import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('queue_reservations', ROOT / 'scripts/gpu_task_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class GpuReservationTests(unittest.TestCase):
    def gpu(self):
        return {
            0: {'uuid': 'gpu0', 'free_gib': 40, 'utilization': 0, 'busy': False},
            1: {'uuid': 'gpu1', 'free_gib': 10, 'utilization': 90, 'busy': True},
        }

    def test_every_prerequisite_must_be_ready_before_reserving(self):
        with tempfile.TemporaryDirectory() as folder:
            required = Path(folder) / 'required'
            gate = Path(folder) / 'gate.json'
            job = {'id': 'critical', 'gpus': [0, 1], 'command': ['true'],
                'reserve_gpus_when_ready': True, 'after': ['main_sft'],
                'requires': [str(required)],
                'gates': [{'path': str(gate), 'key': 'training.complete', 'equals': True}]}
            statuses = {'critical': {'status': 'pending'}, 'main_sft': {'status': 'running'}}
            self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))
            statuses['main_sft']['status'] = 'completed'
            self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))
            required.touch()
            self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))
            gate.write_text('{"training":{"complete":false}}')
            self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))
            gate.write_text('{"training":{"complete":true}}')
            self.assertIn('critical', queue.ready_gpu_reservations([job], statuses, self.gpu()))
            job['command'] = []
            self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))

    def test_only_pending_or_running_automatic_flagged_jobs_reserve(self):
        job = {'id': 'critical', 'gpus': [0, 1], 'command': ['true']}
        statuses = {'critical': {'status': 'pending'}}
        self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))
        job['reserve_gpus_when_ready'] = True
        for status in ('completed', 'failed', 'interrupted'):
            statuses['critical']['status'] = status
            self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))
        statuses['critical']['status'] = 'pending'
        job['external'] = True
        self.assertFalse(queue.ready_gpu_reservations([job], statuses, self.gpu()))

    def test_running_critical_blocks_refill_during_model_loading(self):
        job = {'id': 'critical', 'gpus': [0, 1], 'command': ['true'],
               'priority': 2, 'reserve_gpus_when_ready': True}
        statuses = {'critical': {'status': 'running'}}
        gpu = {i: {'free_gib': 46, 'busy': True} for i in (0, 1)}
        reservations = queue.ready_gpu_reservations([job], statuses, gpu)
        baseline = {'id': 'baseline', 'gpus': [0], 'command': ['true'],
                    'priority': 12, 'minimum_free_gib': 32, 'allow_overlap': True}
        self.assertEqual(reservations['critical']['status'], 'running')
        self.assertTrue(any('reserved for' in reason
                            for reason in queue.reasons(baseline, statuses, gpu, reservations)))

    def test_higher_priority_and_disjoint_jobs_are_not_blocked(self):
        critical = {'id': 'critical', 'gpus': [0, 1], 'command': ['true'],
            'priority': 10, 'reserve_gpus_when_ready': True}
        statuses = {'critical': {'status': 'pending'}}
        reservations = queue.ready_gpu_reservations([critical], statuses, self.gpu())
        higher = {'id': 'urgent', 'gpus': [0], 'command': ['true'], 'priority': 5}
        self.assertEqual(queue.reasons(higher, statuses, self.gpu(), reservations), [])
        cpu = {'id': 'cpu', 'gpus': [], 'command': ['true'], 'priority': 20}
        self.assertEqual(queue.reasons(cpu, statuses, self.gpu(), reservations), [])
        for priority in (10, 20):
            other = {**higher, 'priority': priority}
            self.assertTrue(any('reserved for ready job critical' in reason
                for reason in queue.reasons(other, statuses, self.gpu(), reservations)))

    def test_equal_priority_reservations_do_not_block_each_other_cyclically(self):
        first = {'id': 'first', 'gpus': [0, 1], 'command': ['true'],
            'priority': 10, 'reserve_gpus_when_ready': True}
        second = {**first, 'id': 'second', 'gpus': [0]}
        statuses = {key: {'status': 'pending'} for key in ('first', 'second')}
        reservations = queue.ready_gpu_reservations([first, second], statuses, self.gpu())
        self.assertEqual(list(reservations), ['first'])
        self.assertFalse(any('reserved for' in reason
            for reason in queue.reasons(first, statuses, self.gpu(), reservations)))
        self.assertTrue(any('reserved for ready job first' in reason
            for reason in queue.reasons(second, statuses, self.gpu(), reservations)))

    def tick(self, root, main_complete, reserve=True, baseline_priority=20, gpu=None, independent_active=True, baseline_overlap=False):
        (root / 'deployment').mkdir()
        jobs = [
            {'id': 'main_sft', 'gpus': [1], 'priority': 0},
            {'id': 'independent_sft', 'gpus': [1], 'priority': 0},
            {'id': 'baseline', 'gpus': [0], 'priority': baseline_priority, 'allow_overlap': baseline_overlap},
            {'id': 'critical', 'gpus': [0, 1], 'priority': 10,
                'after': ['main_sft'], 'reserve_gpus_when_ready': reserve},
        ]
        for job in jobs:
            job.update(title=job['id'], command=['true'],
                log_file=str(root / (job['id'] + '.log')),
                pid_file=str(root / (job['id'] + '.pid')),
                started_file=str(root / (job['id'] + '.started.json')),
                exit_file=str(root / (job['id'] + '.exit.json')))
        if main_complete:
            Path(jobs[0]['exit_file']).write_text('{"returncode":0}')
        else:
            Path(jobs[0]['pid_file']).write_text('87')
        if independent_active:
            Path(jobs[1]['pid_file']).write_text('88')
        else:
            Path(jobs[1]['exit_file']).write_text('{"returncode":0}')
        path = root / 'deployment/jobs.json'
        path.write_text(json.dumps({'jobs': jobs}))
        with contextlib.chdir(root), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(queue, 'gpu_state', return_value=gpu or self.gpu()), \
                patch.object(queue, 'alive', side_effect=lambda pid: pid in (87, 88)), \
                patch.object(queue.time, 'monotonic', side_effect=[0, 0, 0, 6]), \
                patch.object(queue.time, 'sleep'), patch.object(queue.os, 'kill') as kill, \
                patch.object(queue.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 999
            queue.supervise(path, 5 / 3600)
        kill.assert_not_called()
        state = json.loads((root / 'deployment/gpu-queue-state.json').read_text())
        return state, spawn

    def test_main_sft_not_completed_allows_single_gpu_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            state, spawn = self.tick(Path(folder), main_complete=False)
        self.assertEqual(state['reservations'], {})
        self.assertEqual(state['jobs']['main_sft']['status'], 'running')
        self.assertEqual(state['jobs']['baseline']['status'], 'running')
        self.assertEqual(spawn.call_args.args[0][-1], 'baseline')

    def test_newly_started_owner_keeps_reservation_in_the_same_tick(self):
        gpu = self.gpu()
        gpu[1].update(free_gib=40, busy=False)
        with tempfile.TemporaryDirectory() as folder:
            state, spawn = self.tick(Path(folder), main_complete=True, gpu=gpu,
                independent_active=False, baseline_overlap=True)
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(spawn.call_args.args[0][-1], 'critical')
        self.assertEqual(state['jobs']['baseline']['status'], 'blocked')
        self.assertEqual(state['reservations']['critical']['status'], 'running')

    def test_ready_two_gpu_job_prevents_low_or_same_priority_refill(self):
        for priority in (10, 20):
            with self.subTest(priority=priority), tempfile.TemporaryDirectory() as folder:
                state, spawn = self.tick(Path(folder), main_complete=True, baseline_priority=priority)
            spawn.assert_not_called()
            self.assertEqual(state['jobs']['independent_sft']['status'], 'running')
            self.assertEqual(state['jobs']['critical']['status'], 'blocked')
            self.assertEqual(state['reservations']['critical']['gpus'], [0, 1])
            self.assertTrue(any('reserved for ready job critical' in reason
                for reason in state['jobs']['baseline']['reasons']))

    def test_default_disabled_reservation_preserves_existing_refill(self):
        with tempfile.TemporaryDirectory() as folder:
            state, spawn = self.tick(Path(folder), main_complete=True, reserve=False)
        self.assertEqual(state['reservations'], {})
        self.assertEqual(state['jobs']['baseline']['status'], 'running')
        self.assertEqual(spawn.call_args.args[0][-1], 'baseline')

    def test_ready_critical_starts_before_earlier_equal_priority_baseline(self):
        gpu = self.gpu()
        gpu[1].update(free_gib=40, busy=False)
        with tempfile.TemporaryDirectory() as folder:
            state, spawn = self.tick(Path(folder), main_complete=True, baseline_priority=10,
                gpu=gpu, independent_active=False)
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(spawn.call_args.args[0][-1], 'critical')
        self.assertEqual(state['jobs']['critical']['status'], 'running')
        self.assertEqual(state['reservations']['critical']['status'], 'running')


class RunningReservationAllowanceTests(unittest.TestCase):
    def setUp(self):
        self.job = {'id': 'evaluation', 'gpus': [1], 'command': ['true'], 'priority': 20,
            'minimum_free_gib': 34, 'allow_overlap': True,
            'allow_with_running_reservation': {'job_id': 'joint', 'gpus': [1]}}
        self.statuses = {'joint': {'status': 'running'}}
        self.gpu = {device: {'free_gib': 40, 'busy': True} for device in (0, 1)}
        self.reservations = {'joint': {'job_id': 'joint', 'gpus': [0, 1], 'priority': 1, 'status': 'running'}}

    def reasons(self, job=None, statuses=None, reservations=None):
        return queue.reasons(self.job if job is None else job,
            self.statuses if statuses is None else statuses, self.gpu,
            self.reservations if reservations is None else reservations)

    def test_exact_running_owner_and_allowed_gpu_can_overlap(self):
        self.assertEqual(self.reasons(), [])
        without_rule = {key: value for key, value in self.job.items() if key != 'allow_with_running_reservation'}
        self.assertTrue(any('reserved for' in reason for reason in self.reasons(without_rule)))

    def test_nonrunning_owner_blocks_even_when_no_reservation_exists(self):
        for status in ('pending', 'failed', 'interrupted', 'blocked', 'awaiting_external_launch'):
            with self.subTest(status=status):
                blocked = self.reasons(statuses={'joint': {'status': status}}, reservations={})
                self.assertTrue(any('awaiting running or completed' in reason for reason in blocked))
        self.assertTrue(self.reasons(statuses={}, reservations={}))
        self.assertEqual(self.reasons(statuses={'joint': {'status': 'completed'}}, reservations={}), [])

    def test_rule_requires_exact_owner_gpu_subset_and_running_reservation(self):
        self.statuses['different'] = {'status': 'running'}
        wrong = self.job | {'allow_with_running_reservation': {'job_id': 'different', 'gpus': [1]}}
        self.assertTrue(any('reserved for' in reason for reason in self.reasons(wrong)))
        for devices in ([0], [0, 1]):
            with self.subTest(devices=devices):
                self.assertTrue(any('reserved for' in reason for reason in self.reasons(self.job | {'gpus': devices})))
        pending = {'joint': self.reservations['joint'] | {'status': 'pending'}}
        self.assertTrue(any('reserved for' in reason for reason in self.reasons(reservations=pending)))
        other = self.reservations | {'other': {'job_id': 'other', 'gpus': [1], 'priority': 1, 'status': 'running'}}
        self.assertTrue(any('ready job other' in reason for reason in self.reasons(reservations=other)))

    def test_malformed_rules_fail_closed(self):
        for rule in (None, True, [], {}, {'job_id': 'joint', 'gpus': []},
                {'job_id': 'joint', 'gpus': [True]}, {'job_id': 'joint', 'gpus': ['1']},
                {'job_id': 'joint', 'gpus': [-1]}, {'job_id': '', 'gpus': [1]}):
            with self.subTest(rule=rule):
                blocked = self.reasons(self.job | {'allow_with_running_reservation': rule})
                self.assertIn('invalid allow_with_running_reservation rule', blocked)

    def test_resource_checks_remain_required(self):
        self.gpu[1]['free_gib'] = 33.9
        self.assertIn('GPU 1 needs more free memory', self.reasons())
        self.gpu[1]['free_gib'] = 40
        self.assertIn('GPU 1 occupied', self.reasons(self.job | {'allow_overlap': False}))
        self.statuses['another'] = {'status': 'running', 'exclusive_group': 'single_evaluation'}
        self.assertTrue(any('active worker in single_evaluation' in reason
            for reason in self.reasons(self.job | {'exclusive_group': 'single_evaluation'})))

    def test_dependencies_and_gates_still_apply_running_or_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            gate = Path(folder) / 'calibration.json'
            gate.write_text('{"complete":false}')
            required = Path(folder) / 'COMMITTED.json'
            for owner_status in ('running', 'completed'):
                statuses = {'joint': {'status': owner_status}, 'prior': {'status': 'pending'}}
                reservations = self.reservations if owner_status == 'running' else {}
                job = self.job | {'after': ['prior'], 'requires': [str(required)],
                    'gates': [{'path': str(gate), 'key': 'complete', 'equals': True}]}
                blocked = self.reasons(job, statuses, reservations)
                self.assertIn('awaiting prior', blocked)
                self.assertIn('missing ' + str(required), blocked)
                self.assertTrue(any('gate pending' in reason for reason in blocked))
            self.assertIn('command not prepared', self.reasons(self.job | {'command': []}))


if __name__ == '__main__':
    unittest.main()
