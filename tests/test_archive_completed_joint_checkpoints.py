from contextlib import redirect_stdout
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
archiver = importlib.import_module('archive_completed_joint_checkpoints')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class JointCheckpointArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name).resolve() / 'project'
        self.stage = self.project / 'runs/full-actor-rl/rl_joint'
        self.metrics = []
        self.commits = []
        for number in range(1, 33):
            checkpoint = self.stage / 'checkpoints' / f'{number:032x}'
            changes = {role: {'max_abs': 0.1, 'l2': 0.2, 'changed_tensors': 1}
                for role in ('actor', 'renderer')}
            gradients = {role: {'gradient_norm': 0.5, 'clip_ratio': 1.0}
                for role in ('actor', 'renderer')}
            row = {'round': number, 'actor_loss': 0.1, 'renderer_loss': 0.2,
                'reward': 0.5, 'zero_advantage_fraction': 0.0}
            commit = {'status': 'COMMITTED', 'transaction_id': checkpoint.name,
                'next_data_cursor': {'round': number}, 'new_version': f'joint:{number}',
                'components': {'actor': 'actor.pt', 'renderer': 'renderer.pt'},
                'optimizer_steps': ['actor', 'renderer'], 'parameter_changes': changes,
                'gradients': gradients, 'probes': {}, 'committed_ns': number,
                'metadata': {'mode': 'joint', 'rl_round_metrics': row.copy(),
                    'config': {'rl': {'joint_effective_updates': 32}}}}
            write_json(checkpoint / 'COMMITTED.json', commit)
            for filename in ('actor.pt', 'renderer.pt', 'training_state.pt'):
                (checkpoint / filename).write_bytes(b'synthetic CPU metadata fixture')
            self.commits.append(commit)
            self.metrics.append(row | {'checkpoint': str(checkpoint), 'gradients': gradients,
                'probes': {}, 'parameter_changes': changes, 'optimizer_steps': ['actor', 'renderer'],
                'effective_components': ['actor', 'renderer'],
                'effective_update_counts': {'actor': number, 'renderer': number,
                    'required_components': number}})
        self.completed = {'status': 'completed', 'mode': 'joint', 'rounds': 32,
            'required_effective_updates': 32, 'effective_target_met': True,
            'required_components': ['actor', 'renderer'],
            'effective_update_counts': {'actor': 32, 'renderer': 32, 'required_components': 32},
            'consecutive_zero_advantage_rounds': 0}
        write_json(self.stage / 'training_completed.json', self.completed)
        self.write_metrics()

    def write_metrics(self):
        (self.stage / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.metrics))

    def test_first_and_second_halves_cover_each_actual_checkpoint_exactly_once(self):
        first = archiver.prepare(self.project, 0)
        second = archiver.prepare(self.project, 16)
        combined = first['selected'] + second['selected']
        self.assertEqual([item['round'] for item in combined], list(range(1, 33)))
        self.assertEqual(len({item['group'] for item in combined}), 32)
        self.assertEqual([item['group'] for item in combined],
            [Path(row['checkpoint']).relative_to(self.project).as_posix() for row in self.metrics])
        for plan in (first, second):
            self.assertEqual(plan['command'].count('--only-group'), 16)
            self.assertEqual(plan['command'][plan['command'].index('--lock-wait-seconds') + 1], '120')
            self.assertEqual(plan['command'][plan['command'].index('--read-mib-per-second') + 1], '16')
            self.assertEqual(plan['command'][plan['command'].index('--chunk-mib') + 1], '256')
            self.assertEqual(plan['command'][plan['command'].index('--compression') + 1], 'gzip')
            self.assertEqual(plan['output'], str(self.project / 'delivery/project-results'))
            self.assertFalse(plan['full_project_delivery_complete'])
        self.assertFalse((self.project / 'delivery').exists())

    def test_incomplete_or_insufficient_joint_completion_is_rejected(self):
        for change in ({'status': 'incomplete_effective_updates'}, {'effective_target_met': False},
                {'effective_update_counts': {'actor': 32, 'renderer': 31, 'required_components': 31}}):
            with self.subTest(change=change):
                write_json(self.stage / 'training_completed.json', self.completed | change)
                with self.assertRaises(ValueError):
                    archiver.prepare(self.project, 0)

    def test_selected_checkpoint_requires_all_three_payloads(self):
        checkpoint = Path(self.metrics[0]['checkpoint'])
        for filename in ('actor.pt', 'renderer.pt', 'training_state.pt'):
            with self.subTest(filename=filename):
                path = checkpoint / filename
                saved = path.read_bytes()
                path.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    archiver.prepare(self.project, 0)
                path.write_bytes(saved)

    def test_duplicate_metric_or_checkpoint_is_rejected(self):
        original = copy.deepcopy(self.metrics)
        for duplicate in ('round', 'checkpoint'):
            with self.subTest(duplicate=duplicate):
                self.metrics = copy.deepcopy(original)
                self.metrics[1][duplicate] = self.metrics[0][duplicate]
                self.write_metrics()
                with self.assertRaises(ValueError):
                    archiver.prepare(self.project, 0)

    def test_metric_checkpoint_outside_joint_stage_is_rejected(self):
        original = copy.deepcopy(self.metrics)
        for outside in (self.project.parent / 'outside', self.project / 'runs/other-stage/checkpoint'):
            with self.subTest(outside=outside):
                write_json(outside / 'COMMITTED.json', self.commits[0])
                self.metrics = copy.deepcopy(original)
                self.metrics[0]['checkpoint'] = str(outside)
                self.write_metrics()
                with self.assertRaises(ValueError):
                    archiver.prepare(self.project, 0)

    def test_symlinked_selected_payload_is_rejected(self):
        payload = Path(self.metrics[0]['checkpoint']) / 'actor.pt'
        outside = self.project.parent / 'outside-payload.pt'
        outside.write_bytes(b'synthetic outside payload')
        payload.unlink()
        payload.symlink_to(outside)
        with self.assertRaises(ValueError):
            archiver.prepare(self.project, 0)

    def test_missing_commit_or_ineffective_metric_is_rejected(self):
        marker = Path(self.metrics[0]['checkpoint']) / 'COMMITTED.json'
        marker.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            archiver.prepare(self.project, 0)
        write_json(marker, self.commits[0])
        self.metrics[0]['effective_components'] = ['actor']
        self.write_metrics()
        with self.assertRaises(ValueError):
            archiver.prepare(self.project, 0)

    def test_only_the_two_authorized_halves_are_allowed(self):
        for start, count in ((1, 16), (32, 16), (0, 32), (0, 0), (True, 16)):
            with self.subTest(start=start, count=count):
                with self.assertRaises(ValueError):
                    archiver.prepare(self.project, start, count)

    def test_packer_exit_code_is_propagated_without_executing_packer(self):
        with patch.object(archiver.shutil, 'disk_usage') as disk, patch.object(archiver.subprocess, 'run') as run, redirect_stdout(io.StringIO()) as output:
            disk.return_value.free = 42_000_000_000
            run.return_value.returncode = 7
            result = archiver.main(['--project', str(self.project), '--start', '16', '--count', '16'])
        self.assertEqual(result, 7)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command.count('--only-group'), 16)
        self.assertEqual(run.call_args.kwargs, {'cwd': str(self.project), 'check': False})
        reports = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(reports[0]['status'], 'selected_scope_validated')
        self.assertEqual(reports[1]['returncode'], 7)
        self.assertFalse((self.project / 'delivery').exists())

    def test_insufficient_space_prevents_packer_start(self):
        with patch.object(archiver.shutil, 'disk_usage') as disk, patch.object(archiver.subprocess, 'run') as run, redirect_stdout(io.StringIO()) as output:
            disk.return_value.free = 41_999_999_999
            with self.assertRaisesRegex(RuntimeError, 'At least42GB'):
                archiver.main(['--project', str(self.project), '--start', '0'])
        run.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report['required_free_bytes'], 42_000_000_000)
        self.assertEqual(report['free_bytes_at_execution'], 41_999_999_999)


if __name__ == '__main__':
    unittest.main()
