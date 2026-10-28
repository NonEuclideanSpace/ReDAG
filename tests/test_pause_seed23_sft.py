from contextlib import ExitStack, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pause_seed23_sft_entry', ROOT / 'scripts/pause_seed23_sft.py')
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class PauseSeed23SftTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()
        self.output = self.project / 'data/actor-full-sft-seed23'
        self.checkpoints = self.project / 'runs/actor-full-seed23/sft_actor/checkpoints'
        self.checkpoint = self.checkpoints / 'transaction-2'
        self.output.mkdir(parents=True)
        self.checkpoint.mkdir(parents=True)
        self.registration = {
            'scope': 'independent_actor_sft_seed_replication_v1',
            'updates': 200,
            'training_seed': 23,
            'independent_family_expansion': False,
            'downstream_rl_planned': False,
            'phase_counts': [5, 7, 5, 3],
            'admission_identity_sha256': 'fixture-admission',
        }
        self.write_json(self.output / 'registration.json', self.registration)
        self.metrics_path = self.checkpoints.parent / 'metrics.jsonl'
        self.metrics = [
            {'step': step, 'stage': 'actor', 'record_ids': list(range(20)),
             'checkpoint': str(self.checkpoint)}
            for step in (1, 2)
        ]
        self.metrics_path.write_text(''.join(json.dumps(row) + '\n' for row in self.metrics))
        self.commit = {
            'committed_ns': 200,
            'status': 'COMMITTED',
            'transaction_id': self.checkpoint.name,
            'new_version': 'actor:2',
            'components': {'actor': 'actor.pt'},
            'optimizer_steps': ['actor'],
        }
        self.write_json(self.checkpoint / 'COMMITTED.json', self.commit)
        self.write_json(self.checkpoints / 'LATEST.json', {
            'transaction_id': self.checkpoint.name, 'version': 'actor:2',
        })
        self.state = {
            'data_cursor': {'step': 2, 'sampler': {'rng': random.Random(23).getstate()}},
            'behavior_version': 'actor:2',
            'rng': {'python': random.Random(23).getstate(), 'torch': [], 'cuda': [], 'numpy': []},
        }
        self.actor = {'model': {'weight': [1]}, 'optimizer': {}, 'scheduler': {}}
        for name in ('actor.pt', 'training_state.pt'):
            (self.checkpoint / name).write_bytes(b'nonempty-fixture')
        self.load_state = mock.Mock(side_effect=lambda path: self.state if path.name == 'training_state.pt' else self.actor)

    def write_json(self, path, value):
        path.write_text(json.dumps(value))

    def test_valid_committed_snapshot_and_malformed_sampler_guard(self):
        snapshot = entry.checkpoint_snapshot(self.project, self.load_state)
        self.assertEqual(snapshot['step'], 2)
        self.assertEqual(snapshot['target_updates'], 200)
        self.assertEqual(snapshot['checkpoint'], str(self.checkpoint))
        self.assertTrue(snapshot['sampler_cursor_valid'])
        self.assertTrue(snapshot['optimizer_scheduler_rng_preserved'])
        self.assertEqual(len(snapshot['files']), 6)
        self.assertTrue(all(row['bytes'] > 0 for row in snapshot['files']))
        self.assertEqual(self.load_state.call_args_list, [
            mock.call(self.checkpoint / 'training_state.pt'),
            mock.call(self.checkpoint / 'actor.pt'),
        ])
        self.state['data_cursor']['sampler']['rng'] = (0, (), None)
        with self.assertRaises(ValueError):
            entry.checkpoint_snapshot(self.project, self.load_state)

    def test_commit_mismatch_and_incomplete_trailing_metrics_reject(self):
        self.commit['new_version'] = 'actor:3'
        self.write_json(self.checkpoint / 'COMMITTED.json', self.commit)
        with self.assertRaisesRegex(ValueError, 'last complete metric do not match'):
            entry.checkpoint_snapshot(self.project, self.load_state)
        self.load_state.assert_not_called()
        self.commit['new_version'] = 'actor:2'
        self.write_json(self.checkpoint / 'COMMITTED.json', self.commit)
        with self.metrics_path.open('a') as stream:
            stream.write('{"step":3')
        with self.assertRaisesRegex(ValueError, 'incomplete trailing line'):
            entry.checkpoint_snapshot(self.project, self.load_state)
        self.load_state.assert_not_called()

    def test_failed_snapshot_resumes_stopped_worker_without_termination(self):
        process = {'pid': 12345, 'start_ticks': '100', 'state': 'S'}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, 'argv', ['pause_seed23_sft.py', '--pid', '12345', '--pause']))
            stack.enter_context(mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=mock.Mock())}))
            stack.enter_context(mock.patch.object(entry, 'process_identity', return_value=process))
            stack.enter_context(mock.patch.object(entry, 'primary_complete', return_value={'bytes': 1}))
            stack.enter_context(mock.patch.object(entry, 'stopped', return_value=True))
            stack.enter_context(mock.patch.object(entry, 'checkpoint_snapshot', side_effect=ValueError('invalid snapshot')))
            proof = stack.enter_context(mock.patch.object(entry, 'write_proof'))
            opened = stack.enter_context(mock.patch.object(entry.os, 'pidfd_open', return_value=99, create=True))
            sent = stack.enter_context(mock.patch.object(entry.signal, 'pidfd_send_signal', create=True))
            stack.enter_context(mock.patch.object(entry.signal, 'signal'))
            stack.enter_context(mock.patch.object(entry.signal, 'setitimer'))
            closed = stack.enter_context(mock.patch.object(entry.os, 'close'))
            errors = stack.enter_context(redirect_stderr(io.StringIO()))
            with self.assertRaisesRegex(ValueError, 'invalid snapshot'):
                entry.main()
            opened.assert_called_once_with(12345, 0)
            self.assertEqual(sent.call_args_list, [
                mock.call(99, entry.signal.SIGSTOP),
                mock.call(99, entry.signal.SIGCONT),
            ])
            proof.assert_not_called()
            closed.assert_called_once_with(99)
            self.assertEqual(json.loads(errors.getvalue())['status'], 'not_paused')

    def test_portable_signals_recheck_identity_before_signaling_expected_pid(self):
        expected = {
            'pid': 12345, 'start_ticks': '100', 'state': 'S',
            'argv': ['python', 'scripts/train_full_actor_sft.py', '--sft-only'],
            'cwd': str(self.project), 'executable': '/usr/bin/python3',
        }
        changes = {
            'start_ticks': '101',
            'argv': ['python', 'unrelated.py'],
            'cwd': '/unrelated',
            'executable': '/different/python3',
        }
        with mock.patch.object(entry, 'process_identity') as identity, mock.patch.object(entry.os, 'kill') as kill:
            calls = mock.Mock()
            calls.attach_mock(identity, 'identity')
            calls.attach_mock(kill, 'kill')
            for kind in (entry.signal.SIGSTOP, entry.signal.SIGTERM, entry.signal.SIGCONT):
                with self.subTest(kind=kind, changed='state'):
                    calls.reset_mock()
                    identity.return_value = {**expected, 'state': 'T'}
                    entry.guarded_signal(None, self.project, expected, kind)
                    self.assertEqual(calls.mock_calls, [
                        mock.call.identity(self.project, expected['pid']),
                        mock.call.kill(expected['pid'], kind),
                    ])
                for key, value in changes.items():
                    with self.subTest(kind=kind, changed=key):
                        calls.reset_mock()
                        identity.return_value = {**expected, key: value}
                        with self.assertRaisesRegex(ValueError, 'identity changed before signal'):
                            entry.guarded_signal(None, self.project, expected, kind)
                        identity.assert_called_once_with(self.project, expected['pid'])
                        kill.assert_not_called()

    def test_portable_failure_resumes_worker_without_termination(self):
        process = {'pid': 12345, 'start_ticks': '100', 'state': 'S'}
        sent = mock.Mock()
        closed = mock.Mock()
        signals = types.SimpleNamespace(
            **{name: getattr(entry.signal, name) for name in ('SIGSTOP', 'SIGTERM', 'SIGCONT', 'SIGALRM', 'ITIMER_REAL')},
            signal=mock.Mock(), setitimer=mock.Mock(),
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, 'argv', ['pause_seed23_sft.py', '--pid', '12345', '--pause']))
            stack.enter_context(mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=mock.Mock())}))
            stack.enter_context(mock.patch.object(entry, 'os', types.SimpleNamespace(kill=sent, close=closed)))
            stack.enter_context(mock.patch.object(entry, 'signal', signals))
            stack.enter_context(mock.patch.object(entry, 'process_identity', return_value=process))
            stack.enter_context(mock.patch.object(entry, 'primary_complete', return_value={'bytes': 1}))
            stack.enter_context(mock.patch.object(entry, 'stopped', return_value=True))
            stack.enter_context(mock.patch.object(entry, 'checkpoint_snapshot', side_effect=ValueError('invalid snapshot')))
            proof = stack.enter_context(mock.patch.object(entry, 'write_proof'))
            errors = stack.enter_context(redirect_stderr(io.StringIO()))
            with self.assertRaisesRegex(ValueError, 'invalid snapshot'):
                entry.main()
            self.assertEqual(sent.call_args_list, [
                mock.call(12345, signals.SIGSTOP),
                mock.call(12345, signals.SIGCONT),
            ])
            proof.assert_not_called()
            closed.assert_not_called()
            self.assertEqual(json.loads(errors.getvalue())['status'], 'not_paused')

    def test_seed47_uses_its_registration_paths_and_gpu_identity(self):
        output = self.project / 'data/actor-full-sft-seed47'
        self.output.rename(output)
        run = self.project / 'runs/actor-full-seed47'
        self.checkpoints.parents[1].rename(run)
        checkpoint = run / 'sft_actor/checkpoints/transaction-2'
        self.registration['training_seed'] = 47
        self.write_json(output / 'registration.json', self.registration)
        metrics = [{**row, 'checkpoint': str(checkpoint)} for row in self.metrics]
        (run / 'sft_actor/metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metrics))
        snapshot = entry.checkpoint_snapshot(self.project, self.load_state, seed=47)
        self.assertEqual(snapshot['checkpoint'], str(checkpoint))
        self.assertEqual(snapshot['step'], 2)
        self.registration['training_seed'] = 23
        self.write_json(output / 'registration.json', self.registration)
        with self.assertRaisesRegex(ValueError, 'seed 47 registration'):
            entry.checkpoint_snapshot(self.project, self.load_state, seed=47)
        proc_root = self.project / 'proc'
        proc = proc_root / '12345'
        proc.mkdir(parents=True)
        executable = self.project / 'python3'
        executable.write_bytes(b'fixture')
        (proc / 'exe').symlink_to(executable)
        (proc / 'cwd').symlink_to(self.project, target_is_directory=True)
        arguments = ['python3', 'scripts/train_full_actor_sft.py',
                     '--config', 'configs/actor-full-seed47.yaml',
                     '--output', 'data/actor-full-sft-seed47',
                     '--run-dir', 'runs/actor-full-seed47', '--updates', '200',
                     '--device', 'cuda:0', '--sft-only']
        (proc / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in arguments))
        (proc / 'status').write_text('State:\tS\nTracerPid:\t0\nSigCgt:\t0\nSigIgn:\t0\nSigBlk:\t0\n')
        (proc / 'stat').write_text('12345 (python3) ' + ' '.join(['S'] + ['0'] * 18 + ['100']))
        with mock.patch.object(entry, 'Path', side_effect=lambda value: proc_root if value == '/proc' else Path(value)):
            identity = entry.process_identity(self.project, 12345, seed=47)
            self.assertEqual(identity['argv'], arguments)
            self.assertEqual(identity['start_ticks'], '100')
            for option, wrong in (('--device', 'cuda:1'), ('--config', 'configs/actor-full-seed23.yaml')):
                with self.subTest(option=option):
                    altered = list(arguments)
                    altered[altered.index(option) + 1] = wrong
                    (proc / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in altered))
                    with self.assertRaisesRegex(ValueError, 'identity differs at ' + option):
                        entry.process_identity(self.project, 12345, seed=47)
        with mock.patch.object(entry, 'process_identity', return_value=identity) as inspect, mock.patch.object(entry.os, 'kill') as kill:
            entry.guarded_signal(None, self.project, identity, entry.signal.SIGSTOP, seed=47)
            inspect.assert_called_once_with(self.project, 12345, seed=47)
            kill.assert_called_once_with(12345, entry.signal.SIGSTOP)


if __name__ == '__main__':
    unittest.main()
