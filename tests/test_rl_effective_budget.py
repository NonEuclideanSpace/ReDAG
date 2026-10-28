import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rl_progress_module', ROOT / 'src/agentic_image/learning/progress.py')
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def proof(mode, round_number, active, zero=0.0):
    return {'status': 'COMMITTED', 'transaction_id': f'commit{round_number}',
        'new_version': f'{mode}:{round_number}', 'next_data_cursor': {'round': round_number},
        'metadata': {'mode': mode, 'rl_round_metrics': {'round': round_number,
            'actor_loss': 1.0, 'renderer_loss': 1.0, 'reward': 0.5,
            'zero_advantage_fraction': zero, 'renders': 1, 'no_renderer_action': False}},
        'optimizer_steps': list(active), 'probes': {},
        'gradients': {name: {'gradient_norm': 1.0} for name in active},
        'parameter_changes': {name: {'changed_tensors': 1, 'l2': 0.1, 'max_abs': 0.1} for name in active}}


class ProgressTests(unittest.TestCase):
    def test_default_budget_and_explicit_caps(self):
        rl = {'agent_rounds': 32, 'joint_rounds': 32}
        self.assertEqual(progress.rl_round_limits(rl, 'agent'), {
            'nominal_rounds': 32, 'effective_updates': 32, 'max_rounds': 32, 'enabled': False})
        rl.update(agent_effective_updates=32, agent_max_rounds=96)
        self.assertEqual(progress.rl_round_limits(rl, 'agent')['max_rounds'], 96)
        with self.assertRaises(ValueError):
            progress.rl_round_limits(rl, 'agent', 31)
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                progress.rl_round_limits(rl | {'agent_effective_updates': value}, 'agent')

    def test_config_rejects_unbounded_or_impossible_effective_targets(self):
        tree = ast.parse((ROOT / 'src/agentic_image/config.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'validate_config')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual_config_validation', 'exec'), namespace)
        base = {'canvas': [512, 512], 'rl': {'tasks': 2, 'roots': 2, 'suffixes': 2, 'agent_rounds': 32, 'joint_rounds': 32},
            'renderer': {'steps': 50, 'replay_strata': 4, 'eta': .05, 'cfg_scale': 4},
            'actor': {'max_input_tokens': 1024, 'max_new_tokens': 128, 'head_chunk_size': 32, 'max_images': 4},
            'episode': {'max_actions': 10}}
        valid = copy.deepcopy(base)
        valid['rl'].update(agent_effective_updates=32, agent_max_rounds=96)
        namespace['validate_config'](valid)
        for patch in ({'agent_max_rounds': 96}, {'agent_effective_updates': 32, 'agent_max_rounds': 31},
                      {'joint_effective_updates': True, 'joint_max_rounds': 96}):
            invalid = copy.deepcopy(base)
            invalid['rl'].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                namespace['validate_config'](invalid)

    def test_joint_requires_both_components_in_the_same_committed_round(self):
        counts = {'actor': 0, 'renderer': 0, 'required_components': 0}
        for index, active in enumerate((('actor',), ('renderer',), ('actor', 'renderer')), 1):
            progress.accumulate_effective_updates(counts, 'joint', proof('joint', index, active))
        self.assertEqual(counts, {'actor': 2, 'renderer': 2, 'required_components': 1})
        invalid = proof('joint', 4, ('actor', 'renderer'))
        invalid['parameter_changes']['renderer']['l2'] = 0.0
        progress.accumulate_effective_updates(counts, 'joint', invalid)
        self.assertEqual(counts['required_components'], 1)
        invalid = proof('joint', 5, ('actor', 'renderer'))
        invalid['gradients']['actor']['gradient_norm'] = 0.0
        progress.accumulate_effective_updates(counts, 'joint', invalid)
        self.assertEqual(counts['required_components'], 1)

    def history(self, root):
        counts = {'actor': 0, 'renderer': 0, 'required_components': 0}
        rows = []
        for index, active in enumerate((('actor',), (), ('actor',)), 1):
            committed = proof('agent', index, active, zero=1.0)
            checkpoint = root / 'checkpoints' / committed['transaction_id']
            write(checkpoint / 'COMMITTED.json', committed)
            progress.accumulate_effective_updates(counts, 'agent', committed)
            rows.append(progress.committed_round_metrics(committed, checkpoint, counts))
        return rows

    def test_resume_recovers_final_metric_from_commit_without_resetting_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.history(root)
            path = root / 'metrics.jsonl'
            path.write_text(json.dumps(rows[0]) + '\n' + json.dumps(rows[1]) + '\n{"round":')
            counts, zero = progress.restore_rl_history(root, 'agent', 3, 'agent:3')
            self.assertEqual(counts, {'actor': 2, 'renderer': 0, 'required_components': 2})
            self.assertEqual(zero, 3)
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], rows)

    def test_resume_rejects_forged_counts_and_duplicate_commit_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.history(root)
            rows[-1]['effective_update_counts']['required_components'] = 32
            (root / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(ValueError):
                progress.restore_rl_history(root, 'agent', 3, 'agent:3')
            duplicate = proof('agent', 3, ('actor',)) | {'transaction_id': 'duplicate'}
            write(root / 'checkpoints/duplicate/COMMITTED.json', duplicate)
            with self.assertRaises(ValueError):
                progress.restore_rl_history(root, 'agent', 3, 'agent:3')


class RLControlFlowTests(unittest.TestCase):
    def run_stage(self, root, active_rounds, nominal, target=None, maximum=None, zero=0.0, resume=False):
        config = {'seed': 11, 'models': {'renderer_policy': 'fixture-policy'},
            'renderer': {'steps': 50, 'eta': 0.05, 'cfg_scale': 4},
            'data': {'public_tasks': 'fixture', 'development_tasks': str(root / 'absent-dev.jsonl')},
            'episode': {'max_actions': 1, 'max_renders': 1}, 'actor': {'max_new_tokens': 1},
            'rl': {'tasks': 1, 'roots': 2, 'suffixes': 2, 'agent_rounds': nominal, 'joint_rounds': nominal,
                'actor_learning_rate': 1e-5, 'renderer_learning_rate': 1e-5, 'epsilon': 1e-8,
                'warmup_rounds': 1, 'beta_actor': 0.02, 'beta_renderer': 0.01, 'max_grad_norm': 1,
                'probe_actor_kl': 0.1, 'probe_renderer_kl': 0.01, 'evaluate_every': 32}}
        if target is not None:
            config['rl'].update(agent_effective_updates=target, agent_max_rounds=maximum)
        write(root / 'sampler_selection.json', {'accepted': True, 'renderer_policy': 'fixture-policy', **config['renderer']})
        actor, renderer = Mock(), Mock()
        actor.has_reference = renderer._reference_exists = True
        actor.policy_parameters.return_value = renderer.policy_parameters.return_value = []
        components = SimpleNamespace(config=config, directory=root, load_all=lambda: None, actor=actor,
            renderer=renderer, judge=SimpleNamespace(thresholds={'semantic': {}}, require_calibration=False))

        class Transaction:
            def __init__(self, folder, components):
                self.folder, self.before = folder, None
                self.folder.mkdir(parents=True, exist_ok=True)

            def begin(self, behavior, cursor, metadata):
                self.index = cursor['round'] + 1
                self.before = {}
                self.manifest = {'metadata': copy.deepcopy(metadata)}
                self.transaction_path = self.folder / f'commit{self.index}'

            def set_batch(self, *args):
                pass

            def step_and_commit(self, behavior, **kwargs):
                row = proof('agent', self.index, ('actor',) if self.index in active_rounds else ())
                row['metadata'] = self.manifest['metadata']
                write(self.transaction_path / 'COMMITTED.json', row)
                self.before = None
                return row

            def load_committed(self):
                maximum = max(int(path.parent.name.removeprefix('commit')) for path in self.folder.glob('*/COMMITTED.json'))
                return {'data_cursor': {'round': maximum}, 'behavior_version': f'agent:{maximum}'}

        class Fraction:
            def __eq__(self, other):
                return self

            def float(self):
                return self

            def mean(self):
                return zero

        batch = {'rewards': [0.0], 'roots': [], 'episodes': []}
        namespace = {name: getattr(progress, name) for name in ('accumulate_effective_updates', 'committed_round_metrics',
            'required_rl_components', 'restore_rl_history', 'rl_round_limits')}
        namespace.update(json=json, random=random, Path=Path, torch=SimpleNamespace(tensor=lambda x: SimpleNamespace(mean=lambda: 0.0)),
            read_tasks=lambda *args, **kwargs: [SimpleNamespace(task_id='fixture', limits=SimpleNamespace(**config['episode']))],
            make_optimizer=lambda *args, **kwargs: Mock(), make_rl_scheduler=lambda *args: Mock(),
            TrainingComponent=lambda *args: SimpleNamespace(active=args[-1]), JointTransaction=Transaction,
            RolloutCollector=lambda c: SimpleNamespace(collect=lambda *args: copy.deepcopy(batch)),
            ScoreWeights=lambda *args: SimpleNamespace(tasks=1), stream_seed=lambda *args: 11,
            hierarchical_rloo=lambda rewards: SimpleNamespace(suffix=Fraction()), UpdateProbe=lambda *args: None,
            sum_policy_gradients=lambda *args: None, atomic_json=write)
        tree = ast.parse((ROOT / 'src/agentic_image/runtime/training.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'train_rl', 'append_metrics', 'prune_checkpoints'}]
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'actual_training_control_flow', 'exec'), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            namespace['train_rl'](components, mode='agent', resume=resume)
        return json.loads((root / 'rl_agent/training_completed.json').read_text())

    def test_32_effective_updates_take_40_actual_rounds_and_stop_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_stage(root, set(range(9, 60)), nominal=32, target=32, maximum=96)
            self.assertEqual(result['rounds'], 40)
            self.assertTrue(result['effective_target_met'])
            metrics = [json.loads(line) for line in (root / 'rl_agent/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['round'] for row in metrics], list(range(1, 41)))
            marker = json.loads((Path(metrics[-1]['checkpoint']) / 'COMMITTED.json').read_text())
            self.assertEqual(marker['new_version'], 'agent:40')
            self.assertEqual(metrics[-1]['effective_update_counts'], result['effective_update_counts'])

    def test_attempt_cap_writes_incomplete_evidence_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'exhausted 3 rounds'):
                self.run_stage(root, {2}, nominal=2, target=2, maximum=3)
            result = json.loads((root / 'rl_agent/training_completed.json').read_text())
            self.assertFalse(result['effective_target_met'])
            self.assertEqual(result['status'], 'incomplete_effective_updates')
            self.assertEqual(result['rounds'], 3)

    def test_resume_continues_effective_count_without_repeating_committed_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'exhausted 3 rounds'):
                self.run_stage(root, {2}, nominal=2, target=2, maximum=3)
            result = self.run_stage(root, {2, 4}, nominal=2, target=2, maximum=6, resume=True)
            self.assertEqual(result['rounds'], 4)
            self.assertEqual(result['effective_update_counts']['required_components'], 2)
            self.assertTrue(result['effective_target_met'])
            rows = [json.loads(line) for line in (root / 'rl_agent/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['round'] for row in rows], [1, 2, 3, 4])

    def test_default_fixed_rounds_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_stage(Path(directory), {2}, nominal=3)
            self.assertEqual(result['rounds'], 3)
            self.assertFalse(result['effective_target_enabled'])
            self.assertFalse(result['effective_target_met'])

    def test_no_reward_difference_guard_survives_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for resume in (False, True):
                with self.assertRaisesRegex(RuntimeError, '25 rounds'):
                    self.run_stage(root, set(), nominal=32, target=32, maximum=96, zero=1.0, resume=resume)
            self.assertEqual(len(list((root / 'rl_agent/checkpoints').glob('*/COMMITTED.json'))), 25)


if __name__ == '__main__':
    unittest.main()
