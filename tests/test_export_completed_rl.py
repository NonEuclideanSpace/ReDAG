import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
exporter = importlib.import_module('export_completed_rl')

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None
    save_file = None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def make_fixture(root, mode='joint', tensors=False):
    run = root / 'runs/full-actor-rl'
    stage = run / ('rl_' + mode)
    actor_initial = root / 'runs/synthetic-sft-actor/policy/policy'
    renderer_initial = root / 'runs/synthetic-sft-renderer/adapters/policy.pt'
    actor_reference = run / 'sft_reference/actor/policy'
    renderer_reference = run / 'sft_reference/renderer/reference.pt'
    config = {
        'run_dir': str(run), 'seed': 11, 'synthetic_cpu_fixture': True,
        'models': {
            'actor': 'synthetic/base-actor', 'actor_revision': 'synthetic-actor-revision',
            'actor_nf4': 'models/synthetic-actor-nf4',
            'renderer': 'models/synthetic-renderer', 'renderer_revision': 'synthetic-renderer-revision',
            'actor_policy': str(actor_initial), 'renderer_policy': str(renderer_initial),
            'actor_reference': str(actor_reference), 'renderer_reference': str(renderer_reference),
        },
        'renderer': {'steps': 50, 'eta': 0.3, 'cfg_scale': 4.0},
        'rl': {'agent_rounds': 2, 'joint_rounds': 2, 'agent_effective_updates': 2,
            'joint_effective_updates': 2, 'agent_max_rounds': 6, 'joint_max_rounds': 6},
    }
    references = {'actor_reference': str(actor_reference), 'renderer_reference': str(renderer_reference)}
    write_json(run / 'sft_reference/references.json', references)
    write_json(run / 'rl_config.yaml', config)
    write_json(run / 'sampler_selection.json', {'accepted': True, 'renderer_policy': str(renderer_initial),
        'steps': 50, 'eta': 0.3, 'cfg_scale': 4.0, 'synthetic_cpu_fixture': True})
    commits = {}
    metrics = []
    active = ['actor'] if mode == 'agent' else ['actor', 'renderer']
    for number in (1, 2):
        transaction = f'{number:032x}'
        checkpoint = stage / 'checkpoints' / transaction
        checkpoint.mkdir(parents=True)
        change = {'max_abs': 0.1, 'l2': 0.1, 'changed_tensors': 1}
        changes = {'actor': change.copy(), 'renderer': change.copy() if mode == 'joint'
            else {'max_abs': 0.0, 'l2': 0.0, 'changed_tensors': 0}}
        gradients = {name: {'gradient_norm': 0.25, 'clip_ratio': 1.0} for name in active}
        probes = {'actor_kl': 0.001, 'renderer_kl': 0.001}
        row = {'round': number, 'actor_loss': 0.1, 'renderer_loss': 0.2 if mode == 'joint' else 0.0,
            'reward': 0.5, 'zero_advantage_fraction': 0.0, 'renders': 1, 'no_renderer_action': False}
        counts = {'actor': number, 'renderer': number if mode == 'joint' else 0,
            'required_components': number}
        commit = {'status': 'COMMITTED', 'transaction_id': transaction,
            'behavior_version': 'sft-pair' if number == 1 else f'{mode}:{number - 1}',
            'new_version': f'{mode}:{number}', 'next_data_cursor': {'round': number},
            'components': {'actor': 'actor.pt', 'renderer': 'renderer.pt'},
            'optimizer_steps': active, 'parameter_changes': changes, 'gradients': gradients,
            'probes': probes, 'committed_ns': 1000000 + number,
            'metadata': {'mode': mode, 'config': copy.deepcopy(config), 'rl_round_metrics': row.copy()}}
        write_json(checkpoint / 'COMMITTED.json', commit)
        for filename in ('actor.pt', 'renderer.pt', 'training_state.pt'):
            (checkpoint / filename).write_bytes(b'synthetic metadata-only fixture; not real weights')
        metrics.append(row | {'checkpoint': str(checkpoint), 'gradients': gradients, 'probes': probes,
            'parameter_changes': changes, 'optimizer_steps': active, 'effective_components': active,
            'effective_update_counts': counts})
        commits[number] = (checkpoint, commit)
    (stage / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metrics))
    checkpoint, commit = commits[2]
    write_json(stage / 'checkpoints/LATEST.json', {'transaction_id': checkpoint.name, 'version': f'{mode}:2'})
    completed = {'status': 'completed', 'mode': mode, 'rounds': 2, 'nominal_rounds': 2,
        'maximum_rounds': 6, 'effective_target_enabled': True, 'required_effective_updates': 2,
        'effective_update_counts': metrics[-1]['effective_update_counts'], 'effective_target_met': True,
        'required_components': active, 'consecutive_zero_advantage_rounds': 0,
        'synthetic_cpu_fixture': True}
    write_json(stage / 'training_completed.json', completed)
    sentinel = run / 'renders/shared-still-running.json'
    write_json(sentinel, {'synthetic_cpu_fixture': True, 'status': 'still_running'})
    fixture = {'root': root, 'run': run, 'stage': stage, 'checkpoint': checkpoint,
        'commits': commits, 'metrics': metrics, 'completed': completed, 'config': config,
        'references': references, 'mode': mode, 'sentinel': sentinel}
    if tensors:
        populate_tensors(fixture)
    return fixture


def populate_tensors(fixture):
    import numpy as np

    mode = fixture['mode']
    actor_reference_state = {'synthetic.weight': torch.tensor([0.25])}
    renderer_reference_state = {'synthetic.weight': torch.tensor([0.5])}
    adapter_config = {'peft_type': 'LORA', 'r': 1, 'lora_alpha': 1,
        'target_modules': ['synthetic'], 'base_model_name_or_path': 'synthetic/base'}
    for name in ('actor_policy', 'actor_reference'):
        directory = Path(fixture['config']['models'][name])
        write_json(directory / 'adapter_config.json', adapter_config)
        save_file(actor_reference_state, directory / 'adapter_model.safetensors')
    renderer_initial = Path(fixture['config']['models']['renderer_policy'])
    renderer_initial.parent.mkdir(parents=True, exist_ok=True)
    torch.save(renderer_reference_state, renderer_initial)
    renderer_reference = Path(fixture['config']['models']['renderer_reference'])
    renderer_reference.parent.mkdir(parents=True, exist_ok=True)
    torch.save(renderer_reference_state, renderer_reference)
    for number, (checkpoint, _) in fixture['commits'].items():
        actor_state = {'policy': {'synthetic.weight': torch.tensor([float(number)])},
            'reference': actor_reference_state}
        renderer_state = {'synthetic.weight': torch.tensor([float(number) + 0.5])} if mode == 'joint' else renderer_reference_state
        optimizer = {'state': {0: {'step': torch.tensor(float(number)), 'exp_avg': torch.tensor([0.1]),
            'exp_avg_sq': torch.tensor([0.2])}}, 'param_groups': [{'params': [0], 'lr': 0.001}]}
        scheduler = {'last_epoch': number, '_step_count': number + 1, 'base_lrs': [0.001]}
        torch.save({'model': actor_state, 'optimizer': optimizer, 'scheduler': scheduler, 'active': True}, checkpoint / 'actor.pt')
        renderer_optimizer = optimizer if mode == 'joint' else {'state': {}, 'param_groups': [{'params': [0], 'lr': 0.001}]}
        torch.save({'model': renderer_state, 'optimizer': renderer_optimizer,
            'scheduler': scheduler, 'active': mode == 'joint'}, checkpoint / 'renderer.pt')
        numpy_state = np.random.get_state()
        rng = {'python': random.getstate(), 'torch': torch.get_rng_state(), 'cuda': [],
            'numpy': {'algorithm': numpy_state[0], 'keys': torch.from_numpy(numpy_state[1].astype('int64')),
                'position': numpy_state[2], 'has_gauss': numpy_state[3], 'cached_gaussian': numpy_state[4]}}
        torch.save({'rng': rng, 'data_cursor': {'round': number}, 'behavior_version': f'{mode}:{number}'}, checkpoint / 'training_state.pt')
    final_actor = fixture['stage'] / 'actor_policy/policy'
    write_json(final_actor / 'adapter_config.json', adapter_config)
    write_json(final_actor.parent / 'actor_config.json', {'synthetic_cpu_fixture': True})
    save_file(actor_state['policy'], final_actor / 'adapter_model.safetensors')
    adapters = fixture['stage'] / 'renderer_adapters'
    adapters.mkdir()
    torch.save(renderer_state, adapters / 'policy.pt')
    torch.save(renderer_reference_state, adapters / 'reference.pt')


class ExportMetadataFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = make_fixture(Path(self.temporary.name).resolve())

    def validate(self):
        return exporter.validate_history(self.fixture['stage'], 'joint', expected_updates=2)

    def test_synthetic_committed_history_validates_without_torch(self):
        result = self.validate()
        self.assertEqual(set(result['commits']), {1, 2})
        self.assertEqual(result['checkpoint'], self.fixture['checkpoint'])
        self.assertEqual(result['config'], self.fixture['config'])

    def test_incomplete_or_unmet_effective_target_is_rejected(self):
        path = self.fixture['stage'] / 'training_completed.json'
        for change in ({'status': 'incomplete_effective_updates'}, {'effective_target_met': False}):
            with self.subTest(change=change):
                write_json(path, self.fixture['completed'] | change)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.validate()

    def test_metric_evidence_must_match_actual_commit(self):
        rows = copy.deepcopy(self.fixture['metrics'])
        rows[-1]['reward'] = 0.9
        (self.fixture['stage'] / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaises((ValueError, RuntimeError)):
            self.validate()

    def test_newer_committed_round_than_completed_marker_is_rejected(self):
        newer = self.fixture['stage'] / 'checkpoints' / f'{3:032x}'
        commit = copy.deepcopy(self.fixture['commits'][2][1])
        commit.update(transaction_id=newer.name, new_version='joint:3', next_data_cursor={'round': 3}, committed_ns=1000003)
        commit['metadata']['rl_round_metrics']['round'] = 3
        write_json(newer / 'COMMITTED.json', commit)
        write_json(newer.parent / 'LATEST.json', {'transaction_id': newer.name, 'version': 'joint:3'})
        with self.assertRaises((ValueError, RuntimeError)):
            self.validate()

    def test_actual_latest_commit_must_match_final_metric_checkpoint(self):
        checkpoint, commit = self.fixture['commits'][1]
        write_json(checkpoint / 'COMMITTED.json', commit | {'committed_ns': 2000000})
        with self.assertRaises((ValueError, RuntimeError)):
            self.validate()

    def test_missing_final_checkpoint_components_are_rejected(self):
        for filename in ('actor.pt', 'renderer.pt', 'training_state.pt'):
            path = self.fixture['checkpoint'] / filename
            saved = path.read_bytes()
            with self.subTest(filename=filename):
                path.unlink()
                with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                    self.validate()
                path.write_bytes(saved)


@unittest.skipIf(torch is None or save_file is None, 'Synthetic tensor fixtures require CPU torch and safetensors')
class ExportTensorFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = make_fixture(Path(self.temporary.name).resolve(), tensors=True)

    def collect(self):
        return exporter.collect_export(self.fixture['root'], self.fixture['run'], 'joint', expected_updates=2)

    def test_final_actor_policy_tensor_mismatch_is_rejected(self):
        path = self.fixture['stage'] / 'actor_policy/policy/adapter_model.safetensors'
        save_file({'synthetic.weight': torch.tensor([99.0])}, path)
        with self.assertRaises((ValueError, RuntimeError)):
            self.collect()

    def test_final_renderer_policy_tensor_mismatch_is_rejected(self):
        torch.save({'synthetic.weight': torch.tensor([99.0])}, self.fixture['stage'] / 'renderer_adapters/policy.pt')
        with self.assertRaises((ValueError, RuntimeError)):
            self.collect()

    def test_frozen_reference_tensor_mismatch_is_rejected(self):
        torch.save({'synthetic.weight': torch.tensor([99.0])}, self.fixture['stage'] / 'renderer_adapters/reference.pt')
        with self.assertRaises((ValueError, RuntimeError)):
            self.collect()

    def test_training_state_behavior_must_match_new_committed_version(self):
        path = self.fixture['checkpoint'] / 'training_state.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        state['behavior_version'] = 'joint:1'
        torch.save(state, path)
        with self.assertRaises((ValueError, RuntimeError)):
            self.collect()

    def test_successful_synthetic_export_preserves_recovery_dependencies_and_excludes_renders(self):
        root = self.fixture['root']
        output = root / 'delivery/priority-checkpoints/synthetic-joint-complete.tar.gz'
        manifest = root / 'deployment/synthetic-joint-export.json'
        exporter.export_completed(root, self.fixture['run'], 'joint', output, manifest,
            expected_updates=2, read_mib_per_second=16, set_priority=False)
        proof = json.loads(manifest.read_text())
        self.assertEqual(proof['status'], 'completed')
        self.assertEqual(proof['bytes'], output.stat().st_size)
        self.assertEqual(proof['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
        with tarfile.open(output, 'r:gz') as archive:
            names = set(archive.getnames())
        expected = [checkpoint / 'COMMITTED.json' for checkpoint, _ in self.fixture['commits'].values()]
        expected.extend(self.fixture['checkpoint'] / name for name in ('actor.pt', 'renderer.pt', 'training_state.pt'))
        expected.extend([self.fixture['stage'] / 'metrics.jsonl', self.fixture['stage'] / 'training_completed.json',
            self.fixture['run'] / 'sampler_selection.json', self.fixture['run'] / 'sft_reference/references.json',
            self.fixture['stage'] / 'actor_policy/policy/adapter_model.safetensors',
            self.fixture['stage'] / 'renderer_adapters/policy.pt',
            self.fixture['stage'] / 'renderer_adapters/reference.pt',
            Path(self.fixture['references']['actor_reference']) / 'adapter_model.safetensors',
            Path(self.fixture['references']['renderer_reference']),
            Path(self.fixture['config']['models']['actor_policy']) / 'adapter_model.safetensors',
            Path(self.fixture['config']['models']['renderer_policy'])])
        self.assertTrue({str(path.relative_to(root)) for path in expected} <= names)
        self.assertFalse(any('/renders/' in name for name in names))

    def test_agent_fixture_preserves_inactive_renderer_checkpoint(self):
        other_root = self.fixture['root'] / 'agent-project'
        fixture = make_fixture(other_root, mode='agent', tensors=True)
        plan = exporter.collect_export(other_root, fixture['run'], 'agent', expected_updates=2)
        self.assertIn(fixture['checkpoint'] / 'renderer.pt', plan['files'])
        self.assertEqual(plan['checkpoint'], fixture['checkpoint'])


if __name__ == '__main__':
    unittest.main()
