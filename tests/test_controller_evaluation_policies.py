import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agentic_image.runtime.controller_evaluation import (
    committed_rl_policy, evaluated_policy, file_identity, freeze_json, load_evaluated_policy,
    policy_identity, sampler_protocol,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PolicyMetadataTests(unittest.TestCase):
    def test_sampler_keeps_qualification_separate_from_base_and_rejects_unaccepted_or_nonnumeric(self):
        tasks = [SimpleNamespace(model_dump=lambda **kw: {'task_id': 'fixture'})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = {name: {'path': f'qualified-{name}', 'bytes': 10, 'mtime_ns': 1} for name in ('actor', 'renderer')}
            for name, identity in identities.items():
                write(root / f'{name}_policy_identity.json', identity)
            write(root / 'registration.json', {'models': {'renderer': 'fixture-base'}, 'tasks': [{'task_id': 'fixture'}],
                'samplers': {'sde50_eta005': [50, .05]}, 'cfg_scale': 4., 'eval_seeds': [11, 23]})
            selection = {'accepted': True, 'status': 'accepted', 'candidate': 'sde50_eta005', 'steps': 50,
                'eta': .05, 'cfg_scale': 4., 'families': 64, 'eval_seeds': [11, 23], 'paired_records': 256,
                'decisions': {'sde50_eta005': {'accepted': True}}, 'renderer_policy_identity': identities['renderer']}
            path = root / 'sampler_selection.json'
            write(path, selection)
            protocol = sampler_protocol(path, tasks)
            base = evaluated_policy({'models': {}}, 'base', protocol, root / 'missing-training-proof')
            self.assertEqual(base['source'], 'untrained_base')
            self.assertEqual(protocol['qualified_on']['policies'], identities)
            for patch in ({'accepted': False}, {'eta': '0.05'}, {'eta': float('nan')}, {'steps': True}, {'steps': 28}):
                with self.subTest(patch=patch):
                    write(path, selection | patch)
                    with self.assertRaises(ValueError):
                        sampler_protocol(path, tasks)

    def test_common_protocol_cannot_silently_change_between_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'common.json'
            freeze_json(path, {'eta': .05})
            freeze_json(path, {'eta': .05})
            with self.assertRaises(ValueError):
                freeze_json(path, {'eta': .3})
            self.assertEqual(json.loads(path.read_text()), {'eta': .05})

    def test_expanded_qualification_keeps_original_failures_and_rejects_overlap_or_relaxed_gate(self):
        original_tasks = [{'task_id': f'old:{i}', 'family_id': f'family:{i}'} for i in range(64)]
        tasks = [SimpleNamespace(model_dump=lambda row=row, **kw: row) for row in original_tasks]
        confirmation_tasks = [{'task_id': f'new:{i}', 'family_id': f'family:{i+64}'} for i in range(64)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'path': 'qualified.pt', 'bytes': 10, 'mtime_ns': 1}
            for role in ('actor', 'renderer'):
                write(root / f'{role}_policy_identity.json', identity)
            registration = {'models': {'renderer': 'fixed-sft'}, 'tasks': original_tasks,
                'samplers': {'sde50_eta005': [50, .05]}, 'cfg_scale': 4., 'eval_seeds': [101, 211]}
            write(root / 'registration.json', registration)
            confirmation_path = root / 'confirmation.json'
            write(confirmation_path, registration | {'tasks': confirmation_tasks})
            margins = {'success_margin': .05, 'defect_margin': .05, 'protection_margin': .05}
            selection = {'accepted': True, 'status': 'accepted', 'candidate': 'sde50_eta005', 'steps': 50,
                'eta': .05, 'cfg_scale': 4., 'families': 128, 'paired_records': 512, 'eval_seeds': [101, 211],
                'decision': margins | {'accepted': True}, 'original_decision': margins | {'accepted': False},
                'independent_confirmation_decision': margins | {'accepted': True}, 'renderer_policy_identity': identity,
                'qualified_on': {'families': 128, 'original_families': 64, 'confirmation_families': 64,
                    'original_registration': str(root / 'registration.json'), 'confirmation_registration': str(confirmation_path)}}
            path = root / 'sampler_selection.json'
            write(path, selection)
            protocol = sampler_protocol(path, tasks)
            self.assertEqual(protocol['qualified_on']['comparison_families'], 128)
            self.assertFalse(protocol['selection']['original_decision']['accepted'])
            write(confirmation_path, registration)
            with self.assertRaises(ValueError):
                sampler_protocol(path, tasks)
            write(confirmation_path, registration | {'tasks': confirmation_tasks})
            selection['decision']['success_margin'] = .08
            write(path, selection)
            with self.assertRaises(ValueError):
                sampler_protocol(path, tasks)

    def fixture(self, root):
        actor = root / 'sft/policy'
        actor.mkdir(parents=True)
        (actor / 'adapter_config.json').write_text('{}')
        (actor / 'adapter_model.safetensors').write_bytes(b'fixture')
        renderer = root / 'renderer.pt'
        renderer.write_bytes(b'fixture')
        config = {'run_dir': str(root), 'models': {'actor': 'actor', 'actor_revision': 'a1', 'renderer': 'renderer',
            'renderer_revision': 'r1', 'renderer_profile': 'small', 'actor_policy': str(actor), 'renderer_policy': str(renderer)},
            'renderer': {'steps': 50, 'eta': .05, 'cfg_scale': 4.}}
        folder = root / 'rl_agent'
        transaction = folder / 'checkpoints/commit1'
        transaction.mkdir(parents=True)
        for name in ('actor.pt', 'renderer.pt', 'training_state.pt'):
            (transaction / name).write_bytes(b'fixture')
        counts = {'actor': 1, 'renderer': 0, 'required_components': 1}
        completion = {'status': 'completed', 'mode': 'agent', 'rounds': 1, 'required_effective_updates': 1,
            'effective_target_met': True, 'effective_update_counts': counts}
        write(folder / 'training_completed.json', completion)
        (folder / 'metrics.jsonl').write_text(json.dumps({'round': 1, 'checkpoint': str(transaction), 'effective_update_counts': counts}) + '\n')
        marker = transaction / 'COMMITTED.json'
        write(marker, {'status': 'COMMITTED', 'transaction_id': 'commit1', 'metadata': {'mode': 'agent', 'config': copy.deepcopy(config)},
            'new_version': 'agent:1', 'components': {'actor': 'actor.pt', 'renderer': 'renderer.pt'}})
        return config, marker, completion

    def test_rl_resolves_final_actual_commit_and_never_treats_sft_config_as_rl_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config, marker, completion = self.fixture(Path(directory))
            policy = committed_rl_policy(config, 'agent_rl')
            self.assertEqual(policy['source'], 'committed_rl_pair')
            self.assertEqual(policy['commit_path'], str(marker.resolve()))
            self.assertNotEqual(policy['components']['actor']['path'], config['models']['actor_policy'])
            with self.assertRaises(ValueError):
                committed_rl_policy(config, 'agent_rl', marker.parent.parent / 'other')
            write(marker, json.loads(marker.read_text()) | {'status': 'PREPARED'})
            with self.assertRaises(ValueError):
                committed_rl_policy(config, 'agent_rl')

    def test_effective_target_32_accepts_40_contiguous_rounds_bound_to_final_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            config, marker, completion = self.fixture(Path(directory))
            folder = marker.parent.parent.parent
            counts = {'actor': 32, 'renderer': 0, 'required_components': 32}
            completion.update(rounds=40, required_effective_updates=32, effective_update_counts=counts,
                              maximum_rounds=96, effective_target_enabled=True)
            write(folder / 'training_completed.json', completion)
            rows = [{'round': index, 'checkpoint': str(marker.parent), 'effective_update_counts': counts}
                    for index in range(1, 41)]
            (folder / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            write(marker, json.loads(marker.read_text()) | {'new_version': 'agent:40'})
            result = committed_rl_policy(config, 'agent_rl')
            self.assertEqual(result['completion']['rounds'], 40)
            self.assertEqual(result['completion']['required_effective_updates'], 32)
            write(folder / 'training_completed.json', completion | {'rounds': 32})
            with self.assertRaises(ValueError):
                committed_rl_policy(config, 'agent_rl')

    def test_sft_requires_completed_four_phase_export_and_final_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _ = self.fixture(root)
            sampler = {'qualified_on': {'policies': {'renderer': policy_identity(config['models']['renderer_policy'], 'renderer')}}}
            completion_path = root / 'sft_completed.json'
            with self.assertRaises(FileNotFoundError):
                evaluated_policy(config, 'sft', sampler, completion_path)
            transaction = root / 'sft_commit'
            transaction.mkdir()
            (transaction / 'actor.pt').write_bytes(b'fixture')
            write(transaction / 'COMMITTED.json', {'status': 'COMMITTED', 'new_version': 'actor:200',
                'optimizer_steps': ['actor'], 'components': {'actor': 'actor.pt'}})
            metrics = root / 'sft_metrics.jsonl'
            metrics.write_text(''.join(json.dumps({'step': index, 'checkpoint': str(transaction)}) + '\n' for index in range(1, 201)))
            completion = {'status': 'completed', 'scope': 'full_four_phase_actor_sft_v1', 'updates': 200,
                'policy': config['models']['actor_policy'], 'metrics': str(metrics)}
            write(completion_path, completion)
            self.assertEqual(evaluated_policy(config, 'sft', sampler, completion_path)['source'], 'completed_sft_exports')
            write(completion_path, completion | {'scope': 'public_protocol_only'})
            with self.assertRaises(ValueError):
                evaluated_policy(config, 'sft', sampler, completion_path)

    def test_rl_incomplete_effective_updates_or_changed_sampler_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            config, marker, completion = self.fixture(Path(directory))
            config['renderer']['eta'] = .3
            with self.assertRaises(ValueError):
                committed_rl_policy(config, 'agent_rl')
            config['renderer']['eta'] = .05
            write(marker.parent.parent.parent / 'training_completed.json', completion | {'effective_target_met': False})
            with self.assertRaises(ValueError):
                committed_rl_policy(config, 'agent_rl')

    def test_queue_has_full_coverage_and_base_does_not_depend_on_full_training(self):
        path = Path(__file__).parents[1] / 'scripts/prepare_controller_evaluation_jobs.py'
        spec = importlib.util.spec_from_file_location('controller_jobs', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        jobs = module.jobs('python', 'output')
        self.assertEqual(len(jobs), 160)
        self.assertEqual(len({row['id'] for row in jobs}), 160)
        for row in jobs:
            self.assertEqual(row['minimum_free_gib'], 40)
            self.assertEqual(row['exclusive_group'], f"renderer-evaluation-gpu{row['gpus'][0]}")
            if '-base-' in row['id']:
                self.assertNotIn('after', row)
                self.assertFalse(any('full-actor' in value or 'actor-full' in value for value in row['requires']))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required for actual tensor restoration tests')
class TensorRestorationTests(unittest.TestCase):
    def test_rl_loads_both_committed_models_without_reference_or_optimizer_and_checks_values(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({'model': {'policy': {'weight': torch.tensor([2.])}, 'reference': {'weight': torch.tensor([99.])}}}, root / 'actor.pt')
            torch.save({'model': {'weight': torch.tensor([3.])}}, root / 'renderer.pt')
            torch.save({'behavior_version': 'joint:1', 'data_cursor': {'round': 1}}, root / 'training_state.pt')
            policy = {'source': 'committed_rl_pair', 'components': {name: file_identity(root / f'{name}.pt') for name in ('actor', 'renderer')},
                'training_state': file_identity(root / 'training_state.pt'), 'commit': {'new_version': 'joint:1'}, 'completion': {'rounds': 1}}
            state = {'actor': {'policy': {'weight': torch.tensor([0.])}}, 'renderer': {'weight': torch.tensor([0.])}}
            actor = SimpleNamespace(adapter_state=lambda: state['actor'], restore_adapter_state=lambda value: state.update(actor=value))
            renderer = SimpleNamespace(policy_state=lambda: state['renderer'], load_policy_state=lambda value: state.update(renderer=value))
            report = load_evaluated_policy(SimpleNamespace(actor=actor, renderer=renderer), policy)
            self.assertEqual(set(state['actor']), {'policy'})
            self.assertEqual(state['actor']['policy']['weight'].item(), 2.)
            self.assertEqual(state['renderer']['weight'].item(), 3.)
            self.assertTrue(report['actor']['exact_tensor_equality'])
            actor.restore_adapter_state = lambda value: None
            state['actor']['policy']['weight'] = torch.tensor([0.])
            with self.assertRaises(ValueError):
                load_evaluated_policy(SimpleNamespace(actor=actor, renderer=renderer), policy)


if __name__ == '__main__':
    unittest.main()
