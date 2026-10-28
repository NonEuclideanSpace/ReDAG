import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class FullActorSftOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        original_cwd = Path.cwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, original_cwd)
        self.root = Path.cwd()
        self.config = {
            'seed': 11, 'run_dir': 'runs/actor-full-200',
            'devices': {'actor': 'cuda:1', 'renderer': 'cuda:0',
                        'encoders': 'cuda:0', 'judge': 'cuda:1'},
            'models': {'actor_policy': 'fixture-initial-policy',
                       'actor_reference': None, 'renderer_reference': None},
            'sft': {'learning_rate': 0.00005},
            'data': {'actor_records': 'fixture-old-records.jsonl'},
            'renderer': {'steps': 50, 'eta': 0.3, 'cfg_scale': 4.0},
        }
        self.calls = {'admission': [], 'training': []}
        self.steps = list(range(1, 201))
        self.write_metrics = True
        self.admission = {
            'records': [{'record_id': f'action:{index}'} for index in range(20)],
            'sources': ['fixture-collection'],
            'counts': {
                'records': 20,
                'independent_families': 2,
                'phase_records': [5, 7, 5, 3],
                'phase_state_bundles': [1, 1, 1, 1],
            },
            'visual_admission': {'review_type': 'ai', 'human_reviewed': False},
            'identity_sha256': 'fixture-admission-identity',
        }
        self.policy = str(self.root / 'fixture-policy')
        self.entry = self.load_entry()

    def load_entry(self):
        def admit(collections, output, visual_reviews, *, seed, updates):
            self.calls['admission'].append((collections, output, visual_reviews, seed, updates))
            output.mkdir(parents=True, exist_ok=True)
            return copy.deepcopy(self.admission)

        def train(components, *, stage, updates, resume):
            self.calls['training'].append((copy.deepcopy(components.config), stage, updates, resume))
            if self.write_metrics:
                values = [{'step': step, 'record_ids': [f'action:{index}' for index in range(20)],
                           'family_ids': ['family:0', 'family:1']} for step in self.steps]
                write_rows(Path(components.config['run_dir']) / 'sft_actor/metrics.jsonl', values)
            return self.policy

        exports = {
            'yaml': {'safe_dump': lambda value, **options: json.dumps(value)},
            'agentic_image.config': {'load_config': lambda path: copy.deepcopy(self.config)},
            'agentic_image.data.io': {'read_jsonl': read_rows, 'write_jsonl': write_rows},
            'agentic_image.data.visual_admission': {'prepare_full_actor_admission': admit},
            'agentic_image.learning.transaction': {'atomic_json': write_json},
            'agentic_image.runtime.components': {
                'Components': lambda config: SimpleNamespace(config=config)},
            'agentic_image.runtime.training': {'train_actor_sft': train},
        }
        modules = {}
        for name, attributes in exports.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        spec = importlib.util.spec_from_file_location(
            'full_actor_sft_only_entry', ROOT / 'scripts/train_full_actor_sft.py')
        entry = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.object(sys, 'path', list(sys.path)):
            spec.loader.exec_module(entry)
        return entry

    def run_entry(self, *, sft_only=False, suffix='', resume=False):
        run_dir = 'runs/actor-full-seed23' if sft_only else 'runs/actor-full-200'
        output = 'data/actor-full-sft-seed23' if sft_only else 'data/actor-full-sft'
        run_dir += suffix
        output += suffix
        args = ['train_full_actor_sft.py', '--collection', 'fixture-collection',
                '--output', output, '--run-dir', run_dir, '--device', 'cuda:0',
                '--visual-reviews', 'fixture-visual-reviews.jsonl']
        if sft_only:
            args.append('--sft-only')
        if resume:
            args.append('--resume')
        with patch.object(sys, 'argv', args):
            self.entry.main()
        return self.root / output, self.root / run_dir

    def expected_main_registration(self):
        counts = self.admission['counts']
        return {
            'scope': 'full_four_phase_actor_sft_v1', 'records': counts['records'],
            'independent_families': counts['independent_families'],
            'phase_records': counts['phase_records'],
            'phase_state_bundles': counts['phase_state_bundles'],
            'phase_counts': [5, 7, 5, 3], 'phase_weights': [0.25, 0.35, 0.25, 0.15],
            'updates': 200, 'planned_action_exposures': 4000,
            'initial_actor_policy': self.config['models']['actor_policy'],
            'sources': self.admission['sources'],
            'teacher_origin': 'executed_public_rules_with_independent_private_positive_repair_filter',
            'review_type': 'ai', 'human_reviewed': False,
            'admission_identity_sha256': self.admission['identity_sha256'],
            'visual_admission': self.admission['visual_admission'],
        }

    def test_sft_only_requires_200_metrics_and_preserves_main_artifacts(self):
        self.config['seed'] = 23
        protected = {
            self.root / 'runs/full-actor-rl/sampler_selection.json': {'keep': 'main sampler'},
            self.root / 'data/actor-full-sft/rl_unfrozen.yaml': {'keep': 'main downstream'},
        }
        for path, value in protected.items():
            write_json(path, value)
        before = {path: path.read_bytes() for path in protected}
        output, run_dir = self.run_entry(sft_only=True, resume=True)
        registration = json.loads((output / 'registration.json').read_text())
        completed = json.loads((output / 'training_completed.json').read_text())
        for value in (registration, completed):
            self.assertEqual(value['scope'], 'independent_actor_sft_seed_replication_v1')
            self.assertEqual(value['experiment_scope'], 'same_data_training_order_and_seed_replication')
            self.assertEqual(value['training_seed'], 23)
            self.assertIs(value['independent_family_expansion'], False)
            self.assertIs(value['downstream_rl_planned'], False)
        self.assertEqual(completed['updates'], 200)
        self.assertEqual(completed['actual_action_exposures'], 4000)
        self.assertEqual(completed['actual_unique_actions_exposed'], 20)
        self.assertEqual(completed['actual_unique_families_exposed'], 2)
        self.assertIs(completed['reference_freeze_pending'], False)
        self.assertEqual(completed['metrics'], str(run_dir / 'sft_actor/metrics.jsonl'))
        self.assertEqual(self.calls['admission'][0][2:], ('fixture-visual-reviews.jsonl', 23, 200))
        training_config, stage, updates, resume = self.calls['training'][0]
        self.assertEqual((stage, updates, resume), ('actor', 200, True))
        self.assertEqual(training_config['data']['actor_records'], str(output / 'actor_records.jsonl'))
        self.assertEqual(training_config['sft']['phase_counts'], [5, 7, 5, 3])
        self.assertEqual(training_config['sft']['actor_updates'], 200)
        self.assertFalse((output / 'rl_unfrozen.yaml').exists())
        self.assertFalse((Path('fixture-collection') / 'sampler_selection.json').exists())
        self.assertEqual(before, {path: path.read_bytes() for path in protected})

    def test_default_preserves_registration_and_restores_downstream_devices(self):
        expected = self.expected_main_registration()
        write_json(self.root / 'data/actor-full-sft/registration.json', expected)
        selection = {'steps': 50, 'eta': 0.15, 'cfg_scale': 4.0, 'qualification': 'fixture'}
        write_json(Path('fixture-collection/sampler_selection.json'), selection)
        output, run_dir = self.run_entry(resume=True)
        registration = json.loads((output / 'registration.json').read_text())
        completed = json.loads((output / 'training_completed.json').read_text())
        self.assertEqual(registration, expected)
        self.assertEqual(completed, {
            'status': 'completed', 'scope': 'full_four_phase_actor_sft_v1',
            'updates': 200, 'policy': self.policy,
            'metrics': str(run_dir / 'sft_actor/metrics.jsonl'),
            'actual_action_exposures': 4000, 'actual_unique_actions_exposed': 20,
            'actual_unique_families_exposed': 2, 'phase_counts': [5, 7, 5, 3],
            'reference_freeze_pending': True,
        })
        training = json.loads((output / 'training.yaml').read_text())
        downstream = json.loads((output / 'rl_unfrozen.yaml').read_text())
        self.assertEqual(training['devices']['actor'], 'cuda:0')
        self.assertEqual(downstream['devices'], self.config['devices'])
        self.assertEqual(downstream['run_dir'], 'runs/full-actor-rl')
        self.assertEqual(downstream['models']['actor_policy'], self.policy)
        self.assertEqual(json.loads(Path('runs/full-actor-rl/sampler_selection.json').read_text()), selection)

    def test_short_or_noncontiguous_metrics_cannot_mark_completed(self):
        variants = [list(range(1, 200)), list(range(1, 200)) + [201],
                    list(range(1, 200)) + [199]]
        for index, steps in enumerate(variants):
            with self.subTest(steps=steps[-2:]):
                self.steps = steps
                with self.assertRaisesRegex(ValueError, 'complete committed metrics'):
                    self.run_entry(sft_only=True, suffix=f'-invalid-{index}')
                output = self.root / f'data/actor-full-sft-seed23-invalid-{index}'
                self.assertFalse((output / 'training_completed.json').exists())
                self.assertFalse((output / 'rl_unfrozen.yaml').exists())
                self.assertFalse((self.root / 'runs/full-actor-rl').exists())

    def test_missing_metrics_cannot_mark_completed(self):
        self.write_metrics = False
        with self.assertRaises(FileNotFoundError):
            self.run_entry(sft_only=True)
        output = self.root / 'data/actor-full-sft-seed23'
        self.assertFalse((output / 'training_completed.json').exists())
        self.assertFalse((self.root / 'runs/full-actor-rl').exists())

    def test_seed23_config_changes_only_seed_and_run_directory(self):
        base = (ROOT / 'configs/actor-full-200.yaml').read_text()
        replica = (ROOT / 'configs/actor-full-seed23.yaml').read_text()
        expected = base.replace('run_dir: runs/actor-full-200\n',
                                'run_dir: runs/actor-full-seed23\n', 1)
        expected = expected.replace('seed: 11\n', 'seed: 23\n', 1)
        self.assertEqual(replica, expected)


if __name__ == '__main__':
    unittest.main()
