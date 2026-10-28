import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('controller_analysis', ROOT / 'scripts/analyze_controller_evaluation.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixtures():
    tasks = [{'task_id': f'development64:sample{i}', 'family_id': f'family{i}', 'split_id': 'development',
              'user_request': {'text': f'Change object{i}'}} for i in range(64)]
    reserved = [{'sample_id': f'sample{i}', 'family_id': f'family{i}', 'split_id': 'development',
                 'evaluation_index': i, 'instruction': f'Change object{i}'} for i in range(64)]
    return tasks, reserved


def episode(method, index, seed=11, success=True, preservation=.7):
    source = {'base': 'untrained_base', 'sft': 'completed_sft_exports'}.get(method, 'committed_rl_pair')
    return {'scope': 'actual_controller_episode', 'method': method, 'index': index, 'task_id': f'development64:sample{index}',
            'family_id': f'family{index}', 'seed': seed, 'private_target_used_for_actions': False, 'training_updates': 0,
            'sampler_parameters': {'steps': 50, 'eta': .15, 'cfg_scale': 4}, 'protocol_failure': False,
            'delivered_image_id': 'artifact:output', 'budget': {'actions': 3, 'renders': 1, 'invalid': 0},
            'terminal': {'success': success, 'preservation_status': 'measured' if preservation is not None else 'not_evaluated',
                         'preservation': preservation, 'atoms': [{'known': True}], 'terminal_reason': 'finished'},
            'loaded_state_verification': {'source': source, 'zero_output_projections_verified': True,
                                         'actor': {'exact_tensor_equality': True}, 'renderer': {'exact_tensor_equality': True}},
            'costs': {'observed_gpu_seconds': 10, 'observed_wall_seconds': 12, 'historical_measurements_complete': True},
            'private_scoring_seconds': 2}


class ControllerEvaluationAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tasks, self.reserved = fixtures()
        self.task_path = self.root / 'tasks.jsonl'
        self.reservation_path = self.root / 'reserved.jsonl'
        for path, rows in [(self.task_path, self.tasks), (self.reservation_path, self.reserved)]:
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.common = {'scope': 'actual_controller_episodes', 'tasks': self.tasks, 'image_seeds': list(analysis.SEEDS),
                       'private_target_used_for_actor_actions': False, 'training_updates_during_evaluation': 0,
                       'sampler_protocol': {'parameters': {'steps': 50, 'eta': .15, 'cfg_scale': 4}}}
        (self.root / 'common_protocol.json').write_text(json.dumps(self.common))

    def save_episode(self, row, path=None):
        method = row['method']
        destination = path or self.root / method / 'episodes' / f"{row['index']:03d}" / f"seed{row['seed']}" / 'episode.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(row))
        (self.root / method / 'registration.json').write_text(json.dumps({'method': method, 'scope': 'actual_controller_episodes',
            'common_protocol': self.common, 'evaluated_policy': {'method': method, 'source': row['loaded_state_verification']['source']}}))

    def analyze(self):
        return analysis.analyze(self.root, self.task_path, self.reservation_path, draws=100)

    def test_partial_coverage_does_not_become_full_benchmark_rate(self):
        self.save_episode(episode('sft', 0))
        report = self.analyze()
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['observed_total_episodes'], 1)
        sft = report['configurations']['sft']
        self.assertEqual(sft['observed_success_rate'], 1)
        self.assertIsNone(sft['full_registered_success_rate'])
        self.assertEqual(len(sft['missing']), 319)
        self.assertIsNone(report['configurations']['agent_rl']['observed_success_rate'])

    def test_identity_drift_duplicate_copy_and_unknown_seed_are_rejected(self):
        row = episode('sft', 0)
        self.save_episode(row)
        copy_path = self.root / 'sft/episodes/000/seed11/copied/episode.json'
        self.save_episode(row, copy_path)
        with self.assertRaisesRegex(ValueError, 'path or identity'):
            self.analyze()
        copy_path.unlink()
        for key, value in [('family_id', 'family1'), ('task_id', 'training:sample0'), ('seed', 7), ('index', True)]:
            broken = copy.deepcopy(row)
            broken[key] = value
            path = self.root / 'sft/episodes/000/seed11/episode.json'
            self.save_episode(broken, path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analyze()

    def test_training_overlap_and_duplicate_registered_family_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            analysis.registered_tasks(self.tasks, self.reserved, [[{'family_id': 'family0', 'task_id': 'train:different'}]])
        broken = copy.deepcopy(self.tasks)
        broken[1]['family_id'] = broken[0]['family_id']
        with self.assertRaisesRegex(ValueError, 'distinct'):
            analysis.registered_tasks(broken, self.reserved, [])

    def test_missing_preservation_and_costs_never_become_zero(self):
        first, second = episode('sft', 0, preservation=.8), episode('sft', 1, preservation=None)
        second['costs'] = None
        second.pop('private_scoring_seconds')
        second['terminal']['atoms'] = [{'known': False}]
        self.save_episode(first)
        self.save_episode(second)
        sft = self.analyze()['configurations']['sft']
        self.assertEqual(sft['preservation']['mean'], .8)
        self.assertEqual(sft['preservation']['missing_count'], 1)
        self.assertEqual(sft['costs']['observed_gpu_seconds']['mean'], 10)
        self.assertEqual(sft['costs']['observed_gpu_seconds']['missing_count'], 1)
        self.assertIsNone(sft['costs']['actual_dit_forwards_observed']['sum'])
        self.assertEqual(sft['unknown_atoms'], 1)

    def test_cluster_resampling_carries_all_seeds_and_reports_family_weights(self):
        values = [('many', 0., 1.)] * 5 + [('one', 1., 0.)]
        result = analysis.paired_statistics(values, 2000, 17)
        self.assertEqual(result['families'], 2)
        self.assertEqual(result['pairs'], 6)
        self.assertAlmostEqual(result['after_minus_before'], 2 / 3)
        self.assertEqual(result['equal_family_difference'], 0)
        self.assertEqual(result['family_cluster_95ci'], [-1, 1])
        self.assertEqual(result, analysis.paired_statistics(values, 2000, 17))
        self.assertIsNone(analysis.paired_statistics(values[:5], 100, 17)['family_cluster_95ci'])

    def test_complete_family_filter_precedes_measured_preservation_filter(self):
        before, after = {}, {}
        for index, seeds in [(0, analysis.SEEDS), (1, analysis.SEEDS[:4])]:
            for seed in seeds:
                before[f'family{index}', seed] = episode('sft', index, seed, False, .8)
                after[f'family{index}', seed] = episode('agent_rl', index, seed, True, .6)
        after['family0', 11]['terminal'].update(preservation_status='not_evaluated', preservation=None)
        result = analysis.comparison(before, after, 100, 17)
        self.assertEqual(result['complete_five_seed_families'], 1)
        complete = result['complete_five_seed_sensitivity']
        self.assertEqual(complete['success']['pairs'], 5)
        self.assertEqual(complete['preservation']['pairs'], 4)
        self.assertAlmostEqual(complete['preservation']['after_minus_before'], -.2)
        self.assertEqual(complete['preservation_missing_pairs'], 1)

    def test_false_policy_proof_and_missing_boolean_are_rejected(self):
        for mutation in ['policy', 'success', 'known']:
            row = episode('agent_rl', 0)
            if mutation == 'policy':
                row['loaded_state_verification']['renderer']['exact_tensor_equality'] = False
            elif mutation == 'success':
                row['terminal']['success'] = None
            else:
                row['terminal']['atoms'][0]['known'] = None
            self.save_episode(row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.analyze()

    def test_nonfinite_cost_and_false_measured_preservation_are_rejected(self):
        for mutation in ['cost', 'preservation']:
            row = episode('sft', 0)
            if mutation == 'cost':
                row['costs']['observed_gpu_seconds'] = float('nan')
            else:
                row['terminal']['preservation'] = None
            self.save_episode(row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.analyze()

    def test_policy_from_another_configuration_is_rejected(self):
        self.save_episode(episode('sft', 0))
        path = self.root / 'sft/registration.json'
        row = json.loads(path.read_text())
        row['evaluated_policy'] = {'method': 'agent_rl', 'source': 'committed_rl_pair'}
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'Policy registration'):
            self.analyze()


if __name__ == '__main__':
    unittest.main()
