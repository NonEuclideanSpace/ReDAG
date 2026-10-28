import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('formal_monitor_probe', Path(__file__).resolve().parents[1] / 'scripts/monitor_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FormalEvaluationProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / 'runs/development64-controller-episodes'
        self.root_patch = patch.object(PROBE, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_file(self, relative, payload='{}'):
        path = self.folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)

    def test_missing_root_is_unknown_and_existing_empty_root_is_zero(self):
        missing = PROBE.formal_evaluation_progress()
        self.assertIsNone(missing['episode_files'])
        self.assertIsNone(missing['all_episode_files_present'])
        self.assertTrue(all(row['episode_files'] is None for row in missing['methods']))
        self.folder.mkdir(parents=True)
        empty = PROBE.formal_evaluation_progress()
        self.assertEqual(empty['episode_files'], 0)
        self.assertFalse(empty['all_episode_files_present'])
        self.assertTrue(all(row['episode_files'] == 0 for row in empty['methods']))

    def test_only_registered_episode_files_count_without_reading_results(self):
        self.write_file('base/episodes/000/seed11/episode.json', '{"terminal":{"success":false}}')
        self.write_file('sft/episodes/063/seed101/episode.json', '{"terminal":{"success":true}}')
        self.write_file('sft/episodes/000/seed11/state.json')
        self.write_file('sft/scores/000/seed11/episode.json')
        self.write_file('sft/episodes/064/seed11/episode.json')
        self.write_file('sft/episodes/000/seed12/episode.json')
        self.write_file('sft/episodes/0/seed11/episode.json')
        self.write_file('agent_rl/episodes/000/seed11/episode.json.part')
        with patch.object(Path, 'read_text', side_effect=AssertionError('Episode contents must not be read by this count')):
            observed = PROBE.formal_evaluation_progress()
        self.assertEqual(observed['episode_files'], 2)
        self.assertEqual({row['method']: row['episode_files'] for row in observed['methods']},
            {'base': 1, 'sft': 1, 'agent_rl': 0, 'joint_rl': 0})
        self.assertFalse(observed['success_scores_included'])
        self.assertFalse(observed['training_updates_included'])

    def test_full_file_coverage_does_not_claim_result_or_training_validation(self):
        for method in ('base', 'sft', 'agent_rl', 'joint_rl'):
            for index in range(64):
                for seed in (11, 23, 47, 73, 101):
                    self.write_file(f'{method}/episodes/{index:03d}/seed{seed}/episode.json')
        observed = PROBE.formal_evaluation_progress()
        self.assertEqual(observed['episode_files'], observed['expected_episodes'])
        self.assertTrue(observed['all_episode_files_present'])
        self.assertTrue(all(row['episode_files'] == 320 for row in observed['methods']))
        self.assertEqual(observed['measurement'], 'saved_episode_file_count')
        self.assertFalse(observed['success_scores_included'])
        self.assertFalse(observed['training_updates_included'])
        self.assertNotIn('status', observed)
        self.assertNotIn('success_count', observed)


if __name__ == '__main__':
    unittest.main()
