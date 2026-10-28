import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import continue_gpu_queue_once as continuation


class QueueContinuationTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'pid': 316619, 'start_ticks': 12345, 'boot_id': 'boot-a',
            'cmdline': ['python', 'scripts/gpu_task_queue.py', '--hours', '6'],
            'cwd': '/workspace/pilot', 'exe': '/bin/python', 'state': 'S'}
        self.state = {'launch_attempted': False}
        self.saved = []
        self.launch = Mock(return_value=400000)

    def step(self, current, boot='boot-a'):
        return continuation.advance(self.expected, current, boot, self.state,
            lambda state: self.saved.append(copy.deepcopy(state)), self.launch)

    def test_live_original_supervisor_never_launches(self):
        self.assertEqual(self.step(self.expected.copy()), 'waiting_for_original_supervisor')
        self.launch.assert_not_called()
        self.assertFalse(self.state['launch_attempted'])

    def test_pid_reuse_and_identity_mismatch_refuse_launch(self):
        for change in ({'start_ticks': 54321}, {'cmdline': ['other']}, {'cwd': '/other'}, {'exe': '/other/python'}):
            with self.subTest(change=change):
                self.assertTrue(self.step(self.expected | change).startswith('refused_'))
        self.assertEqual(self.step(None, boot='boot-b'), 'refused_boot_identity')
        self.launch.assert_not_called()

    def test_natural_exit_commits_before_launch_and_launches_only_once(self):
        def launch():
            self.assertTrue(self.saved[-1]['launch_attempted'])
            self.assertEqual(self.saved[-1]['status'], 'launch_committed')
            return 400000
        self.launch.side_effect = launch
        self.assertEqual(self.step(None), 'continued')
        self.assertEqual(self.step(None), 'continued')
        self.launch.assert_called_once()
        self.assertEqual(self.state['new_supervisor_pid'], 400000)

    def test_busy_original_queue_lock_or_failed_spawn_never_retries(self):
        self.launch.side_effect = BlockingIOError('queue lock is held')
        self.assertEqual(self.step(None), 'launch_refused_or_failed')
        self.assertEqual(self.step(None), 'launch_refused_or_failed')
        self.launch.assert_called_once()
        self.assertTrue(self.state['launch_attempted'])


if __name__ == '__main__':
    unittest.main()
