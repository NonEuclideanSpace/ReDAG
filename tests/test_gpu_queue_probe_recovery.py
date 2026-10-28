import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('queue_probe_recovery', ROOT / 'scripts/gpu_task_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class ProbeRecoveryTests(unittest.TestCase):
    def test_query_failure_keeps_supervisor_alive_and_does_not_launch_jobs(self):
        for error in (subprocess.TimeoutExpired('nvidia-smi', 5), subprocess.CalledProcessError(1, 'nvidia-smi'), ValueError('malformed GPU readout')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'deployment').mkdir()
                jobs = root / 'deployment/jobs.json'
                jobs.write_text('{"jobs":[]}')
                healthy = {0: {'uuid': 'test', 'free_gib': 40.0, 'utilization': 80.0, 'busy': False}}
                with contextlib.chdir(root), contextlib.redirect_stdout(io.StringIO()), patch.object(queue, 'gpu_state', side_effect=[error, healthy]) as probe, patch.object(queue.time, 'sleep'), patch.object(queue.subprocess, 'Popen') as spawn:
                    queue.supervise(jobs, 1 / 3600)
                self.assertEqual(probe.call_count, 2)
                spawn.assert_not_called()
                failures = [json.loads(line) for line in (root / 'deployment/gpu-queue-probe-errors.jsonl').read_text().splitlines()]
                self.assertEqual(len(failures), 1)
                self.assertFalse(failures[0]['new_jobs_admitted'])
                health = json.loads((root / 'deployment/gpu-queue-health.json').read_text())
                self.assertEqual(health['status'], 'healthy')
                self.assertEqual(health['probe_failures'], 1)
                self.assertEqual(health['consecutive_failures'], 0)
                self.assertEqual(len((root / 'deployment/gpu-utilization.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
