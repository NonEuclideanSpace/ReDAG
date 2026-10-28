import contextlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "guard_evaluation_overlap_tested", ROOT / "scripts/guard_evaluation_overlap.py"
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class GuardEvaluationOverlapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "deployment").mkdir()
        self.job_id = "development64-controller-fixture"
        self.command = ["/fixture/python", "scripts/evaluate_development_controller.py", "--fixture"]
        self.job = {
            "id": self.job_id,
            "command": self.command,
            "pid_file": "deployment/evaluation.pid",
            "exit_file": "deployment/evaluation.exit.json",
        }
        self.owner = {
            "id": "full-joint-rl32",
            "exit_file": "deployment/primary.exit.json",
        }
        (self.root / "deployment/gpu-jobs.json").write_text(json.dumps({"jobs": [self.job, self.owner]}))
        (self.root / self.job["pid_file"]).write_text("999")
        self.output = self.root / "deployment" / (self.job_id + ".overlap-guard.json")
        self.identity = {
            "pid": 101,
            "ppid": 999,
            "start_ticks": 42,
            "cmdline": self.command,
            "cwd": str(self.root),
            "exe": "/fixture/python",
        }

    def run_guard(self, seconds=6, free_memory="40960\n", expected_error=None):
        clock = {"elapsed": 0.0}
        sleeps = []
        original_iterdir = Path.iterdir
        original_read_text = Path.read_text

        def iterdir(path):
            return iter([Path("/proc/101")]) if path == Path("/proc") else original_iterdir(path)

        def read_text(path, *args, **kwargs):
            if path == Path("/proc/101/stat"):
                return "101 (python) S 999 0"
            return original_read_text(path, *args, **kwargs)

        def sleep(seconds):
            self.assertGreater(seconds, 0)
            sleeps.append(seconds)
            clock["elapsed"] += seconds
            self.assertLess(len(sleeps), 20, "Guard did not respect its deadline")

        arguments = ["guard_evaluation_overlap.py", "--job", self.job_id, "--seconds", str(seconds)]
        with contextlib.chdir(self.root), \
                patch.object(sys, "argv", arguments), \
                patch.object(Path, "iterdir", iterdir), \
                patch.object(Path, "read_text", read_text), \
                patch.object(guard, "process_identity", return_value=self.identity) as identity, \
                patch.object(guard.time, "monotonic", side_effect=lambda: clock["elapsed"]), \
                patch.object(guard.time, "time", side_effect=lambda: 1000 + clock["elapsed"]), \
                patch.object(guard.time, "sleep", side_effect=sleep), \
                patch.object(guard.subprocess, "check_output", return_value=free_memory) as query, \
                patch.object(guard.os, "kill") as kill:
            if expected_error is None:
                exit_code = guard.main()
            else:
                with self.assertRaises(expected_error):
                    guard.main()
                exit_code = None
            state = json.loads(self.output.read_text())
        return {"exit_code": exit_code, "state": state, "kill": kill, "query": query,
            "identity": identity, "elapsed": clock["elapsed"], "sleeps": sleeps}

    def worker_exit(self, payload):
        (self.root / self.job["exit_file"]).write_text(json.dumps(payload))

    def test_worker_nonzero_exit_returns_failure_without_signal(self):
        payload = {"returncode": 7, "reason": "evaluation failed", "finished_at": 900}
        self.worker_exit(payload)
        result = self.run_guard()
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["state"]["status"], "worker_exited")
        self.assertEqual(result["state"]["guard_exit_code"], 1)
        self.assertEqual(result["state"]["exit"], payload)
        result["kill"].assert_not_called()
        result["query"].assert_not_called()
        result["identity"].assert_not_called()
        self.assertEqual(result["sleeps"], [])

    def test_worker_integer_zero_exit_returns_success_without_signal(self):
        payload = {"returncode": 0, "finished_at": 900}
        self.worker_exit(payload)
        result = self.run_guard()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["state"]["guard_exit_code"], 0)
        self.assertEqual(result["state"]["status"], "worker_exited")
        self.assertEqual(result["state"]["exit"], payload)
        result["kill"].assert_not_called()
        result["query"].assert_not_called()

    def test_worker_exit_requires_strict_integer_zero(self):
        for payload in ({"returncode": False}, {"returncode": True}, {"returncode": 0.0},
                        {"returncode": "0"}, {"returncode": None}, {}, {"exit_code": 0}):
            with self.subTest(payload=payload):
                if self.output.exists():
                    self.output.unlink()
                self.worker_exit(payload)
                result = self.run_guard()
                self.assertEqual(result["exit_code"], 1)
                self.assertEqual(result["state"]["guard_exit_code"], 1)
                self.assertEqual(result["state"]["exit"], payload)
                result["kill"].assert_not_called()
                result["query"].assert_not_called()

    def test_two_low_memory_samples_withdraw_once_and_return_failure(self):
        result = self.run_guard(seconds=30, free_memory="2048\n")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["state"]["status"], "withdrawn_for_primary")
        self.assertEqual(result["state"]["guard_exit_code"], 1)
        self.assertEqual(result["state"]["reason"], "gpu1_free_memory_below_guard_on_two_samples")
        self.assertEqual(result["state"]["signal"], "SIGTERM")
        self.assertEqual(result["state"]["samples"], 2)
        self.assertEqual(result["state"]["minimum_free_gib"], 2)
        self.assertEqual(result["state"]["worker"], self.identity)
        result["kill"].assert_called_once_with(101, signal.SIGTERM)
        self.assertEqual(result["query"].call_count, 2)
        self.assertEqual(result["identity"].call_count, 2)
        self.assertEqual(result["elapsed"], 6)

    def test_deadline_with_unfinished_worker_returns_failure_without_signal(self):
        result = self.run_guard(seconds=6)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["state"]["status"], "bounded_watch_ended")
        self.assertEqual(result["state"]["guard_exit_code"], 1)
        self.assertEqual(result["state"]["samples"], 1)
        self.assertNotIn("exit", result["state"])
        self.assertEqual(result["elapsed"], 6)
        result["kill"].assert_not_called()

    def test_existing_evidence_rejects_repeat_without_signals_or_queries(self):
        evidence = {"status": "previous evidence", "guard_exit_code": 1}
        self.output.write_text(json.dumps(evidence))
        result = self.run_guard(expected_error=FileExistsError)
        self.assertEqual(result["state"], evidence)
        self.assertEqual(result["sleeps"], [])
        result["kill"].assert_not_called()
        result["query"].assert_not_called()
        result["identity"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
