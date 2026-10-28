from contextlib import contextmanager
import json
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'package_rental_results.py'
HOLDER = '''import fcntl
from pathlib import Path
import sys
with Path(sys.argv[1]).open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('locked', flush=True)
    sys.stdin.buffer.read(1)
'''


class PackageRentalLockWaitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.project = root / 'project'
        self.output = root / 'archives'
        self.project.mkdir()
        self.output.mkdir()
        self.payload = b'{"real_sample": "saved-training-result"}\n'
        (self.project / 'result.json').write_bytes(self.payload)
        self.sentinel = b'{ "status": "existing_pass", "unchanged": true }\n'
        (self.output / 'status.json').write_bytes(self.sentinel)
        self.lock_sentinel = b'existing lock metadata\n'
        (self.output / '.pack.lock').write_bytes(self.lock_sentinel)

    def command(self, *arguments):
        return [sys.executable, str(SCRIPT), '--project', str(self.project),
            '--output', str(self.output), '--read-mib-per-second', '128',
            '--chunk-mib', '1', *arguments]

    def assert_untouched(self):
        self.assertEqual((self.output / 'status.json').read_bytes(), self.sentinel)
        self.assertEqual((self.output / '.pack.lock').read_bytes(), self.lock_sentinel)
        self.assertEqual({path.name for path in self.output.iterdir()}, {'.pack.lock', 'status.json'})

    @contextmanager
    def held_lock(self):
        holder = subprocess.Popen([sys.executable, '-u', '-c', HOLDER,
            str(self.output / '.pack.lock')], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(holder.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(timeout=5), 'Lock holder did not become ready')
            self.assertEqual(holder.stdout.readline(), b'locked\n')
            yield holder
        finally:
            if holder.poll() is None:
                holder.stdin.write(b'x')
                holder.stdin.flush()
            holder.communicate(timeout=5)

    def test_default_contention_fails_promptly_without_mutating_output(self):
        with self.held_lock():
            started = time.monotonic()
            result = subprocess.run(self.command(), capture_output=True, text=True, timeout=3)
            elapsed = time.monotonic() - started
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('BlockingIOError', result.stderr)
            self.assertLess(elapsed, 2)
            self.assert_untouched()

    def test_positive_wait_packs_only_after_other_process_releases_lock(self):
        with self.held_lock() as holder:
            packer = subprocess.Popen(self.command('--lock-wait-seconds', '3'),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                time.sleep(0.3)
                self.assertIsNone(packer.poll(), 'Packer exited before the lock was released')
                self.assert_untouched()
                holder.stdin.write(b'x')
                holder.stdin.flush()
                self.assertEqual(holder.wait(timeout=3), 0)
                stdout, stderr = packer.communicate(timeout=5)
                self.assertEqual(packer.returncode, 0, stderr)
                result = json.loads(stdout)
                self.assertEqual(result['status'], 'snapshot_catalog_caught_up')
                self.assertGreater(result['new_archives'], 0)
                passes = list(self.output.glob('*.pass.json'))
                self.assertEqual(len(passes), 1)
                completed = json.loads(passes[0].read_text())
                self.assertEqual(completed['status'], 'snapshot_catalog_caught_up')
                self.assertEqual(completed['pending_changed_or_new_paths'], [])
                self.assertEqual(completed['errors'], [])
                self.assertEqual(completed['source_bytes_read'], len(self.payload))
                self.assertTrue(list(self.output.glob('*.tar.gz')))
                self.assertEqual((self.output / '.pack.lock').read_bytes(), self.lock_sentinel)
            finally:
                if packer.poll() is None:
                    packer.kill()
                packer.communicate(timeout=5)

    def test_timeout_preserves_existing_status_without_creating_any_pass(self):
        with self.held_lock():
            started = time.monotonic()
            result = subprocess.run(self.command('--lock-wait-seconds', '0.25'),
                capture_output=True, text=True, timeout=3)
            elapsed = time.monotonic() - started
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('TimeoutError', result.stderr)
            self.assertIn('lock', result.stderr.lower())
            self.assertGreaterEqual(elapsed, 0.20)
            self.assertLess(elapsed, 2)
            self.assert_untouched()

    def test_invalid_wait_values_are_rejected_without_mutating_output(self):
        for value in ('-1', 'nan', 'inf', '-inf'):
            with self.subTest(value=value):
                result = subprocess.run(self.command('--lock-wait-seconds=' + value),
                    capture_output=True, text=True, timeout=3)
                self.assertNotEqual(result.returncode, 0)
                self.assert_untouched()


if __name__ == '__main__':
    unittest.main()
