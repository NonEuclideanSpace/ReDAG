import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('first_batch_watcher', ROOT / 'scripts/watch_first_controller_batch.py')
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


class FirstBatchWatcherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'deployment').mkdir()
        self.volume = self.root / 'volume'
        self.external = self.volume / 'backup'
        (self.external / 'reports').mkdir(parents=True)
        self.addCleanup(patch.stopall)
        patch.object(watcher, 'VOLUME', self.volume).start()
        patch.object(watcher, 'EXTERNAL', self.external).start()
        patch.object(watcher.os.path, 'ismount', return_value=True).start()
        self.manifest = {'schema': 'first_batch_archive_receipt_v1', 'status': 'completed',
            'source': watcher.ITEM['source'], 'archive': watcher.ITEM['source'], 'bytes': 7, 'mtime': 1234,
            'sha256': 'a' * 64, 'source_files_unchanged': True, 'full1280_complete': False, 'gpu_work_started': False}

    def paths(self):
        target = self.external / 'remote/first-controller-batch/first-controller-batch-8x4.tar.gz'
        report = self.root / 'deployment/first-controller-batch-8x4.transfer.json'
        return target, report

    def completed_local(self):
        target, report = self.paths()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'archive')
        os.utime(target, (1234, 1234))
        report.write_text(json.dumps({'status': 'completed', 'source_checksum_verified': True,
            'source': watcher.ITEM['source'], 'destination': str(target), 'expected_bytes': 7,
            'sha256': self.manifest['sha256'], 'finished_at': 1235}))
        return target, report

    def test_manifest_requires_exact_source_schema_and_completed_evidence(self):
        watcher.validate_manifest(self.manifest)
        invalid = {'source': '/another/archive.tar.gz', 'archive': '/another/archive.tar.gz', 'schema': 'other',
            'status': 'running', 'mtime': True, 'bytes': 0, 'sha256': 'A' * 64,
            'source_files_unchanged': False, 'full1280_complete': True, 'gpu_work_started': True}
        for key, value in invalid.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                watcher.validate_manifest(dict(self.manifest, **{key: value}))
        without_time = dict(self.manifest)
        del without_time['mtime']
        with self.assertRaises(ValueError):
            watcher.validate_manifest(without_time)

    def test_capacity_includes_new_worker_and_rejects_same_source(self):
        self.assertTrue(watcher.transfer_capacity_available([{'workers': 6, 'source': 'other'}]))
        self.assertFalse(watcher.transfer_capacity_available([{'workers': 7, 'source': 'other'}]))
        self.assertFalse(watcher.transfer_capacity_available([{'workers': 1, 'source': watcher.ITEM['source']}]))
        with self.assertRaises(ValueError):
            watcher.transfer_capacity_available([{'workers': 0, 'source': 'other'}])
        target, report = self.paths()
        args = watcher.command(self.root, self.manifest, target, report)
        self.assertEqual(args[args.index('--workers') + 1], '1')
        self.assertEqual(args[args.index('--control') + 1], watcher.CONTROL)
        self.assertEqual(args[args.index('--mtime') + 1], '1234')

    def test_verified_existing_archive_is_not_downloaded_and_receipts_are_copied(self):
        self.completed_local()
        with patch.object(Path, 'is_socket', return_value=True), patch.object(watcher, 'read_remote', return_value=self.manifest), \
                patch.object(watcher.subprocess, 'Popen') as launch, patch.object(watcher, 'active_direct_downloads') as active:
            result = watcher.watch(self.root, .1)
        self.assertEqual(result['status'], 'already_verified')
        launch.assert_not_called()
        active.assert_not_called()
        self.assertEqual(json.loads((self.external / 'reports/first-controller-batch-8x4.received.json').read_text()), self.manifest)
        self.assertTrue((self.external / 'reports/first-controller-batch-8x4.transfer.json').is_file())

    def test_same_bytes_with_changed_timestamp_is_not_already_verified(self):
        target, report = self.completed_local()
        self.assertTrue(watcher.verified(self.manifest, target, report))
        os.utime(target, (1233, 1233))
        self.assertFalse(watcher.verified(self.manifest, target, report))

    def test_busy_capacity_waits_until_bound_without_starting_a_duplicate(self):
        for active in ([{'workers': 7, 'source': 'other'}], [{'workers': 1, 'source': watcher.ITEM['source']}]):
            clock = [0]
            def sleep(seconds):
                clock[0] += seconds
            with patch.object(Path, 'is_socket', return_value=True), \
                    patch.object(watcher, 'read_remote', return_value=self.manifest), \
                    patch.object(watcher, 'active_direct_downloads', return_value=active), \
                    patch.object(watcher.time, 'monotonic', side_effect=lambda: clock[0]), \
                    patch.object(watcher.time, 'sleep', side_effect=sleep), \
                    patch.object(watcher.subprocess, 'Popen') as launch:
                result = watcher.watch(self.root, .01)
            self.assertEqual(result['status'], 'bounded_watch_ended')
            self.assertLessEqual(clock[0], 36)
            launch.assert_not_called()

    def test_disk_device_change_unmount_or_redirect_is_rejected(self):
        device = watcher.check_disk()
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            watcher.check_disk(device + 1)
        with patch.object(watcher.os.path, 'ismount', return_value=False), self.assertRaisesRegex(RuntimeError, 'not mounted'):
            watcher.check_disk(device)
        (self.external / 'remote').symlink_to(self.root)
        with self.assertRaisesRegex(RuntimeError, 'delivery directory'):
            watcher.check_disk(device)

    def test_independent_lock_prevents_duplicate_watcher_before_status_write(self):
        status = self.root / 'deployment/first-controller-batch-transfer-status.json'
        status.write_text('existing status')
        with (self.root / 'deployment/first-controller-batch-transfer.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                watcher.watch(self.root, .1)
        self.assertEqual(status.read_text(), 'existing status')

    def test_time_budget_is_finite_and_at_most_three_hours(self):
        for value in (0, -1, 3.01, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                watcher.watch(self.root, value)
        with patch.object(watcher, 'read_remote') as remote, patch.object(watcher.subprocess, 'Popen') as launch:
            result = watcher.watch(self.root, .001)
        self.assertEqual(result['status'], 'bounded_watch_ended')
        remote.assert_not_called()
        launch.assert_not_called()

    def test_bounded_cleanup_stops_only_its_own_download_group(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired('download', 1)
        with patch.object(watcher.os, 'killpg') as stop, patch.object(watcher.time, 'monotonic', return_value=10):
            watcher.stop_child(child, 11)
        self.assertEqual(stop.call_args_list[0].args, (123, signal.SIGTERM))
        self.assertEqual(stop.call_args_list[1].args, (123, signal.SIGKILL))
        child.wait.assert_called_once_with(timeout=1)


if __name__ == '__main__':
    unittest.main()
