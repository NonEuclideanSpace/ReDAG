import fcntl
from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import archive_reclamation as reclamation
import package_rental_results as packager
import sync_rental_result_archives as sync


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ReclamationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.delivery = self.root / 'delivery/project-results'
        self.delivery.mkdir(parents=True)
        self.name = '1793135136926454764-00001-runs-fixture.tar.gz'
        self.archive = self.delivery / self.name
        self.archive.write_bytes(b'fixture archive payload')
        self.proof = {'archive': self.name, 'archive_bytes': self.archive.stat().st_size,
            'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(), 'group': 'runs/fixture',
            'source_bytes': 40, 'completed_at': 1234.,
            'members': [{'path': 'runs/fixture/result.json', 'stable': True,
                'before': {'size': 40, 'mtime_ns': 999}}]}
        self.manifest = self.delivery / reclamation.manifest_name(self.name)
        write(self.manifest, self.proof)
        destination = '/Volumes/TOSHIBA EXT/fixture/' + self.name
        source = str(self.archive)
        self.receipt = {'archive': self.name, 'archive_bytes': self.proof['archive_bytes'],
            'sha256': self.proof['sha256'], 'manifest_sha256': reclamation.manifest_digest(self.proof),
            'source': source, 'destination': destination, 'external_verification': {
                'status': 'completed', 'method': 'stream_sha256', 'bytes': self.proof['archive_bytes'],
                'sha256': self.proof['sha256'], 'source': source, 'destination': destination,
                'verified_at': time.time()}}
        self.request = {'schema': reclamation.REQUEST_SCHEMA, 'delivery_directory': str(self.delivery),
            'receipts': [self.receipt]}

    def test_missing_archive_without_receipt_is_rejected(self):
        self.archive.unlink()
        with self.assertRaises(ValueError):
            packager.load_catalog(self.delivery)
        with self.assertRaisesRegex(ValueError, 'no previously persisted'):
            reclamation.reclaim(self.delivery, self.request)

    def test_valid_reclamation_preserves_manifest_catalog_and_archive_entry(self):
        before = self.manifest.read_bytes()
        result = reclamation.reclaim(self.delivery, self.request)
        self.assertEqual(result['deleted'], [self.name])
        self.assertFalse(self.archive.exists())
        self.assertEqual(self.manifest.read_bytes(), before)
        catalog, archives = packager.load_catalog(self.delivery)
        self.assertEqual(catalog, {'runs/fixture/result.json': {'size': 40, 'mtime_ns': 999}})
        self.assertEqual(archives[0]['archive'], self.name)
        self.assertTrue(archives[0]['remote_copy_reclaimed'])
        self.assertEqual(archives[0]['sha256'], self.proof['sha256'])
        repeated = reclamation.reclaim(self.delivery, self.request)
        self.assertEqual(repeated['deleted'], [])
        self.assertEqual(repeated['already_absent_finalized'], [self.name])

    def test_wrong_hash_or_manifest_binding_never_authorizes_removal_or_catalog(self):
        for patch_value in ({'sha256': '0' * 64}, {'manifest_sha256': '0' * 64}):
            value = self.receipt | patch_value
            with self.subTest(patch=patch_value), self.assertRaises(ValueError):
                reclamation.reclaim(self.delivery, self.request | {'receipts': [value]})
            self.assertTrue(self.archive.exists())
        self.archive.unlink()
        write(self.delivery / reclamation.receipt_name(self.name), self.receipt | {
            'schema': reclamation.SCHEMA, 'state': 'remote_copy_reclaimed', 'sha256': '0' * 64})
        with self.assertRaises(ValueError):
            packager.load_catalog(self.delivery)

    def test_pack_lock_busy_has_no_deletion_or_receipt_write(self):
        with (self.delivery / '.pack.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = reclamation.reclaim(self.delivery, self.request)
        self.assertEqual(result['status'], 'lock_busy')
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.delivery / reclamation.receipt_name(self.name)).exists())

    def test_path_escape_and_symlink_are_rejected(self):
        for name in ('../outside.tar.gz', '/tmp/outside.tar.gz', '..\\outside.tar.gz'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reclamation.reclaim(self.delivery, self.request | {'receipts': [self.receipt | {'archive': name}]})
        outside = self.root / 'original-data.bin'
        outside.write_bytes(b'original source must remain')
        self.archive.unlink()
        self.archive.symlink_to(outside)
        with self.assertRaises(ValueError):
            reclamation.reclaim(self.delivery, self.request)
        self.assertEqual(outside.read_bytes(), b'original source must remain')
        with self.assertRaises(ValueError):
            reclamation.validate_delivery(self.root / 'delivery/../original-data.bin')

    def test_pending_receipt_survives_crash_and_can_be_finalized(self):
        with patch.object(Path, 'unlink', side_effect=OSError('fixture interrupted before unlink')):
            with self.assertRaisesRegex(OSError, 'fixture interrupted'):
                reclamation.reclaim(self.delivery, self.request)
        pending = self.delivery / reclamation.receipt_name(self.name)
        self.assertEqual(json.loads(pending.read_text())['state'], 'delete_pending')
        self.assertTrue(self.archive.exists())
        self.archive.unlink()
        with self.assertRaises(ValueError):
            packager.load_catalog(self.delivery)
        result = reclamation.reclaim(self.delivery, self.request)
        self.assertEqual(result['deleted'], [])
        self.assertEqual(json.loads(pending.read_text())['state'], 'remote_copy_reclaimed')
        self.assertTrue(packager.load_catalog(self.delivery)[1][0]['remote_copy_reclaimed'])

    def local_fixture(self):
        destination = self.root / 'external'
        destination.mkdir()
        local_archive = destination / self.name
        local_archive.write_bytes(self.archive.read_bytes())
        write(destination / reclamation.manifest_name(self.name), self.proof)
        reports = self.root / 'transfers'
        report = {'status': 'completed', 'sha256': self.proof['sha256'],
            'source_checksum_verified': True, 'expected_bytes': self.proof['archive_bytes'],
            'source': str(self.archive), 'destination': str(local_archive), 'finished_at': time.time() + 1}
        write(reports / (self.name + '.json'), report)
        return destination, reports, report

    def test_verified_local_archive_avoids_all_remote_calls(self):
        destination, reports, report = self.local_fixture()
        self.archive.unlink()
        with patch.object(sync, 'REMOTE', str(self.delivery)), patch.object(sync, 'DESTINATION', destination), \
                patch.object(sync, 'TRANSFER_REPORTS', reports), patch.object(sync, 'remote') as remote:
            result = sync.download(self.proof, 2)
        remote.assert_not_called()
        self.assertTrue(result['reused_verified_download'])

    def test_direct_receipt_reuse_does_not_claim_content_verification(self):
        destination, reports, report = self.local_fixture()
        report.update(source_checksum_verified=False, transport_mode='direct_no_content_hash',
            verification='authenticated_ssh_and_exact_length_only', transfer_returncode=0,
            received_bytes=self.proof['archive_bytes'], expected_sha256_from_manifest=self.proof['sha256'])
        write(reports / (self.name + '.json'), report)
        with patch.object(sync, 'REMOTE', str(self.delivery)), patch.object(sync, 'DESTINATION', destination), \
                patch.object(sync, 'TRANSFER_REPORTS', reports), patch.object(sync, 'remote') as remote:
            result = sync.download(self.proof, 2)
            completed = {}
            sync.reuse_verified_archives({self.name: self.proof}, completed, set())
        remote.assert_not_called()
        for row in (result, completed[self.name]):
            self.assertTrue(row['reused_existing_download'])
            self.assertFalse(row['reused_verified_download'])
            self.assertFalse(row['content_checksum_verified'])

    def test_missing_local_archive_or_mismatched_transfer_identity_is_not_complete(self):
        destination, reports, report = self.local_fixture()
        with patch.object(sync, 'REMOTE', str(self.delivery)), patch.object(sync, 'DESTINATION', destination), \
                patch.object(sync, 'TRANSFER_REPORTS', reports):
            for field, value in [('source', '/wrong/source'), ('destination', '/wrong/destination'),
                                 ('sha256', '0' * 64), ('expected_bytes', 100), ('source_checksum_verified', False)]:
                write(reports / (self.name + '.json'), report | {field: value})
                self.assertFalse(sync.verified_local_archive(self.proof), field)
            write(reports / (self.name + '.json'), report)
            (destination / self.name).unlink()
            self.assertFalse(sync.verified_local_archive(self.proof))
            with patch.object(sync, 'remote', side_effect=FileNotFoundError('remote archive reclaimed')):
                with self.assertRaises(FileNotFoundError):
                    sync.download(self.proof, 2)

    def test_catalog_reuses_more_than_one_batch_without_downloading_or_sleeping(self):
        destination, reports, report = self.local_fixture()
        known = {}
        for index in range(14):
            name = f'fixture-{index:02d}.tar.gz'
            proof = self.proof | {'archive': name}
            known[name] = proof
            local_archive = destination / name
            if index != 12:
                local_archive.write_bytes(self.archive.read_bytes())
            write(destination / reclamation.manifest_name(name), proof)
            write(reports / (name + '.json'), report | {
                'source': str(self.delivery / name), 'destination': str(local_archive)})
        completed = {}
        with patch.object(sync, 'REMOTE', str(self.delivery)), patch.object(sync, 'DESTINATION', destination), \
                patch.object(sync, 'TRANSFER_REPORTS', reports), patch.object(sync, 'download') as download, \
                patch.object(sync, 'remote') as remote, patch.object(sync.time, 'sleep') as sleep:
            sync.reuse_verified_archives(known, completed, {'fixture-13.tar.gz'})
        self.assertEqual(set(completed), {f'fixture-{index:02d}.tar.gz' for index in range(12)})
        self.assertTrue(all(item['reused_verified_download'] for item in completed.values()))
        self.assertEqual(sum(item['bytes'] for item in completed.values()), 12 * self.proof['archive_bytes'])
        download.assert_not_called()
        remote.assert_not_called()
        sleep.assert_not_called()


class ArchiveSyncMainTests(unittest.TestCase):
    def run_main(self, follow_updates=False, hours=16):
        old = {'archive': 'old.tar.gz', 'archive_bytes': 100, 'sha256': 'a' * 64}
        new = {'archive': 'new.tar.gz', 'archive_bytes': 50, 'sha256': 'b' * 64}
        snapshots = [
            {'status': 'snapshot_catalog_caught_up', 'archives': [old]},
            {'status': 'snapshot_catalog_caught_up', 'archives': [old, new]},
        ]
        clock = {'elapsed': 0.0}
        saved, polls, sleeps = [], [], []

        def snapshot(code):
            self.assertIn('status.json', code)
            polls.append(clock['elapsed'])
            return deepcopy(snapshots[min(len(polls) - 1, len(snapshots) - 1)])

        def sleep(seconds):
            self.assertGreater(seconds, 0)
            sleeps.append(seconds)
            clock['elapsed'] += seconds
            self.assertLess(len(sleeps), 100, 'Coordinator did not respect its bounded duration')

        def downloaded(item, workers_per_archive):
            return {'archive': item['archive'], 'bytes': item['archive_bytes'],
                'sha256': item['sha256'], 'verified_at': 1000 + clock['elapsed'],
                'content_checksum_verified': True}

        destination = Path('/Volumes/TOSHIBA EXT/fixture/project-results-archives')
        arguments = ['sync_rental_result_archives.py', '--hours', str(hours),
            '--parallel-archives', '2', '--workers-per-archive', '3']
        if follow_updates:
            arguments.append('--follow-updates')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            status = Path(directory) / 'result-archive-backup-status.json'
            stack.enter_context(patch.object(sync, 'REMOTE', '/fixture/project-results'))
            stack.enter_context(patch.object(sync, 'DESTINATION', destination))
            stack.enter_context(patch.object(sync, 'STATUS', status))
            stack.enter_context(patch.object(sync, 'TRANSFER_REPORTS', Path(directory) / 'transfers'))
            stack.enter_context(patch.object(sync.sys, 'argv', arguments))
            stack.enter_context(patch.object(sync.os.path, 'ismount', return_value=True))
            stack.enter_context(patch.object(Path, 'mkdir'))
            stack.enter_context(patch.object(sync.time, 'monotonic', side_effect=lambda: clock['elapsed']))
            stack.enter_context(patch.object(sync.time, 'time', side_effect=lambda: 1000 + clock['elapsed']))
            stack.enter_context(patch.object(sync.time, 'sleep', side_effect=sleep))
            stack.enter_context(patch.object(sync, 'save', side_effect=lambda state: saved.append(deepcopy(state))))
            remote = stack.enter_context(patch.object(sync, 'remote', side_effect=snapshot))
            verified = stack.enter_context(patch.object(sync, 'verified_local_archive',
                side_effect=lambda item: item['archive'] == old['archive']))
            stack.enter_context(patch.object(sync, 'read_regular_json',
                return_value={'source_checksum_verified': True}))
            download = stack.enter_context(patch.object(sync, 'download', side_effect=downloaded))
            process = stack.enter_context(patch.object(sync.subprocess, 'run',
                side_effect=AssertionError('CPU coordinator test must not start external commands')))
            executor = stack.enter_context(patch.object(sync, 'ThreadPoolExecutor'))
            pool = executor.return_value.__enter__.return_value

            def submit(function, *args):
                future = Future()
                future.set_result(function(*args))
                return future

            pool.submit.side_effect = submit
            sync.main()
            executor.assert_called_once_with(max_workers=2)
            process.assert_not_called()
            self.assertTrue(verified.called)
            return {'states': saved, 'polls': polls, 'sleeps': sleeps, 'elapsed': clock['elapsed'],
                'downloads': download.call_args_list, 'submissions': pool.submit.call_args_list,
                'remote_calls': remote.call_count, 'old': old, 'new': new}

    def test_main_default_exits_when_snapshot_is_verified_without_redownloading(self):
        result = self.run_main(hours=16)
        final = result['states'][-1]
        self.assertEqual(final['status'], 'snapshot_archives_downloaded')
        self.assertEqual(set(final['completed']), {'old.tar.gz'})
        self.assertTrue(final['completed']['old.tar.gz']['reused_verified_download'])
        self.assertEqual(final['verified_bytes'], 100)
        self.assertEqual(final['active'], [])
        self.assertFalse(final['full_delivery_complete'])
        self.assertEqual(result['remote_calls'], 1)
        self.assertEqual(result['downloads'], [])
        self.assertEqual(result['submissions'], [])
        self.assertEqual(result['sleeps'], [])

    def test_main_follow_updates_downloads_only_new_archive_then_ends_at_deadline(self):
        result = self.run_main(follow_updates=True, hours=120 / 3600)
        final = result['states'][-1]
        waiting = [state for state in result['states'] if state['status'] == 'waiting_for_new_archives']
        self.assertTrue(waiting)
        self.assertTrue(any(set(state['completed']) == {'old.tar.gz'} for state in waiting))
        self.assertEqual(final['status'], 'bounded_pass_ended')
        self.assertEqual(set(final['completed']), {'old.tar.gz', 'new.tar.gz'})
        self.assertTrue(final['completed']['old.tar.gz']['reused_verified_download'])
        self.assertEqual(final['verified_archives'], 2)
        self.assertEqual(final['verified_bytes'], 150)
        self.assertEqual(final['active'], [])
        self.assertFalse(final['full_delivery_complete'])
        self.assertEqual(len(result['downloads']), 1)
        self.assertEqual(result['downloads'][0].args, (result['new'], 3))
        self.assertEqual(len(result['submissions']), 1)
        self.assertGreaterEqual(result['remote_calls'], 2)
        for previous, current in zip(result['polls'], result['polls'][1:]):
            self.assertGreaterEqual(current - previous, 45)
        self.assertGreaterEqual(result['elapsed'], 120)
        self.assertLessEqual(result['elapsed'], 165)


if __name__ == '__main__':
    unittest.main()
