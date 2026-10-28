import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_rl_recovery_exports as watcher


class RecoveryWatcherTests(unittest.TestCase):
    def setUp(self):
        self.item = {'mode': 'agent', 'source': '/workspace/agentic-image-pilot-small/delivery/priority-checkpoints/full-agent-rl32-complete.tar.gz'}
        self.manifest = {'schema': 'rl_recovery_export_v1', 'status': 'completed', 'stage': 'agent',
            'source': self.item['source'], 'required_effective_updates': 32, 'effective_target_met': True,
            'source_files_unchanged': True, 'shared_renders_included': False, 'bytes': 3, 'mtime': 123,
            'sha256': 'a' * 64, 'weight_validation': {'actual_checkpoint_tensors_checked': True}}

    def test_incomplete_wrong_stage_and_invalid_hash_are_rejected(self):
        watcher.validate_manifest(self.manifest, self.item)
        for change in ({'status': 'running'}, {'stage': 'joint'}, {'effective_target_met': False},
                {'sha256': None}, {'sha256': 'invalid'}, {'shared_renders_included': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                watcher.validate_manifest(self.manifest | change, self.item)

    def test_single_priority_worker_never_exceeds_seven_transfer_workers_or_duplicates_source(self):
        bulk = [{'source': 'bulk-' + str(i), 'workers': 2} for i in range(3)]
        self.assertTrue(watcher.transfer_capacity_available(self.item, bulk))
        self.assertFalse(watcher.transfer_capacity_available(self.item, bulk + [{'source': 'other', 'workers': 1}]))
        self.assertFalse(watcher.transfer_capacity_available(self.item, [{'source': self.item['source'], 'workers': 1}]))

    def test_reuse_requires_exact_actual_file_and_completed_checksum_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'fixture.tar.gz'
            report = Path(temporary) / 'fixture.transfer.json'
            target.write_bytes(b'abc')
            proof = {'status': 'completed', 'source_checksum_verified': True, 'source': self.item['source'],
                'destination': str(target), 'expected_bytes': 3, 'sha256': 'a' * 64, 'finished_at': time.time() + 1}
            report.write_text(json.dumps(proof))
            self.assertTrue(watcher.verified_local(self.manifest, target, report))
            report.write_text(json.dumps(proof | {'source_checksum_verified': False}))
            self.assertFalse(watcher.verified_local(self.manifest, target, report))
            report.write_text(json.dumps(proof))
            target.write_bytes(b'changed')
            self.assertFalse(watcher.verified_local(self.manifest, target, report))


if __name__ == '__main__':
    unittest.main()
