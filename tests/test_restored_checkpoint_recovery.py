import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "restored_checkpoint_recovery",
    Path(__file__).resolve().parents[1] / "scripts/restored_checkpoint_recovery.py",
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class RestoredCheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()
        self.stage = Path("runs/full-actor-rl/rl_agent")
        self.checkpoint = self.stage / "checkpoints/actual-committed-id"
        self.config = {"seed": 11, "devices": {"actor": "cuda:1", "renderer": "cuda:0",
            "encoders": "cuda:0", "judge": "cuda:1"}}

    def fixture(self, name="valid", modify=None):
        payloads = {
            str(self.checkpoint / "COMMITTED.json"): b'{"status":"COMMITTED"}',
            str(self.checkpoint / "actor.pt"): b"archived actor state",
            str(self.checkpoint / "renderer.pt"): b"archived renderer state",
            str(self.checkpoint / "training_state.pt"): b"archived optimizer and RNG fixture",
        }
        config_path = str(self.stage / "committed_restore_config.json")
        members = [(path, data, tarfile.REGTYPE) for path, data in payloads.items()]
        members.append((config_path, json.dumps(self.config).encode(), tarfile.REGTYPE))
        if modify:
            members = modify(members)
        archive = self.root / (name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            for path, data, kind in members:
                info = tarfile.TarInfo(path)
                info.type = kind
                if kind == tarfile.SYMTYPE:
                    info.linkname = "/outside"
                    stream.addfile(info)
                else:
                    info.size = len(data)
                    stream.addfile(info, io.BytesIO(data))
        record = {
            "schema": "rl_recovery_export_v1",
            "status": "completed",
            "stage": "agent",
            "effective_target_met": True,
            "source": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "checkpoint": str(self.project / self.checkpoint),
            "files": {str(self.project / path): {"bytes": len(data)} for path, data in payloads.items()},
            "committed_config": self.config,
        }
        manifest = self.root / (name + ".manifest.json")
        manifest.write_text(json.dumps(record))
        return archive, manifest, payloads

    def plan(self, name="valid", modify=None):
        archive, manifest, payloads = self.fixture(name, modify)
        return recovery.recovery_plan(self.project, archive, manifest), payloads

    def test_extracts_only_declared_members_and_matching_configuration(self):
        plan, payloads = self.plan()
        destination = self.root / "restored"
        result = recovery.extract_recovery(plan, destination, rate=32 * 2**20)
        self.assertEqual(result["status"], "extracted")
        self.assertEqual(result["files"], len(payloads) + 1)
        self.assertEqual(result["config"], self.config)
        self.assertEqual(result["checkpoint"], str(destination / self.checkpoint))
        self.assertEqual(result["sha256"], plan["manifest"]["sha256"])
        for path, payload in payloads.items():
            self.assertEqual((destination / path).read_bytes(), payload)
        self.assertFalse((self.project / self.stage).exists())

    def test_existing_destination_is_preserved_and_rejected(self):
        plan, _ = self.plan()
        destination = self.root / "restored"
        destination.mkdir()
        sentinel = destination / "preserved"
        sentinel.write_bytes(b"existing result")
        with self.assertRaises(FileExistsError):
            recovery.extract_recovery(plan, destination)
        self.assertEqual(sentinel.read_bytes(), b"existing result")

    def test_rejects_unexpected_traversal_symlink_duplicate_and_wrong_size(self):
        modifications = {
            "extra": lambda items: items + [("unregistered.bin", b"extra", tarfile.REGTYPE)],
            "traversal": lambda items: items + [("../outside.bin", b"escape", tarfile.REGTYPE)],
            "symlink": lambda items: [(items[0][0], b"", tarfile.SYMTYPE), *items[1:]],
            "duplicate": lambda items: items + [items[0]],
            "size": lambda items: [(items[0][0], b"different byte count", tarfile.REGTYPE), *items[1:]],
        }
        for name, modify in modifications.items():
            with self.subTest(name=name):
                plan, _ = self.plan(name, modify)
                with self.assertRaises(ValueError):
                    recovery.extract_recovery(plan, self.root / ("restored-" + name), rate=32 * 2**20)
        self.assertFalse((self.root / "outside.bin").exists())

    def test_rejects_missing_member_and_mismatched_stream_checksum(self):
        plan, _ = self.plan("missing", lambda items: items[1:])
        with self.assertRaisesRegex(ValueError, "incomplete|checksum"):
            recovery.extract_recovery(plan, self.root / "restored-missing", rate=32 * 2**20)
        plan, _ = self.plan("checksum")
        plan["manifest"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            recovery.extract_recovery(plan, self.root / "restored-checksum", rate=32 * 2**20)

    def test_rejects_uncommitted_wrong_source_size_or_outside_manifest_paths(self):
        archive, manifest, _ = self.fixture()
        original = json.loads(manifest.read_text())
        changes = {
            "uncommitted": lambda row: row.update(status="running"),
            "source": lambda row: row.update(source="/different/archive.tar.gz"),
            "size": lambda row: row.update(bytes=row["bytes"] + 1),
            "checkpoint": lambda row: row.update(checkpoint="/outside/checkpoints/checkpoint"),
            "member": lambda row: row["files"].update({"/outside/member.pt": {"bytes": 1}}),
            "required_member": lambda row: row["files"].pop(str(self.project / self.checkpoint / "actor.pt")),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                row = copy.deepcopy(original)
                change(row)
                manifest.write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    recovery.recovery_plan(self.project, archive, manifest)

    def test_rejects_changed_committed_configuration(self):
        def modify(items):
            return [(path, b'{"seed": 99}', kind) if path.endswith("committed_restore_config.json") else (path, data, kind)
                    for path, data, kind in items]
        plan, _ = self.plan("config", modify)
        with self.assertRaisesRegex(ValueError, "configuration"):
            recovery.extract_recovery(plan, self.root / "restored-config", rate=32 * 2**20)

    def test_preserves_original_two_gpu_device_configuration(self):
        original = copy.deepcopy(self.config)
        recovery.validate_recovery_devices(self.config, 2)
        self.assertEqual(self.config, original)
        reversed_config = copy.deepcopy(self.config)
        reversed_config["devices"] = {"actor": "cuda:0", "renderer": "cuda:1",
            "encoders": "cuda:1", "judge": "cuda:0"}
        recovery.validate_recovery_devices(reversed_config, 2)

    def test_restored_configuration_uses_extracted_adapters_and_original_devices(self):
        config = copy.deepcopy(self.config)
        config.update(models={
            "actor_policy": "runs/sft/policy", "actor_reference": str(self.project / "runs/reference/actor"),
            "renderer_policy": "runs/sft/renderer.pt", "renderer_reference": str(self.project / "runs/reference/renderer.pt")},
            resources={}, data={})
        request = {"config": config, "recovered_root": str(self.root / "recovered"),
            "original_project": str(self.project), "output": str(self.root / "verification"),
            "actor_nf4": "/existing/actor-base", "renderer_base": "/existing/renderer-base"}
        actual = recovery.restored_config(request)
        self.assertEqual(actual["devices"], self.config["devices"])
        for key in ("actor_policy", "actor_reference", "renderer_policy", "renderer_reference"):
            self.assertTrue(Path(actual["models"][key]).is_relative_to(self.root / "recovered"))
        self.assertEqual(actual["models"]["actor_nf4"], "/existing/actor-base")
        self.assertEqual(request["config"], config)

    def test_rejects_missing_or_remapped_two_gpu_topology(self):
        for count in (0, 1, 3):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    recovery.validate_recovery_devices(self.config, count)
        cases = [
            {"actor": "cuda:0", "renderer": "cuda:0", "encoders": "cuda:0", "judge": "cuda:1"},
            {"actor": "cuda:1", "renderer": "cpu", "encoders": "cuda:0", "judge": "cuda:1"},
            {"actor": "cuda:1", "renderer": "cuda:0", "encoders": "cuda:2", "judge": "cuda:1"},
            {"actor": "cuda:1", "renderer": "cuda:0", "encoders": "cuda:0"},
        ]
        for devices in cases:
            with self.subTest(devices=devices):
                with self.assertRaises(ValueError):
                    recovery.validate_recovery_devices({"devices": devices}, 2)


if __name__ == "__main__":
    unittest.main()
