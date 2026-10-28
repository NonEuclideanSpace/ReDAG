import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "restore_rental_bundle",
    Path(__file__).resolve().parents[1] / "scripts/restore_rental_bundle.py",
)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


class RestoreRentalBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.backup = self.root / "backup"
        self.archives = self.backup / "remote/project-results-archives"
        self.archives.mkdir(parents=True)
        (self.backup / "code-revisions").mkdir()
        self.original_project = "/workspace/agentic-image-pilot-small"
        self.code = self.backup / "code-revisions/pilot-small-code-20261028T080000Z.tar.gz"
        self.make_code({"src/version.py": b"version = 'latest'\n"})

    def make_code(self, payloads):
        with tarfile.open(self.code, "w:gz") as stream:
            for name, content in payloads.items():
                info = tarfile.TarInfo("pilot-small/" + name)
                info.size = len(content)
                info.mode = 0o644
                stream.addfile(info, io.BytesIO(content))

    def make_archive(self, generation, payloads, unstable=()):
        archive = self.archives / f"{generation:019d}-00001-code.tar.gz"
        members = []
        with tarfile.open(archive, "w:gz") as stream:
            for name, value in payloads.items():
                info = tarfile.TarInfo(name)
                link = value if isinstance(value, tuple) else None
                hardlink = bool(link and link[0] == "hardlink")
                data = b"" if link else value
                if link:
                    info.type = tarfile.LNKTYPE if hardlink else tarfile.SYMTYPE
                    info.linkname = link[1]
                    info.mode = 0o644 if hardlink else 0o777
                    stream.addfile(info)
                else:
                    info.mode = 0o644
                    info.size = len(data)
                    stream.addfile(info, io.BytesIO(data))
                mode = (stat.S_IFLNK if link else stat.S_IFREG) | info.mode
                before = {"size": len(link[1]) if link else len(data), "mtime_ns": generation,
                    "ctime_ns": generation, "mode": mode, "inode": len(members) + 1,
                    "device": 1, "type": "symlink" if link else "file",
                    "link_target": link[1] if link else None}
                if hardlink:
                    before = copy.deepcopy(next(row["before"] for row in members if row["path"] == link[1]))
                members.append({"path": name, "before": before, "after": copy.deepcopy(before),
                    "stable": name not in unstable, "changed_since_inventory": False,
                    "sha256_of_archived_file_bytes": None if link else hashlib.sha256(data).hexdigest(),
                    "archived_file_bytes": len(data), "tar_type": "1" if hardlink else "2" if link else "0",
                    "tar_link_target": link[1] if link else None})
        manifest = {"archive": archive.name, "archive_bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "group": "code",
            "project": self.original_project, "members": members, "completed_at": generation,
            "source_bytes": sum(row["archived_file_bytes"] for row in members),
            "live_changes": list(unstable), "source_deleted": False}
        path = archive.with_name(archive.name.removesuffix(".tar.gz") + ".manifest.json")
        path.write_text(json.dumps(manifest))
        return archive, path

    def plan(self):
        return restore.build_plan(self.backup, self.code)

    def test_latest_stable_version_and_code_overlay_preserve_retained_history(self):
        self.make_archive(2, {"data/value.txt": b"second", "src/version.py": b"old archive source"})
        self.make_archive(1, {"data/value.txt": b"first", "runs/retained.txt": b"retained historical file"})
        plan = self.plan()
        self.assertTrue(plan["ready"], plan["errors"])
        destination = self.root / "restored"
        result = restore.restore_plan(plan, destination)
        self.assertIsInstance(result, dict)
        self.assertEqual((destination / "data/value.txt").read_bytes(), b"second")
        self.assertEqual((destination / "runs/retained.txt").read_bytes(), b"retained historical file")
        self.assertEqual((destination / "src/version.py").read_bytes(), b"version = 'latest'\n")
        self.assertFalse((destination / "pilot-small").exists())

    def test_missing_or_wrong_size_archives_make_plan_unready(self):
        for condition in ("missing", "wrong_size"):
            with self.subTest(condition=condition):
                archive, _ = self.make_archive(1, {"data/value.txt": b"required"})
                if condition == "missing":
                    archive.unlink()
                else:
                    archive.write_bytes(archive.read_bytes() + b"size changed")
                plan = self.plan()
                self.assertFalse(plan["ready"])
                self.assertTrue(plan["errors"])
                with self.assertRaises((ValueError, RuntimeError)):
                    restore.restore_plan(plan, self.root / condition)
                self.assertFalse((self.root / condition).exists())

    def test_latest_unstable_member_cannot_silently_fall_back_to_older_version(self):
        self.make_archive(1, {"data/value.txt": b"stable"})
        self.make_archive(2, {"data/value.txt": b"unstable"}, unstable=("data/value.txt",))
        plan = self.plan()
        self.assertFalse(plan["ready"])
        self.assertTrue(plan["errors"])
        with self.assertRaises((ValueError, RuntimeError)):
            restore.restore_plan(plan, self.root / "restored")

    def test_existing_destination_is_never_overwritten(self):
        self.make_archive(1, {"data/value.txt": b"new"})
        destination = self.root / "existing"
        destination.mkdir()
        marker = destination / "preserve.txt"
        marker.write_bytes(b"keep")
        with self.assertRaises((ValueError, FileExistsError)):
            restore.restore_plan(self.plan(), destination)
        self.assertEqual(marker.read_bytes(), b"keep")
        empty = self.root / "empty-but-existing"
        empty.mkdir()
        with self.assertRaises((ValueError, FileExistsError)):
            restore.restore_plan(self.plan(), empty)
        outside = self.root / "parent-target"
        outside.mkdir()
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, FileExistsError)):
            restore.restore_plan(self.plan(), linked_parent / "new")
        self.assertFalse((outside / "new").exists())

    def test_extraction_rejects_archive_stream_checksum_mismatch(self):
        _, manifest = self.make_archive(1, {"data/value.txt": b"content"})
        record = json.loads(manifest.read_text())
        record["sha256"] = "0" * 64
        manifest.write_text(json.dumps(record))
        plan = self.plan()
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            restore.restore_plan(plan, self.root / "restored")

    def test_internal_relative_and_absolute_symlinks_stay_in_restored_project(self):
        self.make_archive(1, {"runs/original/value.txt": b"image metadata",
            "runs/relative": ("symlink", "original"),
            "runs/absolute": ("symlink", self.original_project + "/runs/original")})
        destination = self.root / "restored"
        plan = self.plan()
        self.assertTrue(plan["ready"], plan["errors"])
        restore.restore_plan(plan, destination)
        for name in ("relative", "absolute"):
            path = destination / "runs" / name
            self.assertTrue(path.is_symlink())
            self.assertTrue(path.resolve().is_relative_to(destination))
            self.assertEqual((path / "value.txt").read_bytes(), b"image metadata")

    def test_external_link_and_member_traversal_never_write_outside(self):
        for name, payload in (
            ("external_link", {"runs/link": ("symlink", "/outside/unowned")}),
            ("relative_escape", {"runs/link": ("symlink", "../../outside")}),
            ("member_escape", {"../escaped.txt": b"forbidden"}),
            ("link_parent", {"runs/link": ("symlink", "original"), "runs/link/nested.txt": b"forbidden"}),
        ):
            with self.subTest(name=name):
                self.make_archive(1, payload)
                try:
                    plan = self.plan()
                except (ValueError, RuntimeError):
                    continue
                with self.assertRaises((ValueError, RuntimeError, OSError)):
                    restore.restore_plan(plan, self.root / ("restore-" + name))
                self.assertFalse((self.root / "escaped.txt").exists())

    def test_missing_code_selection_is_explicitly_unready(self):
        self.make_archive(1, {"data/value.txt": b"content"})
        missing = self.backup / "code-revisions/does-not-exist.tar.gz"
        try:
            plan = restore.build_plan(self.backup, missing)
        except (ValueError, FileNotFoundError):
            return
        self.assertFalse(plan["ready"])
        self.assertTrue(plan["errors"])

    def test_code_pointer_selects_existing_integrated_bundle(self):
        self.make_archive(1, {"data/value.txt": b"content"})
        (self.backup / "code-revisions/code-update-order.json").write_text(json.dumps({
            "recommended_integrated_source_archive": self.code.name,
            "recommended_integrated_source_sha256": hashlib.sha256(self.code.read_bytes()).hexdigest(),
            "recommended_integrated_source_is_standalone": True,
        }))
        plan = restore.build_plan(self.backup)
        self.assertTrue(plan["ready"], plan["errors"])
        destination = self.root / "restored"
        restore.restore_plan(plan, destination)
        self.assertEqual((destination / "src/version.py").read_bytes(), b"version = 'latest'\n")

    def test_code_pointer_checksum_must_match_selected_integrated_bundle(self):
        self.make_archive(1, {"data/value.txt": b"content"})
        (self.backup / "code-revisions/code-update-order.json").write_text(json.dumps({
            "recommended_integrated_source_archive": self.code.name,
            "recommended_integrated_source_sha256": "0" * 64,
            "recommended_integrated_source_is_standalone": True,
        }))
        try:
            plan = restore.build_plan(self.backup)
        except (ValueError, RuntimeError):
            return
        self.assertFalse(plan["ready"])
        self.assertTrue(plan["errors"])

    def test_hardlink_restores_identical_payload_and_inode(self):
        self.make_archive(1, {"runs/original.bin": b"shared payload",
            "runs/linked.bin": ("hardlink", "runs/original.bin"),
            "runs/linked-again.bin": ("hardlink", "runs/linked.bin")})
        plan = self.plan()
        self.assertTrue(plan["ready"], plan["errors"])
        destination = self.root / "restored"
        restore.restore_plan(plan, destination)
        original = destination / "runs/original.bin"
        linked = destination / "runs/linked.bin"
        self.assertEqual(linked.read_bytes(), b"shared payload")
        self.assertEqual(original.stat().st_ino, linked.stat().st_ino)
        self.assertEqual(original.stat().st_ino, (destination / "runs/linked-again.bin").stat().st_ino)
        self.assertFalse(linked.is_symlink())

    def test_hardlink_to_changed_target_version_is_rejected(self):
        self.make_archive(1, {"runs/original.bin": b"original payload",
            "runs/linked.bin": ("hardlink", "runs/original.bin")})
        self.make_archive(2, {"runs/original.bin": b"different later payload"})
        plan = self.plan()
        self.assertFalse(plan["ready"])
        self.assertTrue(plan["errors"])
        with self.assertRaises((ValueError, RuntimeError)):
            restore.restore_plan(plan, self.root / "restored")

    def test_new_safe_link_replaces_an_older_unrestorable_external_link(self):
        self.make_archive(1, {"runs/link": ("symlink", "/old/external")})
        self.make_archive(2, {"runs/real.txt": b"current", "runs/link": ("symlink", "real.txt")})
        plan = self.plan()
        self.assertTrue(plan["ready"], plan["errors"])
        destination = self.root / "restored"
        restore.restore_plan(plan, destination)
        self.assertEqual((destination / "runs/link").read_bytes(), b"current")

    def test_delivery_index_selects_code_and_checks_its_recorded_checksum(self):
        self.make_archive(1, {"data/value.txt": b"content"})
        (self.backup / "code-revisions/code-update-order.json").write_text(json.dumps({
            "recommended_integrated_source_archive": "unavailable-old-code.tar.gz"}))
        index = self.backup / "DELIVERY_INDEX.json"
        index.write_text(json.dumps({"code": {"archive": "code-revisions/" + self.code.name,
            "sha256": hashlib.sha256(self.code.read_bytes()).hexdigest()}}))
        self.assertTrue(restore.build_plan(self.backup)["ready"])
        index.write_text(json.dumps({"code": {"archive": "code-revisions/" + self.code.name,
            "sha256": "0" * 64}}))
        with self.assertRaises(ValueError):
            restore.build_plan(self.backup)

    def test_existing_restore_report_is_not_overwritten(self):
        self.make_archive(1, {"data/value.txt": b"content"})
        destination = self.root / "restored"
        report = self.root / "restored.restore-report.json"
        report.write_text("prior evidence")
        with self.assertRaises(FileExistsError):
            restore.restore_plan(self.plan(), destination)
        self.assertFalse(destination.exists())
        self.assertEqual(report.read_text(), "prior evidence")


if __name__ == "__main__":
    unittest.main()
