import hashlib
import importlib.util
import io
from pathlib import Path, PurePosixPath
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location("isolated_restore", Path(__file__).resolve().parents[1] / "scripts/check_isolated_rental_restore.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def archive_fixture(tmp_path, member_name="opt/test/bin/python"):
    payload = b"synthetic restore test only"
    path = tmp_path / "test.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        member.mode = 0o755
        archive.addfile(member, io.BytesIO(payload))
    return {"archives": {"test": {"file": path.name, "roots": ["/opt/test"],
        "archive_bytes": path.stat().st_size, "raw_bytes": len(payload),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}


def test_extracts_independent_copy_with_stream_identity_and_refuses_overwrite(tmp_path):
    manifest = archive_fixture(tmp_path)
    root = tmp_path / "isolated/root"
    result = restore.extract(tmp_path, manifest, root, 10**9, tmp_path / "report.json")
    assert result["status"] == "extracted" and result["runtime_verified"] is False
    assert (root / "opt/test/bin/python").read_bytes() == b"synthetic restore test only"
    assert result["archives"][0]["stream_verified"] is True
    with pytest.raises(FileExistsError):
        restore.extract(tmp_path, manifest, root, 10**9, tmp_path / "report.json")


def test_extract_rejects_unregistered_member(tmp_path):
    manifest = archive_fixture(tmp_path, "usr/local/live-python")
    with pytest.raises(ValueError, match="outside declared"):
        restore.extract(tmp_path, manifest, tmp_path / "root", 10**9, tmp_path / "report.json")


def test_provenance_requires_actual_restored_inode(tmp_path):
    root = tmp_path / "root"
    file = root / "opt/test/module.py"
    file.parent.mkdir(parents=True)
    file.write_text("pass\n")
    stat = file.stat()
    record = {"path": "/opt/test/module.py", "device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size}
    assert restore.check_file(record, root, [PurePosixPath("/opt/test")]) == str(file)
    record["inode"] += 1
    with pytest.raises(ValueError, match="inode"):
        restore.check_file(record, root, [PurePosixPath("/opt/test")])
    record["path"] = "/usr/local/live-python/module.py"
    with pytest.raises(ValueError, match="outside the restored"):
        restore.check_file(record, root, [PurePosixPath("/opt/test")])


def test_no_active_python_project_or_gpu_device_bindings():
    for path in restore.system_bindings():
        assert not path.startswith(("/usr/local", "/workspace", "/dev/nvidia"))
        assert path not in ("/usr", "/dev", "/lib", "/usr/lib")
