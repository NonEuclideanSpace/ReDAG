import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tarfile
import time
import uuid


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def system_bindings():
    paths = ["/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib64", "/proc",
             "/dev/null", "/dev/random", "/dev/urandom"]
    return [path for path in paths if Path(path).exists()]


def extract(directory, manifest, root, rate, report):
    if root.exists() or root.is_symlink():
        raise FileExistsError("Extraction requires a new unused destination: " + str(root))
    required = sum(row["raw_bytes"] for row in manifest["archives"].values())
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < required + 512 * 1024 * 1024:
        raise RuntimeError("Insufficient space for an independent restored copy")
    root.mkdir()
    marker = str(uuid.uuid4())
    (root / ".restore-check-marker").write_text(marker + "\n")
    all_roots = [PurePosixPath(value) for row in manifest["archives"].values() for value in row["roots"]]
    completed, raw, started, last_report = [], 0, time.monotonic(), 0.0
    for name, expected in manifest["archives"].items():
        path = directory / expected["file"]
        if path.stat().st_size != expected["archive_bytes"]:
            raise ValueError("Archive byte count differs: " + name)
        roots = [PurePosixPath(value) for value in expected["roots"]]
        digest = hashlib.sha256()
        class Reader:
            def __init__(self, stream):
                self.stream = stream
            def read(self, size):
                data = self.stream.read(size)
                digest.update(data)
                return data
        with path.open("rb") as source:
            reader = Reader(source)
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    virtual = PurePosixPath("/") / member.name
                    if PurePosixPath(member.name).is_absolute() or ".." in virtual.parts or not inside(virtual, roots):
                        raise ValueError("Archive member outside declared roots")
                    target = root / str(virtual).lstrip("/")
                    if any(parent.is_symlink() for parent in target.parents if parent != root.parent):
                        raise ValueError("Extraction would traverse an existing symlink")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if member.isdir():
                        target.mkdir(exist_ok=True)
                    elif member.isfile():
                        with archive.extractfile(member) as incoming, target.open("xb") as outgoing:
                            while data := incoming.read(1024 * 1024):
                                outgoing.write(data)
                                raw += len(data)
                                delay = raw / rate - (time.monotonic() - started)
                                if delay > 0:
                                    time.sleep(min(delay, 30))
                                now = time.monotonic()
                                if now - last_report >= 15:
                                    write(report, {"status": "extracting", "archive": name, "raw_bytes": raw,
                                        "required_raw_bytes": required, "elapsed_seconds": now - started,
                                        "runtime_verified": False})
                                    last_report = now
                    elif member.issym():
                        resolved = PurePosixPath(os.path.normpath(str(virtual.parent / member.linkname)))
                        if not inside(resolved, all_roots):
                            raise ValueError("Archive symlink escapes the packaged roots")
                        target.symlink_to(member.linkname)
                    elif member.islnk():
                        link = PurePosixPath("/") / member.linkname
                        if ".." in link.parts or not inside(link, all_roots):
                            raise ValueError("Archive hard link escapes the packaged roots")
                        os.link(root / str(link).lstrip("/"), target, follow_symlinks=False)
                    else:
                        raise ValueError("Unsupported environment archive member")
                    if not member.issym():
                        target.chmod(member.mode)
            while reader.read(1024 * 1024):
                pass
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("Restoration stream SHA differs: " + name)
        completed.append({"archive": name, "sha256": digest.hexdigest(), "stream_verified": True})
    record = {"status": "extracted", "root": str(root), "marker": marker,
        "archives": completed, "raw_bytes": raw, "runtime_verified": False}
    write(root.parent / "extraction_completed.json", record)
    return record


def check_file(record, root, roots):
    virtual = PurePosixPath(record["path"])
    if not virtual.is_absolute() or not inside(virtual, roots):
        raise ValueError("An imported file is outside the restored archive prefixes: " + str(virtual))
    physical = root / str(virtual).lstrip("/")
    if not physical.resolve().is_relative_to(root):
        raise ValueError("Restored file unexpectedly resolves outside the extracted tree")
    stat = physical.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size) != (record["device"], record["inode"], record["bytes"]):
        raise ValueError("Runtime file inode does not match the independently restored file")
    return str(physical)


def verify(manifest, root, proot):
    extraction = json.loads((root.parent / "extraction_completed.json").read_text())
    if extraction["root"] != str(root) or extraction["status"] != "extracted":
        raise ValueError("Independent extraction evidence is absent")
    expected = {name: row["sha256"] for name, row in manifest["archives"].items()}
    if {row["archive"]: row["sha256"] for row in extraction["archives"]} != expected:
        raise ValueError("Extracted archives do not match the supplied manifest")
    marker = extraction["marker"]
    roots = [PurePosixPath(value) for row in manifest["archives"].values() for value in row["roots"]]
    for value in ("tmp", "proc", "dev", "root", "lib/x86_64-linux-gnu", "usr/lib/x86_64-linux-gnu", "lib64"):
        (root / value).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("probe_restored_environment.py"), root / ".restore-probe.py")
    bindings = system_bindings()
    local_libraries = Path(proot).parent.parent / "lib/x86_64-linux-gnu"
    launcher = (["/lib64/ld-linux-x86-64.so.2", "--library-path", str(local_libraries), str(proot)]
                if list(local_libraries.glob("libtalloc.so*")) else [str(proot)])
    base = launcher + ["-r", str(root), "-w", manifest["project"]]
    for path in bindings:
        base += ["-b", path + ":" + path]
    environment = {"HOME": "/root", "TMPDIR": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": manifest["project"] + "/src"}
    results = {}
    for name, inventory in manifest["inventory"].items():
        executable = inventory["executable"]
        env = {**environment, "PATH": str(Path(executable).parent)}
        try:
            process = subprocess.run(base + [executable, "/.restore-probe.py", "--name", name,
                "--project", manifest["project"], "--marker", marker], env=env, capture_output=True,
                text=True, timeout=240)
        except subprocess.TimeoutExpired:
            results[name] = {"verified": False, "error": "Restored runtime exceeded the 240-second CPU check limit"}
            continue
        rows = [line.removeprefix("RESTORE_RESULT=") for line in process.stdout.splitlines()
                if line.startswith("RESTORE_RESULT=")]
        item = {"returncode": process.returncode, "stdout": process.stdout[-6000:], "stderr": process.stderr[-6000:]}
        if process.returncode == 0 and len(rows) == 1:
            payload = json.loads(rows[0])
            if payload["executable"] != executable or payload["prefix"] != inventory["prefix"]:
                raise ValueError("Python original-prefix identity changed")
            origins = {"executable": check_file(payload["executable_file"], root, roots)}
            for module, record in payload["modules"].items():
                if record["status"] == "imported":
                    origins[module] = check_file(record["file"], root, roots)
            if payload.get("ocr", {}).get("status") == "passed":
                for field in ("binary", "language_data"):
                    origins["ocr_" + field] = check_file(payload["ocr"][field], root, roots)
            passed = all(row["status"] == "imported" for row in payload["modules"].values())
            passed &= payload["cpu_tensor"].get("status") == "passed" and payload["cpu_tensor"].get("cuda_device_count") == 0
            passed &= name != "tools" or payload.get("ocr", {}).get("status") == "passed"
            item.update(payload=payload, restored_file_origins=origins, verified=bool(passed))
        else:
            item["verified"] = False
        results[name] = item
    return {"status": "verified" if all(row["verified"] for row in results.values()) else "not_verified",
        "runtime_verified": all(row["verified"] for row in results.values()), "results": results,
        "system_bindings": bindings, "system_binding_access": "ordinary proot binds, not enforced read-only",
        "isolation_scope": "trusted archive pathname and inode provenance, not a security sandbox",
        "host_kernel_and_system_libraries_reused": True, "host_python_or_project_bound": False,
        "gpu_verified": False, "model_weights_or_gpu_kernels_tested": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stage", choices=("plan", "extract", "verify", "all"), default="plan")
    parser.add_argument("--root", type=Path, default=Path("/workspace/env-restore-check/root"))
    parser.add_argument("--proot", type=Path)
    parser.add_argument("--mib-per-second", type=float, default=16)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    root = args.root.absolute()
    report = args.report or root.parent / "restore-verification.json"
    if manifest["status"] != "complete" or manifest["unresolved_non_system_symlinks"]:
        raise ValueError("Environment archive manifest is incomplete")
    if not 0 < args.mib_per_second <= 32:
        raise ValueError("Use a positive bounded I/O rate up to 32 MiB/s")
    result = {"status": "planned", "runtime_verified": False, "root": str(root),
        "archive_bytes": sum(row["archive_bytes"] for row in manifest["archives"].values()),
        "raw_bytes": sum(row["raw_bytes"] for row in manifest["archives"].values()),
        "system_dependencies": ["Linux x86_64 kernel", "proot 5.1", "libtalloc2", "host glibc and system ELF libraries"],
        "active_environment_overwritten": False, "gpu_tested": False}
    if args.stage != "plan":
        if platform.system() != "Linux" or platform.machine() != manifest["machine"]:
            raise RuntimeError("Actual restore verification requires the archived Linux architecture")
        os.nice(19)
        if shutil.which("ionice"):
            subprocess.run([shutil.which("ionice"), "-c", "3", "-p", str(os.getpid())], check=True)
        if args.stage in {"extract", "all"}:
            result = extract(args.directory, manifest, root, args.mib_per_second * 1024 * 1024, report)
        if args.stage in {"verify", "all"}:
            proot = args.proot or shutil.which("proot")
            if not proot:
                raise RuntimeError("proot is unavailable; actual isolated runtime remains unverified")
            result = verify(manifest, root, proot)
    write(report, result)
    print(json.dumps({"report": str(report), "status": result["status"], "runtime_verified": result["runtime_verified"]}))
    if args.stage in {"verify", "all"} and not result["runtime_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
