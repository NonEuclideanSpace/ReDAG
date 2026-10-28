import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time


HOST = "dongzk@219.216.65.42"
PROJECT = "/home/hdd4/dongzk/projects/agentic-image-v3"
STORAGE = "/home/hdd3/dongzk/agentic-image-v3"
BULK = "/mnt/nfs0/dongzk/agentic-image-v3"
SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o",
    "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "Compression=no",
]
MODELS = [
    "geneval2-qwen3-vl-8b", "judge-source", "actor-source", "renderer-source",
    "tools", "geneval",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def remote(code, timeout=None):
    return subprocess.run(
        [*SSH, HOST, "python3 -"], input=code, text=True, capture_output=True,
        check=True, timeout=timeout,
    ).stdout


def inventory():
    return json.loads(remote(
        "from pathlib import Path\nimport os,json\n"
        f"base=Path({STORAGE!r})/'models'\nresult={{}}\n"
        f"for name in {MODELS!r}:\n"
        " root=(base/name).resolve();files={}\n"
        " for folder,dirs,names in os.walk(root,followlinks=True):\n"
        "  dirs[:]=[n for n in dirs if n not in {'.git','.cache'}]\n"
        "  for n in names:\n"
        "   p=Path(folder)/n;files[str(p.relative_to(root))]=p.stat().st_size\n"
        " result[name]={'source':str(root),'files':files,'bytes':sum(files.values())}\n"
        "print(json.dumps(result))\n", timeout=90,
    ))


def prepare_archive(name, parent, members):
    output = f"{STORAGE}/tmp/local-backup-export/{name}.tar.gz"
    code = (
        "from pathlib import Path\nimport subprocess\n"
        f"destination=Path({output!r})\n"
        "destination.parent.mkdir(parents=True,exist_ok=True)\n"
        "if not destination.exists():\n"
        " temporary=destination.with_suffix(destination.suffix+'.part')\n"
        f" subprocess.run(['tar','-I','pigz -1 -p 2','-cf',str(temporary),'-C',{parent!r},*{members!r}],check=True)\n"
        " temporary.replace(destination)\n"
        "print(destination.stat().st_size)\n"
    )
    return output, int(remote(code).strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    if not Path("/Volumes/TOSHIBA EXT").is_mount():
        raise RuntimeError("The destination external disk is not mounted")
    metadata = destination / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    model_manifest = inventory()
    (metadata / "model-files.json").write_text(json.dumps(model_manifest, indent=2) + "\n")
    state = {
        "started_at": now(), "pid": os.getpid(), "status": "running",
        "source_host": HOST, "destination": str(destination),
        "source_project": PROJECT, "source_storage": STORAGE, "source_data": BULK,
        "jobs": {}, "credentials_included": False,
    }
    lock = threading.Lock()

    def save(name=None, **changes):
        with lock:
            if name is not None:
                state["jobs"].setdefault(name, {}).update(changes)
            state["updated_at"] = now()
            temporary = metadata / "backup-status.json.part"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(metadata / "backup-status.json")

    def transfer(name, source, target, expected):
        target.parent.mkdir(parents=True, exist_ok=True)
        save(name, status="transferring", source=source, target=str(target), expected_bytes=expected)
        command = [
            "rsync", "-rltL", "--partial", "--append", "--modify-window=2", "--stats",
            "--exclude=.git", "--exclude=.cache", "--timeout=120", "-e", shlex.join(SSH),
            f"{HOST}:{source}", str(target) + ("/" if source.endswith("/") else ""),
        ]
        for attempt in range(1, 5):
            if not Path("/Volumes/TOSHIBA EXT").is_mount():
                raise RuntimeError("The external disk was disconnected")
            save(name, attempt=attempt)
            with (metadata / f"transfer-{name}.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                return
            save(name, status="retrying", last_exit_code=result.returncode)
            print(f"{now()} {name}: transfer failed ({result.returncode}), attempt {attempt}", flush=True)
            time.sleep(min(5 * attempt, 20))
        raise RuntimeError(f"Transfer failed after four attempts: {name}")

    def model_job(name):
        try:
            details = model_manifest[name]
            target = destination / "models" / name
            transfer(name, details["source"] + "/", target, details["bytes"])
            mismatches = [
                relative for relative, size in details["files"].items()
                if not (target / relative).is_file() or (target / relative).stat().st_size != size
            ]
            if mismatches:
                raise RuntimeError(f"Missing or incomplete model files: {mismatches}")
            save(name, status="complete", completed_at=now(), files=len(details["files"]), local_bytes=details["bytes"])
            print(f"{now()} completed {name}", flush=True)
        except Exception as error:
            save(name, status="failed", error=str(error))
            print(f"{now()} failed {name}: {error}", flush=True)

    def archives_job():
        archives = [
            ("school-project", "/home/hdd4/dongzk/projects", ["agentic-image-v3"], "code"),
            ("linux-environments", STORAGE, ["environments", "tools"], "environments"),
            ("prepared-data-and-runs", BULK, ["data", "runs"], "data"),
        ]
        for name, parent, members, category in archives:
            try:
                save(name, status="packing", original_parent=parent, members=members)
                source, size = prepare_archive(name, parent, members)
                target = destination / category / f"{name}.tar.gz"
                transfer(name, source, target, size)
                if target.stat().st_size != size:
                    raise RuntimeError("The transferred archive size does not match its source")
                save(name, status="complete", completed_at=now(), local_bytes=size)
                print(f"{now()} completed {name}", flush=True)
            except Exception as error:
                save(name, status="failed", error=str(error))
                print(f"{now()} failed {name}: {error}", flush=True)

    for name in MODELS:
        save(name, status="queued", expected_bytes=model_manifest[name]["bytes"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(archives_job)] + [pool.submit(model_job, name) for name in MODELS]
        while any(not future.done() for future in futures):
            time.sleep(30)
            with lock:
                jobs = list(state["jobs"].items())
            total = 0
            for name, job in jobs:
                target = Path(job["target"]) if job.get("target") else None
                if target and target.exists():
                    if target.is_file():
                        size = target.stat().st_size
                    else:
                        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
                    total += size
                    save(name, local_bytes=size)
            print(f"{now()} local transferred bytes: {total}", flush=True)
        for future in futures:
            future.result()
    state["status"] = "complete" if all(job["status"] == "complete" for job in state["jobs"].values()) else "incomplete"
    state["finished_at"] = now()
    save()
    print(json.dumps({"status": state["status"], "finished_at": state["finished_at"]}), flush=True)


if __name__ == "__main__":
    main()
