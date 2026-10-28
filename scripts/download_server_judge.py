import concurrent.futures
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download


project = Path.cwd()
deployment = project / "deployment"
sources = json.loads((deployment / "judge-transfer-sources.json").read_text())
metadata = json.loads((deployment / "judge-hf-metadata.json").read_text())
if sources["mismatched"] or len(sources["weights"]) != 14:
    raise RuntimeError("The domestic source does not contain the required checkpoint")
cache = Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen3-VL-32B-Instruct"
blobs = cache / "blobs"
snapshot = cache / "snapshots" / sources["hf_revision"]
blobs.mkdir(parents=True, exist_ok=True)
snapshot.mkdir(parents=True, exist_ok=True)
weights = {row["rfilename"]: row for row in metadata["siblings"]}
(deployment / "download-judge.pid").write_text(str(os.getpid()))


def download(row):
    name = row["filename"]
    total = row["size"]
    destination = blobs / weights[name]["lfs"]["sha256"]
    temporary = destination.with_name(destination.name + ".incomplete")
    link = snapshot / name
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    revision = row["modelscope_revision"]
    url = f"https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct/resolve/{revision}/{urllib.parse.quote(name)}"
    if destination.exists():
        if destination.stat().st_size != total:
            raise RuntimeError(f"Incomplete existing checkpoint: {name}")
    else:
        for attempt in range(8):
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == total:
                temporary.rename(destination)
                break
            if offset > total:
                raise RuntimeError(f"Unexpected checkpoint size: {name}")
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(url, headers=headers)
            try:
                started = time.monotonic()
                received = 0
                with opener.open(request, timeout=90) as response:
                    if offset and response.status == 200:
                        offset = 0
                    if offset and not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    ):
                        raise RuntimeError(f"Invalid resume response: {name}")
                    with temporary.open("ab" if offset else "wb") as stream:
                        reported = offset
                        while chunk := response.read(4 * 1024**2):
                            stream.write(chunk)
                            offset += len(chunk)
                            received += len(chunk)
                            if offset > total:
                                raise RuntimeError(f"Oversized response: {name}")
                            if offset - reported >= 128 * 1024**2:
                                elapsed = time.monotonic() - started
                                print(name, offset, total, round(received / elapsed / 1024**2, 2), "MiB/s", flush=True)
                                reported = offset
                if temporary.stat().st_size != total:
                    raise RuntimeError(f"Interrupted response: {name}")
                temporary.rename(destination)
                break
            except Exception as error:
                print("RETRY", name, attempt + 1, type(error).__name__, flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 30))
        else:
            raise RuntimeError(f"Download did not complete: {name}")
    if not link.exists():
        link.symlink_to(Path("../..") / "blobs" / destination.name)
    print("COMPLETE", name, total, flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, sources["weights"]))

path = snapshot_download(
    "Qwen/Qwen3-VL-32B-Instruct",
    revision=sources["hf_revision"],
    max_workers=4,
    allow_patterns=["*.json", "*.txt", "*.model", "*.jinja", "LICENSE", "README.md"],
)
(deployment / "download-judge.json").write_text(
    json.dumps({"path": path, "complete": True}, indent=2) + "\n"
)
print("DOWNLOAD_COMPLETE", path, flush=True)
