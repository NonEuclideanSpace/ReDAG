import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub._local_folder import get_local_download_paths, write_download_metadata


source = json.loads(Path("deployment/bert-transfer.json").read_text())
directory = Path("models/tools/bert-base-uncased")
paths = get_local_download_paths(directory, "model.safetensors")
temporary = paths.incomplete_path(source["etag"])
url = (
    "https://modelscope.cn/models/google-bert/bert-base-uncased/resolve/"
    f"{source['revision']}/model.safetensors"
)
subprocess.run(
    [
        "curl",
        "--noproxy",
        "*",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--continue-at",
        "-",
        "--output",
        str(temporary),
        url,
    ],
    check=True,
)
if temporary.stat().st_size != source["size"]:
    raise RuntimeError("The BERT download is incomplete")
temporary.replace(paths.file_path)
write_download_metadata(
    directory,
    "model.safetensors",
    "86b5e0934494bd15c9632b12f734a8a67f723594",
    source["etag"],
)
subprocess.run([sys.executable, "scripts/prepare_tool_assets.py"], check=True)
print("BERT_COMPLETE", flush=True)
