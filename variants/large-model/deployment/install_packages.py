import json
import os
import subprocess
import sys
from pathlib import Path

stage = sys.argv[1]
project = Path.cwd()
python = project / (".venv/bin/python" if stage == "runtime" else ".venv-tools/bin/python")
wheels = json.loads((project / "deployment/torch-wheels.json").read_text())[stage]
os.environ["NO_PROXY"] += ",download.pytorch.org,download-r2.pytorch.org"
os.environ["UV_CONCURRENT_DOWNLOADS"] = "3"
Path(f"deployment/install-{stage}.pid").write_text(str(os.getpid()))

def install(*packages, extra=()):
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--default-index", "https://pypi.tuna.tsinghua.edu.cn/simple", *extra, *packages], check=True)

install(*wheels)
if stage == "runtime":
    install("-e", ".[dev]", "packaging", "ninja", "wheel", "setuptools")
    install("-r", "requirements-gpu.txt")
    install("cuda-toolkit[nvcc,crt,cudart,cccl,nvvm]==13.0.3.0")
    subprocess.run([str(python), "-c", "import torch,transformers,diffusers,peft,bitsandbytes; print(torch.__version__, transformers.__version__, diffusers.__version__, peft.__version__, bitsandbytes.__version__)"], check=True)
else:
    install("transformers==4.48.2", "numpy", "pillow", "pydantic>=2.10,<3", "scipy", "httpx", "fastapi", "uvicorn", "pyyaml", "setuptools", "wheel", "ninja", "packaging")
Path(f"deployment/install-{stage}-packages.json").write_text(json.dumps({"complete": True, "python": str(python)}, indent=2) + "\n")
print("PACKAGES_COMPLETE", stage, flush=True)
