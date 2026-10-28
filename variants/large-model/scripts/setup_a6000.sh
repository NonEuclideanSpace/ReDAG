set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl
uv pip install --python .venv/bin/python -r requirements-gpu.txt
uv pip install --python .venv/bin/python packaging ninja wheel setuptools 'cuda-toolkit[nvcc,crt,cudart,cccl,nvvm]==13.0.3.0'
bash scripts/setup_kernels.sh
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/agentic-image hardware --config configs/a6000.yaml
