set -euo pipefail
cd /workspace/agentic-image-pilot-small
export PATH=/usr/local/miniconda3/envs/py312/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS=6
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p third_party
python -m zipfile -e /workspace/pilot-downloads/causal-conv1d-cd81f0413cad2fc1e6f17e785ac39f59aae690cd.zip third_party
python -m zipfile -e /workspace/pilot-downloads/flash-linear-attention-413ad2c655ece57b94bc9ddc51fdbd43cf09d984.zip third_party
python scripts/patch_causal.py third_party/causal-conv1d-cd81f0413cad2fc1e6f17e785ac39f59aae690cd
python - <<'PY'
from pathlib import Path
p = Path('third_party/causal-conv1d-cd81f0413cad2fc1e6f17e785ac39f59aae690cd/setup.py')
s = p.read_text()
s = s.replace('    if FORCE_CXX11_ABI:', '    if not HIP_BUILD:\n        cc_flag = ["-gencode", "arch=compute_89,code=sm_89"]\n\n    if FORCE_CXX11_ABI:')
p.write_text(s)
PY
python -m pip install --no-build-isolation third_party/causal-conv1d-cd81f0413cad2fc1e6f17e785ac39f59aae690cd
python -m pip install --no-build-isolation third_party/flash-linear-attention-413ad2c655ece57b94bc9ddc51fdbd43cf09d984
python -m pip freeze >deployment/rental-pip-freeze.txt
