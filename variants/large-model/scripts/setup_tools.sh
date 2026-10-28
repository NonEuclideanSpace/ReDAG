set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null
if [ ! -x .venv-tools/bin/python ]; then uv venv --python 3.10 .venv-tools; fi
uv pip install --python .venv-tools/bin/python https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp310-cp310-linux_x86_64.whl https://download-r2.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp310-cp310-linux_x86_64.whl
uv pip install --python .venv-tools/bin/python transformers==4.48.2 numpy pillow 'pydantic>=2.10,<3' scipy httpx fastapi uvicorn pyyaml setuptools wheel ninja packaging
mkdir -p third_party models/tools
if [ ! -d third_party/GroundingDINO ]; then git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO; fi
git -C third_party/GroundingDINO checkout 856dde20aee659246248e20734ef9ba5214f5e44
.venv-tools/bin/python scripts/patch_grounding.py third_party/GroundingDINO
TORCH_CUDA_ARCH_LIST=8.6 uv pip install --python .venv-tools/bin/python --no-build-isolation -e third_party/GroundingDINO
SAM2_BUILD_CUDA=0 uv pip install --python .venv-tools/bin/python --no-build-isolation git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
if [ ! -d third_party/dinov2 ]; then git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2; fi
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
curl --fail --location --continue-at - https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -o models/tools/groundingdino_swint_ogc.pth
curl --fail --location --continue-at - https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -o models/tools/sam2.1_hiera_small.pt
curl --fail --location --continue-at - https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -o models/tools/dinov2_vitb14_pretrain.pth
.venv-tools/bin/python scripts/prepare_tool_assets.py
