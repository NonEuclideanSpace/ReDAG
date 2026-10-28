set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
export CUDA_HOME="$(.venv/bin/python -c 'import sys; print(sys.prefix + "/lib/python3.12/site-packages/nvidia/cu13")')"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-4}"
if [ ! -e "$CUDA_HOME/lib/libcudart.so" ] && [ -f "$CUDA_HOME/lib/libcudart.so.13" ]; then
    ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
mkdir -p third_party/archives
for item in 'Dao-AILab causal-conv1d cd81f0413cad2fc1e6f17e785ac39f59aae690cd' 'fla-org flash-linear-attention 413ad2c655ece57b94bc9ddc51fdbd43cf09d984'; do
    read -r owner name revision <<< "$item"
    archive="third_party/archives/$name-$revision.zip"
    if [ ! -f "$archive" ]; then
        curl --noproxy '*' --fail --location --connect-timeout 20 --retry 3 --retry-all-errors "https://codeload.github.com/$owner/$name/zip/$revision" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    if [ ! -d "third_party/$name-$revision" ]; then
        .venv/bin/python -m zipfile -e "$archive" third_party
    fi
    if [ "$name" = causal-conv1d ]; then
        .venv/bin/python scripts/patch_causal.py "third_party/$name-$revision"
    fi
    uv pip install --python .venv/bin/python --no-build-isolation "third_party/$name-$revision"
done
.venv/bin/python -c 'import torch; import causal_conv1d_cuda; import importlib.metadata as m; print(m.version("causal-conv1d"), m.version("flash-linear-attention"))'
printf 'KERNELS_INSTALLED\n'
