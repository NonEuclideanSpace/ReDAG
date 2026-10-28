set -euo pipefail
cd "$(dirname "$0")/.."
storage="${AGENTIC_STORAGE:-$PWD/.runtime}"
mkdir -p "$storage/tools" "$storage/tmp"
if [ ! -x "$storage/tools/bin/micromamba" ]; then
    curl --fail --location --retry 3 --connect-timeout 20 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2 -o "$storage/tmp/micromamba.tar.bz2"
    tar -xjf "$storage/tmp/micromamba.tar.bz2" -C "$storage/tools" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$storage/tools/mamba"
"$storage/tools/bin/micromamba" create --yes --prefix "$storage/environments/ocr" --override-channels --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge tesseract=5.5.0
curl --fail --location --retry 3 --connect-timeout 20 https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/eng.traineddata -o "$storage/environments/ocr/share/tessdata/eng.traineddata"
ln -sfn "$storage/environments/ocr/bin/tesseract" "$storage/tools/bin/tesseract"
"$storage/tools/bin/tesseract" --version
