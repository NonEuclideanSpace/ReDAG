#!/usr/bin/env bash
# Prepare a fresh AutoDL instance for ReDAG.  Idempotent: rerun after any interruption.  AutoDL no-GPU mode is enough
# (and cheapest) for everything here; switch to GPU mode afterwards and run `bash scripts/launch/go.sh`.
#
#   bash scripts/launch/bootstrap.sh                          packages + model weights (~120 GiB) + checks
#   bash scripts/launch/bootstrap.sh --data-bundle FILE       ... and import a data bundle (manifest from `go.sh bundle export data`)
#   bash scripts/launch/bootstrap.sh --skip-models            e.g. on a cloned instance that already has the weights
#
# Expected base image: Python 3.12, torch 2.12.x built for CUDA 13.0 (see scripts/launch/requirements.lock.txt).
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" || exit 1
BUNDLE="" SKIP_MODELS=0
while [ $# -gt 0 ]; do
    case $1 in
        --data-bundle) BUNDLE=$2; shift 2 ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
step() { echo; echo "=== $* ==="; }

step "1/5 Python and torch"
"$PY" - <<'EOF' || exit 1
import sys
import torch
print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA build', torch.version.cuda,
      '| GPUs visible', torch.cuda.device_count())
if sys.version_info[:2] != (3, 12):
    print('WARNING: validated with Python 3.12')
if not torch.__version__.startswith('2.12'):
    print('WARNING: validated with torch 2.12.1+cu130; other versions may work but are untested')
EOF
df -h /root/autodl-tmp | tail -n 1

step "2/5 packages (pinned, aliyun mirror)"
"$PY" -m pip install -q -r scripts/launch/requirements.lock.txt -i "${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}" || exit 1
"$PY" -m pip install -q --no-deps -e . || exit 1
"$PY" -c "import transformers, diffusers, peft; print('transformers', transformers.__version__, '| diffusers', diffusers.__version__, '| peft', peft.__version__)"

step "3/5 model weights (ModelScope, resumable)"
if [ "$SKIP_MODELS" = 1 ]; then
    "$PY" scripts/launch/download_models.py --check || echo "WARNING: weights incomplete; rerun without --skip-models"
else
    "$PY" scripts/launch/download_models.py --workers "${DOWNLOAD_WORKERS:-16}" || exit 1
fi

step "4/5 data"
if [ -n "$BUNDLE" ]; then
    "$PY" scripts/launch/pipeline.py bundle import "$BUNDLE" || exit 1
else
    echo "no bundle given: go.sh will build the data itself (downloads + conversion; best done in no-GPU mode)"
fi

step "5/5 checks"
"$PY" -m pytest -q -p no:cacheprovider tests 2>&1 | tail -n 2
"$PY" scripts/launch/pipeline.py status --brief
echo
echo "Next: bash scripts/launch/go.sh   (no-GPU mode: CPU data phase only; GPU mode: everything, starting with a GPU preflight)"
