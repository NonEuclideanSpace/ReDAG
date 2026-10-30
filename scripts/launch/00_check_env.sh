#!/usr/bin/env bash
# Preflight after switching to GPU mode (~5-10 min): GPUs, complete weights, kernels, unit tests, a full GPU
# wiring check with tiny models (SFT, rejection sampling, alternating RL and an evaluation session), data files.
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" || exit 1
status=0
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || status=1
require_gpus "${MIN_GPUS:-2}" || status=1   # the scheduler runs this on one GPU (MIN_GPUS=1)
$PY - "$PLANNER" "$CONTROLLER" "$EDITOR" "$VERIFIER" "$DETECTOR" "$SAM" <<'EOF' || status=1
import json, sys
from pathlib import Path
bad = 0
for root in map(Path, sys.argv[1:]):
    if not root.is_dir():
        print('MISSING', root); bad += 1; continue
    partial = list(root.rglob('*.incomplete'))
    missing = []
    for index in root.rglob('*.safetensors.index.json'):  # every shard named in the index must exist and be non-empty
        for shard in set(json.loads(index.read_text())['weight_map'].values()):
            path = index.parent / shard
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path.relative_to(root)))
    weights = sum(p.stat().st_size for p in root.rglob('*.safetensors')) / 2**30
    ok = not partial and not missing and weights > 0
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {root.name}: {weights:.1f} GiB safetensors, partial={len(partial)}, missing_shards={missing[:3]}")
sys.exit(1 if bad else 0)
EOF
$PY - <<'EOF' || status=1
import torch, transformers, diffusers, peft
from transformers import AutoModelForMultimodalLM, AutoModelForImageTextToText, Sam2Model
from transformers.utils.import_utils import is_flash_linear_attention_available
from diffusers import Flux2KleinPipeline
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(),
      '| transformers', transformers.__version__, '| diffusers', diffusers.__version__, '| peft', peft.__version__)
if not is_flash_linear_attention_available():
    print('WARNING: flash-linear-attention missing; Qwen3.5 linear attention will use the slow PyTorch path '
          '(pip install flash-linear-attention)')
EOF
$PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -n 3 || status=1
# flash-linear-attention speeds up Qwen3.5's Gated-DeltaNet layers (controller, planner) several-fold. Its Triton kernels
# only work with a GPU, so it is installed here and kept only if the GPU wiring check passes with it.
FLA_NEW=0
if [ "${INSTALL_FLA:-1}" = 1 ] && ! $PY -c 'import fla' 2>/dev/null; then
    pip install -q flash-linear-attention > "$LOGS/pip_fla.log" 2>&1 && FLA_NEW=1 && echo "installed flash-linear-attention"
fi
if ! $PY scripts/launch/wiring_check.py --device "$GPU0" > "$LOGS/wiring_check.log" 2>&1; then
    if [ $FLA_NEW = 1 ]; then
        echo "wiring check failed with flash-linear-attention; removing it and retrying with the PyTorch fallback"
        pip uninstall -y -q flash-linear-attention fla-core >> "$LOGS/pip_fla.log" 2>&1
        $PY scripts/launch/wiring_check.py --device "$GPU0" > "$LOGS/wiring_check.log" 2>&1 || status=1
    else
        status=1
    fi
fi
grep -E '^(ALL|FAILED)' "$LOGS/wiring_check.log"
$PY scripts/data/check_data.py --allow_missing_canvases || echo "(data check reported problems; see data/final/check_report.json)"
df -h /root/autodl-tmp | tail -n 1
[ $status -eq 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED (see messages above and $LOGS/wiring_check.log)"
exit $status
