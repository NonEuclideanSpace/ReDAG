#!/usr/bin/env bash
# Preflight after switching to GPU mode: GPUs, complete weights, imports, unit tests, data files.
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" || exit 1
status=0
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || status=1
require_gpus 2 || status=1
for model in "$PLANNER" "$CONTROLLER" "$EDITOR" "$VERIFIER" "$DETECTOR" "$SAM"; do
    if [ ! -d "$model" ]; then echo "MISSING  $model"; status=1; continue; fi
    partial=$(find -L "$model" -name '*.incomplete' | wc -l)
    size=$(du -shL "$model" 2>/dev/null | cut -f1)
    if [ -f "$model/config.json" ] || [ -f "$model/model_index.json" ]; then cfg=ok; else cfg=NO-CONFIG; status=1; fi
    [ "$partial" -eq 0 ] || status=1
    echo "$(basename "$model"): size=$size config=$cfg partial_files=$partial"
done
$PY - <<'EOF' || status=1
import torch, transformers, diffusers, peft
from transformers import AutoModelForMultimodalLM, AutoModelForImageTextToText, Sam2Model
from diffusers import Flux2KleinPipeline
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(),
      '| transformers', transformers.__version__, '| diffusers', diffusers.__version__, '| peft', peft.__version__)
EOF
$PY -m pytest -q tests 2>&1 | tail -n 3 || status=1
$PY scripts/data/check_data.py --allow_missing_canvases || echo "(data check reported problems; see data/final/check_report.json)"
df -h /root/autodl-tmp | tail -n 1
exit $status
