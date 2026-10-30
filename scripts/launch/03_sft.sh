#!/usr/bin/env bash
# Phase 3 (editor SFT, planner QLoRA SFT + planner graph evaluation, rejection sampling, controller SFT) through the
# scheduler, including anything still missing before it.  The full pipeline is `bash scripts/launch/go.sh`.
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only '^(sft_controller|eval_planner_sft|eval_planner_zero_shot)$' "$@"
