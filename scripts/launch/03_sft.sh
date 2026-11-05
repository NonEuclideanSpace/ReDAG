#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only '^(sft_controller|eval_planner_sft|eval_planner_zero_shot)$' "$@"
