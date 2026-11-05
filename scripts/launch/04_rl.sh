#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only '^(rl_main|exp2_logs)$' "$@"
