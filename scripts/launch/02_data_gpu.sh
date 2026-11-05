#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only '^check_data_phase2$' "$@"
