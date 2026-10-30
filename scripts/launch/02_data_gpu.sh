#!/usr/bin/env bash
# Phase 2 (graph annotation, K-tier sets, canvases, RL state grounding, data check), run by the scheduler together with
# everything it still needs.  The full pipeline is `bash scripts/launch/go.sh`; this wrapper stops after phase 2.
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only '^check_data_phase2$' "$@"
