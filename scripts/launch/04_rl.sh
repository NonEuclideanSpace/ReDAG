#!/usr/bin/env bash
# Phase 4 (main alternating RL + the three experiment-2 reward runs) through the scheduler, including anything still
# missing before it.  RL_LAYOUT=auto keeps a run on one GPU when every GPU has >= 90 GiB, else uses two GPUs.
source "$(dirname "$0")/env.sh"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only '^(rl_main|exp2_logs)$' "$@"
