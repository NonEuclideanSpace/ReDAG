#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
export EXPERIMENTS=${EXPERIMENTS:-${EXP:-"1 2 3"}}
targets=()
for e in $EXPERIMENTS; do case $e in 1) targets+=(merge_exp1) ;; 2) targets+=(exp2_eval) ;; 3) targets+=(merge_exp3) ;; esac; done
pattern="^($(IFS='|'; echo "${targets[*]}"))\$"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only "$pattern" "$@"
