#!/usr/bin/env bash
# Phase 5 (experiments 1 / 2 / 3, merged) through the scheduler, including anything still missing before it.
# EXP="1 3" (or EXPERIMENTS="1 3") selects experiments, as before.
source "$(dirname "$0")/env.sh"
export EXPERIMENTS=${EXPERIMENTS:-${EXP:-"1 2 3"}}
targets=()
for e in $EXPERIMENTS; do case $e in 1) targets+=(merge_exp1) ;; 2) targets+=(exp2_eval) ;; 3) targets+=(merge_exp3) ;; esac; done
pattern="^($(IFS='|'; echo "${targets[*]}"))\$"
cd "$REDAG_ROOT" && exec "$PY" scripts/launch/pipeline.py run --only "$pattern" "$@"
