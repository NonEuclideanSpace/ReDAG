#!/usr/bin/env bash
# Phase 1 - CPU only, safe in AutoDL no-GPU mode (0.5 core / 2 GiB): raw benchmarks -> sessions -> annotation plan.
# Idempotent: finished steps are skipped (runs/logs/*.done).  Downloads (~32 GiB: MagicBrush ~25 GiB, CoReBench
# FLUX.2-klein-9B canvases ~7 GiB) can wait for another job to end first: WAIT_PID=<pid> (e.g. the weight download).
# MagicBrush dev is fetched and converted first so the converter is checked on real data early.
source "$(dirname "$0")/env.sh"
MB_SPLITS=${MB_SPLITS:-"dev train"}
D=scripts/data
CANVAS_DIR=${CANVAS_DIR:-$REDAG_ROOT/data/raw/corebench_images/FLUX.2-klein-9B}

run_job fetch_corebench $PY $D/fetch_sources.py --what corebench ${WAIT_PID:+--wait_pid $WAIT_PID} || exit 1
for split in $MB_SPLITS; do
    run_job fetch_magicbrush_$split $PY $D/fetch_sources.py --what magicbrush --splits $split --workers "${FETCH_WORKERS:-6}" || exit 1
    run_job prep_magicbrush_$split $PY $D/prep_magicbrush.py --splits $split --size 512 --workers "${WORKERS:-1}" || exit 1
    if [ "$split" = dev ]; then
        run_job fetch_corebench_images $PY $D/fetch_sources.py --what corebench_images --archives FLUX.2-klein-9B || exit 1
    fi
done
CANVAS_ARG=""
[ -d "$CANVAS_DIR" ] && CANVAS_ARG="--canvas_dir $CANVAS_DIR"
run_job prep_checklists $PY $D/prep_checklists.py --what corebench lgbench longbench $CANVAS_ARG || exit 1
run_job plan_annotation $PY $D/plan_annotation.py || exit 1
echo "Phase 1 complete. Sessions: $REDAG_ROOT/data/processed/sessions; next (GPU): 00_check_env.sh, smoke_test.sh, 02_data_gpu.sh"
