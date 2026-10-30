#!/usr/bin/env bash
# Phase 2 - GPU data jobs (~6-8 GPU-h): graph annotation with the planner base model on both GPUs,
# K-tier / split assembly, initial canvases, one-time grounding+scoring of RL states, final check.
source "$(dirname "$0")/env.sh"
require_gpus 2 || exit 1
D=scripts/data
SESSIONS=$(ls "$REDAG_ROOT"/data/processed/sessions/*.jsonl | tr '\n' ' ')
IDS=$REDAG_ROOT/data/processed/sessions/annotate_ids.txt

annotate() {  # $1 = shard, $2 = device
    run_job annotate_shard$1 $PY $D/annotate_graphs.py --sessions $SESSIONS --ids "$IDS" --model "$PLANNER" \
        --device "$2" --shard "$1" --num_shards 2 --batch_size "${ANNOTATE_BATCH:-8}"
}
annotate 0 "$GPU0" & annotate 1 "$GPU1" & wait_all || exit 1

run_job build_sets $PY $D/build_sets.py --per_tier "${PER_TIER:-100}" --n_rs_states "${N_RS_STATES:-1000}" \
    --n_rl_train "${N_RL_TRAIN:-128}" --n_rl_dev "${N_RL_DEV:-16}" || exit 1

render() { run_job canvases_shard$1 $PY $D/render_canvases.py --model "$EDITOR" --device "$2" --shard "$1" --num_shards 2 \
    --size "${CANVAS_SIZE:-768}" --steps "${CANVAS_STEPS:-50}"; }
render 0 "$GPU0" & render 1 "$GPU1" & wait_all || exit 1

run_job precompute_rl_states $PY $D/precompute_states.py --files "$DATA/rl_train.jsonl" "$DATA/rl_dev.jsonl" \
    --verifier_model "$VERIFIER" --detector_model "$DETECTOR" --sam_model "$SAM" --device "$GPU0" || exit 1
run_job check_data_phase2 $PY $D/check_data.py || exit 1
echo "Phase 2 complete; files in $DATA (see manifest.json, check_report.json)"
