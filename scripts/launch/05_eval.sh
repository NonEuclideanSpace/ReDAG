#!/usr/bin/env bash
# Phase 5 - experiments of section 5.2 under one fixed per-session budget for every method.
#   EXP=1  environment ablation, SFT weights: flat_full / tree_rollback_only / redag_full (EXP1_EXTRA=1 adds the
#          spatial_local and spatial_no_gates ablations)
#   EXP=2  credit-assignment diagnostics on eval_credit.jsonl + the three training logs from phase 4
#   EXP=3  five-stage ladder x 3 seeds; one job per stage and shard so any part can be rerun alone
# Default layout: one process per GPU holding all four models (~71 GB); EVAL_LAYOUT=split uses one process over both GPUs.
source "$(dirname "$0")/env.sh"
require_gpus 2 || exit 1
EXP=${EXP:-"1 2 3"}
OUT=$RUNS/eval
mkdir -p "$OUT"
SFT=$RUNS/sft
BUDGET="--max_steps $MAX_STEPS --generation_budget $GEN_BUDGET --verification_budget $VER_BUDGET --num_denoise_steps $STEPS_EVAL"

two_shards() {  # $1 = job prefix, rest = arguments; runs shard 0 on GPU0 and shard 1 on GPU1, outputs <prefix>.s<k>.json
    local name=$1; shift
    if [ "${EVAL_LAYOUT:-shard}" = split ]; then
        run_job "$name" $PY scripts/run_eval_benchmarks.py $RUNTIME $BUDGET "$@" --device "$GPU0" --verifier_device "$GPU0" \
            --editor_device "$GPU1" --output "$OUT/$name.json"
        return
    fi
    for k in 0 1; do
        local dev=$GPU0; [ $k = 1 ] && dev=$GPU1
        run_job "$name.s$k" $PY scripts/run_eval_benchmarks.py $RUNTIME $BUDGET "$@" --device "$dev" --verifier_device "$dev" \
            --editor_device "$dev" --shard $k --num_shards 2 --output "$OUT/$name.s$k.json" &
    done
    wait_all
}

if [[ " $EXP " == *" 1 "* ]]; then
    MODES="flat_full tree_rollback_only redag_full"
    [ "${EXP1_EXTRA:-0}" = 1 ] && MODES="$MODES spatial_local spatial_no_gates"
    two_shards exp1 --exp_id 1 --data "$DATA/eval_main.jsonl" --exp1_modes $MODES \
        --controller_adapter "$SFT/controller" --editor_adapter "$SFT/editor" || exit 1
    [ "${EVAL_LAYOUT:-shard}" = split ] || run_job merge_exp1 $PY scripts/merge_eval_results.py --inputs "$OUT"/exp1.s0.json "$OUT"/exp1.s1.json --output "$OUT/exp1.json" || exit 1
fi

if [[ " $EXP " == *" 2 "* ]]; then
    run_job exp2 $PY scripts/run_eval_benchmarks.py $RUNTIME --exp_id 2 --data "$DATA/eval_credit.jsonl" \
        --training_logs "$RUNS/rl/exp2/training_logs.json" --controller_adapter "$SFT/controller" --editor_adapter "$SFT/editor" \
        --max_steps "${EXP2_MAX_STEPS:-3}" --generation_budget "$GEN_BUDGET" --verification_budget "$VER_BUDGET" \
        --num_denoise_steps "$STEPS_EVAL" --device "$GPU0" --verifier_device "$GPU0" --editor_device "$GPU1" \
        --output "$OUT/exp2.json" || exit 1
fi

if [[ " $EXP " == *" 3 "* ]]; then
    for stage in ${STAGES:-zero_shot sft ctrl_rl edit_rl alternating_rl}; do
        two_shards exp3_$stage --exp_id 3 --data "$DATA/eval_main.jsonl" \
            --stage_checkpoints "$RUNS/rl/main/stage_checkpoints.json" --stages $stage || exit 1
    done
    parts=$(ls "$OUT"/exp3_*.json 2>/dev/null | grep -v blind | tr '\n' ' ')
    run_job merge_exp3 $PY scripts/merge_eval_results.py --inputs $parts --output "$OUT/exp3.json" || exit 1
fi
echo "Evaluation outputs in $OUT (human annotations: run_eval_benchmarks.py --resume_results ... --annotations/--human_ratings)"
