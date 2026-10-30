#!/usr/bin/env bash
# Phase 4 - alternating RL (eq. 34, 39, 55): controller (+verifier/grounder) on GPU0, editor on GPU1.
#   main run      : subgraph_credit_ours on rl_train (default 128 parents), source of the experiment-3 checkpoints
#   experiment 2  : the three reward constructions on the same 64-parent subset with identical settings
#                   (RUN_EXP2=0 skips them; their logs feed run_eval_benchmarks.py --exp_id 2 --training_logs)
source "$(dirname "$0")/env.sh"
require_gpus 2 || exit 1
SFT=$RUNS/sft
COMMON="$RUNTIME --controller_adapter $SFT/controller --editor_adapter $SFT/editor --device $GPU0 --verifier_device $GPU0 \
--editor_device $GPU1 --num_denoise_steps $STEPS_RL --rollout_steps $ROLLOUT_STEPS --dev_data $DATA/rl_dev.jsonl"

run_job rl_main $PY scripts/run_rl_alternating.py $COMMON --data "$DATA/rl_train.jsonl" \
    --reward_mode subgraph_credit_ours --output "$RUNS/rl/main" || exit 1

if [ "${RUN_EXP2:-1}" = 1 ]; then
    head -n "${N_EXP2:-64}" "$DATA/rl_train.jsonl" > "$DATA/rl_train_exp2.jsonl"
    for mode in subgraph_credit_ours step_full_table terminal_broadcast; do
        run_job rl_exp2_$mode $PY scripts/run_rl_alternating.py $COMMON --data "$DATA/rl_train_exp2.jsonl" \
            --reward_mode $mode --output "$RUNS/rl/exp2/$mode" || exit 1
    done
    $PY - "$RUNS/rl/exp2" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
modes = ('terminal_broadcast', 'step_full_table', 'subgraph_credit_ours')
(root / 'training_logs.json').write_text(json.dumps({m: f'{m}/metrics.jsonl' for m in modes}, indent=2))
print('wrote', root / 'training_logs.json')
EOF
fi
echo "Phase 4 complete: stage checkpoints in $RUNS/rl/main/stage_checkpoints.json"
