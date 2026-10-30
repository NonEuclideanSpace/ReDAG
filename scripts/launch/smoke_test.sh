#!/usr/bin/env bash
# First GPU hour: run every GPU stage end-to-end on a handful of records (~30-45 min) to catch
# model/API/memory problems and to measure real per-step costs before the long jobs start.
# Timings land in runs/smoke/logs/timings.jsonl; scripts/launch/README.md explains how to extrapolate.
RUNS=${RUNS:-/root/autodl-tmp/ReDAG/runs}/smoke
source "$(dirname "$0")/env.sh"
require_gpus 2 || exit 1
S=$RUNS/data
$PY scripts/launch/smoke_subset.py "$DATA" "$S" || exit 1

run_job smoke_sft_editor $PY scripts/run_sft_editor.py --data "$S/editor_sft.jsonl" --output "$RUNS/editor" \
    --model "$EDITOR" --device "$GPU1" || exit 1
run_job smoke_rs $PY scripts/collect_controller_rs.py $RUNTIME --editor_adapter "$RUNS/editor" --device "$GPU0" \
    --verifier_device "$GPU0" --editor_device "$GPU1" --num_denoise_steps "$STEPS_EVAL" \
    --data "$S/rs_states.jsonl" --output "$S/controller_sft.jsonl" || exit 1
CTRL_DATA=$S/controller_sft.jsonl
[ -s "$CTRL_DATA" ] || CTRL_DATA=$S/controller_sft_synthetic.jsonl   # no accepted sample: still exercise the trainer
run_job smoke_sft_controller $PY scripts/run_sft_controller.py --data "$CTRL_DATA" --output "$RUNS/controller" \
    --model "$CONTROLLER" --device "$GPU0" || exit 1
head -n 4 "$DATA/planner_sft.jsonl" > "$S/planner_sft.jsonl"
run_job smoke_sft_planner $PY scripts/run_sft_planner.py --data "$S/planner_sft.jsonl" \
    --output "$RUNS/planner" --model "$PLANNER" --device "$GPU1" || exit 1
run_job smoke_rl $PY scripts/run_rl_alternating.py $RUNTIME --controller_adapter "$RUNS/controller" --editor_adapter "$RUNS/editor" \
    --device "$GPU0" --verifier_device "$GPU0" --editor_device "$GPU1" --num_denoise_steps "$STEPS_RL" \
    --data "$S/rl_train.jsonl" --dev_data "$S/rl_dev.jsonl" --output "$RUNS/rl" || exit 1
run_job smoke_eval $PY scripts/run_eval_benchmarks.py $RUNTIME --exp_id 1 --exp1_modes redag_full --allow-small-dataset \
    --data "$S/eval.jsonl" --controller_adapter "$RUNS/controller" --editor_adapter "$RUNS/editor" \
    --max_steps 2 --generation_budget 3 --verification_budget 200 --num_denoise_steps "$STEPS_EVAL" \
    --device "$GPU0" --verifier_device "$GPU0" --editor_device "$GPU1" --output "$RUNS/eval/exp1_smoke.json" || exit 1
echo "--- smoke timings (seconds) ---"; cat "$LOGS/timings.jsonl"
$PY - "$RUNS" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in (root / 'rl' / 'metrics.jsonl').read_text().splitlines() if l.strip()]
for r in rows:
    print(f"RL {r['stage']} step {r['step']}: {r['step_wall_seconds']:.0f}s for {r['sampled_candidates']} candidates "
          f"(peak GiB {[round(v / 2**30, 1) for v in r['peak_cuda_allocated_bytes'].values()]})")
EOF
