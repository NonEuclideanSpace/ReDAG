#!/usr/bin/env bash
# Phase 3 - supervised stages (eq. 17, 19-20, 24), ~17 GPU-h:
#   editor SFT (GPU1) || planner QLoRA SFT (GPU0)  ->  controller rejection sampling with the SFT editor
#   (N_state x 3 candidates, two processes)  ->  controller SFT.
source "$(dirname "$0")/env.sh"
require_gpus 2 || exit 1
SFT=$RUNS/sft
mkdir -p "$SFT"

run_job sft_editor  $PY scripts/run_sft_editor.py  --data "$DATA/editor_sft_train.jsonl" --output "$SFT/editor" \
    --model "$EDITOR" --device "$GPU1" --epochs "${EDITOR_EPOCHS:-1}" &
run_job sft_planner $PY scripts/run_sft_planner.py --data "$DATA/planner_sft.jsonl" --output "$SFT/planner" \
    --model "$PLANNER" --device "$GPU0" --epochs "${PLANNER_EPOCHS:-1}" &
wait_all || exit 1

# Rejection sampling: one process per GPU, each holding controller + editor(SFT) + verifier + grounder (~71 GB).
# RS_LAYOUT=split runs a single process with the editor on GPU1 instead (use it if a shard runs out of memory).
if [ "${RS_LAYOUT:-shard}" = shard ]; then
    $PY - "$DATA/controller_rs_states.jsonl" <<'EOF'
import sys, pathlib
src = pathlib.Path(sys.argv[1]); lines = [l for l in src.read_text().splitlines() if l.strip()]
for k in (0, 1):
    (src.parent / f'controller_rs_states.part{k}.jsonl').write_text(''.join(l + '\n' for i, l in enumerate(lines) if i % 2 == k))
EOF
    rs() { run_job rs_part$1 $PY scripts/collect_controller_rs.py $RUNTIME --editor_adapter "$SFT/editor" \
        --device "$2" --editor_device "$2" --verifier_device "$2" --num_denoise_steps "$STEPS_EVAL" \
        --data "$DATA/controller_rs_states.part$1.jsonl" --output "$DATA/controller_sft.part$1.jsonl" --seed $((42 + $1 * 100000)); }
    rs 0 "$GPU0" & rs 1 "$GPU1" & wait_all || exit 1
    cat "$DATA/controller_sft.part0.jsonl" "$DATA/controller_sft.part1.jsonl" > "$DATA/controller_sft.jsonl"
else
    run_job rs_single $PY scripts/collect_controller_rs.py $RUNTIME --editor_adapter "$SFT/editor" \
        --device "$GPU0" --editor_device "$GPU1" --verifier_device "$GPU0" --num_denoise_steps "$STEPS_EVAL" \
        --data "$DATA/controller_rs_states.jsonl" --output "$DATA/controller_sft.jsonl" || exit 1
fi
echo "accepted rejection-sampling rows: $(wc -l < "$DATA/controller_sft.jsonl")"
run_job check_controller_sft $PY scripts/data/check_data.py || exit 1
run_job sft_controller $PY scripts/run_sft_controller.py --data "$DATA/controller_sft.jsonl" --output "$SFT/controller" \
    --model "$CONTROLLER" --device "$GPU0" --epochs "${CONTROLLER_EPOCHS:-2}" || exit 1
echo "Phase 3 complete: adapters in $SFT/{editor,planner,controller}"
