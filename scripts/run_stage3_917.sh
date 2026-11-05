#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
R=/root/autodl-tmp/ReDAG
M=/root/autodl-tmp/models
cd "$R"
mkdir -p runs/logs runs/eval

STATUS=runs/logs/stage3_status_917.txt
note() { echo "$(date -u '+%F %T UTC') [917-stage3] $*" | tee -a "$STATUS"; }

ADAPTER_SHA=$(sha256sum runs/rl_v51/ctrl_best/adapter_model.safetensors | awk '{print $1}')
if [ "$ADAPTER_SHA" != "60b927d6bc300a0a1e5fb9b7e760391f74ce0ce2c355ba814f3cc482a18fe7d4" ]; then
  note "ERROR: Unexpected ctrl_best SHA256: $ADAPTER_SHA"
  exit 1
fi
note "Verified ctrl_best SHA256=$ADAPTER_SHA and git HEAD=$(git rev-parse --short HEAD)"

MODELS=(
  --controller_model "$M/Qwen/Qwen3.5-9B"
  --editor_model "$M/black-forest-labs/FLUX.2-klein-base-9B"
  --verifier_model "$M/Qwen/Qwen3-VL-8B-Instruct"
  --detector_model "$M/IDEA-Research/grounding-dino-base"
  --sam_model "$M/facebook/sam2.1-hiera-large"
  --device cuda:0
)
CTRL_RL=(--controller_adapter runs/rl_v51/ctrl_best --editor_adapter runs/sft/editor --editor_context_pad 0.25)
BUDGET=(--max_steps 4 --generation_budget 13 --budget_per_turn --verification_budget 500 --num_denoise_steps 28 --seed 42)
V5=(
  --turn_pass --scope_routing --user_text
  --turn_gate net2 --global_gate net2 --local_gate net2
  --global_retries 1 --turn_retries 0 --turn_priority
  --turn_localize --turn_protect
  --clean_instruction --single_stage_verify --focused_turn
  --smart_turn_pass --cascade_priority --repair_gate --monotonic_turn
  --candidate_k 3
)

for shard in 1 10; do
  out="runs/eval/exp1_v51_ctrlrl.s${shard}.json"
  jout="runs/eval/judged_v51_ctrlrl.s${shard}.jsonl"
  note "START Stage 3 eval_main shard ${shard}/12 (25 sessions)"
  "$PY" scripts/run_eval_benchmarks.py "${MODELS[@]}" --exp_id 1 --data data/final/eval_main.jsonl --exp1_modes redag_full \
    "${CTRL_RL[@]}" "${BUDGET[@]}" --shard "$shard" --num_shards 12 --output "$out" "${V5[@]}" \
    > "runs/logs/exp1_v51_ctrlrl.s${shard}.log" 2>&1
  note "DONE Stage 3 eval_main shard ${shard}/12 rc=$?"
  rm -f "$jout"
  "$PY" scripts/judge_finals.py --cases "${out%.json}.cases.jsonl" --model "$M/Qwen/Qwen3-VL-8B-Instruct" \
    --out "$jout" --device cuda:0 > "runs/logs/judge_v51_ctrlrl.s${shard}.log" 2>&1
  note "JUDGED Stage 3 eval_main shard ${shard}/12 rc=$? rows=$(wc -l < "$jout")"
done

note "ALL 917 STAGE 3 SHARDS (1, 10) COMPLETE"
