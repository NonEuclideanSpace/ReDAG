#!/usr/bin/env bash
set -euo pipefail

# Launch script for ReDAG next-stage experiments (preserve all old artifacts).
# Usage:
#   bash scripts/launch_next_stage.sh <task> [gpu_id] [shard] [num_shards]

TASK="${1:-help}"
GPU="${2:-0}"
SHARD="${3:-0}"
NUM_SHARDS="${4:-12}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COMMON_MODELS=(
  --controller_model /root/autodl-tmp/models/Qwen/Qwen3.5-9B
  --editor_model /root/autodl-tmp/models/black-forest-labs/FLUX.2-klein-base-9B
  --verifier_model /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct
  --detector_model /root/autodl-tmp/models/IDEA-Research/grounding-dino-base
  --sam_model /root/autodl-tmp/models/facebook/sam2.1-hiera-large
  --planner_model /root/autodl-tmp/models/Qwen/Qwen3.8-27B
  --device cuda:0
  --verifier_device cuda:0
  --editor_device cuda:0
)

case "${TASK}" in
  t1_ctrl_rl)
    "${PYTHON}" scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}" \
      --exp_id 1 --data data/final/eval_main.jsonl \
      --exp1_modes redag_full \
      --controller_adapter runs/rl/exp2/subgraph_credit_ours/ctrl_rl/controller \
      --editor_adapter runs/sft/editor \
      --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
      --shard "${SHARD}" --num_shards "${NUM_SHARDS}" \
      --output "runs/eval/exp1_redag_ctrlrl.s${SHARD}.json"
    ;;

  t1_context_pad)
    "${PYTHON}" scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}" \
      --exp_id 1 --data data/final/eval_main.jsonl \
      --exp1_modes redag_full \
      --controller_adapter runs/sft/controller \
      --editor_adapter runs/sft/editor \
      --editor_context_pad 0.25 \
      --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
      --shard "${SHARD}" --num_shards "${NUM_SHARDS}" \
      --output "runs/eval/exp1_redag_ctxpad.s${SHARD}.json"
    ;;

  t1_online_planner)
    "${PYTHON}" scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}" \
      --exp_id 1 --data data/final/eval_main.jsonl \
      --exp1_modes redag_full \
      --planner_adapter runs/sft/planner \
      --controller_adapter runs/sft/controller \
      --editor_adapter runs/sft/editor \
      --online_planner \
      --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
      --shard "${SHARD}" --num_shards "${NUM_SHARDS}" \
      --output "runs/eval/exp1_redag_online_planner.s${SHARD}.json"
    ;;

  t2_stress_expanded)
    "${PYTHON}" scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}" \
      --exp_id 1 --data data/final/eval_stress_expanded.jsonl \
      --allow-small-dataset \
      --exp1_modes redag_full spatial_no_gates flat_full \
      --controller_adapter runs/sft/controller \
      --editor_adapter runs/sft/editor \
      --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
      --shard "${SHARD}" --num_shards "${NUM_SHARDS}" \
      --output "runs/eval/stress_expanded.s${SHARD}.json"
    ;;

  t3_rl_counterfactual)
    "${PYTHON}" scripts/run_rl_alternating.py "${COMMON_MODELS[@]}" \
      --train_data data/sft/controller_states.jsonl \
      --dev_data data/sft/controller_dev_states.jsonl \
      --controller_adapter runs/sft/controller \
      --editor_adapter runs/sft/editor \
      --output_dir runs/rl/exp2_v2_counterfactual \
      --reward_mode subgraph_credit_ours \
      --counterfactual_credit \
      --acceptance_bonus 0.05 \
      --restore_best_editor \
      --num_denoise_steps 28 \
      --max_dev_states 64 \
      --eval_every 8
    ;;

  t4_edival100_redag)
    "${PYTHON}" scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}" \
      --exp_id 1 --data data/final/eval_edival100.jsonl \
      --allow-small-dataset \
      --exp1_modes redag_full \
      --controller_adapter runs/sft/controller \
      --editor_adapter runs/sft/editor \
      --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
      --shard "${SHARD}" --num_shards "${NUM_SHARDS}" \
      --output "runs/eval/edival_redag_full.s${SHARD}.json"
    ;;

  t4_external)
    BASELINE_NAME="${3:-mt_opsd}"
    "${PYTHON}" scripts/run_external_baselines.py \
      --baseline "${BASELINE_NAME}" \
      --data data/final/eval_main.jsonl \
      --shards 0 1 2 10 --num_shards 12 \
      --device cuda:0
    ;;

  *)
    echo "Usage: bash scripts/launch_next_stage.sh {t1_ctrl_rl|t1_context_pad|t1_online_planner|t2_stress_expanded|t3_rl_counterfactual|t4_edival100_redag|t4_external} [gpu_id] [shard_or_baseline] [num_shards]"
    exit 1
    ;;
esac
