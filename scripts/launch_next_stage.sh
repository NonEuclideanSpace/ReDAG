#!/usr/bin/env bash
set -euo pipefail


TASK="${1:-help}"
GPU="${2:-0}"
ARG3="${3:-0}"
ARG4="${4:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
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
    SHARD="${ARG3}"
    NUM_SHARDS="${ARG4:-12}"
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
    SHARD="${ARG3}"
    NUM_SHARDS="${ARG4:-12}"
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
    SHARD="${ARG3}"
    NUM_SHARDS="${ARG4:-12}"
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
    SHARD="${ARG3}"
    NUM_SHARDS="${ARG4:-3}"
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
      --data data/final/rl_train_exp2.jsonl \
      --dev_data data/final/rl_dev.jsonl \
      --controller_adapter runs/sft/controller \
      --editor_adapter runs/sft/editor \
      --output runs/rl/exp2/subgraph_credit_counterfactual \
      --reward_mode subgraph_credit_ours \
      --rollout_steps 3 \
      --num_denoise_steps 16 \
      --counterfactual_credit \
      --acceptance_bonus 0.05 \
      --restore_best_editor
    ;;

  t3_rl_counterfactual_eval)
    SHARD="${ARG3}"
    NUM_SHARDS="${ARG4:-12}"
    "${PYTHON}" scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}" \
      --exp_id 1 --data data/final/eval_main.jsonl \
      --exp1_modes redag_full \
      --controller_adapter runs/rl/exp2/subgraph_credit_counterfactual/edit_rl/controller \
      --editor_adapter runs/rl/exp2/subgraph_credit_counterfactual/edit_rl/editor \
      --editor_context_pad 0.25 \
      --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
      --shard "${SHARD}" --num_shards "${NUM_SHARDS}" \
      --output "runs/eval/exp1_redag_rl_counterfactual.s${SHARD}.json"
    ;;

  t4_edival100_redag)
    SHARD="${ARG3}"
    NUM_SHARDS="${ARG4:-1}"
    "${PYTHON}" scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}" \
      --exp_id 1 --data data/final/eval_edival100.jsonl \
      --allow-small-dataset \
      --exp1_modes redag_full \
      --controller_adapter runs/sft/controller \
      --editor_adapter runs/sft/editor \
      --editor_context_pad 0.25 \
      --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
      --shard "${SHARD}" --num_shards "${NUM_SHARDS}" \
      --output "runs/eval/edival100_redag.s${SHARD}.json"
    ;;

  t4_external)
    BASELINE_NAME="${ARG3:-mt_opsd}"
    SHARDS_CSV="${ARG4:-0,1,2,10}"
    IFS=',' read -r -a SHARD_ARR <<< "${SHARDS_CSV}"
    "${PYTHON}" scripts/run_external_baselines.py \
      --baseline "${BASELINE_NAME}" \
      --data data/final/eval_main.jsonl \
      --shards "${SHARD_ARR[@]}" --num_shards 12 \
      --device cuda:0
    ;;

  *)
    echo "Usage: bash scripts/launch_next_stage.sh {t1_ctrl_rl|t1_context_pad|t1_online_planner|t2_stress_expanded|t3_rl_counterfactual|t3_rl_counterfactual_eval|t4_edival100_redag|t4_external} [gpu_id] [shard_or_baseline] [num_shards_or_shards_csv]"
    exit 1
    ;;
esac
