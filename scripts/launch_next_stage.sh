#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ReDAG
export PATH=/root/miniconda3/bin:$PATH

STAGE="${1:-help}"
SHARD="${2:-0}"

COMMON_MODELS=(
  --controller_model /root/autodl-tmp/models/Qwen/Qwen3.5-9B
  --editor_model /root/autodl-tmp/models/black-forest-labs/FLUX.2-klein-base-9B
  --verifier_model /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct
  --detector_model /root/autodl-tmp/models/IDEA-Research/grounding-dino-base
  --sam_model /root/autodl-tmp/models/facebook/sam2.1-hiera-large
  --planner_model /root/autodl-tmp/models/Qwen/Qwen3.8-27B
  --planner_adapter /root/autodl-tmp/ReDAG/runs/sft/planner
)

case "$STAGE" in
  t1_ctrl_rl)
    # Evaluate best ctrl_rl checkpoint (0.533 dev acceptance) + SFT editor on shard $SHARD
    python scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}"       --controller_adapter /root/autodl-tmp/ReDAG/runs/rl/exp2/subgraph_credit_ours/ctrl_rl/controller       --editor_adapter /root/autodl-tmp/ReDAG/runs/sft/editor       --exp_id 1 --data data/final/eval_main.jsonl --num_shards 12 --shard "$SHARD"       --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28       --exp1_modes redag_full --output "runs/eval/exp1_redag_ctrlrl.s${SHARD}.json"
    ;;
  t1_online_planner)
    # Evaluate Planner-in-the-loop (online DAG construction) on shard $SHARD
    python scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}"       --controller_adapter /root/autodl-tmp/ReDAG/runs/sft/controller       --editor_adapter /root/autodl-tmp/ReDAG/runs/sft/editor       --online_planner --allow-small-dataset       --exp_id 1 --data data/final/eval_main.jsonl --num_shards 12 --shard "$SHARD"       --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28       --exp1_modes redag_full --output "runs/eval/exp1_redag_online_planner.s${SHARD}.json"
    ;;
  t2_stress_expanded)
    # Run expanded 355-item stress test (recolor, material, resize, relocate) on shard $SHARD (out of 6 shards)
    python scripts/run_eval_benchmarks.py "${COMMON_MODELS[@]}"       --controller_adapter /root/autodl-tmp/ReDAG/runs/sft/controller       --editor_adapter /root/autodl-tmp/ReDAG/runs/sft/editor       --allow-small-dataset       --exp_id 1 --data data/final/eval_stress_expanded.jsonl --num_shards 6 --shard "$SHARD"       --max_steps 3 --generation_budget 6 --verification_budget 300 --num_denoise_steps 28       --exp1_modes flat_full spatial_no_gates redag_full       --output "runs/eval/stress_expanded.s${SHARD}.json"
    ;;
  t3_rl_counterfactual)
    # Train alternating RL with Counterfactual Pre-Rollback Subgraph Credit + best_edit restore + 28 denoise steps
    python scripts/run_rl_alternating.py "${COMMON_MODELS[@]}"       --controller_adapter /root/autodl-tmp/ReDAG/runs/sft/controller       --editor_adapter /root/autodl-tmp/ReDAG/runs/sft/editor       --data data/final/rl_train_exp2.jsonl       --dev_data data/final/rl_dev.jsonl       --output runs/rl/exp2/subgraph_credit_counterfactual       --reward_mode subgraph_credit_ours       --num_denoise_steps 28       --counterfactual_credit --acceptance_bonus 0.05 --restore_best_editor
    ;;
  *)
    echo "Usage: bash scripts/launch_next_stage.sh {t1_ctrl_rl|t1_online_planner|t2_stress_expanded|t3_rl_counterfactual} [shard]"
    ;;
esac
