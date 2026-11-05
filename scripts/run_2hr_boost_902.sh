#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/ReDAG
PY=/root/miniconda3/bin/python

gpu=$1
if [ "$gpu" = "0" ]; then
  echo "[GPU 0] Starting Fixed Multi-Turn Accumulating Online Planner + CF_RL + 1.5x ContextPad (25 sessions s0)..."
  $PY scripts/run_eval_benchmarks.py --exp_id 1     --data data/final/eval_main.jsonl     --output runs/eval/exp1_redag_online_planner_cfrl.s0.json     --shard 0 --num_shards 12     --exp1_modes redag_full     --controller_model /root/autodl-tmp/models/Qwen/Qwen3.5-9B     --controller_adapter runs/rl/exp2/subgraph_credit_counterfactual/edit_rl/controller     --editor_model /root/autodl-tmp/models/black-forest-labs/FLUX.2-klein-base-9B     --editor_adapter runs/rl/exp2/subgraph_credit_counterfactual/edit_rl/editor     --verifier_model /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct     --planner_model /root/autodl-tmp/models/Qwen/Qwen3.8-27B     --planner_adapter runs/sft/planner     --detector_model /root/autodl-tmp/models/IDEA-Research/grounding-dino-base     --sam_model /root/autodl-tmp/models/facebook/sam2.1-hiera-large     --num_denoise_steps 28 --max_steps 4 --verification_budget 400 --generation_budget 8 --seed 42     --editor_context_pad 0.25     --online_planner     --device cuda:0
  $PY -c '
import json
gt_map = {}
with open("runs/eval/exp1_redag_ctrlrl.s0.cases.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if "case" in r:
            c = r["case"]
            gt_map[r["key"]] = c["views"][c["final"]]["nodes"]
with open("runs/eval/exp1_redag_online_planner_cfrl.s0.cases.jsonl") as fin,      open("runs/eval/exp1_redag_online_planner_cfrl_gt.s0.cases.jsonl", "w") as fout:
    for line in fin:
        r = json.loads(line)
        if "case" in r and r["key"] in gt_map:
            for vk, vv in r["case"]["views"].items():
                vv["nodes"] = gt_map[r["key"]]
        fout.write(json.dumps(r) + "\n")
'
  $PY scripts/judge_finals.py     --cases runs/eval/exp1_redag_online_planner_cfrl_gt.s0.cases.jsonl     --model /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct     --out runs/eval/judged_cfrl_planner.jsonl     --device cuda:0
elif [ "$gpu" = "1" ]; then
  echo "[GPU 1] Starting Complex Stress Benchmark with CF_RL + 1.5x ContextPad (s0, 119 sessions)..."
  $PY scripts/run_eval_benchmarks.py --exp_id 1     --data data/final/eval_stress_expanded.jsonl     --output runs/eval/stress_expanded_cfrl.s0.json     --shard 0 --num_shards 3     --exp1_modes redag_full     --controller_model /root/autodl-tmp/models/Qwen/Qwen3.5-9B     --controller_adapter runs/rl/exp2/subgraph_credit_counterfactual/edit_rl/controller     --editor_model /root/autodl-tmp/models/black-forest-labs/FLUX.2-klein-base-9B     --editor_adapter runs/rl/exp2/subgraph_credit_counterfactual/edit_rl/editor     --verifier_model /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct     --detector_model /root/autodl-tmp/models/IDEA-Research/grounding-dino-base     --sam_model /root/autodl-tmp/models/facebook/sam2.1-hiera-large     --num_denoise_steps 28 --max_steps 4 --verification_budget 400 --generation_budget 8 --seed 42     --editor_context_pad 0.25     --allow-small-dataset     --device cuda:1
  $PY scripts/judge_finals.py     --cases runs/eval/stress_expanded_cfrl.s0.cases.jsonl     --model /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct     --out runs/eval/judged_stress_cfrl.jsonl     --device cuda:1
fi
