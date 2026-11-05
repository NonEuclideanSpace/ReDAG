#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/ReDAG
PY=/root/miniconda3/bin/python
$PY scripts/run_external_baselines.py --baseline direct_flux   --data data/final/eval_stress_expanded.jsonl --shards 0 --num_shards 3   --prefix stress --device cuda:2
$PY scripts/run_external_baselines.py --baseline mt_opsd   --data data/final/eval_stress_expanded.jsonl --shards 0 --num_shards 3   --prefix stress --device cuda:2
$PY scripts/judge_finals.py   --cases runs/eval/stress_direct_flux.s0.cases.jsonl           runs/eval/stress_mt_opsd.s0.cases.jsonl   --model /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct   --out runs/eval/judged_gpu2_boost.jsonl   --device cuda:2
