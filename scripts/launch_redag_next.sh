#!/usr/bin/env bash
set -uo pipefail

TASK="${1:-help}"
GPU="${2:-0}"
ARG3="${3:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
M=/root/autodl-tmp/models
mkdir -p runs/logs
STATUS=runs/logs/next_status.txt

note() {
  echo "$(date '+%F %T') [$(hostname | tail -c 9)] gpu${GPU} $*" >> "${STATUS}"
}

MODELS=(
  --controller_model "${M}/Qwen/Qwen3.5-9B"
  --editor_model "${M}/black-forest-labs/FLUX.2-klein-base-9B"
  --verifier_model "${M}/Qwen/Qwen3-VL-8B-Instruct"
  --detector_model "${M}/IDEA-Research/grounding-dino-base"
  --sam_model "${M}/facebook/sam2.1-hiera-large"
  --device cuda:0
)
SFT=(--controller_adapter runs/sft/controller --editor_adapter runs/sft/editor --editor_context_pad 0.25)
TURN_BUDGET=(--max_steps 3 --generation_budget 3 --budget_per_turn --verification_budget 400 --num_denoise_steps 28 --seed 42)
JUDGE="${M}/Qwen/Qwen3-VL-8B-Instruct"

eval_variant() {
  local name="$1" shard="$2" subset="$3"
  shift 3
  local extra=()
  if [ "${subset}" = "i2i" ]; then
    extra=(--only_kinds i2i)
  elif [ "${subset}" = "k10t2i" ]; then
    extra=(--only_kinds t2i --only_k 10)
  fi
  local out="runs/eval/exp1_next_${name}${subset:+_${subset}}.s${shard}.json"
  note "start ${name} ${subset:-all} shard ${shard}"
  "${PYTHON}" scripts/run_eval_benchmarks.py "${MODELS[@]}" --exp_id 1 --data data/final/eval_main.jsonl --exp1_modes redag_full \
    "${SFT[@]}" "${TURN_BUDGET[@]}" --shard "${shard}" --num_shards 12 --output "${out}" "${extra[@]}" "$@" \
    > "runs/logs/next_${name}${subset:+_${subset}}.s${shard}.log" 2>&1
  note "done ${name} ${subset:-all} shard ${shard} rc=$?"
  "${PYTHON}" scripts/judge_finals.py --cases "${out%.json}.cases.jsonl" --model "${JUDGE}" --out "runs/eval/judged_next_${name}${subset:+_${subset}}.jsonl" \
    --device cuda:0 > "runs/logs/judge_next_${name}${subset:+_${subset}}.s${shard}.log" 2>&1
  note "judged ${name} ${subset:-all} shard ${shard} rc=$?"
}

FULL=(--turn_pass --scope_routing --user_text)
MEAN=(--turn_pass --scope_routing --user_text --turn_gate mean --global_gate mean)

silent_scores() {
  local include="${1:-.}" exclude="${2:-^$}" files=() f
  for f in exp1.s0 exp1.s1 exp1.s10 exp1.s2 exp1_redag_full.s2 exp1_redag_ctxpad.s0 exp1_redag_ctxpad.s1 exp1_redag_ctxpad.s2 exp1_redag_ctxpad.s10 \
           exp1_spatial_no_gates.s0 exp1_spatial_no_gates.s1 exp1_spatial_no_gates.s2 exp1_spatial_no_gates.s10 \
           exp1_redag_rl_counterfactual.s0 exp1_redag_rl_counterfactual.s1 exp1_redag_rl_counterfactual.s2 exp1_redag_rl_counterfactual.s10 \
           stress_expanded.s0 stress_expanded_cfrl.s0 stress_main.s0 stress_main.s1 stress_main.s2; do
    if [ -f "runs/eval/${f}.cases.jsonl" ] && [[ "${f}" =~ ${include} ]] && ! [[ "${f}" =~ ${exclude} ]]; then
      files+=("runs/eval/${f}.cases.jsonl")
    fi
  done
  local usable=()
  for f in "${files[@]}"; do
    "${PYTHON}" - "$f" <<'EOF' && usable+=("$f")
import json, os, sys
lines = open(sys.argv[1]).read().splitlines()[1:]
rows = [json.loads(l) for l in lines if l.strip()]
rows = [r for r in rows if 'case' in r]
sys.exit(0 if rows and all(os.path.exists(r['case']['views'][r['case']['final']]['image']) for r in rows[:5]) else 1)
EOF
  done
  note "silent scoring on ${#usable[@]} case files"
  "${PYTHON}" scripts/score_edit_steps.py --cases "${usable[@]}" --model "${JUDGE}" --out "runs/eval/steps_next_$(hostname | tail -c 9).jsonl" \
    --unit steps --modes redag_full spatial_no_gates flat_full --device cuda:0 > runs/logs/next_silent.log 2>&1
  note "silent scoring done rc=$?"
}

headroom() {
  local vary="$1" samples="$2"
  shift 2
  note "start headroom ${vary} ${samples} $*"
  "${PYTHON}" scripts/rl_headroom.py "${MODELS[@]}" --data data/final/rl_train.jsonl --samples "${samples}" --vary "${vary}" \
    --num_denoise_steps 28 "$@" > "runs/logs/next_headroom_${vary}.log" 2>&1
  note "done headroom ${vary} rc=$?"
}

case "${TASK}" in
  full) eval_variant full "${ARG3}" "" "${FULL[@]}" ;;
  budget) eval_variant budget "${ARG3:-0}" i2i ;;
  mean) eval_variant mean "${ARG3:-0}" i2i "${MEAN[@]}" ;;
  silent) silent_scores "${ARG3:-.}" ;;
  headroom_editor) headroom editor 8 "${SFT[@]}" --limit 80 --out runs/eval/headroom_flux_sft_editor.jsonl ;;
  headroom_controller) headroom controller 4 "${SFT[@]}" --limit 80 --out runs/eval/headroom_flux_sft_controller.jsonl ;;
  pilot_2511)
    headroom editor 4 --controller_adapter runs/sft/controller --editor_kind pipeline --editor_offload --editor_context_pad 0.25 \
      --editor_model "${M}/Qwen/Qwen-Image-Edit-2511" --limit 40 --out runs/eval/headroom_qwen2511_editor.jsonl
    ;;
  chain902_gpu0)
    eval_variant full 0 "" "${FULL[@]}"
    eval_variant full 1 "" "${FULL[@]}"
    ;;
  chain902_gpu1)
    eval_variant full 2 "" "${FULL[@]}"
    eval_variant full 10 "" "${FULL[@]}"
    ;;
  chain902_gpu2)
    silent_scores . rl_counterfactual
    for s in 0 1 2 10; do eval_variant budget "${s}" i2i; done
    for s in 0 1 2 10; do eval_variant mean "${s}" i2i "${MEAN[@]}"; done
    ;;
  chain917)
    silent_scores rl_counterfactual
    headroom editor 8 "${SFT[@]}" --limit 80 --out runs/eval/headroom_flux_sft_editor.jsonl
    headroom controller 4 "${SFT[@]}" --limit 80 --out runs/eval/headroom_flux_sft_controller.jsonl
    if [ -f "${M}/Qwen/Qwen-Image-Edit-2511/model_index.json" ] && ls "${M}/Qwen/Qwen-Image-Edit-2511/transformer/"*.safetensors > /dev/null 2>&1; then
      headroom editor 4 --controller_adapter runs/sft/controller --editor_kind pipeline --editor_offload --editor_context_pad 0.25 \
        --editor_model "${M}/Qwen/Qwen-Image-Edit-2511" --limit 40 --out runs/eval/headroom_qwen2511_editor.jsonl
    else
      note "skip 2511 pilot: weights incomplete"
    fi
    ;;
  start902)
    for g in 0 1 2; do
      nohup bash scripts/launch_redag_next.sh "chain902_gpu${g}" "${g}" > "runs/logs/next_chain902_gpu${g}.log" 2>&1 &
    done
    ;;
  start917)
    nohup bash scripts/launch_redag_next.sh chain917 0 > runs/logs/next_chain917.log 2>&1 &
    ;;
  *)
    echo "tasks: full <shard> | budget <shard> | mean <shard> | silent | headroom_editor | headroom_controller | pilot_2511 | start902 | start917"
    ;;
esac
