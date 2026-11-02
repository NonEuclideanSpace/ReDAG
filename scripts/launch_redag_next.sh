#!/usr/bin/env bash
set -uo pipefail

TASK="${1:-help}"
GPU="${2:-0}"
ARG3="${3:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
PYQ21="${PYQ21:-/root/autodl-tmp/envs/dq21/bin/python}"
MSCLI="${MSCLI:-/root/autodl-tmp/envs/dq21/bin/modelscope}"
VIEW=/root/autodl-tmp/claude_view
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
  elif [ "${subset}" = "t2i" ]; then
    extra=(--only_kinds t2i)
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
NET=(--turn_pass --scope_routing --user_text --turn_gate net --global_gate net --global_retries 1)
MEAN2=(--turn_pass --scope_routing --user_text --turn_gate mean --global_gate mean --global_retries 1)
RETRY=(--turn_pass --scope_routing --user_text --global_retries 1)
NETR=(--turn_pass --scope_routing --user_text --turn_gate net --global_gate net --global_retries 1 --turn_retries 1 --turn_priority)

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
  local tag="$1" py="$2" vary="$3" samples="$4"
  shift 4
  note "start headroom ${tag} ${vary} ${samples}"
  "${py}" scripts/rl_headroom.py "${MODELS[@]}" --data data/final/rl_train.jsonl --samples "${samples}" --vary "${vary}" \
    --num_denoise_steps 28 "$@" > "runs/logs/next_headroom_${tag}.log" 2>&1
  note "done headroom ${tag} rc=$?"
}

model_ready() {
  [ -f "$1/.complete" ] || grep -q "DONE" "$2" 2>/dev/null
}

fetch_one() {
  local repo="$1" dir="$2" log="$3"
  if [ -f "${dir}/.complete" ] || grep -q "DONE" "${log}" 2>/dev/null; then
    return 0
  fi
  mkdir -p "${dir}"
  MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache "${MSCLI}" download --model "${repo}" --local-dir "${dir}" --max-workers 8 > "${log}" 2>&1 \
    && touch "${dir}/.complete" && echo DONE >> "${log}"
}

fetch_models() {
  if pgrep -f "modelscope download" > /dev/null; then
    return 0
  fi
  mkdir -p "${VIEW}" /root/autodl-tmp/ms_cache
  nohup bash -c "$(declare -f fetch_one); MSCLI=${MSCLI}; fetch_one Qwen/Qwen-Image-2.1 ${M}/Qwen/Qwen-Image-2.1 ${VIEW}/dl_q21.log; \
    fetch_one Qwen/Qwen-Image-Edit-2511 ${M}/Qwen/Qwen-Image-Edit-2511 ${VIEW}/dl_2511.log" > /dev/null 2>&1 &
  note "model downloads started"
}

wait_ready() {
  local dir="$1" log="$2" minutes="$3" waited=0
  while ! model_ready "${dir}" "${log}"; do
    if [ "${waited}" -ge "${minutes}" ]; then
      return 1
    fi
    sleep 60
    waited=$((waited + 1))
  done
}

CTRL=(--controller_adapter runs/sft/controller --editor_context_pad 0.25)
PIPE=(--editor_kind pipeline --num_denoise_steps 40)

pilot_flux_zero() {
  headroom flux_zero "${PYTHON}" editor 4 "${CTRL[@]}" --limit 40 --out runs/eval/headroom_flux_zero_editor.jsonl
}

pilot_q21() {
  if wait_ready "${M}/Qwen/Qwen-Image-2.1" "${VIEW}/dl_q21.log" "${1:-90}"; then
    headroom qwen21 "${PYQ21}" editor 4 "${CTRL[@]}" "${PIPE[@]}" --editor_cfg 1.0 --editor_model "${M}/Qwen/Qwen-Image-2.1" \
      --limit 40 --out runs/eval/headroom_qwen21_editor.jsonl
  else
    note "skip qwen21 pilot: weights incomplete"
  fi
}

pilot_2511() {
  if wait_ready "${M}/Qwen/Qwen-Image-Edit-2511" "${VIEW}/dl_2511.log" "${1:-120}"; then
    headroom qwen2511 "${PYTHON}" editor 2 "${CTRL[@]}" "${PIPE[@]}" --editor_offload --editor_cfg 4.0 \
      --editor_model "${M}/Qwen/Qwen-Image-Edit-2511" --limit 40 --out runs/eval/headroom_qwen2511_editor.jsonl
  else
    note "skip 2511 pilot: weights incomplete"
  fi
}

bakeoff_report() {
  "${PYTHON}" scripts/bakeoff_report.py --reference flux_sft --out runs/eval/bakeoff_summary.json --runs \
    flux_sft=runs/eval/headroom_flux_sft_editor.jsonl flux_zero=runs/eval/headroom_flux_zero_editor.jsonl \
    qwen21=runs/eval/headroom_qwen21_editor.jsonl qwen2511=runs/eval/headroom_qwen2511_editor.jsonl > runs/logs/next_bakeoff.log 2>&1
  note "bakeoff report rc=$?"
}

wait_for() {
  while pgrep -f "$1" > /dev/null; do
    sleep 30
  done
}

gate_variants() {
  for s in 0 1 2 10; do eval_variant net "${s}" i2i "${NET[@]}"; done
  for s in 0 1 2 10; do eval_variant mean2 "${s}" i2i "${MEAN2[@]}"; done
}

bakeoff() {
  pilot_q21 90
  headroom flux_sft "${PYTHON}" editor 4 "${SFT[@]}" --limit 40 --out runs/eval/headroom_flux_sft_editor.jsonl
  pilot_flux_zero
  bakeoff_report
  pilot_2511 120
  bakeoff_report
}

case "${TASK}" in
  full) eval_variant full "${ARG3}" "" "${FULL[@]}" ;;
  budget) eval_variant budget "${ARG3:-0}" i2i ;;
  mean) eval_variant mean "${ARG3:-0}" i2i "${MEAN[@]}" ;;
  silent) silent_scores "${ARG3:-.}" ;;
  headroom_editor) headroom flux_sft "${PYTHON}" editor 4 "${SFT[@]}" --limit 40 --out runs/eval/headroom_flux_sft_editor.jsonl ;;
  headroom_controller) headroom flux_sft_ctrl "${PYTHON}" controller 4 "${SFT[@]}" --limit 40 --out runs/eval/headroom_flux_sft_controller.jsonl ;;
  pilot_flux_zero) pilot_flux_zero ;;
  pilot_q21) pilot_q21 0 ;;
  pilot_2511) pilot_2511 0 ;;
  bakeoff) bakeoff ;;
  bakeoff_report) bakeoff_report ;;
  fetch) fetch_models ;;
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
    fetch_models
    bakeoff
    silent_scores rl_counterfactual
    headroom flux_sft_ctrl "${PYTHON}" controller 4 "${SFT[@]}" --limit 40 --out runs/eval/headroom_flux_sft_controller.jsonl
    ;;
  chain917_6h)
    fetch_models
    for s in 0 1 2 10; do eval_variant full "${s}" i2i "${FULL[@]}"; done
    bakeoff
    silent_scores rl_counterfactual
    headroom flux_sft_ctrl "${PYTHON}" controller 4 "${SFT[@]}" --limit 40 --out runs/eval/headroom_flux_sft_controller.jsonl
    ;;
  chain917_6h_b)
    wait_for headroom_qwen21_editor
    note "done headroom qwen21 (waited)"
    headroom flux_sft "${PYTHON}" editor 4 "${SFT[@]}" --limit 40 --out runs/eval/headroom_flux_sft_editor.jsonl
    pilot_flux_zero
    bakeoff_report
    gate_variants
    pilot_2511 30
    bakeoff_report
    ;;
  start917_6h_b)
    nohup bash scripts/launch_redag_next.sh chain917_6h_b 0 > runs/logs/next_chain917_6h_b.log 2>&1 &
    ;;
  chain_t2i)
    case "${ARG3:-net}" in
      net) for s in 0 1 2 10; do eval_variant net "${s}" t2i "${NET[@]}"; done ;;
      netr) for s in 0 1 2 10; do eval_variant netr "${s}" t2i "${NETR[@]}"; done ;;
      mean2) for s in 0 1 2 10; do eval_variant mean2 "${s}" t2i "${MEAN2[@]}"; done ;;
      full) for s in 0 1 2 10; do eval_variant full "${s}" t2i "${FULL[@]}"; done ;;
    esac
    ;;
  start_t2i)
    nohup bash scripts/launch_redag_next.sh chain_t2i "${GPU}" "${ARG3:-net}" > "runs/logs/next_chain_t2i_${ARG3:-net}.log" 2>&1 &
    ;;
  chain_rl_i2i)
    for s in 0 1 2 10; do eval_variant "rl$(basename "${ARG3}" .json)" "${s}" i2i "${FULL[@]}" --strategy_policy "${ARG3}"; done
    ;;
  net) eval_variant net "${ARG3:-0}" i2i "${NET[@]}" ;;
  mean2) eval_variant mean2 "${ARG3:-0}" i2i "${MEAN2[@]}" ;;
  retry) eval_variant retry "${ARG3:-0}" i2i "${RETRY[@]}" ;;
  netr) eval_variant netr "${ARG3:-0}" i2i "${NETR[@]}" ;;
  start917_6h)
    nohup bash scripts/launch_redag_next.sh chain917_6h 0 > runs/logs/next_chain917_6h.log 2>&1 &
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
    echo "tasks: full <shard> | budget <shard> | mean <shard> | silent | headroom_editor | headroom_controller | pilot_flux_zero | pilot_q21 | pilot_2511 | bakeoff | bakeoff_report | net <shard> | mean2 <shard> | retry <shard> | netr <shard> | chain_t2i <gate> | start_t2i <gate> | chain_rl_i2i <policy> | fetch | start902 | start917 | start917_6h | start917_6h_b"
    ;;
esac
