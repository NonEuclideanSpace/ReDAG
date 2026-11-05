#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
R=/root/autodl-tmp/ReDAG
M=/root/autodl-tmp/models
cd "$R"

STATUS=runs/logs/stage2_status_901.txt
note() { echo "$(date -u '+%F %T UTC') [901-stage2] $*" | tee -a "$STATUS"; }

note "WAITING for Stage 1 PID 31293 (/tmp/run_v51_901.sh) to exit cleanly..."
while kill -0 31293 2>/dev/null; do
  sleep 15
done
if ! grep -q "ALL 901 v5.1 STAGES COMPLETE" runs/logs/v51_status_901.txt; then
  note "ERROR: PID 31293 exited without ALL 901 v5.1 STAGES COMPLETE marker!"
  exit 1
fi
sleep 3
note "Stage 1 complete confirmed. Beginning zero-idle Stage 2 handoff."

MODELS=(
  --controller_model "$M/Qwen/Qwen3.5-9B"
  --editor_model "$M/black-forest-labs/FLUX.2-klein-base-9B"
  --verifier_model "$M/Qwen/Qwen3-VL-8B-Instruct"
  --detector_model "$M/IDEA-Research/grounding-dino-base"
  --sam_model "$M/facebook/sam2.1-hiera-large"
  --device cuda:0
)
SFT=(--controller_adapter runs/sft/controller --editor_adapter runs/sft/editor --editor_context_pad 0.25)
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

note "STEP 1: Unifying git commit to 7e51b1b"
git fetch /tmp/v51_7e51b1b.bundle
BUNDLE_SHA=$(git rev-parse FETCH_HEAD)
git reset --hard "$BUNDLE_SHA"
SHORT_SHA=$(git rev-parse --short HEAD)
note "Git HEAD is now $SHORT_SHA ($BUNDLE_SHA)"
if [ "$SHORT_SHA" != "7e51b1b" ]; then
  note "ERROR: Expected 7e51b1b, got $SHORT_SHA"
  exit 1
fi

note "STEP 2: Cross-machine blind reproducibility check (i2i shard 0 K=10: 4 sessions originally run on 917)"
rm -rf /tmp/repro_check_901 && mkdir -p /tmp/repro_check_901
"$PY" scripts/run_eval_benchmarks.py "${MODELS[@]}" --exp_id 1 --data data/final/eval_main.jsonl --exp1_modes redag_full \
  "${SFT[@]}" "${BUDGET[@]}" --shard 0 --num_shards 12 --only_kinds i2i --only_k 10 \
  --output /tmp/repro_check_901/exp1_repro_i2i.s0.json "${V5[@]}" \
  > runs/logs/repro_check_i2i_s0_k10.log 2>&1
"$PY" scripts/judge_finals.py --cases /tmp/repro_check_901/exp1_repro_i2i.s0.cases.jsonl \
  --model "$M/Qwen/Qwen3-VL-8B-Instruct" --out /tmp/repro_check_901/judged_repro_i2i_s0_k10.jsonl --device cuda:0 \
  > runs/logs/judge_repro_check_i2i_s0_k10.log 2>&1

"$PY" - << 'PYREPRO' > runs/logs/repro_check_report.txt
import json
ref = {int(json.loads(l)["key"].split("|")[2]): json.loads(l) for l in open("/tmp/eval_check_100/judged_v51_i2i_main.jsonl")}
new = {int(json.loads(l)["key"].split("|")[2]): json.loads(l) for l in open("/tmp/repro_check_901/judged_repro_i2i_s0_k10.jsonl")}
print("Reproducibility comparison on indices:", sorted(new.keys()))
for idx in sorted(new.keys()):
    r0, r1 = ref[idx], new[idx]
    max_score_diff = max(abs(r0["scores"][k] - r1["scores"][k]) for k in r0["scores"])
    print(f"idx={idx}: sha1_match={r0['image_sha1']==r1['image_sha1']} ({r0['image_sha1'][:12]} vs {r1['image_sha1'][:12]}) | sat={r0['satisfied']} vs {r1['satisfied']} | max_node_score_diff={max_score_diff:.6f}")
PYREPRO
note "DONE Cross-machine reproducibility check (see runs/logs/repro_check_report.txt)"

note "STEP 3: Single-commit 7e51b1b backfill of 5 pre-patch rows in t2i shard 2"
"$PY" - << 'PYSEED'
import json
from pathlib import Path
src = Path("runs/eval/exp1_v5_t2i.s2.cases.jsonl").read_text(encoding="utf-8").splitlines()
dst = Path("runs/eval/exp1_v51_t2i.s2.cases.jsonl")
kept = [src[0]]
for line in src[1:]:
    row = json.loads(line)
    if row.get("code") == "ed1f27b":
        kept.append(line)
assert len(kept) == 13, f"Expected header + 12 ed1f27b rows, got {len(kept)}"
dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
print("Seeded exp1_v51_t2i.s2.cases.jsonl with 12 ed1f27b rows")
PYSEED

"$PY" scripts/run_eval_benchmarks.py "${MODELS[@]}" --exp_id 1 --data data/final/eval_main.jsonl --exp1_modes redag_full \
  "${SFT[@]}" "${BUDGET[@]}" --shard 2 --num_shards 12 --only_kinds t2i \
  --output runs/eval/exp1_v51_t2i.s2.json "${V5[@]}" \
  > runs/logs/exp1_v51_t2i.s2_backfill.log 2>&1
rm -f runs/eval/judged_v51_t2i_s2_clean.jsonl
"$PY" scripts/judge_finals.py --cases runs/eval/exp1_v51_t2i.s2.cases.jsonl \
  --model "$M/Qwen/Qwen3-VL-8B-Instruct" --out runs/eval/judged_v51_t2i_s2_clean.jsonl --device cuda:0 \
  > runs/logs/judge_v51_t2i.s2_clean.log 2>&1

"$PY" - << 'PYMERGE'
import json
from pathlib import Path
s2_clean = [json.loads(l) for l in open("runs/eval/judged_v51_t2i_s2_clean.jsonl")]
s0_rows = [json.loads(l) for l in open("runs/eval/judged_v51_t2i_901.jsonl") if json.loads(l).get("file") == "exp1_v51_t2i.s0.cases.jsonl"]
assert len(s2_clean) == 17 and len(s0_rows) == 19
with open("runs/eval/judged_v51_t2i_901_clean.jsonl", "w", encoding="utf-8") as f:
    for r in s2_clean + s0_rows:
        f.write(json.dumps(r) + "\n")

all_100 = []
for p in ["/tmp/eval_check_100/judged_v51_i2i_main.jsonl", "runs/eval/judged_v51_t2i_901_clean.jsonl", "/tmp/eval_check_100/judged_v51_t2i_917.jsonl"]:
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        shard = r["file"].split(".s")[1].split(".")[0]
        r["file"] = f"exp1_v51.s{shard}.cases.jsonl"
        all_100.append(r)
assert len(all_100) == 100
with open("runs/eval/judged_v51_main_100_clean.jsonl", "w", encoding="utf-8") as f:
    for r in sorted(all_100, key=lambda x: int(x["key"].split("|")[2])):
        f.write(json.dumps(r) + "\n")
print("Wrote unified 100-row judged_v51_main_100_clean.jsonl")
PYMERGE
note "DONE t2i shard 2 clean 17/17 judgement and unified 100-row main file"

note "STEP 4: Install audited Stage 2 RL scripts, run unit tests, commit, build splits, and run Controller GRPO"
cp /tmp/stage2_rl_prep/scripts/build_rl_v51_splits.py scripts/build_rl_v51_splits.py
cp /tmp/stage2_rl_prep/scripts/run_ctrl_rl_v51.py scripts/run_ctrl_rl_v51.py
cp /tmp/stage2_rl_prep/tests/test_ctrl_rl_v51.py tests/test_ctrl_rl_v51.py
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest -q > runs/logs/pytest_stage2_rl.log 2>&1
note "Pytest passed ($(tail -n 1 runs/logs/pytest_stage2_rl.log))"

git add scripts/build_rl_v51_splits.py scripts/run_ctrl_rl_v51.py tests/test_ctrl_rl_v51.py
git commit -m "Add audited inference-aligned v5.1 Controller LoRA GRPO and high-K split builder"
RL_SHA=$(git rev-parse --short HEAD)
note "Committed Stage 2 RL at $RL_SHA"

CUDA_VISIBLE_DEVICES='' "$PY" scripts/build_rl_v51_splits.py --out_dir "$R/data/final" > runs/logs/build_rl_v51_splits.log 2>&1
note "Built high-K zero-overlap RL splits in data/final"

"$PY" scripts/run_ctrl_rl_v51.py "${MODELS[@]}" \
  --controller_adapter runs/sft/controller \
  --editor_adapter runs/sft/editor \
  --data data/final/rl_train_v51.jsonl \
  --dev_data data/final/rl_dev_v51.jsonl \
  --output runs/rl_v51/ctrl_grpo \
  > runs/logs/run_ctrl_rl_v51.log 2>&1
note "DONE Stage 2 Controller LoRA GRPO (rc=$?)"
