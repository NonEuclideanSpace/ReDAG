# Shared settings for every ReDAG job.  Usage:  source scripts/launch/env.sh
# Override any variable before sourcing, e.g.  N_RL_TRAIN=64 source scripts/launch/env.sh
set -o pipefail

export REDAG_ROOT=${REDAG_ROOT:-/root/autodl-tmp/ReDAG}
# machine-local settings that should survive restarts and stay out of git (NOTIFY_URL, AUTO_SHUTDOWN, knobs, ...)
# shellcheck disable=SC1091
[ -f "$REDAG_ROOT/local.env" ] && set -a && . "$REDAG_ROOT/local.env" && set +a
export MODELS=${MODELS:-/root/autodl-tmp/models}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # weights are local; never call the hub mid-run
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PY=${PY:-/root/miniconda3/bin/python}

export PLANNER=$MODELS/Qwen/Qwen3.8-27B
export CONTROLLER=$MODELS/Qwen/Qwen3.5-9B
export EDITOR=$MODELS/black-forest-labs/FLUX.2-klein-base-9B
export VERIFIER=$MODELS/Qwen/Qwen3-VL-8B-Instruct
export DETECTOR=$MODELS/IDEA-Research/grounding-dino-base
export SAM=$MODELS/facebook/sam2.1-hiera-large

export DATA=$REDAG_ROOT/data/final
export RUNS=${RUNS:-$REDAG_ROOT/runs}
export LOGS=$RUNS/logs
mkdir -p "$LOGS"

# ---- budget knobs (see scripts/launch/README.md for the GPU-hour table) ----
export STEPS_RL=${STEPS_RL:-16}          # Flow-SDE steps for RL rollouts (replay dt follows automatically)
export STEPS_EVAL=${STEPS_EVAL:-28}      # steps for rejection sampling and every evaluation
export MAX_STEPS=${MAX_STEPS:-4}         # repair steps per turn (identical for all compared methods)
export GEN_BUDGET=${GEN_BUDGET:-8}       # editor renders per evaluation session
export VER_BUDGET=${VER_BUDGET:-400}     # verifier calls per evaluation session
export ROLLOUT_STEPS=${ROLLOUT_STEPS:-3}  # terminal_broadcast look-ahead in experiment 2
export GPU0=${GPU0:-cuda:0} GPU1=${GPU1:-cuda:1}

RUNTIME="--controller_model $CONTROLLER --editor_model $EDITOR --verifier_model $VERIFIER \
--detector_model $DETECTOR --sam_model $SAM --planner_model $PLANNER"
export RUNTIME

# run_job NAME COMMAND...  -> log to runs/logs/NAME.log, skip when runs/logs/NAME.done exists.  The job lock is the
# one the scheduler (pipeline.py) uses, so the same job never runs twice at once.
run_job() {
    local name=$1; shift
    if [ -f "$LOGS/$name.done" ]; then echo "[skip] $name (delete $LOGS/$name.done to rerun)"; return 0; fi
    echo "[start] $name  $(date '+%F %T')"
    local t0=$SECONDS
    mkdir -p "$RUNS/state/jobs"
    ( flock -n 9 || exit 75   # the log is opened only once the lock is held, so a busy call cannot truncate it
      exec > "$LOGS/$name.log" 2>&1
      cd "$REDAG_ROOT" && "$@" ) 9>"$RUNS/state/jobs/$name.lock"
    local status=$?
    if [ $status -eq 75 ]; then echo "[busy] $name is already running in another process (scheduler?)"; return 75; fi
    local spent=$(( SECONDS - t0 ))
    printf '{"job": "%s", "seconds": %d, "status": %d, "ended": "%s"}\n' "$name" "$spent" "$status" "$(date '+%F %T')" >> "$LOGS/timings.jsonl"
    if [ $status -ne 0 ]; then echo "[FAIL] $name after ${spent}s, see $LOGS/$name.log"; tail -n 20 "$LOGS/$name.log"; return $status; fi
    touch "$LOGS/$name.done"; echo "[done] $name in ${spent}s"
}

# wait_all: wait for background jobs and fail if any failed
wait_all() { local rc=0; for pid in $(jobs -p); do wait "$pid" || rc=1; done; return $rc; }

require_gpus() {
    local n
    n=$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
    if [ "${n:-0}" -lt "${1:-2}" ]; then echo "Need ${1:-2} GPUs, found ${n:-0} (still in no-GPU mode?)"; return 1; fi
}
