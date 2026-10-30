# Running ReDAG

One command runs the whole pipeline: data → editor / planner / controller SFT → rejection sampling → alternating
RL → experiments 1-3. It is resumable at any point. The Chinese hand-over manual is [HANDOFF.md](../../HANDOFF.md).

```bash
cd /root/autodl-tmp/ReDAG
bash scripts/launch/bootstrap.sh     # fresh instance: pinned packages, model weights (~120 GiB, resumable), checks
bash scripts/launch/go.sh            # start or resume in the background (safe to repeat)
bash scripts/launch/go.sh status     # what runs where, progress, ETAs, projected finish
```

## How it works

`pipeline.py` describes about 130 jobs and their dependencies. `orchestrator.py` is a stdlib scheduler:

- **Dispatch.** Whenever a GPU is free, it starts the ready job with the longest remaining critical path. The job is pinned to its GPU(s) with `CUDA_VISIBLE_DEVICES`.
- **Backfilling.** A job that needs more GPUs than are free reserves them. Smaller jobs may use those GPUs only if they finish before the reservation starts (EASY backfilling).
- **Phase overlap.** GPUs never wait for a phase boundary. For example:
  - Editor SFT starts as soon as MagicBrush is converted.
  - Graph annotation of the checklist benchmarks starts before MagicBrush is downloaded.
  - Experiment 3's zero-shot and SFT stages run before RL starts.
  - Each RL stage of experiment 3 is evaluated as soon as its checkpoint is written.
- **On-disk state.** All state lives on disk, so the scheduler can be stopped or killed at any time.
  - Every job holds a lock while it runs. A restarted scheduler *adopts* jobs that are still running instead of starting them twice.
  - Finished jobs leave `runs/logs/<job>.done`.
  - Failed attempts are retried once. Jobs that still fail are retried automatically after the code changes.
- **Resuming inside jobs.**

  | job | resumes from |
  |---|---|
  | SFT | checkpoint every 20 min and every epoch (`CKPT_MINUTES`) |
  | alternating RL | checkpoint every step (LoRA, optimizer, RNG and data order); identical to an uninterrupted run |
  | evaluation, experiment-2 diagnostics, rejection sampling, annotation | per session or state |
  | canvases, state grounding | per file |

- **Layout.**
  - The CPU data phase runs alongside GPU jobs: 1 CPU slot in no-GPU mode, 3 otherwise.
  - RL keeps all models on one GPU when every GPU has ≥ 90 GiB (`RL_LAYOUT=auto`); otherwise each RL run uses two GPUs.
  - Evaluation and rejection sampling use one GPU per process (`EVAL_LAYOUT` / `RS_LAYOUT=split` spreads them over two).
- **Safety.**
  - A GPU preflight (`00_check_env.sh`: weights, kernels, pytest, tiny-model wiring check including resume paths) gates every GPU job. It reruns when the GPUs or the package versions change; `go.sh preflight` re-checks by hand after larger code changes.
  - Settings that change results are recorded in `runs/state/config.json`. A later start with different values is refused.

## Commands (`bash scripts/launch/go.sh …`)

| command | does |
|---|---|
| *(none)* | start or resume in the background; prints status if already running |
| `status [--json]` | running jobs with progress and ETA, per-phase counts, failures, projected finish |
| `logs JOB [-f]` | a job's log |
| `stop` / `drain` | stop now (jobs resume from checkpoints later) / finish running jobs, start nothing new |
| `retry` | start and also retry jobs that failed with the current code |
| `reload` | restart only the scheduler, e.g. after `git pull`; running jobs continue and are adopted |
| `preflight` | re-run the GPU preflight by hand (~2 min on one GPU) |
| `plan [--gpus 1 2 4 8] [--from-scratch]` | projected timeline for other GPU counts |
| `reset JOB [--dependents]` | forget that jobs finished (outputs are kept) |
| `list` | all jobs with dependencies and estimates |
| `bundle export data\|results [--split 3900] [--with-adapters]`, `bundle import MANIFEST` | hand-over / backup archives with sha256 checks |
| `notify-test` | test the `NOTIFY_URL` webhook |

`02_data_gpu.sh` … `05_eval.sh` still work. Each one runs the scheduler up to the end of its phase.

## Settings

- **Where to set them.** Settings live in `env.sh`; machine-local overrides go in `$REDAG_ROOT/local.env`, which git does not track.
- **Budget knobs (identical for every compared method):**
  - `STEPS_RL` / `STEPS_EVAL`: 16 / 28 Flow-SDE steps.
  - `MAX_STEPS`, `GEN_BUDGET`, `VER_BUDGET`: 4, 8, 400 per session.
  - `N_RL_TRAIN`, `N_RL_DEV`, `N_EXP2`: 128, 16, 64.
  - `PER_TIER`: 100.
  - `N_RS_STATES`: 1000.
- **Shards (fixed once a run starts):** `ANNOTATE_SHARDS` 8, `CANVAS_SHARDS` 4, `RS_SHARDS` 4, `EXP1_SHARDS` 12, `EXP3_SHARDS` 12.
- **Machine:**
  - `CPU_SLOTS`, `FETCH_WORKERS`, `WORKERS`, `DOWNLOAD_WORKERS`.
  - `GPU_FREE_MIB`: GPUs with more memory in use by other processes are skipped.
- **Notifications and shutdown:**
  - `NOTIFY_URL`: ServerChan, PushPlus, Feishu, DingTalk, WeCom, Bark, ntfy or generic JSON.
  - `AUTO_SHUTDOWN=done|idle`, `SHUTDOWN_GRACE_MIN`.
  - `MAX_HOURS`: drain after N hours.
- **Several machines:**
  - `EXTERNAL_JOBS` on the main machine and `LOCAL_JOBS` on a helper (the same regex on both).
  - `pipeline.py sync --peer root@host:port --loop 120` exchanges outputs over SSH.

## Planned GPU hours (A800 estimates; `go.sh status` replaces them with measured rates)

| block | GPU-h |
|---|---|
| graph annotation, canvases, state grounding, preflight | ~9 |
| SFT: editor, planner QLoRA + planner evaluation, rejection sampling, controller | ~17 |
| RL: main run + 3 experiment-2 runs (one 96 GB GPU each; twice that on 80 GB pairs) | ~30 (60) |
| experiment 1 (3 environments × 300 sessions) + experiment 2 diagnostics | ~26 |
| experiment 3 (5 stages × 300 sessions × 3 seeds) | ~115 |
| total | ~197 (227 with 80 GB cards) |

Simulated wall time on 96 GB GPUs: 1 GPU 8.2 d, 2 GPUs 4.1 d, 4 GPUs 2.1 d, 8 GPUs 1.2 d. Billed GPU-hours stay within about 3 % of the 1-GPU total up to 4 GPUs.

## Checks that need no GPU

- `python -m pytest -q tests`: unit tests including scheduler, resume and data-pipeline tests (~15 s).
- `python scripts/launch/wiring_check.py --light`: tiny models built from the real configs; fits AutoDL's 2 GiB no-GPU mode.
