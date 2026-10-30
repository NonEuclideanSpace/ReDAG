# Running ReDAG on 2 x A800-80GB

Every script sources `env.sh`. Each step writes `runs/logs/<job>.log`, appends its duration to `runs/logs/timings.jsonl`, and leaves `runs/logs/<job>.done`. Rerunning a phase skips finished steps. Start long phases detached so a closed browser tab does not stop them:

```bash
cd /root/autodl-tmp/ReDAG
nohup bash scripts/launch/03_sft.sh > runs/logs/phase3.out 2>&1 &
```

| phase | script | needs | what it does |
|---|---|---|---|
| 1 | `01_prepare_data.sh` | no GPU | download MagicBrush and CoReBench; build MagicBrush crops and sessions; convert CoReBench, LGBench and LongBench-T2I; choose the sessions to annotate |
| 0 | `00_check_env.sh` | GPUs | GPUs, complete weights (no `*.incomplete`), imports, `pytest`, data check |
| S | `smoke_test.sh` | GPUs | every GPU stage on a few records (~30–45 min); real per-step timings |
| 2 | `02_data_gpu.sh` | GPUs | LLM graph annotation on both GPUs → `build_sets` → canvases → RL state grounding → check |
| 3 | `03_sft.sh` | GPUs | editor SFT ∥ planner QLoRA → rejection sampling (1000 × 3) → controller SFT |
| 4 | `04_rl.sh` | GPUs | main alternating RL (checkpoints for experiment 3) + experiment 2's three reward runs |
| 5 | `05_eval.sh` | GPUs | experiments 1 / 2 / 3 (`EXP="1 3"` and similar select a subset; parts merge automatically) |

## Budget knobs (env.sh)

| variable | default | meaning |
|---|---|---|
| `STEPS_RL` / `STEPS_EVAL` | 16 / 28 | Flow-SDE steps for RL rollouts / for rejection sampling and evaluation (config default 50) |
| `MAX_STEPS`, `GEN_BUDGET`, `VER_BUDGET` | 4, 8, 400 | per-turn repair steps, renders and verifier calls per session; identical for every compared method |
| `N_RL_TRAIN`, `N_RL_DEV`, `N_EXP2` | 128, 16, 64 | RL parents, dev parents (evaluated after every RL step, section 5.3), experiment-2 parents |
| `PER_TIER`, `N_RS_STATES` | 100, 1000 | sessions per K tier (code minimum) and rejection-sampling states (eq. 19) |
| `EVAL_LAYOUT`, `RS_LAYOUT` | shard | one process per GPU with all models (~71 GB); `split` = one process across both GPUs |

## Planned GPU hours (estimate; replace with smoke-test numbers)

| block | GPU-h |
|---|---|
| graph annotation, canvases, state grounding, smoke test | ~10 |
| SFT: editor, planner QLoRA, rejection sampling, controller | ~17 |
| RL: main run + 3 experiment-2 runs | ~60 |
| experiment 1 (3 environments x 300 sessions) | ~23 |
| experiment 3 (5 stages x 300 sessions x 3 seeds) | ~115 |
| total | ~225 |

To extrapolate from the smoke test:
- **Evaluation**: seconds per session in `smoke_eval` × sessions in each experiment.
- **RL**: `step_wall_seconds` in `runs/smoke/rl/metrics.jsonl` × (training parents / 4) × 4 stages.

If a single evaluation step costs more than about 15 s, lower `GEN_BUDGET` before the long runs start.
