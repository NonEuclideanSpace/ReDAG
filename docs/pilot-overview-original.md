# Agentic Image Generation v3

Implementation of the supplied v3 plan: a multimodal Qwen3.8-27B controller, an immutable executable DAG, FLUX.2-klein-base-9B with spatial and reference conditions, a frozen Qwen3-VL-32B judge, linked SFT, hierarchical Agent RL, and joint controller/renderer RL.

The default profile uses one four-card A6000 server. GPU 0 runs the NF4 controller, GPU 1 the BF16 renderer, GPU 2 the frozen Qwen3-8B encoder and FP32 VAE, and GPU 3 the NF4 judge. Visual tools run in their own compatible environment on GPU 3, with one active tool at a time. This uses one replica of each trainable model and gradient accumulation to preserve the specified global batches. It reduces throughput relative to the plan's seven-card deployment without changing the models, conditioning, objective, or sampling distribution.

No server training has been started. Local tests exercise executable state transitions, tensor operations, loss gradients, recovery, and library integration. Full model memory, CUDA kernel behavior, sampling quality, and learning improvements require the target GPUs and authorized model weights.

## Install on the A6000 server

Use Linux, `uv`, Git, a CUDA 13 compatible NVIDIA driver, and a CUDA toolkit suitable for building the two GDN extensions. The scripts create Python 3.12 and Python 3.10 environments. The separate GroundingDINO environment needs the CUDA 12.4 toolkit. Keep the CUDA toolkit selection consistent with the environment being built. The primary environment uses the exact Transformers, PEFT, Diffusers, FLA, and causal-conv1d revisions from the plan.

```bash
bash scripts/setup_a6000.sh
bash scripts/setup_tools.sh
.venv/bin/hf auth login
.venv/bin/agentic-image prepare-models --config configs/a6000.yaml
.venv-tools/bin/python scripts/serve_tools.py --config configs/tools.yaml
```

Accept the FLUX.2-klein-base-9B model license through the official Hugging Face page before downloading it. `prepare-models` creates serialized NF4 bases once; training and inference reload those same bases. It also downloads the complete authorized Klein snapshot. Set the model cache and `run_dir` to storage with sufficient free space. Model weights, paired images, actor snapshots, and full stochastic latent traces are substantial; the plan's memory and disk estimates remain relevant.

Install Tesseract 5.5.0 with `tessdata_fast` 4.1.0 English data if using text checks. The OCR route is limited to explicitly annotated English and numbers. GroundingDINO, SAM2.1-small, DINOv2-B/14, and the fixed BERT snapshot are prepared by `setup_tools.sh`. `configs/tools.yaml` contains their paths.

The school server uses `deployment/environment.sh` for storage, caches, download routes, and executable paths. `bash scripts/run_server.sh` loads this environment before invoking the CLI. Use `bash scripts/run_server.sh tools --config configs/tools.yaml` for the visual service. The project and environments remain separate from shared model sources. `scripts/setup_server_ocr.sh` installs the fixed OCR runtime in the project storage without changing system packages.

The CUDA extension installer uses the CUDA 13 compiler inside the main environment and compiles causal-conv1d for A6000 compute capability 8.6. It creates the missing CUDA runtime linker alias only inside that environment. The fixed FLA source is installed afterwards. The compiler runs with GPUs hidden; installed extensions still require subsequent GPU execution checks.

The server has completed actual CPU inference through GroundingDINO, SAM2.1, DINOv2, and Tesseract. The result is saved in `deployment/visual-tools-cpu.json`. This does not establish GPU training readiness. NF4 export, full controller/renderer/judge runs, GPU backward passes, and training recovery remain required before deployment can be considered complete.

Server preparation is paused for user review. Original controller, judge, and renderer weights are present; the controller and judge NF4 exports are pending. Import processes and the temporary CPU visual service have been stopped. `deployment/status.json` and `deployment/data-pause.json` record the saved state. No GPUs were reserved and no training was started.

All commands accept `--config PATH` and repeated `--set dotted.key=value` overrides. For example:

```bash
.venv/bin/agentic-image hardware --config configs/a6000.yaml
.venv/bin/agentic-image train-sft --stage renderer --set run_dir=runs/main
```

## Data preparation

Public tasks and private evaluation files are separate JSONL files. The controller never receives paired target images, private masks, evaluation IDs, benchmark VQA fields, or terminal scores. `build-pairs` requests a public PLAN first, then checks compatibility against the private target. It does not rewrite a plan using target boxes.

Import the official partitions before assigning families. Include official held-out records in the family pass so a shared underlying image cannot remain in the custom training pool. OmniEdit is streamed and capped rather than downloaded in full.

Imports now retain line-flushed `.partial.jsonl` records on interruption. Add `--resume` to continue the same source, split, revision, and output path. Completed outputs are never overwritten. At the review pause, the saved training indices contain 1,038 MagicBrush records and 1,018 OmniEdit records. MagicBrush dev/test and OmniEdit dev contain all 528, 1,053, and 700 requested records respectively.

MagicBrush's transparent RGBA areas represent the edit region. The importer decodes alpha before considering grayscale masks and preserves masks at either original image resolution. Equal-aspect source/target pairs with different resolutions still require independent registration evidence before transformation into a shared canvas. The earlier imported MagicBrush records require a mask backfill using `scripts/prepare_magicbrush_masks.py`; this backfill is not complete. Neither raw imports nor registration checks constitute verified training labels.

The school server's direct Hugging Face CDN route timed out during data streaming. The proxy route returned actual data during the bounded check, so the saved environment now routes Hugging Face CDN requests through the proxy. Large-file sustained throughput after this change has not been established. Model downloads are already complete; remaining dataset imports are paused.

```bash
.venv/bin/agentic-image import-data --source magicbrush --split train --limit 8807 --output data/raw/mb_train.jsonl
.venv/bin/agentic-image import-data --source magicbrush --split dev --limit 528 --output data/raw/mb_dev.jsonl
.venv/bin/agentic-image import-data --source magicbrush --split test --limit 1053 --output data/raw/mb_test.jsonl
.venv/bin/agentic-image import-data --source omniedit --split train --limit 12000 --output data/raw/omni_train.jsonl
.venv/bin/agentic-image import-data --source omniedit --split dev --limit 700 --output data/raw/omni_dev.jsonl
.venv/bin/agentic-image merge-records --input data/raw/mb_train.jsonl data/raw/mb_dev.jsonl data/raw/mb_test.jsonl data/raw/omni_train.jsonl data/raw/omni_dev.jsonl --output data/raw/all.jsonl
.venv/bin/agentic-image split-data --input data/raw/all.jsonl --output data/families.jsonl
.venv/bin/agentic-image annotation-packet --candidates data/families.jsonl --split train --limit 200 --output data/review/train_pilot
```

Family grouping includes sessions, identical decoded images, reference sources, and near-duplicate candidates confirmed with the frozen DINO tool. `--reviewed-pairs` supplies adjudicated overrides; `--no-visual-review` quarantines unresolved perceptual candidates. The partition report records independent families and shortages. Complete the emitted annotation JSONL while reviewing the HTML image pairs. The templates start unverified: a target is not automatically a valid label, a difference mask is not automatically an edit annotation, and an instruction is not a complete caption. Review the five specified dimensions: completion, masks, identity, registration, and request coverage. Do not change thresholds to fill a quota.

Annotations include an independently checked `evaluation` with nonempty `eval_atoms`, provenance, paired semantic compatibility, registered source/target geometry, and reliable entity regions where available. `expected_value` can contain structured count, geometry, OCR, or reference checks; the exact accepted fields are implemented in `checks/evidence.py`. User-provided controls belong in `public_controls`; private target annotations do not.

The frozen teacher can prepare the bulk annotations. It extracts evaluation requirements from the original request and source, separately audits their coverage, and only then evaluates the private target. It proposes complete captions and checks entity geometry with the actual detector and segmenter. Confident rows carry machine-verification provenance; uncertain rows retain explicit review reasons. Held-out task admission depends on a valid task definition, not whether the supplied target receives a successful score.

```bash
.venv/bin/agentic-image propose-annotations --candidates data/families.jsonl --output data/review/verified.jsonl --split train --limit 12000
.venv/bin/agentic-image propose-annotations --candidates data/families.jsonl --output data/review/development.jsonl --split development --no-captions
.venv/bin/agentic-image propose-annotations --candidates data/families.jsonl --output data/review/calibration.jsonl --split calibration --no-captions
```

Verified complete captions can become the T2I portion of the linked dataset:

```bash
.venv/bin/agentic-image caption-candidates --input data/families.jsonl --annotations data/review/verified.jsonl --output data/captions.jsonl
.venv/bin/agentic-image merge-records --input data/families.jsonl data/captions.jsonl --output data/candidate_pool.jsonl
.venv/bin/agentic-image merge-records --input data/review/verified.jsonl data/captions.annotations.jsonl --output data/candidate_annotations.jsonl
.venv/bin/agentic-image appearance-candidates --input data/candidate_pool.jsonl --annotations data/candidate_annotations.jsonl --output data/appearance_pool.jsonl
.venv/bin/agentic-image build-pairs --candidates data/appearance_pool.jsonl --annotations data/candidate_annotations.jsonl --output data/aligned --limit 12000
.venv/bin/agentic-image build-tasks --candidates data/families.jsonl --annotations data/review/development.jsonl --split development --output data/development
.venv/bin/agentic-image build-tasks --candidates data/families.jsonl --annotations data/review/calibration.jsonl --split calibration --output data/calibration/tasks
.venv/bin/agentic-image build-tasks --candidates data/appearance_pool.jsonl --annotations data/candidate_annotations.jsonl --split train --limit 2000 --output data/rl
```

Caption candidates retain their parent family and need verified full-caption evaluation annotations. The appearance builder consumes `appearance_reference` annotations with `verified`, `provenance`, `source_box`, and `entity_name`, and crops only the public source. It retains all candidate rows and reports missing eligible references. `build-pairs` reserves 10% appearance coverage inside each source/region bucket and reports actual quotas and gaps; it never copies records to manufacture 8,000 independent pairs. Renderer records retain the shared PLAN, bundle identity, target, valid/edit masks, and a reconstructable condition recipe. Actor PLAN records are selected at one half of each accepted source/region bucket.

Audit 100 accepted records per source/region bucket, or every record when a bucket is smaller. The packet includes image pairs, actual masks, the proposed plan, and independent evaluation atoms. Complete the five human ratings in `reviews.jsonl`, then score them. Renderer and full actor SFT require an accepted result at `data.paired_audit`; a bucket above 5% error must be repaired and resampled.

```bash
.venv/bin/agentic-image pair-audit --pairs data/aligned --annotations data/candidate_annotations.jsonl --output data/aligned/audit
.venv/bin/agentic-image score-pair-audit --packet data/aligned/audit --reviews data/aligned/audit/reviews.jsonl
```

Point `data.public_tasks`, `data.private_tasks`, `data.actor_records`, and `data.renderer_records` in the configuration at the resulting files. Held-out task construction performs no success-based filtering. Keep the development, calibration, diagnostic, and official benchmark families out of both SFT and RL.

## Judge calibration

The production judge compares complete `Yes + im_end` and `No + im_end` sequence likelihoods. It uses the fixed second-template route and treats semantic uncertainty separately from execution failure. Structured tools supplement count, geometry, reference, preservation, and OCR checks.

Prepare calibration cases with `task_id`, `image_id`, `partition` (`selection` or `confirmation`), `labels` mapping private evaluation IDs to human booleans, and `adjudicated`. The registered calibration split contains 400 selection families and 200 confirmation families, with at least 100 labeled atoms per type. Use examples of both success and failure and adjudicated human labels.

```bash
.venv/bin/agentic-image score-calibration --tasks data/calibration/tasks/public_tasks.jsonl --cases data/calibration/cases.jsonl --output data/calibration/scored.jsonl --set data.private_tasks=data/calibration/tasks/private_tasks.jsonl
.venv/bin/agentic-image calibrate --records data/calibration/scored.jsonl --output data/calibration/thresholds.json
```

Threshold selection uses the prescribed Wilson precision bound and independent confirmation. Unsupported types remain unconfirmed. RL requires a calibrated judge; annotation and calibration are real preparation work, not supplied synthetic training labels.

## Training sequence

The commands below use `runs/main` and the data paths in `configs/a6000.yaml`. Update the data fields to the actual merged files first. Fresh stages rebuild their SFT optimizer/schedule while loading the previous policy explicitly.

1. Build public protocol demonstrations, then train renderer SFT and controller protocol SFT. These stages do not need visual repair labels.

```bash
.venv/bin/agentic-image build-protocol --tasks data/aligned/public_tasks.jsonl --output data/protocol
.venv/bin/agentic-image build-complex-plans --tasks data/complex/public_tasks.jsonl --output data/complex_plans
.venv/bin/agentic-image merge-records --input data/aligned/actor_records.jsonl data/protocol/actor_records.jsonl data/complex_plans/actor_records.jsonl --output data/sft/protocol.jsonl
.venv/bin/agentic-image train-sft --stage renderer --set data.renderer_records=data/aligned/renderer_records.jsonl
.venv/bin/agentic-image train-sft --stage protocol --set data.actor_records=data/sft/protocol.jsonl
```

Default budgets are 800 renderer updates with global batch 12 and 200 protocol updates with global batch 20. The complex-plan builder accepts up to 96 actually complex valid plans, with at most 192 proposals; the recovery builder has a separate 1,024-record cap.

2. Load both trained policies, perform the development sampler comparison, and lock the accepted sampler before collecting repairs.

```bash
.venv/bin/agentic-image sample-comparison --tasks data/development/public_tasks.jsonl --output runs/main/sampler --set models.actor_policy=runs/main/sft_protocol/policy --set models.renderer_policy=runs/main/sft_renderer/adapters/policy.pt --set data.private_tasks=data/development/private_tasks.jsonl
.venv/bin/agentic-image select-sampler --records runs/main/sampler/comparison.jsonl --candidate sde50 --reference ode50 --set models.renderer_policy=runs/main/sft_renderer/adapters/policy.pt
```

The comparison generates the specified 64 tasks × two seeds × four samplers. Fill `severe_defect` with blinded visual-quality judgments before selection. The decision uses paired intervals and preservation as well as success. `--fallbacks` generates the two specified eta-0.15 alternatives if needed. Shorter chains must be compared with an already qualified stochastic baseline and show a throughput gain. Set `renderer.steps` and `renderer.eta` to the selected values for every subsequent stage. The default remains 50 full stochastic steps, eta 0.3, CFG 4.

3. Collect real same-state repairs, controlled interventions, and rollback/finish records. Use the frozen renderer SFT policy and selected sampler.

```bash
.venv/bin/agentic-image build-repairs --tasks data/aligned/public_tasks.jsonl --output data/repairs --renderer-version renderer-sft-v1 --sampler-id sde50 --set models.actor_policy=runs/main/sft_protocol/policy --set models.renderer_policy=runs/main/sft_renderer/adapters/policy.pt --set data.private_tasks=data/aligned/private_tasks.jsonl
.venv/bin/agentic-image build-injections --tasks data/aligned/public_tasks.jsonl --specifications data/injection_specs.jsonl --output data/injections --renderer-version renderer-sft-v1 --sampler-id sde50 --set models.actor_policy=runs/main/sft_protocol/policy --set models.renderer_policy=runs/main/sft_renderer/adapters/policy.pt --set data.private_tasks=data/aligned/private_tasks.jsonl
.venv/bin/agentic-image merge-records --input data/sft/protocol.jsonl data/repairs/actor_records.jsonl data/injections/actor_records.jsonl --output data/sft/actor.jsonl
.venv/bin/agentic-image train-sft --stage actor --set models.actor_policy=runs/main/sft_protocol/policy --set data.actor_records=data/sft/actor.jsonl
```

Natural collection caps initial states, two actual candidate renders, bounded second-seed verification, and explicit retry controls. Intervention specifications identify a saved public plan action, injected typed patches, and private target evaluation IDs. Normal, bad, and online-recovered images must all be actually generated and meet their thresholds. All verification seeds compare against the same original bad state. Uncertain outcomes are not positive labels. The combined initial repair cap is 4,800 renders.

Full controller SFT defaults to 800 updates with phase counts 5/7/5/3. Each sample is normalized by its supervised action tokens; phase weights use the global batch counts. Perform two self-correction rounds with `build-refresh --round-index 0` and `--round-index 1`, at most 256 tasks and three renders per task, and 100 `train-sft --stage refresh` updates per round. Load the latest actor policy each time and use distinct `run_dir` values for the two refresh training checkpoints. The renderer stays fixed. Merge verified refresh records with the phase-balanced training pool if a refresh subset does not cover every phase; never renormalize missing phases silently.

4. Load the completed SFT policies and freeze the independent reference adapters.

```bash
.venv/bin/agentic-image freeze-reference --set models.actor_policy=runs/main/sft_actor/policy --set models.renderer_policy=runs/main/sft_renderer/adapters/policy.pt
```

If self-correction was run, substitute its final actor checkpoint. The command writes `runs/main/rl_config.yaml` with the loaded policies and independent reference paths. No adapters are merged into the base and no model is re-quantized.

5. Run Agent RL, the two independent renderer LR pilots, and joint RL from the common Agent checkpoint.

```bash
.venv/bin/agentic-image train-rl --config runs/main/rl_config.yaml --mode agent --rounds 100
.venv/bin/agentic-image train-rl --config runs/main/rl_config.yaml --mode joint --pilot --rounds 10 --initialize-from runs/main/rl_agent/checkpoints --set run_dir=runs/pilot_3e5 --set rl.renderer_learning_rate=0.00003
.venv/bin/agentic-image train-rl --config runs/main/rl_config.yaml --mode joint --pilot --rounds 10 --initialize-from runs/main/rl_agent/checkpoints --set run_dir=runs/pilot_1e4 --set rl.renderer_learning_rate=0.0001
.venv/bin/agentic-image train-rl --config runs/main/rl_config.yaml --mode joint --rounds 400 --initialize-from runs/main/rl_agent/checkpoints
```

When initializing a new run, the trainer inherits and validates the accepted sampler-selection record from the common checkpoint's run directory. Policy, fixed reference, data, and calibration paths must point to the same registered artifacts. Compare pilot outputs on the same 64 development tasks. Choose 3e-5 if the evidence does not clearly favor 1e-4. Formal joint training restarts from the common Agent checkpoint, preserving the actor's Adam history and initializing renderer Adam. Do not continue from the better pilot's final weights.

For the main B3/B4/B5 comparison, branch from the same 100-round Agent checkpoint into 400 rounds of `--mode agent`, `--mode renderer`, and `--mode joint`, each in its own run directory. Use the same seed, task pool, sampler, and budgets. Repeat the complete shared warmup and three branches for each training seed.

`--mode renderer` freezes the actor for the corresponding single-component baseline. The joint collector always collects complete P6/N2/M2 groups. Root actions are sampled once and shared across two independent suffixes; all actual renders, including abandoned images, contribute to the generator score. Zero-render branches and constant-reward groups stay in the objective. There is one optimizer update per new batch, no PPO density-ratio approximation, and no hidden SFT replay inside RL.

## Recovery and practical checks

`--resume` restores the latest committed adapter pair, optimizers, schedulers, RNG state, and data cursor. Per-action snapshots and pending records retain the original sampled tokens and processor tensors; interrupted work is resumed with the same seed and behavior version. A failed service does not become a negative-reward example. Both components roll back if an update cannot be safely published. Completed checkpoint payloads are retained for the most recent two updates and every 25th update; lightweight manifests remain.

```bash
.venv/bin/agentic-image preflight --tasks data/preflight/public_tasks.jsonl --component actor
.venv/bin/agentic-image preflight --tasks data/preflight/public_tasks.jsonl --component renderer --cycles 32
.venv/bin/agentic-image preflight --tasks data/preflight/public_tasks.jsonl --component judge
.venv/bin/agentic-image train-rl --mode joint --resume
```

Use a real source-plus-reference development task for the maximum renderer layout. The preflight runs actual sampling, full-prefix backward, or CFG transition replay as appropriate; it records memory and produces an actual decoded image. Use `--cycles 100` for the sustained resource check. The four-A6000 profile uses a 42 GiB ceiling per 48 GiB training card. The original 32 GiB 5090 deployment would need its separate 28 GiB limit.

The default renderer backend is explicitly `_native_efficient`; it does not silently discard masks or use full quadratic math attention. If the target wheel cannot execute that kernel, select `--set renderer.attention_backend=query_chunk` before a new stage and repeat the actual resource/probability checks. The fallback chunks query rows while preserving full-key softmax and gradients. `renderer.saved_tensor_offload=true` is a separate explicit training-memory option.

Local verification is intentionally focused:

```bash
PYTHONPATH=src .venv-check/bin/python -m pytest -q
```

The final local run passed 31 tests on Python 3.12 with PyTorch 2.13.0, Transformers 5.12.1, PEFT 0.20.0, and Diffusers 0.38.0. This includes small instances of the actual Qwen hybrid model and Flux2 Transformer, multimodal cached/full replay, FM and RL gradients, public/private boundaries, full hierarchical collection with interruption, and atomic optimizer recovery. No full pretrained weights or CUDA training were used. Syntax, command entrypoints, shell scripts, and the absence of code comments/docstrings were also checked.

The local check environment is for development on this Mac. Recreate the CUDA environments with the server scripts rather than copying virtual environments between machines. No file-hash verification framework is used. Content identities are used only where they implement artifact deduplication, condition caching, deterministic RNG, or same-job recovery.

## Generate and evaluate

```bash
.venv/bin/agentic-image generate --request "A red mug to the left of a blue book on a wooden table" --output runs/example.png
.venv/bin/agentic-image generate --request "Change the book text to HELLO and preserve everything else" --source input.png --output runs/edit.png
.venv/bin/agentic-image benchmark --benchmark geneval2 --input benchmarks/geneval2_data.jsonl --output runs/geneval2 --seeds 101
.venv/bin/agentic-image compare --records results/main_comparison.jsonl --output results/statistics.json
```

Inference runs one budgeted episode and delivers only the selected/finished image. GenEval2 generation receives a sanitized prompt-only projection; the private VQA file remains with the official scorer. GenEval, GenEval2, and MagicBrush wrappers retain their official scoring paths. MagicBrush provides independent-turn and true iterative-source tracks. See `python -m agentic_image.evaluation.official --help` for the separate official evaluation environment.

`benchmark --method B0` runs the raw renderer with fixed public conditions; `--method B1` uses the explicitly loaded renderer SFT adapter. Their `--renders 3` option selects among three actual renders using public checks only. B2–B5 use one controller trajectory and the loaded policy pair; the method label does not load a checkpoint automatically. `--mask-provided` explicitly enables the separate MagicBrush mask-provided input track.

Comparison rows contain `method`, `training_seed`, `task_id`, `family_id`, `eval_seed`, and the metric. Statistics pair complete task sets, resample training seeds and families, and apply Holm correction to B5 versus B2/B3/B4. Blind human-audit utilities preserve inclusion probabilities, stratified 34/33/33 sampling, adjudication, and cluster intervals.

## Code map

| Package | Responsibility |
| --- | --- |
| `core` | Typed DAG, immutable versions, artifact/frame storage, public aliases, protocol, branches, repair compilation, views and paging |
| `actor` | Fixed NF4 base, language LoRA, FP32 GDN cache, exact multimodal action snapshots, full-support sampling, frozen-head VJP |
| `render` | Frozen Klein codecs, region/reference masks and IDs, DiT LoRA, FP32 Flow-SDE, sequential CFG VJP, FM loss, optional depth/pose residuals |
| `checks` | Public decomposition, independent private terminal evaluation, structured visual evidence, preservation reward, NF4 judge calibration |
| `data` | Bounded streaming import, family isolation, paired PLAN alignment, verified repair/rollback/finish demonstrations, injections and refresh |
| `learning` | Hierarchical advantages, global loss coefficients, phase sampling, optimizers, atomic joint commits |
| `runtime` | Four-GPU component loading, collection journals, SFT/RL loops, update probes, inference and sampler comparison |
| `evaluation` | Official benchmark adapters, paired statistics, blind human audits |

The optional depth/pose extension is disabled in the main configuration. It has explicit public provenance, independent reference parameters, target-only residuals, and its own supervision requirement. It does not reinterpret appearance images as depth or pose.

The implementation imports the official model/processor code rather than vendoring or reimplementing those networks. `SOURCES.md` records the specific upstream implementations used.
