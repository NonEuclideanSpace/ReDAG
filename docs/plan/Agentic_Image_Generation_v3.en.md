# Agentic Image Generation v3: Complete English Translation

**Historical status: written before the small-model GPU experiments.**

This is a complete English translation of the supplied 47-page research plan. The original plan targets a larger 27B controller, 9B renderer, and 32B judge. **That larger-model implementation has not been run or validated on a GPU.** Later GPU experiments used the adapted small-model implementation in the repository root. Their results do not validate the original larger-model configuration.

The document preserves the original design, mathematical definitions, configurations, budgets, proposed acceptance gates, source references, and historical verification claims. First-person statements and claims of prior checks below belong to the original plan; this translation does not claim that those checks were repeated during repository preparation. Proposed experiments remain proposals unless separately supported by recorded results.

The original PDF is preserved unchanged as [Agentic_Image_Generation_v3.original.pdf](Agentic_Image_Generation_v3.original.pdf). The corresponding original implementation is preserved in [variants/large-model](../../variants/large-model/). See [implementation status](../STATUS.md) and [validation evidence](../VALIDATION.md) for the distinction between historical GPU work and current CPU checks.

Source-page markers preserve traceability across all 47 pages. All embedded source hyperlinks have been retained. This is a translation, not a summary or an updated experimental protocol.

---

**Source page 01**

# Agentic Image Generation v3

## Multimodal DAGs, a 27B Controller, a 9B Generator, and Joint Learning

Complete Research Design and Implementation Specification | October 2026 · v3 Final Draft (Final Review Complete)

This document targets one server with 4 × A6000 48 GB GPUs and one server with 3 × RTX 5090 32 GB GPUs. The present deliverable is a research, design, and validation plan. This round inspected public source code and metadata and performed CPU mathematics/interface arithmetic; it did not load the research models or start training. All performance, peak-memory, and benefit acceptance tests are explicitly left for execution on the target machines.

### Contents

| Section | Source page |
|---|---:|
| Agentic Image Generation v3 | 1 |
| 0. The central idea: align intent, conditions, and evidence | 2 |
| 1. Main configuration and model selection | 3 |
| 2. System components, pinned versions, and reuse | 3 |
| 3. Complete semantics of tasks, identities, artifacts, and the DAG | 5 |
| 4. Action protocol, state machine, and error localization | 7 |
| 5. Multimodal interfaces and geometric conventions | 10 |
| 6. Trainable parameters, 27B quantized adaptation, and reference models | 12 |
| 7. Data sources, isolation, construction, and filtering | 17 |
| 8. Exact definitions of SFT on both sides | 20 |
| 9. Evaluators, tool interfaces, and reward reliability | 22 |
| 10. Return definition and edge cases | 23 |
| 11. Behavior records and joint probability | 24 |
| 12. Complete mathematics and implementation of Flow-SDE | 26 |
| 13. Hierarchical sampling, credit assignment, losses, and regularization | 28 |
| 14. Equivalent backpropagation with low memory use | 30 |
| 15. Training stages, sampling and updates, and consistent commits | 32 |
| 16. Failures, retries, zero signal, and termination | 33 |
| 17. Roles, resources, and default configurations of the two servers | 34 |
| 18. Extensions with explicit interfaces | 38 |
| 19. Implementation modules, dependency order, and consolidated acceptance tests | 39 |
| 20. Baselines, ablations, statistics, and troubleshooting | 40 |
| 21. Source-code differences, adoption scope, and specific fixes | 42 |
| 22. Pinned sources and retrieval locations | 43 |
| 23. Actual verification in this round and deliverable boundaries | 45 |

**Source page 02**

## 0. The central idea: align intent, conditions, and evidence

This system has a clear task: understand complex image requirements, propose an executable generation plan, observe the execution results, revise the relevant branches responsible for problems, and improve planning and generation together through these real interactions.

The main models are a **Qwen3.8-27B controller, a FLUX.2-klein-base-9B generator, and a frozen Qwen3-VL-32B evaluator**. The 27B and 32B models use fixed NF4 bases. The 27B model trains language LoRA; the 32B model only judges. The 9B model retains a BF16 base and trains DiT LoRA. This uses the per-GPU capacity and service division of the seven available GPUs without requiring large-scale model parallelism. [S1, S2, S23]

### 0.1 Three layers of state, one closed loop

The DAG stores three corresponding layers of state:

| Layer | Stored content | Role in the loop |
|---|---|---|
| Intent | Original requirements, entities, relationships, subtasks, and control choices | Explains what must be accomplished. The original text is fixed; the Agent's plan can be revised. |
| Conditions | Actual layout, source/target masks, reference bindings, text blocks, latents, and position IDs | Determines exactly what the generator is asked to execute this time; constructed by a deterministic compiler. |
| Evidence | Image versions, public checks, crops/detections, failures, and repair results | Supports identifying problems and deciding what to change next; every piece of evidence is bound to a real image and version. |

A repair proceeds by finding mutable nodes associated with failure evidence, proposing a patch, compiling the changed conditions, executing one image operation, checking the result, and deciding whether to retain it, revert, or deliver. Old versions continue to exist; dependencies determine what must be recomputed. The DAG therefore organizes execution and preserves the actual decision context needed for learning.

Models exchange typed artifacts with coordinates and provenance. JSON expresses actions and references; RGB, masks, latents, embeddings, and attention visibility domains travel through fixed tensor interfaces. Vision models can look again at the original and current images, and the generator can directly consume spatial and reference conditions.

### 0.2 What the three learning stages address

1. **Linked SFT on both sides.** The public PLAN for the same task is both a controller demonstration and the source of generator conditions. The controller learns to propose executable plans; the generator learns to execute the same compiled conditions. Real failure states further teach repair, rollback, and termination. Each supervised loss has its own gradients, while the data are linked through the same bundle.
2. **Agent RL.** Temporarily freeze the generator after SFT, allowing the 27B controller to improve plans, checks, branch selection, and stopping decisions from complete interaction outcomes.
3. **Joint RL.** Collect complete trajectories from the same behavior version. Controller action probabilities and generator stochastic-transition probabilities jointly explain the trajectory distribution; both sides update using the same terminal task return. Better planning changes the conditions received by the generator, while better execution changes the evidence the Agent subsequently sees.

SFT establishes a usable initial way to cooperate; RL adjusts that cooperation based on actual execution. The main RL loss contains only policy gradient and KL to a fixed SFT reference. It does not mix multiple forms of supervised replay and additional calibrators into every round. Models, objectives, and validation remain directly connected.

### 0.3 Key decisions in this version

- Model changes are verified against actual configurations, weight shapes, and runtime code. Qwen3.8-27B uses `Qwen3_5ForConditionalGeneration`; it is not a proportionally enlarged 4B architecture. The 9B model's text dimensions, projections, parameters, and attention backend are all recalculated.
- After SFT on each side, retain the policy LoRA and copy a frozen reference; RL continues updating the policy. There is no merge followed by requantization, and no newly created RL adapter with a zero-initialized B matrix.
- `FINISH(image_id)` unifies delivery. Actual images determine completion. This removes the weak dominance caused by the previous version's asymmetric ACCEPT/STOP rewards.
- Retain four designs with direct training implications: same-state repair/rollback/termination supervision, stage-balanced SFT, hierarchical credit for root plans and subsequent execution, and weighted replay over four diffusion-time segments.
- The main working set is 512², two active regions, and two image references. The DAG can express more entities and execute them in steps when necessary. The 27B model permits at most 6,144 input tokens and 1,024 action tokens. The 768² configuration is an extension with a separate resource gate.

The rest of the document gives the complete implementation contract. Section 21 consolidates differences from open-source implementations; §23 distinguishes completed verification from acceptance tests on the target machines. The focus is to make the loop work and test the effects of learning; fact-checking serves that goal.


Source links embedded on source page 02:

- [Source link 1](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- [Source link 2](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py)
- [Source link 3](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/1e6810acf5d8ebd189b4e75590877b64358e454c)

**Source page 03**

## 1. Main configuration and model selection

### 1.1 The single main design

| Component | Fixed choice | Trainable content and rationale |
|---|---|---|
| Controller | `Qwen/Qwen3.8-27B` | Frozen NF4 base plus language rank-16 LoRA; one model provides stronger visual understanding, planning, and feedback capabilities. |
| Renderer | `FLUX.2-klein-base-9B` | Frozen BF16 base plus rank-16 DiT LoRA; an undistilled flow base, with a shared interface for generation and multi-reference editing. |
| Public checker / terminal judge | `Qwen3-VL-32B-Instruct` | NF4, fully frozen; augmented by geometry, detection, and feature evidence so that weak small-model criteria do not become the principal capability bottleneck. |
| Offline teacher | Frozen 27B snapshot | Reads only the AgentView available at the target action; reuses Actor hardware. Proposals must pass compilation and real-execution filtering. |
| Klein condition encoding | Qwen3-8B plus VAE from the same 9B snapshot | Fully frozen; BF16 text model and FP32 VAE; loaded separately from the 27B controller. |
| Main training sampler | 50 steps, CFG 4, full-step Flow-SDE, η = .3 | Positive-variance transitions provide a well-defined image-policy score; 20/28 steps must pass the quality gate in §12. |
| Per-episode budget | 10 actions, 3 renders, 2 INSPECT, 2 READ_NODES | Supports initial generation, two repairs, and necessary checks/rollback. |
| Per joint round | P = 6 tasks, N = 2 root plans, M = 2 continuation branches | 24 complete trajectories; replay four time segments for every actual render. |
| Updates | One optimizer step per side for each batch of fresh trajectories | Calculate both gradients at the same behavior point; reduce only adapter gradients within each side's server. |

The user specified a 27B capacity requirement without naming an exact checkpoint. This version independently selects Qwen3.8-27B based on its public implementation, and compares the 3.5/3.6 configurations to confirm that the three share the same principal architecture. The official model card describes planning, environmental feedback, and long-horizon Agent capability as subsequent improvements; these motivate the selection. Whether this project benefits is determined by task experiments with the same base, not by treating the publisher's results as measurements of this project. [S1]

The 9B model remains a working set whose adapter can be trained on one 5090: DiT weights occupy 16.91 GiB, policy/reference and optimizer static state approximately 0.60 GiB, and frozen encoders reside separately on A2. It uses the same class of packed latents and control entry points as 4B. The upgrade primarily concerns execution capability and capacity rather than introducing a different system. [S2, S24]

### 1.2 Quantization, licensing, and scope of adoption

NF4 reduces frozen-weight storage and arithmetic precision while preserving the model's complete 27B architecture and context capability; quantized quality still requires a separate comparison. Main training, sampling, reference computation, and formal evaluation always use the same serialized NF4 base. BF16 and NF4 are not alternated as if they were the same policy.

The official 9B weights use the FLUX Non-Commercial License; downloading requires an account to accept the model terms. In this round, public metadata, the official BFL architecture, and Diffusers conversion code could be cross-checked. Access restrictions were not bypassed when controlled files returned 401. The specification completes the design on this basis. At implementation time, once downloading is authorized, perform strict checks of actual configuration, key sets, and shapes. This is a model-admission step for research use and does not require changing the research direction. [S2, S24]

Main algorithm comparisons consistently use 27B + 9B; the older 4B model is used only for an additional capacity ablation. This separates gains from model upgrades from gains due to the DAG, SFT, RL, and joint learning.

## 2. System components, pinned versions, and reuse

### 2.1 Module boundaries

Each loop is: construct visible state → Agent samples an action → type validation and DAG revision → condition compilation and necessary execution → save images and evidence → decide again. Learning consumes actual behavior records; it must not infer history backward from a final image.

| Module | Input → output | Responsibility |
|---|---|---|
| Task / CheckSpec builder | Request and user images → public task and fixed public checks | Same function in training and deployment; private evaluation criteria are stored independently. |
| TypedDag / BranchManager | Typed patch and artifact references → immutable revision | Execution dependencies, observation provenance, entity bindings, branches, and recomputation closure. |
| ArtifactStore | RGB, masks, tensors, metadata → content address | Value identity, provenance identity, coordinates, cross-machine locations, and caches. |
| AgentViewBuilder | Revision, observed evidence, budget → multimodal view within 6,144 tokens | Retain original source and current image; page through the remaining public nodes. |
| 27B Actor worker | View → raw action tokens, probabilities, action | Planning, control choices, localization, revision, rollback, and delivery. |
| ConditionCompiler | Plan, regions, bindings → ConditionBundle | Construct the text, latents, masks, and IDs actually passed into 9B. |


Source links embedded on source page 03:

- [Source link 1](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- [Source link 2](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py)
- [Source link 3](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15)

**Source page 04**

| Module | Input → output | Responsibility |
|---|---|---|
| Frozen encoder service | Text/RGB → frozen embeddings/latents; latent → RGB | Run Qwen3-8B and the VAE on A2; cache by version. |
| 9B renderer worker | Bundle, seed, policy version → FP32 trajectory and image | The same CFG velocity entry point is used for sampling and learning. |
| Observer / Reward services | Image plus public/private criteria → respective results | Weights may be shared; inputs, DTOs, caches, and response channels are separate. |
| TrajectoryStore | All actions, calls, results → complete hierarchical batch | Save failures, rollbacks, and actual renders that were abandoned. |
| JointLearner | Batch → θ/φ gradients and new versions | Hierarchical scores, fixed-reference KL, and consistent commits on both sides. |

### 2.2 Pinned versions

| Object | Fixed version |
|---|---|
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Complete FLUX.2-klein-base-9B components | `32773329fbe7e81a90ef971740e8ba4b0364ecf3` |
| Qwen3-VL-32B-Instruct | `1e6810acf5d8ebd189b4e75590877b64358e454c` |
| Transformers 5.12.1 | `ddb849abe009d1089e6c691bfc897f27211c663c`, plus the minimal FP32-cache fix in §6 |
| PEFT 0.20.0 | `a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e` |
| Diffusers | `275869dcae4ebcfee6a80253fdabc56033335020` |
| bitsandbytes 0.50.2 | `08a9956b939857ae2c4fbd56753960cc370fa7cf` |
| flash-linear-attention 0.6.0 | `413ad2c655ece57b94bc9ddc51fdbd43cf09d984` |
| causal-conv1d 1.7.0 | `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` |
| Main environment | Python 3.12, PyTorch 2.13.0+cu130, torchvision 0.28.0 |

Each service saves its wheel hashes, CUDA/driver, actual kernels, and model/adapter/quantization/processor hashes. Controller full attention initially uses native SDPA; GDN uses the pinned FLA. The 9B model explicitly selects Diffusers `_native_efficient`. Do not enable `torch.compile`, cross-action KV reuse, or replacement inference engines. Code support and execution on the target GPU are two distinct forms of evidence. Pass the consolidated acceptance tests in §19 before training. [S3, S5, S17, S25, S26]

### 2.3 Specific sources of reuse

| Open-source implementation | Actual parts adopted | What this project must add |
|---|---|---|
| Tencent UniRL | Configuration-driven Klein workers, condition DTOs, stepwise trajectories, SFT/RL organization | Qwen3_5 NF4 worker, multi-turn DAG interaction, explicit masks/IDs, FP32 chain, hierarchical advantages, and consistent publication. |
| Diffusers / BFL | 9B Transformer, Qwen3 text encoding, VAE/BN, img2img LoRA training, efficient attention | Region compilation, separate target/source masks, multi-reference roles, and complete Gaussian replay. |
| RePlan | Independent regional-text encoding and spatial visibility domains | Separate source/target frames, reference versions, correct two-branch CFG, and auditable candidates. |
| Flow-GRPO | Flow-SDE and recomputation after sampling | KL using actual transition variance, hierarchical task returns, fixed denominators, and full-trajectory scores. |
| TRL RLOO | Reference implementation for independent samples under the same prompt, leave-one-out, and masks | Custom root/suffix grouping and gradients for two models; its existing trainer is not equivalent to this objective. |
| ComfyUI | Ready queues, dependency signatures with slot names, and caching ideas | Immutable history, observation edges, and branch isolation; training replay is separate from the `inference_mode` executor. |
| UMM-Reflection | Organization of update stages and transaction state | PREPARED → COMMITTED publication of both sides' adapters/optimizers/RNG. |

Function-level sources and fixes appear in §§21-22. Connecting several trainers does not by itself constitute joint training. Sections 11-15 define both sampling distributions, scores, references, reductions, and version consistency together.


Source links embedded on source page 04:

- [Source link 1](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Source link 2](https://github.com/huggingface/peft/tree/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e)
- [Source link 3](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [Source link 4](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/attention_dispatch.py#L3187-L3219)
- [Source link 5](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/cache_utils.py#L917-L941)

**Source page 05**

## 3. Complete semantics of tasks, identities, artifacts, and the DAG

### 3.1 Four objects and separate public/private channels

| Object | Source and mutability | Agent visibility |
|---|---|---|
| UserRequest | Immutable original text, original source image, and references/layout explicitly supplied by the user | Always visible. |
| PlanRequirement | The Agent's decomposition of the original text, entities, controls, and dependencies; patchable | Visible and mutable. |
| CheckSpec | Produced by a frozen decomposer using only UserRequest; fixed within an episode | Visible after the root PLAN, as a checking basis also available during deployment. |
| EvaluationSpec | Independently verified scoring requirements; frozen within an episode | Never enters Agent input or the public DAG. |

The root PLAN reads only the original request and permitted user images. The frozen decomposer may precompute CheckSpec; public checks and source-image checking evidence are revealed after the first legal PLAN. Training, development, formal benchmarks, and actual deployment use the same function, prompt, processor, decoding settings, and cache rules. The external decomposer may omit requirements or produce incorrect ones; it must not be treated as a gold standard. Its coverage and false-positive rate are evaluated separately.

The decomposer reuses the frozen Qwen3-VL-32B NF4 on A#3, with `do_sample=False`, input ≤ 8,192 tokens, and output ≤ 1,024 tokens. Admission also checks the decomposer's own complete mandatory input. If it exceeds 8,192, use a separate larger-context configuration rather than truncating the original text. The fixed task prompt requires decomposition into checkable requirements using only the original text and user attachments; each item must cite an actual character span in the original text. Separate entity counts, attributes, spatial relations, text, and reference appearance. Do not introduce unrequested objects, backgrounds, or styles; preserve `unknown` for ambiguous items. First normalize the original text to NFC and fix its hash. Spans use Unicode code-point half-open intervals `[start,end)` in that string; also save the original wording. Schema validation must confirm exact agreement between a span and its cited text.

On parsing failure, retry once using the same public input and a fixed format-correction prompt. If it still fails, use a single public check, "Does the overall result satisfy the original request?", and set `decomposition_fallback=true`. EvaluationSpec must not fill omissions. Public requirements beyond 12 items are executed in blocks of 12 and displayed through pagination, retaining every item; resource accounting increases with the actual number of items. Private requirements in the main training tasks initially have a limit of 12. Tasks exceeding that capacity use an explicitly expanded configuration. Never silently remove requirements from a formal benchmark.

```text
PublicTask:
  task_id, split_id, family_id, source_dataset, license_record
  user_request{text, spans, source_image?, appearance_images[], user_controls?}
  canvas{height=512,width=512}, task_mode{generate|edit}
  check_spec{version, extractor_version, atoms[], fallback}
  limits{max_entities=16,max_active_regions=2,check_atoms_per_batch=12}
PublicCheckAtom:
  check_id, origin{request|source_inference}, raw_span_ids[]
  predicate, entity_refs[], expected_value, public_reference_ids[], confidence
PrivateEvaluation:
  private_task_key, eval_atoms[], original_protected_mask?
  source_baseline_observations, annotation_provenance, evaluator_version
PrivateEvalAtom:
  eval_id, raw_span, predicate, expected_value, checker_route, threshold, mandatory
```

`check_id` and `eval_id` use independent namespaces, with no mutually visible mapping. Hidden targets, benchmark `vqa_list`, correct answers, injection causes, and private protected masks remain on the scoring side. The scoring process may associate images through an internal task key; the Agent holds only PublicTask and public artifacts. If the user actually supplies a protected region, that user version is visible. Additional internal evaluation annotations cannot thereby become control inputs.

By default, automatic checks evaluate only the fixed CheckSpec, preventing the Agent from deleting checks by changing its plan. The Agent may request its own plan-consistency check using `INSPECT(kind=plan_consistency)`. Its results are marked `origin=plan`; they cannot overwrite the base CheckSpec or determine final rewards. Error localization may compare the original text, current plan, and public visual evidence. A "missing requirement" is a finding of the public checker, not an answer directly supplied by the gold standard.

An independent EvaluationSpec means independently annotated requirements, not a guarantee that visual criteria never err. Every item is still evaluated under the calibration and known/unknown rules in §9. Private scoring is calculated only after the episode ends, on the image actually delivered. The original source may be scored once in advance through the private channel to establish a regression baseline. Hidden scores for intermediate images are used only for offline SFT filtering or post-termination diagnosis; they cannot change an already executed episode.

SFT may select actions retrospectively using private scores, but target actions may reference only already public nodes, check IDs, or original-text spans. The data-generation teacher reads public state. Hidden atoms must not leak through teacher prompts, candidate rankings, logs, pagination, cache descriptions, or "error explanations". Empty-requirement and explicitly contradictory tasks are isolated and recorded during task construction; training groups must not be retrospectively selected according to generation success or failure.

**Source page 06**

### 3.2 Graph definition

$$G=(V,E_d\cup E_o).$$

$V$ contains **immutable versions** of requests, plan items, entity bindings, control artifacts, operations, image/tensor artifacts, observations, and decisions. $E_d$ represents computational inputs actually consumed; $E_o$ represents observations actually read by a decision or diagnosis. Scene relations such as `left_of`, `wearing`, and `same_identity` are node content, not execution edges themselves.

For example, if the controller has viewed the entire image before modifying a cup on the left, that decision's observation dependency must point to the whole-image version. Recording only the crop dependency would incorrectly permit reuse of the old decision after the whole image changes.

```text
NodeSpec:
  node_version_id, logical_node_id, episode_id, branch_id
  kind, schema_version, operation, typed_parameters
  inputs[(slot_name,index,artifact_value_id,producer_version_id)]
  data_parents[], observation_parents[]
  request_span_ids[], plan_requirement_ids[], entity_ids[]
  frame_ids[], version_vector, rng_key?
NodeEvent:
  event_seq, node_version_id, event_type, timestamp
  output_artifact_ids[], error_code?, evidence_ids[]
BranchManifest:
  branch_id, revision_id, parent_revision_id
  bindings{logical_node_id:node_version_id}
  selected_image_id, selected_plan_revision, active_control_set_id
```

The state machine is CREATED → READY → RUNNING → SUCCEEDED/FAILED; a cache hit records CACHE_HIT → SUCCEEDED. A retry creates a new attempt event and does not overwrite old output. Cancellation records CANCELLED. STALE is a view property meaning that a particular revision cannot reuse a particular output, not global deletion of a node.

New decisions in the execution graph read only completed versions. Operations and outputs are created in topological order. Imported plan dependencies first undergo type and SCC checks, rejecting self-loops and multi-node cycles; execution nodes are expanded only after validation. Controller episodes unfold by appending versions, keeping the entire history acyclic.

### 3.3 Value identity, provenance identity, and caching

$$v(A)=\operatorname{SHA256}(\mathrm{type,shape,dtype,frame,payload}),$$
$$p(A)=\operatorname{SHA256}(\mathrm{producer,ordered\ slots,stamps}).$$

Canonical payloads: RGB uses decoded, color-normalized uint8 HWC pixels; PNG-compressed bytes are not the basis for pixel equality. Tensors use contiguous row-major order and explicit little-endian dtypes. Text uses NFC UTF-8. Metadata uses canonical encoding with sorted fields; floating-point parameters are fixed to finite float32. Reject NaN/Inf before writing. Scaling, padding, and source dimensions in the frame definition all participate in identity.

The same value may have multiple provenances. Caches reuse by value, while audits retain the actual producer. **Input signatures must include slot names and list order; simply sorting input hashes is insufficient.** Swapping source and appearance changes the meaning of the conditions even if the set of input images is unchanged.

$$K_v=H(\mathrm{op,schema,ordered\ input\ slots,args,model/adapters,compiler,backend,dtype,RNG}).$$

Encoding caches include the tokenizer, template, extraction layers, VAE/BN, and geometry version. Checking caches include image value identity, requirement, judge weights, template, preprocessing, and deterministic seed. Render caches also include complete conditions, the sigma schedule, CFG, noise seed, and behavior version.

Online RL shares only frozen, deterministic preprocessing caches across trajectories. Every new stochastic render has a new job identity and RNG stream; generation from a sibling trajectory cannot replace sampling. Retrying the same job may idempotently retrieve its completed result. Training replay must perform a fresh forward pass and build a graph; it must not read inference activation caches.

### 3.4 Revision and recomputation

For the modified node set $U$ and the output set $O$ requested by a new revision:

$$\mathcal C(U,O)=\left(U\cup\operatorname{Desc}_{E_d\cup E_o}(U)\right)\cap\operatorname{Anc}^{*}_{E_d\cup E_o}(O).$$

Execute in order: validate `expected_old_version` → create new nodes → update bindings → check graph and coordinates → calculate closure → query caches → execute topologically → check every public CheckSpec requirement on the new image. Old decisions whose observation basis has changed remain only as history; they are not automatically executed as new actions.

**Source page 07**

Sibling branches share immutable artifacts, but not `selected_image`, candidate queues, or budget objects. SELECT restores both the historical image and its associated plan/control revision; the next edit explicitly uses the selected image as its source. Failure in a new branch does not lose an old branch.

DAG caching saves operations that actually hit the cache, such as encoding, geometry processing, and checks. Under the main configuration, each repair still runs the complete DiT; spatial attention also allows information to propagate across layers. Record actual calls and FLOPs. Do not equate "local node repair" with strictly unchanged pixels or local DiT computation.

### 3.5 Entity identity binding

Entity IDs are stable within a task, for example `entity:0003`; changing color, position, or appearance binding does not renumber them. Each reference binding explicitly specifies `entity_id`, `reference_image_id`, `crop_mask_id`, `role{appearance|style}`, and `source_frame`. Anonymous detection boxes are matched to entities before entering observations.

Matching uses constrained Hungarian assignment. Class-incompatible pairs have infinite cost; other pairs use

$$c_{ij}=0.5(1-\operatorname{IoU}_{ij})+0.5(1-\cos(f_i,f_j))/2.$$

Here $f$ is a frozen DINOv2 crop feature. Without a historical position, use only appearance and renormalize its weight. Add dummy nodes with cost 0.7. Mark `identity_unknown` for unmatched entities or when the gap between first and second choices is below 0.1. Calibrate these thresholds on the development set and then freeze them. A move operation makes position changes legitimate, so matching may omit the IoU term, but this must be triggered by an explicit movement instruction and recorded. This matching is diagnostic evidence; final identity requirements are still verified using independent references, semantic criteria, and human evaluation.

## 4. Action protocol, state machine, and error localization

### 4.1 Unified actions

Sample one JSON object each time; it may contain a brief `reason` of at most 40 tokens. These are also actually sampled tokens, whose probability is not removed in RL. The protocol version is fixed. Arbitrary Python, arbitrary URLs, and unregistered tools are prohibited.

| Action | Required inputs | Execution semantics |
|---|---|---|
| PLAN | Original-text spans, entities, subtasks, dependencies, global text, initial control bindings | Create plan and control revisions without generating an image. |
| RENDER | `control_set_id,expected_revision` | Compile and generate once, check public CheckSpec, and select the new image. |
| DRAFT_REPAIR | `source_image_id,evidence_ids,suspect_node_ids,patch_ops,instruction` | Construct and validate 1-3 distinct candidates without generating an image. |
| APPLY | `candidate_id,expected_revision` | Commit the candidate revision, execute one render, check, and select the new image. |
| INSPECT | `image_id,kind,entity_id/region` | Crop, detect, segment, recheck public requirements, or check plan consistency; consumes the visual-inspection budget. |
| READ_NODES | `node_ids[]` or public `page_cursor` | Read only the public DAG/CheckSpec projection; ≤ 16 records and ≤ 1,024 tokens per read, at most two reads per episode; still consumes one action. |
| SELECT | Historical `image_id` | Atomically switch image and associated manifest; editing may continue afterward. |
| FINISH | Optional `image_id,reason` | Omission delivers the selected image; a string delivers that legal historical image; explicit null terminates without an image. |

FINISH must distinguish an absent field from explicit `null`: the parser preserves field presence rather than converting both to the same `None` default. The root state still requires a plan first; giving up without an image is expressed by `FINISH(null)` after a legal PLAN.

PLAN is the only legal initial action. Grammar does not forcibly guarantee success: other root actions or illegal JSON are recorded as one policy-invalid action; after the fork, each branch can recover with PLAN. Sample the root action only once, counting it toward the 10-step limit. Copy its deterministic effects, revision, budget, and invalid-action count into each suffix. The first image generation occurs after the fork. Editing tasks may FINISH immediately after a legal PLAN, delivering the existing source image.

PLAN is legal only when no valid plan exists; revise an existing plan through typed patches. Actions may reference only IDs currently or previously disclosed and still legal within the current branch, except for read-only artifacts from a shared prefix. Access across sibling branches is prohibited. The view supplies an ID directory and pagination cursors for omitted nodes, which READ_NODES uses; the model is not expected to guess unseen IDs. The terminator does not automatically select the "best image" using hidden reward.

**Source page 08**

A pagination cursor is bound to `episode/branch/revision/collection_kind/offset/projection_hash`; CheckSpec atoms also belong to the pageable public projection. An expired cursor returns `STALE_CURSOR` and a currently legal cursor. It consumes the action/read but does not count as policy invalid; it does not silently switch to another revision. If one record exceeds the page budget, return a summary and public artifact reference. Subsequent field-level reads still consume reads. If the budget is insufficient to read everything, FINISH may acknowledge incompletion; hidden scoring cannot fill the gap.

The original source image of an editing task may become a candidate after its initial check. Delivering a correct source image with zero renders is legal; pure generation fails when there is no image to deliver. This trains correct stopping and rollback without artificially requiring image generation in every episode.

Public source-image checks may be computed ahead of time, but the root view provides only the original request and image. CheckSpec and its checking evidence are revealed only after successful PLAN execution. A recovery PLAN following an invalid root follows the same rule. Private source-image scores are never disclosed.

### 4.2 Typed patches and candidate compilation

```text
PatchOperation:
  target_logical_node_id, expected_old_version_id
  op{add|replace|retire}, field_path, typed_value_or_artifact_id
RepairCandidate:
  candidate_id, source_image_id, parent_revision
  patch_operations[], scope{local|expanded|global|custom|retry}
  changed_nodes[], recompute_nodes[], compiled_condition_hash
  required_render_calls=1, validation_result
```

Allowed changes include plan interpretation, counts/attributes, dependencies, layout, regional text, reference bindings, and source-image bindings. Changing the original request, fixed public CheckSpec, independent scoring criteria, original protected region, historical image bytes, evaluators, or budgets is prohibited. When the old version does not match, return `STALE_PATCH`; do not guess the controller's new target for it.

`add` requires the target not to exist and `expected_old_version_id=null`; `replace/retire` must match the current version. The source for an ordinary edit is late-bound at compilation to the current selected image. After restoring a historical manifest, do not reuse the old source that originally produced that historical image. `DRAFT_REPAIR.source_image_id` must equal selected; SELECT first to edit another historical image. The candidate saves this image and manifest hash, and APPLY verifies that neither has changed. There is an explicit retry exception: copy the selected image producer's complete historical bundle and change only the seed, recording the actual source. This resamples the original operation rather than pretending to continue editing the current image.

Default candidate-generation rules:

1. **local:** apply the target-node patch and change only its corresponding region controls; leave other entities and requirements unchanged.
2. **expanded:** extend local to include controls sharing an entity or having IoU > 0.2 with the target region; dilate the target effect mask by two latent patches, or 32 pixels. Preserve neighboring regions' original text rather than rewriting every object with one common instruction.
3. **global:** compile the revised complete target description into global conditions and remove local-text partitions, while retaining all original requirements and explicit reference bindings.

Calculate source and target effect regions independently; an object move cannot overwrite the source mask with the target mask. If the target region is missing, first construct it from reliable detection evidence; if localization remains impossible, generate only global/custom candidates. Plan-level errors may directly repair ancestor nodes without first forcing a local image edit. Retry explicitly keeps conditions unchanged and resamples; account for it separately from condition revision.

Deduplicate candidates by complete compiled conditions; return the actual number when fewer than three remain. Candidate descriptions contain actual changes, expected recomputation operations, and cost, not post-execution images or rewards. The controller selects; the compiler does not select the best candidate on its behalf.

If automatic expansion exceeds two active regions, return that candidate's `CAPACITY_EXCEEDED` reason and retain the other legal candidates, without merging or dropping entities. The controller may choose global or explicitly submit custom patches in steps; user requirements remain complete. Automatically generated candidate-validation failures do not add policy-invalid counts; count them only if the original action itself violates the contract.

### 4.3 Error-node localization

Contract errors return the field's producer node. Requirements missing according to the public checker's reading of the original text point to plan ancestors; incorrect references point to bindings; reliable position evidence points to layouts. When valid conditions fail in execution, retain render and condition nodes as multiple candidates. For a failed observation $o$, first calculate

$$\mathcal B(o)=\operatorname{Anc}_{E_d\cup E_o}(o)\cap V_{\mathrm{mutable}},$$

then narrow candidates using original-text spans, entities, frames, and evidence. Graph reachability supplies a candidate range; it does not automatically establish visual causation. When a natural failure lacks a reliable unique cause, return a candidate set. SFT must not fabricate unique root-cause labels.

**Source page 09**

### 4.4 Visible state and deterministic execution

The view contains the original request, current plan/bindings, relevant nodes and ancestors, the three most recent errors, public CheckSpec and evidence, candidate summaries, and budgets. It contains at most four images in this order: original source (mandatory for editing), current image, active crop, appearance reference. Artifacts with identical pixels are encoded once while retaining multiple role labels. Generation tasks have no source slot. Qwen3.8 processes at most 256 visual tokens per image: 512 × 512 gives 256 tokens under its patch-16, merge-2 structure; 448 × 448 gives 196. Each merged visual token has width 5,120. This controller has no DeepStack injection. [S1, S3]

If total input exceeds 6,144, remove the oldest errors, redundant summaries, and irrelevant nodes in that order; the optional crop may be removed last. Never truncate the original request, original source, current image, required fields of the current action, or complete image blocks. Task admission uses the real tokenizer to calculate the worst-case view containing the original text, the maximum mandatory images, a 1,024-token plan, and mandatory fields. Tasks over the limit enter a separate larger-context configuration before sampling. If `VIEW_OVERFLOW` still occurs online because of an implementation error, pause and retain the complete group; after repair, resume from the same behavior version and RNG. Do not change context halfway through, drop the branch, or fabricate a policy failure. Larger context is enabled only for a new task configuration/stage and must pass memory acceptance again. Save processor outputs and image hashes; do not reconstruct an old view from a new DAG afterward.

Set processor `max_pixels` to $256(16\cdot2)^2=262144$. Actual `pixel_values` for static images is `[P,1536]` float32, where $P=\sum\operatorname{prod}(\mathrm{image\_grid\_thw})$ and $1536=3\times2\times16\times16$. A 512² image has grid `[1,32,32]`, yielding 256 visual tokens after merging. This is not the VAE's BCHW tensor. Check counts against the processor's actual grid and freeze resize/normalization settings. [S3]

In the following pseudocode, the root is turn 0 and an invalid root increments the invalid count to 1. `run_suffix` executes independently for each branch. Every sampled action consumes one turn, including an action that returns an error.

```python
root = sample_validate_and_execute_root_PLAN()
for branch in fork_independent_suffixes(root.state, M=2):
    run_suffix(branch, first_turn=1, last_turn=9)

for turn in range(1, 10):
    obs = build_view(revision, evidence, budget)
    raw = policy.sample(obs, max_new_tokens=1024)
    record_raw_tokens_and_behavior(raw, obs)
    action, error = validate(raw, revision, budget)
    if error:
        record_error(error)
        if error.is_policy_invalid:
            invalid_count += 1
        if invalid_count >= 2:
            return terminal_protocol_failure(-1)
        continue
    if action.kind == "SELECT":
        revision = restore_image_and_manifest(action.image_id)
    elif action.kind in {"RENDER", "APPLY"}:
        revision = execute_one_render_and_measure(action, revision)
    elif action.kind in {"FINISH"}:
        return finalize(action, revision)
    else:
        revision = execute_nonrender_action(action, revision)
return finalize_budget_exhausted(revision.selected_image_id)
```

Each sampled action consumes one step. Render usage increases only when a job is actually accepted and launched. Automatic public checks are a fixed part of rendering; additional INSPECT and READ_NODES are counted separately. Once a render/inspect/read budget is exhausted, reject the corresponding action and consume a step without an additional penalty. SELECT and FINISH remain available. Exhausting 10 steps returns the selected image: calculate normal reward if an image exists, or -1 otherwise. The finite budget guarantees termination.

Illegal JSON, unknown IDs, out-of-bounds/degenerate geometry, and illegal states count as policy-invalid actions. The first is recoverable; the second terminates with protocol failure. Text beyond encoding capacity explicitly returns `TEXT_TOO_LONG`, also counts as invalid, and must never be silently truncated. Service exceptions, OOM, and version mismatches are not disguised as illegal actions; see §16.

Each Actor worker executes one complete action request at a time. Internal RoPE state and the within-action cache belong to that request; tokens from multiple branches must not be interleaved. Every action prefills afresh from its public view; caching accelerates only that action's autoregression. The specific 27B template, EOS, FP32 recurrent state, and reference switching are described in §§6 and 11.


Source links embedded on source page 09:

- [Source link 1](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- [Source link 2](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

**Source page 10**

## 5. Multimodal interfaces and geometric conventions

### 5.1 Artifacts and coordinates

| Artifact | Shape / dtype | Semantics |
|---|---|---|
| Persistent RGB image | `[H,W,3] uint8` | sRGB, 0-255; apply EXIF rotation first and composite alpha onto RGB(128,128,128). |
| Model image | `[B,3,H,W] float32` | 0-1; convert to -1-1 before the VAE. |
| Pixel mask | `[H,W] float32` | 0-1, with explicit image, frame, and purpose; do not conflate soft and binary masks. |
| bbox | `[4] float32` | Normalized half-open xyxy; finite and satisfying $0\le x_0<x_1\le1$, $0\le y_0<y_1\le1$. |
| Geometric transform | `[3,3] float64` | Homogeneous transform from the original image to canvas and its inverse. |
| Packed latent | `[B,L,128]` | FP32 state and BF16 reference cache; slot specifies role. |
| Text embedding | `[B,Ltext,12288] bfloat16` | Fixed encoder, template, layers, and padding protocol. |

Aspect-preserving letterboxing: first set $s=\min(512/W_0,512/H_0)$, then $W'=\operatorname{round}(sW_0)$ and $H'=\operatorname{round}(sH_0)$. Record the actual $s_x=W'/W_0$, $s_y=H'/H_0$, and centered integer padding $(p_x,p_y)$. The transform is

$$A=\begin{bmatrix}s_x&0&p_x\\0&s_y&p_y\\0&0&1\end{bmatrix}.$$

RGB uses fixed Lanczos resampling; binary masks use nearest; soft masks use area/bilinear, with the algorithm recorded. Source images and paired targets must share a registration transform. If original sizes or crops differ, verify registration first; do not letterbox them independently and then directly calculate preservation loss. Unregisterable pairs are excluded from pixel-protection supervision. Appearance references have independent transforms; no correspondence with output pixels is assumed.

All padding is RGB(128,128,128). Canvas geometry places integer coordinates on image boundaries and pixel centers at $(x+0.5,y+0.5)$. If external keypoints use 0 for the first pixel center, add 0.5 before applying the transform. Bboxes retain half-open boundaries; mask transforms sample pixel centers. Record the source coordinate convention in the frame.

RegionControl explicitly specifies `target_mask`, `source_mask`, `entity_ids`, `region_text`, and `appearance_bindings`. A newly added object may have no source mask; moved objects occupy different positions on the two sides; a deleted object's target edit region covers its original location. A null pointer and an all-zero mask have different meanings: the former means that this side is not applicable; the latter is an error when a region is declared.

### 5.2 VAE and latents

For a 512 × 512 image:

$$z_v=\operatorname{mode}\,q_{\mathrm{VAE}}(z\mid2I-1)\in\mathbb R^{B\times32\times64\times64},$$
$$z_p=\operatorname{patchify}_{2\times2}(z_v)\in\mathbb R^{B\times128\times32\times32},$$
$$z=(z_p-\mu_{\mathrm{BN}})/\sqrt{v_{\mathrm{BN}}+\epsilon_{\mathrm{BN}}},$$
$$Z=\operatorname{pack}(z)\in\mathbb R^{B\times1024\times128}.$$

The patchify channel order is $c'=4c+2\delta_h+\delta_w$; pack traverses spatial positions in row-major order. BN mean, variance, and epsilon come from the same checkpoint and are all frozen. VAE encoding and decoding use FP32 and the posterior mode. Target, source, and appearance share one encoding core; their wrappers return either a canonical target or a packed reference with a role. Decoding strictly performs unpack, inverse whitening, unpatchify, and VAE decode. [S4, S6]

Missing BN statistics, incorrect channels, or shape mismatches block startup of that model configuration. Validation includes exact inverses for patchify/pack, numerical comparison of encoding with the official single-image path, and unchanged BN buffers before and after training.


Source links embedded on source page 10:

- [Source link 1](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
- [Source link 2](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555)

**Source page 11**

### 5.3 Text, position IDs, and bundles

Use one global text block and at most two regional blocks. Each block independently uses Klein's accompanying Qwen3 template and a fixed length of 512. The Qwen3-8B text encoder in the same 9B snapshot concatenates hidden states 9, 18, and 27 into 12,288 dimensions. Preserve the official padding embeddings rather than zeroing them. Check length after templating; do not rely on silent tokenizer truncation. [S4]

Let the number of regions be $n_r$, and the total reference count be $n_c\in\{0,1,2\}$, with source also counted as a reference segment:

$$L_t=512(1+n_r),\qquad L=1024,\qquad S=L_t+L(1+n_c)\le4608.$$

Joint-attention order is `[global text, regional text | target noise | source?, appearance?]`. Number image segments in actual input-list order: target is 0 and subsequent segment $j$ has $j\ge1$:

$$\operatorname{imgid}(j,r,c)=(10j,r,c,0),\qquad\operatorname{txtid}(i,\ell)=(0,0,0,\ell).$$

Each text block restarts $\ell$ from 0; this is part of the regional-compilation design. Reference-image time coordinates follow the official list order. Without source, an appearance image is the first reference, with `time=10`; source plus appearance use 10 and 20 respectively. Store roles separately; do not infer them from time.

```text
ConditionBundle:
  bundle_id, canvas_hw, region_count, num_target_tokens=1024
  positive_embeds[B,Lt,12288], negative_embeds[B,Lt,12288]?
  text_ids[Lt,4], image_ids[(1+nc)*1024,4]
  reference_tokens[B,nc*1024,128]
  token_span_table, source_membership, target_membership, appearance_membership
  positive_mask[B,1,S,S], negative_mask[B,1,S,S]
  cfg_scale, sigma_table_id, attn_impl, layout_signature
  artifact_bindings, model_versions, compiler_version
```

Save IDs as int64 and convert as required at the model entry point. Reference/text forward passes use BF16. Masks are bool, with True meaning access is permitted. Different tasks can share a batch only if their layout signatures match: grids, IDs, segment lengths, and ordering must be identical, while embeddings and masks may differ per sample. The first version uses microbatch = 1 and does not depend on batching across layouts.

### 5.4 Regional attention

Rasterize each region separately on source and target grids:

$$\rho_{ip}=\frac1{256}\sum_{(x,y)\in P_p}m_i(x,y),\qquad b_{ip}=\mathbf1[\rho_{ip}\ge0.25].$$

If a positive-area region has no patch meeting the threshold, retain the patch with maximum coverage, breaking ties by smallest index, and record the fallback. Reject genuinely empty regions. Overlapping regions use multilabel OR; later writes must not overwrite earlier ones.

Let $G,T_i,Y,C,A_j$ denote global text, regional text, target, source, and appearance respectively. $Y_i,C_i$ are determined by their respective masks; $B_j$ denotes regions bound to the reference.

An entity's `crop_mask_id` within a reference undergoes that reference's own transform and pooling, producing token subset $A_{j,i}\subseteq A_j$. Regional-reference connections in the table below connect only these subsets. A reference token connects to a corresponding region only when it belongs to that entity subset. `reference_scope=global` connects the entire reference segment. Without active regions, ordinary global references also use this explicit scope so references do not become completely disconnected. If global repair removes regions, it must also record the reference-scope conversion.

| Query | Readable keys |
|---|---|
| $G$ | $G\cup Y\cup C$, plus references whose scope is global. |
| $T_i$ | $T_i\cup Y_i\cup C_i\cup\bigcup_{j:i\in B_j}A_{j,i}$. |
| $y_p\in Y$ | $G\cup Y\cup C$, all global references, and regional text covering the target position plus its references. |
| $c_p\in C$ | $G\cup Y\cup C$, all global references, and regional text covering the source position plus its references. |
| $a_p\in A_j$ | $A_j\cup G$, plus $T_i,Y_i,C_i$ for entities to which this token belongs; global references additionally access all visual segments. |

Unlisted edges are False, and every row must be nonempty. This assertion enforces interface semantics; it does not claim that empty rows produce NaN in every PyTorch backend.


Source links embedded on source page 11:

- [Source link 1](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)

**Source page 12**

Masks are not explicitly copied across heads. The largest single-sample bool matrix is 20.25 MiB.

The native Flux2 processor consumes `attention_mask`. Check every attention processor's argument signature at startup. This wrapper raises an error for unknown kwargs because the library may merely warn and discard them. Passing RePlan's `flex_block_mask` to a native processor does not establish that it has taken effect. [S2, S7]

### 5.5 The single velocity entry point and CFG

Negative text consists of empty-string blocks from the same encoder, with the same lengths, IDs, layout, source/references, and masks as the positive branch. It is the "text-unconditional branch given spatial and image conditions".

$$v_\phi^{\mathrm{cfg}}=wv_\phi^+ +(1-w)v_\phi^-,\qquad w=4.$$

Reuse `predict_velocity(bundle,z,sigma,adapter)` for sampling, reference, and replay: concatenate noisy target and reference tokens, call the Transformer, and take only the first 1,024 output tokens. Pass $\sigma$ at the model entry point; the model internally multiplies it by 1,000. Use `guidance=None` and assert the checkpoint has `guidance_embeds=false`. Differentiate both positive and negative branches with respect to current LoRA; the negative-branch coefficient is -3. If CFG > 1 and negative text, mask, or matching IDs are missing, raise an error immediately rather than silently reverting to one branch. See §14 for the memory implementation. [S2, S4]

SFT uses one branch with CFG = 1; the mixed CFG velocity is not treated as an ordinary flow target. In 10% of SFT samples, replace the entire text group with empty blocks while retaining layout and references. SFT's continuous time sampling differs from the rollout's discrete sigma schedule; this is intentional.

### 5.6 Invariants after condition compilation

The complete conditions of an execution are the bundle's tensors and their provenance, not a newly generated description. Changing layout changes target membership and the attention mask; changing reference bindings changes role segments and their visibility domains; replacing the original image changes source latents and source frame. All these changes enter the bundle hash. The DAG connects user intent, executed controls, and image evidence, allowing the controller to revise specific nodes. Regional masks restrict information visibility; **they do not guarantee strictly unchanged pixels outside the region**. Preservation quality is learned jointly through source conditions, supervised loss, and terminal protection metrics. Tasks explicitly requiring pixel-level locking use a separate configuration with explicit compositing operators and are not reported together with this default generator's outputs.

## 6. Trainable parameters, 27B quantized adaptation, and reference models

Let $\theta$ denote trainable controller LoRA, $\phi$ trainable generator LoRA, and $\omega$ frozen criteria. All bases, vision towers, condition encoders, VAEs, and criteria remain unchanged. Each side's policy has a fixed SFT reference. The following specifies complete module, precision, state, and gradient boundaries.

### 6.1 Architectural facts that cannot be obtained by simply multiplying 4B

Qwen3.8-27B has 64 layers arranged as `[linear,linear,linear,full] × 16`, hidden width 5,120, FFN 17,408, and vocabulary 248,320: 48 GatedDeltaNet blocks and 16 full-attention blocks. Full attention has 24 Q heads, 4 KV heads, and explicit `head_dim=256`; head dimension must not be derived from 5120/24. Linear blocks have 16 QK heads and 48 V heads, each 128-dimensional.

| Language module | Input → output | Count per model | Quantization | LoRA |
|---|---|---:|---|---|
| Full `q_proj` | 5120 → 12288 | 16 | NF4 | Yes |
| Full `k_proj,v_proj` | 5120 → 1024 | 16 each | NF4 | Yes |
| Full `o_proj` | 6144 → 5120 | 16 | NF4 | Yes |
| Linear `in_proj_qkv` | 5120 → 10240 | 48 | NF4 | Yes |
| Linear `in_proj_z` | 5120 → 6144 | 48 | NF4 | Yes |
| Linear `in_proj_a,in_proj_b` | 5120 → 48 | 48 each | BF16 | Yes |
| Linear `out_proj` | 6144 → 5120 | 48 | NF4 | Yes |
| MLP `gate_proj,up_proj` | 5120 → 17408 | 64 each | NF4 | Yes |
| MLP `down_proj` | 17408 → 5120 | 64 | NF4 | Yes |

Full-attention `q_proj` output splits into Q and an output gate, each 6,144-dimensional; the gate is sigmoid. GDN's z gate is SiLU, b is sigmoid, and a forms negative decay through softplus and `exp(A_log)`. The new configuration field `output_gate_type="swish"` must not be misread as replacing full-attention sigmoid with SiLU. Pinned HF source `Qwen3_5Attention.forward` and `Qwen3_5RMSNormGated.forward` explicitly specify two distinct gates; current main does likewise. Reuse the official forward directly rather than independently interpreting the field and changing operators.

Pinned source: GDN class lines 371-557, full attention 645-718, RMSNormGated 187-203.


Source links embedded on source page 12:

- [Source link 1](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py)
- [Source link 2](https://github.com/JIA-Lab-research/RePlan/tree/6c9b12f0c619cc8ca6f50b536b24aff377525f8c)
- [Source link 3](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
- [Source link 4](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L371-L557)
- [Source link 5](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L645-L718)
- [Source link 6](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L187-L203)

**Source page 13**

### 6.2 Exact parameters and LoRA coverage

I used HTTP Range to retrieve only each safetensors file's 8-byte length and JSON header, parsed all 1,199 weight shapes, and checked that the index key set matched. Results:

| Quantity | Actual metadata value |
|---|---:|
| Parameters in the complete published checkpoint | 27,781,427,952 |
| Of which MTP weights | 424,699,392 |
| Resident model parameters in this design, without instantiating MTP | 27,356,728,560 |
| Vision tower, including merger | 460,730,096 |
| Token embedding | 1,271,398,400 |
| `lm_head`, independent and not tied | 1,271,398,400 |
| Large Linear modules converted to NF4 | 400 modules; 24,326,963,200 parameters |
| Other unquantized parameters | 3,029,765,360 |
| LoRA targets | 496 Linear modules |
| Parameters per rank-16 adapter | 116,727,808 |

Excluding MTP from residency is not an independently invented removal of a model branch: the pinned `Qwen3_5ForConditionalGeneration.__init__` creates only `model` and `lm_head`; `Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected=[r"^mtp.*"]` explicitly ignores these weights. See lines 812-824 and 1709-1720. Retain complete upstream download verification; do not silently present MTP as a component trained by this project.

LoRA uses rank 16, alpha 32, dropout 0, and bias none, covering the projections above in every language block. Freeze vision tower/merger, embedding, `lm_head`, norms, `conv1d`, `A_log`, and `dt_bias`.

For a linear map $W\in\mathbb R^{d_o\times d_i}$, one adapter has $r(d_i+d_o)$ parameters. Thus:

$$N_{\mathrm{linear},16}=16\{15360+11264+5168+5168+11264+3(22528)\}=1,852,928,$$
$$N_{\mathrm{full},16}=16\{17408+6144+6144+11264+3(22528)\}=1,736,704,$$
$$N_{\mathrm{adapter}}=48N_{\mathrm{linear},16}+16N_{\mathrm{full},16}=116,727,808.$$

The PEFT target regex uses actual fully qualified module names:

```text
^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$
```

At startup, enumerate the instantiated model and assert exactly 496 targets and 116,727,808 parameters; save a names/shapes manifest. Excluding the 96 small a/b gates from quantization does not exclude them from LoRA.

### 6.3 Main execution path: one complete NF4 model on each of A0/A1

Resident BF16 27B weights occupy approximately 50.956 GiB and do not fit on one A6000. The main design uses a frozen NF4 base, BF16 arithmetic, and FP32 adapters without shrinking 27B to a small model. Each of two A6000s holds a complete replica, alternating sampling and training, with gradients summed only across these two GPUs. A2 runs the frozen generator encoder/VAE, A3 runs the frozen visual judge/tools, and the three 5090s run DiT. There is no cross-machine FSDP or per-generated-token PCIe all-gather of 27B weights.

Quantization-configuration design:

```text
load_in_4bit=true
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=true
bnb_4bit_compute_dtype=bfloat16
bnb_4bit_quant_storage=uint8
llm_int8_skip_modules=["lm_head", "model.visual", "in_proj_a", "in_proj_b"]
device_map={"": local_cuda_index}
```

Although its name contains int8, the pinned 4-bit quantizer actually reads `llm_int8_skip_modules`. See the 4-bit quantizer at lines 121-143, the module-matching function, and Linear4bit replacement.

After first quantization, save one serialized NF4 base and hashes for every quantized tensor and `quant_state`; both workers load this same artifact. Do not independently requantize each stage or replica. Official NF4 block size 64 and double-quantization second-level block size 256 can be checked in `functional.py` lines 917-948.


Source links embedded on source page 13:

- [Source link 1](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L812-L824)
- [Source link 2](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1709-L1720)
- [Source link 3](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/quantizers/quantizer_bnb_4bit.py#L121-L143)
- [Source link 4](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/quantizers/quantizers_utils.py#L25-L43)
- [Source link 5](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/integrations/bitsandbytes.py#L164-L230)
- [Source link 6](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py#L917-L948)

**Source page 14**

### 6.3.1 Precision handling in `prepare_model_for_kbit_training`

The pinned PEFT implementation freezes the base and converts every non-Params4bit FP16/BF16 parameter to FP32. Do not call the function and then uncritically reuse a 17 GiB budget. See `other.py` lines 151-244.

Executable preparation order:

1. Load the fixed NF4 base onto one GPU using the official multimodal `Qwen3_5ForConditionalGeneration`, not a text-only class.
2. Call `prepare_model_for_kbit_training(...,gradient_checkpointing_kwargs={"use_reentrant":False})`.
3. Restore unquantized, frozen embedding, `lm_head`, vision Linear/Conv, language `conv1d`, and a/b projections to BF16; leave norms, `A_log`, and `dt_bias` in FP32. **Do not call `.to(bfloat16)` on the whole model or modify Params4bit payloads.**
4. Insert the policy adapter with FP32 parameters. Enable non-reentrant decoder checkpointing; extract frozen vision-tower features under eval/no_grad. The current decoder is in train mode, all dropout is 0, and replay uses `use_cache=False`.
5. Check dtype and the `requires_grad` allowlist parameter by parameter; record actual allocated/reserved memory.

Norms, `A_log`, and `dt_bias` total 806,144 parameters, adding approximately 1.54 MiB relative to keeping all unquantized parameters BF16; large embedding/head parameters no longer mistakenly add approximately 4.74 GiB.

### 6.3.2 Quantization does not cut off gradients

One model projection is

$$y=\widehat W_0x+\frac\alpha r BAx,\qquad\widehat W_0=D(Q_{\mathrm{NF4}}(W_0)).$$

Freezing $\widehat W_0$ means only that the quantized base is not updated; it must still propagate $\partial L/\partial x$ to upstream activations. The bitsandbytes `MatMul4Bit.backward` computes a dequantized matrix multiplication for input gradients and returns None for quantized-weight gradients. PEFT's bnb layer adds the LoRA branch to the base output. See bnb backward lines 365-386 and PEFT bnb forward 510-558. Therefore the entire frozen language-base forward must not be wrapped in `no_grad`, which would remove gradients for earlier LoRA layers.

### 6.4 From SFT to RL: copy a same-rank adapter as reference

Use one active adapter on a shared base. Retain the policy adapter after SFT, copy it into a frozen reference, and continue updating policy during RL. There is no SFT merge, requantization, or new zero-B adapter.

$$\pi_\theta=\pi(\widehat W_0,\theta),\qquad\theta_{t=0}=\theta_{\mathrm{SFT}},\qquad\pi_{\mathrm{ref}}=\pi(\widehat W_0,\theta_{\mathrm{SFT}}^{\mathrm{frozen}}).$$

Interface:

```python
get_peft_model(base, lora_config, adapter_name="policy")
```

When SFT finishes, save the policy checkpoint, then load that checkpoint:

```python
load_adapter(checkpoint, adapter_name="reference", is_trainable=False)
```

After copying, check that every tensor has equal values but distinct `data_ptr`/storage. Reference must not point to the same tensor and then change with policy updates.

For each replay microbatch:

1. `model.set_adapter("reference", inference_mode=True)`; `eval()`; calculate reference logp/KL targets under `no_grad`. Cache only required values and finish this gradient-free computation.
2. `model.set_adapter("policy", inference_mode=False)`; restore the language-LoRA allowlist and keep vision frozen; current decoder `train()`, dropout 0, `use_cache=False`.
3. Build the current graph, calculate that stage's CE or PG/KL, and complete backward.
4. Switch adapters only after the current graph finishes. Do not change the active adapter to reference during checkpoint recomputation.

Pinned API evidence: `PeftModel.set_adapter` lines 1591-1613 and `BaseTunerLayer.set_adapter` 1609-1635. `set_adapter` changes `requires_grad`; it is not merely routing. `disable_adapter()` returns the pre-SFT base, not this design's reference, and cannot substitute for it.


Source links embedded on source page 14:

- [Source link 1](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/utils/other.py#L151-L244)
- [Source link 2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/autograd/_functions.py#L365-L386)
- [Source link 3](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/bnb.py#L510-L558)
- [Source link 4](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/peft_model.py#L1591-L1613)
- [Source link 5](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/tuners_utils.py#L1609-L1635)

**Source page 15**

### 6.5 Vision, tokenizer, and action-probability interfaces

#### 6.5.1 The actual processor

`preprocessor_config` specifies `Qwen3VLProcessor + Qwen2VLImageProcessorFast`. The pinned version has no `processing_qwen3_5.py`; use that revision's `AutoProcessor.from_pretrained` directly.

For one 512 × 512 image with patch 16, temporal patch 2, and merge 2, after duplicating/assembling temporal patches for a static image:

```text
pixel_values: [1024, 3*2*16*16] = [1024,1536]
image_grid_thw: [1,3], value [1,32,32]
vision encoder output: [1024,1152]
after merger: [256,5120]
```

Four images concatenate into `pixel_values[4096,1536]`, grid `[4,3]`, and 1,024 image placeholders in total. Vision features are frozen tensors, cacheable by image hash + processor + vision hash; width is 5,120, not v2's 2,560, and they must not be sent into Klein's dedicated text encoder. Qwen3.8 vision configuration has `deepstack_visual_indexes=[]`, and the model implements a single vision-merger injection. **Do not save or pass v2 Qwen3-VL's three-layer DeepStack features.** See VisionModel lines 1018-1050 and multimodal forward 1535-1605.

Save and replay `input_ids,attention_mask,mm_token_type_ids,image_grid_thw,pixel_values` or frozen vision features, `prefix_image_positions,position_ids`, and template parameters. `mm_token_type_ids` is 1 for images and 0 for text; append 0 for generated actions. With images present, the model throws an error if this field is missing; do not "guess" positions from placeholder counts. See position IDs lines 1490-1527. The generation path uses text + 3D `[4,B,S] position_ids`; ordinary 3D image positions are `[3,B,S]`. Save and check the actual behavior-path products; never borrow old `rope_deltas` across actions. Full replay consistently uses the prefix-slot assembly in §11.1.1 to prevent anomalous completion visual tokens from triggering native placeholder-count conflicts.

#### 6.5.2 Thinking and tool templates

Qwen3.8 defaults to thinking + xhigh and preserves historical thinking by default. This design's main action protocol explicitly sets:

```python
enable_thinking=False
preserve_thinking=False
tools=None
add_generation_prompt=True
```

The reason is that the system trains executable planning/repair actions under a finite budget. Planning, diagnosis, and evidence fields themselves carry the necessary judgments. The main path does not need long free-form CoT plus another accounting system and output-truncation rules. A future study of thinking budgets may include all generated thinking tokens in action probabilities and budgets and introduce a separately preregistered comparison; it must not silently remove their probabilities.

In non-thinking mode, the official template adds the fixed prefix `<think>\n\n</think>\n\n`. This belongs to the input, not sampled tokens. Passing `tools` makes the template request XML `<tool_call><function=...>`. The current protocol outputs one JSON object, so do not pass tools; put function schemas in the fixed system prompt. Preserve raw output tokens; do not parse/reformat before calculating logp.

#### 6.5.3 EOS, padding, and distributions

Vocabulary size is 248,320. EOS is `[248046,248044]`, where 248046 is `im_end` and 248044 is `endoftext`; PAD is also 248044. SFT target sequences end with `im_end`. Either EOS actually sampled in RL contributes to the score and terminates generation. Determine padding with actual-length masks; `token_id == pad_id` must not remove a real EOS.

Official `generation_config` defaults to top_k 20, top_p .95, and temperature 1. To train with full-support categorical sampling, explicitly override `top_k=0,top_p=1,min_p=0,temperature=the project-defined value,repetition_penalty=1`; apply no other logits processors unaccounted for in the score. Saved sampling logp must come from the completely normalized logits after the same temperature transform.

SFT, RL, and reference use the same chat template, image resizing, positions, quantized base, and kernel manifest. Maximum action length is fixed at 1,024 and input at 6,144, totaling 7,168. Actual training peak memory is accepted in §19; static budgets are not measurements.

### 6.6 Actual hybrid-cache defects and the minimal fix

This is the main implementation gap that must be addressed when changing models in this round, independent of how a paper describes it.

#### 6.6.1 Pinned 5.12.1 incorrectly stores recurrent state in the convolution dtype

In `cache_utils.py` lines 917-941, `LinearAttentionLayer.lazy_initialization` first sets `self.dtype` from `conv_states`, then creates `recurrent_states` with that dtype. On a BF16 model, even FP32 `final_state` returned by FLA is copied into BF16 storage; the configuration's `mamba_ssm_dtype` is not read here.

The FLA recurrent kernel explicitly creates FP32 `final_state`, and its backward is directly `NotImplemented`. See final-state allocation lines 211-215 and backward 300-306.


Source links embedded on source page 15:

- [Source link 1](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1018-L1050)
- [Source link 2](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1535-L1605)
- [Source link 3](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1490-L1527)
- [Source link 4](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/cache_utils.py#L917-L941)
- [Source link 5](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/gated_delta_rule/fused_recurrent.py#L211-L215)
- [Source link 6](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/gated_delta_rule/fused_recurrent.py#L300-L306)

**Source page 16**

**Minimal fix design:** add `FP32LinearAttentionLayer(LinearAttentionLayer)`, overriding only `lazy_initialization`. For convolution, call the original parent implementation. For recurrence, assert that the incoming dtype is FP32, use `zeros_like(recurrent_states)`, set the initialized flag, and do not pass `self.dtype`. After creating `DynamicCache(config=model.config)`, replace its 48 linear layers with this subclass before the first prefill; leave the 16 DynamicLayer instances unchanged. Assert layer by layer that conv is BF16, recurrence FP32, and full-attention KV BF16.

This agrees with the later upstream fix: upstream FP32-fix lines 1189-1195 use `zeros_like(recurrent_states)`. The current cache API has been comprehensively redesigned; this project need not immediately upgrade the whole stack for this one fix.

#### 6.6.2 Cache is an optimization within one action, not DAG state

The old linear cache's `crop()` is a no-op and cannot undo recurrent state; see lines 912-915. Therefore every Agent action prefills anew from its current public view, uses caching for within-action autoregression, and destroys the cache when the action ends. DAG SELECT/rollback changes artifact pointers and revision; it neither reuses nor crops the old language cache.

Replay always uses the complete prefix plus action, `past_key_values=None,use_cache=False`, and chunk GDN with backward support. Do not detach prefix state and backpropagate only through the suffix to save memory; that would lose parameter gradients through prefix encoding. Default microbatch is 1 with no sequence packing, also avoiding linear recurrence streaming across packed samples.

#### 6.6.3 Acceptance tests for exact-probability consistency

"Same weights" cannot substitute for a sampling/replay comparison, because cached decode and full-sequence chunk execution have different numerical paths. Before formal training, use fixed cases with 0/1/2/4 images, short/long inputs, and both EOS tokens:

1. During action sampling, record the normalized logp of every sampled token and full logits at sampled positions.
2. Recompute with gradient-free, teacher-forced full replay at the same behavior version; report mean/p99/maximum logp differences and full-vocabulary KL at sampled positions.
3. Recompute the same results with gradient checkpointing and training mode enabled, dropout 0; check that language-LoRA gradients are nonzero and vision/base gradients absent.
4. If thresholds are exceeded, first check FP32 cache, position IDs, temperature, EOS, template, NF4 version, and kernels. Do not conceal the problem by changing stored logp.

Language caches for root plans and suffix branches must not share mutable objects; each view starts a new prefill. For a fixed token sequence, cache is only an implementation acceleration. RL cannot start before the consistency gate passes. Thresholds should be based on and record actual floating-point noise on the development machine. For example, mean absolute logp ≤ 0.01 nat, p99 ≤ 0.05 nat, and sampled-position KL ≤ 1e-3 are first-round implementation checks. If they fail, repair the path rather than directly relaxing them. None of these has yet been executed on a GPU.

#### 6.6.4 Additional note: RoPE offsets belong to the model instance, not Cache

Pinned `Qwen3_5Model.__init__` sets `self.rope_deltas` at line 1246. PEFT's `get_base_model`, lines 1099-1103, returns the wrapped HF model for LoRA. The actual reset path is therefore `peft_model.get_base_model().model.rope_deltas = None`.

The necessity must be scoped accurately: the pure-text prefill branch of `compute_3d_position_ids` (`input_ids` present, `past_len=0`) returns None without clearing old deltas. A custom token loop followed by cached decode may add the previous image request's delta at lines 1520-1530. **Normal HF `.generate` already zeroes it for a new text-only request in its position-preparation function, lines 1903-1918; its default path must not be incorrectly described as necessarily broken.**

Unified worker convention: clear `rope_deltas` once at the entry to every sample request, every reference full replay, and every policy full replay. Each action gets a new cache; a complete generate/full backward runs serially on the same instance. Reset occurs at the beginning of a request, **not before every decode token within the action**. Default sampling lets `HF.generate` rebuild positions and deltas from that request's IDs/grid/mm_token_type_ids. Full replay recomputes complete position IDs from saved input; reference/current share the same position tensor. If a custom token loop passes explicit position IDs, prefill must obtain its delta through `get_rope_index` for the same request and calculate subsequent positions accordingly rather than borrowing the instance's old value. Add cache/reset comparisons for image → text-only → image and branch switching.

### 6.7 Trainable projections in the 9B DiT

DiT hidden width is 4,096, with eight dual-stream blocks and 24 single-stream blocks. Dual-stream training targets `attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}`; single-stream training targets `attn.{to_qkv_mlp_proj,to_out}`. The single-stream fused input is 4096 → 36864 and output is 16384 → 4096, so LoRA also covers the fused MLP path. With rank 16, alpha 16, and dropout 0:

$$N_\phi=8\cdot8\cdot16(4096+4096)+24\cdot16[(4096+36864)+(16384+4096)]=31,981,568.$$


Source links embedded on source page 16:

- [Source link 1](https://github.com/huggingface/transformers/blob/c0d024fbc873ab0325e1b40fc1bd69069609de65/src/transformers/cache_utils.py#L1189-L1195)
- [Source link 2](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/cache_utils.py#L912-L915)
- [Source link 3](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1246)
- [Source link 4](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/peft_model.py#L1099-L1103)
- [Source link 5](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1520-L1530)
- [Source link 6](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1903-L1918)

**Source page 17**

There are 112 modules and 224 A/B tensors; the fixed base has 9,078,581,248 parameters. Both policy and reference use FP32 LoRA, while DiT forward uses BF16. SFT initializes A randomly and B = 0; RL continues from SFT-trained A/B, so the claim that "A necessarily has zero gradient on the first RL step" does not apply. Freeze VAE/BN, Qwen3-8B encoder, original DiT parameters, and reference. Independently copy adapter values from the SFT snapshot, with the same gradient boundaries as 27B; do not mix the two frameworks' APIs. [S2, S5, S24]

DiT uses native Diffusers `PeftAdapterMixin`, not an outer `PeftModel` wrapper. Initialize with `transformer.add_adapter(config, adapter_name="policy")`. After SFT, obtain CPU-cloned LoRA values using `get_peft_model_state_dict(transformer, adapter_name="policy")`; create reference with the same configuration and copy using `set_peft_model_state_dict(..., adapter_name="reference")`. Verify equal values and independent storage tensor by tensor, explicitly convert to FP32, and freeze reference. DiT's `transformer.set_adapter("reference")` does not accept `inference_mode`. After switching, explicitly freeze every non-policy parameter, run reference forward under no_grad, and restore `requires_grad` only for policy A/B after switching back. Reference switching must not inadvertently clear policy gradients. See pinned Diffusers `loaders/peft.py` lines 508-590. The call with `inference_mode` in §6.4 applies only to the 27B PeftModel.

### 6.8 How gradients pass through the system

SFT's CE updates $\theta$ and FM updates $\phi$. Shared tasks and bundles align the two supervised objectives, but there is no backpropagation through discrete PLAN. In joint RL, raw action-token scores update $\theta$, while Gaussian-transition scores update $\phi$. Sampled images, JSON parsing, hard masks, compilation, frozen encoders, and rewards are fixed conditions or environment operations. Their influence on later decisions is included in the complete trajectory distribution.

A frozen language/DiT base does not mean its forward can be wrapped in no_grad: LoRA in each layer still needs to pass input gradients through fixed projections to earlier layers. Only reference, pure rollout, vision tower, condition encoders, and criteria use gradient-free paths. Both models optimize the same task objective; mathematics appears in §§11-14.

## 7. Data sources, isolation, construction, and filtering

### 7.1 Data and licensing

| Data | Pinned source and use | Treatment |
|---|---|---|
| MagicBrush | train/dev/test multi-turn editing; HF `6c254560879df7851233ea0fd2ae2a62764d0ef2` | Use train for construction; retain official dev/test. Split by session and underlying source image. |
| OmniEdit-Filtered-1.2M | HF `b6eec40a8a6994aa380497e33b1964742d7e028b` | Stream and filter a subset rather than downloading the approximately TB-scale full dataset; editing instructions are not presented as complete captions. |
| Self-constructed diagnostic and repair trajectories | Images from the training partition and independent task definitions | Save injections, recovery, measured candidates, and private labels. |
| GenEval / GenEval2 | Official evaluation | Exclude from SFT, RL, teacher-prompt selection, and reward calibration. |
| IV-Edit | No explicit dataset-license declaration was found in the pinned data card | Not a mandatory default dependency; register as supplementary evaluation only after establishing an explicit basis for use. |

MagicBrush's current official description says test is public. Train/dev/test contain respectively 4,512/266/535 sessions and 8,807/528/1,053 turns. Older statements that the test set requires an application are outdated. MagicBrush is labeled CC-BY-4.0, and OmniEdit's data card MIT. Underlying image-source records must still be retained; a data-card license does not automatically eliminate source-image constraints. [S12, S13]

Previous byte-for-byte verification confirmed that Tencent `datasets/geneval2/synthetic/test.jsonl` is identical to the official GenEval2 800-entry file: both have 423,220 bytes and SHA-256 `09233adbc9f877fba205f1a20efa4a8640e7e62a91dfd55893562bed8a928e43`. The entire file, rewrites of it, and teacher-derived trajectories are prohibited from the training pool. [S14]

### 7.2 Split before expansion

Build data families from connections through the same source/target image, editing session, entity-reference source, near-duplicate image, controlled template instance, or its counterfactuals. A connected component is the minimum splitting unit.

Preprocessing order: isolate official test → deduplicate exact content hashes → review candidates with pHash Hamming distance ≤ 6 → manually/rule-review candidates with DINOv2 cosine ≥ 0.98 → fix families → hash-split into train/development/calibration/diagnostic-test. If the same underlying image crosses official splits, remove its family from the self-constructed training pool; do not change official test data.

Development, calibration, and the 500 specialized test tasks retain at least 200, 600, and 500 independent families respectively. If numbers are insufficient, expand candidate data rather than making different crops of the same source image count as independent samples. The RL task pool defaults to 2,000 training tasks and may share the training partition with SFT; record overlap and additionally report results on unseen task combinations.


Source links embedded on source page 17:

- [Source link 1](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py)
- [Source link 2](https://github.com/huggingface/peft/tree/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e)
- [Source link 3](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15)
- [Source link 4](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/loaders/peft.py#L508-L590)
- [Source link 5](https://huggingface.co/datasets/TainU/IV-Edit/blob/964d92584ade3fadbd0d6a55f069e28936c94842/README.md)
- [Source link 6](https://huggingface.co/datasets/osunlp/MagicBrush/tree/6c254560879df7851233ea0fd2ae2a62764d0ef2)
- [Source link 7](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/tree/b6eec40a8a6994aa380497e33b1964742d7e028b)
- [Source link 8](https://github.com/facebookresearch/GenEval2/tree/a6e82d2289e8d418f27f0adee77908b07060eea3)

**Source page 18**

### 7.3 The initial 8,000 DiT SFT samples

| Source | No region | One region | Two regions | Total |
|---|---:|---:|---:|---:|
| MagicBrush | 2,000 | 1,200 | 800 | 4,000 |
| OmniEdit | 1,400 | 600 | 400 | 2,400 |
| Complete-caption T2I from training images | 600 | 200 | 800 | 1,600 |
| Total | 4,000 | 2,000 | 2,000 | 8,000 |

These are source and condition-coverage quotas. A two-region example may mean one edited region plus one protected/reference-bound region. Store a separate `two_changed_entities` label rather than advertising all such examples as simultaneous two-object editing capability.

Initial OmniEdit filtering is `sc_score_1≥8,sc_score_2≥8,pq_score≥8`, followed by independent checks of target completion, content consistency, protected regions, and registration. It does not have universally reliable native regional masks; RGB differences can propose candidates but cannot directly serve as edit ground truth. Reliable regions come from annotated masks, reviewed detection plus segmentation, explicit bboxes, or controlled image construction. Samples without reliable regions enter only the global-condition bucket.

T2I samples require complete image descriptions verified for entities, counts, relations, and backgrounds by a frozen teacher and manual spot-checks. "Make the hat red" is not a complete caption. Target-region annotations supply supervision weights and compatibility filtering; regions actually input to DiT come from the public PLAN below or controls originally provided by the user. Target pixels and private mask IDs must not enter PLAN/ConditionBundle.

Appearance samples comprise 10% of the total, allocated within each bucket. Prefer cropping the target entity from source; cross-image same-identity tasks require identity evidence. By default, hidden target crops are not used to reconstruct conditions (0%), and complete targets are not references. New appearance-reference tasks must explicitly supply the corresponding crop in the public task definition and let the Actor actually read it; it cannot be temporarily added from a hidden target. Evaluate cross-view identity capability separately.

#### 7.3.1 Both sides' SFT share the same plans and conditions

For each candidate paired sample, construct a public root view containing the request, original source, and genuinely allowed user controls. A frozen 27B teacher proposes PLAN from this view; the compiler produces $c_i=\operatorname{Compile}(G_i)$. The private target and annotations enter the filter only after proposal. Only semantically compatible tuples $(x_i,G_i,c_i,I_i^*)$ with legal public references and reliable geometry enter the linked dataset.

All 8,000 DiT samples save `task_id / root_view_hash / plan_token_ids / graph_revision / compiled_condition_hash / target_id`; DiT consumes $c_i$. Take half from each source/region cell, totaling 4,000 PLAN examples for Actor SFT:

| Source | Global PLAN | Single-region PLAN | Two-region PLAN | Total |
|---|---:|---:|---:|---:|
| MagicBrush | 1,000 | 600 | 400 | 2,000 |
| OmniEdit | 700 | 300 | 200 | 1,200 |
| Complete-caption T2I | 300 | 100 | 400 | 800 |
| Total | 2,000 | 1,000 | 1,000 | 4,000 |

Filtering checks at least: all PLAN references and types are legal; textual requirements agree with the paired target; for samples with reliable entity boxes whose plans use regions, PLAN target regions have IoU ≥ .5 with annotated entity boxes; source regions are checked independently when source is involved; object identity/edit instructions remain consistent. Without reliable regional evidence, use only the global bucket. Do not write target boxes back into the teacher's PLAN and still call it a plan produced from public inputs. Tasks in which the user explicitly provides layout may be constructed, but layout must appear in that task's public root view; label them separately and cap their share at 25%. Formal evaluation may use only controls actually supplied by the user.

The teacher proposes at most twice per original candidate. The second proposal corrects only formatting or public-contract issues; target-check results are not fed back as prompts for repeatedly trying answers. Review at most 12,000 original candidates, with at most 24,000 initial PLAN teacher calls. An additional 96 protocol PLANs permit at most 192 proposals; 1,024 rule-based recovery demonstrations are counted separately. Report actual counts when quotas are not met. Each Actor cell takes $\lfloor n/2\rfloor$ of its actual valid samples; do not duplicate data to fabricate 4,000 independent tasks. Account separately for source-image verification, teacher calls, and renders.

Both sides' initial supervision thus has the same condition semantics: Actor learns to propose the public source of $c_i$, and DiT learns to execute $c_i$. Private targets remain only supervision and filtering inputs, adding no information unavailable at inference.

**Source page 19**

### 7.4 Same-state SFT for repair, rollback, and termination

Organize data into "unique state bundles" rather than promising 16,000 mutually independent repair cases. A bundle stores the actual pre-action state and multiple execution-verified successors:

```text
StateBundle:
  family_id, state_id, renderer_version, sampler_id, compiler_version
  public_before_view, public_dag_revision, source_image_id
  candidate_actions[], private_outcomes[], public_after_views[]
  verified_action_records[], intervention_node_set?, repair_node_set?
ActionRecord:
  state_id, phase, raw_input_ids, processor_artifacts
  target_ids, target_mask, public_reference_ids, verifier_provenance
```

Split by family first, then expand states and actions. The offline teacher reuses a frozen Qwen3.8-27B snapshot on A0/A1; its input is strictly that action's `serialized_actor_view(state_id)`. Rule-based patches supply candidates, while the teacher adds wording and compositional planning. First validate schema, references, frames, capacity, and condition hashes; then execute at most two truly distinct candidates. When natural failure has no unique root cause, supervise only effective revisions and evidence rather than forcing a unique erroneous node.

One failed state may provide before → DRAFT_REPAIR, after candidate return → APPLY, and after actual success → FINISH. A regressed result may provide → SELECT of a historical image and revision, followed after restoration by → FINISH or a new legal revision. Every target has its own real pre-action view; an image obtained after success cannot be inserted into the pre-repair input. Historical erroneous actions have CE mask = 0; the current verified target has mask = 1. Effective repair does not imply immediate acceptance: if fixing a cup breaks existing text, the data should teach rollback or further repair.

For a fixed state $s=(I_{\mathrm{before}},G_{\mathrm{revision}},h)$ and candidate $c$:

$$I_c^{(r)}=\operatorname{Render}_{\phi_{\mathrm{SFT}}}(c,s;\epsilon^{(r)}),\qquad\Delta_c^{(r)}=Q_{\mathrm{dense}}(I_c^{(r)})-Q_{\mathrm{dense}}(I_{\mathrm{before}}).$$

Improvement for every verification seed subtracts **the same original before image**. The second seed changes only candidate-generation noise. Regenerating a different before image creates a new state that must have a separate state ID, not be mixed into the same action label.

The first round uses one seed per candidate. A positive repair requires $\Delta_c\ge0.1$, and every previously known-passing mandatory requirement must remain known-passing in every output used for verification. All $J$ requirements participating in fixed $Q_{\mathrm{dense}}$ must be known in before and every verified after; do not recompute the denominator from the intersection of known items. Cases with $\Delta\in[0.05,0.15]$, conflicting criteria, or unclear regressions receive priority for the second-seed quota. If an action remains uncertain after that quota is exhausted, do not give it a positive label. Necessary temporary degradation is labeled as a multistep positive example only after verification of the complete subsequent trajectory.

Second-seed verification fixes before, executes both candidates once more, and reserves one explicit retry comparison. Retry reuses the selected image's historical producer's complete conditions, source/references, and versions, changing only the seed. It is another legal action in the same state; **it does not replace before and is not an RL baseline**. A small difference between one retry and two repairs is insufficient to establish an expected-benefit advantage; use it only as selection and diagnostic evidence. Paired noise is used only for offline comparisons of this kind; online sibling RLOO remains independent.

Controlled injection saves two kinds of evidence separately. First, with identical original input, versions, and exogenous noise, change only injected nodes and verify a target score ≥ 0.8 for the normal image and ≤ 0.6 for the bad image. Then execute legal DRAFT/APPLY from the bad image's actual public state and verify a restored score ≥ 0.8. The second part defaults to the bad image as source and demonstrates online recoverability; it is not a strict counterfactual in which only one variable is reverted. `intervention_node_set` and actual `repair_node_set` may differ. Prohibit using the correct target as reference, changing source outside the logs, or copying the normal image directly as recovery. Two injection seeds produce two distinct bad states; save their snapshots separately.

FINISH preferably comes from complete success, budget exhaustion, or a state following real rollback. "Nonpositive benefit from continuing" refers only to tested candidates and current budget, not proof of optimal stopping over the entire action space. Gold data is used only for retrospective filtering. Targets cannot reference hidden eval IDs, future images, or private answers.

#### 7.4.1 Finite data budgets and stage dependencies

First construct the 4,000 PLAN demonstrations aligned with paired samples, plus 96 complex protocol PLANs and 1,024 programmatically verifiable contract-error/recovery demonstrations, all with a 1,024-output-token limit. These require no new DiT renders and may run in parallel with paired-image DiT SFT. **Visual repair data must wait until DiT SFT and sampler selection in §12.4 are complete.** By default, do not directly use ODE28 to label SDE50 repair actions positive. If it is separately used for prescreening, positive labels must still be verified with the final sampler and additional calls accounted for.

| Construction item | Budget definition | Render limit |
|---|---|---:|
| Natural-state collection | 1,024 starting-task attempts, one initial generation each | 1,024 |
| Candidate execution | At most two candidates per state above, one seed each | 2,048 |
| Boundary verification | At most 256 states: second seed for both candidates plus one retry | 768 |
| Controlled injection | 256 normal/bad/executable-recovery triplets | 768 |

**Source page 20**

| Construction item | Budget definition | Render limit |
|---|---|---:|
| Injection verification | Add one new-seed triplet to at most 64 injection instances | 192 |
| Main construction subtotal | Includes failed, unknown, and unlabeled attempts | **4,800** |
| Two rounds of own-state correction | 2 × 256 tasks, at most three renders per task | 1,536 |
| Total | Excludes the sampler gate, RL pilot, and formal evaluation | **6,336** |

A 600-render data pilot consists of 128 natural attempts, 32 injections, 32 natural verifications, and eight injection verifications, all counted within the quotas above. The number 1,024 denotes attempts, not a guarantee of 1,024 failed states; successful initial images can also generate stopping supervision. When no label is obtained, do not sample indefinitely until reaching "16,000 records" or lower passing thresholds. Report actual unique family/state/action counts first, followed by training presentation counts and oversampling ratios. Any expansion requires an explicit next-batch budget.

Own-state correction collects real trajectories using the current controller and fixed DiT SFT version. The three renders for each of 256 tasks per round include candidate verification; no additional candidate enumeration may be added outside the table. Each round performs 100 Controller SFT updates only on correction actions actually executed and verified. DiT is not updated by default. If DiT SFT must change, first create a new renderer/sampler version and then recollect affected repair records; old outcomes cannot establish the new model's effectiveness.

The loader samples family first, StateBundle second, then an action under §8's stage quotas. At most two records in a global batch come from the same bundle. During SFT construction, verify the 1,024-output-token capacity with the real tokenizer. Correct demonstrations that are too long must be split into legal multiple actions or registered under an expanded protocol; never truncate and supervise them. Inputs may reference only artifacts disclosed at the time. A private artifact ID for a target mask cannot pretend that the Agent already knew its location.

### 7.5 Data-reliability acceptance tests

Inspect 100 examples from every data bucket, or all examples when fewer than 100 exist. Record correctness rates for task completion, masks, identity, registration, and original-text coverage. If any key label's error rate exceeds 5%, pause generation for that bucket, repair the rules, and resample; do not lower thresholds to fill counts. Before training, check family overlap, known benchmark prompts, image hashes, target leakage, and future information.

If a quota is short, first expand candidates within the same training partition and add manual annotation. If still short, publish a new data configuration, report actual counts, and preserve no-region/single-region/two-region sampling ratios. Do not copy the same image as a new independent sample or fill gaps with official test data. In the first round, use 200 candidates per bucket to estimate pass rates, then estimate later cost from valid counts divided by pass rates. The initial original-pair DiT screening reviews at most 12,000 pairs to target the 8,000 pairs in §7.3; if insufficient, report actual yield and the gap rather than filling it indefinitely. Repair data follows §7.4's separate attempt limits. Original image-pair verification, teacher proposals, public checks, and private scores are accounted for separately.

## 8. Exact definitions of SFT on both sides

### 8.1 SFT normalized by decision stage, then by action

This design prevents long PLAN/PATCH actions from taking almost all tokens while short SELECT/FINISH actions receive almost no training share. It changes SFT demonstration weights, not the RL trajectory-score definition.

Let $g\in\{1,2,3,4\}$ be the action stage, with stage weights $q=(0.25,0.35,0.25,0.15)$. Complete repair SFT uses global batch = 20:

| Stage | Actions | Records per batch $n_g$ | Weight |
|---|---|---:|---:|
| Planning | PLAN, initial controls | 5 | 0.25 |
| Repair execution | DRAFT_REPAIR, APPLY, including legal retry | 7 | 0.35 |
| Delivery and rollback | SELECT, FINISH | 5 | 0.25 |
| Evidence acquisition / protocol recovery | INSPECT, READ_NODES, legal recovery after invalid actions | 3 | 0.15 |

Assign each record exactly one phase in advance. A recovery PLAN following an invalid action belongs to category 4, not also category 1. Let $m_{i\ell}\in\{0,1\}$ be the valid supervision mask for action $i$, and $d_i=\sum_\ell m_{i\ell}>0$:

$$\ell_i^{\mathrm{CE}}=-\frac1{d_i}\sum_\ell m_{i\ell}\log\pi_\theta(y_{i\ell}\mid h_{i\ell}),\qquad\mathcal L_\theta^{\mathrm{SFT}}=\sum_{g=1}^4\frac{q_g}{n_g}\sum_{i:g_i=g}\ell_i^{\mathrm{CE}}.$$

System/user/tool tokens, the fixed empty-think prefix, image placeholders, historical errors, and padding have mask 0; the current verified target action and correct `im_end` EOS have mask 1. Generate masks from actual lengths and role positions rather than PAD token IDs, because the 27B model's PAD equals one EOS token.

**Source page 21**

Remove unconfirmed root-cause wording or set its spans to 0 while continuing to supervise valid patches. Token boundaries use original IDs and construction-time spans, not string searches. Reject all-zero targets. Distributed padding does not enter denominators.

Each rank contributes $q_g\ell_i/n_g$ per action, where $n_g$ is the global quota. After SUM all-reduce, do not divide by world size again. If a phase is short, keep accumulating until its full quota is met; do not temporarily renormalize the remaining phases. Repeated sampling is explicit oversampling and must not receive another application of the same stage weight. When protocol warmup does not yet have all four categories, average per-action CE over the available protocol actions and record it as a separate stage; do not present it as complete phase-balanced training.

SFT permits demonstration EOS; in RL, an artificially appended EOS must have policy mask 0. The main RL stage does not reintroduce CE/FM replay; fixed-reference KL preserves SFT capabilities. **RL does not average by each action's actual length** and still uses §13's fixed budget denominator, since otherwise the objective changes.

Same-state repair bundles teach when to repair, retain, and roll back; stage balancing gives these actions explicit supervision shares. For DiT, §8.2 adapts regional text, source/target masks, and reference conditions together. Thus SFT affects actual control execution, not just JSON formatting. Both effects require comparisons removing the corresponding SFT.

### 8.2 DiT flow matching

Let sample $b$'s packed target be $x_{0,b}\in\mathbb R^{1024\times128}$:

$$u_b\sim\mathcal N(0,1),\qquad\sigma_b=\operatorname{clip}(\operatorname{sigmoid}u_b,10^{-4},1-10^{-4}),$$
$$\epsilon_b\sim\mathcal N(0,I_D),\qquad D=1024\cdot128,$$
$$x_{\sigma,b}=(1-\sigma_b)x_{0,b}+\sigma_b\epsilon_b,$$
$$v_b^\star=\epsilon_b-x_{0,b}.$$

$\sigma=1$ is noise and $\sigma=0$ is data; velocity is the derivative with respect to $\sigma$. Spatial supervision is on the target grid, with $p$ denoting patches and $c$ the 128 channels:

$$w_{bp}=m_{bp}^{\mathrm{valid}}(1+3m_{bp}^{\mathrm{edit}}),$$
$$e_b=\frac{\sum_{p,c}w_{bp}(v_{\phi,bpc}-v_{bpc}^\star)^2}{128\sum_p w_{bp}},$$
$$\mathcal L_\phi^{\mathrm{FM}}=\frac{\sum_bm_b^{\mathrm{sample}}e_b}{\sum_bm_b^{\mathrm{sample}}}.$$

`valid` is area pooling of actual valid pixels; `edit` is area pooling of the union of trusted edit regions; `sample` handles sample-level padding. Slice away reference and source outputs directly; they do not contribute to stochastic dimension or target loss. Without reliable edit masks, edit = 0. Reject a real sample with $\sum_pw_{bp}=0$ rather than hiding bad data with epsilon. Compute $e_b$ only for samples with sample = 1. Dummies contribute structural zero; do not calculate 0/0 and then multiply by 0. $\sum_bm_b^{\mathrm{sample}}$ is the aggregated global valid-sample count. All ranks use that denominator and do not divide by world size again after summing gradients. Skip the update if the global valid count is 0.

Derive independent streams for $\sigma$, $\epsilon$, and text dropout from `H(sample_id,epoch,step,stream_tag)`. SFT uses single-branch prediction with whole-text-group dropout probability 0.1. Tencent's generic SFT training path does not use its evaluation path's sample loss mask. Therefore this project's `train/fm_loss.py` returns non-detached per-sample errors before reduction; detached auxiliary loss cannot be used for backward. [S6]

### 8.3 A directly constructible data example

The original request is: "The red cup on the wooden table is to the left of the blue book; change the text on the book to HELLO and preserve everything else." Public inputs are only the original image and this sentence. The Agent creates cup/book entities and separate source/target regions. The frozen public decomposer supplies CheckSpec's color, position, text, and preservation checks. Their IDs differ from hidden scoring IDs.

The first result fixes the text but turns the cup blue. The public image and color check narrow error candidates to cup controls/bindings and this render; they do not establish a unique root cause. From this real before view, the teacher proposes "repair only the cup" and "expand into the adjacent book region". The compiled condition hashes must differ. If one candidate yields a red cup while HELLO remains correct, it can supply before → DRAFT/APPLY and after → FINISH supervision. If the other candidate fixes the cup but degrades text, that after state supplies context only for SELECT of an old image or further repair; it cannot be labeled successful acceptance.

Both results come from the same before state, preserving their respective actual public observations. Hidden scoring determines only which target actions pass offline filtering; it never tells the Agent as input which candidate is best. A second seed still retains this blue-cup before image and only resamples candidate execution.


Source links embedded on source page 21:

- [Source link 1](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555)

**Source page 22**

This organization puts DAG revision, visual checks, preservation of previously correct items, and stopping decisions into one set of executable records, without inventing unique causal labels for natural failures.

## 9. Evaluators, tool interfaces, and reward reliability

### 9.1 Tool-output contracts

Public observations return `image_hash,check_id,value,known,verdict,evidence_ids,tool_version,frame,entity_matches`; private scoring uses independent eval IDs and storage. Observer accesses only CheckSpec; RewardService returns to the learner only after an episode ends. The same judge weights may be reused, but request queues, DTOs, cache namespaces, and log permissions are separate. `unknown` means content cannot be reliably judged; `infra_error` means the service did not finish. Do not combine them.

GroundingDINO's standard output is normalized cxcywh; convert once to canvas xyxy. Initial box/text thresholds are 0.35/0.25; add within-class NMS with IoU = 0.5. Raw detection scores are not correctness probabilities. SAM2 predictor image-feature state must be bound to the image hash; each concurrent request needs an independent instance or a serialization lock. DINOv2 ViT-B/14 supports appearance and protected-region evidence but does not replace identity ground truth. [S15]

Pin tool variants to GroundingDINO Swin-T OGC (`GroundingDINO_SwinT_OGC.py`, `groundingdino_swint_ogc.pth`), SAM2.1 Hiera-small (`configs/sam2.1/sam2.1_hiera_s.yaml`, `sam2.1_hiera_small.pt`), and DINOv2 `dinov2_vitb14` without register tokens. GroundingDINO's BERT dependency is pinned to `google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594`, with configuration pointing to a fixed local snapshot. Calculate SHA-256 for each weight at implementation download time; filenames or HTTP lengths are not verification substitutes.

Vision tools use a separate A6000 environment: Python 3.10, torch 2.6.0+cu124, torchvision 0.21.0, and Transformers 4.48.2. Compile the GroundingDINO extension using CUDA 12.4 toolkit and `TORCH_CUDA_ARCH_LIST=8.6`. In the project compatibility patch, change CUDA-source dispatch from `value.type()` to `value.scalar_type()`, and `.type().is_cuda()`/`.data<T>()` to `.is_cuda()`/`.data_ptr<T>()`. This pinned kernel dispatches only float/double, and Python has no BF16 handling branch. Thus detector models/inputs default to FP32 with autocast disabled. At startup, verify `_C` loading and CPU/GPU results; extension failure must not be treated as automatic fallback. [S15]

Install SAM with `SAM2_BUILD_CUDA=0`. `build_sam2(...,apply_postprocessing=True)` retains native stability fallback. Set predictor `mask_threshold=0,max_hole_area=0,max_sprinkle_area=0`, without calling optional connected-component CUDA postprocessing. Input is RGB uint8. Convert normalized xyxy box prompts to pixels by multiplying by canvas dimensions; use `multimask_output=False`. SAM may use BF16 forward, converting persistent masks to float32. Set `XFORMERS_DISABLED=1` before importing DINO; use ordinary dense batches, FP32, and native SDPA.

Detection retains official deterministic resizing: short side 800, long side ≤ 1,333, and ImageNet mean/std. Internal DINO inputs are uniformly 224 × 224. Resize the whole canvas proportionally with gray padding; crop entities by bbox before proportional resize/padding. Fix bicubic and ImageNet mean/std. CLS appearance features are 768-dimensional and L2-normalized. Protection comparisons use the 16 × 16 patch grid; apply the same transform and 14 × 14 area pooling to the protected mask. Nonfinite or zero-norm features are tool errors, not fabricated similarities. SAM retains its own processor; do not pass it normalized DINO/detection tensors directly.

By default, OCR handles only short English/numeric text explicitly annotated in the dataset. Use Tesseract 5.5.0, `tessdata_fast@4.1.0/eng.traineddata`, OEM = 1, and PSM = 11. Compare exact strings under Unicode NFC with task-specific case sensitivity; the first round does not include uncalibrated Chinese OCR tasks. OCR results include crop and confidence. Initially, confidence < 80 or invisible text means unknown; freeze the threshold through the same independent calibration process. Include tool-binary and traineddata hashes in the manifest. [S19]

### 9.2 Frozen 32B criteria and bounded second review

Qwen3-VL-32B-Instruct uses a fixed NF4 base, BF16 arithmetic, no-grad, and microbatch 1. Quantize only 448 large Linear modules in language layers, totaling 31,205,621,760 parameters; skip `model.visual` and `lm_head`, while embedding is naturally not converted. The official index gives 33,357,390,064 total parameters, of which 2,151,768,304 remain unquantized. Static memory is approximately 19.00 GiB; fix model, processor, and quantization state. It differs from the 27B Actor: 64 full-attention layers, hidden width 5,120, 64 Q/8 KV heads, head dimension 128, vocabulary 151,936, and DeepStack `[8,16,24]`. Use its own AutoProcessor and official forward rather than borrowing the 27B no-DeepStack path. [S23]

Input consists of the candidate image, necessary source/identity references (at most two images), and one requirement. Each image is limited to 262,144 pixels and total input to 8,192 tokens; route over-limit inputs during task admission instead of silently truncating them. With patch 16, temporal 2, and merge 2, a 512² image gives grid `[1,32,32]`, `pixel_values[1024,1536]`, and 256 visual tokens. Every check reads images and requirements, not the Agent's self-reported success.

The criterion does not generate a long explanation and parse it afterward. Compare conditional likelihoods of fixed answer sequences `Yes+im_end` and `No+im_end`. The real tokenizer may split answers into multiple tokens; save their actual IDs rather than assuming a single token:


Source links embedded on source page 22:

- [Source link 1](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)
- [Source link 2](https://github.com/tesseract-ocr/tesseract/tree/5.5.0)
- [Source link 3](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/1e6810acf5d8ebd189b4e75590877b64358e454c)

**Source page 23**

$$\ell_c=\sum_{k=1}^{|v_c|}\log p_\omega(v_{c,k}\mid h,v_{c,<k}),\qquad p_Y=\operatorname{sigmoid}(\ell_Y-\ell_N),\qquad c\in\{Y,N\}.$$

Do not average answers by length. EOS is this judge's own `im_end` (151645), distinct from the Actor's 248046. Calculate logits only at completion positions. Yes/No for the same question may be batched; record logical queries and physical forward counts separately.

The task portions of the two fixed English templates are below. Entity names, requirements, and references come from the corresponding CheckSpec/EvaluationSpec fields:

```text
Template A:
Inspect the attached image(s). The first image is the candidate.
Any additional image has the role stated below.
Request context: {original_request}
Image roles: {roles}
Check exactly this property: {atomic_property}
Does the candidate visibly satisfy the entire property?
Answer only Yes or No. Judge the image, not an assistant's claim.

Template B:
Compare the candidate with the stated requirement and any named reference.
Request context: {original_request}
Image roles: {roles}
Required property: {atomic_property}
Is this property clearly supported by what is visible in the candidate?
Answer only Yes or No. Do not infer a requested object is present merely
because it is mentioned in the requirement.
```

Execute A first to obtain $p_1$. Execute B only when $p_1\in[.2,.8]$, structured evidence conflicts, or the type is preregistered as difficult (exact counting, text, or cross-image identity). Put second-route rules and thresholds into configuration before calibration; do not change them opportunistically based on training gains. With A only, $q_j=p_1$; with B, $q_j=\min(p_1,p_2)$. $|p_1-p_2|>.25$, unresolved tool conflicts, or insufficient semantic evidence means unknown. Nonfinite values, timeouts, or device errors are infra_error, not content uncertainty. Calibrate final passing thresholds by type; .8 is only an initial candidate value.

Split 600 independent calibration tasks by family into 400 for threshold selection and 200 for confirmation, with two annotators and adjudication of disagreements. Each type needs at least 100 atomic annotations. Report precision/recall, ECE, Brier, and unknown rate, with task-clustered intervals. Among `{.50,.55,…,.95}`, choose the lowest threshold whose passing precision has a Wilson 95% lower bound ≥ .85. Do not reselect thresholds on the confirmation set. If a type fails admission, obtain more annotations or use an explicit structured route; its unreliable Yes must not still count as success. Calibrate the actual NF4 model plus complete routing, rather than calibrating BF16 and directly swapping in a quantized model.

Even when public feedback and private rewards share one frozen set of weights, their input criteria, IDs, queues, DTOs, cache namespaces, and response recipients must remain separate. CheckSpec comes from the frozen 32B decomposer reading the public request; evaluate its coverage, false positives, and hallucinated-constraint rate separately. Two phrasings are not two independent ground truths. Final effectiveness is also checked using official criteria and human evaluation. Calls total one to two queries per atom; see the budget in §17.

### 9.3 Geometry, counting, and references

After reliable detection and identity matching, the soft counting score is $\exp(-|n-k|)$ and exact passing requires $n=k$. Left/right relations use center coordinates with tolerance 0.02 of canvas width. Containment uses intersection divided by the contained box's area, with passing threshold 0.9. Use layout IoU only when **the user or dataset annotations explicitly supply target boxes**; the Agent's self-selected bbox cannot be reward ground truth.

Conflicts between VLM and geometry, missed detections, or unclear identity mean unknown; "not detected" is not proof that an object is absent. Appearance requirements check both DINO similarity on entity crops and semantic consistency with references. Select similarity thresholds on a calibration set for that requirement type; a fixed cosine is not an identity guarantee. Protected regions come from the original task and cannot be shrunk by the Agent.

## 10. Return definition and edge cases

**Every requirement, score, CS, and protection term in this section is a private terminal-evaluation quantity, not within-episode feedback.** Only the image actually delivered and the original source enter the default return. Hidden scoring of intermediate images requires additional offline work and separate cost reporting.

Let the fixed requirement count be $J\ge1$, valid judgments $o_j\in\{0,1\}$, passing thresholds $\tau_j$, and soft scores $q_j\in[0,1]$:

$$\widetilde q_j=o_jq_j,\qquad p_j=o_j\mathbf1[q_j\ge\tau_j],\qquad\mathrm{CS}=\prod_{j=1}^{J}p_j.$$

For types with an exact discrete passing condition, $p_j$ is the conjunction of that condition and known; soft scores are still used for dense feedback.


**Source page 24**

$$
Q_{\mathrm{dense}}=0.7\frac{1}{J}\sum_j\tilde q_j+0.3\min_j\tilde q_j.
$$

The mean plus minimum retains the signal from improvements to other requirements when one item is still zero; complete success is represented separately by CS. The geometric mean, in which one zero item makes the entire dense term zero, is not retained.

For editing tasks, $I_0$ is the original source canvas, not the first render. $P_0$ is the set of requirements that the original image already passes and that this task requires preserving:

$$
\mathrm{reg}=\frac{1}{|P_0|}\sum_{j\in P_0}[p_j(I_0)-p_j(I_T)]_+.
$$

When $P_0$ is empty, reg = 0. The protection mask is $M=m^{\mathrm{protect}}m^{\mathrm{valid}}$:

$$
d_{\mathrm{pix}}=\frac{\sum_{x,y,c}M_{xy}|I_{T,xyc}-I_{0,xyc}|}{3\sum_{x,y}M_{xy}},
$$

$$
\mathrm{keep}=\frac12 e^{-d_{\mathrm{pix}}/0.05}+\frac12\operatorname{mean}_{p\in\Omega}\frac{1+\cos(f_p(I_T),f_p(I_0))}{2}.
$$

$\Omega$ consists of DINO patches with protection coverage $\geq0.5$. If valid protected pixels exist but $\Omega$ is empty, keep uses only the pixel term, without multiplying it by $1/2$. If there is no protected region, keep = 1. Features are computed after fixed resizing/padding, with the mask transformed identically; the latent's $32\times32$ grid must not directly index DINO's grid.

The logical cost is

$$
\tilde C=\frac{n_{\mathrm{render}}+0.1n_{\mathrm{inspect}}+0.01n_{\mathrm{action}}}{3+0.1\cdot2+0.01\cdot10}\in[0,1].
$$

Cache hits are still charged as logical requests. Actual cache savings are reported separately as a system metric, so sibling branches do not affect one another's rewards through a shared cache. FINISH gives no additional reward or penalty for self-reporting success. The return for normal delivery is

$$
R=\mathrm{CS}+0.3Q_{\mathrm{dense}}-0.15\mathrm{reg}-0.05(1-\mathrm{keep})-0.1\tilde C-0.02n_{\mathrm{invalid}}.
$$

A normal episode contains at most one invalid action, so the total negative terms are bounded above by $P_{\max}=0.32$. Thus every completely successful episode has $R\geq0.68$, while every episode without complete success has $R\leq0.3$: success strictly dominates. When weights change, the startup check must reverify $1-P_{\max}>w_{\mathrm{dense}}$ (currently $w_{\mathrm{dense}}=0.3$). Two invalid actions, an explicit `FINISH(null)`, or the absence of a deliverable image all receive a return of $-1$, with no additional cost deduction. Therefore the overall range is $R\in[-1,1.3]$.

Unknown sets the corresponding $\tilde q_j,p_j$ to zero without removing them from the denominator; holding everything else fixed, it cannot increase the task score. When infrastructure prevents an image or score from being obtained, the entire group remains incomplete and is not entered into training as unknown or $-1$. Optimization can still exploit evaluators' content blind spots. Independent audits and human evaluation must monitor this; the bounds above alone do not establish that the reward cannot be exploited.

The original ACCEPT and STOP differed only in their failure penalties for the same image, so STOP weakly dominated ACCEPT. v3 uses a single FINISH to remove this degree of freedom, which does not help decision-making. The return encourages complete satisfaction, preservation of already correct content, and restraint in execution cost. The termination action itself has no extra language-label reward. The main path does not add a potential-function term that would cancel in episode-level RLOO.

## 11. Behavior Recording and Joint Probability

**Source page 25**

### 11.1 Token, Image, and Version Records

For every action, save `raw_prompt_ids`, `raw_completion_ids`, `behavior_log_probs`, `response_mask`, `finish_reason`, `view_hash`, and the processor's actual `pixel_values`, `image_grid_thw`, `attention_mask`, and `mm_token_type_ids`, together with the model/processor revision. The RL mask covers every action token, rationale, invalid content, and EOS actually sampled by the model. Environment tokens, input-prefix image placeholders, padding, and manually appended EOS have mask 0. Special visual tokens actually sampled in the completion still contribute to the score. Actual tokens at the length limit retain mask 1.

The 27B worker saves the actual processor outputs for this call, template switches, the quantized base hash, and the position-generation rules. Inputs with images must supply `mm_token_type_ids`; text type 0 is appended when the action is concatenated. Each action is sampled using a fresh FP32-recurrent hybrid cache. Replay sets `use_cache=False,past_key_values=None`, computing positions and differentiable chunk GDN from the complete input. Do not detach the prefix and train only the suffix. Reference and current policies use the same inputs/positions; see §6 for the specific cache repair. The 27B has no DeepStack; the 32B judge's DeepStack remains handled by its own forward.

Fix the template settings to `enable_thinking=False,preserve_thinking=False,tools=None`. An empty think block is part of the prefix and has mask 0; all raw sampled content, including incorrect JSON and EOS, is a policy event. Explicitly set sampling to temperature 1, top_p 1, top_k 0, min_p 0, and repetition_penalty 1, disabling logits filters not included in the score. SFT uses `im_end`; RL scores both actual EOS types. PAD is excluded by position and length masks.

Split the processor's multi-image `pixel_values` by actual grid lengths into immutable per-image tensor artifacts. The view records ordered artifact references and concatenation offsets, avoiding repeated storage of the same source image for every action. Replay reconstructs these saved tensors and verifies the concatenation hash; it must not retrieve images again from a changed DAG. Retain the full current stage, and use the same audit-retention policy for stage archives and latent chains.

For each render, save:

```text
render_id, task/root/branch IDs, behavior_version, bundle_hash
z_fp32[T+1,1024,128], sigma_fp32[T+1], transition_std[T]
seed_init, seed_steps[T], stored_mean_logp[T]
probe_mean_fp32[selected_steps]?, final_image_id
compiler/encoder/judge/backend versions, status, logical_cost
```

Record `task_id`, `root_id`, and `branch_id` each round; do not conflate them. Node IDs visible to the Agent are stable aliases local to the branch. Globally unique IDs are used only for internal provenance tracking, preventing scheduling order from leaking information about sibling branches.

### 11.1.1 Visual Prefix Assembly and Complete Replay of Exceptional Actions

A full-support policy can sample special tokens such as `image_token_id=248056` in its completion. The native multimodal forward's `get_placeholder_mask` counts image placeholders across the entire ID sequence and does not read `mm_token_type_ids`. An image prefix followed by such a completion causes a feature-count mismatch. Cached decoding stops passing `pixel_values` after its first iteration, so the sampling event itself can occur. This must not cause the trajectory to be discarded, tokens to be silently deleted, or the policy support to be changed. S3

Replay consistently executes the frozen input-assembly portion of the official forward explicitly:

1. When generating a behavior view, save the actual `prefix_image_positions`, prefix length, image order, and merger feature offsets/hash. Only these slots are images. The number of prefix slots must equal the number of image features; do not search the whole completion during replay.
2. Use the same frozen embedding to construct `[1,L,5120]` BF16 `inputs_embeds` for the full prefix plus completion. Scatter frozen merger features only into the saved prefix slots. Every completion token retains its ordinary embedding and type 0.
3. Call `hf.model.get_rope_index` using the saved IDs, grid, attention mask, and appended type 0 to obtain `[3,1,L]` positions. With no image, all three axes use ordinary sequential positions. Concatenate ordinary text positions `[1,1,L]` to obtain `[4,1,L]`, shared by reference/current. Every prefix and completion position must align with actual cached generation.
4. Set `hf=peft_model.get_base_model()`. Call `hf.model.language_model(inputs_embeds=..., position_ids=..., attention_mask=..., past_key_values=None, use_cache=False, return_dict=True)` to obtain the last-layer hidden states. Language LoRA and checkpointing remain inside this module, preserving gradients through the complete prefix.
5. Following §14, compute reference no-grad first, then the full policy graph; chunk frozen-head positions and perform one decoder VJP. Do not rematerialize all layers' hidden states.

The pinned source's multimodal forward ultimately calls `language_model` this way as well; see S3 lines 1564-1606. TextModel distinguishes text positions from 3D positions among the four axes at lines 1173-1197. This is an explicit decomposition of the same official computation. Vision/embedding computations may be frozen and cached; language activations cannot replace differentiable replay.

JSON parsing uses decoding with `skip_special_tokens=False`. Remove one actual terminal EOS only according to the true sampling boundary; do not silently delete other special tokens inside the completion to repair JSON. Add fixed acceptance cases with an image prefix and completion containing special image/video/start/end IDs: replay must complete, every raw score must be preserved, and the protocol may classify the action as invalid. For ordinary 0/1/4-image paths, compare hidden states/logp for equivalence with the native multimodal implementation.

**Source page 26**

### 11.2 Probability Events and Objective

Let the task be $\xi$, the complete trajectory $\tau$, the action the raw sampled token sequence $a_t$, and the actual condition for render $e$ be $c_e$. With compiler, tools, and evaluators fixed:

$$
p_{\theta,\phi}(\tau\mid\xi)=p_{\mathrm{env}}(\tau\mid\xi)\prod_t\pi_\theta(a_t\mid s_t)\cdot\prod_{e\in\tau}\prod_{k=0}^{T-1}q_\phi(z_{e,k+1}\mid z_{e,k},c_e),
$$

$$
J(\theta,\phi)=\mathbb E_{\xi\sim\mathcal D,\tau\sim p_{\theta,\phi}}[R(\tau)].
$$

$p_{\mathrm{env}}$ includes the initial standard normal and deterministic state transitions and contains no trainable parameters directly. If tools are stochastic, they use fixed, independent RNGs and are included in the environment. An action's probability is the product of the conditional probabilities of all its sampled tokens.

$$
\nabla_\theta J=\mathbb E\left[R\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right],
\qquad
\nabla_\phi J=\mathbb E\left[R\sum_{e,k}\nabla_\phi\log q_\phi(z_{e,k+1}\mid z_{e,k},c_e)\right].
$$

Given the action, $c_e$ no longer depends directly on $\theta$; the controller's influence on condition selection is already captured in action probabilities. Every actual render contributes to the score, including abandoned images, because it can affect subsequent repair and stopping. Unexecuted candidates have no generation score. For a valid zero-render editing trajectory, the $\phi$ sum is empty, but its return still enters the branch baseline.

Default sampling uses temperature = 1, top-p = 1, and top-k = 0, without grammar truncation. If constrained decoding is used later, save the allowed set at every position and renormalize probabilities over the actual set; original full-softmax probabilities cannot replace behavior probabilities.

## 12. Complete Flow-SDE Mathematics and Implementation

### 12.1 One Sigma Schedule

The number of tokens in the noise segment is $L=HW/256=1024$, and the number of steps is $T=50$. Following the pinned official implementation, define

$$
m_{200}=0.00016927L+0.45666666,
\qquad m_{10}=0.0000873809524L+1.89833333,
$$

$$
\mu_{\mathrm{shift}}=\begin{cases}
m_{10}+\frac{T-10}{190}(m_{200}-m_{10}),&L\leq4300,\\
m_{200},&L>4300,
\end{cases}
$$

$$
t_k=1-k/T,\quad k=0,\ldots,T-1,
\qquad
\sigma_k=\frac{e^{\mu_{\mathrm{shift}}}}{e^{\mu_{\mathrm{shift}}}+1/t_k-1},\quad\sigma_T=0.
$$

This is an exponential shift; $\mu_{\mathrm{shift}}$ is not a Gaussian mean. Compute the schedule using only the number of target noise tokens; do not add text/source/appearance lengths to $L$. Construct it once and save the complete float32 array and hash. Sampling, current replay, and reference share it rather than recomputing separately. S4

### 12.2 Transition Kernel

Flatten `[1024,128]` to $D=131072$ dimensions. At every step:

**Source page 27**

$$
\Delta_k=\sigma_{k+1}-\sigma_k<0,
\qquad
\tilde\sigma_k=\begin{cases}\sigma_1,&k=0,\\\sigma_k,&k>0,\end{cases}
$$

$$
d_k=\eta\sqrt{\frac{\sigma_k}{1-\tilde\sigma_k}},\qquad s_k=d_k\sqrt{-\Delta_k},\qquad\eta=0.3,
$$

$$
a_k=1+\frac{d_k^2\Delta_k}{2\sigma_k},\qquad
b_k=\left(1+\frac{d_k^2(1-\sigma_k)}{2\sigma_k}\right)\Delta_k,
$$

$$
\mu_{\phi,k}=a_kz_k+b_kv_\phi^{\mathrm{cfg}}(z_k,\sigma_k,c_e),\qquad
q_\phi(\cdot\mid z_k,c_e)=\mathcal N(\mu_{\phi,k},s_k^2I_D),
$$

$$
z_0\sim\mathcal N(0,I_D),\qquad z_{k+1}=\mu_{\phi_{\mathrm{old}},k}+s_k\epsilon_k,\qquad\epsilon_k\sim\mathcal N(0,I_D).
$$

$d_k$ is the diffusion coefficient; $s_k$ is the actual one-step standard deviation. All primary states, means, sampling, logp, and KL use FP32; only DiT inputs and forward computation use BF16. Convert model outputs to FP32 before CFG mixing. Store the FP32 chain $z_0,\ldots,z_T$; derive actual BF16 inputs using the same cast.

Ordinary arithmetic gives $\mu_{\mathrm{shift}}\approx1.7019562073$, $\sigma_1\approx0.9962928450$, $\sigma_{49}\approx0.1006643969$, $s_0=0.3$, and $s_{49}\approx0.0318446337$. The final step still has positive variance; this is not a conclusion about quality.

### 12.3 Density and Step Subsampling

The total log density and its per-dimension mean are

$$
\ell_{\phi,e,k}=-\frac12\sum_{d=1}^D\left[\frac{(\operatorname{sg}(z_{e,k+1,d})-\mu_{\phi,e,k,d})^2}{s_k^2}+2\log s_k+\log(2\pi)\right],
\qquad\bar\ell_{\phi,e,k}=\ell_{\phi,e,k}/D.
$$

sg denotes stop-gradient. During replay, the current/next states, conditions, sigma, and standard deviation are all fixed; differentiation applies only to the LoRA of the current velocity. Reference tokens are not random actions. Configuration checks require every $s_k>10^{-8}$ and finite; silently dropping steps during execution is prohibited.

By default all 50 steps are stochastic, and 51 states are saved. During learning, an independent RNG samples one step from each of the four time strata in §13.4. `stochastic_indices=range(T)` and `train_indices` are separate fields. Tencent's `sde_indices` controls both noise injection and saving; pass all steps, not the four replay steps. S6

An all-step SDE makes the gradient definition of this scheme, with black-box reward, per-step score, and a detached entire chain, complete. A deterministic suffix using the same trainable parameters would omit part of those parameters' influence if its path derivative were neither computed nor reflected in a changed policy definition. Windowed SDE can serve as another surrogate but cannot be silently mixed into this objective. ODE with $\eta=0$ is used only as a separate inference comparison; do not fabricate a Gaussian logp for it.

### 12.4 Sampler Quality Gate: 50 Steps by Default; 20/28 Steps Pending Validation

CPU recomputation using the same source and $\eta=0.3$ gives:

| T | Shift $\mu$ | Last nonzero $\sigma$ | Actual final-step standard deviation |
|---|---:|---:|---:|
| 20 | 1.91634762 | 0.26345430 | 0.09209302 |
| 28 | 1.85917658 | 0.19206305 | 0.06410270 |
| 50 | 1.70195621 | 0.10066440 | 0.03184463 |

Final noise injection at 20 steps is approximately 2.89 times that at 50 steps. Mathematical completeness does not imply identical finite-step image quality. Nor may the last step be silently changed to ODE while retaining the current Gaussian objective. S4 S9

After DiT SFT, fix 64 development tasks/control conditions covering 0/1/2 regions, generation/editing, and references, with two predetermined exogenous seeds per task,

**Source page 28**

and compare 50-step ODE against 50/28/20-step all-SDE sampling (CFG 4, $\eta=.3$), totaling 512 renders. Keep conditions and tasks identical. Different step counts inherently define different transition chains; align only initial seeds and exogenous random rules where definable, without claiming step-by-step noise equivalence.

First examine 50-step SDE relative to ODE for task completion, protection, and blinded visual quality. Initial acceptance thresholds are a success-rate point difference no worse than -5 percentage points and an increase in severe visual defects no greater than 5 percentage points, with paired confidence intervals also reported. If intervals cannot exclude a larger degradation, extend verification to independent development tasks rather than treating a nonsignificant difference as equivalence. If the default fails the quality gate, test 50 steps at $\eta=.15$, then 80 steps at $\eta=.15$, each with 128 renders. The six configurations require at most 768 initial-screening renders. Additional independent verification has a separate budget.

20/28 steps replace the default only after passing the same quality gate against an already qualified 50-step configuration and improving measured end-to-end throughput. Retain 50 steps when uncertain. Once selected, lock the sampler ID: repair SFT, Agent RL, joint RL, the main test, and every direct comparison use it. Report 50-step ODE separately as an additional inference result; do not silently change samplers only for the main test. If no usable all-SDE configuration exists, repair sampling numerics/condition adaptation and repeat this gate before beginning large-scale data collection with an unvalidated short chain.

This version reruns the sampler gate on 9B; a pass obtained with 4B images cannot be reused. The 50-step sigma values remain the same because the 512 grid still has $L=1024$, not because the two models have identical image quality. The gate also compares native global conditioning, regional conditioning, and source/reference editing on the model after DiT SFT, ruling out condition-compilation degradation before repair SFT/RL.

## 13. Hierarchical Sampling, Credit Assignment, Losses, and Regularization

### 13.1 Sampling and Independence

Each round uses $P=6,N=2,M=2$. Let $p$ index tasks, $n$ root plans, and $m$ suffixes. First sample $N$ root actions independently, then fork $M$ suffixes for each root action; the first render occurs after the fork. Keep coincidentally identical roots without deduplication or replacement sampling.

Startup assertions require $P\geq1,N\geq2,M\geq2,T\geq4,1\leq K_s\leq T$, and all budgets and token caps to be positive integers. Reject this RLOO configuration if N or M is 1 rather than silently substituting a zero baseline. If a small batch is incomplete, defer collection until a full group is available; do not complete an update by reducing group size.

Each suffix has independent controller, initial-noise, and SDE streams:

$$
\mathrm{seed}_{\mathrm{ctrl}}=H(\mathrm{run},\mathrm{round},p,n,m,t,\mathrm{ctrl}),
$$
$$
\mathrm{seed}_{\mathrm{init}}=H(\mathrm{run},\mathrm{round},p,n,m,e,\mathrm{init}),
$$
$$
\mathrm{seed}_{\mathrm{SDE}}=H(\mathrm{run},\mathrm{round},p,n,m,e,k,\mathrm{sde}).
$$

The root-action RNG does not include $m$; training step selection uses a separate `step_subset` tag. Truncate the hash to an unsigned 64-bit seed and also save the original tuple. Tools do not use a global RNG. Sibling branches have independent visible IDs, budgets, candidates, image pointers, and reward costs. Shared read-only encoding caches expose neither cache hits nor other branches' queuing delays.

### 13.2 Hierarchical RLOO

$$
\bar R_{pn}=\frac1M\sum_mR_{pnm},
$$
$$
A^0_{pn}=\bar R_{pn}-\frac1{N-1}\sum_{n'\neq n}\bar R_{pn'},
\qquad
A^+_{pnm}=R_{pnm}-\frac1{M-1}\sum_{m'\neq m}R_{pnm'}.
$$

The root baseline is independent of the current root action. Given the root action, the sibling-suffix baseline is independent of the current suffix. Thus $\mathbb E[b\nabla\log p]=0$. Use $A^0$ for the root and $A^+$ for every suffix action and every render. Since $\sum_m A^+_{pnm}=0$, applying suffix advantages to the shared root would cancel its gradient. This is a limitation of this estimator, not a claim that all shared-prefix methods are untrainable.

Do not divide by the group standard deviation, clip advantages, or filter failures or zero returns. Given the overall return range, advantages lie in $[-2.3,2.3]$; when $M=2$, they are simply the difference between the two returns. Constant-return groups stay in the denominator, with PG naturally equal to zero.

**Source page 29**

### 13.3 Controller PG and KL

Let $s^0_{pn}$ be the sum of logp over all policy tokens in the root action and $s^+_{pnm}$ the sum of logp over subsequent actions in that branch. Fix $C_\theta=10\times1024=10240$:

$$
\mathcal L_{\theta,\mathrm{PG}}=-\frac1{PNC_\theta}\sum_{p,n}\operatorname{sg}(A^0_{pn})s^0_{pn}-\frac1{PNMC_\theta}\sum_{p,n,m}\operatorname{sg}(A^+_{pnm})s^+_{pnm}.
$$

Count each root once; do not average by actual output length. The full-vocabulary KL at each position is

$$
K_\theta(h)=\sum_{v\in\mathcal V}\pi_\theta(v\mid h)\log\frac{\pi_\theta(v\mid h)}{\pi_{\mathrm{ref}}(v\mid h)}.
$$

Let the root and suffix policy-token sets be $\mathcal T^0,\mathcal T^+$. Aggregation must explicitly be

$$
\mathcal K_\theta=\frac1{PNC_\theta}\sum_{p,n}\sum_{\ell\in\mathcal T^0_{pn}}K_\theta(h_\ell)+\frac1{PNMC_\theta}\sum_{p,n,m}\sum_{\ell\in\mathcal T^+_{pnm}}K_\theta(h_\ell).
$$

KL is not multiplied by advantage. Normalize over the complete vocabulary; do not softmax vocabulary chunks separately. From a full decoder forward, take only last-layer hidden states at the predecessor positions of target tokens. Following §14.2, project the complete vocabulary in chunks of 32 positions, then send the hidden-state gradient back through the decoder once. This does not mean materializing logits for all actions and then slicing them. The decoder uses non-reentrant checkpointing. S3

### 13.4 Renderer PG and KL Covering Four Time Strata

All T steps of every actual render are stochastic policy actions; sample $K_s=4$ steps only for backward computation. Partition $0,\ldots,T-1$ into four contiguous, disjoint time strata $\mathcal S_h$. Let $q=\lfloor T/K_s\rfloor,r=T\bmod K_s$; the first r strata have length q+1 and the rest length q. For T = 50, the strata are `[0,13)`, `[13,26)`, `[26,38)`, `[38,50)`, with lengths 13, 13, 12, and 12.

Using an independent replay RNG, sample one $k_{e,h}$ uniformly from each stratum, with weight $\omega_h=|\mathcal S_h|/T$. Save `stratum_id,step_index,sampling_probability=1/size,weight`. Sampling does not inspect advantages, scores, or gradients; online siblings remain independent.

$$
\mathcal L_{\phi,\mathrm{PG}}=-\frac1{PNMB_{\max}}\sum_{p,n,m}\operatorname{sg}(A^+_{pnm})\sum_{e\in\tau_{pnm}}\sum_{h=1}^{K_s}\omega_h\bar\ell_{\phi,e,k_{e,h}},\qquad B_{\max}=3.
$$

$$
\mathbb E\left[\sum_h\omega_h\nabla\bar\ell_{k_h}\right]=\sum_h\frac{|\mathcal S_h|}{T}\frac1{|\mathcal S_h|}\sum_{k\in\mathcal S_h}\nabla\bar\ell_k=\frac1T\sum_{k=0}^{T-1}\nabla\bar\ell_k.
$$

This remains a fixed $1/(B_{\max}TD)$ scaling of the original full-chain $\phi$ gradient. At T = 50, weights are .26/.26/.24/.24, not uniformly 1/4. Do not additionally divide by Ks, T, or D, or average by the actual render count. The motivation is to cover early, middle, and late portions with limited replay, not to claim exclusive semantic roles for particular time intervals or guaranteed variance reduction for every gradient distribution.

The per-dimension reference KL between equal-variance Gaussians is

$$
\bar K_{\phi,e,k}=\frac{\|\mu_{\phi,e,k}-\operatorname{sg}(\mu_{\mathrm{ref},e,k})\|_2^2}{2Ds_k^2},
\qquad
\mathcal K_\phi=\frac1{PNMB_{\max}}\sum_{p,n,m}\sum_{e\in\tau_{pnm}}\sum_h\omega_h\bar K_{\phi,e,k_{e,h}}.
$$

The denominator uses $s_k^2=d_k^2(-\Delta_k)$. The inspected Flow-GRPO Kontext path omits the $-\Delta_k$ factor from the complete transition variance; repair it according to the analytic expression. S9

**Source page 30**

Each batch is updated only once, using the score loss directly, so no PPO ratio is needed. $\exp(\bar\ell_\phi-\bar\ell_{\mathrm{old}})$ is the D-th root of the complete Gaussian density ratio and cannot be treated as the original ratio. Stored logp is used for replay consistency. Zero-advantage groups remain in the denominator: PG is naturally zero, while fixed-reference KL is still computed as defined. If the entire batch has no renders, $\phi$ has no PG/KL. Explicitly skip that optimizer step and record `no_renderer_action`; do not silently use an FM update and present it as image RL. $\phi$ is frozen during the Agent-only stage.

### 13.5 A Simple Main Loss and Fixed References

The main RL losses are

$$
\mathcal L_\theta=\mathcal L_{\theta,\mathrm{PG}}+.02\mathcal K_\theta,\qquad
\mathcal L_\phi=\mathcal L_{\phi,\mathrm{PG}}+.01\mathcal K_\phi.
$$

Each reference is an independent frozen adapter from the end of its corresponding SFT and is not refreshed during training. The two KL terms provide local stabilization at currently sampled states; they are not claimed to include the complete regularization gradient accounting for changes in the new policy's occupancy distribution. The main path has no CE/FM replay, whole-loss $\kappa$ scaling, group-standard-deviation scaling, or eight-batch coefficient calibration.

This is an explicit choice: SFT first establishes protocol and condition-execution competence, RL adjusts the policies according to actual task returns, and fixed references control deviation. If the development set confirms forgetting of protocol or control capabilities, introduce a separately named ablation/recovery configuration with fixed supervised replay and the same total training budget. Do not quietly add supervised terms to the main experiment and attribute all gains to RL.

The default $\theta$ LR is $5\times10^{-6}$ and $\phi$ LR is $3\times10^{-5}$. Compare the latter with $10^{-4}$ in two ten-round pilots following §15. Use AdamW, FP32 parameters and states, epsilon $10^{-8}$, weight decay 0, and global gradient clipping at 1. Gradient clipping protects against extreme batches; it need not trigger every round. Nor can learning effectiveness be inferred by claiming parameter changes cannot exceed the LR.

Each round records the advantage distribution, fraction of groups with nonzero PG, pre-clipping norm, clipping fraction, effective $BA$ changes, reference KL, and task metrics. In rounds 1-5 and every 25 rounds thereafter, separately measure PG and KL gradients and output responses on a fixed small probe; do not copy three parameter sets for the entire batch every round. Measure output response through old-to-new action distributions or normalized velocity changes, comparing them with same-parameter replay differences. A response above numerical differences does not imply a task gain; actual effectiveness must still be assessed through frozen-side comparisons and development learning curves.

If five consecutive nonzero-PG updates yield no distinguishable output change, check LoRA coverage, correct masks, score signs, the FP32 path, and whether Adam epsilon dominates, in that order, before running a new pilot. If necessary, test epsilon $10^{-12}$ separately and register the configuration. Increasing KL to an arbitrary target magnitude does not prove learning, particularly because at D = 131072 a per-dimension KL of $10^{-3}$ already corresponds to a total KL of about 131 for one transition. If there are no advantage differences anywhere in the batch, first check evaluator criteria and task coverage, then add compositional tasks using predefined difficulty buckets. Keep the original zero-advantage groups.

### 13.6 Intuition and Examples for Credit Assignment

Two root plans for the same task each have two independent suffixes. For returns `(1.1,.2)` and `(.9,.1)`, the root means are .65 and .50, producing root advantages +.15 and -.15. The within-root suffix advantages are respectively `(+.9,-.9)` and `(+.8,-.8)`. Root plans learn from average subsequent executability. Every inspection, selection, repair, and termination action in a suffix, together with every render it actually performs, is compared against the independent alternative outcome under that root.

This does not precisely assign responsibility for a bad pixel to one token, nor fabricate causal rewards from graph distance. Error-node localization organizes possible revisions onto the correct dependency branch; complete-trajectory return evaluates whether the request is ultimately met. The two sides share returns and synchronize sampling and updates while preserving their own actual probabilities and gradient boundaries. Four time strata only organize renderer backward coverage; they do not permit unexecuted candidates to enter policy probabilities.

## 14. Equivalent Low-Memory Backward Computation

### 14.1 Sequential CFG VJP for 9B

For a selected transition, let the time-stratum coefficient be $c=\omega_h/(PNMB_{\max})$, $A=A^+_{pnm}$, $z'=z_{e,k+1}$, and $\beta=\beta_\phi$. Its loss is $c[-A\bar\ell+\beta\bar K]$, with transition mean $\mu=a_kz+b_kv^{\mathrm{cfg}}$.

First obtain the current positive/negative velocities and the reference mean with no-grad, then compute

$$
g=\operatorname{sg}\left[\frac{cb_k}{Ds_k^2}\{A(\mu-z')+\beta(\mu-\mu_{\mathrm{ref}})\}\right]\in\mathbb R^D.
$$

Then build and backpropagate the graphs sequentially:

$$
L_+=\sum_dw g_dv^+_{\phi,d},\qquad L_-=\sum_d(1-w)g_dv^-_{\phi,d}.
$$

**Source page 31**

The sum of the two gradients equals the first derivative of the original loss. Use a sum for the inner product, not another mean. At $w=4$, the negative branch must backpropagate with coefficient -3. Fix the order to reference no-grad → both current no-grad branches → current positive forward/backward/release → current negative forward/backward/release. Do not step, zero gradients, or switch adapters between the two backward passes.

This method adds two current no-grad forwards in exchange for retaining activations for only one CFG branch at a time. Log the actual Gaussian loss, not the VJP inner-product value as if it were the original loss. The initial implementation must be compared against simultaneous two-branch autograd on small shapes. If the maximum shape permits direct simultaneous branches, that implementation may also be used, retaining the same mathematical objective and a separate configuration record.

### 14.2 Full-Vocabulary KL and Head Chunking for 27B

The vocabulary has 248320 entries; per-position KL cannot be replaced by softmax over a vocabulary subset. Because `lm_head` is frozen and has no LoRA, backward computation through the head and decoder can be separated exactly:

1. Switch to reference. Under no-grad, assemble the frozen visual prefix as in §11.1.1 and call the underlying `model.language_model` to obtain last-layer hidden states, saving only $H_{\mathrm{ref}}\in\mathbb R^{K\times5120}$ at action-prediction positions. Do not request `hidden_states` for every layer.
2. Switch to policy and restore training mode/the freezing allowlist. Using the same input assembly, run `language_model` on the complete prefix plus action to obtain $H_\theta$, retaining the decoder computation graph. Use microbatch 1 and $K\leq1024$.
3. Create a gradient-enabled copy $\tilde H=\operatorname{detach}(H_\theta)$. For each 32 action positions, use the same frozen BF16 `lm_head` to compute policy/reference logits; compute log-softmax, full-vocabulary KL, and those positions' PG in FP32. Backpropagate each block to $\tilde H$ and immediately release large vocabulary tensors.
4. Finally perform one vector-product backward: $H_\theta.\mathrm{backward}(\operatorname{detach}(\partial L/\partial\tilde H))$, updating LoRA through the complete decoder graph. The prefix's influence is still propagated through that graph; the prefix is not detached.

Let $z_i=W_{\mathrm{head}}H_i$, $p_i=\operatorname{softmax}(z_i)$, and let $q_i$ be the reference distribution. Then

$$
K_i=\sum_vp_{iv}(\log p_{iv}-\log q_{iv}),\qquad
\frac{\partial K_i}{\partial z_{iv}}=p_{iv}[(\log p_{iv}-\log q_{iv})-K_i].
$$

Chunking only positions while retaining the entire vocabulary therefore gives the same gradient as full KL. Each position's coefficient directly uses the root/suffix denominators and advantages from §13, without another average over K. PG, KL, and CE can all use automatic differentiation for head blocks; a hand-coded softmax derivative is unnecessary. The expression above is for verification.

For prefix length $q$ and zero-indexed action token $j$, the prediction position is $q+j-1$ and its label is at concatenated-sequence position $q+j$. Actual EOS in the completion is scored normally; remove padding only according to the actual response mask. Reference hidden states occupy at most 10 MiB, and a 32-position FP32 logits block about 30.31 MiB, so the complete reference vocabulary matrix need not be stored. Do not switch adapters while the live policy graph exists. Reference projection through the shared frozen head requires no switch back to reference.

### 14.3 Backend and Fallback for 9B Regional Attention

The main path directly calls the pinned Diffusers API `transformer.set_attention_backend("_native_efficient")`, retains both native Flux2 processors, and passes `joint_attention_kwargs={"attention_mask": mask}`. This internally forces PyTorch `EFFICIENT_ATTENTION`; verify that the kernel is actually used at runtime. S25

This choice has a specific source-code basis: the pinned PyTorch native Flash branch rejects a nonempty `attn_mask`, while hardware checks for the memory-efficient branch allow sm50-sm121, including the 5090's sm120. It is insufficient to write only "use FlashAttention to save memory," and automatic selection must not silently fall back to full $S^2$ math. At the largest 512 layout, one 32-head FP32 score tensor is 2.53125 GiB; multiple intermediates can exceed 9B capacity. S25

If the target wheel's efficient kernel cannot handle the working set, first repair the installation/build. If limitations remain, enable the defined equivalent fallback: chunk only query rows in blocks of 128, read all K/V, and apply non-reentrant checkpointing to every block:

$$
O_{a:b}=\operatorname{softmax}(Q_{a:b}K^\top/\sqrt{d_h}+M_{a:b,:})V.
$$

Do not also truncate K and thereby change softmax normalization. Convert the bool mask inside the operator to 0 for allowed entries and $-\infty$ for forbidden entries; every row has at least one allowed entry. In the fallback, Q/K scoring, softmax, and multiplication of probabilities by V use FP32; convert V to FP32 before multiplication and cast the output back to BF16. The main efficient path uses the library's dtype handling. The fallback preserves conditions and the objective but may be slower; report measured speed. If memory pressure remains, use training-compatible saved-tensor CPU offload rather than presenting inference-only pipeline offload as training offload. Record every replacement before starting a complete stage and rerun the sampling/replay consistency gate.

**Source page 32**

## 15. Training Stages, Sampling Updates, and Consistent Commits

### 15.1 Stage Dependencies and Initial Budgets

| Stage | Parameters, quantities, and initial LR | Entry conditions |
|---|---|---|
| Source/environment/evaluator gates | No training | Pass the structure, state, gradient, probability, and resource gates in §19; complete evaluator calibration |
| Linked data | Target 8,000 pairs, including 4,000 shared PLAN records | Public-only PLAN and the same bundle pass retrospective compatibility filtering against the paired target |
| DiT SFT | $\phi$, 800 updates, batch 12, LR 5e-5 | Same 9B condition interface; correct target-only FM and masks |
| Actor protocol SFT | $\theta$, 200 updates, batch 20, LR 5e-5 | PLAN/contract data; may run in parallel with DiT SFT |
| Fixed DiT and repair data | No $\phi$ updates; sampler gate plus at most 4,800 renders | DiT SFT complete; do not merge policy or change precision; determine the sampler first |
| Full Actor SFT | $\theta$, 800 updates, batch 20, LR 5e-5 | Shared PLAN and same-state repair/rollback/FINISH data |
| Two rounds of correction on self-generated states | 256 tasks per round, at most 3 renders/task; 100 Actor updates each | Current Actor plus fixed DiT SFT; actual executed corrective actions pass filtering |
| Fixed paired references | Copy both SFT policy adapters; no training | Equal values, independent storage, unchanged base/precision/conditions |
| Agent RL warmup | $\theta$, 100 rounds, $\phi$ frozen | Complete trajectories from the current Actor/Renderer and the reference contract |
| Joint LR pilot | $\phi$ LR 3e-5 and 1e-4, 10 rounds each; $\theta$ fixed at 5e-6 | Independent starts from the same warmup checkpoint |
| Main joint RL stage | $\theta,\phi$, 400 rounds | Restart from the common warmup point using the selected LR; update synchronously from the same behavior batch |

Actor SFT totals 1,200 updates and 24,000 action presentations; DiT SFT totals 9,600 sample presentations. These are training budgets, not counts of independent samples or promises of convergence. Family-separated validation determines whether overfitting occurs. If protocol/conditioning still fails its gate, repair data or interfaces first rather than using RL to compensate for incorrect labels. Extend training, when needed, in explicit new budget segments of 200 rounds while preserving the original curves.

| Item | Actor SFT | DiT SFT | Actor RL | DiT RL |
|---|---|---|---|---|
| LR | 5e-5 | 5e-5 | 5e-6 | Start at 3e-5; compare with 1e-4 |
| Global batch | 20 actions | 12 images | P6/N2/M2 | Same trajectory batch |
| Microbatch | 1 action | 1 image | 1 action | 1 image × 1 transition |
| AdamW $\beta$ | (.9,.999) | Same | Same | Same |
| Epsilon | 1e-8 | Same | Same | Same |
| Weight decay | .01 | .01 | 0 | 0 |
| Global norm clip | 1 | 1 | 1 | 1 |
| Main regularization | No extra KL | No extra KL | .02 × reference KL | .01 × reference KL |

For each SFT stage, warm up over the first 5% of updates, then cosine-decay to 10% of the starting LR. Rebuild optimizer/scheduler for each new stage while continuing model parameters. Rebuild the RL optimizer when freezing the reference at the end of SFT. Warm up for the first 10 Agent rounds. On entering joint training, retain $\theta$'s Adam history and initialize $\phi$'s Adam when it is first activated; warm up both joint LRs for 10 rounds and then keep them fixed. The pilot directly uses the LRs being compared so that all ten rounds are not spent at a low warmup LR. Every 25 rounds, evaluate 64 fixed development tasks, with at most 3 renders/task.

### 15.2 Twenty-Round Joint Pilot

Both branches begin from the same 100-round warmup checkpoint and $\theta$ optimizer. They have identical $\phi$ SFT adapters but separate freshly initialized optimizers. Branch A uses $\phi$ LR 3e-5 and branch B 1e-4; both use $\theta$ LR 5e-6, with ten fresh sampling/update rounds each. Pair task sequences and exogenous seeds across configurations; within each configuration, root plans and suffixes remain independent. Evaluate both branches on the same 64 development tasks at their endpoints.

Selection first requires passing numerical/protocol/resource gates, then compares task completion, protection, and visual quality. Choose 3e-5 if the difference is insufficient to distinguish them clearly. When no PG response appears, check gradients and evaluator criteria first rather than selecting an LR by an arbitrary KL target. If numerics are normal and images respond but gains have not appeared, examine learning curves and extend according to the predefined budget; do not claim that ten rounds establish ultimate superiority.

The pilot has at most $20\times72=1,440$ training renders plus $2\times64\times3=384$ endpoint development renders, totaling 1,824. Formal joint training restarts from the common warmup point. Do not combine weights from the two branches or count gains from pilot selection as test performance. Fixed $\beta$ requires no extra eight-batch calibration.

**Source page 33**

### 15.3 One Joint Update Cycle

```python
snapshot = begin_transaction(policy_adapters, optimizers, rng, data_cursor)
versions = freeze_and_ack_all_workers(snapshot)
batch = collect_complete_groups(P=6, N=2, M=2, versions=versions)
assert_complete_groups_and_public_private_separation(batch)
assert_replay_contract(batch)
root_adv, suffix_adv = hierarchical_rloo(batch)
zero_policy_gradients()

for action in batch.unique_roots_and_actual_suffix_actions:
    # reference hidden first; policy full graph; head chunks; decoder VJP
    backward_controller_pg_kl(action, root_adv, suffix_adv)
for render in batch.all_executed_renders:
    for step, weight in stratified_steps(render.T, strata=4):
        backward_cfg_vjp(render, step, time_weight=weight)

sum_allreduce_actor_policy_grads()
sum_allreduce_renderer_policy_grads()
assert_finite_on_all_ranks()
clip_global_policy_norms(1.0)
wait_both_components_ready()
step_active_optimizers_once()
validate_small_update_probes()
write_both_checkpoint_payloads_and_hashes()
write_COMMITTED_manifest_last()
publish_new_pair_and_wait_ack()
```

Agent-only training performs no $\phi$ backward/step. If the entire batch contains no renders, the $\phi$ optimizer does not step, and the transaction registers its old version. Each local loss already contains the global denominator; do not divide by world size after manual SUM. Empty ranks provide zero gradients and participate in reduction. If default averaging DDP is adopted later, multiply the local loss by world size to compensate; do not average twice. Neither component generates the next rollout batch while updates are underway.

### 15.4 References, Training State, and Transactions

Compute reference no-grad first. After switching to policy, put the decoder/DiT in train mode, keep the frozen vision tower in eval mode, set dropout to 0 and `use_cache=False`, and use non-reentrant checkpointing. Do not switch adapters, step, or clear gradients until all policy graphs have completed backward. BN/encoder/judge remain frozen. `set_adapter` changes `requires_grad`, so restore an explicit allowlist each time policy is reactivated; optimizer parameter lists contain only policy parameters.

Checkpoints save references to both policy/reference pairs, optimizer/scheduler, RNG, data cursor, behavior version, quantization/condition/evaluator manifests, and batch ID. Write COMMITTED last, after both files and hashes are complete. If one side fails, restore the complete state of both sides; reverting only a version number or restoring weights alone is insufficient. S11

Use 16 fixed action views and 16 transitions with different times/layouts as lightweight publication probes. Reject publication, roll both sides back, and halve LR if mean old-to-new KL over action positions exceeds .1 nat or DiT mean per-dimension old-to-new KL exceeds .01. Retry at most twice; if abnormality persists, stop that update for diagnosis. These thresholds are engineering limits against destructive updates, not theoretical trust regions or mandatory learning targets. Also report maxima per position/transition.

Reuse the batch only if both parameter sets and optimizers have been restored exactly to their sampling-time state; otherwise collect complete groups again. Include all retries, uncommitted computation, and failed pilots in cost accounting. Final gains are determined by the task comparisons in §20.

## 16. Failures, Retries, Zero Signal, and Termination

| Situation | Explicit handling |
|---|---|
| Invalid JSON/reference/geometry/action state | Preserve original tokens; feedback on the first occurrence, protocol failure with -1 on the second |
| Valid generation with poor effects, or repair introducing regression | Ordinary low reward; preserve all calls and scores |
| Length truncation | `finish_reason=length`; retain every actual token; handle parse failure according to the protocol |
| Ambiguous content or detector/VLM conflict | Semantic unknown; fixed requirement denominator; soft score 0 |
| RPC timeout/checksum failure | Retry the same job/input/version/seed at most twice |
| Service still unavailable | Keep the batch incomplete, stop updates, and complete full groups; do not train only successful siblings |
| Reproducible NaN/unknown OOM | Stop the sampling transaction, restore the last commit, and inspect numerics/memory; do not convert it to a low-reward image |
| Nonfinite gradients | Neither side steps or advances its scheduler; retain diagnostic data |
| All advantages zero | PG = 0; fixed-reference KL is still computed as defined; record zero PG and do not falsely report task learning |
| No renders in the entire batch | $\phi$ PG/KL are empty; no $\phi$ step that round, and no undeclared FM |
| Revision/behavior version/condition hash mismatch | Reject replay; identify the cause and recollect complete groups |

**Source page 34**

Initial execution timeouts are 900 seconds for the controller, 1200 seconds for rendering, and 300 seconds for one inspection, with queue time recorded separately. After the first service-profiling round, fix new timeouts to `max(initial value,3×P99)`. Timeouts and queuing delays do not enter policy rewards, preventing machine scheduling from coupling branches. Interruption accounting lists already consumed generation, failed jobs, and uncommitted batches; it must not hide extra computation.

If the zero-advantage fraction exceeds 80% for 25 consecutive rounds, or five consecutive PG-bearing updates produce no distinguishable output response, pause the current update segment. Investigate evaluator resolution → group independence → token/gradient interfaces → data difficulty and exploration, in that order. Do not delete the original complete groups; register reconfiguration as a new stage/config. Protocol invalidity and semantic unknowns may provide learning signals; device/service faults cannot be fabricated into negative rewards.

## 17. Responsibilities, Resources, and Default Configuration for Two Servers

### 17.1 GPU Responsibilities: Complete Replicas, Stage Switching, and Local Reduction

| GPU | Resident components | Specific responsibilities |
|---|---|---|
| A0, A1: 48 GB A6000 each | One 27B NF4 base plus policy/reference LoRA per GPU | One complete action request at a time per GPU; sampling and training in separate phases; update microbatch 1 and local SUM gradients |
| A2: 48 GB A6000 | Frozen Qwen3-8B text encoder plus FP32 VAE | Condition encoding and final-latent decoding; text/image artifact caches |
| A3: 48 GB A6000 | Frozen 32B NF4 evaluator plus one visual tool at a time | Public CheckSpec/inspection and private terminal-scoring queues; load/unload tools on demand; evaluator microbatch 1 |
| B0, B1, B2: 32 GB 5090 each | One 9B BF16 DiT plus policy/reference LoRA per GPU | Three generation workers; one image/transition per update, sequential CFG VJP, and local SUM gradients |

Nominal memory for A6000 and 5090 is 48 GB and 32 GB respectively, and compute capability is 8.6 and 12.0. Do not add their memory together as if it were one GPU's capacity or assume an existing NVLink bridge, P2P, or particular network bandwidth. S17 Only conditions, final latents, images/evidence, and version messages cross machines; **there is no cross-machine gradient synchronization**. Each policy uses the same GPU type, quantized artifact, and kernel configuration for online sampling and replay.

The offline PLAN teacher uses the same frozen 27B on A0/A1. During repair-data creation, free A6000s may temporarily run 9B inference or evaluator replicas after unloading the Actor. Measure queue bottlenecks before allocating resources, and do not count one GPU as two concurrent resources. Fix device type and backend within each paired candidate group. Independent human/official benchmark evaluation runs while training is paused.

Compute the reference policy by switching to an independent SFT reference adapter, then switch back to policy and restore its gradient allowlist. Do not use `disable_adapter` to obtain the pre-SFT base. Do not switch adapters while a policy computation graph is live. A2 instantiates only the encoder/VAE, not a complete bundle constructor that would also load the DiT. S5 S6

### 17.2 Verifiable GPU-Memory Budget

The following are static budgets computed from actual shapes and dtypes, **not measured peaks**. Define 1 GiB = $2^{30}$ bytes and 1 MiB = $2^{20}$ bytes.

For $Q$ NF4 parameters and $U$ nonquantized BF16 parameters, approximate double-quantized storage is

$$
M_{\mathrm{base}}=\frac Q2+\frac Q{64}+\frac{4Q}{64\cdot256}+2U\quad\text{bytes}.
$$

These terms are the 4-bit payload, 8-bit first-level scale payload, FP32 second-level scales, and nonquantized weights, respectively. Small per-tensor codebooks, alignment, and allocator overhead are additional. The extra memory for FP32 norms is listed separately. Trainable parameters, gradients, both Adam moments, and the frozen reference are all FP32, giving a primary resident amount of $20N$ bytes, not just one copy of LoRA.

| Item | Static quantity and meaning |
|---|---|
| 27B quantized base | $Q=24,326,963,200,U=3,029,765,360$; approximately 17.331 GiB, with about 1.54 MiB extra for norm precision |
| 27B policy + gradient + Adam m/v + reference | $20\times116,727,808$ bytes = 2.1742 GiB; with the base, approximately 19.51 GiB/GPU |
| 27B full-attention KV | $16\times2\times4\times256\times7168\times2$ bytes = 448 MiB, one copy per action |
| 27B GDN state | Recurrent: $48\times48\times128\times128\times4$ = 144 MiB; BF16 convolution state 3.75 MiB |
| 27B checkpoint layer inputs | $64\times7168\times5120\times2$ = 4.375 GiB; not all activations |
| 27B head chunks | Full-vocabulary FP32 logits for 32 positions = 30.3125 MiB; do not materialize the complete vocabulary at 7168 positions |
| Largest single NF4 dequantized matrix | $17408\times5120\times2$ = 170 MiB; backward workspace is additional |

**Source page 35**

| Item | Static quantity and meaning |
|---|---|
| 9B BF16 base | $9,078,581,248\times2$ bytes = 16.9102 GiB |
| 9B policy + gradient + Adam m/v + reference | $20\times31,981,568$ bytes = 0.5957 GiB |
| 9B checkpoint block inputs | $32\times4608\times4096\times2$ = 1.125 GiB; still excludes some activations |
| 9B bool attention mask | $4608^2$ bytes = 20.25 MiB |
| Accidental math-attention path | One layer's 32-head FP32 score tensor is 2.53125 GiB, excluding backward and multiple intermediates |
| A2: 8B encoder / FP32 VAE | Approximately 15.2564 GiB / 0.313 GiB, plus encoding activations and caches |
| A3: 32B NF4 evaluator | 31,205,621,760 quantized and 2,151,768,304 nonquantized parameters; approximately 19.00 GiB of weights, with no LoRA/optimizer |

Target-machine startup gates require peak reserved memory $\leq42$ GiB for Actor training GPUs and $\leq28$ GiB for DiT GPUs; A2/A3 also retain at least 6 GiB of device headroom. Run 32 complete maximum-layout cycles first, then 100 stability cycles. Record allocated/reserved memory, CPU RSS, and service queues. Headroom gates are engineering targets; before running the actual models, do not describe them as already fitting.

The memory priority is explicit: remove concurrent graphs within each worker; enable block checkpointing and the sequential VJP of §14; explicitly select `_native_efficient` for DiT; if that kernel fails on the actual machine, use validated query-row chunking plus checkpointing; only then consider CPU saved-tensor offload if limits are still exceeded. Do not disable regional masks, remove the negative CFG branch, or reduce reference precision while reporting results as the original configuration. Resolution 768 is not part of the main configuration: the maximum sequence grows to 8448 and a single-layer math score already occupies 8.5078 GiB, so it requires independent acceptance testing. S2 S25

### 17.3 Actual Computation and Communication per Round

$P=6,N=M=2$ produces 24 episodes and at most 72 renders. One `[1024,128]` FP32 latent is 0.5 MiB. The 51-state chain for 50 steps occupies 25.5 MiB; an entire round requires at most 1836 MiB, stored on B's CPU/NVMe. B sends only the final 0.5 MiB per image to A2 for decoding, at most 36 MiB/round.

One `[512,12288]` BF16 text block is 12 MiB. Three positive blocks plus two reference latents total 36.5 MiB/render. Cache the negative empty-string block by encoder version, and reconstruct the bool mask on B from the geometry manifest. Maximum A-to-B transfer is 2628 MiB/round; repeated conditions reduce actual transfers through content caching. Measure network bandwidth and scheduling rather than inferring wall-clock time from GPU counts.

Actor gradients are 445.28 MiB/rank and DiT gradients 122 MiB/rank. Typical per-rank local ring-SUM traffic is $2(W-1)/W$ times these amounts, approximately 445.28 MiB and 162.67 MiB respectively; record the actual environment's collective algorithm. Behavior-old uses policy values from transaction start, with small adapter/optimizer snapshots saved on CPU. Do not keep another resident old large base.

| Sampling configuration | Trajectories / maximum renders | Rollout DiT forwards | Replay forwards / backwards |
|---|---:|---:|---:|
| Main T50, Ks4 | 24 / 72 | 7200 | 1728 / 576 |
| Quality-gated T28, Ks4 | 24 / 72 | 4032 | 1728 / 576 |
| Quality-gated T20, Ks4 | 24 / 72 | 2880 | 1728 / 576 |

The formula is $F_{\mathrm{roll}}=PNMB_{\max}\cdot2T$. There are at most $72\times4=288$ replay transitions per round. Each transition includes both current/reference branches and both current VJP branches, totaling 6 forwards and 2 backwards, before checkpoint recomputation, probes, and Actor computation. Four-time-stratum sampling is not replay of the entire 50-step chain; the full chain must still be saved to reconstruct selected positions.

With root-action sharing, the maximum number of generated tokens is

$$
[PN+PNM(H_{\max}-1)]L_{\max}=[12+24\times9]\times1024=233,472\quad\text{tokens/round}.
$$

The 116,736,000 tokens over 500 formal rounds are an extreme upper bound, not an expected count. Large-model costs must include these token-by-token decisions, full-prefix replay, and judging. Scheduling cannot consider only 9B image forwards.

### 17.4 Total Accounting Including Data, Pilots, and Evaluation Criteria

| One main training path | Maximum render count |
|---|---:|
| Repair, injection, and two rounds of correction on self-generated states | 6336 |
| Initial sampler screening, including two fallback configurations | 768 |
| Twenty-round LR pilot plus endpoint development | 1824 |
| Formal Agent 100 + joint 400 rounds, 500 × 72 | 36,000 |

**Source page 36**

| One main training path | Maximum render count |
|---|---:|
| Development on 64 tasks every 25 rounds, 20 × 64 × 3 | 3840 |
| Subtotal | 48,768 |

There is no extra eight-batch coefficient calibration and no undeclared CE/FM replay inside RL. The 800 DiT SFT and 1200 Actor SFT updates are training forwards/backwards and are accounted separately, as are final benchmarks, multiple baselines, multiple training seeds, optional controls, original-data verification, additional retries, and follow-up experiments. The 600-render data pilot is a subset of the 6336. Joint 400 rounds replay at most 115,200 transitions; report the actual number with nonzero PG.

**Evaluator accounting.** Let $K_c,K_g$ be the numbers of public/private atoms and $b\leq3$ the actual render count. The first template in §9 always runs, while the second runs only when required by the fixed route. Hence

$$
K_cb+K_g\leq N_{\mathrm{query}}\leq2(K_cb+K_g),\qquad N_{\mathrm{answer\ sequence}}=2N_{\mathrm{query}}.
$$

For $K_c=K_g=12,b=3$, one trajectory uses 48-96 logical binary-choice queries and 96-192 candidate answer sequences; 24 trajectories use 1152-2304 queries. Six new editing sources add another 144-288 queries if neither channel is cached. INSPECT, CheckSpec decomposition, and paginated atoms are accounted separately using actual counts.

If all 6336 repair-data images are checked using 12 atoms in each of two channels, they require 152,064-304,128 queries. For at most 12,000 original pairs, 12 atoms each for source and target require 288,000-576,000 queries. Reliable human labels, rule checks, and caches can reduce actual model calls; the ledger must record which evidence was used. At most 24,000 public PLAN teacher proposals plus at most 192 proposals for an additional 96 records are counted separately from these queries; the 1024 contract-recovery records can be constructed by rules. Repair-teacher and refresh proposals are recorded by actual use as specified in §7. CheckSpec is decomposed once per unique request, with at most one format correction, so $N_{\mathrm{decompose}}\leq2N_{\mathrm{unique\ request}}$.

This ledger exposes the bottlenecks to measure first: A3's evaluator queue and 27B token-by-token generation may be slower than 9B training. Add evaluator replicas in scheduled offline periods while keeping the online configuration fixed. Do not covertly lower cost by changing inspection standards.

### 17.5 Host Memory, Storage, and Scheduling

Suggested starting available host memory is at least 128 GiB on A and 64 GiB on B; comfortable configurations are A 256/B 128 GiB. Suggested free SSD capacity is at least 1 TB on A and 2 TB on B. These are requirements to inventory, not assumptions about the user's existing hardware. Stream only selected OmniEdit samples rather than downloading its TB-scale complete dataset. S13

For 8,000 pairs averaging two text blocks, storage is about 187.5 GiB. A 64 GiB disk LRU on each machine can regenerate entries using original text and the fixed encoder. The original 27B checkpoint is about 55.56 GB and serialized NF4 about 18.6 GB; the original 32B evaluator is about 66.7 GB and NF4 about 20.4 GB; 9B is about 18.16 GB and the 8B encoder about 16.38 GB. These are disk quantities and must not be added to resident GPU memory.

A $512^2$ Actor-processor FP32 image tensor is 6 MiB. Tensor storage for the formal 36,000 generated images has an upper bound of 210.94 GiB; 24,000 active crops have an upper bound of 140.625 GiB. Source, reference, and offline images are additional. Deduplicate by image/processor hash. Retain the complete actual RL view tensors during the current stage; for SFT, save original images, IDs, processor versions, and a bounded cache.

The upper bound for complete latent chains over 500 rounds is 896.48 GiB: about 179.30 GiB for Agent 100 and 717.19 GiB for joint 400. After stage acceptance, use a preregistered archival policy to retain all manifests, images, and checkpoints, a stratified 1% sample of complete chains, and recovery chains for the most recent two rounds. Do not let multi-seed experiments accumulate without bounds. If B has only 1 TB, stage archives or external storage are necessary; do not promise that one disk can hold all intermediates. Keep one fixed version of each base, and checkpoint only adapters, optimizers, and references.

The actual scheduling procedure is 20 warmups and 100 timed runs per service type, followed by ten rounds of the real dependency queue. Record p50/p95/P99, peak GPU memory, per-round token/render/query counts, collectives, and transaction time. Extrapolate the complete budget from these measurements; do not guarantee a number of days using theoretical TFLOPS or an untested seconds-per-image estimate.

### 17.6 Consolidated Default Configuration

```yaml
protocol: dag-agent-v3
canvas: [512, 512]
train_max_eval_atoms: 12
check_atoms_per_batch: 12
max_entities: 16
max_active_regions: 2
max_source_images: 1
max_appearance_images: 1
episode: {max_actions: 10, max_renders: 3, max_inspects: 2, max_reads: 2}
feedback:
  public_source: frozen_request_only_CheckSpec
```

**Source page 37**

```yaml
  reveal_after_first_valid_plan: true
  gold_visible_to_actor: false
  private_reward_timing: terminal_plus_source_baseline
read_nodes: {max_nodes: 16, max_tokens: 1024}
actor:
  model: Qwen/Qwen3.8-27B
  class: Qwen3_5ForConditionalGeneration
  quant: nf4_double_quant
  compute_dtype: bfloat16
  policy_reference_dtype: float32
  recurrent_cache_dtype: float32
  max_view_images: 4
  input_tokens: 6144
  output_tokens: 1024
  enable_thinking: false
  preserve_thinking: false
  temperature: 1.0
  top_p: 1.0
  top_k: 0
  min_p: 0
  repetition_penalty: 1.0
  lora: {r: 16, alpha: 32, dropout: 0.0}
  one_request_per_worker: true
  fresh_cache_each_action: true
  lm_head_positions_per_chunk: 32
judge:
  model: Qwen/Qwen3-VL-32B-Instruct
  quant: nf4_double_quant
  input_tokens: 8192
  max_images: 2
  second_template: fixed_uncertainty_conflict_hardtype_route
sft:
  actor_global_batch: 20
  phase_counts: [5, 7, 5, 3]
  phase_weights: [0.25, 0.35, 0.25, 0.15]
  action_token_mean: true
  renderer_global_batch: 12
  aligned_target_pairs: 8000
  shared_actor_plans: 4000
  text_dropout: 0.1
  repair_data_after_renderer_sft: true
  repair_render_cap_with_refresh: 6336
renderer:
  model: black-forest-labs/FLUX.2-klein-base-9B
  text_hidden: 12288
  dit_hidden: 4096
  steps: 50
  efficient_step_candidates: [28, 20]
  require_sampler_quality_gate: true
  cfg: 4.0
  sde_eta: 0.3
  stochastic_steps: all
  state_dtype: float32
  model_dtype: bfloat16
  policy_reference_dtype: float32
  text_block_tokens: 512
  attention_backend: _native_efficient
  lora: {r: 16, alpha: 16, dropout: 0.0}
rl:
  tasks_P: 6
  roots_N: 2
  suffixes_M: 2
  sampled_steps_Ks: 4
  step_sampling: contiguous_strata_one_each
  step_weights: stratum_size_over_T
  actor_fixed_denominator: 10240
  actor_lr: 5.0e-6
  renderer_lr_default: 3.0e-5
  renderer_lr_pilot: [3.0e-5, 1.0e-4]
  pilot_rounds_each: 10
  updates_per_collection: 1
  ppo_ratio: false
  gradient_reduce: sum
  beta_actor: 0.02
  beta_renderer: 0.01
  supervised_replay: false
```

**Source page 38**

```yaml
  max_global_grad_norm: 1.0
  agent_only_rounds: 100
  joint_rounds: 400
runtime:
  compile: false
  microbatch: 1
  activation_checkpoint: non_reentrant
  sequential_cfg_vjp: true
  cross_machine_gradient_sync: false
  retry_same_job: 2
  seed_streams: independent
```

The lock manifest records complete revisions, wheel/quantization/adapter hashes, model-license access status, sampler, reward, data, and acceptance records. Freeze candidate step counts/LRs after selecting them through the established development procedure; do not change the algorithm's definition during training.

## 18. Extensions with Explicit Interfaces

### 18.1 Genuine Depth/Pose Control

The main configuration already uses images, layout, regional masks, entity references, and inspection evidence. Depth and pose maps require the model to learn their respective semantics; placing them in an appearance slot does not establish control. When needed, add a small residual adapter and paired supervision. It is a control-node type within the same DAG, not a replacement for the overall architecture.

One target latent token corresponds to $16\times16$ canvas pixels. Normalize depth as relative inverse depth using the 5th/95th percentiles of valid values, with near = 1 and far = 0, clipped to `[0,1]`. Set invalid positions to 0 and include a validity channel; set constant depth to .5. Pixel-unshuffle the two channels to 512 dimensions. Pose uses COCO17 joint heatmaps plus validity, 18 channels total; heatmaps have an 8-pixel standard deviation and peaks equal to confidence, with missing joints set to zero. Unshuffle to 4608 dimensions. Apply the same canvas transform to coordinates first.

Each control produces `[B,1024,d_in]`, then `Linear(d_in,64) → GELU → Linear(64,4096)` produces `[B,1024,4096]`. Add it only to the first 1024 target tokens of the `x_embedder` output; source and appearance reference segments have zero residual. Sum the two controls if both exist; absent controls use an explicit zero bypass. Initialize the first layer randomly and the second layer's weights/bias to zero, without adding another multiplicative gate that is also zero-initialized.

Including biases, parameter count is $d_{\mathrm{in}}\cdot64+64+64\cdot4096+4096$: 299,072 for depth and 561,216 for pose, totaling 860,288. The extended generation side has 32,841,856 trainable parameters. At 20 bytes/parameter for policy, gradients, both Adam moments, and reference, extra static storage is approximately 16.41 MiB, with no increase in token sequence length.

Use at least 2,000 pairs from real human controls/annotations or verifiable synthetic scenes, separated by family. Target-derived supervision may only be used as offline loss/filtering. Depth/pose used as a condition must be explicitly defined in training tasks as user-provided or produced by a public tool; hidden targets must not supply it at test time. Default control SFT uses 400 updates, batch 12, LR 5e-5, the FM of §8, whole-group text dropout .1, and whole-control-type dropout .1. Keep controls in both positive and negative CFG branches.

Use 200 isolated tasks to evaluate pairwise near/far depth-order accuracy and PCK@.05. A valid joint passes when its error is $\leq.05\times512$ pixels. A joint that should appear but is not generated fails; genuine occlusion is reported separately according to annotations. Enter RL only if there is an identifiable control response relative to the same text without that control, without material degradation in quality/identity.

The new MLP belongs to $\phi$. After SFT, copy an independent frozen `control_adapter_ref`; switch LoRA and the control MLP together for policy/reference. Register a new SFT reference point for the new branch and begin RL from it, rather than changing the reference midway through the main experiment. S2 S5

### 18.2 Variable Inference Budgets and Alternative Generation Objectives

The main configuration explicitly disables unrestricted thinking but retains necessary explanations in PLAN and revision fields. If a thinking budget is added, all actually generated thinking tokens enter the score, length, and cost; retain raw tokens completely and handle termination/truncation by the same rules. Do not train only the final JSON while ignoring sampled thinking that determined it. First compare quality/total tokens/time on fixed development tasks, then decide whether to adopt it as a new configuration. S1 S3

DiffusionNFT provides another generation-side post-training objective worth considering as a later independent branch. However, without derivation and same-budget validation, do not add it alongside the current Gaussian score to the main loss. S16 The main scheme already has a verifiable joint distribution and two gradient channels; first complete effectiveness comparisons for these channels.

Likewise, do not add an impact predictor, extra value network, or continuous latent-language bridge by default. If one is needed, define its actual bottleneck, inputs/outputs, additional training data, and cost as an explicit project. DAG error localization and branch revision are already handled by public evidence, dependency closure, and Agent actions. This specifies implementation order rather than restricting the research direction or what may be investigated.

**Source page 39**

## 19. Implementation Modules, Dependency Order, and Consolidated Acceptance

### 19.1 Implement in This Order

| Order | Planned file/module | Deliverables and completion conditions |
|---|---|---|
| 0: Environment and model locks | `configs/lock.json`, `runtime/probe.py` | Driver, wheel, topology, model/quantization hashes; operator probes for both GPU architectures |
| 1: State and artifacts | `schema/`, `dag/`, `store/` | Immutable revisions, slot types, cycle detection, branches and SELECT; verifiable without large models |
| 2: 27B worker | `models/qwen35_actor.py`, `models/fp32_linear_cache.py` | NF4 allowlist, 496 LoRA targets, template/vision/EOS, per-action cache and RoPE isolation |
| 3: 9B encoding/decoding | `models/klein_codec.py`, `models/klein_renderer.py` | 8B encoder, VAE/BN, 112 LoRA targets, DiT-only worker, efficient attention |
| 4: Condition compilation | `conditioning/compiler.py`, `conditioning/geometry.py` | 12288-dimensional text, target/source frames, reference roles, bool masks, four-axis IDs and bundle hash |
| 5: Sampling and trajectories | `sampling/flow_sde.py`, `trajectory/` | One CFG velocity, complete FP32 chains, raw action IDs, independent random streams, idempotent retries |
| 6: Checks and data | `checks/public_spec.py`, `reward/private_eval.py`, `data/aligned_pair.py` | DTO/storage isolation, evaluator calibration, public-only paired PLAN, family partitions |
| 7: Paired SFT | `train/agent_sft.py`, `train/fm_loss.py`, `data/state_bundle.py` | DiT and protocol warmup → sampler gate → real repair → full/corrective Actor SFT |
| 8: Reference | `models/reference.py` | Independent SFT adapters on the same base, no merge/requantization; correct APIs for both frameworks and gradient allowlists |
| 9: Agent RL | `train/hierarchical_pg.py`, `train/actor_head_vjp.py` | Root/suffix grouping, full-vocabulary KL, complete-prefix gradients, 100-round learning curves |
| 10: Joint RL | `train/stratified_replay.py`, `train/cfg_vjp.py`, `train/transaction.py` | Two score channels, four-time-stratum reduction, consistent two-sided commits, pilot and 400-round main stage |
| 11: Evaluation | `eval/official_adapters/`, `eval/paired_stats.py` | Main comparisons on the same tasks/budgets/seeds, ablations, and independent human audits |

This table is the implementation checklist for the future system. This round only writes the research document and lightweight verification scripts; it has not implemented these research modules or started training. Each step has explicit prerequisite artifacts, so implementers need not decide for themselves which model trains first or which output enters which model.

### 19.2 Eight Acceptance Groups to Resolve Correctness Centrally

**Gate 1: Structure, operators, and quantization.** On actual devices, assert 400 NF4 modules, 496 LoRA targets, and 116,727,808 parameters for 27B, and 112 targets with 31,981,568 parameters for 9B. Confirm complete checkpoint loading; 9B must not replace missing weights with zero tensors. After obtaining its gated snapshot, verify differences from the public structure. Check norm, GDN state, quantized payload, LoRA, and encoder dtypes individually. Original zero-increment LoRA should reproduce the original base; after copying SFT to reference, reference should reproduce the SFT policy, with independent storage.

Before training, fix 128 public development views and compare NF4 versus BF16 27B for action validity, requirement coverage, and visual identity/geometry questions. Use BF16 only for temporary offline inference split by layers across two A6000s, then restore the main layout; do not train two-GPU BF16. If NF4 shows clear concentrated degradation, expand the nonquantized module allowlist according to error location, recompute resources, and repeat the gate. Do not assume a smaller model is safer. Quantization quality awaits validation; performance is not inferred from parameter count.

**Gate 2: Vision, cache, and probability.** Fix cases covering 0/1/2/4 images, long/short prefixes, both EOS types, image → pure text → image, and branch switches. Save actual processor outputs and positions. Sampling records raw tokens/logp; recompute full replay at the same behavior version, then recompute with train/dropout 0/checkpoint enabled. Initial tolerances follow §6.6: mean absolute logp difference $\leq.01$ nat, p99 $\leq.05$ nat, and full-vocabulary KL at sampled positions $\leq1e-3$. Report the maximum as well and investigate large differences. Confirm FP32 GDN recurrent state, BF16 conv/KV, independent per-action caches, and no detached prefix gradients. If a gate fails, repair implementation/kernel consistency first, not stored logp to conceal discrepancies.

**Gate 3: Losses, gradients, and reduction.** Validate first on small CPU models, then small layouts on real models.

- For $z=\phi+\epsilon,\epsilon\sim N(0,1),R=z$, the expected score gradient is 1. Independent-sibling RLOO should lie within three estimated standard errors; a counterexample must catch zero differences caused by reusing the same noise. For $r\sim\operatorname{Bernoulli}(\operatorname{sigmoid}\theta)$, $z=\phi+r+\epsilon,R=z^2$, check $\partial_\phi J=2(\phi+p)$ and $\partial_\theta J=p(1-p)(2\phi+1)$.
- Enumerate stratified step sampling for small T: expectations weighted by stratum length/T equal the full-time mean. SUM over unequal rank partitions and empty ranks must equal a single process. Do not divide repeatedly by token/sample masks or the fixed $C_\theta,B_{\max},T,D$.

**Source page 40**

- Compare head-chunked VJP against direct full-vocabulary KL/PG backward, and CFG VJP against simultaneous two-branch backward. Require relative gradient error $\leq1e-4$ in FP32 and $\leq1e-2$ with cosine $\geq.999$ in actual BF16. Cover positive/negative branches and reference switching separately.
- Actual target tokens, including EOS, have CE; system/user/tool/old incorrect content/padding do not. Policy target parameters have gradients, while base/visual/encoder/reference gradients are empty. RL does not begin with a fresh zero B, so zero A gradients after SFT are not an expected invariant.

**Gate 4: Continuous policies and controls actually enter the network.** Recompute strictly decreasing sigma down to 0, verify every $s_k>0$, and match empirical SDE mean/variance to the analytic kernel. Gaussian KL uses actual $s_k^2$. Require relative replay error $\leq1e-5$ on small FP32 models. As a starting tolerance for actual BF16 velocity replay, require RMS of mean error divided by $s_k$ to be $\leq1e-3$. Decode the saved final $z_T$ directly; do not claim SDE reward using the final image from a different ODE chain.

First align the no-region, single-source-reference path with the official 9B pipeline. For the regional path, check 0/1/2 regions, overlaps, swapped bindings between two objects of the same class, source/appearance slot swaps, different source/target positions, and whether the second reference segment affects the specified object. Use known Q/K/V to inspect allowed edges in one layer; do not demand that a multilayer model have absolutely no indirect information propagation. Perturb layout/bindings: the bundle, attention outputs, and generation response should change accordingly. After passing the same-condition sampler quality gate in §12, fix the same sampler for training and testing.

**Gate 5: DAG and public-information contract.** Cover omitted-requirement repair, invalidation of descendants after ancestor revisions, cache reuse for unrelated artifacts, invalidation of dependent decisions when whole-image observations change, SELECT restoring manifests and actual sources, rejection of stale candidates, retries changing only the seed, inheritance of one invalid root by each suffix while actions remain $\leq10$, and zero-render trajectories that directly FINISH with the original source. Deliver only the designated final image; do not automatically choose the image with the highest private reward.

Fix public inputs, models, and RNG, then shuffle hidden EvaluationSpec order, answers, IDs, benchmark fields, or targets. Before termination, AgentView bytes, tokens, public calls, bundle hashes, and generated images must remain unchanged; only private terminal evaluation may change. Public pagination queries only the PublicEpisode projection. SFT teacher input equals the ActorView at the same state; private FM weighting masks must not enter conditions. Complete judge calibration and human spot-checks of 100 SFT records per bucket together. Critical-label error rate must be $\leq5\%$; 1024 tokens must accommodate valid actions; report actual quota fulfillment.

**Gate 6: Maximum working set and throughput.** Following §17, run 32 complete maximum-layout forward/backward/step cycles plus 100 stability cycles. Require A $\leq42$ GiB, B $\leq28$ GiB, with no continual memory growth. Confirm that masks reach the efficient kernel. Fallback query chunks must match full attention numerically and in gradients. Time 27B, 9B, encoder, and judge separately, then measure ten actual queue rounds. If static estimates fit but execution OOMs, first follow §17's explicit memory-handling order.

**Gate 7: Updates, publication, and recovery.** Both sides should have nonzero task PG and distinguishable output responses; report reference KL, parameter changes, and actual success rates as well. Fault injection covers duplicate RPCs, corrupted artifacts, judge timeouts, empty ranks, one optimizer stepping before the other side fails, half-written checkpoints, and disconnection before publication. After recovery, both parameter sets, Adam states, RNGs, cursors, and versions must match the last COMMITTED state. Do not update without a complete group.

**Gate 8: Task effects.** Complete the same-budget B2/B3/B4/B5 comparisons in §20 and report confidence intervals and categories of gains/degradations. At minimum, answer whether SFT helps valid planning/condition execution, whether RL improves real decisions, whether the generator receives effective return gradients, and whether joint training outperforms the corresponding single-component alternatives. Gate 8 yields an experimental conclusion, not a promise of improvement. If effects are insufficient, repair data, control, or optimization according to failure types, preserving complete curves and costs. Loss reduction is not a substitute for the research objective.

### 19.3 Levels of Completion Criteria

"Runnable" means passing the structural, numerical, resource, and recovery checks in gates 1-7. "Substantive gains" require gate 8 and independent evaluation. CPU formula checks resolve only formula/reduction questions; they do not replace actual 27B/9B GPU acceptance. Every acceptance record stores minimal input, expectation, versions, actual results, and repair diffs, avoiding repeated expansion of the same issue into new research modules in different sections.

## 20. Baselines, Ablations, Statistics, and Troubleshooting

### 20.1 Main Comparisons and Fair Budgets

All main comparisons use the same Qwen3.8-27B NF4 controller, Klein-base-9B, and tools. Do not subtract v2's 4B results directly from v3 and attribute the difference to the algorithm. Model-capacity comparisons are separate experiments with consistent interfaces/data/budgets.

| ID | Controller | Renderer | Purpose |
|---|---|---|---|
| B0 | Fixed request/fixed workflow | Original Klein | Base-model and generation-cost reference |
| B1 | Fixed workflow | DiT SFT | Isolate conditional SFT |
| B2 | Controller SFT + complete DAG | DiT SFT | Shared starting point after SFT on both sides |
| B3 | Agent RL + complete DAG | Fixed DiT SFT | Train only the controller |
| B4 | Fixed common Agent warmup checkpoint + DAG | DiT RL | Update only the generator after the same starting point; name the pure dual-SFT start separately as B4-SFT |
| B5 | Joint RL + complete DAG | Joint RL | Main system |

Also include same-budget fixed prompt rewriting, sequential repair, best-of-3, and separately trained-then-combined controls. Best-of-3 selects images using only CheckSpec:

**Source page 41**

rank first by the number of known passing items, then by the public fixed-denominator soft score, then by the score for user-public protection constraints; break remaining ties by earliest generation. Do not select using terminal-evaluation ground truth. Align total training render/forward budgets for separately trained-then-combined models with joint training. B0/B1 may report both single-generation and three-generation-budget points. Every method reports render count, DiT forwards, output tokens, GPU seconds, wall-clock time, actual cache hits, and logical action cost; do not report only a favorable cost axis.

The minimum complete main comparison continues the common 100-round Agent warmup checkpoint for 400 rounds each of $\theta$-only, $\phi$-only, and joint training. All three use the same P/N/M/T, task sequence, and maximum render budget, directly isolating update mode after the same starting point. The $\phi$-only branch explicitly includes common Agent warmup and must not be described as never having trained the Agent. Separately, 500-round $\theta$-only and $\phi$-only runs may start from pure dual-SFT and be compared with the 100+400 joint path as a total-budget comparison. Account for these extra experiments separately rather than hiding them within the main system's 500 rounds.

One common warmup plus three continuations requires at most $(100+3\times400)\times72=93,600$ training renders per training seed; three training seeds have an upper bound of 280,800, plus data, pilots, and evaluation. Execute a complete loop and main comparison for one seed first, then the other two. Data and frozen encodings may be shared, but model/optimizer/RNG states must not be mixed. If only the first seed is completed, report current results and uncertainty rather than presenting them as a three-seed conclusion.

Main evaluation runs one budget-constrained trajectory per task. Training's N×M hierarchical sampling is not default best-of-4 inference search. If inference search is additionally evaluated, identify search and selection costs separately. Final scoring evaluates only the image actually delivered by SELECT/FINISH; do not replace it with the trajectory's oracle-best image.

### 20.2 Required Ablations

- **SFT alignment:** Shared public PLAN/compiled bundles versus independently constructed data for the two sides. Both groups use the same original image pairs, family partitions, teacher calls, and repair-render budgets. Report actual usable counts and geometric compatibility rates, testing whether learning to propose conditions must align with learning to execute them.
- **SFT contribution:** No Controller SFT, no DiT SFT, and no SFT on either side. Compare three data variants: successful demonstrations, adding failure repair, then adding rollback/stopping. Compare three reductions: original token mean, single-action mean, and phase plus action mean. Align families, validation renders, and training token/update budgets; report actual repetition rates. Examine short termination-action accuracy, regression rate, condition-execution rate, and final success, not only loss.
- **Joint training:** Train only $\theta$, only $\phi$, both simultaneously, or train separately in stages and combine. Include gradient norms/update magnitudes on both sides to show that both receive task-return signals in the main system.
- **DAG:** Sequential histories with identical observation/tool budgets, recording a DAG without using it for decisions, removing observation dependencies, prohibiting ancestor repair, and recomputing every whole branch. Measure success, regression, and actual cache savings separately. Do not assume DAG gains imply local DiT compute savings.
- **Multimodal conditions:** Text only; layout plus masks; adding identity references; shared source/target masks; removing reference-entity binding; disabling regional attention. Keep evaluation requirements identical across variants.
- **Credit and reward:** Flat episode credit versus hierarchical credit; uniform Ks2 as a cost-effect point; at fixed Ks4, equal rounds, and equal total replay transitions, compare uniform step sampling with four-stratum coverage, measuring gradient variance, responses by time interval, and final metrics. Remove the minimum term, remove protection terms, or use only the final-image soft score. The biased shared-noise implementation is only a numerical counterexample, not a recommended training method. Rereport actual units/scales for every reward definition; changing the requirement denominator is prohibited.

Run interface ablations first on a fixed development subset, then formal evaluation of complete main controls. Report both statistical significance and effect size; do not select the best configuration using one seed.

### 20.3 Datasets and Statistical Protocol

Construct 500 hidden diagnostic tasks in five classes of 100 each: planning coverage/omissions, entity identity/relation binding, local editing/protection, ancestor-error propagation, and multibranch revision/rollback. Divide each class into natural errors and controlled injections and report them separately. Injection labels do not enter Agent or judge inputs. Separate 200 development tasks, 600 calibration tasks, and 500 diagnostic test tasks by family. Public benchmarks do not participate in reward tuning.

Report official tests in parallel: GenEval uses the official category macro-average; GenEval2 uses the official 800 records, Qwen3-VL-8B, and explicit `--method soft_tifa_gm`, taking the geometric mean of question-answer soft scores within each prompt, then averaging across prompts and multiplying by 100; MagicBrush reports all turns and final turns using its official DINOv1 metric. S12 S14 Official GenEval2 actually sums first-token probabilities of acceptable answer variants, rather than internally normalizing complete Yes/No+EOS sequences. Preserve its original algorithm and separately record a sanity check for duplicate token IDs; do not silently correct it and continue calling the result official. Run the official README's Transformers 4.57 environment separately with fixed model revisions. Name the internal 32B NF4 evaluator, DINOv2 preservation score, and this document's CS separately. S14 The availability of the public test in 2026 has been rechecked; do not retain outdated claims that no test exists.

Establish two MagicBrush tracks. Independent editing reads the official source for each turn, writes `inde_*`, and aggregates all turns. Iterative editing starts with the first session image, then uses the actually delivered previous-turn image as the next source, writes `iter_*`, and aggregates final turns. Reset the 10-action/3-render budget each turn while preserving session-level provenance. Do not present independent-track outputs as iterative results. Default mask-free editing receives only source plus instruction; packaged targets, masks, or extra global descriptions are not automatically given to the Agent. Create a separate mask-provided comparison with fair inputs. Only the scoring process loads GenEval2's `vqa_list` and other scoring fields. The official prompt-to-image_path dictionary accommodates only one image per prompt; run and save scores separately for multiple seeds, without overwriting one dictionary.

**Source page 42**

Default RL training seeds for the main comparisons are 11, 23, and 47, sharing the same registered dual-SFT starting point. These replicates measure randomness in RL sampling/optimization; SFT construction/initialization variance is reported separately in SFT ablations. To control evaluation cost on the actual hardware, fix exogenous seed 101 for every main-evaluation task. Add seeds 211, 307, and 401 on 200 preselected development/specialized-test tasks for robustness evaluation. Extending all benchmarks to four seeds requires a separately stated threefold additional inference budget, applied equally to all methods. Pairing the same task/seed across methods controls evaluation variance and is allowed; online-training siblings remain independent. First report per-seed results and across-seed dispersion. Compute 95% intervals with 10,000 stratified paired bootstrap replicates, resampling training seeds first, then tasks/families/sessions. Sibling images and different-seed images of the same task are not independent samples. Use original sessions as the MagicBrush unit. Apply Holm correction to primary B5 versus B2/B3/B4 comparisons, and report percentage-point differences and intervals.

The human audit contains 300 pairs: 100 each for B5 versus B2/B3/B4. For each comparison, divide pairs into mutually exclusive tertiles of the mean of the two methods' terminal scores, sampling 34/33/33 pairs with a fixed seed and saving inclusion probabilities. Randomize left/right order and anonymize methods. Two people independently judge task completion, original-image preservation, identity consistency, and visual quality; a third adjudicates disagreements. Report each stratum and overall inverse-inclusion-probability-weighted win/tie/loss rates, indeterminate rates, Cohen $\kappa$, and clustered intervals. Do not treat raw proportions from an oversampled stratum as overall win rates. Official benchmark evaluators are independent of the internal 32B reward. Agreement among VLMs is not equivalent to correctness; report human conclusions separately.

During training, audit 50 outputs offline every 25 rounds, half high-reward and half random, with stratified human verification every 100 rounds. Record the fraction that passes training evaluators but fails human review, including missed objects, blurred images, textual false evidence, background changes, and reference confusion. If thresholds or evaluators must change, end the old reward version, recalibrate, and start a new stage; do not present before/after curves as one reward scale.

### 20.4 Operational Order for Diagnosing Problems

| Symptom | Check first | Handling and reverification |
|---|---|---|
| Loss changes but images do not | Target slicing, adapter activation, B gradients, condition hash | Perturb parameters and controls for one step at a fixed seed; verify velocity and image responses |
| Repair becomes a new edit of an earlier source | Selected pointer, late binding of historical sources, candidate binding | Minimal SELECT → APPLY regression case; inspect actual source artifact ID |
| High reward but incorrect count/identity | Missed detections, entity matching, VLM calibration, denominator for unknowns | Recheck with humans and structured evidence; retest after freezing new criteria |
| RL has zero advantages for a long period | Identical sibling seeds, root/leaf grouping, requirement soft scores, task difficulty | First toy gradients and independence, then predefined difficulty strata/corrective SFT; do not delete zero groups |
| Agent improves but DiT does not change | Empty renders, $\phi$ trainable set, VJP negative branch, scaling | Separate PG and reference KL; compare PG norm and parameter differences; repeat gradient and LR pilots |
| Large rollout/replay logp discrepancy | CFG, masks, IDs, GDN cache, RoPE, sigma, FP32 states | Compare field hashes and align layer by layer; do not directly enlarge clipping or reduce reward |
| 5090 OOM or abnormal speed | SDPA backend, score matrices, two-branch activations, cache leaks | Optimize in §17 order; record peaks and time rather than concealing problems by disabling controls |
| Selecting an old image still returns a new one | Finalize using hidden best or last-generated pointer | Explicitly use only the delivery ID; assert corresponding image/evidence/hash |
| One rank is slow or the batch incomplete | Job retries, same-version acknowledgments, task allocation, stale workers | Restore complete groups and configure timeouts; do not gain speed by dropping low-score samples |

Open questions for training feasibility are actual maximum-layout peaks, masked-SDPA speed on both GPU types, evaluator reliability by category, fulfillment of the 8,000-pair supervision quota, effective learning signals, and actual network utilization. Each has an independent acceptance and repair path above. These measurements determine implementation parameters and resource scheduling, not whether to reject or narrow the research topic.

## 21. Source Differences, Adopted Scope, and Specific Repairs

The following classifications describe code behavior under the specified configurations. They do not infer author intent or equate all capabilities discussed in papers with what public repositories already implement.

| Inspected location | Finding and applicable conditions | Planned repair / verification |
|---|---|---|
| Transformers 5.12.1 hybrid cache S26 | Recurrent initialization inherits conv dtype and may store FLA FP32 state as BF16; linear crop cannot undo state | Minimal FP32 cache subclass; fresh prefill per action; complete-prefix chunk replay; compare with subsequent upstream fixes |
| Qwen3.8 processor/template/forward S1 S3 | Qwen3_5 hybrid architecture, thinking enabled by default; EOS overlaps PAD; no Qwen3-VL DeepStack | Official class/processor; explicit template switches; position masks; frozen visual-prefix injection and true-position replay |
| PEFT `prepare_kbit` and adapter switching S5 | Nonquantized parameters become FP32; `set_adapter` changes `requires_grad`; quantized merge is not an identity | Explicit precision restoration, 496-target allowlist, independent SFT-adapter copy; do not rebuild the same quantized base |
| Diffusers `PeftAdapterMixin` S5 | `set_adapter` does not accept PeftModel's `inference_mode` argument | Separate DiT add/set interface; explicit reference freezing and policy allowlist |

**Source page 43**

| Inspected location | Finding and applicable conditions | Planned repair / verification |
|---|---|---|
| Diffusers attention dispatch / PyTorch SDPA S25 | Native Flash rejects nonempty masks; the existing efficient backend supports masks, but actual kernels still require acceptance testing | `_native_efficient` main path; equivalent query-row-chunked fallback; no silent switch to math/removal of masks |
| BFL 9B metadata/config S24 | Text encoder is Qwen3-8B CausalLM, not the controller; FP8 in an official path is not equivalent to this BF16 conditioning | Complete 9B snapshot, BF16 frozen encoder, 12288 text/4096 DiT dimensions; regenerate 9B repair labels |
| Tencent `diffusion.py` 31-111, 287-346 S6 | Both branches have `kwargs=None`; missing negative text skips negative CFG; `sde_indices` controls noise injection and saving together | Explicit masks/IDs; error on missing negative conditions; stochastic steps cover all T, with independent train indices; same velocity in sampling/replay |
| Tencent Qwen `bundle.py` 38-94 S21 | Class name is hard-coded to Qwen2.5, so changing only the Qwen3 checkpoint string is insufficient | New `qwen35_actor.py` using the Qwen3_5 multimodal class and NF4; complete `mm_token_type_ids`, grids, FP32 GDN cache, and behavior-logp checks |
| Tencent `bundle.py` 169-186, `vae.py` 70-132, `track_builder.py` 738-756 S6 | DiT-only is already implemented; generic SFT calls and Klein VAE signatures/return formats do not yet form a complete interface | `klein_codec.py` shares encoding core and separates target/reference wrappers; shape and decode round-trip checks |
| Tencent `algorithms/sft.py` 195-320 S6 | Training does not use evaluation's sample loss mask; nontrivial masks expose a reduction gap; aux is already detached | `fm_loss.py` returns differentiable per-sample loss with a unified valid denominator; verify 4+1 split-batch equivalence |
| G-U-N `edit_grpo_trainer.py` 805-855 S8 | For CFG > 1, rollout has two branches but replay only the positive branch, making policies inconsistent | One velocity definition and sequential two-branch VJP; compare with direct autograd |
| G-U-N, same file 213-253, 504-534 S8 | Unfreezing backbone/lm_head after PEFT violates this scheme's LoRA scope; equal weighting of tail-batch means can bias reduction | Trainable allowlist and global denominator; check parameter counts and unequal microbatches |
| RePlan `flux_klein.py` 1050-1074, 1158-1193 S7 | Accessing `image[0]` limits pure T2I; multi-reference loops/indexing have shape or overwrite risks in this integration path | Handle None first; per-reference grids/offsets; independent source/target transforms; 0/1/2-reference matrix |
| RePlan, same file 1299-1326; `flex_attn.py` 516-647 S7 | Positive/negative text layouts differ; flex masks are not native attention masks | Identical layouts for both branches; native SDPA bool masks; signature checks and forbidden-edge tests |
| Flow-GRPO SDE 49-93 and Kontext training 840-850 S9 | Returns noise coefficient d; this KL expression omits complete transition variance $d^2(-\Delta)$ | Analytic $s^2$ kernel from §12; KL Monte Carlo with different $\Delta$ |
| ComfyUI `execution.py` 733-801 S10 | `inference_mode` is reasonable for inference but does not support training graphs; residual nodes are not exact SCCs | Own training replay; SCC cycle rejection, ordered-slot caching, and branch history |
| GroundingDINO CUDA dispatch 65/135 and wrapper 330-350 S15 | Old Tensor API needs compatibility patching with new torch; no BF16 kernel, and CUDA inputs do not automatically fall back to CPU | `tools/patches/groundingdino_torch26.patch`, FP32 worker, and reference checks for build/load/forward |
| EMU rollout 478-498, Gen-Searcher workflow 602-618 S16 | Falling back to all-ones after token-alignment failure contaminates supervision; grouping by unique episode and filtering zero reward is unsuitable for RLOO group filtering | Preserve original token boundaries; quarantine bad records as infrastructure failures; explicit parent/group IDs and complete-group retention |
| State casts and continuous density S6 S8 | Verify sampled actions against saved states; preserving original continuous actions can support an extended probabilistic interpretation, rather than declaring both repositories entirely wrong | Save complete FP32 states/samples; BF16 only for network inputs; verify each path individually |
| UMM transactions / DiffusionNFT S11 S16 | Transactions and alternative objectives are useful references; they do not provide this complete two-machine DAG joint implementation | Add two-sided COMMITTED protocol; NFT only as a separate objective comparison |

Distinguish facts from design: findings in the table are limited to inspected pinned files/branches; repairs are changes this system plans to adopt and have not yet run on GPUs. The gated 9B configuration/weights were not downloaded in this round. Structure comes from official public definitions and metadata; strict loading after authorized access is a startup gate. Record paper descriptions, README support lists, and executed results separately; do not generalize one file's gap into invalidity of an entire project.

For every port, retain the original file/function, commit, modification diff, and test results. Neither copy a repository wholesale nor rewrite already verified reliable Transformers, VAEs, and processors.

## 22. Pinned Sources and Retrieval Locations

The following links point directly to public source code or model configurations. Repository commits are snapshots inspected in this round; official hardware/installation documentation consists of dynamic pages, verified on October 26-27, 2026. Record SHA256 of downloaded content separately during implementation. Line numbers assist location; functions and filenames are the primary anchors.

- **S1 | 27B controller snapshot.** Qwen3.8-27B configuration, weight index, processor, template, `generation_config`, and model card. This round also parsed JSON headers from 18 shards and 1199 shapes, without downloading weight payloads.
- **S2 | Klein Transformer.** Pinned 9B model snapshot and public card; configuration/weights require authorization, with structure also checked via S24. Diffusers Transformer: double processor 325-391, projections 526-545, kwargs filtering 550-567, single processor 569-625, fused projections 745-763, timestep 1231, and IDs 1268-1280.
- **S3 | Qwen3_5 multimodal forward.** `modeling_qwen3_5.py`: GDN 371-557, full attention 645-718, model/MTP 812-824, vision 1018-1118, language forward 1137-1221, placeholders 1446-1477, positions 1490-1533, multimodal injection 1535-1608, and generation positions 1903-1918. Reuses Qwen3VLProcessor and its image processor.

**Source page 44**

- **S4 | Klein encoding/decoding and sigma.** `pipeline_flux2_klein.py`: shift 63-78, text 208-263, IDs 265-365, patch/pack 369-399, VAE 462-475, sampling 811-880, decode 904-918; FlowMatch scheduler.
- **S5 | PEFT and LoRA.** kbit preparation, `PeftModel.set_adapter`, and Diffusers adapter interface; initialization in `layer.py` 328-341 and target matching in `tuners_utils.py` 1894-1934.
- **S6 | Tencent UniRL.** Pinned tree; `diffusion.py`, `bundle.py`, `vae.py`, `track_builder.py`, `algorithms/sft.py`, SDE kernels, and `trainer/pe.py`.
- **S7 | RePlan.** `flux_klein.py`, `flex_attn.py`: regional encoding 732-825, input handling 1050-1074, references 1158-1193, and CFG 1299-1326; flex import initialization 28-32 and processor 516-647.
- **S8 | G-U-N UniRL.** Klein pipeline: SDE 45-98, CFG 333-357; `edit_grpo_trainer.py`: unfreezing 213-253, reduction 504-534, replay 805-855.
- **S9 | Flow-GRPO.** SDE/logp and Kontext KL.
- **S10 | ComfyUI.** `graph.py`: ready 117-240 and residual cycles 375-392; `caching.py` 83-149; `execution.py` 733-801.
- **S11 | UMM-Reflection transactions.** `v20_update_transaction.py`, primarily update stages and state checks at lines 1-220.
- **S12 | MagicBrush.** Pinned dataset version, README, `image_eval.py`: final turn 165, all turns 208, DINO 400; DINOv1 hubconf.
- **S13 | OmniEdit.** Pinned Filtered-1.2M dataset card, train 1,202,797 and dev 700; download manifest, schema, scoring fields, and license checked separately.
- **S14 | Official evaluation.** GenEval `summary_scores.py`; GenEval2 `evaluation.py`: first-token answer variants and count handling 73-145, geometric mean 187-188; 800 records and Tencent's byte-identical copy; official Qwen3-VL-8B revision.
- **S15 | Visual tools.** GroundingDINO `inference.py` 53-99; SAM2 predictor 85-129; pinned DINOv2 source. Installation locations: GroundingDINO CUDA kernel, Python dtype branch, torch 2.6 Dispatch, and SAM postprocessing. Public weights: GroundingDINO SwinT, SAM2.1 small, DINOv2 B/14, and pinned BERT.
- **S16 | Other training references and limitations.** DiffusionNFT SD3 training; EMU rollout; Gen-Searcher workflow.
- **S17 | Hardware and runtime environment.** NVIDIA A6000 datasheet, RTX 5090 specifications, GPU compute capabilities, CUDA compatibility table, and PyTorch version installation table.
- **S22 | Training mode and checkpointing.** Transformers `GradientCheckpointingLayer`, enabled only in training mode; separate frozen-module states from current-policy training state.
- **S20 | Adam and RL configurations.** PyTorch 2.13 Adam update and Flow-GRPO default LR. Used to check external-review claims about step sizes and source defaults, not to assume another project's LR is optimal here.
- **S21 | Qwen3 worker integration gap.** Tencent Qwen loader; Qwen3 processor.
- **S19 | OCR.** Tesseract 5.5.0 and tessdata_fast 4.1.0. Used only for this plan's calibrated English/digit tasks; record data and binary hashes separately.
- **S23 | Frozen 32B evaluator.** Qwen3-VL-32B configuration, weight index, processor, and Qwen3-VL forward. DeepStack remains in the evaluator's own path.
- **S24 | Official 9B structure and text encoding.** BFL Klein9BParams, official loading configuration `util.py`, `text_encoder.py`, 9B conversion definition, and official img2img LoRA example. Cross-check the 8B structure against Qwen3-8B configuration. Actual deployment still loads the encoder packaged with the same 9B snapshot; do not substitute one merely by name.
- **S25 | Actual masked-attention backend.** Diffusers `_native_efficient_attention` 3187-3219; PyTorch native Flash mask check 259-266; CUDA efficient architecture/dtype checks. Source-code permission is not the same as passing tests on the target wheel/GPU.
- **S26 | NF4 and GDN cache.** bnb quantization, input gradients, installation/architecture documentation, pinned cache initialization, subsequent upstream fix, FLA recurrent kernel, chunk kernel, and causal-conv1d.
- **S27 | RLOO implementation reference.** TRL `rloo_trainer.py`. Draws on implementation experience with independent repeated sampling and leave-one-out; root/suffix grouping and two-model scores must be implemented according to §13, not replaced by an existing trainer in lieu of the derivation.

**Source page 45**

## 23. Actual Verification Records and Delivery Boundaries for This Round

### 23.1 Source, Metadata, and CPU Checks Performed

This round reread the 27B configuration, template, processor, hybrid-layer forward, cache, PEFT, bitsandbytes, and FLA; official 9B structure, converter, encoder, mask backend, SDE/training paths; and the 32B evaluator's configuration and index. Direct sources are listed in §22. For 27B, HTTP Range requests read the length fields and JSON headers of all 18 safetensors shards, verifying 1199 shapes against the index without downloading full weights. Gated 9B weights/configuration were not obtained by bypassing permissions. Public structure agrees with public parameter metadata; actual strict loading remains a startup acceptance check.

The following values were actually computed or recomputed on CPU using NumPy floating-point arithmetic. They are not model-performance experiments.

| Check | Method and actual result |
|---|---|
| Hierarchical RLOO, the same $\theta$ controlling root and suffix | Exact enumeration of 1024 outcomes, compared with centered differences of expected return: $\theta$ score .020730804773 versus difference .020730804740; $\phi$ .034420899272 versus .034420899353; maximum absolute difference $8.09\times10^{-11}$ |
| Hierarchical RLOO with valid zero-render branches | Exact enumeration of 324 outcomes; maximum difference between the three expected root/suffix/renderer scores and direct expected derivatives $1.249\times10^{-16}$; total probability .9999999999999998 |
| Sequential CFG VJP | Small D7 two-branch tanh model, CFG 4, negative coefficient -3; compared with finite differences of original Gaussian PG + KL: maximum difference $4.20\times10^{-12}$, relative L2 $1.45\times10^{-10}$ |
| Frozen-head position-chunked VJP | 11 words, hidden dimension 3, complete causal prefix, PG + full-vocabulary KL with EOS/padding; chunked versus single-pass VJP difference 0; maximum finite-difference error $4.21\times10^{-12}$, relative L2 $2.24\times10^{-9}$ |
| Four-time-stratum step selection | T20/T50 expectations weighted by stratum length/T differ from full-time means by $\leq1.11\times10^{-16}$; incorrect equal weights at T50 give .532770 versus the correct mean .524120 |
| Phase/action SFT reduction | Quotas 5/7/5/3, inserted mask-0 padding and unequal/empty ranks; original-check difference 0, another random unequal partition $2.22\times10^{-16}$ |
| Gaussian KL | 500,000 samples, std = .17, mean difference = .025; analytic .01081315, MC .01086738, difference .261 standard errors; incorrect denominator gives a value 16 times too small in this example |
| Sigma and transition standard deviation | At T20/28/50, all sigma values strictly decrease to 0, variances are positive, first-step std = .3; final-step std .09209302/.06410270/.03184463 |
| Query-row-chunked attention | NumPy full QK softmax versus query chunks reading all K/V; maximum difference $1.11\times10^{-16}$; does not imply a tested GPU backend |
| EOS and PAD boundaries | `[20,248044,248044,248044]` with actual length 2: positional mask preserves 2 tokens including EOS; masking by PAD ID incorrectly preserves only 1 |
| 27B structure and storage arithmetic | 496 LoRA targets, 116,727,808 parameters; 400 quantized projections, 24,326,963,200 parameters, matching actual shape inventory |
| 9B structure arithmetic | Official-module sum 9,078,581,248 matches public parameter metadata; 112 LoRA targets, 31,981,568 parameters; static 16.9102 + .5957 GiB |
| 32B evaluator arithmetic | Index corresponds to 33,357,390,064 parameters; 448 quantized projections with 31,205,621,760 parameters; static NF4 approximately 19.00 GiB |
| Termination actions and reward bounds | For the same image, old STOP exceeds ACCEPT by .05 on failure and is equal on success; new FINISH has conservative success lower bound .68, failure upper bound .3, full range [-1,1.3] |
| Cost reduction | One main path 48,768 renders; joint training at most 115,200 replay transitions; three-branch main comparison 93,600 training renders/seed |

**Reproducible small environments.** The first check uses $r\sim\operatorname{Bernoulli}(\operatorname{sigmoid}\theta)$, $a\sim\operatorname{Bernoulli}(\operatorname{sigmoid}(\theta+.7r))$, and $z\sim\operatorname{Bernoulli}(\operatorname{sigmoid}(\phi+.4r-.3a))$; $R=\mathbf1[a=r]+.8\mathbf1[z=r]-.15a$, with $\theta=.31,\phi=-.27,N=M=2$, and centered-difference step $10^{-6}$. Enumerate two roots and two `(a,z)` pairs per root using the respective root/suffix coefficients of §13, covering one $\theta$ influencing both stages.

The second check, with zero renders, uses root logit .3, suffix render/no-render logit $-.4+.2r$, and image Bernoulli logit $.6-.3r$. No-render return is $.15+.05r$; rendering gives return 1.0 when $z=r$, otherwise -.05. Exact enumeration of 324 outcomes for two roots with two suffixes each gives direct derivatives `(.0342076777921,.0678066996559,.0253506372788)`, identical to RLOO. This verifies that zero-render branches must not be deleted from groups.

The head check uses seed 20261027. Initialize $X\in\mathbb R^{7\times4}$ and $A\in\mathbb R^{4\times3}$ from normal distributions with standard deviation .2, and frozen $W\in\mathbb R^{11\times3}$ with standard deviation .3. The decoder is $H=\tanh(\operatorname{cumsum}(XA))$. Use positions `[2,3,4,5]`, targets `[0,10,8,9]`, mask `[1,1,1,0]`, advantages `[.7,.7,-.2,-.2]`, coefficients `[1/120,1/120,1/240,1/240]`, and $\beta=.02$. The reference uses `A+.03` with stop-gradient. Chunk every two positions and use centered differences of $10^{-6}$ on A. Intermediate prefix gradients are nonzero, so chunking does not remove the prefix's influence.

Additionally, an Adam counterexample gives gradient 1 after 999 steps of zero gradients, with $\beta=(.9,.999)$ and epsilon 1e-8. Displacement at step 1000 is $2.514566\times\mathrm{LR}$. This demonstrates that claiming each parameter moves at most LR per step cannot prove that DiT necessarily fails to learn. The actual changes still adopt a reasonable higher-LR pilot and a more sufficient transition

**Source page 46**

budget, evaluated through output response and task effects; see §§13 and 15. With the same initial state and zero terminal potential, potential-function rewards cancel in episode-level RLOO, so the main objective does not add this ineffective wrapper.

### 23.2 Items Still Requiring Actual GPUs

The complete 27B/9B/32B models, NF4 quality comparisons, FLA cached/full probability consistency, masked efficient kernels, maximum-layout backward, service throughput, SFT, and RL have not been run. Small CPU errors do not establish that these gates have passed. Implementation especially needs to confirm:

- Configuration/strict loading of the authorized 9B snapshot and consistency of its bundled encoder.
- Whether complete-prefix 27B QLoRA and FP32 GDN caches on A6000 meet probability/peak-memory gates.
- Whether bool-masked efficient attention and dual-CFG VJP on 5090 meet the 28 GiB gate.
- Category-specific calibration and queue costs for the quantized 32B evaluator.
- Paired-SFT geometric compatibility, quality of actual repair labels, and actual fulfillment of the 8,000 quota.
- Task PG, output response, and same-budget single-component/joint comparisons on both sides.

For each item, §19 already specifies inputs, thresholds, repair order, and recording requirements. The current deliverable is **a design checked against source code and mathematics and implementable module by module**; future GPU acceptance, training gains, and wall-clock times are not presented as measured results.

### 23.3 Keeping the Research Path Complete

The DAG connects intent, executable controls, and visual evidence. Same-condition SFT teaches the Agent and generator to use a shared interface. Agent RL learns planning and revision in the real closed loop. Joint RL adjusts both policies from the same task outcome. Error-node localization, ancestor/branch revision, identity and spatial propagation, rollback, and termination all remain in the main system. Next implement §19, then use §20 to test the actual contribution of each stage; there is no need to add unrelated modules or change the research topic to fill implementation gaps.

### 23.4 Final Implementability Review and Finalization Confirmation

**Final conclusion: no unaddressed major design contradiction was found that would cause the plan to fail when implemented as specified. v3 can be finalized and implemented according to §19.** This round reread the complete document, compared it with the initial external review, and separately checked model execution, generator and DAG, SFT/RL, and joint gradients. Pinned source code was retrieved again online rather than relying solely on papers or previous conclusions. This round does not change the main models, training objectives, default parameters, DAG protocol, or resource allocation. The following verification records are added.

| Final review scope | Basis and conclusion |
|---|---|
| 27B loading, quantization, and actual calls | Official configuration, HF forward/generation, PEFT, and bnb implementations agree. Reparsed 1199 weight shapes: still 496 LoRA targets and 116,727,808 adapter parameters. Allocation of a complete NF4 replica to one GPU remains consistent. S1 S3 S5 S26 |
| 27B sampling and differentiable replay | HF accepts the repaired Cache supplied to it; append multimodal type 0 for new tokens and extend four-axis positions. Complete replay using saved visual-prefix slots matches actual generation semantics. Retain the FP32 recurrent repair and head VJP without adding a new model branch |
| 9B conditioning and execution paths | Pinned BFL/Diffusers/Tencent implementations verify 12288-dimensional text, 128-channel packed latents, target-first ordering, reference times 10/20, mask propagation through both processors, and dual-branch CFG. NumPy patchify/pack and inverse transforms on nonsquare, multichannel tensors are exactly equal elementwise. S2 S4 S6 S24 S25 |
| Three substantive issues in the original external review | CheckSpec/EvaluationSpec channels are defined; $\phi$ learning response, LR pilots, and single-component controls are scheduled; repair labels follow DiT SFT and sampler locking, with data/evaluator costs accounted. No evidence was found that changing models broke these repairs |
| SFT and joint probability | Complete prefixes, valid supervision masks, global denominators, zero renders, scores of all actual renders, all-step positive-variance SDE, actual-transition-variance KL, and one same-version update on both sides form a consistent whole. Added multiround enumeration appears below. S6 S9 S27 |
| Hardware and resources | Recomputed 27B base plus training/reference states: 19.5053 GiB, plus approximately 1.54 MiB for norm precision; 9B: 16.9102 + 0.5957 GiB. Both are static quantities, with existing 42/28 GiB measured-peak gates. No cross-machine gradient synchronization or pooled-memory assumption is added |

Supplementary generation-entry source locations: HF acceptance of a custom Cache, and extension of generated-token types and positions. These checks support the existing implementation contract without adding required revisions.

**Added multiround joint-gradient verification.** In a finite Bernoulli environment, the same $\theta$ controls the root plan, whether to render initially, and the continuation decision after observing the first image. The same $\phi$ controls both renders. Let $s(x)=1/(1+e^{-x})$, $\theta=.27$, and $\phi=-.31$:

**Source page 47**

$$
r\sim\operatorname{Bernoulli}(s(\theta)),
$$
$$
u\mid r\sim\operatorname{Bernoulli}(s(\theta-.3r)),
$$
$$
y\mid r,u=1\sim\operatorname{Bernoulli}(s(\phi+.4r)),
$$
$$
v\mid r,y,u=1\sim\operatorname{Bernoulli}(s(\theta+.8y-.5)),
$$
$$
z\mid r,y,u=v=1\sim\operatorname{Bernoulli}(s(\phi-.6y+.2r)).
$$

When $u=0$, there are zero renders and $R=.14+.07r$. When $u=1,v=0$, deliver the first image and use $R=.85y+.12r-.08$. When $u=v=1$, deliver the second image and use $R=1.1z+.2\mathbf1[z=r]-.16-.09y$. Each root has seven suffix outcomes. With N = M = 2, enumerate all 9,604 combinations and use centered finite differences with step $10^{-5}$:

| Check | Actual CPU result in this round |
|---|---:|
| Total probability of combinations | 1.0000000000000069 |
| Direct expected $\theta,\phi$ score derivatives | (.03971170210184353, .10282906553219126) |
| Hierarchical RLOO expected $\theta,\phi$ values | (.039711702101844146, .10282906553219191) |
| Maximum absolute RLOO versus direct-score difference | $6.52\times10^{-16}$ |
| Maximum absolute finite-difference versus direct-score difference | $1.14\times10^{-12}$ |
| $\phi$ gradient bias when incorrectly removing the score of the abandoned first render that influences subsequent decisions | .009262809558049229 |

The results support the retention of scores for all actual generation and the hierarchical baseline already adopted in §§11-13. There is no gap that omits intermediate-image gradients in pursuit of only the final image. The valid zero-render enumeration, complete-prefix head VJP, SFT reduction, and budget checks in §23.1 were also rerun with consistent results. All of the above are source/metadata and CPU-arithmetic checks; no models were loaded, research system implemented, or training started.

**After finalization, the next step is implementation through the established startup gates, not continued expansion of the design review.** Strict loading of real models, GPU operators/peaks, sampling quality, and learning gains still require evidence from actual machines. Specific entry points, thresholds, and repair orders have been provided; these are not design gaps left for implementers to fill. Basic operation and normal generation can be accepted item by item accordingly. The actual improvements from SFT, Agent RL, and joint training are to be established through the fair comparisons in §20.


## Preserved Hyperlink Destinations from Source Pages 24-47

This translation index preserves the original PDF's embedded URL targets, which are not visible in plain-text extraction. The numbered S citations and descriptions above retain their original source identifiers. Repeated identical targets within one page are listed once.

| Source page | Link | Original destination |
|---|---|---|
| 25 | P25-01 | [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| 26 | P26-01 | [pipeline_flux2_klein.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) |
| 27 | P27-01 | [8cfbac17765004afb68fef08a4e011a4c8daf555](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555) |
| 27 | P27-02 | [pipeline_flux2_klein.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) |
| 27 | P27-03 | [879042cf5707f8b90daa98d147d7deac2317c5da](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) |
| 29 | P29-01 | [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| 29 | P29-02 | [879042cf5707f8b90daa98d147d7deac2317c5da](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) |
| 31 | P31-01 | [attention_dispatch.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/attention_dispatch.py#L3187-L3219) |
| 33 | P33-01 | [v20_update_transaction.py](https://github.com/waltstephen/UMM-Reflection/blob/1e4194505d53c06cbaa8f58171130bf749e9f660/src/unify_rl/train/v20_update_transaction.py) |
| 34 | P34-01 | [minor-version-compatibility.html](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) |
| 34 | P34-02 | [a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e](https://github.com/huggingface/peft/tree/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e) |
| 34 | P34-03 | [8cfbac17765004afb68fef08a4e011a4c8daf555](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555) |
| 35 | P35-01 | [transformer_flux2.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py) |
| 35 | P35-02 | [attention_dispatch.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/attention_dispatch.py#L3187-L3219) |
| 36 | P36-01 | [b6eec40a8a6994aa380497e33b1964742d7e028b](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/tree/b6eec40a8a6994aa380497e33b1964742d7e028b) |
| 38 | P38-01 | [transformer_flux2.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py) |
| 38 | P38-02 | [a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e](https://github.com/huggingface/peft/tree/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e) |
| 38 | P38-03 | [1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| 38 | P38-04 | [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| 38 | P38-05 | [cbb14f84b8312b620390dfcbe2fab69c1383b104](https://github.com/NVlabs/DiffusionNFT/tree/cbb14f84b8312b620390dfcbe2fab69c1383b104) |
| 41 | P41-01 | [6c254560879df7851233ea0fd2ae2a62764d0ef2](https://huggingface.co/datasets/osunlp/MagicBrush/tree/6c254560879df7851233ea0fd2ae2a62764d0ef2) |
| 41 | P41-02 | [a6e82d2289e8d418f27f0adee77908b07060eea3](https://github.com/facebookresearch/GenEval2/tree/a6e82d2289e8d418f27f0adee77908b07060eea3) |
| 42 | P42-01 | [cache_utils.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/cache_utils.py#L917-L941) |
| 42 | P42-02 | [1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| 42 | P42-03 | [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| 42 | P42-04 | [a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e](https://github.com/huggingface/peft/tree/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e) |
| 43 | P43-01 | [attention_dispatch.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/attention_dispatch.py#L3187-L3219) |
| 43 | P43-02 | [50fe5162777813d869182b139e83b10743caef15](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15) |
| 43 | P43-03 | [8cfbac17765004afb68fef08a4e011a4c8daf555](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555) |
| 43 | P43-04 | [bundle.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/models/qwen_vl/bundle.py) |
| 43 | P43-05 | [53c28c9ae209583c358e2afc67d0a97f3a636afe](https://github.com/G-U-N/UniRL/tree/53c28c9ae209583c358e2afc67d0a97f3a636afe) |
| 43 | P43-06 | [6c9b12f0c619cc8ca6f50b536b24aff377525f8c](https://github.com/JIA-Lab-research/RePlan/tree/6c9b12f0c619cc8ca6f50b536b24aff377525f8c) |
| 43 | P43-07 | [879042cf5707f8b90daa98d147d7deac2317c5da](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) |
| 43 | P43-08 | [cdf272f3514353aab04dc2d9fceef57a260dc156](https://github.com/Comfy-Org/ComfyUI/tree/cdf272f3514353aab04dc2d9fceef57a260dc156) |
| 43 | P43-09 | [2b90b9f5ceec907a1c18123530e92e794ad901a4](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4) |
| 43 | P43-10 | [cbb14f84b8312b620390dfcbe2fab69c1383b104](https://github.com/NVlabs/DiffusionNFT/tree/cbb14f84b8312b620390dfcbe2fab69c1383b104) |
| 43 | P43-11 | [v20_update_transaction.py](https://github.com/waltstephen/UMM-Reflection/blob/1e4194505d53c06cbaa8f58171130bf749e9f660/src/unify_rl/train/v20_update_transaction.py) |
| 43 | P43-12 | [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) |
| 43 | P43-13 | [model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json) |
| 43 | P43-14 | [preprocessor_config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/preprocessor_config.json) |
| 43 | P43-15 | [chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja) |
| 43 | P43-16 | [generation_config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/generation_config.json) |
| 43 | P43-17 | [README.md](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md) |
| 43 | P43-18 | [32773329fbe7e81a90ef971740e8ba4b0364ecf3](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/32773329fbe7e81a90ef971740e8ba4b0364ecf3) |
| 43 | P43-19 | [transformer_flux2.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py) |
| 43 | P43-20 | [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| 43 | P43-21 | [processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_vl/processing_qwen3_vl.py) |
| 43 | P43-22 | [image_processing_qwen2_vl.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py) |
| 44 | P44-01 | [pipeline_flux2_klein.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) |
| 44 | P44-02 | [scheduling_flow_match_euler_discrete.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L648-L652) |
| 44 | P44-03 | [other.py](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/utils/other.py#L151-L244) |
| 44 | P44-04 | [peft_model.py](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/peft_model.py#L1591-L1634) |
| 44 | P44-05 | [peft.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/loaders/peft.py#L508-L590) |
| 44 | P44-06 | [layer.py](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/layer.py#L328-L341) |
| 44 | P44-07 | [tuners_utils.py](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/tuners_utils.py#L1894-L1934) |
| 44 | P44-08 | [8cfbac17765004afb68fef08a4e011a4c8daf555](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555) |
| 44 | P44-09 | [diffusion.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/models/flux2_klein/diffusion.py) |
| 44 | P44-10 | [bundle.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/models/flux2_klein/bundle.py) |
| 44 | P44-11 | [vae.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/models/flux2_klein/vae.py) |
| 44 | P44-12 | [track_builder.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/train/sft/track_builder.py) |
| 44 | P44-13 | [sft.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/algorithms/sft.py) |
| 44 | P44-14 | [kernels.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/sde/kernels.py#L159-L243) |
| 44 | P44-15 | [pe.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/trainer/pe.py) |
| 44 | P44-16 | [flux_klein.py](https://github.com/JIA-Lab-research/RePlan/blob/6c9b12f0c619cc8ca6f50b536b24aff377525f8c/replan/pipelines/flux_klein.py) |
| 44 | P44-17 | [flex_attn.py](https://github.com/JIA-Lab-research/RePlan/blob/6c9b12f0c619cc8ca6f50b536b24aff377525f8c/replan/pipelines/flex_attn.py) |
| 44 | P44-18 | [flux2_klein_pipeline.py](https://github.com/G-U-N/UniRL/blob/53c28c9ae209583c358e2afc67d0a97f3a636afe/unimodel/qwenklein/flux2_klein_pipeline.py) |
| 44 | P44-19 | [edit_grpo_trainer.py](https://github.com/G-U-N/UniRL/blob/53c28c9ae209583c358e2afc67d0a97f3a636afe/unirl/trainer/edit_grpo_trainer.py) |
| 44 | P44-20 | [sd3_sde_with_logprob.py](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py#L49-L93) |
| 44 | P44-21 | [train_flux_kontext.py](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_flux_kontext.py#L840-L850) |
| 44 | P44-22 | [graph.py](https://github.com/Comfy-Org/ComfyUI/blob/cdf272f3514353aab04dc2d9fceef57a260dc156/comfy_execution/graph.py) |
| 44 | P44-23 | [caching.py](https://github.com/Comfy-Org/ComfyUI/blob/cdf272f3514353aab04dc2d9fceef57a260dc156/comfy_execution/caching.py#L83-L149) |
| 44 | P44-24 | [execution.py](https://github.com/Comfy-Org/ComfyUI/blob/cdf272f3514353aab04dc2d9fceef57a260dc156/execution.py#L733-L801) |
| 44 | P44-25 | [v20_update_transaction.py](https://github.com/waltstephen/UMM-Reflection/blob/1e4194505d53c06cbaa8f58171130bf749e9f660/src/unify_rl/train/v20_update_transaction.py) |
| 44 | P44-26 | [6c254560879df7851233ea0fd2ae2a62764d0ef2](https://huggingface.co/datasets/osunlp/MagicBrush/tree/6c254560879df7851233ea0fd2ae2a62764d0ef2) |
| 44 | P44-27 | [README.md](https://github.com/OSU-NLP-Group/MagicBrush/blob/d2d7f84d71ff23154e3f16f0c8f19392782bd201/README.md) |
| 44 | P44-28 | [image_eval.py](https://github.com/OSU-NLP-Group/MagicBrush/blob/d2d7f84d71ff23154e3f16f0c8f19392782bd201/evaluation/image_eval.py) |
| 44 | P44-29 | [hubconf.py](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/hubconf.py) |
| 44 | P44-30 | [b6eec40a8a6994aa380497e33b1964742d7e028b](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/tree/b6eec40a8a6994aa380497e33b1964742d7e028b) |
| 44 | P44-31 | [summary_scores.py](https://github.com/djghosh13/geneval/blob/af4902f24d3ca90ebbb446dd9891a59e0f82725f/evaluation/summary_scores.py) |
| 44 | P44-32 | [evaluation.py](https://github.com/facebookresearch/GenEval2/blob/a6e82d2289e8d418f27f0adee77908b07060eea3/evaluation.py) |
| 44 | P44-33 | [geneval2_data.jsonl](https://github.com/facebookresearch/GenEval2/blob/a6e82d2289e8d418f27f0adee77908b07060eea3/geneval2_data.jsonl) |
| 44 | P44-34 | [test.jsonl](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/datasets/geneval2/synthetic/test.jsonl) |
| 44 | P44-35 | [0c351dd01ed87e9c1b53cbc748cba10e6187ff3b](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) |
| 44 | P44-36 | [inference.py](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/util/inference.py#L53-L99) |
| 44 | P44-37 | [sam2_image_predictor.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py#L85-L129) |
| 44 | P44-38 | [7764ea0f912e53c92e82eb78a2a1631e92725fc8](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) |
| 44 | P44-39 | [ms_deform_attn_cuda.cu](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu#L65) |
| 44 | P44-40 | [ms_deform_attn.py](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/models/GroundingDINO/ms_deform_attn.py#L330-L350) |
| 44 | P44-41 | [Dispatch.h](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/Dispatch.h#L101-L103) |
| 44 | P44-42 | [transforms.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/utils/transforms.py#L76-L104) |
| 44 | P44-43 | [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| 44 | P44-44 | [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| 44 | P44-45 | [dinov2_vitb14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) |
| 44 | P44-46 | [86b5e0934494bd15c9632b12f734a8a67f723594](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594) |
| 44 | P44-47 | [train_nft_sd3.py](https://github.com/NVlabs/DiffusionNFT/blob/cbb14f84b8312b620390dfcbe2fab69c1383b104/scripts/train_nft_sd3.py) |
| 44 | P44-48 | [vllm_rollout_emu_agentic.py](https://github.com/bubble65/EMU-Agentic-PostTrain/blob/c4a122136a2621070e58cec7ae4ce574f79b7381/RL/UniVR_RL/verl/workers/rollout/emu_agentic/vllm_rollout_emu_agentic.py#L478-L498) |
| 44 | P44-49 | [agent_workflow_engine.py](https://github.com/tulerfeng/Gen-Searcher/blob/e5078d31859bafee6b6b610f0cd40095cc72e2a4/Gen-DeepResearch-RL/rllm/rllm/engine/agent_workflow_engine.py#L602-L618) |
| 44 | P44-50 | [www.nvidia.com source](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20%281%29.pdf) |
| 44 | P44-51 | [www.nvidia.com](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) |
| 44 | P44-52 | [gpus](https://developer.nvidia.com/cuda/gpus) |
| 44 | P44-53 | [minor-version-compatibility.html](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) |
| 44 | P44-54 | [pytorch.org](https://pytorch.org/get-started/previous-versions/) |
| 44 | P44-55 | [modeling_layers.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/modeling_layers.py#L59-L92) |
| 44 | P44-56 | [adam.py](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/optim/adam.py#L528-L546) |
| 44 | P44-57 | [base.py](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/config/base.py#L74-L99) |
| 44 | P44-58 | [bundle.py](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/models/qwen_vl/bundle.py#L38-L94) |
| 44 | P44-59 | [processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_vl/processing_qwen3_vl.py) |
| 44 | P44-60 | [5.5.0](https://github.com/tesseract-ocr/tesseract/tree/5.5.0) |
| 44 | P44-61 | [4.1.0](https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0) |
| 44 | P44-62 | [config.json](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/1e6810acf5d8ebd189b4e75590877b64358e454c/config.json) |
| 44 | P44-63 | [model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/1e6810acf5d8ebd189b4e75590877b64358e454c/model.safetensors.index.json) |
| 44 | P44-64 | [preprocessor_config.json](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/1e6810acf5d8ebd189b4e75590877b64358e454c/preprocessor_config.json) |
| 44 | P44-65 | [modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) |
| 44 | P44-66 | [model.py](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/model.py#L25-L35) |
| 44 | P44-67 | [util.py](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/util.py) |
| 44 | P44-68 | [text_encoder.py](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/text_encoder.py) |
| 44 | P44-69 | [convert_flux2_to_diffusers.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/scripts/convert_flux2_to_diffusers.py#L430-L451) |
| 44 | P44-70 | [train_dreambooth_lora_flux2_klein_img2img.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py#L704-L710) |
| 44 | P44-71 | [config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) |
| 44 | P44-72 | [attention_dispatch.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/attention_dispatch.py#L3187-L3219) |
| 44 | P44-73 | [sdp_utils_cpp.h](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L259-L266) |
| 44 | P44-74 | [sdp_utils.cpp](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L456-L513) |
| 44 | P44-75 | [functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/functional.py#L917-L948) |
| 44 | P44-76 | [_functions.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/08a9956b939857ae2c4fbd56753960cc370fa7cf/bitsandbytes/autograd/_functions.py#L365-L386) |
| 44 | P44-77 | [installation](https://huggingface.co/docs/bitsandbytes/main/en/installation) |
| 44 | P44-78 | [cache_utils.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/cache_utils.py#L917-L941) |
| 44 | P44-79 | [cache_utils.py](https://github.com/huggingface/transformers/blob/c0d024fbc873ab0325e1b40fc1bd69069609de65/src/transformers/cache_utils.py#L1189-L1195) |
| 44 | P44-80 | [fused_recurrent.py](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/gated_delta_rule/fused_recurrent.py) |
| 44 | P44-81 | [chunk.py](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/gated_delta_rule/chunk.py) |
| 44 | P44-82 | [cd81f0413cad2fc1e6f17e785ac39f59aae690cd](https://github.com/Dao-AILab/causal-conv1d/tree/cd81f0413cad2fc1e6f17e785ac39f59aae690cd) |
| 44 | P44-83 | [rloo_trainer.py](https://github.com/huggingface/trl/blob/636cb0fccbea797cb6ca8f17e0d1a45847c43ae6/trl/trainer/rloo_trainer.py) |
| 46 | P46-01 | [1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| 46 | P46-02 | [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| 46 | P46-03 | [a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e](https://github.com/huggingface/peft/tree/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e) |
| 46 | P46-04 | [cache_utils.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/cache_utils.py#L917-L941) |
| 46 | P46-05 | [transformer_flux2.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_flux2.py) |
| 46 | P46-06 | [pipeline_flux2_klein.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) |
| 46 | P46-07 | [8cfbac17765004afb68fef08a4e011a4c8daf555](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555) |
| 46 | P46-08 | [50fe5162777813d869182b139e83b10743caef15](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15) |
| 46 | P46-09 | [attention_dispatch.py](https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/attention_dispatch.py#L3187-L3219) |
| 46 | P46-10 | [879042cf5707f8b90daa98d147d7deac2317c5da](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) |
| 46 | P46-11 | [rloo_trainer.py](https://github.com/huggingface/trl/blob/636cb0fccbea797cb6ca8f17e0d1a45847c43ae6/trl/trainer/rloo_trainer.py) |
| 46 | P46-12 | [utils.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/generation/utils.py#L1869-L1881) |
| 46 | P46-13 | [utils.py](https://github.com/huggingface/transformers/blob/ddb849abe009d1089e6c691bfc897f27211c663c/src/transformers/generation/utils.py#L930-L947) |
