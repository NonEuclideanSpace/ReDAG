# Sources and preservation boundaries

The current root implementation is the adapted small-model pilot from the final handover snapshot, `pilot-small-code-20261028-final.tar.gz`. The original larger-model implementation is preserved separately in `variants/large-model/`. The two variants intentionally retain the same internal Python package name and should be run in separate environments or processes.

The original research plan was written before the small-model GPU experiments. Its complete English translation and unchanged source PDF are in `docs/plan/`. Historical first-person verification statements in the plan belong to the source document; they are not claims that publication preparation repeated those checks. The larger-model implementation has not been run on a GPU.

## Implementation provenance

- `provenance/code-source-map.json` records every selected source file, its original hash, its publication location, and any reviewed change.
- `provenance/code-validation.json` records the actual local CPU checks and their limits.
- `provenance/assets/project-versions.jsonl.gz` preserves the recorded experiment-file versions.
- `provenance/assets/project-latest.jsonl.gz` identifies the latest recorded version of each project path.
- `provenance/handover/` retains the original handover records. Older incomplete status fields are historical observations, not the current publication status.

## Third-party software

The implementations use PyTorch, Transformers, PEFT, Diffusers, bitsandbytes, Flash Linear Attention, causal-conv1d, GroundingDINO, SAM 2, DINOv2, and other dependencies documented by the original setup scripts and frozen environments. Preserve the separate main and visual-tool environments: their Python, PyTorch, and CUDA versions differ.

The original larger-model source references and attribution are retained in `variants/large-model/SOURCES.md`. They describe that original configuration and must not be relabeled as the pilot's model inventory. Applicable bundled notices are retained in `THIRD_PARTY_LICENSES/` and in archived third-party source packages. This repository does not replace upstream licenses or assign a new license to third-party content.

Pinned third-party source revisions used by the setup scripts include:

| Project | Revision |
| --- | --- |
| GroundingDINO | `856dde20aee659246248e20734ef9ba5214f5e44` |
| SAM 2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| DINOv2 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| causal-conv1d | `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` |
| Flash Linear Attention | `413ad2c655ece57b94bc9ddc51fdbd43cf09d984` |

## Pilot models

| Role | Repository | Revision |
| --- | --- | --- |
| Actor | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Renderer | `black-forest-labs/FLUX.2-klein-base-4B` | `a3b4f4849157f664bdbc776fd7453c2783562f4d` |
| Judge | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |

Configuration, tokenizer, index, model-card, and license metadata are preserved under `models/metadata/`. Large base-model weight files are intentionally omitted. Project-trained adapters, optimizer states, training traces, conditions, and experiment outputs are research assets rather than base-model downloads and are included in the release selection.

## Original research data

Original task text, annotations, numerical values, image identity, and source links are preserved. A task that requires Chinese text in an image is not translated into a different task. AI review records remain labeled as AI reviews, including `human_reviewed=false` where recorded. Preserving a historical failed run does not make it a successful result.

The offline package inventory records known gaps. Existing environment archives and fixed source revisions improve recoverability, but neither the package inventory nor the local CPU tests prove a complete offline GPU restoration on a new machine.
