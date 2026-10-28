# Implementations and evidence boundaries

The repository preserves two distinct implementations. Their validation status must be kept separate.

| Implementation | Location | Model profile | GPU status |
| --- | --- | --- | --- |
| Small-model pilot | Repository root | Qwen3.5-4B actor, FLUX.2-klein-base-4B renderer, Qwen3-VL-8B-Instruct judge | **PARTIAL GPU VALIDATION** from the historical rental experiments. The full planned experiment was not completed. |
| Original large-model implementation | `variants/large-model/` | Qwen3.8-27B actor, FLUX.2-klein-base-9B renderer, Qwen3-VL-32B-Instruct judge | **GPU UNVERIFIED. This implementation has never been run on GPU.** |

The original plan predates the small-model GPU experiments. It describes the intended research design and does not establish that either implementation completed that design. The original PDF is retained with the large-model variant; the repository's plan materials are in `docs/plan/`.

## Small-model pilot

The root source was reconstructed from the final handover archive `pilot-small-code-20261028-final.tar.gz`. All 305 ordinary archive members have a destination in `provenance/code-source-map.json`. The archive contains 77 package files, 133 scripts, 48 test files, 16 configuration files, 23 deployment records, four monitor files, and four remaining project files.

The archived README is preserved at `docs/pilot-overview-original.md`; the repository README is a separate placeholder. The internal package name remains `agentic_image`, and the distribution and command remain `agentic-image`. This retains imports, command lines, serialized references, and the existing script layout.

The root also contains `configs/a6000.yaml`, an original large-model profile retained in the final pilot snapshot. Its presence is not evidence of a large-model GPU run. Use `configs/pilot-small.yaml` and the explicitly named pilot/seed configurations when interpreting the small-model experiments.

Two small implementation changes were made during publication preparation:

1. `monitor/transfer_server.py` adopts the four-line `manual_handoff` display handling from the original workspace's newer pilot copy. It clears the displayed active transfer and live indicator after a manual handoff.
2. `scripts/sync_rental_result_archives.py` restores the historical `reused_verified_download` result field. It is true only for a receipt that explicitly records content checksum verification. Direct transfers checked by authenticated transport and exact byte length remain distinguishable through `reused_existing_download` and `content_checksum_verified`.

The archive-sync tests were updated to supply explicit verification evidence in their mocked receipts, and one regression test now checks that a direct receipt never claims content verification. No model, objective, sampler, training, or evaluation algorithm was rewritten. The source map records original hashes and final hashes for every preserved source file.

Five prose-only Python comment tokens were removed from four pilot delivery scripts while preserving whitespace and line structure. Python ASTs before and after removal are identical. The original large-model variant was not edited, and docstrings were retained.

## Original large-model implementation

The separate variant preserves the original workspace's package, tests, scripts, configurations, deployment records, dependency files, README, provenance notices, and plan. These 131 source files were copied without editing their contents. Its own `src/agentic_image` remains independent of the pilot package.

Run its commands and tests from `variants/large-model/` in a separate process or environment. Both implementations use the same Python package and distribution names; installing them into one environment as simultaneous editable packages is ambiguous.

`variants/large-model/STATUS.md` states the validation boundary. Existing deployment records and the archived README remain historical source material and do not override the explicit GPU-unverified status. CPU tests exercise small synthetic models and local fixtures; they do not establish that the configured 27B/9B/32B profile fits, trains, or evaluates on the intended GPU hardware.

## Environment and portability

The training environment is Python 3.12 with the pinned upstream model libraries and a CUDA 13.0 PyTorch build in the historical setup. The separate visual-tools environment uses Python 3.10 and CUDA 12.4. Preserve these as separate environments.

The original setup scripts install PyTorch and the GPU libraries separately from the base `pyproject.toml` dependencies. Installing the package alone does not supply the complete training stack. Frozen requirements and deployment records include machine-local editable paths, and several operational scripts retain the original rental-server and external-drive paths. These files were preserved for reproducibility and provenance; they are not portable defaults for a new machine.

Do not execute historical transfer, reclamation, or finalization scripts as a generic installation step. Review their configured paths and operational purpose first. Model weights, project checkpoints, datasets, environment bundles, and archive metadata have separate asset inventories and restoration requirements.

See `docs/VALIDATION.md` and `provenance/code-validation.json` for the checks performed during publication preparation and their limits.
