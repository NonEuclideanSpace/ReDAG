# Code validation

Publication checks completed on 2026-10-28 using an existing Python 3.12.14 environment on macOS arm64. No CUDA device was available, and no dependency packages were installed during these checks. The machine-readable results and per-test outcomes are in `provenance/code-validation.json`.

| Check | Small-model pilot | Original large-model variant |
| --- | --- | --- |
| Existing tests plus one targeted pilot regression | 316 passed; 211 subtests passed | 34 passed |
| Python 3.12 syntax parsing | 254 files passed | 90 files passed |
| Shell syntax | 7 files passed | 6 files passed |
| CLI help smoke check | Passed | Passed |
| Original source mapping | All 305 archive members mapped | All 131 original files mapped and content unchanged |
| GPU execution during publication preparation | Not run | Not run |

**The original large-model implementation has never been run on GPU.** The root pilot has historical partial GPU evidence from the rental experiments. The local CPU checks above do not extend either claim to a complete GPU training, evaluation, or restoration run.

## What the tests establish

The suites exercise the existing DAG/runtime logic, numerical routines, small synthetic model computations, data/evaluation rules, and archive/restore bookkeeping. The pilot suite also checks GPU queue behavior through local fixtures and mocks; those checks do not launch remote GPU work.

Both suites emit one PEFT warning in the tiny Flux model test about an existing `peft_config` while attaching multiple adapters. The affected test passes. There were no skipped tests in the final runs.

The validation environment passed `python -m pip check`. Relevant installed versions were PyTorch 2.13.0, Transformers 5.12.1, PEFT 0.20.0, Diffusers 0.38.0, NumPy 2.5.3, and pytest 9.1.1. The environment contains the packages needed by these tests. This is not a clean-install verification of the archived requirements on a new machine.

## Issues found and resolved

The final handover version of the archive-sync script had replaced the `reused_verified_download` result field. Existing callers and regression assertions still expected that field. The field is now retained, but it is true only when the underlying receipt explicitly records content checksum verification. A transfer accepted by SSH completion and exact length remains reusable without being described as checksum verified.

Two coordinator test fixtures were updated to provide explicit verification metadata rather than relying on the old implied contract. One additional regression verifies that a direct-transfer receipt keeps both content-verification flags false. The training and evaluation algorithms were not changed.

Five initial sampler-confirmation failures exposed missing experimental metadata in the source-only archive. The original records were restored from the project archives and checked against their member hashes. These files are required by the existing suite:

- `data/evaluation/sampler_confirm64/public_tasks.jsonl`
- `data/evaluation/sampler_confirm64/reserved.frozen.jsonl`
- `data/evaluation/sampler_confirm64/selection_rule.json`
- `data/evaluation/development64/public_tasks.jsonl`
- `data/evaluation/development64/reserved.frozen.jsonl`
- `data/calibration/thresholds.json`
- `data/calibration/thresholds.json.model.json`

No replacement experimental data was fabricated to make the tests pass.

Five prose-only comment tokens in four pilot delivery scripts were removed after the test run. Their Python ASTs were compared before and after and are identical; no executable expression changed. The source map records the updated hashes. The original large-model source was left unchanged.

## Repeating the local checks

Use a Python 3.12 environment with the recorded test dependencies. Run each implementation in a separate process from its own directory. The two implementations use the same Python package name.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m pytest -q -p no:cacheprovider
PYTHONPATH=src python -m agentic_image --help
```

From `variants/large-model/`:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m pytest -q -p no:cacheprovider
PYTHONPATH=src python -m agentic_image --help
```

The checks did not download the large base models, rerun CUDA training, execute the full planned experiment, test a new-machine complete offline restoration, or certify every historical operational path. Passing the listed checks is bounded evidence, not a guarantee that no bugs remain.
