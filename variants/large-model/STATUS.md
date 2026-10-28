# Original large-model implementation: GPU UNVERIFIED

**This implementation has never been run on GPU.** It is the original larger-model implementation, preserved separately from the later small-model pilot experiments.

Its `configs/a6000.yaml` specifies a Qwen3.8-27B actor, FLUX.2-klein-base-9B renderer, and Qwen3-VL-32B-Instruct judge across four GPU placements. The plan in `plan/Agentic_Image_Generation_v3.pdf` predates the small-model pilot experiments.

The original 131 source files were copied without content changes. The README, deployment records, and provenance notes are historical materials. Their presence does not establish a completed GPU installation, training run, benchmark, or end-to-end validation for this implementation.

Publication checks passed all 34 existing CPU tests on macOS arm64 with Python 3.12.14. No CUDA device was available. These tests include small synthetic model computations and do not validate the full configured model sizes or the intended GPU deployment.

Keep this variant separate from the root pilot implementation. Both expose the `agentic_image` Python package and `agentic-image` command. Run commands from this directory in a separate process or environment.

The source map and excluded generated/environment directories are recorded in `../../provenance/code-source-map.json`. Detailed validation is recorded in `../../docs/VALIDATION.md` and `../../provenance/code-validation.json`.
