# Implementation and experiment status

This repository preserves two different implementations. They must not be treated as interchangeable experimental evidence.

| Implementation | Location | GPU validation status |
| --- | --- | --- |
| Adapted small-model pilot | Repository root | GPU experiments were performed. The recorded formal evaluation is partial: 637 of 1,280 cases. |
| Original larger-model implementation | `variants/large-model/` | **Not run or validated on a GPU.** CPU checks do not establish GPU correctness, feasibility, memory use, or model quality. |

The original research plan describes a 27B controller, a 9B renderer, and a 32B judge. It was written **before the small-model GPU experiments**. Its proposed measurements, budgets, and acceptance gates are a historical design, not completed experimental results. The English translation preserves that historical content rather than updating it to match the later pilot.

The small-model pilot instead uses Qwen3.5-4B, FLUX.2-klein-base-4B, and Qwen3-VL-8B-Instruct, with subsequent implementation changes and experiment-specific settings. Pilot outcomes do not validate the larger-model configuration.

The formal evaluation was frozen with Base 141, SFT 320, Agent RL 160, and Joint RL 16 completed cases. The remaining 643 cases were not completed in the recorded run. Existing reports include failures and incomplete historical checkpoint payloads; preservation does not resolve those experimental limitations.

See [implementation details](IMPLEMENTATIONS.md), [validation evidence](VALIDATION.md), and [asset restoration](ASSETS.md). The root README intentionally remains a placeholder.

Code and research-data integrity take priority over cosmetic language changes. Original task text, identifiers, third-party code, legal notices, and historical evidence are retained when changing them would alter behavior or provenance.
