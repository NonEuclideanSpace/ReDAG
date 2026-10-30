# data/

Everything under this folder except this README is git-ignored. It is produced by `scripts/data/*` and `scripts/launch/01_prepare_data.sh` and `02_data_gpu.sh`.

```
data/
  raw/                        downloaded or provided sources
    magicbrush/data/*.parquet osunlp/MagicBrush (train 51 shards, dev 4)
    corebench/splits/*.jsonl  lioooox/T2I-CoReBench (12 x 90 prompts with yes/no checklists)
    benchmarks/lgbench/       t2i_1000.json, i2i_1000.json, i2i_source_images*/ (TruemanV5/LGBench)
    benchmarks/longbench_t2i/ instruction.jsonl
  processed/
    magicbrush/               aligned 512 px turn images, binary edit masks, turns_<split>.jsonl
    editor_sft/<split>/       crops around each edit (source, target, mask)
    sessions/*.jsonl          unified raw sessions; annotate_ids.txt = sessions sent to the LLM
    graphs/*.jsonl            validated condition DAGs (annotate_graphs.py)
    canvases/                 initial canvases (render_canvases.py) and requests.jsonl
  final/                      inputs of scripts/*.py (image paths are relative to this folder)
```

## Files in `final/` and the script that reads them

| file | consumer | one record |
|---|---|---|
| `editor_sft_train.jsonl` | `run_sft_editor.py` | `source`, `target`, `mask` crop paths (same size; mask 255 = editable), `instruction` |
| `planner_sft.jsonl` (+`_dev`) | `run_sft_planner.py` | `user_prompt`, `previous_dag` or null, `response` = new nodes/edges (+`superseded_ids`) |
| `controller_rs_states.jsonl` | `collect_controller_rs.py` | `id`, `image`, `graph` (N_state = 1000 failure states) |
| `controller_sft.jsonl` | `run_sft_controller.py` | produced by rejection sampling: `graph`, `scores`, `target_node_id`, `action` |
| `rl_train.jsonl`, `rl_dev.jsonl` | `run_rl_alternating.py` | failure states with cached `scores`, `detected_counts`, grounded boxes; session-disjoint |
| `eval_main.jsonl` | `run_eval_benchmarks.py --exp_id 1/3` | `image`, `k`, `turns: [{user_prompt, graph, superseded_ids}]`; final K is exactly 10, 20 or 30 |
| `eval_credit.jsonl` | `run_eval_benchmarks.py --exp_id 2` | subset of `eval_main.jsonl` |

Graphs use the repository schema: `nodes` with `node_id, node_type, question, entity_phrase, bbox, weight, threshold, expected_count, scope, depends_on`, plus `comp_edges`, `supp_edges` and `int_edges`.

## Splits

Evaluation is held out before anything else is sampled. The evaluation pool is MagicBrush dev plus a hash-selected half of every other benchmark. Training material (planner pairs, rejection-sampling states, RL states) comes only from the rest, and RL train and dev states are session-disjoint. `final/manifest.json` records the counts, sources and seed. `python scripts/data/check_data.py` checks every file with the repository's own loaders.
