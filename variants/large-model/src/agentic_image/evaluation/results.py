import json
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmarks import read_jsonl


def geneval_result_rows(path, method, training_seed, eval_seed=101):
    records = read_jsonl(path)
    categories = Counter(record["tag"] for record in records)
    rows = []
    for record in records:
        index = Path(record["filename"]).parent.parent.name
        rows.append(
            {
                "method": method,
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "task_id": f"geneval:{index}",
                "family_id": f"geneval:{index}",
                "category": record["tag"],
                "value": float(record["correct"]),
                "task_weight": 1.0 / categories[record["tag"]],
                "metric": "geneval_category_macro",
            }
        )
    return rows


def geneval2_result_rows(public_prompt_path, scores_path, method, training_seed, eval_seed=101):
    prompts = read_jsonl(public_prompt_path)
    scores = json.loads(Path(scores_path).read_text())
    if len(prompts) != len(scores):
        raise ValueError(
            "GenEval2 score output must include every prompt, including no-image outcomes"
        )
    rows = []
    for index, (prompt, values) in enumerate(zip(prompts, scores)):
        values = np.asarray(values, dtype=float)
        if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid GenEval2 question probabilities")
        value = 0.0 if np.any(values == 0) else float(np.exp(np.log(values).mean()))
        task_id = prompt.get("task_id", f"geneval2:{index}")
        rows.append(
            {
                "method": method,
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "task_id": task_id,
                "family_id": prompt.get("family_id", task_id),
                "value": value,
                "metric": "geneval2_soft_tifa_gm",
            }
        )
    return rows
