import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from agentic_image.core.types import PrivateEvaluation


parser = argparse.ArgumentParser()
parser.add_argument("--batch", required=True)
arguments = parser.parse_args()
root = Path("data/calibration/ai_reserved")
reserved = [json.loads(line) for line in (root / "reserved_600.jsonl").read_text().splitlines()]
output = root / "independent_requests.jsonl"
existing = {row["calibration_index"] for row in map(json.loads, output.read_text().splitlines())} if output.exists() else set()
batch = json.loads(Path(arguments.batch).read_text())
records = []
for key, draft in batch.items():
    index = int(key)
    if index in existing:
        raise ValueError("Existing independent request drafts must not be overwritten")
    row = reserved[index]
    atoms = []
    for atom_index, (span, predicate, expected, route) in enumerate(draft["atoms"]):
        if row["instruction"].count(span) != 1:
            raise ValueError(f"Index {index}: draft span is not an exact unique original quote")
        atoms.append({"eval_id": f"eval:{atom_index:04d}", "raw_span": span,
                      "predicate": predicate, "expected_value": expected,
                      "checker_route": route, "threshold": 0.8, "mandatory": True})
    evaluation = PrivateEvaluation.model_validate({
        "private_task_key": row["sample_id"], "eval_atoms": atoms,
        "source_baseline_observations": [], "evaluator_version": "assistant-calibration-request-draft-v1",
        "annotation_provenance": {"origin": "assistant_text_or_public_source_draft",
                                  "input_scope": "original_public_request_only",
                                  "human_reviewed": False, "private_target_seen": False,
                                  "controller_plan_used": False, "calibrated": False,
                                  "source_grounding_note": draft.get("source_grounding_note"),
                                  "preserved_eval_ids": [],
                                  "created_at": datetime.now(timezone.utc).isoformat()},
    })
    records.append({"calibration_index": index, "sample_id": row["sample_id"],
                    "family_id": row["family_id"], "calibration_fold": row["calibration_fold"],
                    "request": row["instruction"], "evaluation": evaluation.model_dump(mode="json"),
                    "review_type": "ai_draft", "human_reviewed": False,
                    "source_correct": None, "target_correct": None, "ai_visual_review_complete": False})
with output.open("a") as stream:
    for row in records:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"added": len(records), "total": len(existing) + len(records),
                  "atoms_added": sum(len(row["evaluation"]["eval_atoms"]) for row in records),
                  "ai_visual_labels": 0, "human_reviewed": False}))
