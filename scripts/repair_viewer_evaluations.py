import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.config import load_config
from agentic_image.data.io import append_jsonl, read_jsonl, write_report
from agentic_image.data.proposals import _evaluation
from agentic_image.runtime.components import Components
from propose_viewer_review import ReviewJudge, emit


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
arguments = parser.parse_args()
output = Path(arguments.output)
rows = {row["sample_id"]: row for row in read_jsonl(output / "proposals.jsonl")}
pending = [row for row in rows.values() if "evaluation" not in row]
if not pending:
    raise ValueError("No missing evaluation proposals require repair")
config = load_config(arguments.config)
config["run_dir"] = str(output / "repair_runtime")
config["judge"]["calibration"] = None
config["devices"]["judge"] = "cuda:0"
components = Components(config)
components.load_judge(private=False)
started = time.monotonic()
repaired = 0
for index, row in enumerate(pending):
    correction = "Propose explicit original-request requirements for human review even when a referent is ambiguous. Preserve the exact original noun or pronoun and set ambiguous=true in that case. Do not guess which object or delete an explicitly requested change. An instruction such as changing an object's color still has an atom describing that requested color, even when its precise referent needs human review. You are drafting requirements, not deciding target success. Do not return empty atoms for a nonempty visual editing request. "
    evidence = []
    judge = ReviewJudge(components.judge, evidence, correction)
    image = components.artifacts.get_image(row["source_image_id"])
    try:
        evaluation, raw, covered = _evaluation(judge, row["request"], [image], row["sample_id"], 12)
        row.update(evaluation=evaluation.model_dump(mode="json"), evaluation_proposal=raw,
                   automated_coverage=covered, evaluation_repair="single_source_only_human_review_draft")
        repaired += 1
    except (ValueError, KeyError, TypeError) as error:
        row["errors"].append({"stage": "bounded_evaluation_repair", "error": str(error)})
    row["evaluation_repair_call_outputs"] = evidence
    append_jsonl(output / "proposals.jsonl", [row])
    emit(output, {"stage": "evaluation_repair", "completed": index + 1,
                  "total": len(pending), "sample_id": row["sample_id"],
                  "has_evaluation": "evaluation" in row,
                  "elapsed_seconds": round(time.monotonic() - started, 2)})
judge = None
components.release_frozen()
write_report(output / "evaluation_repairs.report.json", {
    "attempted": len(pending), "repaired": repaired,
    "missing_after": len(pending) - repaired, "human_reviewed": 0,
    "calibrated": False, "paired_training_admitted": 0,
    "elapsed_seconds": time.monotonic() - started,
})
