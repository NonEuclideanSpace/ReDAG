import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--verdicts", required=True)
parser.add_argument("--output", required=True)
arguments = parser.parse_args()
base = Path("data/calibration/ai_reserved")
drafts = {row["calibration_index"]: row for row in map(json.loads, (base / "independent_requests.jsonl").read_text().splitlines())}
prepared = {row["calibration_index"]: row for row in map(json.loads, (base / "batch_600.jsonl").read_text().splitlines())}
verdicts = json.loads(Path(arguments.verdicts).read_text())
output = Path(arguments.output)
if output.exists():
    raise ValueError("Existing actual review evidence must not be overwritten")
records = []
for key, decisions in verdicts.items():
    index = int(key)
    row, draft = prepared[index], drafts[index]
    if row.get("quarantined"):
        raise ValueError("Quarantined family cannot enter ordinary calibration labels")
    images = {image["candidate_role"]: image for image in row["images"]}
    for role in ("source", "target"):
        if len(decisions[role]) != len(draft["evaluation"]["eval_atoms"]):
            raise ValueError("Actual verdict count does not match the frozen atoms")
        for atom, (correct, observation) in zip(draft["evaluation"]["eval_atoms"], decisions[role]):
            if correct is not None and type(correct) is not bool:
                raise ValueError("Actual labels must be Boolean or unknown")
            paths = [str((base / "contact_sheets" / f"{index // 4 * 4:03d}-{index // 4 * 4 + 3:03d}.jpg").resolve())]
            paths.extend(str((base / path).resolve()) for path in decisions.get("reviewed_extra", []))
            if any(not Path(path).is_file() for path in paths):
                raise ValueError("Actual reviewed image evidence is missing")
            records.append({"calibration_index": index, "sample_id": row["sample_id"], "family_id": row["family_id"], "calibration_fold": row["calibration_fold"], "candidate_id": row["sample_id"] + ":" + role, "candidate_role": role, "candidate_image_id": images[role]["artifact_id"], "candidate_value_id": images[role]["value_id"], "source_image_id": images["source"]["artifact_id"], "eval_id": atom["eval_id"], "atom": atom, "correct": correct, "known": correct is not None, "reviewer": "codex_ai_calibration_reviewer_b", "review_type": "ai", "human_reviewed": False, "review_evidence": {"reviewer": "codex_ai_calibration_reviewer_b", "observation": observation, "record_path": str(output.resolve())}, "reviewed_image_paths": paths, "candidate_image_files": [str((base / "local_images" / f"{index:03d}_{item}.jpg").resolve()) for item in ("source", "target")], "reviewed_at": datetime.now(timezone.utc).isoformat(), "raw_judge_scores_seen": False, "draft_evaluation_sha256": sha256(json.dumps(draft["evaluation"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
report = {"families": len(verdicts), "records": len(records), "true": sum(row["correct"] is True for row in records), "false": sum(row["correct"] is False for row in records), "unknown": sum(row["correct"] is None for row in records), "indices": sorted(map(int, verdicts)), "review_type": "ai", "human_reviewed": False, "raw_judge_scores_seen": False}
output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
