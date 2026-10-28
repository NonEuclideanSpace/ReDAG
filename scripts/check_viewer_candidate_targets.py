import json
from pathlib import Path

from agentic_image.data.families import HammingIndex
from agentic_image.data.io import read_jsonl, write_report


root = Path("data")
raw = list(read_jsonl(root / "raw/all_saved.jsonl"))
required = {row[field] for row in raw if row["official_split"] in {"dev", "test"}
            for field in ("source_image_id", "target_image_id")}
images = {row["original_artifact_id"]: row for row in read_jsonl(root / "viewer/images.jsonl")
          if row["status"] == "downloaded"}
if required - images.keys():
    raise ValueError("Official held-out evidence is incomplete")
index = HammingIndex()
for identifier in required:
    index.add(int(images[identifier]["phash"]), identifier)
targets = list(read_jsonl(root / "viewer/training_candidate_images.jsonl"))
results = []
for row in targets:
    matches = index.query(int(row["phash"]), radius=6)
    results.append({"sample_id": row["sample_id"], "target_image_id": row["artifact_id"],
                    "heldout_near_matches": matches, "quarantined": bool(matches)})
report = {"profile": "official-viewer-reencoded-v1", "radius": 6,
          "official_reference_count": len(required), "targets_checked": len(targets),
          "scope": "32 actual candidate targets against all official held-out sources and targets",
          "quarantined": sum(row["quarantined"] for row in results), "results": results,
          "human_reviewed": False, "source_isolation_report": "data/public/protocol_pilot_tasks.report.json"}
write_report(root / "review/viewer_candidate_target_isolation.json", report)
print(json.dumps({key: value for key, value in report.items() if key != "results"}))
