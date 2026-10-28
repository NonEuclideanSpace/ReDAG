from datetime import datetime, timezone
import json
from pathlib import Path

from agentic_image.core.types import PrivateEvaluation
from build_complete_viewer_review import read_jsonl, write_jsonl, write_report


DRAFTS = {
    "magicbrush:513551:1": [("leave only the scissor on  the cup", "Only the scissor remains on the cup; other items on the cup are absent.", "only the scissor on the cup")],
    "magicbrush:435242:1": [("leave only words on the page", "Words remain on the page.", "words present"), ("leave only words on the page", "Non-word content on the page is absent.", "only words on the page")],
    "magicbrush:327726:1": [("change the table for a dog", "The requested table is replaced by a dog.", "dog replacing the table")],
    "magicbrush:181992:1": [("let the car be a sports car", "The car referred to in the request is a sports car.", "sports car")],
    "magicbrush:236951:1": [("add a bullet train", "A bullet train is added to the scene.", "added bullet train")],
    "magicbrush:10909:1": [("remove the microwave and stove", "The microwave is removed.", "microwave absent"), ("remove the microwave and stove", "The stove is removed.", "stove absent"), ("put a refrigerator in it's place", "A refrigerator is placed at the location referred to by 'it's place' in the request.", "refrigerator at the requested replacement location")],
    "magicbrush:442416:1": [("add a phone", "A phone is added to the scene.", "added phone")],
    "magicbrush:504170:1": [("The background should be of a mountain.", "The background depicts a mountain.", "mountain background")],
    "magicbrush:173530:1": [("another car", "The scene contains the requested additional car.", "another car"), ("on it", "The additional car is on the referent of 'it'; the referent remains unresolved by the request alone.", "on the unspecified referent of 'it'")],
    "magicbrush:23781:1": [("a bouquet of red flowers", "A bouquet of flowers is added to the scene.", "added bouquet of flowers"), ("red flowers", "The flowers in the added bouquet are red.", "red flowers")],
}
AMBIGUITIES = {
    "magicbrush:513551:1": "The scope of 'only' is limited to items on the cup; no changes elsewhere are inferred.",
    "magicbrush:10909:1": "The antecedent and exact spatial extent of 'it's place' need source-grounded review; no target location is invented.",
    "magicbrush:173530:1": "The referent of 'it', identity of 'another car', and intended configuration are unresolved. This candidate must not be accepted without clarification.",
}
root = Path("data/review/viewer_proposals_gpu")
rows = {row["sample_id"]: row for row in read_jsonl(root / "proposals.jsonl")}
drafts = []
for sample, atoms in DRAFTS.items():
    row = rows[sample]
    if "evaluation" in row:
        raise ValueError("Existing independent evaluations must not be overwritten")
    for span, _, _ in atoms:
        if row["request"].count(span) != 1:
            raise ValueError("Draft spans must match the original request exactly and uniquely")
    evaluation = PrivateEvaluation.model_validate({
        "private_task_key": sample,
        "eval_atoms": [{"eval_id": f"eval:{index:04d}", "raw_span": span,
                        "predicate": predicate, "expected_value": expected,
                        "checker_route": "semantic", "threshold": 0.8, "mandatory": True}
                       for index, (span, predicate, expected) in enumerate(atoms)],
        "source_baseline_observations": [],
        "annotation_provenance": {"origin": "assistant_text_or_public_source_draft",
                                  "input_scope": "original_public_request_only",
                                  "private_target_seen": False, "private_mask_seen": False,
                                  "controller_plan_used": False, "human_reviewed": False,
                                  "preserved_eval_ids": [], "ambiguous": sample in AMBIGUITIES,
                                  "ambiguity_note": AMBIGUITIES.get(sample),
                                  "created_at": datetime.now(timezone.utc).isoformat()},
        "evaluator_version": "assistant-request-only-unreviewed-v1",
    })
    row.update(evaluation=evaluation.model_dump(mode="json"),
               evaluation_teacher="assistant_text_or_public_source_draft",
               automated_coverage=False, evaluation_ai_reviewed=False)
    drafts.append({"sample_id": sample, "request": row["request"],
                   "evaluation": evaluation.model_dump(mode="json"),
                   "reviewer": "assistant_text_or_public_source_draft",
                   "review_type": "ai_draft", "human_reviewed": False})
    rows[sample] = row
destination = Path("data/review/assistant_request_drafts.jsonl")
if destination.exists():
    raise FileExistsError("Independent drafts are versioned and must not be overwritten")
write_jsonl(destination, drafts)
write_jsonl(root / "proposals.with_assistant_drafts.jsonl", rows.values())
write_report(Path("data/review/assistant_request_drafts.report.json"), {
    "drafted": len(drafts), "total_independent_evaluations": len(rows),
    "source": "original_public_requests_only", "human_reviewed": False,
    "ai_visual_review_complete": False, "calibrated": False,
    "ambiguous_samples": list(AMBIGUITIES),
})
print(json.dumps({"drafted": len(drafts), "total_evaluations": len(rows), "human_reviewed": False}))
