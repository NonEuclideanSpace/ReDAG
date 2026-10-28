import argparse
from pathlib import Path

from agentic_image.core.types import PlanSpec, PrivateEvaluation
from build_complete_viewer_review import read_jsonl, write_jsonl, write_report


DIMENSIONS = {"task_completion", "mask", "identity", "registration",
              "original_request_coverage", "plan_compatible", "evaluation_approved"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    packet, output = Path(arguments.packet), Path(arguments.output)
    if output.exists():
        raise FileExistsError("Reviewed annotations must not be overwritten")
    keys = {row["sample_id"]: row for row in read_jsonl(packet / "keys.jsonl")}
    annotations = {row["sample_id"]: row for row in read_jsonl(packet / "annotations.pending.jsonl")}
    reviews = list(read_jsonl(arguments.reviews))
    if len({row["sample_id"] for row in reviews}) != len(reviews):
        raise ValueError("Duplicate human review rows are not allowed")
    accepted, plans, retained = [], [], []
    for review in reviews:
        key = keys[review["sample_id"]]
        if review.get("review_revision") != key["review_revision"]:
            raise ValueError("Review does not match the exact displayed proposal and images")
        if review.get("scope") != "independent_source_only_plan_and_annotation_review_v1":
            raise ValueError("Human review scope is missing or incompatible")
        retained.append(review)
        if not review.get("reviewed"):
            continue
        if not review.get("reviewer", "").strip() or not review.get("reviewed_at"):
            raise ValueError("Completed reviews require a named reviewer and timestamp")
        ratings = review.get("ratings", {})
        if set(ratings) != DIMENSIONS or any(type(value) is not bool for value in ratings.values()):
            raise ValueError("Completed reviews require all seven explicit boolean ratings")
        if not all(ratings.values()):
            continue
        if not key["review_ready"] or key["isolation"]["quarantined"]:
            raise ValueError("Incomplete or quarantined candidates cannot be admitted")
        PrivateEvaluation.model_validate(key["evaluation"])
        PlanSpec.model_validate(key["plan"])
        annotation = annotations[key["sample_id"]]
        annotation.update(verified=True, semantic_compatible=True, registration_verified=True,
                          human_reviewed=True, task_definition_verified=True)
        annotation["provenance"] = {"annotator": key["proposal_teacher"],
                                    "reviewer": review["reviewer"],
                                    "annotation_date": review["reviewed_at"],
                                    "source": "explicit_human_review_of_source_only_proposal",
                                    "review_revision": key["review_revision"],
                                    "human_review_path": str(Path(arguments.reviews).resolve()),
                                    "human_reviewed": True}
        accepted.append(annotation)
        plans.append({**key["compiled_plan"], "human_reviewed": True,
                      "review_revision": key["review_revision"],
                      "actor_teacher_trajectory": False})
    output.mkdir(parents=True)
    write_jsonl(output / "annotations.jsonl", accepted)
    write_jsonl(output / "approved_plan_recipes.jsonl", plans)
    write_jsonl(output / "human_reviews.jsonl", retained)
    write_report(output / "report.json", {"input_reviews": len(reviews),
                  "accepted_independent_annotations": len(accepted),
                  "actor_teacher_trajectories": 0, "calibration_complete": False,
                  "final_actor_generated_pair_audit_complete": False,
                  "status": "independent_annotations_imported_final_pair_audit_still_required"})


if __name__ == "__main__":
    main()
