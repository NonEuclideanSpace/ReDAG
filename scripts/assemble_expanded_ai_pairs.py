import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from agentic_image.config import load_config
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import box_mask, transform_box
from agentic_image.core.types import BoundingBox, PlanSpec
from agentic_image.data.paired import _canvas_artifacts
def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def report(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


DIMENSIONS = ("task_completion", "mask", "identity", "registration", "original_request_coverage", "plan_compatible")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/renderer-expanded-200.yaml")
    parser.add_argument("--root", default="data/training_expansion")
    parser.add_argument("--output", default="data/expanded-ai-pairs-v1")
    parser.add_argument("--minimum", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    config = load_config(args.config)
    if config.get("review_mode") != "ai_exploratory":
        raise ValueError("The explicit AI review profile is required")
    coverage = json.loads((root / "isolation/coverage.after.json").read_text())
    isolation = {row["sample_id"]: row for row in read(root / "isolation/candidate_isolation.jsonl")}
    store = ArtifactStore(config["data"]["artifacts"])
    components = SimpleNamespace(config=config, artifacts=store)
    qualified, rejected, reviews, reviewed_samples = [], [], [], set()
    for directory in sorted(root.glob("candidates*")):
        if not (directory / "review_keys.jsonl").is_file():
            continue
        keys = {row["sample_id"]: row for row in read(directory / "review_keys.jsonl")}
        prepared = {row["sample_id"]: row for row in read(directory / "prepared.jsonl")}
        for path in sorted(directory.glob("ai_reviews*.jsonl")):
            for review in read(path):
                sample = review["sample_id"]
                if sample in reviewed_samples:
                    raise ValueError("Duplicate AI reviews require explicit adjudication")
                reviewed_samples.add(sample)
                key, row = keys[sample], prepared[sample]
                if review.get("review_type") != "ai" or review.get("human_reviewed") is not False:
                    raise ValueError("Every review must retain explicit AI provenance")
                if review.get("scope") != "original_request_global_pair_v1" or review.get("review_revision") != key["review_revision"]:
                    raise ValueError("Review does not match its frozen displayed images and request")
                if not review.get("reviewer") or not review.get("evidence") or not review.get("reviewed_image_paths"):
                    raise ValueError("Actual visual review evidence is required")
                plan = PlanSpec(global_text=key["request"])
                if PlanSpec.model_validate(review["global_plan"]) != plan:
                    raise ValueError("The global plan must preserve the original request exactly")
                reviews.append(review)
                reasons = []
                if review.get("accepted") is not True or any(review["ratings"].get(name) is not True for name in DIMENSIONS):
                    reasons.append("visual_review_rejected_or_unknown")
                screened = isolation.get(sample, {})
                if not screened.get("isolation_passed") or not screened.get("full_candidate_family_value_coverage") or not screened.get("full_guarded_value_coverage"):
                    reasons.append("complete_family_isolation_not_passed")
                exclusions = review.get("target_exclusion_boxes", review.get("private_loss_exclusions", []))
                for exclusion in exclusions:
                    coordinates = exclusion.get("bbox_xyxy_normalized", exclusion) if isinstance(exclusion, dict) else exclusion
                    coordinates = {key: coordinates[key] for key in ("x0", "y0", "x1", "y1")} if isinstance(coordinates, dict) else dict(zip(("x0", "y0", "x1", "y1"), coordinates))
                    box = BoundingBox.model_validate(coordinates)
                    if (box.x1 - box.x0) * (box.y1 - box.y0) > 0.02:
                        reasons.append("private_exclusion_area_exceeds_two_percent")
                source, target = row["images"]
                if source["artifact_id"] != key["source_image_id"] or target["artifact_id"] != key["target_image_id"]:
                    raise ValueError("Review image identities changed")
                if source["width"] * target["height"] != target["width"] * source["height"]:
                    reasons.append("source_target_aspect_ratio_mismatch")
                if reasons:
                    rejected.append({"sample_id": sample, "family_id": row["family_id"], "expansion_index": row["expansion_index"],
                                     "reasons": reasons, "source_hw": [source["height"], source["width"]],
                                     "target_hw": [target["height"], target["width"]]})
                else:
                    qualified.append((row, review))
    qualified.sort(key=lambda pair: pair[0]["expansion_index"])
    if args.selection:
        selection = read(args.selection)
        selected_ids = [row["sample_id"] for row in selection]
        available = {row["sample_id"]: (row, review) for row, review in qualified}
        if len(set(selected_ids)) != len(selected_ids) or any(sample not in available for sample in selected_ids):
            raise ValueError("Every frozen selected sample must independently pass every qualification gate")
        qualified = [available[sample] for sample in selected_ids]
        for expected, (actual, _) in zip(selection, qualified):
            if any(expected[key] != actual[key] for key in ("sample_id", "family_id", "expansion_index", "source_dataset")):
                raise ValueError("Frozen selection identity mismatch")
    if args.limit:
        qualified = qualified[:args.limit]
    if len({row["family_id"] for row, _ in qualified}) != len(qualified):
        raise ValueError("The qualified training pool contains duplicate families")
    status = {"visual_reviews": len(reviews), "qualified_independent_families": len(qualified),
              "minimum_required": args.minimum, "minimum_reached": len(qualified) >= args.minimum,
              "source_counts": dict(Counter(row["source_dataset"] for row, _ in qualified)),
              "rejection_reason_counts": dict(Counter(reason for row in rejected for reason in row["reasons"])),
              "review_type": "ai", "human_reviewed": False, "unverified_registration_stretch_applied": False,
              "full_guarded_phash_screening_complete": coverage.get("full_guarded_phash_screening_complete", False),
              "full_candidate_family_phash_screening_complete": coverage.get("full_candidate_family_phash_screening_complete", False)}
    report(root / "qualification_status.json", status)
    write(root / "qualification_rejections.jsonl", rejected)
    write(root / "qualified_selection.jsonl", [{"sample_id": row["sample_id"], "family_id": row["family_id"],
           "expansion_index": row["expansion_index"], "source_dataset": row["source_dataset"]} for row, _ in qualified])
    print(json.dumps(status), flush=True)
    if not args.build:
        return
    if len(qualified) < args.minimum:
        raise ValueError("The real qualified family count does not meet the requested minimum")
    if output.exists():
        raise FileExistsError("A frozen training package must not be overwritten")
    records = []
    for row, review in qualified:
        raw = {"sample_id": row["sample_id"], "source_image_id": row["images"][0]["artifact_id"],
               "target_image_id": row["images"][1]["artifact_id"], "edit_mask_id": None}
        source, target, valid, edit, frame = _canvas_artifacts(components, raw, {"registration_verified": True})
        exclusions = review.get("target_exclusion_boxes", review.get("private_loss_exclusions", []))
        boxes = []
        for exclusion in exclusions:
            coordinates = exclusion.get("bbox_xyxy_normalized", exclusion) if isinstance(exclusion, dict) else exclusion
            coordinates = {key: coordinates[key] for key in ("x0", "y0", "x1", "y1")} if isinstance(coordinates, dict) else dict(zip(("x0", "y0", "x1", "y1"), coordinates))
            box = BoundingBox.model_validate(coordinates)
            if (box.x1 - box.x0) * (box.y1 - box.y0) > 0.02:
                raise ValueError("A private artifact exclusion may not conceal a large target area")
            boxes.append(box)
        if boxes:
            mask = store.get_array(valid).copy()
            for box in boxes:
                mask *= 1 - box_mask(transform_box(box, frame), frame.canvas_hw)
            valid = store.put_array(mask, frame=frame, metadata={"role": "valid_supervision", "target_artifact_exclusions": [box.model_dump(mode="json") for box in boxes], "reviewer": review["reviewer"]}).artifact_id
        records.append({"sample_id": row["sample_id"], "family_id": row["family_id"], "task_id": row["sample_id"],
                        "condition_recipe": {"plan": PlanSpec(global_text=row["request"]).model_dump(mode="json"), "source_image_id": source},
                        "target_image_id": target, "valid_mask_id": valid, "edit_mask_id": edit,
                        "source_dataset": row["source_dataset"], "region_count": 0, "split_id": "train", "verified": True,
                        "verification_provenance": {"review_type": "ai", "human_reviewed": False,
                        "reviewer": review["reviewer"], "review_revision": review["review_revision"],
                        "target_teacher": "deterministic_original_request_plan", "actor_sampled_plan": False,
                        "full_family_isolation": True, "target_artifact_exclusions": [box.model_dump(mode="json") for box in boxes]}})
    output.mkdir(parents=True)
    write(output / "renderer.jsonl", records)
    write(output / "ai_reviews.jsonl", [review for _, review in qualified])
    write(output / "rejected.jsonl", rejected)
    counts = Counter(row["source_dataset"] + ":0" for row in records)
    selected = {bucket: min(100, count) for bucket, count in counts.items()}
    audit = {"status": "accepted", "review_mode": "ai_exploratory", "human_reviewed": False,
             "user_authorized_ai_review": True, "ai_review_path": str((output / "ai_reviews.jsonl").resolve()),
             "pair_source_path": str(output.resolve()), "paused_buckets": [], "reviewed": sum(selected.values()),
             "selected_by_bucket": selected, "population_by_bucket": dict(counts),
             "screened_candidates": len(reviews), "rejected_candidates": len(rejected),
             "ai_visual_reviewed_accepted_population": len(records), "scope": "original_request_global_pair_v1",
             "screening_is_independent_human_audit": False,
             "rate_interpretation": "Rates describe rechecked accepted-pool AI labels, not an independent population-error estimate or judge calibration",
             "buckets": {bucket: {name: {"count": count, "correct": count, "error_rate": 0.0} for name in DIMENSIONS} for bucket, count in selected.items()}}
    report(output / "audit.json", audit)
    report(output / "completed.json", {**status, "status": "built", "renderer_records": len(records)})


if __name__ == "__main__":
    main()
