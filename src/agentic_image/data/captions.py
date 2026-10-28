from collections import Counter
from pathlib import Path

from .io import read_jsonl, write_jsonl, write_report


def caption_candidates(input_path, annotations_path, output_path, max_records=1600):
    if max_records < 1:
        raise ValueError("Caption candidate limit must be positive")
    annotations = {row["sample_id"]: row for row in read_jsonl(annotations_path)}
    candidates, caption_annotations = [], []
    reasons = Counter()
    seen_images = set()
    for row in read_jsonl(input_path):
        if len(candidates) >= max_records:
            break
        if row.get("split_id") != "train":
            continue
        annotation = annotations.get(row["sample_id"], {})
        caption = annotation.get("complete_caption", "").strip()
        if (
            not caption
            or not annotation.get("complete_caption_verified")
            or not annotation.get("provenance")
        ):
            reasons["requires_verified_complete_caption"] += 1
            continue
        if annotation.get("benchmark_derived"):
            reasons["benchmark_derived"] += 1
            continue
        role = annotation.get("caption_image_role", "target")
        if role not in {"target", "source"}:
            raise ValueError("Caption image role must be target or source")
        image_id = row.get(f"{role}_image_id")
        if not image_id:
            reasons["missing_caption_image"] += 1
            continue
        if image_id in seen_images:
            reasons["duplicate_caption_image"] += 1
            continue
        seen_images.add(image_id)
        sample_id = f"t2i:{row['sample_id']}:{role}"
        candidate = {
            "sample_id": sample_id,
            "source_dataset": "t2i",
            "official_split": "train",
            "split_id": "train",
            "family_id": row["family_id"],
            "session_id": row.get("session_id", row["family_id"]),
            "family_links": [row["family_id"]],
            "instruction": caption,
            "source_image_id": None,
            "target_image_id": image_id,
            "edit_mask_id": None,
            "appearance_image_ids": [],
            "license_record": row.get("license_record", {}),
            "registration": {"verified": True, "provenance": "single_image_generation_target"},
            "metadata": {
                "parent_sample_id": row["sample_id"],
                "complete_caption_verified": True,
                "caption_provenance": annotation["provenance"],
                "public_reference_source_image_id": row.get("source_image_id")
                if role == "target" and row.get("source_image_id") != image_id
                else None,
            },
        }
        candidates.append(candidate)
        caption_annotation = {
            "sample_id": sample_id,
            "verified": False,
            "semantic_compatible": False,
            "complete_caption_verified": True,
            "registration_verified": True,
            "geometry_verified": bool(
                annotation.get("caption_geometry_verified", annotation.get("geometry_verified"))
            ),
            "provenance": annotation["provenance"],
            "entities": annotation.get("caption_entities", []),
            "box_frame": annotation.get("box_frame", "original"),
            "evaluation": annotation.get("caption_evaluation", {}),
            "public_controls": annotation.get("caption_public_controls", {}),
            "appearance_reference": annotation.get("caption_appearance_reference", {}),
        }
        if annotation.get("caption_evaluation_verified") and caption_annotation["evaluation"].get(
            "eval_atoms"
        ):
            caption_annotation["verified"] = True
            caption_annotation["semantic_compatible"] = True
        caption_annotations.append(caption_annotation)
    write_jsonl(output_path, candidates)
    annotation_path = Path(output_path).with_suffix(".annotations.jsonl")
    write_jsonl(annotation_path, caption_annotations)
    report = {
        "candidates": len(candidates),
        "independently_verified_evaluations": sum(row["verified"] for row in caption_annotations),
        "unique_families": len({row["family_id"] for row in candidates}),
        "requested_cap": max_records,
        "shortfall": max(0, max_records - len(candidates)),
        "rejections": dict(reasons),
        "annotations_path": str(annotation_path.resolve()),
    }
    write_report(Path(output_path).with_suffix(".report.json"), report)
    return report
