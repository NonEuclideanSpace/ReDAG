from collections import Counter, defaultdict
import math
from pathlib import Path

from agentic_image.core.types import BoundingBox
from agentic_image.learning.sampling import stream_seed

from .io import read_jsonl, write_jsonl, write_report


def appearance_candidates(
    input_path, annotations_path, output_path, artifacts, fraction=0.1, seed=20261027
):
    if not 0 < fraction <= 0.25:
        raise ValueError(
            "Public appearance sample fraction must be positive and at most one quarter"
        )
    rows = list(read_jsonl(input_path))
    annotations = {row["sample_id"]: row for row in read_jsonl(annotations_path)}
    pools = defaultdict(list)
    eligible = defaultdict(list)
    rejected = Counter()
    for index, row in enumerate(rows):
        if row.get("split_id") != "train":
            continue
        bucket = row.get("source_dataset", "curated")
        pools[bucket].append(index)
        reference = annotations.get(row["sample_id"], {}).get("appearance_reference", {})
        public_source_id = row.get("source_image_id") or row.get("metadata", {}).get(
            "public_reference_source_image_id"
        )
        if (
            not public_source_id
            or not reference.get("verified")
            or not reference.get("provenance")
            or not reference.get("source_box")
            or not reference.get("entity_name")
        ):
            rejected["missing_verified_public_source_crop"] += 1
            continue
        if (
            artifacts.get_ref(public_source_id).value_id
            == artifacts.get_ref(row["target_image_id"]).value_id
        ):
            rejected["source_equals_hidden_target"] += 1
            continue
        if row.get("appearance_image_ids"):
            rejected["already_has_appearance_reference"] += 1
            continue
        if reference.get("origin", "source") != "source":
            raise ValueError("Default appearance construction only accepts the public source image")
        eligible[bucket].append(index)
    counts, desired = {}, {}
    for bucket, indices in pools.items():
        desired[bucket] = round(len(indices) * fraction)
        ordered = sorted(
            eligible[bucket],
            key=lambda index: stream_seed(seed, rows[index]["sample_id"], "appearance_selection"),
        )
        counts[bucket] = 0
        for index in ordered[: desired[bucket]]:
            row = rows[index]
            reference = annotations[row["sample_id"]]["appearance_reference"]
            box = BoundingBox(
                x0=reference["source_box"][0],
                y0=reference["source_box"][1],
                x1=reference["source_box"][2],
                y1=reference["source_box"][3],
            )
            source_id = (
                row.get("source_image_id") or row["metadata"]["public_reference_source_image_id"]
            )
            image = artifacts.get_image(source_id)
            crop = image.crop(
                (
                    math.floor(box.x0 * image.width),
                    math.floor(box.y0 * image.height),
                    math.ceil(box.x1 * image.width),
                    math.ceil(box.y1 * image.height),
                )
            )
            crop_ref = artifacts.put_image(
                crop,
                producer_id=row["sample_id"],
                input_slots=(("public_source", 0, artifacts.get_ref(source_id).value_id),),
                metadata={
                    "role": "public_appearance",
                    "entity_name": reference["entity_name"],
                    "original_source_id": source_id,
                    "source_box": box.model_dump(),
                    "annotation_provenance": reference["provenance"],
                },
            )
            row["appearance_image_ids"] = [crop_ref.artifact_id]
            metadata = dict(row.get("metadata", {}))
            metadata["public_appearance"] = {
                "image_id": crop_ref.artifact_id,
                "entity_name": reference["entity_name"],
                "role": "appearance",
                "source": "verified_public_source_crop",
            }
            row["metadata"] = metadata
            counts[bucket] += 1
    write_jsonl(output_path, rows)
    report = {
        "rows": len(rows),
        "fraction": fraction,
        "selected_by_source": counts,
        "desired_by_source": desired,
        "shortfalls": {key: max(0, desired[key] - counts[key]) for key in desired},
        "rejections": dict(rejected),
        "reference_source": "public_original_source_only",
    }
    write_report(Path(output_path).with_suffix(".report.json"), report)
    return report
