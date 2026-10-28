from collections import defaultdict
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from agentic_image.learning.sampling import stream_seed

from .io import read_jsonl, write_jsonl, write_report


DIMENSIONS = ("task_completion", "mask", "identity", "registration", "original_request_coverage")


def prepare_pair_audit(
    pairs_dir, annotations_path, output_dir, artifacts, per_bucket=100, seed=20261027
):
    if per_bucket != 100:
        raise ValueError(
            "The registered paired-data audit samples 100 records per available bucket"
        )
    pairs_dir, output = Path(pairs_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError("An existing audit packet must not be overwritten")
    output.mkdir(parents=True)
    previews = output / "previews"
    previews.mkdir()
    annotations = {item["sample_id"]: item for item in read_jsonl(annotations_path)}
    tasks = {item["task_id"]: item for item in read_jsonl(pairs_dir / "public_tasks.jsonl")}
    evaluations = {
        item["task_id"]: item["evaluation"]
        for item in read_jsonl(pairs_dir / "private_tasks.jsonl")
    }
    buckets = defaultdict(list)
    for row in read_jsonl(pairs_dir / "renderer_records.jsonl"):
        buckets[f"{row['source_dataset']}:{row['region_count']}"].append(row)
    keys, review_rows, sections = [], [], []
    for bucket, records in sorted(buckets.items()):
        selected = sorted(
            records, key=lambda item: stream_seed(seed, item["sample_id"], "pair_audit")
        )[:per_bucket]
        for row in selected:
            index = len(keys)
            task = tasks[row["task_id"]]
            annotation = annotations.get(row["sample_id"], {})
            source_id = task["user_request"].get("source_image_id")
            images = []
            for role, image_id in (("source", source_id), ("target", row["target_image_id"])):
                if image_id is None:
                    continue
                image = artifacts.get_image(image_id)
                image.thumbnail((512, 512))
                name = f"{index:05d}_{role}.png"
                image.save(previews / name)
                images.append(
                    f'<figure><img src="previews/{name}" alt="{role}"><figcaption>{role}</figcaption></figure>'
                )
            if row.get("edit_mask_id"):
                mask = artifacts.get_array(row["edit_mask_id"])
                name = f"{index:05d}_edit_mask.png"
                Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255)).save(previews / name)
                images.append(
                    f'<figure><img src="previews/{name}" alt="Private edit mask"><figcaption>Private edit mask</figcaption></figure>'
                )
            applicable = {
                "task_completion": True,
                "mask": bool(row.get("edit_mask_id") or row["region_count"]),
                "identity": bool(source_id or task["user_request"].get("appearance_image_ids")),
                "registration": bool(source_id),
                "original_request_coverage": True,
            }
            key = {
                "sample_id": row["sample_id"],
                "bucket": bucket,
                "applicable": applicable,
                "family_id": row["family_id"],
                "condition_recipe": row.get("condition_recipe"),
                "annotation": annotation,
                "evaluation": evaluations[row["task_id"]],
            }
            keys.append(key)
            review_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "reviewer": "",
                    "reviewed": False,
                    "ratings": {
                        name: None if active else "not_applicable"
                        for name, active in applicable.items()
                    },
                    "notes": "",
                }
            )
            details = json.dumps(
                {
                    "plan": row.get("condition_recipe", {}).get("plan"),
                    "independent_annotation": annotation,
                    "evaluation": evaluations[row["task_id"]],
                },
                indent=2,
            )
            sections.append(
                f"<section><h2>{escape(row['sample_id'])}</h2><p>Bucket: {escape(bucket)}</p><p>{escape(task['user_request']['text'])}</p><div>{''.join(images)}</div><details><summary>Plan and private annotations</summary><pre>{escape(details)}</pre></details></section>"
            )
    write_jsonl(output / "keys.jsonl", keys)
    write_jsonl(output / "reviews.jsonl", review_rows)
    style = "body{font:16px system-ui;margin:32px;background:#f4f5f6;color:#15202b}section{background:white;padding:24px;margin:24px 0;border-radius:12px}section>div{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0}img{max-width:400px;max-height:400px}pre{white-space:pre-wrap}p{max-width:1000px;line-height:1.5}"
    (output / "index.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired supervision audit</title><style>{style}</style><body><h1>Paired supervision audit</h1><p>Review every sampled record against the original request, source and private target. Fill only human ratings in reviews.jsonl: true means correct; false means incorrect. Keep not_applicable entries unchanged. Record completion, masks, identity, registration and complete request coverage independently. A bucket with any error rate above 5% must be repaired and resampled.</p>{"".join(sections)}</body></html>'
    )
    report = {
        "selected": len(keys),
        "by_bucket": {bucket: min(len(records), per_bucket) for bucket, records in buckets.items()},
        "population_by_bucket": {bucket: len(records) for bucket, records in buckets.items()},
        "pair_source_path": str(pairs_dir.resolve()),
        "annotations_source_path": str(Path(annotations_path).resolve()),
        "status": "awaiting_human_review",
        "reviews_path": str((output / "reviews.jsonl").resolve()),
    }
    write_report(output / "report.json", report)
    return report


def evaluate_pair_audit(packet_dir, reviews_path):
    packet = Path(packet_dir)
    keys = {item["sample_id"]: item for item in read_jsonl(packet / "keys.jsonl")}
    reviews = list(read_jsonl(reviews_path))
    if (
        not keys
        or len(reviews) != len(keys)
        or {item["sample_id"] for item in reviews} != set(keys)
    ):
        raise ValueError("Every selected audit sample requires exactly one human review")
    buckets = defaultdict(lambda: defaultdict(list))
    for row in reviews:
        if row.get("reviewed") is not True or not row.get("reviewer", "").strip():
            raise ValueError("Human audit ratings require a completed named reviewer")
        key = keys[row["sample_id"]]
        if set(row.get("ratings", {})) != set(DIMENSIONS):
            raise ValueError("Audit review must retain all five registered quality dimensions")
        for dimension in DIMENSIONS:
            value = row["ratings"][dimension]
            if key["applicable"][dimension]:
                if type(value) is not bool:
                    raise ValueError("Applicable audit ratings must be human true or false values")
                buckets[key["bucket"]][dimension].append(value)
            elif value != "not_applicable":
                raise ValueError("An inapplicable quality dimension must remain not_applicable")
    rates = {
        bucket: {
            dimension: {
                "count": len(values),
                "correct": sum(values),
                "error_rate": 1 - sum(values) / len(values),
            }
            for dimension, values in dimensions.items()
        }
        for bucket, dimensions in buckets.items()
    }
    paused = [
        bucket
        for bucket, dimensions in rates.items()
        if any(value["error_rate"] > 0.05 + 1e-12 for value in dimensions.values())
    ]
    packet_report = json.loads((packet / "report.json").read_text())
    report = {
        "status": "accepted" if not paused else "repair_and_resample",
        "buckets": rates,
        "paused_buckets": paused,
        "reviewed": len(reviews),
        "selected_by_bucket": packet_report["by_bucket"],
        "population_by_bucket": packet_report["population_by_bucket"],
        "pair_source_path": packet_report["pair_source_path"],
        "annotations_source_path": packet_report["annotations_source_path"],
        "human_review_path": str(Path(reviews_path).resolve()),
    }
    write_report(packet / "result.json", report)
    return report


def require_pair_audit(path):
    if not path:
        raise ValueError("Renderer and full actor SFT require the paired-data human audit result")
    report = json.loads(Path(path).read_text())
    if (
        report.get("status") != "accepted"
        or report.get("paused_buckets")
        or not report.get("pair_source_path")
    ):
        raise ValueError("Paired supervision has not passed its per-bucket human quality audit")
    selected = report.get("selected_by_bucket", {})
    population = report.get("population_by_bucket", {})
    if (
        not selected
        or set(selected) != set(population)
        or set(selected) != set(report.get("buckets", {}))
    ):
        raise ValueError("Paired-data audit is missing a source or condition bucket")
    if report.get("reviewed") != sum(selected.values()) or any(
        selected[key] != min(100, population[key]) for key in selected
    ):
        raise ValueError("Paired-data audit does not cover the registered sample count")
    for bucket, dimensions in report["buckets"].items():
        if not {"task_completion", "original_request_coverage"} <= set(dimensions):
            raise ValueError("Paired-data audit omitted mandatory quality dimensions")
        if any(
            not 0 < values["count"] <= selected[bucket] or values["error_rate"] > 0.05 + 1e-12
            for values in dimensions.values()
        ):
            raise ValueError("Paired-data labels exceed the five-percent error limit")
    return report
