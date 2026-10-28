from pathlib import Path
import json
import os
import sys

import numpy as np

from agentic_image.core.geometry import normalize_image

from .io import read_jsonl, write_report


SOURCES = {
    "magicbrush": {
        "dataset": "osunlp/MagicBrush",
        "revision": "6c254560879df7851233ea0fd2ae2a62764d0ef2",
        "license": "CC-BY-4.0",
    },
    "omniedit": {
        "dataset": "TIGER-Lab/OmniEdit-Filtered-1.2M",
        "revision": "b6eec40a8a6994aa380497e33b1964742d7e028b",
        "license": "MIT",
    },
}


def decode_edit_mask(mask):
    if "A" in mask.getbands() or "transparency" in mask.info:
        alpha = np.asarray(mask.convert("RGBA"), dtype=np.uint8)[..., 3]
        if np.any(alpha < 255):
            return 1 - alpha.astype(np.float32) / 255, "transparent_edit_region"
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    if np.any(rgb[..., 0] != rgb[..., 1]) or np.any(rgb[..., 0] != rgb[..., 2]):
        raise ValueError("An opaque color image is not an official edit mask")
    return rgb[..., 0].astype(np.float32) / 255, "white_edit_region"


def store_edit_mask(mask, artifacts, provenance):
    array, encoding = decode_edit_mask(mask)
    if not np.any(array > 0):
        return None
    return artifacts.put_array(
        array,
        metadata={
            **provenance,
            "role": "private_edit_annotation",
            "annotation_source": "official_mask",
            "encoding": encoding,
            "original_size": list(mask.size),
        },
    ).artifact_id


def stream_import(
    source, output_path, artifacts, max_records, split="train", max_scanned=None, resume=False
):
    from datasets import load_dataset

    if source not in SOURCES:
        raise ValueError(f"Supported sources are {sorted(SOURCES)}")
    if max_records < 1 or split not in {"train", "dev", "test"}:
        raise ValueError("A positive record cap and an official split are required")
    if source == "omniedit" and split == "test":
        raise ValueError("OmniEdit does not provide an official test split")
    specification = SOURCES[source]
    destination = Path(output_path)
    partial = destination.with_suffix(".partial.jsonl")
    if destination.exists():
        raise FileExistsError(f"A completed import already exists: {destination}")
    if partial.exists() and not resume:
        raise FileExistsError(f"Use --resume to continue the saved import: {partial}")
    previous = list(read_jsonl(partial)) if partial.exists() else []
    seen = {row["sample_id"] for row in previous}
    if len(seen) != len(previous) or len(previous) > max_records:
        raise ValueError("The saved import contains duplicate IDs or exceeds the record cap")
    if any(
        row["source_dataset"] != source
        or row["official_split"] != split
        or row["license_record"]["revision"] != specification["revision"]
        for row in previous
    ):
        raise ValueError("The saved import does not match this dataset, split, and revision")
    print(f"Importing {source}/{split}, limit {max_records}", file=sys.stderr, flush=True)
    if previous:
        print(f"Resuming {len(previous)} saved records", file=sys.stderr, flush=True)
    dataset = load_dataset(
        specification["dataset"], revision=specification["revision"], split=split, streaming=True
    )
    report = {
        "source": source,
        "split": split,
        "scanned": 0,
        "accepted": len(previous),
        "resumed": len(previous),
        "score_filtered": 0,
        "missing_instruction": 0,
        "limit": max_records,
        "revision": specification["revision"],
    }
    max_scanned = max_records * 20 if max_scanned is None else max_scanned

    def records():
        for row in dataset:
            if report["accepted"] >= max_records or report["scanned"] >= max_scanned:
                break
            report["scanned"] += 1
            if source == "magicbrush":
                identifier = f"{row['img_id']}:{row['turn_index']}"
                session = str(row["img_id"])
                instruction = row["instruction"]
                source_image, target_image = row["source_img"], row["target_img"]
                mask = row.get("mask_img")
                metadata = {"turn_index": int(row["turn_index"]), "base_image_id": session}
            else:
                scores = {key: float(row[key]) for key in ("sc_score_1", "sc_score_2", "pq_score")}
                if any(score < 8 for score in scores.values()):
                    report["score_filtered"] += 1
                    continue
                identifier = str(row["omni_edit_id"])
                session = identifier
                instructions = row["edited_prompt_list"]
                instruction = max(
                    (text for text in instructions if isinstance(text, str) and text.strip()),
                    key=len,
                    default="",
                )
                source_image, target_image = row["src_img"], row["edited_img"]
                mask = None
                metadata = {
                    "task_category": row["task"],
                    "scores": scores,
                    "instruction_variants": instructions,
                }
            if f"{source}:{identifier}" in seen:
                continue
            if not instruction.strip():
                report["missing_instruction"] += 1
                continue
            source_image = normalize_image(source_image)
            target_image = normalize_image(target_image)
            provenance = {
                "dataset": specification["dataset"],
                "revision": specification["revision"],
                "record": identifier,
                "official_split": split,
            }
            source_ref = artifacts.put_image(
                source_image, metadata={**provenance, "role": "source"}
            )
            target_ref = artifacts.put_image(
                target_image, metadata={**provenance, "role": "private_target"}
            )
            mask_id = None
            if mask is not None:
                if mask.size not in {source_image.size, target_image.size}:
                    raise ValueError(f"Official mask has an unrecognized frame: {identifier}")
                mask_id = store_edit_mask(mask, artifacts, provenance)
            report["accepted"] += 1
            if report["accepted"] % 100 == 0:
                print(
                    f"Imported {report['accepted']}/{max_records} from {source}/{split}",
                    file=sys.stderr,
                    flush=True,
                )
            yield {
                "sample_id": f"{source}:{identifier}",
                "source_dataset": source,
                "official_split": split,
                "session_id": f"{source}:{session}",
                "instruction": instruction,
                "source_image_id": source_ref.artifact_id,
                "target_image_id": target_ref.artifact_id,
                "edit_mask_id": mask_id,
                "appearance_image_ids": [],
                "license_record": {
                    "license": specification["license"],
                    "source": specification["dataset"],
                    "revision": specification["revision"],
                    "original_image_provenance": metadata.get("base_image_id", identifier),
                },
                "registration": {
                    "same_dimensions": source_image.size == target_image.size,
                    "verified": source == "magicbrush" and source_image.size == target_image.size,
                    "provenance": "official_registered_edit_pair"
                    if source == "magicbrush"
                    else "requires_independent_registration_review",
                },
                "metadata": metadata,
            }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", buffering=1) as stream:
        for row in records():
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            if report["accepted"] % 100 == 0:
                stream.flush()
                os.fsync(stream.fileno())
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(destination)
    write_report(Path(output_path).with_suffix(".report.json"), report)
    return report
