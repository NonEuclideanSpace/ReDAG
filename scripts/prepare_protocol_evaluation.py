from collections import Counter
import io
import json
from pathlib import Path

import httpx
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.types import (
    CheckSpec,
    PublicCheckAtom,
    PublicTask,
    RequestSpan,
    UserRequest,
)
from agentic_image.data.families import HammingIndex
from agentic_image.data.io import read_jsonl, write_jsonl, write_report


def main():
    root = Path("data")
    store = ArtifactStore(root / "artifacts")
    rows = sorted(
        read_jsonl(root / "families.protocol.jsonl"),
        key=lambda row: (row["family_id"], row["sample_id"]),
    )
    images = {
        row["original_artifact_id"]: row
        for row in read_jsonl(root / "viewer/images.jsonl")
        if row["status"] == "downloaded"
    }
    training = list(read_jsonl(root / "public/protocol_pilot_tasks.jsonl"))
    training_families = {row["family_id"] for row in training}
    seen = set()
    fingerprints = HammingIndex()
    counts = Counter()
    tasks = []
    references = []
    unavailable = 0
    restored = 0
    viewer = {}
    for path in (root / "viewer/api").glob("magicbrush_dev_*.json"):
        for item in json.loads(path.read_text())["rows"]:
            value = item["row"]
            viewer[f"magicbrush:{value['img_id']}:{value['turn_index']}"] = value
    for row in rows:
        source = row["source_dataset"]
        family = row["family_id"]
        if (
            row["split_id"] != "official-dev"
            or counts[source] >= 8
            or family in seen | training_families
        ):
            continue
        entry = images[row["source_image_id"]]
        try:
            path = store.path(entry["artifact_id"])
        except (FileNotFoundError, KeyError):
            path = store.root / entry["relative_path"]
        if not path.is_file():
            if source != "magicbrush" or row["sample_id"] not in viewer or restored >= 8:
                unavailable += 1
                continue
            value = viewer[row["sample_id"]]
            if value["instruction"] != row["instruction"]:
                raise ValueError("Official evaluation source no longer matches its request")
            response = httpx.get(value["source_img"]["src"], follow_redirects=True, timeout=20)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            reference = store.put_image(
                image,
                metadata={
                    "dataset_profile": "official-viewer-reencoded-v1",
                    "original_artifact_id": row["source_image_id"],
                    "sample_id": entry["sample_id"],
                    "role": "source",
                    "viewer_revision": entry["viewer_revision"],
                    "original_revision": row["license_record"]["revision"],
                },
            )
            if (
                reference.value_id != entry["value_id"]
                or reference.artifact_id != entry["artifact_id"]
            ):
                raise ValueError("Restored evaluation source differs from its registered identity")
            restored += 1
            path = store.path(reference.artifact_id)
        fingerprint = int(entry["phash"])
        if fingerprints.query(fingerprint, radius=6):
            continue
        fingerprints.add(fingerprint, family)
        seen.add(family)
        counts[source] += 1
        text = row["instruction"]
        task = PublicTask(
            task_id="protocol-eval:" + row["sample_id"],
            family_id=family,
            split_id="official-dev",
            source_dataset=source,
            license_record={
                **row["license_record"],
                "dataset_profile": "official-viewer-reencoded-v1",
                "purpose": "protocol_evaluation_only",
            },
            user_request=UserRequest(
                text=text,
                source_image_id=entry["artifact_id"],
                spans=(RequestSpan(span_id="span:request", start=0, end=len(text), text=text),),
            ),
            task_mode="edit",
            check_spec=CheckSpec(
                extractor_version="public-protocol-only",
                fallback=True,
                atoms=(
                    PublicCheckAtom(
                        check_id="check:request",
                        raw_span_ids=("span:request",),
                        predicate=text,
                        expected_value=text,
                        confidence=0.0,
                    ),
                ),
            ),
        )
        tasks.append(task.model_dump(mode="json"))
        reference = store.get_ref(entry["artifact_id"])
        references.append(
            {
                "task_id": task.task_id,
                "original_sample_id": row["sample_id"],
                "family_id": family,
                "artifact_id": reference.artifact_id,
                "value_id": reference.value_id,
                "ref": str(store.root / "refs" / (reference.artifact_id.split(":")[1] + ".json")),
                "value": str(path),
            }
        )
        if len(tasks) == 16:
            break
    if len(tasks) != 16 or counts != {"magicbrush": 8, "omniedit": 8}:
        raise ValueError(
            f"Insufficient available independent official development families: {dict(counts)}"
        )
    destination = root / "public/protocol_eval_16_tasks.jsonl"
    write_jsonl(destination, tasks)
    write_jsonl(root / "public/protocol_eval_16_artifacts.jsonl", references)
    report = {
        "status": "ready",
        "tasks": 16,
        "independent_exact_families": 16,
        "by_source": dict(counts),
        "split": "official-dev",
        "purpose": "protocol_schema_and_recovery_evaluation_only",
        "visual_quality_evaluation": False,
        "training_family_overlap": 0,
        "calibration_used": False,
        "existing_reservations_changed": False,
        "selection": "Eight available source images per dataset, distinct exact families, pairwise unresolved pHash matches excluded; no success filtering",
        "train_to_heldout_evidence": "data/public/protocol_pilot_tasks.report.json",
        "locally_unavailable_images_skipped": unavailable,
        "source_images_restored_from_official_viewer": restored,
        "tasks_path": str(destination),
    }
    write_report(root / "public/protocol_eval_16_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
