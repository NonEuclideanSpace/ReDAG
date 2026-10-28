import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.types import (
    CheckSpec,
    PublicCheckAtom,
    PublicTask,
    RequestSpan,
    UserRequest,
)
from agentic_image.data.families import partition_families
from agentic_image.data.io import read_jsonl, write_jsonl, write_report


class PerceptualStore(ArtifactStore):
    def __init__(self, root, thumbnails):
        super().__init__(root)
        self.thumbnails = thumbnails

    def get_image(self, identifier):
        reference = self.get_ref(identifier)
        path = self.thumbnails / (reference.value_id.split(":")[1] + ".png")
        with Image.open(path) as image:
            return image.copy()


def image_paths(root, rows):
    identifiers = {
        identifier
        for row in rows
        for identifier in (
            row.get("source_image_id"),
            row["target_image_id"],
            *row.get("appearance_image_ids", []),
        )
        if identifier
    }
    store = ArtifactStore(root / "artifacts")
    references = [store.get_ref(identifier) for identifier in sorted(identifiers)]
    paths = sorted({reference.relative_path for reference in references})
    return references, paths


def selected_families(rows, limit):
    pools = defaultdict(list)
    seen = set()
    for row in sorted(rows, key=lambda item: (item["family_id"], item["sample_id"])):
        if row["split_id"] != "train" or row["family_id"] in seen:
            continue
        seen.add(row["family_id"])
        pools[row["source_dataset"]].append(row)
    selected = []
    while len(selected) < limit and any(pools.values()):
        for source in sorted(pools):
            if pools[source] and len(selected) < limit:
                selected.append(pools[source].pop(0))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "partition", "protocol"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    arguments = parser.parse_args()
    root = arguments.data.resolve()
    rows = list(read_jsonl(root / "raw/all_saved.jsonl"))
    if arguments.stage == "inventory":
        references, paths = image_paths(root, rows)
        destination = root / "transfer/referenced_image_values.txt"
        destination.write_text("".join(path + "\n" for path in paths))
        report = {
            "rows": len(rows),
            "image_references": len(references),
            "unique_image_values": len(paths),
            "decoded_rgb_bytes": sum(
                reference.shape[0] * reference.shape[1] * reference.shape[2]
                for reference in {r.value_id: r for r in references}.values()
            ),
            "value_list": str(destination),
        }
        write_report(root / "transfer/inventory.json", report)
    elif arguments.stage == "partition":
        report = partition_families(
            root / "raw/all_saved.jsonl",
            root / "families.jsonl",
            PerceptualStore(root / "artifacts", root / "transfer/phash_thumbnails"),
            minimums={"development": 64, "calibration": 600, "diagnostic-test": 32},
        )
    else:
        rows = list(read_jsonl(root / "families.jsonl"))
        selected = selected_families(rows, 32)
        if len(selected) < 16:
            raise ValueError("At least sixteen isolated training families are required")
        tasks = []
        store = ArtifactStore(root / "artifacts")
        for row in selected:
            store.get_image(row["source_image_id"])
            request = UserRequest(text=row["instruction"], source_image_id=row["source_image_id"])
            request = request.model_copy(
                update={
                    "spans": (
                        RequestSpan(
                            span_id="span:request",
                            start=0,
                            end=len(request.text),
                            text=request.text,
                        ),
                    )
                }
            )
            task = PublicTask(
                task_id=row["sample_id"],
                family_id=row["family_id"],
                split_id="train",
                source_dataset=row["source_dataset"],
                license_record=row["license_record"],
                user_request=request,
                task_mode="edit",
                check_spec=CheckSpec(
                    extractor_version="public-protocol-only",
                    fallback=True,
                    atoms=(
                        PublicCheckAtom(
                            check_id="check:request",
                            raw_span_ids=("span:request",),
                            predicate=request.text,
                            expected_value=request.text,
                            confidence=0.0,
                        ),
                    ),
                ),
            )
            tasks.append(
                PublicTask.model_validate(task.model_dump(mode="json")).model_dump(mode="json")
            )
        public = root / "public"
        public.mkdir(exist_ok=True)
        path = public / "protocol_pilot_tasks.jsonl"
        write_jsonl(path, tasks)
        report = {
            "purpose": "public_protocol_recovery_only",
            "tasks": len(tasks),
            "unique_families": len({task["family_id"] for task in tasks}),
            "by_source": dict(Counter(task["source_dataset"] for task in tasks)),
            "family_report": str(root / "families.report.json"),
            "task_file": str(path),
            "visual_requirements_decomposed": False,
            "visual_success_labels": False,
            "human_audit_accepted": False,
            "public_fields": ["original_request", "original_source", "request_span"],
        }
        write_report(public / "protocol_pilot_tasks.report.json", report)
        candidates = selected_families(rows, 128)
        for row in candidates:
            row["metadata"] = dict(row.get("metadata", {}))
            if row["source_dataset"] == "magicbrush":
                row["metadata"]["legacy_mask_requires_backfill"] = row.get("edit_mask_id")
                row["edit_mask_id"] = None
        review = root / "review"
        review.mkdir(exist_ok=True)
        write_jsonl(review / "pilot_candidates.jsonl", candidates)
        write_report(
            review / "pilot_candidates.report.json",
            {
                "status": "unverified_candidates",
                "rows": len(candidates),
                "by_source": dict(Counter(row["source_dataset"] for row in candidates)),
                "magicbrush_masks": "old_masks_excluded_pending_official_alpha_backfill",
                "human_audit_accepted": False,
            },
        )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
