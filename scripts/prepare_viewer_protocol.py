from collections import defaultdict
from hashlib import blake2b
import json
from pathlib import Path

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.types import (
    CheckSpec,
    PublicCheckAtom,
    PublicTask,
    RequestSpan,
    UserRequest,
)
from agentic_image.data.families import HammingIndex, UnionFind, perceptual_hash
from agentic_image.data.io import read_jsonl, write_jsonl, write_report


def main():
    root = Path("data")
    store = ArtifactStore(root / "artifacts")
    rows = list(read_jsonl(root / "raw/all_saved.jsonl"))
    row_index = {row["sample_id"]: index for index, row in enumerate(rows)}
    union = UnionFind(len(rows))
    owners = {}
    for index, row in enumerate(rows):
        keys = [("session", row["session_id"])]
        keys.extend(("family_link", value) for value in row.get("family_links", []))
        if row.get("template_instance_id"):
            keys.append(("template", row["template_instance_id"]))
        for identifier in (
            row["source_image_id"],
            row["target_image_id"],
            *row.get("appearance_image_ids", []),
        ):
            keys.append(("value", store.get_ref(identifier).value_id))
        for key in keys:
            if key in owners:
                union.union(index, owners[key])
            else:
                owners[key] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[union.find(index)].append(index)
    families = {
        group: "family:"
        + blake2b(
            "\n".join(sorted(rows[index]["sample_id"] for index in members)).encode(),
            digest_size=16,
        ).hexdigest()
        for group, members in groups.items()
    }
    official_groups = {
        union.find(index)
        for index, row in enumerate(rows)
        if row["official_split"] in {"dev", "test"}
    }
    images = {
        row["original_artifact_id"]: row
        for row in read_jsonl(root / "viewer/images.jsonl")
        if row["status"] == "downloaded"
    }
    required = {
        row[field]
        for row in rows
        if row["official_split"] in {"dev", "test"}
        for field in ("source_image_id", "target_image_id")
    }
    missing = sorted(required - images.keys())
    report = {
        "profile": "official-viewer-reencoded-v1",
        "source_rows": len(rows),
        "exact_families": len(groups),
        "official_records": sum(row["official_split"] in {"dev", "test"} for row in rows),
        "official_image_references_required": len(required),
        "official_image_references_missing": len(missing),
        "human_audit_accepted": False,
        "visual_success_labels": False,
        "paired_registration_reusable": False,
        "renderer_sft_allowed": False,
        "full_actor_sft_allowed": False,
    }
    output = root / "public"
    output.mkdir(exist_ok=True)
    if missing:
        report.update(status="blocked_missing_official_images", missing_artifacts=missing)
        write_report(output / "protocol_pilot_tasks.report.json", report)
        raise ValueError(
            f"Official held-out image evidence is incomplete: {len(missing)} references"
        )
    heldout = HammingIndex()
    for identifier in sorted(required):
        entry = images[identifier]
        if "phash" not in entry:
            raise ValueError("An official held-out image lacks its measured perceptual fingerprint")
        heldout.add(int(entry["phash"]), identifier)
    candidates = []
    rejected = []
    seen = set()
    for entry in images.values():
        sample = entry["sample_id"]
        row = rows[row_index[sample]]
        if row["official_split"] != "train" or entry["role"] != "source":
            continue
        group = union.find(row_index[sample])
        if group in seen:
            continue
        seen.add(group)
        if group in official_groups:
            rejected.append(
                {
                    "sample_id": sample,
                    "family_id": families[group],
                    "reason": "exact_official_overlap",
                }
            )
            continue
        fingerprint = int(
            entry.get("phash", perceptual_hash(store.get_image(entry["artifact_id"])))
        )
        matches = heldout.query(fingerprint, radius=6)
        if matches:
            rejected.append(
                {
                    "sample_id": sample,
                    "family_id": families[group],
                    "reason": "unresolved_official_near_duplicate",
                    "matched_artifacts": matches,
                }
            )
            continue
        candidates.append((row, group, entry, fingerprint))
    local = HammingIndex()
    ambiguous = set()
    for _, group, _, fingerprint in candidates:
        matches = local.query(fingerprint, radius=6)
        ambiguous.update(matches)
        if matches:
            ambiguous.add(group)
        local.add(fingerprint, group)
    selected = [item for item in candidates if item[1] not in ambiguous][:32]
    for row, group, _, _ in candidates:
        if group in ambiguous:
            rejected.append(
                {
                    "sample_id": row["sample_id"],
                    "family_id": families[group],
                    "reason": "unresolved_training_near_duplicate",
                }
            )
    if len(selected) < 16:
        report.update(
            status="blocked_insufficient_isolated_protocol_families",
            protocol_families=len(selected),
            quarantined=rejected,
        )
        write_report(output / "protocol_pilot_tasks.report.json", report)
        raise ValueError(
            f"Only {len(selected)} protocol families passed the registered isolation scope"
        )
    selected_groups = {item[1] for item in selected}
    rejected_families = {item["family_id"] for item in rejected}
    rejected_groups = {group for group, family in families.items() if family in rejected_families}
    eligible = sorted(
        (
            group
            for group in groups
            if group not in official_groups | selected_groups | rejected_groups
        ),
        key=lambda group: blake2b(f"20261027:{families[group]}".encode(), digest_size=16).digest(),
    )
    minimums = {"development": 64, "calibration": 600, "diagnostic-test": 32}
    if len(eligible) < sum(minimums.values()):
        raise ValueError(
            "Independent exact families cannot fill the declared held-out reservations"
        )
    reservations = {}
    offset = 0
    for split, count in minimums.items():
        reservations.update({group: split for group in eligible[offset : offset + count]})
        offset += count
    tasks = []
    transfer = []
    for row, group, entry, _ in selected:
        text = row["instruction"]
        task = PublicTask(
            task_id=row["sample_id"],
            family_id=families[group],
            split_id="train",
            source_dataset=row["source_dataset"],
            license_record={**row["license_record"], "dataset_profile": report["profile"]},
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
        transfer.extend(
            [
                str(root / "artifacts/refs" / (entry["artifact_id"].split(":")[1] + ".json")),
                str(root / "artifacts" / entry["relative_path"]),
            ]
        )
    for index, row in enumerate(rows):
        group = union.find(index)
        row["family_id"] = families[group]
        row["split_id"] = (
            f"official-{row['official_split']}"
            if row["official_split"] in {"dev", "test"}
            else "quarantine-official-overlap"
            if group in official_groups
            else "train-protocol-only"
            if group in selected_groups
            else "quarantine-near-duplicate"
            if group in rejected_groups
            else reservations.get(group, "quarantine-not-screened-for-pilot")
        )
    write_jsonl(root / "families.protocol.jsonl", rows)
    write_jsonl(output / "protocol_pilot_tasks.jsonl", tasks)
    report.update(
        status="ready_for_protocol_recovery_only",
        tasks=len(tasks),
        unique_families=len(selected_groups),
        training_stages_allowed=["protocol-recovery-only"],
        isolation_scope="All saved records use exact image-value and session links; every exposed protocol source is screened against all official held-out source and target viewer images; unresolved Hamming-distance <=6 matches are quarantined.",
        paired_family_perceptual_audit_complete=False,
        reserved_exact_families=minimums,
        reserved_family_visual_audit_complete=False,
        calibration_human_labels_available=False,
        quarantined=rejected,
        quarantined_candidate_families=len(rejected_groups),
    )
    write_report(output / "protocol_pilot_tasks.report.json", report)
    transfer.extend(
        [
            str(output / "protocol_pilot_tasks.jsonl"),
            str(output / "protocol_pilot_tasks.report.json"),
            str(root / "families.protocol.jsonl"),
        ]
    )
    (root / "transfer").mkdir(exist_ok=True)
    (root / "transfer/protocol_ready_files.txt").write_text(
        "".join(path + "\n" for path in sorted(set(transfer)))
    )
    print(
        json.dumps({key: value for key, value in report.items() if key != "quarantined"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
