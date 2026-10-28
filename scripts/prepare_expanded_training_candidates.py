import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import io
import json
from pathlib import Path
import time

import httpx
from PIL import Image, ImageDraw

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import normalize_image
from agentic_image.data.families import HammingIndex, perceptual_hash
from fetch_viewer_heldout import REVISIONS, request
from prepare_reserved_calibration import lookup


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def report(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/training_expansion/candidates_128")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--exclude-candidates", action="append", default=[])
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--magicbrush-count", type=int, default=64)
    parser.add_argument("--omniedit-count", type=int, default=64)
    arguments = parser.parse_args()
    quotas = {"magicbrush": arguments.magicbrush_count, "omniedit": arguments.omniedit_count}
    expected = sum(quotas.values())
    if expected < 1 or min(quotas.values()) < 0:
        raise ValueError("Candidate quotas must be nonnegative and nonempty")
    output = Path(arguments.output)
    for name in ("api", "local_images", "contact_sheets"):
        (output / name).mkdir(parents=True, exist_ok=True)
    manifest = read("data/families.protocol.jsonl")
    store = ArtifactStore("data/artifacts")
    reserved = read("data/calibration/ai_reserved/reserved_600.jsonl")
    exposed = read("data/review/viewer_candidates.jsonl")
    existing = read("data/review/ai-global-reviews-a.jsonl") + read("data/review/ai_review_b/reviews.request_global.jsonl")
    accepted = [row for row in existing if row.get("accepted") is True or row.get("decision") == "accept"]
    cohort = {"magicbrush:" + item for item in ("513551:1", "442654:1", "138270:1", "262146:1", "327726:1", "61184:1", "442416:1", "357633:1")}
    assert len(accepted) == 17 and cohort <= {row["sample_id"] for row in accepted}
    blocked_rows = [row for row in manifest if row["split_id"] != "quarantine-not-screened-for-pilot"] + reserved + exposed
    replacement_registry = Path("data/calibration/ai_reserved/replacements/reserved_replacements.frozen.jsonl")
    if replacement_registry.exists():
        blocked_rows += read(replacement_registry)
    blocked_families = {row["family_id"] for row in blocked_rows}
    for exclusion in arguments.exclude_candidates:
        blocked_families.update(row["family_id"] for row in read(exclusion))
    blocked_samples = {row["sample_id"] for row in blocked_rows}
    blocked_ids = {row[field] for row in blocked_rows for field in ("source_image_id", "target_image_id") if row.get(field)}
    blocked_values = {store.get_ref(identifier).value_id for identifier in blocked_ids}
    groups = defaultdict(list)
    excluded = Counter()
    for row in manifest:
        if row["official_split"] != "train" or row["split_id"] != "quarantine-not-screened-for-pilot":
            continue
        if row["family_id"] in blocked_families or row["sample_id"] in blocked_samples:
            excluded["blocked_family_or_sample"] += 1
            continue
        values = {store.get_ref(row[field]).value_id for field in ("source_image_id", "target_image_id")}
        if values & blocked_values:
            excluded["blocked_exact_value"] += 1
            continue
        groups[row["family_id"]].append(row)
    ordered = sorted(groups, key=lambda family: sha256(("expanded-training-128-v1:" + family).encode()).hexdigest())
    selected, counts, seen_values = [], Counter(), set()
    for family in ordered:
        row = sorted(groups[family], key=lambda item: (item.get("metadata", {}).get("turn_index", 1), item["sample_id"]))[0]
        source = row["source_dataset"]
        if counts[source] >= quotas[source]:
            continue
        values = {store.get_ref(row[field]).value_id for field in ("source_image_id", "target_image_id")}
        if values & seen_values:
            excluded["selected_exact_value_conflict"] += 1
            continue
        seen_values.update(values)
        selected.append({**row, "expansion_index": arguments.index_offset + len(selected), "original_split_id": row["split_id"],
                         "training_eligible": False, "candidate_status": "pending_full_visual_isolation_and_ai_review",
                         "public_plan": {"global_text": row["instruction"], "regions": []},
                         "plan_rule": "original_request_global_pair_v1", "human_reviewed": False})
        counts[source] += 1
    if len(selected) != expected:
        raise ValueError("The requested independent candidate quota cannot be met")
    frozen = output / "candidates.frozen.jsonl"
    if frozen.exists() and read(frozen) != selected:
        raise ValueError("The frozen expansion candidates must not change")
    write(frozen, selected)
    write(output / "existing_accepted17.registry.jsonl", [{"sample_id": row["sample_id"], "family_id": row["family_id"],
          "existing_fixed_cohort8": row["sample_id"] in cohort, "candidate128_member": False,
          "review_type": "ai", "human_reviewed": False, "review_revision": row["review_revision"]} for row in accepted])
    selection_report = {"selected_candidates": len(selected), "distinct_families": len({row["family_id"] for row in selected}),
                        "source_counts": dict(counts), "identity_eligible_family_pool": len(groups),
                        "excluded": dict(excluded), "reserved_calibration_family_count": len({row["family_id"] for row in reserved}),
                        "exact_blocked_value_count": len(blocked_values), "existing_accepted17_unchanged": True,
                        "fixed_cohort8_unchanged": True, "training_ready_count": 0,
                        "full_heldout_phash_complete": False, "ai_review_complete": False,
                        "estimated_compressed_download_bytes": 41107000,
                        "estimate_basis": "Existing calibration download means: MagicBrush 73153 and OmniEdit 247997 bytes per image"}
    report(output / "selection.report.json", selection_report)
    print(json.dumps(selection_report), flush=True)
    if arguments.select_only:
        return
    prepared_path = output / "prepared.jsonl"
    prepared = {row["sample_id"]: row for row in read(prepared_path)} if prepared_path.exists() else {}
    pending = [row for row in selected if row["sample_id"] not in prepared]
    pages = defaultdict(list)
    for source in ("magicbrush", "omniedit"):
        positions = {row["sample_id"]: index for index, row in enumerate(row for row in manifest if row["source_dataset"] == source and row["official_split"] == "train")}
        for row in pending:
            if row["source_dataset"] == source:
                pages[(source, positions[row["sample_id"]] // 100 * 100)].append(row)
    started, total_bytes, failures, viewers = time.monotonic(), 0, [], {}
    with httpx.Client(follow_redirects=True, timeout=35, limits=httpx.Limits(max_connections=20)) as client:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(lookup, client, source, offset, rows, output / "api"): (source, offset) for (source, offset), rows in pages.items()}
            for future in as_completed(futures):
                try:
                    viewers.update(future.result())
                except Exception as error:
                    failures.append({"stage": "metadata", "page": futures[future], "error": type(error).__name__ + ": " + str(error)[:250]})
                print(json.dumps({"stage": "official_metadata", "matched": len(viewers), "failures": len(failures), "seconds": round(time.monotonic() - started, 1)}), flush=True)

        def prepare(row):
            viewer = viewers[row["sample_id"]]
            columns = ("source_img", "target_img") if row["source_dataset"] == "magicbrush" else ("src_img", "edited_img")
            images = []
            for role, column, field in zip(("source", "target"), columns, ("source_image_id", "target_image_id")):
                url = viewer[column]["src"]
                revision = REVISIONS[row["source_dataset"]]
                if f"/--/{revision}/--/" not in url:
                    raise ValueError("The official viewer revision does not match the recorded data profile")
                payload = request(client, url).content
                image = normalize_image(Image.open(io.BytesIO(payload)))
                if image.size != (viewer[column]["width"], viewer[column]["height"]):
                    raise ValueError("The downloaded image dimensions do not match official metadata")
                path = output / "local_images" / f"{row['expansion_index']:03d}_{role}.jpg"
                path.write_bytes(payload)
                original = store.get_ref(row[field])
                ref = store.put_image(image, metadata={"dataset_profile": "official-viewer-reencoded-v1", "original_artifact_id": row[field],
                      "sample_id": row["sample_id"], "role": role if role == "source" else "private_target",
                      "viewer_revision": revision, "original_revision": row["license_record"]["revision"]})
                images.append({"candidate_role": role, "original_artifact_id": row[field], "original_value_id": original.value_id,
                               "artifact_id": ref.artifact_id, "value_id": ref.value_id, "relative_path": ref.relative_path,
                               "local_review_path": str(path.resolve()), "phash": perceptual_hash(image), "bytes_downloaded": len(payload),
                               "width": image.width, "height": image.height, "viewer_revision": revision})
            return {"expansion_index": row["expansion_index"], "sample_id": row["sample_id"], "family_id": row["family_id"],
                    "source_dataset": row["source_dataset"], "request": row["instruction"], "public_plan": row["public_plan"],
                    "images": images, "verified_viewer_row_index": viewer["_viewer_row_index"],
                    "training_eligible": False, "human_reviewed": False, "ai_review_complete": False}

        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            futures = {executor.submit(prepare, row): row for row in pending if row["sample_id"] in viewers}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                    prepared[row["sample_id"]] = result
                    total_bytes += sum(image["bytes_downloaded"] for image in result["images"])
                    with prepared_path.open("a") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as error:
                    failures.append({"stage": "image", "sample_id": row["sample_id"], "error": type(error).__name__ + ": " + str(error)[:250]})
                state = {"stage": "images", "prepared_pairs": len(prepared), "expected_pairs": expected,
                         "download_bytes_this_run": total_bytes, "seconds": round(time.monotonic() - started, 1), "failures": len(failures)}
                report(output / "progress.json", state)
                print(json.dumps(state), flush=True)
    pairs = sorted(prepared.values(), key=lambda row: row["expansion_index"])
    for offset in range(arguments.index_offset, arguments.index_offset + expected, 4):
        group = [row for row in pairs if offset <= row["expansion_index"] < offset + 4]
        if len(group) != min(4, arguments.index_offset + expected - offset):
            continue
        canvas = Image.new("RGB", (1024, 2208), "white")
        draw = ImageDraw.Draw(canvas)
        for y, row in enumerate(group):
            for x, image in enumerate(row["images"]):
                with Image.open(image["local_review_path"]) as original:
                    thumb = original.convert("RGB")
                    thumb.thumbnail((512, 512))
                    canvas.paste(thumb, (x * 512 + (512 - thumb.width) // 2, y * 552 + 40 + (512 - thumb.height) // 2))
                draw.text((x * 512 + 8, y * 552 + 12), f"{row['expansion_index']:03d} | {image['candidate_role']} | {row['sample_id']}", fill="black")
        canvas.save(output / "contact_sheets" / f"{offset:03d}-{offset + 3:03d}.jpg", quality=95)
    final = {**selection_report, "prepared_pairs": len(pairs), "download_bytes_this_run": total_bytes,
             "elapsed_seconds": time.monotonic() - started, "failures": failures,
             "status": "downloaded_pending_isolation_and_ai_review" if len(pairs) == expected else "partial_download"}
    report(output / "download.report.json", final)
    print(json.dumps({key: value for key, value in final.items() if key != "failures"}), flush=True)


if __name__ == "__main__":
    main()
