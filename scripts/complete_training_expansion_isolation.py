import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time

import httpx
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import normalize_image
from agentic_image.data.families import HammingIndex, perceptual_hash
from fetch_viewer_heldout import REVISIONS, request
from prepare_reserved_calibration import lookup
from prepare_expanded_training_candidates import read, write, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/training_expansion/isolation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--official-index", default="data/viewer/images.complete.remote.jsonl")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--candidates", action="append")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    (output / "fingerprint_images").mkdir(parents=True, exist_ok=True)
    manifest = read("data/families.protocol.jsonl")
    store = ArtifactStore("data/artifacts")
    refs = {}
    candidate_directories = [Path(value) for value in arguments.candidates] if arguments.candidates else [Path("data/training_expansion/candidates_128"), Path("data/training_expansion/candidates_128_reserve"), Path("data/training_expansion/candidates_extra_magicbrush")]
    candidate_directories = [directory for directory in candidate_directories if (directory / "candidates.frozen.jsonl").is_file()]
    frozen_candidates = [row for directory in candidate_directories for row in read(directory / "candidates.frozen.jsonl")]
    candidates = [row for directory in candidate_directories if (directory / "prepared.jsonl").is_file() for row in read(directory / "prepared.jsonl")]
    candidate_families = {row["family_id"] for row in frozen_candidates}

    def value(identifier):
        if identifier not in refs:
            refs[identifier] = store.get_ref(identifier).value_id
        return refs[identifier]

    known = {}
    for row in read("data/calibration/ai_reserved/batch_600.jsonl"):
        for image in row["images"]:
            known[image["original_value_id"]] = {**image, "evidence_source": "reserved_calibration_actual_download"}
    for name in ("data/viewer/images.jsonl", "data/viewer/training_candidate_images.jsonl", arguments.official_index):
        if not Path(name).is_file():
            continue
        for row in read(name):
            if row.get("phash") is not None and row.get("original_value_id"):
                known[row["original_value_id"]] = {**row, "evidence_source": name}
    ready = output / "supplementary_images.jsonl"
    if ready.exists():
        for row in read(ready):
            known[row["original_value_id"]] = row
    for row in candidates:
        for image in row["images"]:
            known[image["original_value_id"]] = {**image, "evidence_source": "expanded_candidate_actual_download"}
    replacements_path = Path("data/calibration/ai_reserved/replacements/prepared.jsonl")
    replacements = read(replacements_path) if replacements_path.is_file() else []
    replacement_families = {row["family_id"] for row in replacements}
    for row in replacements:
        for image in row["images"]:
            known[image["original_value_id"]] = {**image, "evidence_source": "reserved_calibration_replacement_actual_download"}
    guarded = [row for row in manifest if row["split_id"] != "quarantine-not-screened-for-pilot"]
    guarded += [row for row in manifest if row["family_id"] in replacement_families]
    aliases = defaultdict(list)
    for row in guarded:
        for role, field in (("source", "source_image_id"), ("target", "target_image_id")):
            aliases[value(row[field])].append((row, role, row[field]))
    candidate_aliases = defaultdict(list)
    for row in manifest:
        if row["family_id"] in candidate_families:
            for role, field in (("source", "source_image_id"), ("target", "target_image_id")):
                candidate_aliases[value(row[field])].append((row, role, row[field]))
    combined_aliases = {**aliases, **candidate_aliases}
    missing = {identity: rows for identity, rows in combined_aliases.items() if identity not in known and rows[0][0]["official_split"] == "train"}
    overview = {"guarded_families": len({row["family_id"] for row in guarded}), "guarded_unique_values": len(aliases),
                "known_guarded_values": len(set(aliases) & set(known)), "missing_train_reserved_values": len(missing),
                "missing_train_reserved_by_split": dict(Counter(rows[0][0]["split_id"] for rows in missing.values())),
                "official_index_exists": Path(arguments.official_index).is_file(), "training_eligible": False}
    overview.update({"candidate_family_unique_values": len(candidate_aliases),
                     "missing_candidate_family_values": len(set(candidate_aliases) - set(known))})
    report(output / "coverage.before.json", overview)
    print(json.dumps(overview), flush=True)
    started, failures, bytes_downloaded = time.monotonic(), [], 0
    if arguments.download and missing:
        pages = defaultdict(dict)
        for source in ("magicbrush", "omniedit"):
            positions = {row["sample_id"]: index for index, row in enumerate(row for row in manifest if row["source_dataset"] == source and row["official_split"] == "train")}
            for rows in missing.values():
                row = rows[0][0]
                if row["source_dataset"] == source:
                    pages[(source, positions[row["sample_id"]] // 100 * 100)][row["sample_id"]] = row
        viewers = {}
        with httpx.Client(follow_redirects=True, timeout=35, limits=httpx.Limits(max_connections=16)) as client:
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {executor.submit(lookup, client, source, offset, list(rows.values()), Path("data/training_expansion/candidates_128/api")): (source, offset) for (source, offset), rows in pages.items()}
                for future in as_completed(futures):
                    try:
                        viewers.update(future.result())
                    except Exception as error:
                        failures.append({"stage": "metadata", "page": futures[future], "error": type(error).__name__})
                    print(json.dumps({"stage": "supplementary_metadata", "matched_samples": len(viewers), "failures": len(failures)}), flush=True)

            def fetch(identity, rows):
                row, role, identifier = rows[0]
                viewer = viewers[row["sample_id"]]
                column = (("source_img" if role == "source" else "target_img") if row["source_dataset"] == "magicbrush" else ("src_img" if role == "source" else "edited_img"))
                url = viewer[column]["src"]
                revision = REVISIONS[row["source_dataset"]]
                if f"/--/{revision}/--/" not in url:
                    raise ValueError("The official viewer revision differs from the fixed profile")
                payload = request(client, url).content
                image = normalize_image(Image.open(io.BytesIO(payload)))
                if image.size != (viewer[column]["width"], viewer[column]["height"]):
                    raise ValueError("Official image dimensions do not match the downloaded pixels")
                destination = output / "fingerprint_images" / (identity.split(":")[1] + ".jpg")
                destination.write_bytes(payload)
                return {"original_value_id": identity, "original_artifact_id": identifier, "phash": perceptual_hash(image),
                        "evidence_source": "actual_supplementary_download", "bytes_downloaded": len(payload),
                        "viewer_revision": revision, "sample_id": row["sample_id"], "role": role,
                        "local_image_path": str(destination.resolve()), "width": image.width, "height": image.height,
                        "family_ids": sorted({item[0]["family_id"] for item in rows}),
                        "training_eligible": False, "purpose": "reserved_family_visual_isolation_only"}

            with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
                futures = {executor.submit(fetch, identity, rows): identity for identity, rows in missing.items() if rows[0][0]["sample_id"] in viewers}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        known[result["original_value_id"]] = result
                        bytes_downloaded += result["bytes_downloaded"]
                        with ready.open("a") as stream:
                            stream.write(json.dumps(result) + "\n")
                    except Exception as error:
                        failures.append({"stage": "image", "value_id": futures[future], "error": type(error).__name__})
                    progress = {"stage": "supplementary_images", "downloaded_this_run": len(set(missing) & set(known)),
                                "expected_this_run": len(missing), "bytes_this_run": bytes_downloaded,
                                "elapsed_seconds": round(time.monotonic() - started, 1), "failures": len(failures)}
                    report(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    coverage = {**overview, "known_guarded_values": len(set(aliases) & set(known)),
                "missing_guarded_values": len(set(aliases) - set(known)), "download_bytes_this_run": bytes_downloaded,
                "elapsed_seconds": time.monotonic() - started, "failures": failures,
                "full_visual_isolation_complete": False}
    report(output / "coverage.after.json", coverage)
    if set(aliases) - set(known):
        print(json.dumps({"status": "incomplete_guarded_fingerprint_coverage", "missing": len(set(aliases) - set(known))}), flush=True)
        return
    if set(candidate_aliases) - set(known):
        coverage.update({"missing_candidate_family_values": len(set(candidate_aliases) - set(known)),
                         "full_candidate_family_phash_screening_complete": False})
        report(output / "coverage.after.json", coverage)
        print(json.dumps({"status": "incomplete_candidate_family_fingerprint_coverage", "missing": len(set(candidate_aliases) - set(known))}), flush=True)
        return
    write(output / "guarded_value_index.jsonl", [{"original_value_id": identity,
          "family_ids": sorted({item[0]["family_id"] for item in items}),
          "split_ids": sorted({item[0]["split_id"] for item in items}),
          "phash": known[identity]["phash"], "evidence_source": known[identity]["evidence_source"]}
          for identity, items in sorted(aliases.items())])
    write(output / "expanded_family_value_index.jsonl", [{"original_value_id": identity,
          "family_ids": sorted({item[0]["family_id"] for item in items}), "phash": known[identity]["phash"],
          "evidence_source": known[identity]["evidence_source"]} for identity, items in sorted(candidate_aliases.items())])
    index = HammingIndex()
    for identity, items in aliases.items():
        for family in sorted({item[0]["family_id"] for item in items}):
            index.add(int(known[identity]["phash"]), family)
    expected_candidate_count = len(frozen_candidates)
    if not candidates:
        return
    if len({row["family_id"] for row in candidates}) != len(candidates):
        raise ValueError("Expanded candidates contain duplicate families")
    peer = HammingIndex()
    conflicts = defaultdict(set)
    external_by_family = defaultdict(set)
    for identity, items in sorted(candidate_aliases.items()):
        fingerprint = int(known[identity]["phash"])
        for family in sorted({item[0]["family_id"] for item in items}):
            external_by_family[family].update(index.query(fingerprint, radius=6))
            matches = set(peer.query(fingerprint, radius=6)) - {family}
            for previous in matches:
                conflicts[family].add(previous)
                conflicts[previous].add(family)
            peer.add(fingerprint, family)
    results = []
    for row in sorted(candidates, key=lambda item: item["expansion_index"]):
        external = set(external_by_family[row["family_id"]])
        for image in row["images"]:
            external.update(index.query(int(image["phash"]), radius=6))
            matches = set(peer.query(int(image["phash"]), radius=6)) - {row["family_id"]}
            for family in matches:
                conflicts[row["family_id"]].add(family)
                conflicts[family].add(row["family_id"])
            peer.add(int(image["phash"]), row["family_id"])
        results.append({"expansion_index": row["expansion_index"], "sample_id": row["sample_id"], "family_id": row["family_id"],
                        "full_guarded_value_coverage": True, "phash_radius": 6, "external_near_families": sorted(external),
                        "full_candidate_family_value_coverage": True,
                        "training_eligible": False, "ai_review_complete": False})
    for row in results:
        row["candidate_near_families"] = sorted(conflicts[row["family_id"]])
        row["isolation_passed"] = not row["external_near_families"] and not row["candidate_near_families"]
    write(output / "candidate_isolation.jsonl", results)
    coverage.update({"full_visual_isolation_complete": False,
                     "full_guarded_phash_screening_complete": len(results) == expected_candidate_count,
                     "full_candidate_family_phash_screening_complete": len(results) == expected_candidate_count,
                     "missing_candidate_family_values": 0,
                     "screening_method": "Complete guarded source/target value coverage with pHash Hamming radius 6; every near match is rejected pending separate visual review",
                     "candidate_count": len(results), "expected_candidate_count": expected_candidate_count,
                     "isolation_passed": sum(row["isolation_passed"] for row in results),
                     "quarantined_candidates": sum(not row["isolation_passed"] for row in results)})
    report(output / "coverage.after.json", coverage)
    print(json.dumps({key: val for key, val in coverage.items() if key != "failures"}), flush=True)


if __name__ == "__main__":
    main()
