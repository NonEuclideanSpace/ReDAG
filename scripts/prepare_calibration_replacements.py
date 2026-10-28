from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import io
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import normalize_image


ROOT = Path("data/calibration/ai_reserved/replacements")
REVISIONS = {"magicbrush": "6c254560879df7851233ea0fd2ae2a62764d0ef2", "omniedit": "82455c6cd66db7f0e5bfce8d7a236441af59d6df"}
REPOSITORIES = {"magicbrush": "osunlp/MagicBrush", "omniedit": "TIGER-Lab/OmniEdit-Filtered-1.2M"}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def fetch(url):
    for attempt in range(2):
        try:
            with urlopen(Request(url, headers={"User-Agent": "dag-pilot-calibration-replacements"}), timeout=25) as response:
                return response.read()
        except Exception:
            if attempt:
                raise


def fingerprint(image):
    array = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    matrix = np.cos(np.pi * (np.arange(32) + 0.5)[None, :] * np.arange(32)[:, None] / 32) * np.sqrt(2 / 32)
    matrix[0] /= np.sqrt(2)
    coefficients = (matrix @ array @ matrix.T)[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return sum(int(value) << index for index, value in enumerate(bits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--drafts", default="independent_requests_600_602.frozen.jsonl")
    arguments = parser.parse_args()
    root = arguments.root
    started = time.monotonic()
    rows = read(root / "reserved_replacements.frozen.jsonl")
    drafts = read(root / arguments.drafts)
    if {row["sample_id"] for row in rows} != {row["sample_id"] for row in drafts}:
        raise ValueError("Public request drafts must be frozen before image retrieval")
    manifest = read("data/families.protocol.jsonl")
    guard = read("data/training_expansion/isolation/guarded_value_index.jsonl")
    peer = [item | {"split_ids": ["training_expansion_reserved"]} for item in read("data/training_expansion/isolation/expanded_family_value_index.jsonl")]
    if len(guard) < 5624 or len(peer) < 978:
        raise ValueError("Complete current guarded and expanded training family fingerprints are required")
    known = guard + peer
    known_families = {family for item in known for family in item["family_ids"]}
    reserved_families = {item["family_id"] for item in read("data/calibration/ai_reserved/reserved_600.jsonl")}
    other_replacements = Path("data/calibration/ai_reserved/replacements/reserved_replacements.frozen.jsonl")
    if root.resolve() != other_replacements.parent.resolve():
        reserved_families.update(item["family_id"] for item in read(other_replacements))
    if any(row["family_id"] in known_families | reserved_families for row in rows):
        raise ValueError("Replacement family intersects a reserved or training family")
    if any(len([item for item in manifest if item["family_id"] == row["family_id"]]) != 1 for row in rows):
        raise ValueError("Replacement must be a complete single-row train family")
    verification = read("data/training_expansion/candidates_128/prepared.jsonl")[0]["images"][0]
    if fingerprint(Image.open(verification["local_review_path"])) != verification["phash"]:
        raise ValueError("Local orthonormal DCT-II fingerprint differs from the saved pipeline fingerprint")
    positions = {}
    for source in REPOSITORIES:
        positions[source] = {row["sample_id"]: index for index, row in enumerate(row for row in manifest if row["source_dataset"] == source and row["official_split"] == "train")}
    (root / "api").mkdir(exist_ok=True)
    (root / "local_images").mkdir(exist_ok=True)
    jobs = {}
    for row in rows:
        key = row["source_dataset"], positions[row["source_dataset"]][row["sample_id"]] // 100 * 100
        jobs.setdefault(key, []).append(row)

    def page(key, requested):
        source, offset = key
        url = "https://datasets-server.huggingface.co/rows?" + urlencode({"dataset": REPOSITORIES[source], "config": "default", "split": "train", "offset": offset, "length": 100})
        payload = json.loads(fetch(url))
        (root / "api" / f"{source}_{offset}.json").write_text(json.dumps(payload))
        lookup = {row["sample_id"]: row for row in requested}
        found = {}
        for entry in payload["rows"]:
            row = entry["row"]
            sample = f"magicbrush:{row['img_id']}:{row['turn_index']}" if source == "magicbrush" else "omniedit:" + row["omni_edit_id"]
            if sample not in lookup:
                continue
            instruction = row["instruction"] if source == "magicbrush" else max((text for text in row["edited_prompt_list"] if isinstance(text, str) and text.strip()), key=len)
            if instruction != lookup[sample]["instruction"] or entry.get("truncated_cells"):
                raise ValueError("Official sample/request mismatch")
            found[sample] = {**row, "_viewer_row_index": entry["row_idx"]}
        if set(found) != set(lookup):
            raise ValueError("Official exact sample lookup incomplete")
        return found

    viewers = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(page, key, requested) for key, requested in jobs.items()]):
            viewers.update(future.result())
            print(json.dumps({"stage": "metadata", "matched": len(viewers), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    store = ArtifactStore("data/artifacts")

    def image_job(row, role):
        viewer = viewers[row["sample_id"]]
        column = ("source_img" if role == "source" else "target_img") if row["source_dataset"] == "magicbrush" else ("src_img" if role == "source" else "edited_img")
        url = viewer[column]["src"]
        revision = REVISIONS[row["source_dataset"]]
        if f"/--/{revision}/--/" not in url:
            raise ValueError("Official revision mismatch")
        payload = fetch(url)
        image = normalize_image(Image.open(io.BytesIO(payload)))
        if image.size != (viewer[column]["width"], viewer[column]["height"]):
            raise ValueError("Official image dimensions mismatch")
        original = store.get_ref(row[f"{role}_image_id"])
        reference = store.put_image(image, metadata={"dataset_profile": "official-viewer-reencoded-v1", "original_artifact_id": original.artifact_id, "sample_id": row["sample_id"], "role": "source" if role == "source" else "private_target", "viewer_revision": revision, "original_revision": row["license_record"]["revision"]})
        path = root / "local_images" / f"{row['calibration_index']:03d}_{role}.jpg"
        path.write_bytes(payload)
        hashed = fingerprint(image)
        matches = [{"family_ids": item["family_ids"], "split_ids": item["split_ids"], "original_value_id": item["original_value_id"], "hamming_distance": (hashed ^ int(item["phash"])).bit_count()} for item in known if (hashed ^ int(item["phash"])).bit_count() <= 6 or original.value_id == item["original_value_id"]]
        return row["calibration_index"], {"sample_id": row["sample_id"], "original_artifact_id": original.artifact_id, "original_value_id": original.value_id, "artifact_id": reference.artifact_id, "value_id": reference.value_id, "relative_path": reference.relative_path, "candidate_role": role, "role": "source" if role == "source" else "private_target", "profile": "official-viewer-reencoded-v1", "viewer_revision": revision, "official_split": "train", "status": "downloaded", "width": image.width, "height": image.height, "bytes_downloaded": len(payload), "phash": hashed, "local_review_path": str(path.resolve()), "review_image_url": url, "external_near_matches": matches}

    images = {row["calibration_index"]: [] for row in rows}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(image_job, row, role) for row in rows for role in ("source", "target")]):
            index, image = future.result()
            images[index].append(image)
            print(json.dumps({"stage": "images", "images": sum(map(len, images.values())), "bytes": sum(image["bytes_downloaded"] for values in images.values() for image in values), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    prepared = []
    for row in rows:
        family_rows = [item for item in manifest if item["family_id"] == row["family_id"]]
        if len(family_rows) != 1:
            raise ValueError("Replacement requires additional full-family images")
        prepared.append({"sample_id": row["sample_id"], "family_id": row["family_id"], "calibration_index": row["calibration_index"], "calibration_fold": row["calibration_fold"], "request": row["instruction"], "source_dataset": row["source_dataset"], "license_record": row["license_record"], "images": sorted(images[row["calibration_index"]], key=lambda item: item["candidate_role"]), "verified_viewer_row_index": viewers[row["sample_id"]]["_viewer_row_index"], "quarantined": any(item["external_near_matches"] for item in images[row["calibration_index"]]), "human_reviewed": False, "ai_review_complete": False, "source_correct": None, "target_correct": None, "full_family_rows": len(family_rows), "expanded_training_additional_turn_check_pending": False})
    (root / "prepared.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prepared))
    report = {"prepared": len(prepared), "images": sum(map(len, images.values())), "guarded_unique_values": len(guard), "expanded_training_unique_values": len(peer), "expanded_training_families": len({family for item in peer for family in item["family_ids"]}), "quarantined": sum(row["quarantined"] for row in prepared), "download_bytes": sum(image["bytes_downloaded"] for values in images.values() for image in values), "elapsed_seconds": time.monotonic() - started, "labels_available": 0, "human_reviewed": False, "original_reservation_unchanged": True, "calibration_complete": False}
    (root / "preparation.report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
