from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
from urllib.parse import quote

import httpx
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import normalize_image
from agentic_image.data.families import perceptual_hash
from agentic_image.data.io import append_jsonl, read_jsonl, write_report


ROOT = Path("data")
DIRECTORY = ROOT / "viewer"
PROFILE = "official-viewer-reencoded-v1"
REVISIONS = {
    "magicbrush": "6c254560879df7851233ea0fd2ae2a62764d0ef2",
    "omniedit": "82455c6cd66db7f0e5bfce8d7a236441af59d6df",
}
REPOSITORIES = {"magicbrush": "osunlp/MagicBrush", "omniedit": "TIGER-Lab/OmniEdit-Filtered-1.2M"}


def request(client, url):
    for attempt in range(2):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, OSError):
            if attempt:
                raise
            time.sleep(1)


def page(client, source, split, offset):
    path = DIRECTORY / "api" / f"{source}_{split}_{offset}.json"
    if path.is_file() and time.time() - path.stat().st_mtime < 1800:
        data = json.loads(path.read_text())
    else:
        url = (
            "https://datasets-server.huggingface.co/rows?dataset="
            + quote(REPOSITORIES[source], safe="")
            + f"&config=default&split={split}&offset={offset}&length=100"
        )
        data = request(client, url).json()
        path.write_text(json.dumps(data))
    return source, split, data["rows"]


def download(client, store, aliases):
    source, split, item, column, original = aliases[0]
    url = item[column]["src"]
    revision = REVISIONS[source]
    if f"/--/{revision}/--/" not in url:
        raise ValueError("Dataset viewer revision differs from the recorded data profile")
    payload = request(client, url).content
    image = normalize_image(Image.open(io.BytesIO(payload)))
    if image.size != (item[column]["width"], item[column]["height"]):
        raise ValueError("Dataset viewer dimensions do not match its image metadata")
    fingerprint = perceptual_hash(image)
    outputs = []
    for index, (source, split, item, column, original) in enumerate(aliases):
        role = "source" if column in {"source_img", "src_img"} else "private_target"
        ref = store.put_image(
            image,
            metadata={
                "dataset_profile": PROFILE,
                "original_artifact_id": original["artifact_id"],
                "sample_id": original["sample_id"],
                "role": role,
                "viewer_revision": revision,
                "original_revision": original["revision"],
            },
        )
        outputs.append(
            {
                "sample_id": original["sample_id"],
                "original_artifact_id": original["artifact_id"],
                "original_value_id": original["value_id"],
                "artifact_id": ref.artifact_id,
                "value_id": ref.value_id,
                "relative_path": ref.relative_path,
                "bytes_downloaded": len(payload) if index == 0 else 0,
                "role": role,
                "official_split": split,
                "status": "downloaded",
                "profile": PROFILE,
                "phash": fingerprint,
                "viewer_revision": revision,
            }
        )
    return outputs


def main():
    DIRECTORY.mkdir(exist_ok=True)
    (DIRECTORY / "api").mkdir(exist_ok=True)
    store = ArtifactStore(ROOT / "artifacts")
    raw = {row["sample_id"]: row for row in read_jsonl(ROOT / "raw/all_saved.jsonl")}
    completed = {
        row["original_artifact_id"]
        for row in read_jsonl(DIRECTORY / "images.jsonl")
        if row["status"] == "downloaded"
    }
    requests = [
        (source, split, offset)
        for source, split, count in (
            ("magicbrush", "dev", 528),
            ("magicbrush", "test", 1053),
            ("omniedit", "dev", 700),
        )
        for offset in range(0, count, 100)
    ]
    jobs = {}
    errors = []
    api_rows = 0
    started = time.monotonic()
    with httpx.Client(
        follow_redirects=True, timeout=20, limits=httpx.Limits(max_connections=16)
    ) as client:
        with ThreadPoolExecutor(max_workers=8) as executor:
            pending = {executor.submit(page, client, *values): values for values in requests}
            for future in as_completed(pending):
                try:
                    source, split, rows = future.result()
                    for entry in rows:
                        row = entry["row"]
                        identifier = (
                            f"magicbrush:{row['img_id']}:{row['turn_index']}"
                            if source == "magicbrush"
                            else f"omniedit:{row['omni_edit_id']}"
                        )
                        original = raw.get(identifier)
                        if original is None or original["official_split"] != split:
                            raise ValueError("Viewer row is absent from the saved official split")
                        instruction = (
                            row["instruction"]
                            if source == "magicbrush"
                            else max(
                                (
                                    text
                                    for text in row["edited_prompt_list"]
                                    if isinstance(text, str) and text.strip()
                                ),
                                key=len,
                                default="",
                            )
                        )
                        if instruction != original["instruction"] or entry["truncated_cells"]:
                            raise ValueError("Viewer row failed exact sample/instruction matching")
                        api_rows += 1
                        columns = (
                            ("source_img", "target_img")
                            if source == "magicbrush"
                            else ("src_img", "edited_img")
                        )
                        for column, field in zip(columns, ("source_image_id", "target_image_id")):
                            identifier = original[field]
                            if identifier in completed:
                                continue
                            reference = store.get_ref(identifier)
                            metadata = {
                                "sample_id": original["sample_id"],
                                "artifact_id": identifier,
                                "value_id": reference.value_id,
                                "revision": original["license_record"]["revision"],
                            }
                            jobs.setdefault(reference.value_id, []).append(
                                (source, split, row, column, metadata)
                            )
                    print(
                        json.dumps(
                            {
                                "stage": "official_metadata",
                                "matched_rows": api_rows,
                                "unique_image_jobs": len(jobs),
                                "seconds": round(time.monotonic() - started, 1),
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    failure = {"stage": "metadata", "page": pending[future], "error": str(error)}
                    errors.append(failure)
                    append_jsonl(DIRECTORY / "failures.jsonl", [failure])
                    print(
                        json.dumps(
                            {
                                "stage": "metadata_failure",
                                "page": pending[future],
                                "error_type": type(error).__name__,
                            }
                        ),
                        flush=True,
                    )
        successes = 0
        image_references = 0
        total_bytes = 0
        report_at = 0
        with ThreadPoolExecutor(max_workers=12) as executor:
            pending = {
                executor.submit(download, client, store, aliases): key
                for key, aliases in jobs.items()
            }
            for future in as_completed(pending):
                try:
                    outputs = future.result()
                    append_jsonl(DIRECTORY / "images.jsonl", outputs)
                    successes += 1
                    image_references += len(outputs)
                    total_bytes += sum(item["bytes_downloaded"] for item in outputs)
                except Exception as error:
                    failure = {
                        "stage": "image",
                        "original_value_id": pending[future],
                        "error": str(error),
                    }
                    errors.append(failure)
                    append_jsonl(DIRECTORY / "failures.jsonl", [failure])
                    print(
                        json.dumps(
                            {
                                "stage": "image_failure",
                                "original_value_id": pending[future],
                                "error_type": type(error).__name__,
                            }
                        ),
                        flush=True,
                    )
                elapsed = time.monotonic() - started
                if elapsed >= report_at:
                    report = {
                        "stage": "official_images",
                        "images": successes,
                        "image_references": image_references,
                        "total_jobs": len(jobs),
                        "bytes_downloaded": total_bytes,
                        "seconds": round(elapsed, 1),
                        "failures": len(errors),
                    }
                    print(json.dumps(report), flush=True)
                    write_report(DIRECTORY / "heldout_progress.json", report)
                    report_at = elapsed + 30
        required = {
            row[field]
            for row in raw.values()
            if row["official_split"] in {"dev", "test"}
            for field in ("source_image_id", "target_image_id")
        }
        present = {
            row["original_artifact_id"]
            for row in read_jsonl(DIRECTORY / "images.jsonl")
            if row["status"] == "downloaded"
        }
        missing = sorted(required - present)
        report = {
            "status": "complete" if not missing else "quarantine_missing_official_images",
            "metadata_rows": api_rows,
            "images": successes,
            "bytes_downloaded": total_bytes,
            "missing_official_references": missing,
            "errors": errors,
            "seconds": round(time.monotonic() - started, 1),
            "profile": PROFILE,
        }
        write_report(DIRECTORY / "heldout_report.json", report)
        print(
            json.dumps(
                {**report, "errors": len(errors), "missing_official_references": len(missing)}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
