from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time

import httpx
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.data.ingest import store_edit_mask
from agentic_image.data.io import read_jsonl, write_report
from fetch_viewer_heldout import download, page, request


root = Path("data")
store = ArtifactStore(root / "artifacts")
candidates = list(read_jsonl(root / "review/viewer_candidates.jsonl"))
raw = {row["sample_id"]: row for row in read_jsonl(root / "raw/all_saved.jsonl")}
started = time.monotonic()
with httpx.Client(timeout=30, follow_redirects=True, limits=httpx.Limits(max_connections=16)) as client:
    _, _, items = page(client, "magicbrush", "train", 0)
    viewer = {f"magicbrush:{item['row']['img_id']}:{item['row']['turn_index']}": item["row"] for item in items}

    def restore(candidate):
        sample = candidate["sample_id"]
        original = raw[sample]
        item = viewer[sample]
        if item["instruction"] != original["instruction"] or candidate["instruction"] != original["instruction"]:
            raise ValueError("Request differs from the exact saved sample")
        store.get_image(candidate["source_image_id"])
        size = 0
        try:
            store.get_image(candidate["target_image_id"])
        except (FileNotFoundError, OSError, ValueError, KeyError):
            ref = store.get_ref(original["target_image_id"])
            metadata = {"sample_id": sample, "artifact_id": ref.artifact_id,
                        "value_id": ref.value_id, "revision": original["license_record"]["revision"]}
            restored = download(client, store, [("magicbrush", "train", item, "target_img", metadata)])[0]
            if restored["artifact_id"] != candidate["target_image_id"]:
                raise ValueError("Restored target differs from the reviewed pixel artifact")
            size += restored["bytes_downloaded"]
        try:
            store.get_array(candidate["edit_mask_id"])
        except (FileNotFoundError, OSError, ValueError, KeyError):
            payload = request(client, item["mask_img"]["src"]).content
            mask = Image.open(io.BytesIO(payload))
            mask.load()
            identifier = store_edit_mask(mask, store, {
                "dataset": "osunlp/MagicBrush", "revision": original["license_record"]["revision"],
                "sample_id": sample, "dataset_profile": "official-viewer-reencoded-v1",
            })
            if identifier != candidate["edit_mask_id"]:
                raise ValueError("Restored official alpha mask differs from the reviewed artifact")
            size += len(payload)
        return {"sample_id": sample, "source_image_id": candidate["source_image_id"],
                "target_image_id": candidate["target_image_id"], "edit_mask_id": candidate["edit_mask_id"],
                "download_bytes": size, "status": "exact_reviewed_artifacts_available"}

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=12) as executor:
        pending = {executor.submit(restore, row): row["sample_id"] for row in candidates}
        for future in as_completed(pending):
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({"sample_id": pending[future], "error": str(error)})
            print(json.dumps({"ready": len(completed), "failed": len(failures),
                              "bytes": sum(row["download_bytes"] for row in completed),
                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
write_report(root / "review/viewer_pair_assets.report.json", {
    "ready": len(completed), "failed": len(failures), "results": completed,
    "failures": failures, "download_bytes": sum(row["download_bytes"] for row in completed),
    "elapsed_seconds": time.monotonic() - started,
})
if failures:
    raise RuntimeError("Some reviewed pair assets could not be restored exactly")
