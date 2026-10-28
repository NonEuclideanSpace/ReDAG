from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time

import httpx
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.data.ingest import store_edit_mask
from agentic_image.data.io import append_jsonl, read_jsonl, write_jsonl, write_report
from fetch_viewer_heldout import download, request


def main():
    root = Path("data")
    store = ArtifactStore(root / "artifacts")
    raw = {row["sample_id"]: row for row in read_jsonl(root / "raw/all_saved.jsonl")}
    sources = list(read_jsonl(root / "viewer/images.jsonl"))[:32]
    page = json.loads((root / "viewer/api/magicbrush_train_0.json").read_text())
    originals = {
        f"magicbrush:{entry['row']['img_id']}:{entry['row']['turn_index']}": entry["row"]
        for entry in page["rows"]
    }
    prepared = []
    errors = []
    bytes_downloaded = 0
    started = time.monotonic()
    with httpx.Client(timeout=20, follow_redirects=True) as client:

        def prepare(source):
            row = dict(raw[source["sample_id"]])
            viewer = originals[row["sample_id"]]
            if viewer["instruction"] != row["instruction"]:
                raise ValueError("Source instruction differs from the original record")
            target = store.get_ref(row["target_image_id"])
            metadata = {
                "sample_id": row["sample_id"],
                "artifact_id": target.artifact_id,
                "value_id": target.value_id,
                "revision": row["license_record"]["revision"],
            }
            output = download(
                client, store, [("magicbrush", "train", viewer, "target_img", metadata)]
            )[0]
            mask_bytes = request(client, viewer["mask_img"]["src"]).content
            mask = Image.open(io.BytesIO(mask_bytes))
            mask.load()
            corrected = store_edit_mask(
                mask,
                store,
                {
                    "dataset": "osunlp/MagicBrush",
                    "revision": row["license_record"]["revision"],
                    "sample_id": row["sample_id"],
                    "dataset_profile": "official-viewer-reencoded-v1",
                },
            )
            row.update(
                source_image_id=source["artifact_id"],
                target_image_id=output["artifact_id"],
                edit_mask_id=corrected,
                split_id="pending-protocol-isolation",
            )
            row["registration"] = {
                "verified": False,
                "provenance": "viewer_reencoded_pair_requires_new_independent_review",
            }
            row["metadata"] = {
                **row["metadata"],
                "dataset_profile": "official-viewer-reencoded-v1",
                "mask_backfill": "official_alpha_decoded_at_original_size",
                "human_verified": False,
            }
            return row, output, len(mask_bytes)

        with ThreadPoolExecutor(max_workers=12) as executor:
            pending = {executor.submit(prepare, source): source["sample_id"] for source in sources}
            for future in as_completed(pending):
                try:
                    row, output, mask_bytes = future.result()
                    prepared.append(row)
                    bytes_downloaded += output["bytes_downloaded"] + mask_bytes
                    append_jsonl(root / "viewer/training_candidate_images.jsonl", [output])
                except Exception as error:
                    errors.append(
                        {
                            "sample_id": pending[future],
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "prepared": len(prepared),
                            "errors": len(errors),
                            "bytes_downloaded": bytes_downloaded,
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    write_jsonl(root / "review/viewer_candidates_pending.jsonl", prepared)
    report = {
        "status": "unverified_candidates",
        "prepared": len(prepared),
        "errors": errors,
        "download_bytes": bytes_downloaded,
        "official_mask_decode_complete": True,
        "paired_registration_verified": False,
        "human_audit_accepted": False,
    }
    write_report(root / "review/viewer_candidates_pending.report.json", report)


if __name__ == "__main__":
    main()
