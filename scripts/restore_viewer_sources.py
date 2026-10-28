from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path

import httpx
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.data.families import perceptual_hash
from agentic_image.data.io import read_jsonl, write_jsonl


def main():
    root = Path("data")
    store = ArtifactStore(root / "artifacts")
    packets = list(read_jsonl(root / "viewer/priority_source_urls.jsonl"))
    restored = {}
    total_bytes = 0
    with httpx.Client(timeout=20, follow_redirects=True) as client:

        def restore(packet):
            existing = store.root / packet["relative_path"]
            if existing.is_file():
                try:
                    image = store.get_image(packet["artifact_id"])
                    return packet, image, 0
                except (OSError, ValueError):
                    existing.unlink()
            for attempt in range(2):
                try:
                    response = client.get(packet["url"])
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                    image.load()
                    reference = store.put_image(image, metadata=packet["metadata"])
                    if (
                        reference.value_id != packet["value_id"]
                        or reference.artifact_id != packet["artifact_id"]
                    ):
                        raise ValueError(
                            "Restored source differs from the registered viewer image identity"
                        )
                    return packet, image, len(response.content)
                except (httpx.HTTPError, OSError):
                    if attempt:
                        raise

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(restore, packet) for packet in packets]
            for future in as_completed(futures):
                packet, image, count = future.result()
                restored[packet["original_artifact_id"]] = perceptual_hash(image)
                total_bytes += count
                print(
                    json.dumps(
                        {"restored_sources": len(restored), "bytes_downloaded": total_bytes}
                    ),
                    flush=True,
                )
    records = list(read_jsonl(root / "viewer/images.jsonl"))
    for record in records:
        if record["original_artifact_id"] in restored:
            record["phash"] = restored[record["original_artifact_id"]]
    write_jsonl(root / "viewer/images.jsonl", records)


if __name__ == "__main__":
    main()
