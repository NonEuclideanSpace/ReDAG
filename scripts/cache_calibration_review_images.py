import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image, ImageDraw


parser = argparse.ArgumentParser()
parser.add_argument("--batch", required=True)
parser.add_argument("--workers", type=int, default=12)
arguments = parser.parse_args()
batch = Path(arguments.batch)
root = batch.parent
images_dir, sheets_dir = root / "local_images", root / "contact_sheets"
images_dir.mkdir(exist_ok=True)
sheets_dir.mkdir(exist_ok=True)
rows = sorted((json.loads(line) for line in batch.read_text().splitlines()), key=lambda row: row["calibration_index"])
started = time.monotonic()


def fetch(row, image):
    path = images_dir / f"{row['calibration_index']:03d}_{image['candidate_role']}.jpg"
    if path.exists():
        with Image.open(path) as decoded:
            decoded.load()
        return path, 0
    error = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(image["review_image_url"], timeout=25) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as decoded:
                decoded.load()
            path.write_bytes(payload)
            return path, len(payload)
        except Exception as caught:
            error = caught
            if not attempt:
                time.sleep(1)
    raise error


def sheets():
    created = []
    for offset in range(0, len(rows), 4):
        group = rows[offset:offset + 4]
        destination = sheets_dir / f"{group[0]['calibration_index']:03d}-{group[-1]['calibration_index']:03d}.jpg"
        if destination.exists():
            continue
        if not all((images_dir / f"{row['calibration_index']:03d}_{role}.jpg").exists() for row in group for role in ("source", "target")):
            continue
        sheet = Image.new("RGB", (1024, 552 * len(group)), "white")
        draw = ImageDraw.Draw(sheet)
        for y, row in enumerate(group):
            for x, role in enumerate(("source", "target")):
                with Image.open(images_dir / f"{row['calibration_index']:03d}_{role}.jpg") as image:
                    image = image.convert("RGB")
                    image.thumbnail((512, 512))
                    sheet.paste(image, (512 * x + (512 - image.width) // 2, 552 * y + 40 + (512 - image.height) // 2))
                draw.text((512 * x + 12, 552 * y + 12), f"{row['calibration_index']:03d} | {role} | {row['sample_id']}", fill="black")
        sheet.save(destination, quality=95)
        created.append(str(destination.resolve()))
    return created


count, total_bytes, failures = 0, 0, []
with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
    futures = {executor.submit(fetch, row, image): (row["calibration_index"], image["candidate_role"])
               for row in rows for image in row["images"]}
    for future in as_completed(futures):
        try:
            _, size = future.result()
            count += 1
            total_bytes += size
        except Exception as error:
            failures.append({"image": futures[future], "error": str(error)})
        created = sheets()
        if created or count % 10 == 0 or failures:
            print(json.dumps({"images": count, "bytes": total_bytes, "seconds": round(time.monotonic() - started, 2),
                              "new_contact_sheets": created, "failures": len(failures)}), flush=True)
report = {"images": count, "expected_images": len(rows) * 2, "download_bytes": total_bytes,
          "elapsed_seconds": time.monotonic() - started, "failures": failures,
          "actual_ai_labels_created": 0, "human_reviewed": False}
(root / (batch.stem + ".local_images.report.json")).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
