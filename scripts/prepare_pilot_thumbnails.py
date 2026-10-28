from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from PIL import Image


def prepare(path, destination):
    try:
        with Image.open(path) as image:
            value = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
            value.save(destination)
        return True
    except (OSError, ValueError):
        return False


def main():
    root = Path("data")
    folder = root / "transfer/phash_thumbnails"
    folder.mkdir(exist_ok=True)
    paths = (root / "transfer/referenced_image_values.txt").read_text().splitlines()
    pending = {
        value: folder / Path(value).name
        for value in paths
        if not (folder / Path(value).name).is_file()
    }
    started = time.monotonic()
    report_at = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        while pending:
            available = [
                (value, target)
                for value, target in pending.items()
                if (root / "artifacts" / value).exists()
            ]
            futures = {
                value: executor.submit(prepare, root / "artifacts" / value, target)
                for value, target in available
            }
            for value, future in futures.items():
                if future.result():
                    del pending[value]
            elapsed = time.monotonic() - started
            if elapsed >= report_at:
                print(
                    json.dumps(
                        {
                            "prepared": len(paths) - len(pending),
                            "total": len(paths),
                            "seconds": round(elapsed, 1),
                        }
                    ),
                    flush=True,
                )
                report_at = elapsed + 30
            if pending:
                time.sleep(2)
    print(json.dumps({"prepared": len(paths), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
