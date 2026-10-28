import argparse
import json
from pathlib import Path

from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import normalize_image


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--image-root", required=True)
arguments = parser.parse_args()
store = ArtifactStore("data/artifacts")
count = 0
for row in map(json.loads, Path(arguments.manifest).read_text().splitlines()):
    for image in row["images"]:
        reference = store.get_ref(image["artifact_id"])
        path = Path(arguments.image_root) / f"{row['calibration_index']:03d}_{image['candidate_role']}.jpg"
        restored = store.put_image(normalize_image(Image.open(path)), metadata=reference.metadata)
        if restored.artifact_id != reference.artifact_id or restored.value_id != image["value_id"]:
            raise ValueError("Restored real image does not match the frozen artifact identity")
        count += 1
print(json.dumps({"restored_exact_artifacts": count}), flush=True)
