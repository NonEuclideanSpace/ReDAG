import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.checks.tools import VisualTools
from agentic_image.core.artifacts import ArtifactStore


torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/tools.yaml")
arguments = parser.parse_args()
config = yaml.safe_load(Path(arguments.config).read_text())
config["device"] = "cpu"
artifacts = ArtifactStore("data/deployment_probes")
tools = VisualTools(artifacts, config)
image = artifacts.put_image("third_party/GroundingDINO/.asset/cat_dog.jpeg")
detected = tools.detect(image.artifact_id, text="cat . dog .")
if len(detected["detections"]) < 2:
    raise RuntimeError("The official cat and dog example was not detected")
print("DETECTION_OK", detected["detections"], flush=True)
segmented = tools.segment(image.artifact_id, detected["detections"][0]["box"])
if not segmented["known"]:
    raise RuntimeError("SAM2 returned an empty object mask")
print("SEGMENTATION_OK", segmented["score"], flush=True)
features = tools.features(image.artifact_id)
if len(features["cls"]) != 768:
    raise RuntimeError("DINOv2 returned an unexpected feature dimension")
print("FEATURES_OK", len(features["cls"]), flush=True)
text_image = Image.new("RGB", (640, 140), "white")
font = ImageFont.truetype("DejaVuSans.ttf", 58)
ImageDraw.Draw(text_image).text((24, 28), "HELLO 123", font=font, fill="black")
text_ref = artifacts.put_image(text_image)
ocr = tools.ocr(text_ref.artifact_id)
if ocr["text"] != "HELLO 123" or not ocr["known"]:
    raise RuntimeError(f"Unexpected OCR output: {ocr['text']}")
print("OCR_OK", ocr["text"], flush=True)
Path("deployment/visual-tools-cpu.json").write_text(
    json.dumps(
        {
            "device": "cpu",
            "detection": detected,
            "segmentation": segmented,
            "feature_dimensions": len(features["cls"]),
            "ocr": ocr,
        },
        indent=2,
    )
    + "\n"
)
