import os
import subprocess
import threading
from pathlib import Path

import httpx
import numpy as np
from PIL import Image


class ToolClient:
    def __init__(self, url, timeout=300):
        self.client = httpx.Client(base_url=url, timeout=timeout, trust_env=False)

    def call(self, operation, **arguments):
        response = self.client.post(f"/{operation}", json=arguments)
        response.raise_for_status()
        return response.json()


class VisualTools:
    def __init__(self, artifacts, config):
        self.artifacts = artifacts
        self.config = config
        self.device = config.get("device", "cuda:3")
        self.active = None
        self.model = None
        self.lock = threading.RLock()

    def _load(self, kind):
        import gc
        import torch

        if self.active == kind:
            return self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if kind == "detect":
            from groundingdino.util.inference import load_model

            self.model = load_model(
                self.config["grounding_config"], self.config["grounding_weights"], self.device
            )
        elif kind == "segment":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(
                self.config["sam_config"],
                self.config["sam_weights"],
                device=self.device,
                apply_postprocessing=True,
            )
            self.model = SAM2ImagePredictor(
                model, mask_threshold=0, max_hole_area=0, max_sprinkle_area=0
            )
        elif kind == "features":
            os.environ["XFORMERS_DISABLED"] = "1"
            self.model = torch.hub.load(
                self.config["dino_repository"], "dinov2_vitb14", source="local", pretrained=False
            )
            weights = torch.load(self.config["dino_weights"], map_location="cpu", weights_only=True)
            self.model.load_state_dict(weights, strict=True)
            self.model = self.model.float().to(self.device).eval()
            self.model.requires_grad_(False)
        else:
            raise ValueError(f"Unknown tool: {kind}")
        self.active = kind
        return self.model

    def detect(self, image_id, text=None, entity_id=None, region=None):
        import torch
        import groundingdino.datasets.transforms as transforms
        from groundingdino.util.inference import predict
        from torchvision.ops import nms

        if not text:
            raise ValueError("Detection needs an explicit public entity description")
        with self.lock, torch.no_grad():
            model = self._load("detect")
            image = self.artifacts.get_image(image_id)
            transform = transforms.Compose(
                [
                    transforms.RandomResize([800], max_size=1333),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            tensor, _ = transform(image, None)
            boxes, scores, labels = predict(
                model=model,
                image=tensor.float(),
                caption=text,
                box_threshold=0.35,
                text_threshold=0.25,
                device=self.device,
            )
            xyxy = torch.cat(
                (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1
            ).clamp(0, 1)
            keep = []
            for label in sorted(set(labels)):
                indices = torch.tensor([i for i, item in enumerate(labels) if item == label])
                keep.extend(indices[nms(xyxy[indices], scores[indices], 0.5)].tolist())
            detections = [
                {"box": xyxy[i].tolist(), "score": float(scores[i]), "label": labels[i]}
                for i in keep
            ]
            return {"known": bool(detections), "detections": detections, "image_id": image_id}

    def segment(self, image_id, region, entity_id=None):
        import torch

        box = (
            [region[key] for key in ("x0", "y0", "x1", "y1")]
            if isinstance(region, dict)
            else region
        )
        with self.lock, torch.no_grad():
            predictor = self._load("segment")
            image = self.artifacts.get_image(image_id)
            predictor.set_image(np.array(image, copy=True))
            pixels = np.asarray(box, dtype=np.float32) * np.asarray([image.width, image.height] * 2)
            masks, scores, _ = predictor.predict(box=pixels, multimask_output=False)
            mask = masks[0].astype(np.float32)
            reference = self.artifacts.put_array(
                mask,
                frame=self.artifacts.get_ref(image_id).frame,
                producer_id=image_id,
                metadata={"entity_id": entity_id, "kind": "binary"},
            )
            return {
                "known": bool(mask.any()),
                "artifact_ids": [reference.artifact_id],
                "score": float(scores[0]),
            }

    def _dino_input(self, image):
        import torch
        from agentic_image.core.geometry import make_frame

        frame = make_frame((image.height, image.width), (224, 224), "dino")
        resized_h, resized_w = frame.resized_hw
        canvas = Image.new("RGB", (224, 224), (128, 128, 128))
        canvas.paste(
            image.resize((resized_w, resized_h), Image.Resampling.BICUBIC), frame.padding_xy
        )
        tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float().div(255)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return ((tensor - mean) / std).unsqueeze(0).to(self.device), frame

    def features(self, image_id, region=None, entity_id=None):
        import torch

        with self.lock, torch.no_grad():
            model = self._load("features")
            image = self.artifacts.get_image(image_id)
            if region:
                box = (
                    [region[key] for key in ("x0", "y0", "x1", "y1")]
                    if isinstance(region, dict)
                    else region
                )
                image = image.crop(
                    tuple(value * size for value, size in zip(box, [image.width, image.height] * 2))
                )
            tensor, frame = self._dino_input(image)
            output = model.forward_features(tensor)
            cls = output["x_norm_clstoken"].float()
            norm = cls.norm(dim=-1, keepdim=True)
            if not torch.isfinite(cls).all() or (norm <= 0).any():
                raise FloatingPointError("Invalid DINO appearance features")
            patches = output["x_norm_patchtokens"].float()
            return {
                "known": True,
                "cls": (cls / norm)[0].cpu().tolist(),
                "patches": patches[0].cpu().tolist(),
                "frame": frame.model_dump(),
            }

    def preservation(self, source_image_id, image_id, mask_id):
        from agentic_image.core.geometry import Frame, transform_mask

        left = self.features(source_image_id)
        right = self.features(image_id)
        frame = Frame.model_validate(left["frame"])
        mask = transform_mask(self.artifacts.get_array(mask_id), frame, "soft")
        coverage = mask.reshape(16, 14, 16, 14).mean((1, 3)).reshape(-1)
        return {
            "source_features": left["patches"],
            "output_features": right["patches"],
            "patch_coverage": coverage.tolist(),
        }

    def ocr(self, image_id, entity_id=None, region=None):
        import csv
        import io
        import tempfile
        import unicodedata

        image = self.artifacts.get_image(image_id)
        artifact_ids = []
        if region is not None:
            box = (
                [region[key] for key in ("x0", "y0", "x1", "y1")]
                if isinstance(region, dict)
                else region
            )
            image = image.crop(
                tuple(value * size for value, size in zip(box, [image.width, image.height] * 2))
            )
            crop = self.artifacts.put_image(
                image, producer_id=image_id, metadata={"role": "ocr_crop", "box": box}
            )
            artifact_ids.append(crop.artifact_id)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            image.save(path)
            command = [
                self.config.get("tesseract", "tesseract"),
                str(path),
                "stdout",
                "-l",
                "eng",
                "--oem",
                "1",
                "--psm",
                "11",
                "tsv",
            ]
            result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
        words = []
        for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
            if row["text"].strip():
                words.append(
                    {
                        "text": unicodedata.normalize("NFC", row["text"]),
                        "confidence": float(row["conf"]),
                        "box": [int(row[key]) for key in ("left", "top", "width", "height")],
                    }
                )
        return {
            "known": bool(words) and min(word["confidence"] for word in words) >= 80,
            "text": " ".join(word["text"] for word in words),
            "words": words,
            "artifact_ids": artifact_ids,
        }


def serve_tools(artifacts, config, host="127.0.0.1", port=8091):
    import uvicorn
    from fastapi import FastAPI, HTTPException

    worker = VisualTools(artifacts, config)
    app = FastAPI()

    @app.post("/{operation}")
    def call(operation: str, arguments: dict):
        if operation not in {"detect", "segment", "features", "preservation", "ocr"}:
            raise HTTPException(404, "Unknown visual tool")
        return getattr(worker, operation)(**arguments)

    uvicorn.run(app, host=host, port=port)
