import math
import unicodedata

import numpy as np

from agentic_image.core.geometry import box_iou
from agentic_image.core.types import BoundingBox

from .reward import AtomScore


def box_value(value):
    if value is None:
        return None
    if isinstance(value, BoundingBox):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        value = dict(zip(("x0", "y0", "x1", "y1"), value, strict=True))
    return BoundingBox.model_validate(value).model_dump()


class StructuredChecks:
    def __init__(self, tools):
        self.tools = tools

    def _detections(self, image_id, description):
        if not description:
            raise ValueError("A structured visual requirement needs an explicit object description")
        return self.tools.call("detect", image_id=image_id, text=description)

    def _single(self, image_id, description):
        result = self._detections(image_id, description)
        boxes = result.get("detections", [])
        if not result.get("known", False) or len(boxes) != 1:
            return None
        return BoundingBox.model_validate(box_value(boxes[0]["box"]))

    def evaluate(self, atom_id, image_id, kind, expected, reference_id=None):
        specification = expected if isinstance(expected, dict) else {"value": expected}
        route = specification.get("kind", kind)
        routes = {
            "count",
            "text",
            "identity",
            "left_of",
            "right_of",
            "above",
            "below",
            "inside",
            "contains",
            "layout",
            "geometry",
        }
        if route not in routes:
            return None
        if self.tools is None:
            raise RuntimeError(f"The {route} checker requires the configured visual tool service")
        if route == "geometry":
            route = specification.get("relation")
            if route not in routes - {"geometry", "count", "text", "identity"}:
                raise ValueError("A geometric requirement needs a supported relation")
        if route == "count":
            target = specification.get("count", specification.get("value"))
            if not isinstance(target, int) or isinstance(target, bool) or target < 0:
                raise ValueError("An exact count requirement needs a nonnegative integer count")
            result = self._detections(image_id, specification.get("object"))
            count = len(result.get("detections", []))
            known = bool(result.get("known", False)) and count >= target and count > 0
            return AtomScore(
                atom_id, math.exp(-abs(count - target)), known, exact_pass=count == target
            )
        if route == "text":
            target = specification.get("text", specification.get("value"))
            if not isinstance(target, str) or not target or not target.isascii():
                raise ValueError(
                    "The registered OCR route accepts explicitly specified short English or numeric text"
                )
            result = self.tools.call(
                "ocr", image_id=image_id, region=box_value(specification.get("region"))
            )
            actual = unicodedata.normalize("NFC", result.get("text", "")).strip()
            target = unicodedata.normalize("NFC", target).strip()
            if not specification.get("case_sensitive", True):
                actual, target = actual.casefold(), target.casefold()
            passed = actual == target
            return AtomScore(
                atom_id,
                float(passed),
                bool(result.get("known", False)),
                exact_pass=passed,
                evidence=tuple(result.get("artifact_ids", ())),
            )
        if route == "identity":
            reference_id = specification.get("reference_image_id", reference_id)
            threshold = specification.get("similarity_threshold")
            if not reference_id or threshold is None or not -1 <= float(threshold) <= 1:
                raise ValueError(
                    "Identity evidence requires a named reference and a calibrated similarity threshold"
                )
            candidate_box = box_value(specification.get("candidate_box"))
            if candidate_box is None:
                detected = self._single(image_id, specification.get("object"))
                if detected is None:
                    return AtomScore(atom_id, 0, False, exact_pass=False)
                candidate_box = detected.model_dump()
            reference_box = box_value(specification.get("reference_box"))
            left = self.tools.call("features", image_id=image_id, region=candidate_box)
            right = self.tools.call("features", image_id=reference_id, region=reference_box)
            first = np.asarray(left["cls"], dtype=np.float64)
            second = np.asarray(right["cls"], dtype=np.float64)
            norm = np.linalg.norm(first) * np.linalg.norm(second)
            if (
                first.shape != second.shape
                or not np.isfinite(first).all()
                or not np.isfinite(second).all()
                or norm <= 0
            ):
                raise FloatingPointError("Invalid identity feature vectors")
            similarity = float(np.clip(first @ second / norm, -1, 1))
            return AtomScore(
                atom_id,
                (1 + similarity) / 2,
                bool(left.get("known", False) and right.get("known", False)),
                exact_pass=similarity >= float(threshold),
            )
        first = self._single(image_id, specification.get("subject", specification.get("object")))
        if first is None:
            return AtomScore(atom_id, 0, False, exact_pass=False)
        if route == "layout":
            if specification.get("box_source") not in {"user", "annotation"}:
                raise ValueError(
                    "Layout rewards require a user-provided or independently annotated target box"
                )
            target = BoundingBox.model_validate(box_value(specification.get("target_box")))
            overlap = box_iou(first, target)
            return AtomScore(
                atom_id,
                overlap,
                True,
                exact_pass=overlap >= float(specification.get("iou_threshold", 0.5)),
            )
        second = self._single(image_id, specification.get("object"))
        if second is None:
            return AtomScore(atom_id, 0, False, exact_pass=False)
        if route in {"inside", "contains"}:
            inner, outer = (first, second) if route == "inside" else (second, first)
            intersection = max(0, min(inner.x1, outer.x1) - max(inner.x0, outer.x0)) * max(
                0, min(inner.y1, outer.y1) - max(inner.y0, outer.y0)
            )
            overlap = intersection / ((inner.x1 - inner.x0) * (inner.y1 - inner.y0))
            return AtomScore(atom_id, overlap, True, exact_pass=overlap >= 0.9)
        axis = "x" if route in {"left_of", "right_of"} else "y"
        first_center = (getattr(first, axis + "0") + getattr(first, axis + "1")) / 2
        second_center = (getattr(second, axis + "0") + getattr(second, axis + "1")) / 2
        distance = second_center - first_center
        if route in {"right_of", "below"}:
            distance = -distance
        return AtomScore(
            atom_id,
            1 / (1 + math.exp(-distance / 0.05)),
            abs(distance) > 0.02,
            exact_pass=distance > 0.02,
        )
