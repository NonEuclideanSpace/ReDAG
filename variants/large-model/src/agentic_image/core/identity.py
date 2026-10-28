from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import box_iou
from .types import BoundingBox


@dataclass(frozen=True)
class IdentityItem:
    item_id: str
    category: str
    feature: np.ndarray
    box: BoundingBox | None = None


@dataclass(frozen=True)
class IdentityMatch:
    entity_id: str
    detection_id: str | None
    cost: float
    identity_unknown: bool


def match_entities(
    entities: list[IdentityItem],
    detections: list[IdentityItem],
    moved_entity_ids: set[str] | None = None,
    dummy_cost: float = 0.7,
    ambiguity_margin: float = 0.1,
) -> list[IdentityMatch]:
    if not entities:
        return []
    moved_entity_ids = moved_entity_ids or set()
    costs = np.full((len(entities), len(detections) + len(entities)), dummy_cost, dtype=np.float64)
    for row, entity in enumerate(entities):
        source = np.asarray(entity.feature, dtype=np.float64).reshape(-1).copy()
        source /= max(np.linalg.norm(source), 1e-12)
        for column, detection in enumerate(detections):
            if entity.category != detection.category:
                costs[row, column] = 1e6
                continue
            target = np.asarray(detection.feature, dtype=np.float64).reshape(-1).copy()
            if (
                target.shape != source.shape
                or not np.isfinite(source).all()
                or not np.isfinite(target).all()
            ):
                raise ValueError("Identity features must be finite with matching dimensions")
            target /= max(np.linalg.norm(target), 1e-12)
            appearance_cost = (1 - np.clip(source @ target, -1, 1)) / 2
            if (
                entity.box is not None
                and detection.box is not None
                and entity.item_id not in moved_entity_ids
            ):
                costs[row, column] = (
                    0.5 * (1 - box_iou(entity.box, detection.box)) + 0.5 * appearance_cost
                )
            else:
                costs[row, column] = appearance_cost
    rows, columns = linear_sum_assignment(costs)
    result = []
    for row, column in zip(rows, columns, strict=True):
        row_values = np.sort(np.concatenate((costs[row, : len(detections)], [dummy_cost])))
        ambiguous = len(row_values) >= 2 and row_values[1] - row_values[0] < ambiguity_margin
        unknown = column >= len(detections) or costs[row, column] >= dummy_cost or ambiguous
        result.append(
            IdentityMatch(
                entities[row].item_id,
                None if column >= len(detections) else detections[column].item_id,
                float(costs[row, column]),
                bool(unknown),
            )
        )
    return result
