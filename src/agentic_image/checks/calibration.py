import math
from collections import defaultdict

import numpy as np


def wilson_lower(successes, count, z=1.959963984540054):
    if count == 0:
        return 0.0
    probability = successes / count
    return (
        probability
        + z * z / (2 * count)
        - z * math.sqrt(probability * (1 - probability) / count + z * z / (4 * count * count))
    ) / (1 + z * z / count)


def score_calibration(records, threshold):
    probabilities = np.asarray([row["value"] for row in records])
    labels = np.asarray([row["label"] for row in records], dtype=bool)
    known = np.asarray([row["known"] for row in records], dtype=bool)
    exact = np.asarray([row.get("exact_pass") is not False for row in records], dtype=bool)
    positive = known & (probabilities >= threshold) & exact
    tp = int((positive & labels).sum())
    count = int(positive.sum())
    ece = 0.0
    for start in np.arange(0, 1, 0.1):
        selected = known & (probabilities >= start) & (probabilities < start + 0.1 + 1e-12)
        if selected.any():
            ece += float(
                selected.mean() * abs(probabilities[selected].mean() - labels[selected].mean())
            )
    return {
        "precision": tp / count if count else 0.0,
        "recall": tp / max(1, int(labels.sum())),
        "precision_lower": wilson_lower(tp, count),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": ece,
        "unknown_rate": float(1 - known.mean()),
        "count": len(records),
    }


def calibrate(records, review_mode="human"):
    if review_mode not in {"human", "ai_exploratory"}:
        raise ValueError("Unknown calibration review mode")
    grouped = defaultdict(lambda: defaultdict(list))
    family_partitions = {}
    for row in records:
        if row["partition"] not in {"selection", "confirmation"}:
            raise ValueError("Calibration partitions must be selection or confirmation")
        prior = family_partitions.setdefault(row["family_id"], row["partition"])
        if prior != row["partition"]:
            raise ValueError("Calibration family leaked across partitions")
        if not row.get("adjudicated", False) or type(row.get("label")) is not bool:
            raise ValueError("Calibration requires an adjudicated boolean label")
        if review_mode == "ai_exploratory":
            evidence = row.get("review_evidence")
            if (
                row.get("review_type") != "ai"
                or row.get("human_reviewed") is not False
                or not isinstance(evidence, dict)
                or any(not evidence.get(key) for key in ("reviewer", "observation", "record_path"))
            ):
                raise ValueError("AI calibration requires explicit visual-review provenance")
        elif row.get("review_type") == "ai" or row.get("human_reviewed") is False:
            raise ValueError("AI labels cannot enter the human calibration profile")
        grouped[row["kind"]][row["partition"]].append(row)
    counts = {
        partition: sum(value == partition for value in family_partitions.values())
        for partition in ("selection", "confirmation")
    }
    if counts["selection"] < 400 or counts["confirmation"] < 200:
        raise ValueError(
            "Judge calibration requires at least 400 selection and 200 confirmation families"
        )
    result = {}
    for kind, groups in grouped.items():
        selection = groups["selection"]
        confirmation = groups["confirmation"]
        selected = None
        if len(selection) + len(confirmation) >= 100 and selection and confirmation:
            for threshold in np.arange(0.5, 0.951, 0.05):
                report = score_calibration(selection, float(threshold))
                if report["precision_lower"] >= 0.85:
                    selected = round(float(threshold), 2)
                    break
        if selected is None:
            result[kind] = {
                "threshold": 1.0,
                "confirmed": False,
                "reason": "insufficient_precision_or_labels",
            }
            continue
        report = score_calibration(confirmation, selected)
        result[kind] = {
            "threshold": selected,
            "confirmed": report["precision_lower"] >= 0.85,
            "selection": score_calibration(selection, selected),
            "confirmation": report,
            "family_counts": counts,
        }
    return result
