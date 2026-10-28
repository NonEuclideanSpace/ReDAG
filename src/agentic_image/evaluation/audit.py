from collections import defaultdict

import numpy as np


def sample_blind_pairs(pairs, quotas=(34, 33, 33), seed=101):
    if len(quotas) != 3 or any(quota < 1 for quota in quotas):
        raise ValueError("Human audit uses three positive stratum quotas")
    rng = np.random.default_rng(seed)
    pairs = list(pairs)
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Audit pair identifiers must be unique")
    permutation = rng.permutation(len(pairs))
    ordered = sorted(
        permutation,
        key=lambda index: (float(pairs[index]["score_a"]) + float(pairs[index]["score_b"])) / 2,
    )
    strata = np.array_split(np.asarray(ordered, dtype=np.int64), 3)
    public, keys = [], []
    for stratum, (indices, quota) in enumerate(zip(strata, quotas)):
        if len(indices) < quota:
            raise ValueError(
                f"Audit stratum {stratum} has {len(indices)} pairs for a quota of {quota}"
            )
        sampled = rng.choice(indices, quota, replace=False)
        probability = quota / len(indices)
        for index in sampled:
            pair = pairs[int(index)]
            left_is_a = bool(rng.integers(2))
            public.append(
                {
                    "pair_id": pair["pair_id"],
                    "prompt": pair["prompt"],
                    "source_image": pair.get("source_image"),
                    "left_image": pair["image_a"] if left_is_a else pair["image_b"],
                    "right_image": pair["image_b"] if left_is_a else pair["image_a"],
                    "dimensions": [
                        "task_completion",
                        "source_preservation",
                        "identity",
                        "visual_quality",
                    ],
                }
            )
            keys.append(
                {
                    "pair_id": pair["pair_id"],
                    "family_id": pair.get("family_id", pair["pair_id"]),
                    "stratum": stratum,
                    "inclusion_probability": probability,
                    "left_is_a": left_is_a,
                    "method_a": pair.get("method_a", "B5"),
                    "method_b": pair["method_b"],
                }
            )
    rng.shuffle(public)
    return public, keys


def _unblind(value, left_is_a):
    if value not in {"left", "right", "tie", "unknown"}:
        raise ValueError("Audit answers must be left, right, tie or unknown")
    if value in {"tie", "unknown"}:
        return value
    return "win" if (value == "left") == left_is_a else "loss"


def cohen_kappa(first, second, weights=None):
    if len(first) != len(second) or not len(first):
        raise ValueError("Two reviewer label arrays must have the same nonzero length")
    categories = sorted(set(first) | set(second))
    weights = np.ones(len(first)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(first),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Reviewer weights must be finite and positive")
    total = weights.sum()
    observed = (
        sum(weight for left, right, weight in zip(first, second, weights) if left == right) / total
    )
    expected = sum(
        sum(weight for value, weight in zip(first, weights) if value == category)
        * sum(weight for value, weight in zip(second, weights) if value == category)
        / total**2
        for category in categories
    )
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def analyze_audit(keys, ratings, dimension="task_completion", iterations=10000, seed=101):
    keys = {row["pair_id"]: row for row in keys}
    ratings = {row["pair_id"]: row for row in ratings}
    if keys.keys() != ratings.keys():
        raise ValueError("Every selected audit pair must have a reviewer record")
    first, second, labels, weights, families, strata = [], [], [], [], [], []
    for pair_id, key in keys.items():
        rating = ratings[pair_id]
        left = rating["reviewer_1"][dimension]
        right = rating["reviewer_2"][dimension]
        if left == right:
            final = left
        else:
            final = rating.get("adjudicator", {}).get(dimension)
            if final is None:
                raise ValueError(f"Pair {pair_id} needs a third review for {dimension}")
        first.append(_unblind(left, key["left_is_a"]))
        second.append(_unblind(right, key["left_is_a"]))
        labels.append(_unblind(final, key["left_is_a"]))
        probability = float(key["inclusion_probability"])
        if not 0 < probability <= 1:
            raise ValueError("Audit inclusion probabilities must lie in (0, 1]")
        weights.append(1 / probability)
        families.append(key["family_id"])
        strata.append(key["stratum"])
    weights = np.asarray(weights)
    categories = ("win", "tie", "loss", "unknown")
    outcomes = np.asarray(
        [[float(label == category) for category in categories] for label in labels]
    )
    weighted = (outcomes * weights[:, None]).sum(0) / weights.sum()
    family_indices = defaultdict(list)
    for index, family in enumerate(families):
        family_indices[family].append(index)
    family_names = sorted(family_indices)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((iterations, len(categories)))
    for iteration in range(iterations):
        chosen = rng.integers(len(family_names), size=len(family_names))
        indices = np.concatenate([family_indices[family_names[index]] for index in chosen])
        bootstrap[iteration] = (outcomes[indices] * weights[indices, None]).sum(0) / weights[
            indices
        ].sum()
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    stratum_results = {}
    for stratum in sorted(set(strata)):
        indices = np.asarray([index for index, value in enumerate(strata) if value == stratum])
        stratum_results[str(stratum)] = dict(zip(categories, outcomes[indices].mean(0).tolist()))
    return {
        "dimension": dimension,
        "pairs": len(keys),
        "families": len(family_names),
        "weighted_rates": dict(zip(categories, weighted.tolist())),
        "confidence_intervals": {
            category: intervals[:, index].tolist() for index, category in enumerate(categories)
        },
        "strata": stratum_results,
        "cohen_kappa": cohen_kappa(first, second),
        "weighted_cohen_kappa": cohen_kappa(first, second, weights),
        "disagreement_rate": float(np.mean(np.asarray(first) != np.asarray(second))),
    }


def training_audit_sample(outputs, count=50, seed=101):
    outputs = list(outputs)
    if len(outputs) < count:
        raise ValueError("The training audit needs enough outputs for its fixed quota")
    high_count = count // 2
    order = sorted(
        range(len(outputs)), key=lambda index: float(outputs[index]["reward"]), reverse=True
    )
    high = order[:high_count]
    remaining = order[high_count:]
    random_indices = np.random.default_rng(seed).choice(
        remaining, count - high_count, replace=False
    )
    return [
        {**outputs[index], "audit_selection": "high_reward" if index in high else "random"}
        for index in high + random_indices.tolist()
    ]
