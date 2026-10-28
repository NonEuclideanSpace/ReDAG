from collections import defaultdict

import numpy as np


def holm_adjust(p_values):
    keys = sorted(p_values, key=p_values.get)
    adjusted, previous = {}, 0.0
    for index, key in enumerate(keys):
        value = float(p_values[key])
        if not 0 <= value <= 1:
            raise ValueError("P-values must lie in [0, 1]")
        previous = max(previous, min(1.0, value * (len(keys) - index)))
        adjusted[key] = previous
    return adjusted


def paired_bootstrap(rows, method_a, method_b, metric="value", iterations=10000, seed=101):
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    by_method = {method_a: {}, method_b: {}}
    observation_weights = {}
    for row in rows:
        if row["method"] not in by_method:
            continue
        key = (
            row["training_seed"],
            row.get("family_id", row["task_id"]),
            row["task_id"],
            row.get("eval_seed", 101),
        )
        if key in by_method[row["method"]]:
            raise ValueError("Repeated method/task/seed observations are not independent samples")
        by_method[row["method"]][key] = float(row[metric])
        weight = float(row.get("task_weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Task weights must be finite and positive")
        if key in observation_weights and observation_weights[key] != weight:
            raise ValueError("Both methods must use identical task weights")
        observation_weights[key] = weight
    left, right = by_method[method_a], by_method[method_b]
    if not left or left.keys() != right.keys():
        raise ValueError(
            "Paired comparison requires the same complete task/family/seed set for both methods"
        )
    clustered = defaultdict(lambda: defaultdict(list))
    for key in left:
        training_seed, family, task, eval_seed = key
        clustered[training_seed][family].append((left[key], right[key], observation_weights[key]))
    training_seeds = sorted(clustered)
    families = sorted(clustered[training_seeds[0]])
    if any(set(clustered[value]) != set(families) for value in training_seeds):
        raise ValueError("Each training seed must be evaluated on the same family set")
    totals = np.zeros((len(training_seeds), len(families), 2), dtype=np.float64)
    counts = np.zeros((len(training_seeds), len(families)), dtype=np.float64)
    for i, training_seed in enumerate(training_seeds):
        for j, family in enumerate(families):
            values = np.asarray(clustered[training_seed][family], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Metrics must be finite; incomplete evaluations cannot be dropped")
            totals[i, j] = (values[:, :2] * values[:, 2:]).sum(0)
            counts[i, j] = values[:, 2].sum()
    estimates = totals.sum((0, 1)) / counts.sum()
    difference = float(estimates[0] - estimates[1])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations)
    for index in range(iterations):
        sampled_seeds = rng.integers(len(training_seeds), size=len(training_seeds))
        sampled_families = rng.integers(len(families), size=len(families))
        sampled_totals = totals[sampled_seeds[:, None], sampled_families[None, :]].sum((0, 1))
        sampled_count = counts[sampled_seeds[:, None], sampled_families[None, :]].sum()
        bootstrap[index] = (sampled_totals[0] - sampled_totals[1]) / sampled_count
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    p_value = float(
        (np.count_nonzero(np.abs(bootstrap - difference) >= abs(difference)) + 1) / (iterations + 1)
    )
    per_seed = {}
    for index, value in enumerate(training_seeds):
        means = totals[index].sum(0) / counts[index].sum()
        per_seed[str(value)] = {
            method_a: float(means[0]),
            method_b: float(means[1]),
            "difference": float(means[0] - means[1]),
        }
    return {
        "method_a": method_a,
        "method_b": method_b,
        "means": estimates.tolist(),
        "difference": difference,
        "confidence_interval": interval,
        "p_value": p_value,
        "training_seeds": len(training_seeds),
        "families": len(families),
        "observations": len(left),
        "per_seed": per_seed,
        "bootstrap_iterations": iterations,
    }


def compare_main_methods(
    rows, baseline="B5", alternatives=("B2", "B3", "B4"), metric="value", iterations=10000, seed=101
):
    results = {
        method: paired_bootstrap(rows, baseline, method, metric, iterations, seed)
        for method in alternatives
    }
    corrected = holm_adjust({method: result["p_value"] for method, result in results.items()})
    for method in results:
        results[method]["holm_p_value"] = corrected[method]
    return results


def sampler_quality_decision(
    rows,
    candidate,
    reference="ode50",
    success_margin=0.05,
    defect_margin=0.05,
    iterations=10000,
    seed=101,
    require_faster=False,
):
    success = paired_bootstrap(rows, candidate, reference, "success", iterations, seed)
    defects = paired_bootstrap(rows, candidate, reference, "severe_defect", iterations, seed)
    protection = paired_bootstrap(rows, candidate, reference, "protection", iterations, seed)
    acceptable = (
        success["confidence_interval"][0] >= -success_margin
        and defects["confidence_interval"][1] <= defect_margin
        and protection["confidence_interval"][0] >= -success_margin
    )
    timing = {}
    if require_faster:
        timing = paired_bootstrap(rows, candidate, reference, "seconds", iterations, seed)
        acceptable = acceptable and timing["confidence_interval"][1] < 0
    return {
        "candidate": candidate,
        "reference": reference,
        "accepted": bool(acceptable),
        "status": "accepted" if acceptable else "needs_review",
        "success": success,
        "severe_defects": defects,
        "protection": protection,
        "timing": timing,
    }
