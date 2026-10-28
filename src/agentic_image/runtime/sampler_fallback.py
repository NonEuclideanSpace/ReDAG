import json
from pathlib import Path

from agentic_image.learning.transaction import atomic_json
from .sampler import EVALUATION_SEEDS, frozen_json, policy_identity, severe_review_status


FALLBACK_SAMPLERS = {"ode50": (50, 0.0), "sde50_eta005": (50, 0.05), "sde50_eta015": (50, 0.15)}
FALLBACK_CANDIDATES = ("sde50_eta005", "sde50_eta015")


def reuse_link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except FileExistsError:
        if destination.resolve() != source:
            raise ValueError(f"Existing fallback link points elsewhere: {destination}")


def fallback_registration(config, tasks, output, reference_output, reference_reviews):
    output, reference = Path(output), Path(reference_output).resolve()
    if output.resolve() == reference:
        raise ValueError("Fallback must use an independent output directory")
    original = json.loads((reference / "registration.json").read_text())
    identities = {name: json.loads((reference / f"{name}_policy_identity.json").read_text())
                  for name in ("actor", "renderer")}
    if (len(tasks) != 64 or original["tasks"] != [task.model_dump(mode="json") for task in tasks]
            or original["eval_seeds"] != list(EVALUATION_SEEDS)
            or original["training_seed"] != config["seed"]
            or original["canvas"] != config["canvas"] or original["cfg_scale"] != 4
            or config["renderer"]["cfg_scale"] != 4):
        raise ValueError("Fallback source tasks, seeds or canvas differ from the original comparison")
    for key, value in original["models"].items():
        if config["models"].get(key) != value:
            raise ValueError(f"Fallback model registration changed: {key}")
    if (policy_identity(config["models"]["renderer_policy"]) != identities["renderer"]
            or policy_identity(config["models"]["actor_policy"], actor=True) != identities["actor"]):
        raise ValueError("Fallback policies differ from the original frozen tensor files")
    labels = {}
    for line in Path(reference_reviews).read_text().splitlines():
        row = json.loads(line)
        if row["image_id"] in labels:
            raise ValueError("Duplicate original quality review")
        labels[row["image_id"]] = row
    plans, references, selected_labels = [], [], []
    for index, task in enumerate(tasks):
        plan = json.loads((reference / "plans" / f"{index:03d}" / "frozen.json").read_text())
        if (plan["task"] != task.model_dump(mode="json") or plan.get("actual_sampled_plan") is not True
                or plan["actor_policy_identity"] != identities["actor"] or not Path(plan["bundle_path"]).is_file()):
            raise ValueError("Fallback controller PLAN is not the original actual frozen PLAN")
        plans.append({"index": index, "bundle_id": plan["bundle_id"], "bundle_path": plan["bundle_path"],
                      "source_image_id": task.user_request.source_image_id, "plan": plan["plan"]})
        for seed in EVALUATION_SEEDS:
            stem = f"{index:03d}_{seed}"
            image = json.loads((reference / "images" / "ode50" / f"{stem}.json").read_text())
            score = json.loads((reference / "scores" / "ode50" / f"{stem}.json").read_text())
            if (image["method"] != "ode50" or image["steps"] != 50 or image["eta"] != 0
                    or image["eval_seed"] != seed or image["task_id"] != task.task_id
                    or image["family_id"] != task.family_id or image["bundle_id"] != plan["bundle_id"]
                    or image["source_image_id"] != task.user_request.source_image_id
                    or image["renderer_policy_identity"] != identities["renderer"]
                    or not Path(image["image"]).is_file()
                    or score["image_id"] != image["image_id"] or score["image_value_id"] != image["image_value_id"]):
                raise ValueError("The reused ODE image/score identity changed")
            review = labels.get(image["image_id"])
            if severe_review_status(review, image) not in {"reviewed", "reviewed_unknown"}:
                raise ValueError("The reused ODE image lacks its actual quality review")
            references.append({"index": index, "eval_seed": seed, "image_id": image["image_id"],
                               "image_value_id": image["image_value_id"], "image": image["image"]})
            selected_labels.append(review)
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    for method in FALLBACK_CANDIDATES:
        (output / "images" / method).mkdir(parents=True, exist_ok=True)
        (output / "scores" / method).mkdir(parents=True, exist_ok=True)
    record = {**original, "version": "development64-frozen-controller-fallback-v1",
        "samplers": {key: list(value) for key, value in FALLBACK_SAMPLERS.items()},
        "expected_renders": 384, "new_renders": 256, "reused_ode_renders": 128,
        "reference_output": str(reference), "candidate_priority": list(FALLBACK_CANDIDATES),
        "reference_identity": references, "frozen_plan_identity": plans,
        "success_margin": 0.05, "defect_margin": 0.05, "protection_margin": 0.05,
        "acceptance_requires_actual_scores_and_blind_quality_review": True,
        "ode_rerender_allowed": False, "accepted": False,
        "declared_pilot_extension": "User-authorized eta0.05 candidate added after original eta0.3 failure; original quality margins unchanged",
        "deferred_candidates": {"sde28_eta005": [28, 0.05], "sde20_eta005": [20, 0.05]},
        "deferred_activation": "Only after sde50_eta005 passes; compare each shortened chain against that accepted same-eta50 reference before any final sampler lock"}
    frozen_json(output / "registration.json", record)
    frozen_json(output / "reference_reviews.json", selected_labels)
    for name, identity in identities.items():
        frozen_json(output / f"{name}_policy_identity.json", identity)
    frozen_json(output / "calibration.json", json.loads((reference / "calibration.json").read_text()))
    reuse_link(reference / "plans", output / "plans")
    reuse_link(reference / "images" / "ode50", output / "images" / "ode50")
    reuse_link(reference / "scores" / "ode50", output / "scores" / "ode50")
    return record


def fallback_reviews(output, review_paths):
    output = Path(output)
    labels = {row["image_id"]: row for row in json.loads((output / "reference_reviews.json").read_text())}
    for path in review_paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing = labels.get(row["image_id"])
            if existing is not None and existing != row:
                raise ValueError("Fallback quality reviews conflict for one exact image")
            labels[row["image_id"]] = row
    path = output / "quality_reviews_merged.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels.values()))
    temporary.replace(path)
    return path


def select_fallback(config, tasks, output, candidate, review_paths, iterations=10000):
    from .sampler import complete_comparison_rows, sampler_decision_with_protection

    if candidate not in FALLBACK_CANDIDATES:
        raise ValueError("Only a registered fallback candidate may be selected")
    output = Path(output)
    labels = fallback_reviews(output, review_paths)
    rows, report = complete_comparison_rows(tasks, output, labels, ("ode50", candidate), FALLBACK_SAMPLERS)
    target = output / f"sampler_selection.{candidate}.json"
    if report["status"] != "ready_for_decision":
        result = {**report, "candidate": candidate, "renderer_policy": config["models"]["renderer_policy"]}
        atomic_json(target, result)
        atomic_json(output / "sampler_selection.json", result)
        return result
    identity = json.loads((output / "renderer_policy_identity.json").read_text())
    if identity != policy_identity(config["models"]["renderer_policy"]):
        raise ValueError("Fallback policy changed after sampling")
    decision = sampler_decision_with_protection(rows, candidate, "ode50", iterations)
    steps, eta = FALLBACK_SAMPLERS[candidate]
    result = {"accepted": decision["accepted"], "status": decision["status"], "candidate": candidate,
        "reference": "ode50", "decision": decision, "decisions": {candidate: decision},
        "steps": steps, "eta": eta, "cfg_scale": 4.0, "families": 64,
        "eval_seeds": list(EVALUATION_SEEDS), "paired_records": 256,
        "renderer_policy": config["models"]["renderer_policy"], "renderer_policy_identity": identity,
        "calibration": str((output / "calibration.json").resolve()),
        "severe_reviews": str(labels.resolve()), "human_reviewed": False,
        "review_scope": "Actual AI blind quality review with worst-case reviewed unknowns and unchanged private noninferiority gates",
        "reference_reused": True, "ode_rerendered": 0,
        "declared_pilot_extension": candidate == "sde50_eta005",
        "final_sampler_lock_required_before_repair_sft_rl_and_main_test": True}
    atomic_json(target, result)
    atomic_json(output / "sampler_selection.json", result)
    return result


def prepare_fallback_review(tasks, output):
    import os
    import random
    from PIL import Image

    output = Path(output)
    slots = [(index, task, method, seed) for index, task in enumerate(tasks)
             for seed in EVALUATION_SEEDS for method in FALLBACK_CANDIDATES]
    if len(slots) != 256:
        raise ValueError("Fallback blind review requires all 64 fixed tasks")
    random.Random(176211).shuffle(slots)
    folder = output / "blinded_quality_review"
    folder.mkdir(parents=True, exist_ok=True)
    records, key = [], []
    for number, (index, task, method, seed) in enumerate(slots):
        review_id = f"fallback-quality-{number:04d}"
        key.append({"review_id": review_id, "method": method, "index": index, "eval_seed": seed})
        path = output / "images" / method / f"{index:03d}_{seed}.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        dest = folder / f"{review_id}.png"
        if dest.exists():
            if not os.path.samefile(row["image"], dest):
                raise ValueError("A fallback blind image changed after registration")
        else:
            os.link(row["image"], dest)
        preview = folder / f"{review_id}.jpg"
        if not preview.exists():
            with Image.open(dest) as image:
                image.save(preview, quality=85, optimize=True)
        records.append({"review_id": review_id, "image_id": row["image_id"],
            "candidate_image_value_id": row["image_value_id"], "candidate_path": str(dest.resolve()),
            "preview_path": str(preview.resolve()), "request": task.user_request.text,
            "source_image_id": task.user_request.source_image_id,
            "severe_defect": None, "known": False, "review_type": "ai", "human_reviewed": False,
            "review_evidence": None, "reviewed_image_paths": []})
    frozen_json(output / "quality_review_key.private.json", key)
    atomic_json(folder / "records.json", records)
    return {"status": "prepared" if len(records) == 256 else "partial", "images": len(records),
            "required": 256, "records": str((folder / "records.json").resolve()), "known_labels": 0,
            "reference_review_reused": 128}
