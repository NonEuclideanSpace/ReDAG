import json
import time
from pathlib import Path

from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json

from .rollout import RolloutCollector, read_tasks


def sample_comparison(components, tasks_path, output_directory, limit=64, fallbacks=False):
    tasks = read_tasks(tasks_path)[:limit]
    if any(task.split_id == "train" for task in tasks):
        raise ValueError("Sampler selection requires held-out development tasks")
    if len(tasks) < limit:
        raise ValueError("Provide the complete registered development task count")
    configurations = [
        ("ode50", 50, 0.0),
        ("sde50", 50, 0.3),
        ("sde28", 28, 0.3),
        ("sde20", 20, 0.3),
    ]
    if fallbacks:
        configurations = [("sde50_eta015", 50, 0.15), ("sde80_eta015", 80, 0.15)]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    collector = RolloutCollector(components)
    rows = []
    for index, task in enumerate(tasks):
        environment = collector.environment(task, f"sampler:{index}")
        for attempt in range(2):
            snapshot = collector.snapshot(environment)
            sample = components.actor.sample(
                snapshot, stream_seed(101, index, attempt, "plan"), "sampler-teacher"
            )
            environment.step(sample.text)
            if environment.state.plan:
                break
        if not environment.state.plan:
            raise RuntimeError(
                f"Task {task.task_id} has no executable public plan; repair this input before the comparison"
            )
        plan = environment.state.plan
        bundle = components.compiler.compile(plan, task.user_request.source_image_id)
        atomic_json(output / f"plan_{index:03d}.json", plan.model_dump(mode="json"))
        for seed in (101, 211):
            for method, steps, eta in configurations:
                directory = output / method
                directory.mkdir(exist_ok=True)
                path = directory / f"{index:03d}_{seed}.png"
                record_path = path.with_suffix(".json")
                if record_path.is_file():
                    rows.append(json.loads(record_path.read_text()))
                    continue
                started = time.monotonic()
                if eta == 0:
                    latent = components.renderer.rollout_ode(bundle, seed, steps)
                else:
                    trace = components.renderer.rollout(
                        bundle, seed, steps=steps, eta=eta, behavior_version="sampler-comparison"
                    )
                    latent = trace.final_latent
                    trace.save(path.with_suffix(".pt"))
                image = components.encoders.decode_image(latent, bundle.canvas_hw)
                image.save(path)
                reference = components.artifacts.put_image(
                    image, producer_id=f"sampler:{method}:{index}:{seed}"
                )
                reward = components.evaluator.evaluate(
                    task, reference.artifact_id, renders=1, inspects=0, actions=2, invalids=0
                )
                row = {
                    "method": method,
                    "training_seed": components.config["seed"],
                    "task_id": task.task_id,
                    "family_id": task.family_id,
                    "eval_seed": seed,
                    "success": float(reward.success),
                    "protection": reward.preservation,
                    "preservation_status": reward.preservation_status,
                    "dense": reward.dense,
                    "severe_defect": None,
                    "seconds": time.monotonic() - started,
                    "image": str(path.resolve()),
                    "steps": steps,
                    "eta": eta,
                    "cfg_scale": bundle.cfg_scale,
                    "bundle_id": bundle.bundle_id,
                }
                atomic_json(record_path, row)
                rows.append(row)
    with (output / "comparison.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return {
        "renders": len(rows),
        "records": str((output / "comparison.jsonl").resolve()),
        "remaining": "Fill severe_defect with blinded image-quality judgments before select-sampler",
    }


SAMPLERS = {"ode50": (50, 0.0), "sde50": (50, 0.3), "sde28": (28, 0.3), "sde20": (20, 0.3)}
EVALUATION_SEEDS = (101, 211)


def frozen_json(path, value):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"Frozen registration changed: {path}")
        else:
            atomic_json(path, value)
    return value


def comparison_tasks(tasks_path, reservation_path, task_prefix="development64:", expected_split="development"):
    tasks = read_tasks(tasks_path)
    reserved = [json.loads(line) for line in Path(reservation_path).read_text().splitlines() if line.strip()]
    if len(tasks) != 64 or len(reserved) != 64 or len({task.family_id for task in tasks}) != 64:
        raise ValueError("Sampler comparison requires the fixed 64 development families")
    for index, (task, row) in enumerate(zip(tasks, reserved)):
        if (task.split_id != expected_split or row["split_id"] != expected_split
                or row["evaluation_index"] != index or task.task_id != task_prefix + row["sample_id"]
                or task.family_id != row["family_id"] or task.user_request.text != row["instruction"]):
            raise ValueError(f"Development registration mismatch at {index}")
    return tasks


def comparison_registration(config, tasks, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    for method in SAMPLERS:
        (output / "scores" / method).mkdir(parents=True, exist_ok=True)
    if config["renderer"]["cfg_scale"] != 4 or tuple(config["canvas"]) != (512, 512):
        raise ValueError("The fixed comparison requires a 512 canvas and CFG 4")
    models = config["models"]
    if not models.get("actor_policy") or not models.get("renderer_policy"):
        raise ValueError("Comparison requires explicit trained actor and renderer policies")
    return frozen_json(output / "registration.json", {
        "version": "development64-frozen-controller-sampler-v1",
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "samplers": {key: list(value) for key, value in SAMPLERS.items()},
        "eval_seeds": list(EVALUATION_SEEDS), "training_seed": config["seed"],
        "expected_renders": 512, "canvas": config["canvas"], "cfg_scale": 4.0,
        "models": {key: models.get(key) for key in (
            "actor", "actor_revision", "actor_policy", "renderer", "renderer_revision", "renderer_policy",
            "renderer_profile", "judge", "judge_revision")},
        "actor_sampling": {key: config["actor"][key] for key in ("temperature", "max_new_tokens", "max_input_tokens")},
        "controller_seed": 101, "controller_attempts": 2,
        "private_evaluation_separate": True, "human_reviewed": False,
    })


def shard_indices(start, count):
    if count != 8 or start not in range(0, 64, 8):
        raise ValueError("Use one registered eight-task shard")
    return range(start, start + count)


def shard_name(start, count):
    return f"{start:03d}_{start + count - 1:03d}"


def policy_identity(path, actor=False):
    if actor:
        from agentic_image.actor.config import adapter_directory
        path = adapter_directory(path) / "adapter_model.safetensors"
    path = Path(path)
    if path.is_dir():
        path = path / "policy.pt"
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def require_training(exit_path, metrics_path, updates):
    result = json.loads(Path(exit_path).read_text())
    code = result.get("returncode") if isinstance(result, dict) else result
    metrics = [json.loads(line) for line in Path(metrics_path).read_text().splitlines() if line.strip()]
    if type(code) is not int or code != 0 or not metrics or metrics[-1]["step"] != updates:
        raise ValueError(f"Training has not committed {updates} updates with exit zero")


def freeze_comparison_plans(components, tasks, output, start=0, count=8):
    import torch
    from agentic_image.actor.records import ActionSample
    from agentic_image.core.runtime import EpisodeEnvironment
    from agentic_image.data.paired import actor_snapshot, public_environment
    from agentic_image.learning.transaction import atomic_torch_save

    output = Path(output)
    actor_identity = policy_identity(components.config["models"]["actor_policy"], actor=True)
    frozen_json(output / "actor_policy_identity.json", actor_identity)
    (output / "shards").mkdir(parents=True, exist_ok=True)
    tokenizer = components.actor.processor.tokenizer
    records, errors = [], []
    for index in shard_indices(start, count):
        task = tasks[index]
        folder = output / "plans" / f"{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        ready = folder / "frozen.json"
        if ready.is_file():
            record = json.loads(ready.read_text())
            if (record["task"] != task.model_dump(mode="json") or not Path(record["bundle_path"]).is_file()
                    or record.get("actual_sampled_plan") is not True
                    or record.get("actor_policy_identity") != actor_identity):
                raise ValueError("A frozen plan changed or lost its genuine sampled bundle")
            records.append(record)
            continue
        progress = folder / "state.json"
        if progress.is_file():
            saved = json.loads(progress.read_text())
            environment = EpisodeEnvironment.restore(
                saved["environment"], components.artifacts, components.compiler, None, components.observer,
                components.compiler.load,
                token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                max_input_tokens=saved["view_token_budget"],
            )
            attempt = saved["attempts"]
        else:
            environment = public_environment(components, task, f"sampler:{index}")
            attempt = 0
            atomic_json(progress, {"environment": environment.snapshot(), "attempts": 0,
                                  "view_token_budget": environment.view_builder.max_input_tokens})
        while environment.state.plan is None and attempt < 2 and not environment.state.done:
            sample_path = folder / f"action_{attempt:02d}.pt"
            seed = stream_seed(101, index, attempt, "plan")
            if sample_path.is_file():
                sample = ActionSample.from_state_dict(torch.load(sample_path, map_location="cpu", weights_only=True))
                if sample.seed != seed or sample.snapshot.metadata["revision_id"] != environment.state.manifest.revision_id:
                    raise ValueError("Interrupted PLAN sample does not match its saved state")
            else:
                snapshot = actor_snapshot(components, environment)
                snapshot.save(folder / f"snapshot_{attempt:02d}.pt")
                sample = components.actor.sample(snapshot, seed, "sampler-frozen-controller")
                atomic_torch_save(sample_path, sample.state_dict())
            result = environment.step(sample.text, seed=seed)
            atomic_json(folder / f"action_{attempt:02d}.json", {
                "seed": seed, "text": sample.text, "finish_reason": sample.finish_reason,
                "result": result.model_dump(mode="json"), "snapshot": str((folder / f"snapshot_{attempt:02d}.pt").resolve()),
            })
            attempt += 1
            atomic_json(progress, {"environment": environment.snapshot(), "attempts": attempt,
                                  "view_token_budget": environment.view_builder.max_input_tokens})
        if environment.state.plan is None:
            failure = {"index": index, "task_id": task.task_id, "status": "controller_plan_failed", "attempts": attempt}
            atomic_json(folder / "failure.json", failure)
            errors.append(failure)
            print(json.dumps(failure), flush=True)
            continue
        plan = environment.state.plan
        bundle = components.compiler.compile(plan, task.user_request.source_image_id)
        bundle_path = folder / "condition.pt"
        bundle.save(bundle_path)
        record = {
            "index": index, "task": task.model_dump(mode="json"), "plan": plan.model_dump(mode="json"),
            "bundle_id": bundle.bundle_id, "bundle_path": str(bundle_path.resolve()),
            "source_image_id": task.user_request.source_image_id, "attempts": attempt,
            "actor_policy_identity": policy_identity(components.config["models"]["actor_policy"], actor=True),
            "controller_seed": 101, "actual_sampled_plan": True, "target_access": False,
            "snapshot_directory": str(folder.resolve()), "status": "frozen",
        }
        atomic_json(ready, record)
        records.append(record)
        print(json.dumps({"stage": "freeze", "index": index, "bundle_id": bundle.bundle_id, "attempts": attempt}), flush=True)
    report = {"stage": "freeze", "status": "completed" if not errors else "blocked",
              "start": start, "count": count, "completed": len(records), "errors": errors}
    atomic_json(output / "shards" / f"freeze_{shard_name(start, count)}.json", report)
    return report


def render_comparison_shard(components, tasks, output, start=0, count=8, methods=None, samplers=None):
    import torch
    from agentic_image.render.types import ConditionBundle

    samplers = SAMPLERS if samplers is None else samplers
    methods = comparison_methods(methods, samplers)
    suffix = comparison_stage_suffix(methods, samplers)
    output = Path(output)
    identity = policy_identity(components.config["models"]["renderer_policy"])
    frozen_json(output / "renderer_policy_identity.json", identity)
    supplied = torch.load(identity["path"], map_location="cpu", weights_only=True)
    loaded = components.renderer.policy_state()
    if supplied.keys() != loaded.keys() or any(not torch.equal(supplied[key], loaded[key]) for key in supplied):
        raise ValueError("Renderer policy differs from the registered trained tensors")
    if not any(torch.count_nonzero(value) for key, value in supplied.items() if "lora_B" in key):
        raise ValueError("Renderer policy has no trained LoRA contribution")
    del supplied, loaded
    rows = []
    for index in shard_indices(start, count):
        task = tasks[index]
        record = json.loads((output / "plans" / f"{index:03d}" / "frozen.json").read_text())
        if record["task"] != task.model_dump(mode="json") or not record["actual_sampled_plan"]:
            raise ValueError("Frozen controller task changed")
        bundle = ConditionBundle.load(record["bundle_path"])
        components.compiler.validate_bundle(bundle)
        if (bundle.bundle_id != record["bundle_id"] or bundle.artifact_bindings["plan"] != record["plan"]
                or bundle.artifact_bindings["source_image_id"] != task.user_request.source_image_id
                or bundle.cfg_scale != 4 or bundle.canvas_hw != (512, 512)):
            raise ValueError("Frozen condition changed before rendering")
        for seed in EVALUATION_SEEDS:
            for method in methods:
                steps, eta = samplers[method]
                folder = output / "images" / method
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{index:03d}_{seed}.png"
                row_path = path.with_suffix(".json")
                identity_fields = {"method": method, "index": index, "task_id": task.task_id,
                    "family_id": task.family_id, "eval_seed": seed, "training_seed": components.config["seed"],
                    "steps": steps, "eta": eta, "cfg_scale": 4.0, "bundle_id": bundle.bundle_id,
                    "renderer_policy_identity": identity, "source_image_id": task.user_request.source_image_id}
                if row_path.is_file():
                    row = json.loads(row_path.read_text())
                    if any(row.get(key) != value for key, value in identity_fields.items()) or not path.is_file():
                        raise ValueError("A resumed sampler image has different inputs or lost its pixels")
                    rows.append(row)
                    continue
                device = components.config["devices"]["renderer"]
                torch.cuda.synchronize(device)
                started = time.monotonic()
                if eta == 0:
                    latent = components.renderer.rollout_ode(bundle, seed, steps=steps)
                    trace_path = None
                else:
                    trace = components.renderer.rollout(bundle, seed, steps=steps, eta=eta,
                        behavior_version=components.config.get("sampler_behavior_version", "sampler-expanded-200"), render_id=f"sampler:{method}:{index}:{seed}")
                    latent = trace.final_latent
                    trace_path = path.with_suffix(".pt")
                    trace.save(trace_path)
                    del trace
                image = components.encoders.decode_image(latent, bundle.canvas_hw)
                torch.cuda.synchronize(device)
                seconds = time.monotonic() - started
                image.save(path)
                preview = path.with_suffix(".jpg")
                image.save(preview, quality=80, optimize=True)
                ref = components.artifacts.put_image(image, producer_id=f"sampler:{method}:{index}:{seed}")
                row = {**identity_fields, "status": "rendered", "image_id": ref.artifact_id,
                    "image_value_id": ref.value_id, "image": str(path.resolve()), "preview": str(preview.resolve()),
                    "trace": str(trace_path.resolve()) if trace_path else None, "seconds": seconds,
                    "timing_scope": "diffusion_trace_and_decode_excludes_compile_and_judge",
                    "success": None, "protection": None, "preservation_status": "not_evaluated",
                    "dense": None, "severe_defect": None, "severe_defect_status": "unknown",
                    "calibrated_evaluation": False, "accepted": False}
                atomic_json(row_path, row)
                rows.append(row)
                print(json.dumps({"stage": "render", "index": index, "method": method, "seed": seed,
                                  "seconds": seconds, "completed_in_shard": len(rows)}), flush=True)
                atomic_json(output / "shards" / f"render{suffix}_{shard_name(start, count)}.progress.json",
                            {"completed": len(rows), "required": count * 2 * len(methods), "last": row})
    report = {"stage": "render", "status": "completed", "start": start, "count": count,
              "completed": len(rows), "required": count * 2 * len(methods), "render_seconds": sum(row["seconds"] for row in rows),
              "methods": list(methods), "calibrated_evaluation": False, "accepted": False}
    atomic_json(output / "shards" / f"render{suffix}_{shard_name(start, count)}.json", report)
    return report


def evaluation_routes(evaluation):
    return sorted({atom.expected_value.get("kind", atom.checker_route)
                   if isinstance(atom.expected_value, dict) else atom.checker_route
                   for atom in evaluation.eval_atoms})


def score_comparison_shard(components, tasks, output, start=0, count=8, methods=None, samplers=None):
    samplers = SAMPLERS if samplers is None else samplers
    methods = comparison_methods(methods, samplers)
    suffix = comparison_stage_suffix(methods, samplers)
    output = Path(output)
    components.judge.require_calibration = True
    components.evaluator.unconfirmed_route_policy = "unknown"
    calibration = components.judge.thresholds
    if not calibration:
        raise ValueError("Private sampler scoring requires confirmed checkpoint-specific calibration")
    calibration_record = {"identity": components.judge.config.calibration_identity, "routes": calibration}
    frozen_json(output / "calibration.json", calibration_record)
    rows, failures = [], []
    for index in shard_indices(start, count):
        task = tasks[index]
        evaluation = components.evaluator.evaluations.get(task.task_id)
        for seed in EVALUATION_SEEDS:
            for method in methods:
                image_path = output / "images" / method / f"{index:03d}_{seed}.json"
                image = json.loads(image_path.read_text())
                score_path = output / "scores" / method / f"{index:03d}_{seed}.json"
                if image["task_id"] != task.task_id or image["family_id"] != task.family_id:
                    raise ValueError("Private score input does not match its registered image")
                if score_path.is_file():
                    row = json.loads(score_path.read_text())
                    if row.get("status") == "scored":
                        if (row["image_id"] != image["image_id"] or evaluation is None
                                or row["private_evaluation"] != evaluation.model_dump(mode="json")
                                or row["calibration"] != calibration_record):
                            raise ValueError("A completed private score changed its evidence")
                        rows.append(row)
                        continue
                started = time.monotonic()
                try:
                    if evaluation is None:
                        raise ValueError("Private evaluation is unavailable for this frozen family")
                    missing = [kind for kind in evaluation_routes(evaluation)
                               if not calibration.get(kind, {}).get("confirmed", False)]
                    reward = components.evaluator.evaluate(task, image["image_id"], renders=1, inspects=0, actions=2, invalids=0)
                    if reward.preservation_status not in {"measured", "not_applicable"}:
                        raise ValueError("Preservation evidence is not complete")
                    route_by_atom = {atom.eval_id: atom.expected_value.get("kind", atom.checker_route)
                                     if isinstance(atom.expected_value, dict) else atom.checker_route
                                     for atom in evaluation.eval_atoms}
                    if any(atom.known and route_by_atom[atom.atom_id] in missing for atom in reward.atoms):
                        raise ValueError("An unconfirmed calibration route produced a known score")
                    row = {**image, "status": "scored", "success": float(reward.success),
                        "protection": reward.preservation, "preservation_status": reward.preservation_status,
                        "dense": reward.dense, "reward": reward.to_dict(),
                        "private_evaluation": evaluation.model_dump(mode="json"), "calibration": calibration_record,
                        "scoring_seconds": time.monotonic() - started, "calibrated_evaluation": not missing,
                        "evaluation_complete": True, "known_scores_calibrated": True,
                        "unconfirmed_routes": missing, "unknown_atom_ids": [atom.atom_id for atom in reward.atoms if not atom.known],
                        "calibration_policy": "confirmed_routes_only_unconfirmed_explicit_unknown",
                        "severe_defect": None, "severe_defect_status": "unknown", "accepted": False}
                    rows.append(row)
                except (ValueError, KeyError, RuntimeError) as error:
                    if "CUDA" in str(error) or "out of memory" in str(error).lower():
                        raise
                    row = {**image, "status": "score_blocked", "success": None, "protection": None,
                           "preservation_status": "not_evaluated", "dense": None,
                           "scoring_error": {"type": type(error).__name__, "message": str(error)},
                           "calibrated_evaluation": False, "accepted": False}
                    failures.append({"index": index, "method": method, "eval_seed": seed, "error": row["scoring_error"]})
                atomic_json(score_path, row)
                print(json.dumps({"stage": "score", "index": index, "method": method, "seed": seed,
                                  "status": row["status"], "completed": len(rows), "blocked": len(failures)}), flush=True)
    report = {"stage": "score", "status": "completed" if not failures else "blocked",
              "start": start, "count": count, "completed": len(rows), "required": count * 2 * len(methods),
              "failures": failures, "methods": list(methods), "unknown_atoms": sum(len(row.get("unknown_atom_ids", [])) for row in rows), "accepted": False, "severe_defect_status": "unknown"}
    atomic_json(output / "shards" / f"score{suffix}_{shard_name(start, count)}.json", report)
    return report


def severe_review_status(review, row):
    if not isinstance(review, dict):
        return None
    evidence = review.get("review_evidence")
    if not (
        row.get("image_id") and row.get("image_value_id")
        and review.get("image_id") == row["image_id"]
        and review.get("candidate_image_value_id") == row["image_value_id"]
        and review.get("review_type") in {"ai", "human"}
        and (review["review_type"] != "ai" or review.get("human_reviewed") is False)
        and isinstance(evidence, dict) and evidence
        and isinstance(review.get("reviewed_image_paths"), list) and review["reviewed_image_paths"]
    ):
        return None
    if review.get("known") is True and type(review.get("severe_defect")) is bool:
        return "reviewed"
    if (review.get("known") is False and review.get("severe_defect") is None
            and evidence.get("actual_visual_inspection") is True
            and evidence.get("reviewer") and evidence.get("observation")):
        return "reviewed_unknown"
    return None


def complete_comparison_rows(tasks, output, severe_reviews=None, methods=None, samplers=None):
    samplers = SAMPLERS if samplers is None else samplers
    methods = comparison_methods(methods, samplers)
    suffix = comparison_stage_suffix(methods, samplers)
    required = 64 * 2 * len(methods)
    output = Path(output)
    labels = {}
    if severe_reviews:
        for line in Path(severe_reviews).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["image_id"]
            if key in labels:
                raise ValueError("Severe-defect image has duplicate reviews")
            labels[key] = row
    records, missing = [], []
    for index, task in enumerate(tasks):
        for seed in EVALUATION_SEEDS:
            for method in methods:
                score_path = output / "scores" / method / f"{index:03d}_{seed}.json"
                image_path = output / "images" / method / f"{index:03d}_{seed}.json"
                if not image_path.is_file():
                    missing.append({"index": index, "method": method, "eval_seed": seed, "missing": "render"})
                    continue
                image = json.loads(image_path.read_text())
                if not image.get("image") or not Path(image["image"]).is_file():
                    missing.append({"index": index, "method": method, "eval_seed": seed, "missing": "render_pixels"})
                    continue
                row = json.loads(score_path.read_text()) if score_path.is_file() else image
                if (row["task_id"] != task.task_id or row["family_id"] != task.family_id
                        or row["image_id"] != image["image_id"] or row["bundle_id"] != image["bundle_id"]
                        or row["image_value_id"] != image["image_value_id"]
                        or row["method"] != method or row["eval_seed"] != seed):
                    raise ValueError("Comparison record identity mismatch")
                if not score_protocol_complete(row):
                    missing.append({"index": index, "method": method, "eval_seed": seed, "missing": "calibrated_score"})
                review = labels.get(row["image_id"])
                quality_status = severe_review_status(review, row)
                if quality_status:
                    value = float(review["severe_defect"]) if quality_status == "reviewed" else None
                    row = {**row, "severe_defect": value,
                           "severe_defect_status": quality_status, "severe_defect_review": review}
                else:
                    row = {**row, "severe_defect": None, "severe_defect_status": "unknown"}
                    missing.append({"index": index, "method": method, "eval_seed": seed, "missing": "severe_defect_review"})
                records.append(row)
    path = output / f"comparison{suffix}.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    temporary.replace(path)
    report = {"status": "ready_for_decision" if not missing and len(records) == required else "incomplete",
              "records": len(records), "required_records": required, "methods": list(methods), "families_required": 64,
              "missing": missing, "accepted": False,
              "reviewed_unknown_defect_count": sum(row["severe_defect_status"] == "reviewed_unknown" for row in records),
              "reviewed_unknown_defects": [row["image_id"] for row in records if row["severe_defect_status"] == "reviewed_unknown"]}
    atomic_json(output / f"comparison{suffix}_status.json", report)
    return records, report


def sampler_decision_with_protection(rows, candidate, reference, iterations=10000, expected_families=64):
    import math
    from agentic_image.evaluation.statistics import paired_bootstrap

    subset = [row for row in rows if row["method"] in {candidate, reference}]
    if expected_families not in {64, 128} or len(subset) != expected_families * 4:
        raise ValueError(f"A decision requires both samplers over all {expected_families} registered families and two seeds")
    by_key = {}
    for row in subset:
        key = (row["training_seed"], row["family_id"], row["task_id"], row["eval_seed"])
        by_key.setdefault(key, {})[row["method"]] = row
        quality_status = severe_review_status(row.get("severe_defect_review"), row)
        reviewed = (
            row.get("severe_defect_status") == "reviewed"
            and quality_status == "reviewed"
            and row.get("severe_defect") == float(row["severe_defect_review"]["severe_defect"])
        )
        reviewed_unknown = (
            row.get("severe_defect_status") == "reviewed_unknown"
            and row.get("severe_defect") is None
            and quality_status == "reviewed_unknown"
        )
        if not score_protocol_complete(row) or not (reviewed or reviewed_unknown):
            raise ValueError("Unknown image quality or uncalibrated scores cannot pass selection")
    if len(by_key) != expected_families * 2 or len({key[1] for key in by_key}) != expected_families:
        raise ValueError("The complete paired family and evaluation-seed set is required")
    family_keys = {}
    for key in by_key:
        family_keys.setdefault(key[1], []).append(key)
    if len({key[0] for key in by_key}) != 1 or any(
        len(keys) != 2 or len({key[2] for key in keys}) != 1
        or {key[3] for key in keys} != set(EVALUATION_SEEDS) for keys in family_keys.values()
    ):
        raise ValueError("Each fixed family requires one task and the two registered evaluation seeds")
    protection_rows, na_keys = [], []
    for key, pair in by_key.items():
        if set(pair) != {candidate, reference}:
            raise ValueError("Sampler observations are not paired")
        left, right = pair[candidate], pair[reference]
        if (left["bundle_id"] != right["bundle_id"] or left["source_image_id"] != right["source_image_id"]
                or left["renderer_policy_identity"] != right["renderer_policy_identity"]):
            raise ValueError("Paired samplers must use exactly the same frozen conditions and policy")
        statuses = {left.get("preservation_status"), right.get("preservation_status")}
        if statuses == {"not_applicable"} and left["protection"] is None and right["protection"] is None:
            na_keys.append(list(key))
        elif statuses == {"measured"}:
            if any(row["protection"] is None or not math.isfinite(row["protection"]) for row in pair.values()):
                raise ValueError("Measured protection requires a finite measurement")
            protection_rows.extend(pair.values())
        else:
            raise ValueError("Missing or inconsistent protection cannot be omitted from selection")
    success = paired_bootstrap(subset, candidate, reference, "success", iterations)
    unknowns = [row for row in subset if row["severe_defect_status"] == "reviewed_unknown"]
    upper_rows = [{**row, "severe_defect": float(row["method"] == candidate)}
                  if row["severe_defect_status"] == "reviewed_unknown" else row for row in subset]
    lower_rows = [{**row, "severe_defect": float(row["method"] == reference)}
                  if row["severe_defect_status"] == "reviewed_unknown" else row for row in subset]
    defects = paired_bootstrap(upper_rows, candidate, reference, "severe_defect", iterations)
    lower_defects = paired_bootstrap(lower_rows, candidate, reference, "severe_defect", iterations) if unknowns else defects
    defects = {**defects,
        "analysis": "worst_case_completion_of_reviewed_unknowns" if unknowns else "fully_observed",
        "difference_identification_interval": [lower_defects["difference"], defects["difference"]],
        "robust_confidence_interval": [lower_defects["confidence_interval"][0], defects["confidence_interval"][1]],
        "worst_case_difference_upper_bound": defects["difference"],
        "worst_case_confidence_upper_bound": defects["confidence_interval"][1],
        "reviewed_unknown_count": len(unknowns),
        "reviewed_unknown_by_method": {method: sum(row["method"] == method for row in unknowns) for method in (candidate, reference)},
        "known_label_count_by_method": {method: sum(row["method"] == method and row["severe_defect_status"] == "reviewed" for row in subset) for method in (candidate, reference)},
        "reviewed_unknown_images": [{"method": row["method"], "image_id": row["image_id"],
            "image_value_id": row["image_value_id"], "review_id": row["severe_defect_review"].get("review_id"),
            "family_id": row["family_id"], "task_id": row["task_id"], "eval_seed": row["eval_seed"]} for row in unknowns],
        "completion_rule": "Analysis copies only: unknown candidate=1, unknown reference=0 for the upper bound; the reverse for the lower bound. Original labels remain null and are not known observations.",
        "bound_justification": "With identical paired family/seed resamples and positive weights, each upper-completion difference is at least every admissible binary completion; its 97.5th percentile bounds every completion's upper confidence endpoint.",
    }
    protection = paired_bootstrap(protection_rows, candidate, reference, "protection", iterations) if protection_rows else None
    timing = paired_bootstrap(subset, candidate, reference, "seconds", iterations) if candidate in {"sde28", "sde20"} else None
    accepted = (success["confidence_interval"][0] >= -0.05
                and defects["worst_case_difference_upper_bound"] <= 0.05
                and defects["worst_case_confidence_upper_bound"] <= 0.05
                and (protection is None or protection["confidence_interval"][0] >= -0.05)
                and (timing is None or timing["confidence_interval"][1] < 0))
    return {"candidate": candidate, "reference": reference, "accepted": bool(accepted),
            "status": "accepted" if accepted else "needs_review", "success": success,
            "severe_defects": defects, "protection": protection, "timing": timing,
            "protection_applicable_observations": len(protection_rows) // 2,
            "protection_not_applicable_observations": len(na_keys), "protection_not_applicable_keys": na_keys,
            "success_margin": 0.05, "defect_margin": 0.05, "protection_margin": 0.05,
            "unknown_atoms_by_method": {method: sum(len(row.get("unknown_atom_ids", [])) for row in subset if row["method"] == method) for method in (candidate, reference)},
            "observations_with_unknown_atoms_by_method": {method: sum(bool(row.get("unknown_atom_ids")) for row in subset if row["method"] == method) for method in (candidate, reference)}}


def select_comparison_sampler(config, tasks, output, severe_reviews, iterations=10000):
    output = Path(output)
    base_rows, base_report = complete_comparison_rows(tasks, output, severe_reviews, ("ode50", "sde50"))
    if base_report["status"] != "ready_for_decision":
        atomic_json(output / "sampler_selection.json", {**base_report, "renderer_policy": config["models"]["renderer_policy"]})
        return base_report
    registered_policy = json.loads((output / "renderer_policy_identity.json").read_text())
    if policy_identity(config["models"]["renderer_policy"]) != registered_policy:
        raise ValueError("The renderer policy changed after comparison images were generated")
    decisions = {"sde50": sampler_decision_with_protection(base_rows, "sde50", "ode50", iterations)}
    chosen = None
    pending = []
    if decisions["sde50"]["accepted"]:
        chosen = "sde50"
        for candidate in ("sde28", "sde20"):
            rows, report = complete_comparison_rows(tasks, output, severe_reviews, ("sde50", candidate))
            if report["status"] != "ready_for_decision":
                decisions[candidate] = {"candidate": candidate, "reference": "sde50", "accepted": False,
                    "status": "incomplete", "missing": report["missing"]}
                pending.append(candidate)
                continue
            decisions[candidate] = sampler_decision_with_protection(rows, candidate, "sde50", iterations)
            if decisions[candidate]["accepted"]:
                chosen = candidate
    full_rows, full_report = complete_comparison_rows(tasks, output, severe_reviews)
    result = {"accepted": chosen is not None, "status": "accepted" if chosen else "needs_review",
              "candidate": chosen, "decisions": decisions, "renderer_policy": config["models"]["renderer_policy"],
              "renderer_policy_identity": registered_policy,
              "base_pair_records": 256, "full_experiment_records": len(full_rows), "full_experiment_required": 512,
              "full_experiment_complete": full_report["status"] == "ready_for_decision",
              "families": 64, "eval_seeds": list(EVALUATION_SEEDS), "pending_shorter_samplers": pending,
              "cfg_scale": 4.0, "calibration": str((output / "calibration.json").resolve()),
              "severe_reviews": str(Path(severe_reviews).resolve()), "human_reviewed": False,
              "review_scope": "AI visual defect review and conservative private scoring with unknown atoms retained"}
    if chosen:
        result["steps"], result["eta"] = SAMPLERS[chosen]
    atomic_json(output / "sampler_selection.json", result)
    for candidate, decision in decisions.items():
        if decision["accepted"]:
            proof = {**result, "candidate": candidate, "steps": SAMPLERS[candidate][0], "eta": SAMPLERS[candidate][1],
                     "accepted": True, "decision": decision}
            atomic_json(output / f"sampler_selection.{candidate}.json", proof)
    return result


def prepare_severe_review(tasks, output, samplers=None, review_prefix="sampler-quality"):
    samplers = SAMPLERS if samplers is None else samplers
    required = 64 * 2 * len(samplers)
    import os
    import random
    from PIL import Image

    output = Path(output)
    slots = [(index, task, method, seed) for index, task in enumerate(tasks)
             for seed in EVALUATION_SEEDS for method in samplers]
    if len(slots) != required:
        raise ValueError("The blinded review registration requires all 64 fixed tasks")
    random.Random(738211).shuffle(slots)
    folder = output / "blinded_quality_review"
    folder.mkdir(parents=True, exist_ok=True)
    records, key = [], []
    for number, (index, task, method, seed) in enumerate(slots):
        review_id = f"{review_prefix}-{number:04d}"
        key.append({"review_id": review_id, "method": method, "index": index, "eval_seed": seed})
        path = output / "images" / method / f"{index:03d}_{seed}.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        dest = folder / f"{review_id}.png"
        if dest.exists():
            if not os.path.samefile(row["image"], dest):
                raise ValueError("A blinded image changed after review registration")
        else:
            os.link(row["image"], dest)
        with Image.open(dest) as image:
            preview = folder / f"{review_id}.jpg"
            if not preview.exists():
                image.save(preview, quality=85, optimize=True)
        records.append({"review_id": review_id, "image_id": row["image_id"],
            "candidate_image_value_id": row["image_value_id"], "candidate_path": str(dest.resolve()),
            "preview_path": str(preview.resolve()), "request": task.user_request.text,
            "source_image_id": task.user_request.source_image_id,
            "severe_defect": None, "known": False, "review_type": "ai", "human_reviewed": False,
            "review_evidence": None, "reviewed_image_paths": []})
    frozen_json(output / "quality_review_key.private.json", key)
    atomic_json(folder / "records.json", records)
    return {"status": "prepared" if len(records) == required else "partial", "images": len(records),
            "required": required, "records": str((folder / "records.json").resolve()), "known_labels": 0}


def cached_freeze_report(config, tasks, output, start=0, count=8):
    output = Path(output)
    paths = [output / "plans" / f"{index:03d}" / "frozen.json" for index in shard_indices(start, count)]
    if not all(path.is_file() for path in paths):
        return None
    actor_identity = policy_identity(config["models"]["actor_policy"], actor=True)
    frozen_json(output / "actor_policy_identity.json", actor_identity)
    for index, path in zip(shard_indices(start, count), paths):
        row = json.loads(path.read_text())
        if (row["task"] != tasks[index].model_dump(mode="json") or row.get("actual_sampled_plan") is not True
                or row["actor_policy_identity"] != actor_identity or not Path(row["bundle_path"]).is_file()):
            raise ValueError("Cached controller plans do not match this frozen registration")
    report = {"stage": "freeze", "status": "completed", "start": start, "count": count,
              "completed": count, "errors": [], "all_frozen_reused": True, "models_loaded": False}
    atomic_json(output / "shards" / f"freeze_{shard_name(start, count)}.json", report)
    return report


def comparison_methods(methods=None, samplers=None):
    samplers = SAMPLERS if samplers is None else samplers
    methods = tuple(samplers) if methods is None else tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(samplers):
        raise ValueError("Use distinct registered sampler methods")
    return methods


def comparison_stage_suffix(methods, samplers=None):
    samplers = SAMPLERS if samplers is None else samplers
    methods = comparison_methods(methods, samplers)
    if set(methods) == set(samplers):
        return ""
    if set(methods) == {"ode50", "sde50"}:
        return "_base"
    if set(methods) == {"sde28", "sde20"}:
        return "_short"
    return "_" + "_".join(methods)


def score_protocol_complete(row):
    return row.get("status") == "scored" and (
        row.get("calibrated_evaluation") is True
        or row.get("evaluation_complete") is True and row.get("known_scores_calibrated") is True
    )


def cached_render_report(config, tasks, output, start=0, count=8, methods=None, samplers=None):
    output = Path(output)
    samplers = SAMPLERS if samplers is None else samplers
    methods = comparison_methods(methods, samplers)
    keys = [(index, seed, method) for index in shard_indices(start, count)
            for seed in EVALUATION_SEEDS for method in methods]
    paths = [output / "images" / method / f"{index:03d}_{seed}.json" for index, seed, method in keys]
    if not all(path.is_file() for path in paths):
        return None
    identity = policy_identity(config["models"]["renderer_policy"])
    frozen_json(output / "renderer_policy_identity.json", identity)
    rows = []
    for (index, seed, method), path in zip(keys, paths):
        row = json.loads(path.read_text())
        plan = json.loads((output / "plans" / f"{index:03d}" / "frozen.json").read_text())
        task = tasks[index]
        expected = {"index": index, "method": method, "eval_seed": seed, "task_id": task.task_id,
                    "family_id": task.family_id, "source_image_id": task.user_request.source_image_id,
                    "bundle_id": plan["bundle_id"], "renderer_policy_identity": identity,
                    "steps": samplers[method][0], "eta": samplers[method][1], "cfg_scale": 4.0,
                    "training_seed": config["seed"]}
        if any(row.get(key) != value for key, value in expected.items()) or not Path(row["image"]).is_file():
            raise ValueError("Cached sampler image differs from the frozen source, PLAN, policy or seed")
        if row["eta"] > 0 and (not row.get("trace") or not Path(row["trace"]).is_file()):
            raise ValueError("Cached stochastic image lost its actual diffusion trace")
        rows.append(row)
    report = {"stage": "render", "status": "completed", "start": start, "count": count,
              "completed": len(rows), "required": count * 2 * len(methods), "methods": list(methods),
              "render_seconds": sum(row["seconds"] for row in rows), "all_images_reused": True,
              "models_loaded": False, "accepted": False, "calibrated_evaluation": False}
    suffix = comparison_stage_suffix(methods, samplers)
    atomic_json(output / "shards" / f"render{suffix}_{shard_name(start, count)}.json", report)
    return report
