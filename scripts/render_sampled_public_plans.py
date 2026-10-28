import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.config import load_config
from agentic_image.core.types import PlanSpec, PublicTask
from agentic_image.learning.transaction import atomic_json
from agentic_image.render.types import ConditionBundle
from agentic_image.runtime.components import Components


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--training-exit", required=True)
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid deterministic shard selection")
    collection = Path(args.collection).resolve()
    output = Path(args.output).resolve()
    if output == collection or collection in output.parents:
        raise ValueError("Render outputs must be independent of the immutable collection")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    metadata = read_json(collection / "metadata.json")
    if read_json(collection / "progress.json")["stage"] != "complete":
        raise RuntimeError("The public PLAN collection is incomplete")
    tasks = [PublicTask.model_validate(row) for row in metadata["tasks"]]
    records = [json.loads(row) for row in (collection / "results.jsonl").read_text().splitlines()]
    if len(tasks) != len(records) or len({task.family_id for task in tasks}) != len(tasks):
        raise ValueError("Collection must contain one completed task per training family")
    selected = []
    missing = []
    for index, (task, record) in enumerate(zip(tasks, records)):
        if task.split_id != "train" or record["task_id"] != task.task_id or record["family_id"] != task.family_id:
            raise ValueError("Only the fixed original training families may be rendered")
        attempt = next((row for row in record["attempts"] if row["plan_contract_succeeded"] and row["bundle_compiled"]), None)
        if attempt is None:
            missing.append(task.task_id)
            continue
        if attempt["teacher"] != metadata["teacher"] or attempt["teacher"]["source"] != "actual_actor_sample":
            raise ValueError("The PLAN must come from the registered actual actor sample")
        if index % args.shard_count == args.shard_index:
            selected.append((index, task, attempt))
    outcome = read_json(args.training_exit)
    if outcome.get("returncode", outcome.get("exit_code")) != 0:
        raise RuntimeError("Renderer training has not exited successfully")
    metrics = [json.loads(row) for row in Path(args.training_metrics).read_text().splitlines()]
    if not metrics or metrics[-1]["step"] != 20:
        raise RuntimeError("This render collection requires exactly 20 renderer updates")
    checkpoint = Path(metrics[-1]["checkpoint"])
    if not (checkpoint / "COMMITTED.json").is_file():
        raise RuntimeError("The last renderer update has no committed checkpoint")
    policy = Path(args.policy).resolve()
    policy_stat = policy.stat()
    config = load_config(args.config)
    config["run_dir"] = str(output)
    config["seed"] = 11
    config["renderer"].update(steps=50, eta=0.3, cfg_scale=4.0)
    config["devices"].update(renderer=args.device, encoders=args.device)
    config["models"].update(renderer_policy=str(policy), renderer_reference=None)
    registration = {
        "scope": "actual_sampled_training_plan_initial_renders",
        "collection": str(collection), "actor_policy": metadata["teacher"],
        "renderer_policy": str(policy), "renderer_policy_bytes": policy_stat.st_size,
        "renderer_policy_mtime_ns": policy_stat.st_mtime_ns, "renderer_updates": 20,
        "renderer_revision": config["models"]["renderer_revision"],
        "steps": 50, "seed": 11, "eta": 0.3, "cfg_scale": 4.0,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "selected": [{"index": index, "task_id": task.task_id, "family_id": task.family_id, "bundle_id": attempt["bundle_id"]} for index, task, attempt in selected],
        "without_compiled_plan": missing, "condition_scope": "saved_public_source_only_bundle",
        "visual_success": None, "repair_label": None, "eligible_for_training": False,
        "ai_review": "pending", "human_reviewed": False, "judge_calibrated": False,
        "environment_render_step_executed": False,
    }
    registration_path = output / "registration.json"
    if registration_path.exists() and read_json(registration_path) != registration:
        raise ValueError("Existing render registration differs")
    atomic_json(registration_path, registration)
    result_path = output / "results.json"
    results = read_json(result_path) if result_path.exists() else []
    if len(results) > len(selected) or any(row["task_id"] != selected[i][1].task_id for i, row in enumerate(results)):
        raise ValueError("Existing render results differ from the fixed shard order")
    atomic_json(progress_path, {"stage": "loading_renderer", "pid": os.getpid(), "completed": len(results), "total": len(selected), "updated_at": time.time()})
    components = Components(config)
    components.record_memory("before_renderer_load")
    components.load_renderer()
    components.render_bridge.behavior_version = "renderer-global-ai:step20"
    supplied = torch.load(policy, map_location="cpu", weights_only=True)
    loaded = components.renderer.policy_state()
    if supplied.keys() != loaded.keys() or any(not torch.equal(supplied[key], loaded[key]) for key in supplied):
        raise RuntimeError("The trained renderer policy was not loaded exactly")
    changed = sum(int(torch.count_nonzero(value)) for key, value in supplied.items() if "lora_B" in key)
    if not changed:
        raise RuntimeError("The trained renderer has no nonzero LoRA B parameters")
    atomic_json(output / "policy_load.json", {"exact_tensor_equality": True, "tensor_count": len(supplied), "nonzero_lora_b_elements": changed, "policy_path": str(policy)})
    del supplied, loaded
    components.record_memory("renderer_loaded")
    for position, (index, task, attempt) in enumerate(selected):
        if position < len(results):
            if not Path(results[position]["image"]).is_file() or not Path(results[position]["result"]["trajectory_id"]).is_file():
                raise RuntimeError("A completed render is missing its image or trace")
            continue
        started = time.monotonic()
        plan = PlanSpec.model_validate(attempt["internal_plan"])
        allowed = {task.user_request.source_image_id, *task.user_request.appearance_image_ids}
        if attempt["source_image_id"] != task.user_request.source_image_id or not set(attempt["accessed_public_artifacts"]) <= allowed:
            raise ValueError("The original collection accessed nonpublic conditioning artifacts")
        for key in ("bundle_path", "bundle_recipe_path", "snapshot_path", "action_sample_path", "before_environment_path", "after_environment_path", "raw_action_path"):
            path = Path(attempt[key]).resolve()
            if collection not in path.parents or not path.is_file():
                raise ValueError(f"Missing original collection evidence: {key}")
        before = read_json(attempt["before_environment_path"])
        initial = read_json(attempt["after_environment_path"])
        if before["plan"] is not None or initial["plan"] != attempt["internal_plan"] or initial["task"] != task.model_dump(mode="json") or initial["executed_renders"]:
            raise ValueError("The initial state is not the actual unrendered sampled PLAN state")
        recipe = read_json(attempt["bundle_recipe_path"])
        if recipe["plan"] != plan.model_dump(mode="json") or recipe["source_image_id"] != task.user_request.source_image_id:
            raise ValueError("The bundle recipe differs from the sampled public PLAN")
        destination = output / "bundles" / Path(attempt["bundle_path"]).name
        if not destination.exists():
            os.link(attempt["bundle_path"], destination)
        if not destination.with_suffix(".json").exists():
            shutil.copy2(attempt["bundle_recipe_path"], destination.with_suffix(".json"))
        bundle = ConditionBundle.load(destination)
        components.compiler.validate_bundle(bundle)
        bindings = bundle.artifact_bindings
        slots = bindings.get("reference_slots", [])
        if bundle.bundle_id != attempt["bundle_id"] or bindings.get("plan") != attempt["internal_plan"] or bindings.get("source_image_id") != task.user_request.source_image_id or any(slot["image_id"] not in allowed for slot in slots):
            raise ValueError("Saved bundle bindings differ from the actual public PLAN")
        if bundle.canvas_hw != tuple(config["canvas"]) or bundle.cfg_scale != 4.0 or bundle.sigma_table_id != "flow-sde-50-eta0.3-v3":
            raise ValueError("Saved bundle sampling settings differ from the fixed collection protocol")
        atomic_json(progress_path, {"stage": "rendering", "pid": os.getpid(), "completed": len(results), "total": len(selected), "task_id": task.task_id, "updated_at": time.time()})
        result = components.render_bridge.render(bundle, 11, f"sampled-plan-initial:{index:03d}")
        image_path = output / f"{index:03d}.png"
        components.artifacts.get_image(result.image_id).save(image_path)
        row = {
            "index": index, "task_id": task.task_id, "family_id": task.family_id,
            "source_image_id": task.user_request.source_image_id, "request": task.user_request.text,
            "plan": attempt["internal_plan"], "sample_attempt": attempt["attempt"],
            "actor_policy": attempt["teacher"], "renderer_policy": str(policy),
            "bundle_path": str(destination), "bundle_recipe_path": str(destination.with_suffix(".json")),
            "pre_action_snapshot": attempt["snapshot_path"], "action_sample": attempt["action_sample_path"],
            "before_plan_state": attempt["before_environment_path"], "initial_unrendered_state": attempt["after_environment_path"],
            "raw_action_path": attempt["raw_action_path"], "result": result.model_dump(mode="json"),
            "image": str(image_path), "seconds": time.monotonic() - started,
            "visual_success": None, "repair_label": None, "eligible_for_training": False,
            "judge_calibrated": False, "ai_review": "pending", "human_reviewed": False,
            "environment_render_step_executed": False,
        }
        results.append(row)
        atomic_json(result_path, results)
        components.record_memory(f"initial_render:{index:03d}")
        atomic_json(progress_path, {"stage": "rendering", "pid": os.getpid(), "completed": len(results), "total": len(selected), "last_seconds": row["seconds"], "updated_at": time.time()})
        print(json.dumps(row), flush=True)
    summary = {**registration, "status": "completed", "completed": len(results), "render_seconds": sum(row["seconds"] for row in results), "peak_reserved_gib": torch.cuda.max_memory_reserved(args.device) / 2**30}
    atomic_json(output / "summary.json", summary)
    atomic_json(progress_path, {"stage": "complete", "pid": os.getpid(), "completed": len(results), "total": len(selected), "updated_at": time.time()})
    atomic_json(output / "completed.json", {"status": "completed", "summary": str(output / "summary.json")})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
