import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
from peft import load_peft_weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collect_cached_public_trajectories import CachedCompiler, CachedRenderBridge, read_json
from evaluate_public_plans import policy_identity, score
from agentic_image.actor.records import ActionSample, ActorSnapshot
from agentic_image.config import load_config
from agentic_image.core.protocol import parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import ContractError, PlanSpec
from agentic_image.data.io import read_jsonl, write_jsonl
from agentic_image.data.paired import actor_snapshot
from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json, atomic_torch_save
from agentic_image.render.types import ConditionBundle
from agentic_image.runtime.components import Components
from agentic_image.runtime.prompt import SYSTEM_PROMPT
from agentic_image.runtime.rollout import read_tasks


def committed_updates(policy):
    directory = next((parent for parent in Path(policy["path"]).parents if (parent / "metrics.jsonl").is_file()), None)
    if directory is None:
        raise ValueError(f"Policy has no committed training metrics: {policy['path']}")
    metrics = list(read_jsonl(directory / "metrics.jsonl"))
    steps = [row["step"] for row in metrics]
    if not steps or steps != list(range(1, len(steps) + 1)):
        raise ValueError(f"Training metrics are not a complete ordered update sequence: {directory}")
    marker = Path(metrics[-1]["checkpoint"]) / "COMMITTED.json"
    commit = read_json(marker)
    if commit.get("status") != "COMMITTED" or commit.get("new_version") != f"protocol:{steps[-1]}" or "actor" not in commit.get("optimizer_steps", []):
        raise ValueError(f"The final actor update lacks its exact committed transaction: {marker}")
    return {"policy": policy, "updates": steps[-1], "metrics_path": str((directory / "metrics.jsonl").resolve()), "final_commit_path": str(marker.resolve()), "final_commit_version": commit["new_version"]}


def training_lineage(policies, plan_metadata_path, transition_completed_path):
    plan_metadata = read_json(plan_metadata_path)
    completion = read_json(transition_completed_path)
    transition_registration = read_json(Path(transition_completed_path).parent / "registration.json")
    if completion.get("status") != "completed" or policy_identity(completion["policy"]) != policies["after"] or policy_identity(transition_registration["initial_policy"]) != policies["before"]:
        raise ValueError("The evaluated policies do not match the completed transition training lineage")
    recovery = committed_updates(policy_identity(plan_metadata["initial_policy"]))
    planning = committed_updates(policies["before"])
    transition = committed_updates(policies["after"])
    if completion["updates"] != transition["updates"] or transition_registration["updates"] != transition["updates"]:
        raise ValueError("Transition completion and registration disagree with the actual committed updates")
    before = recovery["updates"] + planning["updates"]
    return {"stages": {"initial_recovery": recovery, "global_plan": planning, "stage_transitions": transition}, "cumulative_policy_updates": {"before": before, "after": before + transition["updates"]}, "plan_metadata_path": str(Path(plan_metadata_path).resolve()), "transition_completed_path": str(Path(transition_completed_path).resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--before-policy", required=True)
    parser.add_argument("--after-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--tasks", default="data/public/protocol_eval_16_tasks.jsonl")
    parser.add_argument("--training-tasks", default="data/public/protocol_pilot_tasks.jsonl")
    parser.add_argument("--plan-evaluation", default="runs/plan-global-evaluation")
    parser.add_argument("--renders", default="runs/renderer-global-ai-evaluation")
    parser.add_argument("--plan-training-metadata", default="data/reviewed-global-plan-sft/metadata.json")
    parser.add_argument("--transition-completed", default="data/public-stage-transitions/training_completed.json")
    parser.add_argument("--initialization-completed")
    args = parser.parse_args()
    tasks = read_tasks(args.tasks)
    training = read_tasks(args.training_tasks)
    if len(tasks) != 16 or any(task.split_id != "official-dev" for task in tasks) or {task.family_id for task in tasks} & {task.family_id for task in training}:
        raise ValueError("The fixed sixteen official-dev families must remain outside training")
    selected = tasks[:4]
    renders = {row["task_id"]: row for row in read_json(Path(args.renders) / "results.json")}
    if read_json(Path(args.renders) / "summary.json")["status"] != "completed":
        raise ValueError("The fixed trained-renderer heldout images must already be complete")
    config = load_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies = {"before": policy_identity(args.before_policy), "after": policy_identity(args.after_policy)}
    if args.initialization_completed:
        initialization = read_json(args.initialization_completed)
        initialization_registration = read_json(Path(args.initialization_completed).parent / "registration.json")
        transition_registration = read_json(Path(args.transition_completed).parent / "registration.json")
        if initialization.get("status") != "completed" or policy_identity(initialization["policy"]) != policies["after"] or policy_identity(initialization["initial_policy"]) != policies["before"]:
            raise ValueError("Initialization evaluation policies do not match completed training")
        previous = training_lineage({"before": policy_identity(transition_registration["initial_policy"]), "after": policies["before"]}, args.plan_training_metadata, args.transition_completed)
        latest = committed_updates(policies["after"])
        if latest["updates"] != initialization["updates"] or latest["updates"] != initialization_registration["updates"]:
            raise ValueError("Initialization updates disagree with their committed metrics")
        before = previous["cumulative_policy_updates"]["after"]
        lineage = {"prior_lineage": previous, "initialization": latest, "cumulative_policy_updates": {"before": before, "after": before + latest["updates"]}}
    else:
        lineage = training_lineage(policies, args.plan_training_metadata, args.transition_completed)
    config["run_dir"] = str(output)
    config["models"].update(actor_policy=policies["before"]["path"], actor_reference=None)
    config["judge"]["calibration"] = None
    config["actor"]["max_new_tokens"] = 384
    for name in ("actor", "judge", "encoders", "renderer"):
        config["devices"][name] = args.device
    components = Components(config)
    actor = components.load_actor()
    components.load_judge(private=False)
    tokenizer = actor.processor.tokenizer
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    budget = config["actor"]["max_input_tokens"] - count(SYSTEM_PROMPT) - 128
    registration = {"scope": "fixed_heldout_public_stage_transition_contract_v1", "tasks": [task.task_id for task in selected], "family_ids": [task.family_id for task in selected], "policies": policies, "contexts": ["initial_plan", "after_plan", "after_plan_error", "after_render_error"], "families": 4, "max_samples_per_context_per_policy": 1, "max_new_tokens": 384, "sampling_seed": 101, "environment_render_seed": 11, "numerics_profile": actor.config.numerics_profile, "optimizer_updates": 0, "new_renderer_generations": 0, "judge_calibrated": False, "visual_success_evaluation": False}
    registration["training_lineage"] = lineage
    reg_path = output / "registration.json"
    if reg_path.exists() and read_json(reg_path) != registration:
        raise ValueError("The frozen heldout evaluation registration changed")
    atomic_json(reg_path, registration)
    contexts = []
    for index, task in enumerate(selected):
        row = renders[task.task_id]
        if row["family_id"] != task.family_id or row["source_image_id"] != task.user_request.source_image_id or row["request"] != task.user_request.text:
            raise ValueError("The saved heldout render no longer matches its public task")
        stem = row["result"]["bundle_id"].replace(":", "_")
        bundle = ConditionBundle.load(Path(args.renders) / "bundles" / f"{stem}.pt")
        compiler = CachedCompiler(components, bundle)
        state_root = Path(args.plan_evaluation) / "tasks" / f"{index:03d}"
        for name in registration["contexts"]:
            folder = output / "states" / f"{index:03d}" / name
            folder.mkdir(parents=True, exist_ok=True)
            snapshot_path = folder / "snapshot.pt"
            state_path = folder / "before.json"
            renderer = CachedRenderBridge(components, row, 0)

            def restore(path):
                return EpisodeEnvironment.restore(read_json(path), components.artifacts, compiler, renderer, components.observer, compiler.load, token_counter=count, max_input_tokens=budget)

            if not state_path.exists():
                environment = restore(state_root / ("before.json" if name == "initial_plan" else "after.after.json"))
                setup = []
                if name == "after_render_error":
                    action = {"kind": "RENDER", "control_set_id": environment.state.manifest.active_control_set_id, "expected_revision": environment.state.manifest.revision_id}
                    result = environment.step(action, 11)
                    if result.error or result.render is None:
                        raise RuntimeError("The actual heldout cached render failed")
                    setup.append({"action": action, "result": result.model_dump(mode="json"), "render_events": renderer.events})
                if name in {"after_plan_error", "after_render_error"}:
                    action = {"kind": "PLAN", "plan": environment.state.plan.model_dump(mode="json")}
                    result = environment.step(action, 11)
                    if result.error is None or result.error.code != "PLAN_ALREADY_EXISTS" or result.done:
                        raise RuntimeError("The controlled heldout duplicate-PLAN error was not actually produced")
                    setup.append({"action": action, "result": result.model_dump(mode="json"), "origin": "explicit_fixed_contract_error_setup_not_actor_sample"})
                snapshot = actor_snapshot(components, environment)
                atomic_torch_save(snapshot_path, snapshot.state_dict())
                atomic_json(state_path, environment.snapshot())
                atomic_json(folder / "setup.json", setup)
            contexts.append({"index": index, "task": task, "name": name, "state_path": state_path, "snapshot_path": snapshot_path, "bundle": bundle, "render": row})
    result_path = output / "results.jsonl"
    results = {(row["policy"], row["task_id"], row["context"]): row for row in read_jsonl(result_path)} if result_path.exists() else {}
    for label in ("before", "after"):
        weights = load_peft_weights(policies[label]["path"], device="cpu", local_files_only=True)
        actor.restore_adapter_state({"policy": weights})
        actual = actor.adapter_state()["policy"]
        if actual.keys() != weights.keys() or any(not torch.equal(actual[key], weights[key]) for key in weights):
            raise RuntimeError("The evaluation policy was not restored exactly")
        del actual, weights
        for context in contexts:
            task, name = context["task"], context["name"]
            key = (label, task.task_id, name)
            if key in results:
                continue
            compiler = CachedCompiler(components, context["bundle"])
            renderer = CachedRenderBridge(components, context["render"], 0)
            renderer.cached_used = name == "after_render_error"
            environment = EpisodeEnvironment.restore(read_json(context["state_path"]), components.artifacts, compiler, renderer, components.observer, compiler.load, token_counter=count, max_input_tokens=budget)
            snapshot = ActorSnapshot.load(context["snapshot_path"])
            if name == "initial_plan":
                target = {"kind": "PLAN", "plan": PlanSpec(global_text=task.user_request.text).model_dump(mode="json")}
                expected = {"PLAN"}
            elif name == "after_render_error":
                target = {"kind": "READ_NODES", "node_ids": ["plan"], "fields": ["typed_parameters"]}
                expected = {"READ_NODES"}
            else:
                target = {"kind": "RENDER", "control_set_id": environment.state.manifest.active_control_set_id, "expected_revision": environment.state.manifest.revision_id}
                expected = {"RENDER"} if name == "after_plan" else {"RENDER", "READ_NODES"}
            target_raw = json.dumps(environment.public_action(target), ensure_ascii=True, separators=(",", ":"))
            target_ids = tokenizer.encode(target_raw, add_special_tokens=False) + [actor.config.action_eos_token_id]
            folder = output / "samples" / label / f'{context["index"]:03d}' / name
            folder.mkdir(parents=True, exist_ok=True)
            sample_path = folder / "action.pt"
            seed = stream_seed(101, "heldout-stage-transition", task.task_id, name)
            if sample_path.exists():
                sample = ActionSample.from_state_dict(torch.load(sample_path, map_location="cpu", weights_only=True))
                actor._validate_sample_numerics(sample)
            else:
                sample = actor.sample(snapshot, seed, f"heldout-stage-transition:{label}", probe_positions=(0, 15, 31))
                atomic_torch_save(sample_path, sample.state_dict())
            (folder / "raw.txt").write_text(sample.text)
            likelihood, replay = score(actor, snapshot, target_ids, sample)
            parsed = None
            parse_error = None
            try:
                parsed = parse_action(environment.resolve_action(sample.text)).model_dump(mode="json")
            except ContractError as error:
                parse_error = {"code": error.code, "message": str(error)}
            result = environment.step(sample.text, 11)
            record = {"policy": label, "task_id": task.task_id, "family_id": task.family_id, "context": name, "snapshot_path": str(context["snapshot_path"]), "sample_path": str(sample_path), "raw_action": sample.text, "parsed_action": parsed, "parse_error": parse_error, "expected_action_kinds": sorted(expected), "expected_transition_executed": parsed is not None and parsed["kind"] in expected and result.error is None and not result.protocol_failure, "environment_result": result.model_dump(mode="json"), "render_events": renderer.events, "target_likelihood": likelihood, "replay": replay, "judge_calibrated": False, "visual_success": None}
            atomic_json(folder / "after.json", environment.snapshot())
            results[key] = record
            write_jsonl(result_path, results.values())
            atomic_json(output / "progress.json", {"policy": label, "task_id": task.task_id, "context": name, "completed": len(results), "total": 32})
            print(json.dumps({"policy": label, "task_id": task.task_id, "context": name, "expected_transition_executed": record["expected_transition_executed"], "replay_passed": replay["passed"]}), flush=True)
    summary = {**registration, "status": "completed", "policies": {}}
    for label in ("before", "after"):
        subset = [row for row in results.values() if row["policy"] == label]
        summary["policies"][label] = {"samples": len(subset), "expected_transitions": sum(row["expected_transition_executed"] for row in subset), "contexts": {name: {"samples": sum(row["context"] == name for row in subset), "expected_transitions": sum(row["context"] == name and row["expected_transition_executed"] for row in subset)} for name in registration["contexts"]}, "action_kinds": dict(Counter((row["parsed_action"] or {}).get("kind", "invalid") for row in subset)), "all_replay_passed": all(row["replay"]["passed"] for row in subset), "maximum_replay_p99": max(row["replay"]["p99"] for row in subset)}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
