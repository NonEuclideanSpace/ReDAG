import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.actor.records import ActionSample
from agentic_image.config import load_config
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.protocol import PlanAction, parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import ContractError
from agentic_image.data.io import write_jsonl
from agentic_image.data.paired import actor_snapshot, public_environment
from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json, atomic_torch_save
from agentic_image.runtime.components import Components
from agentic_image.runtime.rollout import read_tasks


class PublicArtifacts(ArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.allowed = set()
        self.accessed = set()

    def get_ref(self, artifact_id):
        if artifact_id not in self.allowed:
            raise ContractError("NONPUBLIC_ARTIFACT", "The collection only permits original public request images")
        reference = super().get_ref(artifact_id)
        if reference.kind != "rgb":
            raise ContractError("NONPUBLIC_ARTIFACT", "The collection cannot read masks or non-image artifacts")
        self.accessed.add(artifact_id)
        return reference


def collect(arguments):
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(arguments.config)
    config["run_dir"] = str(output)
    for name in ("actor", "encoders", "judge"):
        config["devices"][name] = arguments.device
    config["models"]["actor_policy"] = arguments.policy
    config["judge"]["calibration"] = None
    if arguments.max_new_tokens is not None:
        if not 1 <= arguments.max_new_tokens <= config["actor"]["max_new_tokens"]:
            raise ValueError("The collection token limit must fit the registered actor budget")
        config["actor"]["max_new_tokens"] = arguments.max_new_tokens
    tasks = read_tasks(arguments.tasks)
    if len(tasks) != 32 or len({task.family_id for task in tasks}) != 32:
        raise ValueError("This collection requires exactly 32 distinct public training families")
    if any(task.split_id != "train" or not task.user_request.source_image_id for task in tasks):
        raise ValueError("Only original public training requests with source images are supported")
    policy = Path(arguments.policy).resolve()
    weight = policy / "adapter_model.safetensors"
    identity = {
        "model_id": config["models"]["actor"],
        "revision": config["models"]["actor_revision"],
        "policy_path": str(policy),
        "policy_weight_bytes": weight.stat().st_size,
        "policy_weight_mtime_ns": weight.stat().st_mtime_ns,
        "training_updates": arguments.policy_updates,
        "source": "actual_actor_sample",
    }
    components = Components(config)
    components.artifacts = PublicArtifacts(config["data"]["artifacts"])
    components.record_memory("before_actor_load")
    actor = components.load_actor()
    components.record_memory("actor_loaded")
    identity["numerics_profile"] = actor.config.numerics_profile
    metadata = {
        "teacher": identity, "tasks": [task.model_dump(mode="json") for task in tasks],
        "max_samples_per_task": 2, "read_scope": "public_request_source_and_appearance_rgb_only",
        "judge_calibrated": False, "ai_review": "pending", "target_compatibility_review": "pending",
        "eligible_for_training": False, "max_new_tokens": config["actor"]["max_new_tokens"],
    }
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("An existing collection has a different policy or public task identity")
    atomic_json(metadata_path, metadata)
    records = []
    started = time.monotonic()

    def progress(stage, task_id=None):
        report = {
            "stage": stage, "pid": os.getpid(), "completed_tasks": len(records), "total_tasks": len(tasks),
            "current_task": task_id, "real_samples": sum(len(row["attempts"]) for row in records),
            "compiled_actor_plans": sum(row["compiled_actor_plan"] for row in records),
            "ai_review": "pending", "target_compatibility_review": "pending", "eligible_for_training": False,
            "judge_calibrated": False, "seconds": time.monotonic() - started,
        }
        atomic_json(output / "progress.json", report)
        print(json.dumps(report), flush=True)

    def restore(path, template):
        return EpisodeEnvironment.restore(
            json.loads(path.read_text()), components.artifacts, components.compiler, None, components.observer,
            components.compiler.load if components.compiler else None,
            token_counter=template.token_counter,
            max_input_tokens=template.view_builder.max_input_tokens,
        )

    try:
        for index, task in enumerate(tasks):
            folder = output / "tasks" / f"{index:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            task_result = folder / "result.json"
            if task_result.exists():
                records.append(json.loads(task_result.read_text()))
                continue
            components.artifacts.allowed = {task.user_request.source_image_id, *task.user_request.appearance_image_ids}
            components.artifacts.accessed = set()
            environment = public_environment(components, task, f"actor-plan:{index:03d}")
            attempts = []
            progress("sampling", task.task_id)
            for attempt in range(2):
                stem = folder / f"attempt_{attempt:02d}"
                result_path = stem.with_suffix(".json")
                before_path = stem.with_suffix(".before.json")
                after_path = stem.with_suffix(".after.json")
                sample_path = stem.with_suffix(".action.pt")
                snapshot_path = stem.with_suffix(".snapshot.pt")
                raw_path = stem.with_suffix(".raw.txt")
                if result_path.exists():
                    row = json.loads(result_path.read_text())
                    attempts.append(row)
                    environment = restore(after_path, environment)
                    if row["plan_contract_succeeded"] or environment.state.done:
                        break
                    continue
                seed = stream_seed(config["seed"], "public-plan-candidate", task.task_id, attempt)
                if sample_path.exists():
                    environment = restore(before_path, environment)
                    sample = ActionSample.from_state_dict(torch.load(sample_path, map_location="cpu", weights_only=True))
                    actor._validate_sample_numerics(sample)
                else:
                    snapshot = actor_snapshot(components, environment)
                    atomic_json(before_path, environment.snapshot())
                    atomic_torch_save(snapshot_path, snapshot.state_dict())
                    sample = actor.sample(snapshot, seed, f"public-plan-candidates:updates-{arguments.policy_updates}")
                    atomic_torch_save(sample_path, sample.state_dict())
                raw_path.write_text(sample.text)
                row = {
                    "sample_id": task.task_id, "task_id": task.task_id, "family_id": task.family_id,
                    "attempt": attempt, "seed": seed, "raw_action": sample.text,
                    "raw_action_path": str(raw_path), "action_sample_path": str(sample_path),
                    "snapshot_path": str(snapshot_path), "before_environment_path": str(before_path),
                    "after_environment_path": str(after_path), "teacher": identity,
                    "plan_contract_succeeded": False, "bundle_compiled": False,
                    "judge_calibrated": False, "ai_review": "pending", "target_compatibility_review": "pending",
                    "eligible_for_training": False, "source_image_id": task.user_request.source_image_id,
                }
                parsed = None
                try:
                    parsed = parse_action(environment.resolve_action(sample.text))
                    row["parsed_action"] = parsed.model_dump(mode="json")
                except ContractError as error:
                    row["parse_error"] = {"code": error.code, "message": str(error)}
                if isinstance(parsed, PlanAction):
                    progress("loading_public_plan_execution", task.task_id)
                    components.load_encoders()
                    components.load_judge(private=False)
                    environment.compiler = components.compiler
                    environment.observer = components.observer
                    components.record_memory("public_plan_services_loaded")
                    if torch.cuda.max_memory_reserved(actor.device) / 2**30 > 42:
                        raise RuntimeError("Public PLAN collection exceeded its 42 GiB actor GPU ceiling")
                try:
                    result = environment.step(sample.text, seed)
                    row["environment_result"] = result.model_dump(mode="json")
                    if isinstance(parsed, PlanAction) and result.error is None and environment.state.plan is not None:
                        row["plan_contract_succeeded"] = True
                        row["internal_plan"] = environment.state.plan.model_dump(mode="json")
                        bundle = environment._compile(environment.state.plan, task.user_request.source_image_id)
                        if bundle.artifact_bindings.get("plan") != row["internal_plan"]:
                            raise RuntimeError("Compiled bundle differs from the actually executed actor PLAN")
                        if bundle.artifact_bindings.get("source_image_id") != task.user_request.source_image_id:
                            raise RuntimeError("Compiled bundle changed the original public source")
                        bundle_path, recipe_path = components.compiler._paths(bundle.bundle_id)
                        row.update(bundle_compiled=True, bundle_id=bundle.bundle_id, bundle_path=str(bundle_path), bundle_recipe_path=str(recipe_path))
                except ContractError as error:
                    row["execution_error"] = {"code": error.code, "message": str(error)}
                row["raw_public_observations"] = [observation.model_dump(mode="json") for observation in environment.state.observations.values()]
                row["accessed_public_artifacts"] = sorted(components.artifacts.accessed)
                atomic_json(after_path, environment.snapshot())
                atomic_json(result_path, row)
                attempts.append(row)
                if row["plan_contract_succeeded"] or environment.state.done:
                    break
            record = {
                "sample_id": task.task_id, "task_id": task.task_id, "family_id": task.family_id,
                "teacher": identity, "attempts": attempts,
                "compiled_actor_plan": any(row["plan_contract_succeeded"] and row["bundle_compiled"] for row in attempts),
                "ai_review": "pending", "target_compatibility_review": "pending", "eligible_for_training": False,
                "judge_calibrated": False,
            }
            atomic_json(task_result, record)
            records.append(record)
            write_jsonl(output / "results.jsonl", records)
            progress("collecting")
        progress("complete")
        write_jsonl(output / "results.jsonl", records)
    finally:
        components.release_frozen(encoders=True, judge=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", default="runs/public-plan-candidates")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--policy-updates", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--pid-file", default="deployment/public-plan-candidates.pid")
    parser.add_argument("--exit-file", default="deployment/public-plan-candidates.exit.json")
    arguments = parser.parse_args()
    Path(arguments.pid_file).write_text(str(os.getpid()) + "\n")
    status = {"pid": os.getpid(), "exit_code": 1}
    try:
        collect(arguments)
        status["exit_code"] = 0
    except BaseException as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        raise
    finally:
        atomic_json(arguments.exit_file, status)


if __name__ == "__main__":
    main()
