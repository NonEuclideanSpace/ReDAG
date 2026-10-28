import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from peft import load_peft_weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.actor.config import adapter_directory
from agentic_image.actor.head import head_log_probs
from agentic_image.actor.precision import actor_matmul_precision
from agentic_image.actor.records import ActionSample, ActorSnapshot
from agentic_image.config import load_config
from agentic_image.core.protocol import PlanAction, parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import ContractError, PlanSpec
from agentic_image.data.io import read_jsonl, write_jsonl
from agentic_image.data.paired import actor_snapshot, public_environment
from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json, atomic_torch_save
from agentic_image.runtime.components import Components
from agentic_image.runtime.rollout import read_tasks
from collect_public_plan_candidates import PublicArtifacts


def policy_identity(path):
    path = adapter_directory(path).resolve()
    weight = path / "adapter_model.safetensors"
    return {"path": str(path), "bytes": weight.stat().st_size, "mtime_ns": weight.stat().st_mtime_ns}


@torch.no_grad()
def score(actor, snapshot, target_ids, sample):
    with actor.lock, actor_matmul_precision():
        actor._set_adapter("policy", training=False)
        tokens = torch.tensor(target_ids, device=actor.device, dtype=torch.long)
        hidden = actor._hidden(snapshot, tokens)
        target = head_log_probs(hidden, actor.base.lm_head, tokens, actor.config.temperature, actor.config.head_chunk_size)
    replay = actor.replay_log_probs(sample)
    differences = (replay - sample.behavior_log_probs).abs().float()
    probes = actor.distribution_probe(sample, positions=tuple(sample.probe_logits))
    kls = []
    for position, logits in sample.probe_logits.items():
        behavior = F.log_softmax(logits.float(), dim=-1)
        kls.append(float((behavior.exp() * (behavior - probes[position])).sum()))
    audit = {"mean": float(differences.mean()), "p99": float(torch.quantile(differences, 0.99)), "max": float(differences.max()), "max_probe_kl": max(kls, default=0.0)}
    audit["passed"] = audit["mean"] <= 0.01 and audit["p99"] <= 0.05 and audit["max_probe_kl"] <= 1e-3
    return {"mean_nll": float(-target.mean()), "sum_nll": float(-target.sum()), "tokens": len(target_ids)}, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot-small.yaml")
    parser.add_argument("--tasks", default="data/public/protocol_eval_16_tasks.jsonl")
    parser.add_argument("--training-tasks", default="data/public/protocol_pilot_tasks.jsonl")
    parser.add_argument("--before-policy", default="runs/protocol-warmup/sft_protocol/policy/policy")
    parser.add_argument("--after-policy", default="runs/plan-global-ai/sft_protocol/policy/policy")
    parser.add_argument("--output", default="runs/plan-global-evaluation")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=101)
    arguments = parser.parse_args()
    tasks = read_tasks(arguments.tasks)
    training = read_tasks(arguments.training_tasks)
    if len(tasks) != 16 or len({task.family_id for task in tasks}) != 16 or any(task.split_id != "official-dev" for task in tasks):
        raise ValueError("PLAN evaluation requires the fixed 16 official-dev families")
    if {task.family_id for task in tasks} & {task.family_id for task in training}:
        raise ValueError("Evaluation and training families overlap")
    if {task.user_request.source_image_id for task in tasks} & {task.user_request.source_image_id for task in training}:
        raise ValueError("Evaluation and training public sources overlap")
    config = load_config(arguments.config)
    if not 1 <= arguments.max_new_tokens <= config["actor"]["max_new_tokens"]:
        raise ValueError("The evaluation token limit exceeds the actor profile")
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identities = {"before": policy_identity(arguments.before_policy), "after": policy_identity(arguments.after_policy)}
    config["run_dir"] = str(output)
    config["models"]["actor_policy"] = identities["before"]["path"]
    config["models"]["actor_reference"] = None
    config["judge"]["calibration"] = None
    config["actor"]["max_new_tokens"] = arguments.max_new_tokens
    for name in ("actor", "encoders", "judge"):
        config["devices"][name] = arguments.device
    components = Components(config)
    components.artifacts = PublicArtifacts(config["data"]["artifacts"])
    components.record_memory("before_actor")
    actor = components.load_actor()
    registration = {
        "scope": "heldout_public_plan_contract_v1", "tasks": [task.model_dump(mode="json") for task in tasks],
        "policies": identities, "numerics_profile": actor.config.numerics_profile,
        "max_samples_per_task_per_policy": 1, "max_new_tokens": arguments.max_new_tokens,
        "seed": arguments.seed, "optimizer_updates": 0, "private_rewards": 0,
        "visual_quality_evaluation": False, "judge_calibrated": False,
        "target_label_origin": "deterministic_original_request_plan",
    }
    registration_path = output / "registration.json"
    if registration_path.exists() and json.loads(registration_path.read_text()) != registration:
        raise ValueError("Use a fresh output for a different registered evaluation")
    atomic_json(registration_path, registration)
    result_path = output / "results.jsonl"
    results = {(row["policy"], row["task_id"]): row for row in read_jsonl(result_path)} if result_path.exists() else {}
    started = time.monotonic()
    baseline_state = actor.adapter_state()["policy"]
    try:
        for label in ("before", "after"):
            if label == "after":
                trained = load_peft_weights(identities[label]["path"], device="cpu", local_files_only=True)
                if trained.keys() != baseline_state.keys() or any(trained[key].shape != baseline_state[key].shape for key in trained):
                    raise ValueError("The trained policy does not match the original adapter structure")
                actor.restore_adapter_state({"policy": trained})
                loaded = actor.adapter_state()["policy"]
                if any(not torch.equal(loaded[key], trained[key]) for key in trained):
                    raise RuntimeError("The trained adapter was not restored exactly")
                del trained, loaded
                gc.collect()
            for index, task in enumerate(tasks):
                if (label, task.task_id) in results:
                    continue
                tick = time.monotonic()
                components.artifacts.allowed = {task.user_request.source_image_id, *task.user_request.appearance_image_ids}
                folder = output / "tasks" / f"{index:03d}"
                folder.mkdir(parents=True, exist_ok=True)
                template = public_environment(components, task, f"plan-eval:{index:03d}")
                before_path = folder / "before.json"
                snapshot_path = folder / "snapshot.pt"
                if not before_path.exists():
                    snapshot = actor_snapshot(components, template)
                    atomic_json(before_path, template.snapshot())
                    atomic_torch_save(snapshot_path, snapshot.state_dict())
                else:
                    snapshot = ActorSnapshot.load(snapshot_path)
                environment = EpisodeEnvironment.restore(
                    json.loads(before_path.read_text()), components.artifacts, components.compiler, None, components.observer,
                    components.compiler.load if components.compiler else None,
                    token_counter=template.token_counter, max_input_tokens=template.view_builder.max_input_tokens,
                )
                expected = PlanSpec(global_text=task.user_request.text)
                target_action = environment.public_action({"kind": "PLAN", "plan": expected.model_dump(mode="json")})
                target_ids = actor.processor.tokenizer.encode(json.dumps(target_action, ensure_ascii=True, separators=(",", ":")), add_special_tokens=False) + [actor.config.action_eos_token_id]
                seed = stream_seed(arguments.seed, "heldout-plan", task.task_id)
                sample_path = folder / f"{label}.action.pt"
                if sample_path.exists():
                    sample = ActionSample.from_state_dict(torch.load(sample_path, map_location="cpu", weights_only=True))
                    actor._validate_sample_numerics(sample)
                else:
                    sample = actor.sample(snapshot, seed, f"heldout-plan:{label}", probe_positions=(0, 15, 31))
                    atomic_torch_save(sample_path, sample.state_dict())
                (folder / f"{label}.raw.txt").write_text(sample.text)
                likelihood, replay = score(actor, snapshot, target_ids, sample)
                row = {
                    "policy": label, "task_id": task.task_id, "family_id": task.family_id,
                    "seed": seed, "raw_action": sample.text, "action_sample_path": str(sample_path),
                    "snapshot_path": str(snapshot_path), "finish_reason": sample.finish_reason, "sampled_tokens": sample.token_ids.numel(),
                    "request_copy_likelihood": likelihood, "replay": replay,
                    "parsed_plan": False, "contract_succeeded": False, "bundle_compiled": False, "exact_original_request_plan": False,
                    "judge_calibrated": False, "visual_success_label": None,
                }
                parsed = None
                try:
                    parsed = parse_action(environment.resolve_action(sample.text))
                    row["parsed_kind"] = parsed.kind
                    row["parsed_plan"] = isinstance(parsed, PlanAction)
                except ContractError as error:
                    row["parse_error"] = {"code": error.code, "message": str(error)}
                if isinstance(parsed, PlanAction):
                    components.load_encoders()
                    components.load_judge(private=False)
                    environment.compiler = components.compiler
                    environment.observer = components.observer
                    components.record_memory("public_plan_evaluation")
                    if torch.cuda.max_memory_reserved(actor.device) / 2**30 > 42:
                        raise RuntimeError("Public PLAN evaluation exceeded its 42 GiB GPU ceiling")
                try:
                    result = environment.step(sample.text, seed)
                    row["environment_result"] = result.model_dump(mode="json")
                    if isinstance(parsed, PlanAction) and result.error is None and environment.state.plan is not None:
                        row["contract_succeeded"] = True
                        row["internal_plan"] = environment.state.plan.model_dump(mode="json")
                        row["exact_original_request_plan"] = environment.state.plan == expected
                        bundle = environment._compile(environment.state.plan, task.user_request.source_image_id)
                        if bundle.artifact_bindings.get("plan") != row["internal_plan"]:
                            raise RuntimeError("The compiled bundle changed the actual PLAN")
                        if bundle.artifact_bindings.get("source_image_id") != task.user_request.source_image_id:
                            raise RuntimeError("The compiled bundle changed the original public source")
                        bundle_path, recipe = components.compiler._paths(bundle.bundle_id)
                        row.update(bundle_compiled=True, bundle_id=bundle.bundle_id, bundle_path=str(bundle_path), bundle_recipe_path=str(recipe))
                except ContractError as error:
                    row["execution_error"] = {"code": error.code, "message": str(error)}
                row["raw_public_observations"] = [value.model_dump(mode="json") for value in environment.state.observations.values()]
                row["seconds"] = time.monotonic() - tick
                atomic_json(folder / f"{label}.after.json", environment.snapshot())
                results[label, task.task_id] = row
                write_jsonl(result_path, results.values())
                progress = {"policy": label, "completed": sum(key[0] == label for key in results), "total": 16, "last_task": task.task_id, "compiled": row["bundle_compiled"], "replay_passed": replay["passed"]}
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        summary = {"status": "complete", "scope": registration["scope"], "families": 16, "optimizer_updates": 0, "visual_quality_evaluation": False, "judge_calibrated": False, "policies": {}}
        for label in ("before", "after"):
            rows = [row for row in results.values() if row["policy"] == label]
            summary["policies"][label] = {
                "samples": len(rows), "parsed_plans": sum(row["parsed_plan"] for row in rows),
                "contract_successes": sum(row["contract_succeeded"] for row in rows),
                "compiled_plans": sum(row["bundle_compiled"] for row in rows),
                "exact_original_request_plans": sum(row["exact_original_request_plan"] for row in rows),
                "mean_request_copy_nll": sum(row["request_copy_likelihood"]["mean_nll"] for row in rows) / len(rows),
                "replay_passed": all(row["replay"]["passed"] for row in rows),
                "maximum_replay_p99": max(row["replay"]["p99"] for row in rows),
                "action_kinds": dict(Counter(row.get("parsed_kind", "invalid") for row in rows)),
            }
        summary["after_ready_for_public_collection"] = summary["policies"]["after"]["compiled_plans"] > 0 and summary["policies"]["after"]["replay_passed"]
        summary["seconds"] = time.monotonic() - started
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "completed.json", {"status": "complete", "summary": str(output / "summary.json")})
        print(json.dumps(summary), flush=True)
    finally:
        components.release_frozen(encoders=True, judge=True)


if __name__ == "__main__":
    main()
