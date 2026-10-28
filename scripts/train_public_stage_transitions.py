import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collect_cached_public_trajectories import CachedCompiler, CachedRenderBridge, read_json
from agentic_image.actor.records import ActionSample
from agentic_image.config import load_config
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import PlanSpec, StepResult
from agentic_image.data.audit import require_pair_audit
from agentic_image.data.io import read_jsonl, write_jsonl
from agentic_image.data.paired import actor_snapshot, save_action_record
from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json, atomic_torch_save
from agentic_image.render.types import ConditionBundle
from agentic_image.runtime.components import Components
from agentic_image.runtime.prompt import SYSTEM_PROMPT
from agentic_image.runtime.training import train_actor_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--initial-render-root", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--plan-records", required=True)
    parser.add_argument("--recovery-records", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_root = Path(args.audit_root).resolve()
    audit = require_pair_audit(audit_root / "audit.json", review_mode="ai_exploratory")
    if audit.get("scope") != "original_request_global_pair_v1" or audit.get("reviewed") != 8:
        raise ValueError("This transition pilot is fixed to the accepted eight-family original-request cohort")
    pairs = list(read_jsonl(audit_root / "renderer.jsonl"))
    approved = {}
    review_dimensions = ("task_completion", "mask", "identity", "registration", "original_request_coverage", "plan_compatible")
    reviews = {}
    for row in read_jsonl(audit_root / "ai_reviews.jsonl"):
        if row.get("scope") != audit["scope"] or row.get("review_type") != "ai" or row.get("human_reviewed") is not False or not row.get("reviewer") or not row.get("evidence") or row.get("accepted") is False or not all(row.get("ratings", {}).get(name) is True for name in review_dimensions):
            continue
        key = (row["sample_id"], row["review_revision"])
        if key in reviews:
            raise ValueError(f"Duplicate accepted review binding: {key}")
        reviews[key] = row
    for pair in pairs:
        provenance = pair.get("verification_provenance", {})
        review = reviews.get((pair["sample_id"], provenance.get("review_revision")))
        if not pair.get("verified") or provenance.get("review_type") != "ai" or provenance.get("human_reviewed") is not False or review is None or review["family_id"] != pair["family_id"] or review["reviewer"] != provenance.get("reviewer"):
            raise ValueError(f"The admitted family lacks its exact accepted AI review: {pair['sample_id']}")
        if PlanSpec.model_validate(review["global_plan"]) != PlanSpec.model_validate(pair["condition_recipe"]["plan"]):
            raise ValueError(f"The admitted PLAN differs from its reviewed PLAN: {pair['sample_id']}")
        approved[pair["sample_id"]] = pair
    if len(approved) != 8 or len({row["family_id"] for row in pairs}) != 8:
        raise ValueError("The admitted cohort must contain exactly eight distinct families")
    families = {row["family_id"] for row in pairs}
    policy = Path(args.policy).resolve()
    weight = policy / "adapter_model.safetensors"
    registration = {
        "scope": "partial_public_protocol_stage_transition_sft_v1",
        "audit_scope": audit["scope"], "audit_path": str(audit_root / "audit.json"),
        "audit_sha256": hashlib.sha256((audit_root / "audit.json").read_bytes()).hexdigest(),
        "families": sorted(families), "initial_policy": str(policy),
        "initial_policy_bytes": weight.stat().st_size, "initial_policy_mtime_ns": weight.stat().st_mtime_ns,
        "phase_counts": [1, 0, 0, 3], "actor_batch": 4, "updates": args.updates,
        "independent_families": len(families), "planned_record_exposures": 4 * args.updates,
        "planned_phase_exposures": {"0": args.updates, "3": 3 * args.updates},
        "stage": "protocol", "loss": "mean_of_four_token_normalized_action_CE_losses",
        "phase_semantics": {"0": "reviewed_original_request_plan", "3": "executed_recovery_from_actual_public_contract_errors"},
        "phase1_verified_repairs": 0, "phase2_verified_finish_or_rollback": 0,
        "visual_success_supervision": False, "full_four_phase_actor_sft": False,
        "review_type": "ai", "human_reviewed": False, "judge_calibrated": False,
    }
    metadata_path = output / "registration.json"
    if metadata_path.exists() and read_json(metadata_path) != registration:
        raise ValueError("The transition dataset registration changed")
    atomic_json(metadata_path, registration)
    config = load_config(args.config)
    config["run_dir"] = args.run_dir
    config["review_mode"] = "ai_exploratory"
    config["models"].update(actor_policy=str(policy), actor_reference=None)
    config["judge"]["calibration"] = None
    config["devices"].update(actor=args.device, judge=args.device, encoders=args.device, renderer=args.device)
    config["sft"].update(actor_batch=4, protocol_phase_counts=[1, 0, 0, 3], protocol_updates=args.updates)
    config["data"].update(actor_records=str(output / "actor_records.jsonl"), paired_audit=str(audit_root / "audit.json"))
    config["pilot"]["scope"] = registration["scope"]
    (output / "training.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    components = Components(config)
    actor = components.load_actor()
    if not (output / "completed.json").is_file():
        components.load_judge(private=False)
        tokenizer = actor.processor.tokenizer
        count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        budget = config["actor"]["max_input_tokens"] - count(SYSTEM_PROMPT) - 128
        records = []
        original_plan_records = [row for row in read_jsonl(args.plan_records) if row["family_id"] in families]
        original_recovery_records = [row for row in read_jsonl(args.recovery_records) if row["family_id"] in families]
        if len(original_plan_records) != 8 or len(original_recovery_records) != 16:
            raise ValueError("The fixed cohort needs eight retained PLAN and sixteen original recovery records")
        for category, rows, phase in (("retained_original_plan", original_plan_records, 0), ("retained_original_recovery", original_recovery_records, 3)):
            for row in rows:
                if not row.get("verified") or row["phase"] != phase or not Path(row["snapshot_path"]).is_file() or row["target_ids"][-1] != 248046:
                    raise ValueError("A retained action record lacks its original verified snapshot or EOS")
                records.append({**row, "transition_bucket": category})
        rendered = []
        for folder in sorted(Path(args.initial_render_root).glob("shard-*")):
            if read_json(folder / "summary.json")["status"] != "completed":
                raise ValueError("The actual initial renders are incomplete")
            rendered.extend(read_json(folder / "results.json"))
        selected = [row for row in sorted(rendered, key=lambda row: row["index"]) if row["task_id"] in approved]
        if len(selected) != 8:
            raise ValueError("Every reviewed family must retain its actual initial render")
        evidence = []
        started = time.monotonic()

        def verified_record(environment, action, kind, state_id, provenance):
            snapshot = actor_snapshot(components, environment)
            before = environment.snapshot()
            raw = json.dumps(environment.public_action(action), ensure_ascii=True, separators=(",", ":"))
            record_id = f"stage-transition:{environment.task.task_id}:{kind}"
            stem = output / "evidence" / record_id.replace(":", "_")
            stem.parent.mkdir(exist_ok=True)
            atomic_json(stem.with_suffix(".before.json"), before)
            result = environment.step(raw, 11)
            if result.error is not None or result.done or result.protocol_failure:
                raise RuntimeError(f"The actual public teacher action failed: {result.model_dump(mode='json')}")
            if action["kind"] == "READ_NODES":
                plans = [row for row in result.output.get("records", []) if row["id"] == "plan"]
                if not plans or plans[0]["value"]["typed_parameters"]["global_text"] != environment.task.user_request.text:
                    raise RuntimeError("The real public read did not retrieve the fixed original-request plan")
                if environment.state.budget.reads != before["budget"]["reads"] + 1:
                    raise RuntimeError("The real public read did not increment its actual budget")
            elif result.render is None or not result.render.metadata.get("cache_hit"):
                raise RuntimeError("The initial RENDER must execute against the strictly matched existing trace")
            atomic_json(stem.with_suffix(".after.json"), environment.snapshot())
            provenance = {**provenance, "scope": registration["scope"], "verification": "actual_environment_step_public_contract_only", "teacher_origin": "deterministic_public_contract_transition", "actor_sampled": False, "visual_success_label": False, "repair_success_label": False, "judge_calibrated": False, "human_reviewed": False, "pre_action_state": str(stem.with_suffix(".before.json")), "post_action_state": str(stem.with_suffix(".after.json"))}
            record = save_action_record(components, environment, raw, output / "snapshots", record_id, 3, state_id, provenance, snapshot)
            record.update(task_id=environment.task.task_id, split_id="train", transition_bucket=kind, target_action=json.loads(raw))
            records.append(record)
            evidence.append({"record_id": record_id, "task_id": environment.task.task_id, "family_id": environment.task.family_id, "action": json.loads(raw), "actual_result": result.model_dump(mode="json"), "snapshot_path": record["snapshot_path"], "provenance": provenance})
            write_jsonl(output / "actor_records.jsonl", records)
            write_jsonl(output / "execution_evidence.jsonl", evidence)

        for initial in selected:
            pair = approved[initial["task_id"]]
            if pair["family_id"] != initial["family_id"] or initial["plan"] != PlanSpec(global_text=initial["request"]).model_dump(mode="json"):
                raise ValueError("The actual sampled PLAN differs from the admitted original-request scope")
            bundle = ConditionBundle.load(initial["bundle_path"])
            compiler = CachedCompiler(components, bundle)
            renderer = CachedRenderBridge(components, initial, 0)

            def restore(path):
                return EpisodeEnvironment.restore(read_json(path), components.artifacts, compiler, renderer, components.observer, compiler.load, token_counter=count, max_input_tokens=budget)

            environment = restore(initial["initial_unrendered_state"])
            provenance = {"paired_review": pair["verification_provenance"], "audit_path": registration["audit_path"], "original_actual_plan_state": initial["initial_unrendered_state"], "original_render_trace": initial["result"]["trajectory_id"]}
            attempt = output / "actual_after_plan_samples" / f'{initial["index"]:03d}'
            attempt.parent.mkdir(exist_ok=True)
            sample_seed = stream_seed(11, "actual-after-plan-transition", environment.task.task_id)
            if attempt.with_suffix(".json").exists():
                saved = read_json(attempt.with_suffix(".json"))
                outcome = StepResult.model_validate(saved["result"])
                environment = restore(attempt.with_suffix(".after.json"))
                renderer.events = saved["render_events"]
                renderer.cached_used = bool(renderer.events)
            else:
                if attempt.with_suffix(".action.pt").exists():
                    environment = restore(attempt.with_suffix(".before.json"))
                    sample = ActionSample.from_state_dict(torch.load(attempt.with_suffix(".action.pt"), map_location="cpu", weights_only=True))
                    actor._validate_sample_numerics(sample)
                else:
                    snapshot = actor_snapshot(components, environment)
                    atomic_torch_save(attempt.with_suffix(".snapshot.pt"), snapshot.state_dict())
                    atomic_json(attempt.with_suffix(".before.json"), environment.snapshot())
                    sample = actor.sample(snapshot, sample_seed, "stage-transition-behavior:updates40")
                    atomic_torch_save(attempt.with_suffix(".action.pt"), sample.state_dict())
                attempt.with_suffix(".raw.txt").write_text(sample.text)
                outcome = environment.step(sample.text, 11)
                atomic_json(attempt.with_suffix(".after.json"), environment.snapshot())
                atomic_json(attempt.with_suffix(".json"), {"actor_sampled": True, "sample_seed": sample_seed, "environment_seed": 11, "raw_action": sample.text, "result": outcome.model_dump(mode="json"), "render_events": renderer.events, "supervised": False})
            if outcome.error is not None and outcome.error.code == "PLAN_ALREADY_EXISTS" and not outcome.done and environment.state.budget.invalid == 1:
                action = {"kind": "RENDER", "control_set_id": environment.state.manifest.active_control_set_id, "expected_revision": environment.state.manifest.revision_id, "reason": "The plan already exists; render it now."}
                verified_record(environment, action, "render_after_actual_plan_error", f'{environment.episode_id}:actual-after-plan-error', {**provenance, "failure_code": "PLAN_ALREADY_EXISTS", "original_failed_step": str(attempt.with_suffix(".json")), "original_failed_sample": str(attempt.with_suffix(".action.pt")), "error_was_actual_current_policy_sample": True})
            read = {"kind": "READ_NODES", "node_ids": ["plan"], "fields": ["typed_parameters"], "reason": "Read the current plan before choosing the next action."}
            trajectory = Path(args.trajectories) / "tasks" / f'{initial["index"]:03d}'
            failed_step = trajectory / "action_00.json"
            if failed_step.is_file():
                failure = read_json(failed_step)
                if failure["result"].get("error", {}).get("code") != "PLAN_ALREADY_EXISTS":
                    raise ValueError("The retained real failure differs from the intended recovery case")
                environment = restore(failure["after_state"])
                if environment.state.done or environment.state.budget.invalid != 1:
                    raise ValueError("The real failed action state is not recoverable")
                recovery = {**read, "reason": "Read the existing plan after PLAN_ALREADY_EXISTS."}
                verified_record(environment, recovery, "recover_repeated_plan", f'{environment.episode_id}:actual-first-plan-error', {**provenance, "original_failed_step": str(failed_step.resolve()), "original_failed_sample": failure["action_sample"], "failure_code": "PLAN_ALREADY_EXISTS"})
            report = {"stage": "building_public_transitions", "records": len(records), "buckets": dict(Counter(row["transition_bucket"] for row in records)), "seconds": time.monotonic() - started}
            atomic_json(output / "progress.json", report)
            print(json.dumps(report), flush=True)
        counts = Counter(row["transition_bucket"] for row in records)
        if counts["retained_original_plan"] != 8 or counts["retained_original_recovery"] != 16 or counts["recover_repeated_plan"] != 4 or not 2 <= counts["render_after_actual_plan_error"] <= 8:
            raise ValueError(f"The fixed transition dataset is incomplete: {dict(counts)}")
        report = {**registration, "records": len(records), "record_buckets": dict(counts), "record_phase_counts": dict(Counter(row["phase"] for row in records)), "actual_new_teacher_steps": len(evidence), "all_new_targets_end_with_eos": all(row["target_ids"][-1] == 248046 for row in records), "numerics_profile": actor.config.numerics_profile, "status": "completed"}
        atomic_json(output / "audit.json", report)
        atomic_json(output / "completed.json", {"status": "completed", "records": len(records), "audit": str(output / "audit.json")})
    components.release_frozen(encoders=True, judge=True)
    if args.train:
        print(json.dumps({"stage": "starting_transition_sft", "updates": args.updates, "batch": 4, "quotas": [1, 0, 0, 3], "initial_policy": str(policy), "independent_families": len(families), "planned_record_exposures": 4 * args.updates}), flush=True)
        result = train_actor_sft(components, stage="protocol", updates=args.updates, resume=args.resume)
        metrics = list(read_jsonl(Path(args.run_dir) / "sft_protocol" / "metrics.jsonl"))
        steps = {row["step"]: row for row in metrics}
        if set(steps) != set(range(1, args.updates + 1)):
            raise ValueError("Committed training metrics do not cover the requested updates")
        exposures = Counter(record_id for row in steps.values() for record_id in row["record_ids"])
        trained_families = {family_id for row in steps.values() for family_id in row["family_ids"]}
        atomic_json(output / "training_completed.json", {"status": "completed", "policy": result, "updates": args.updates, "scope": registration["scope"], "independent_families": len(families), "actually_exposed_families": len(trained_families), "unique_exposed_records": len(exposures), "record_exposures": sum(exposures.values()), "repeat_record_exposures": sum(exposures.values()) - len(exposures), "exposures_by_record": dict(exposures), "metrics_path": str((Path(args.run_dir) / "sft_protocol" / "metrics.jsonl").resolve())})


if __name__ == "__main__":
    main()
