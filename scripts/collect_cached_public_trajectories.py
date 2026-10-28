import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.actor.records import ActionSample
from agentic_image.config import load_config
from agentic_image.core.protocol import parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import ContractError, RenderResult
from agentic_image.data.audit import require_pair_audit
from agentic_image.data.paired import actor_snapshot
from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json, atomic_torch_save
from agentic_image.render.types import ConditionBundle, RenderTrace
from agentic_image.runtime.components import Components
from agentic_image.runtime.prompt import SYSTEM_PROMPT


def read_json(path):
    return json.loads(Path(path).read_text())


class CachedCompiler:
    def __init__(self, components, bundle):
        self.components = components
        self.bundle = bundle

    def compile(self, plan, source_id=None):
        bindings = self.bundle.artifact_bindings
        if plan.model_dump(mode="json") == bindings["plan"] and source_id == bindings["source_image_id"]:
            return self.bundle
        self.components.load_encoders()
        return self.components.compiler.compile(plan, source_id)

    def load(self, bundle_id):
        if bundle_id == self.bundle.bundle_id:
            return self.bundle
        self.components.load_encoders()
        return self.components.compiler.load(bundle_id)

    def __getattr__(self, name):
        self.components.load_encoders()
        return getattr(self.components.compiler, name)


class CachedRenderBridge:
    def __init__(self, components, initial, maximum):
        self.components = components
        self.initial = initial
        self.maximum = maximum
        self.cached_used = False
        self.extra_used = 0
        self.events = []

    def render(self, bundle, seed, job_id):
        if not self.cached_used:
            cached = RenderResult.model_validate(self.initial["result"])
            if bundle.bundle_id != cached.bundle_id or seed != cached.metadata["seed"] or cached.metadata["behavior_version"] != "renderer-global-ai:step20":
                raise RuntimeError("The initial cached render identity differs from the real environment request")
            trace = RenderTrace.load(cached.trajectory_id)
            if trace.bundle_id != bundle.bundle_id or trace.behavior_version != cached.metadata["behavior_version"] or trace.steps != 50 or trace.eta != 0.3 or trace.cfg_scale != 4.0:
                raise RuntimeError("The original cached trace has incompatible conditions or renderer behavior")
            event = {"kind": "initial_cache_hit", "requested_job_id": job_id, "original_render_id": trace.render_id, "trace": cached.trajectory_id, "image_id": cached.image_id, "bundle_id": bundle.bundle_id, "seed": seed, "behavior_version": trace.behavior_version, "new_generation": False}
            self.events.append(event)
            self.cached_used = True
            return cached.model_copy(update={"metadata": {**cached.metadata, "cache_hit": True, "cache_source_render_id": trace.render_id, "requested_environment_job_id": job_id, "new_generation": False}})
        if self.extra_used >= self.maximum:
            raise ContractError("COLLECTION_RENDER_LIMIT", "The bounded collection permits only one additional render per family")
        self.components.load_renderer()
        bridge = self.components.render_bridge
        bridge.behavior_version = "renderer-global-ai:step20"
        result_file = bridge.directory / job_id.replace(":", "_").replace("/", "_") / "result.json"
        existed = result_file.exists()
        result = bridge.render(bundle, seed, job_id)
        self.extra_used += 1
        self.events.append({"kind": "additional_render", "requested_job_id": job_id, "trace": result.trajectory_id, "image_id": result.image_id, "bundle_id": bundle.bundle_id, "seed": seed, "behavior_version": bridge.behavior_version, "new_generation": not existed, "resumed_existing_bridge_result": existed})
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--initial-render-root", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--actor-policy", required=True)
    parser.add_argument("--renderer-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--max-extra-renders", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.families <= 4 or not 1 <= args.max_actions <= 4 or not 0 <= args.max_extra_renders <= 1:
        raise ValueError("The initial trajectory pilot is limited to four families, four sampled actions and one additional render")
    audit_root = Path(args.audit_root).resolve()
    audit = require_pair_audit(audit_root / "audit.json", review_mode="ai_exploratory")
    if audit.get("scope") != "original_request_global_pair_v1" or audit.get("human_reviewed") is not False:
        raise ValueError("The cohort requires the explicit AI original-request global-pair audit")
    pairs = [json.loads(line) for line in (audit_root / "renderer.jsonl").read_text().splitlines()]
    reviewed = {row["sample_id"]: row for row in pairs if row.get("verified") and row.get("verification_provenance", {}).get("review_type") == "ai" and row["verification_provenance"].get("human_reviewed") is False}
    if len(reviewed) != audit["reviewed"]:
        raise ValueError("The actual verified cohort differs from the audited count")
    review_rows = [json.loads(line) for line in (audit_root / "ai_reviews.jsonl").read_text().splitlines()]
    accepted = {(row["sample_id"], row["review_revision"]): row for row in review_rows if row.get("accepted") is True and row.get("scope") == audit["scope"]}
    collection = Path(args.collection).resolve()
    collection_metadata = read_json(collection / "metadata.json")
    actor_policy = Path(args.actor_policy).resolve()
    renderer_policy = Path(args.renderer_policy).resolve()
    if Path(collection_metadata["teacher"]["policy_path"]).resolve() != actor_policy:
        raise ValueError("The continuation actor differs from the actual sampled PLAN policy")
    rendered = []
    for folder in sorted(Path(args.initial_render_root).resolve().glob("shard-*")):
        summary = read_json(folder / "summary.json")
        if summary["status"] != "completed" or Path(summary["renderer_policy"]).resolve() != renderer_policy or summary["renderer_policy_mtime_ns"] != renderer_policy.stat().st_mtime_ns:
            raise ValueError("Initial render shard or trained renderer policy identity differs")
        rendered.extend(read_json(folder / "results.json"))
    selected = [row for row in sorted(rendered, key=lambda row: row["index"]) if row["task_id"] in reviewed][:args.families]
    if len(selected) != args.families or len({row["family_id"] for row in selected}) != args.families:
        raise ValueError("The audited cohort has insufficient distinct actually rendered families")
    bindings = []
    for row in selected:
        pair = reviewed[row["task_id"]]
        provenance = pair["verification_provenance"]
        review = accepted.get((row["task_id"], provenance["review_revision"]))
        if review is None or pair["family_id"] != row["family_id"] or review["family_id"] != row["family_id"]:
            raise ValueError("The selected family lacks its exact accepted AI review")
        bindings.append({"index": row["index"], "task_id": row["task_id"], "family_id": row["family_id"], "review_revision": provenance["review_revision"], "reviewer": provenance["reviewer"]})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registration = {"scope": "unverified_actual_public_execution_after_cached_initial_render", "cohort_scope": audit["scope"], "audit_path": str(audit_root / "audit.json"), "audit_sha256": hashlib.sha256((audit_root / "audit.json").read_bytes()).hexdigest(), "selected": bindings, "actor_policy": collection_metadata["teacher"], "renderer_policy": str(renderer_policy), "max_sampled_actions_per_family": args.max_actions, "max_additional_renders_per_family": args.max_extra_renders, "judge_calibrated": False, "verified": False, "repair_success": None, "eligible_for_training": False, "trajectory_ai_review": "pending", "human_reviewed": False}
    registration_path = output / "registration.json"
    if registration_path.exists() and read_json(registration_path) != registration:
        raise ValueError("The continuation registration changed during resume")
    atomic_json(registration_path, registration)
    config = load_config(args.config)
    config["run_dir"] = str(output)
    config["models"].update(actor_policy=str(actor_policy), renderer_policy=str(renderer_policy), actor_reference=None, renderer_reference=None)
    config["judge"]["calibration"] = None
    config["renderer"].update(steps=50, eta=0.3, cfg_scale=4.0)
    for name in ("actor", "renderer", "encoders", "judge"):
        config["devices"][name] = args.device
    components = Components(config)
    actor = components.load_actor()
    components.load_judge(private=False)
    tokenizer = actor.processor.tokenizer
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    token_budget = config["actor"]["max_input_tokens"] - token_count(SYSTEM_PROMPT) - 128
    records = []
    started = time.monotonic()

    def progress(stage, task_id=None, actions=0):
        row = {"stage": stage, "completed_families": len(records), "total_families": len(selected), "task_id": task_id, "current_sampled_actions": actions, "seconds": time.monotonic() - started, "updated_at": time.time(), "verified": False, "judge_calibrated": False}
        atomic_json(output / "progress.json", row)
        print(json.dumps(row), flush=True)

    for initial in selected:
        folder = output / "tasks" / f'{initial["index"]:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            records.append(read_json(result_path))
            continue
        bundle = ConditionBundle.load(initial["bundle_path"])
        if bundle.artifact_bindings["plan"] != initial["plan"] or bundle.artifact_bindings["source_image_id"] != initial["source_image_id"]:
            raise ValueError("The initial bundle no longer matches the actual sampled PLAN and public source")
        compiler = CachedCompiler(components, bundle)
        renderer = CachedRenderBridge(components, initial, args.max_extra_renders)

        def restore(path):
            return EpisodeEnvironment.restore(read_json(path), components.artifacts, compiler, renderer, components.observer, compiler.load, token_counter=token_count, max_input_tokens=token_budget)

        initial_after = folder / "initial.after.json"
        if initial_after.exists():
            environment = restore(initial_after)
            renderer.cached_used = True
            renderer.events = read_json(folder / "initial.step.json")["render_events"]
        else:
            environment = restore(initial["initial_unrendered_state"])
            if environment.task.split_id != "train" or environment.task.family_id != initial["family_id"] or environment.state.executed_renders:
                raise ValueError("The original state is not the selected unrendered public training episode")
            snapshot = actor_snapshot(components, environment)
            atomic_torch_save(folder / "initial.snapshot.pt", snapshot.state_dict())
            atomic_json(folder / "initial.before.json", environment.snapshot())
            action = {"kind": "RENDER", "control_set_id": environment.state.manifest.active_control_set_id, "expected_revision": environment.state.manifest.revision_id, "reason": "Render the actual sampled public plan."}
            raw = json.dumps(environment.public_action(action), separators=(",", ":"))
            progress("executing_cached_initial_render", initial["task_id"])
            outcome = environment.step(raw, 11)
            if outcome.error is not None or not environment.state.executed_renders:
                raise RuntimeError(f"The real cached initial RENDER failed: {outcome.model_dump(mode='json')}")
            atomic_json(folder / "initial.step.json", {"action_origin": "explicit_public_contract_render", "raw_action": raw, "seed": 11, "result": outcome.model_dump(mode="json"), "render_events": renderer.events, "observer": "actual_uncalibrated_8b_public_observer", "actor_sampled": False, "verified": False})
            atomic_json(initial_after, environment.snapshot())
        actions = []
        for turn in range(args.max_actions):
            if environment.state.done:
                break
            stem = folder / f"action_{turn:02d}"
            step_path = stem.with_suffix(".json")
            if step_path.exists():
                saved = read_json(step_path)
                actions.append(saved)
                environment = restore(stem.with_suffix(".after.json"))
                renderer.events.extend(saved["render_events"])
                renderer.extra_used += sum(event["kind"] == "additional_render" for event in saved["render_events"])
                continue
            sample_path = stem.with_suffix(".action.pt")
            seed = stream_seed(11, "cached-public-continuation", initial["task_id"], turn)
            if sample_path.exists():
                environment = restore(stem.with_suffix(".before.json"))
                sample = ActionSample.from_state_dict(torch.load(sample_path, map_location="cpu", weights_only=True))
                actor._validate_sample_numerics(sample)
            else:
                snapshot = actor_snapshot(components, environment)
                atomic_torch_save(stem.with_suffix(".snapshot.pt"), snapshot.state_dict())
                atomic_json(stem.with_suffix(".before.json"), environment.snapshot())
                progress("sampling_actual_action", initial["task_id"], turn)
                sample = actor.sample(snapshot, seed, "actual-plan-policy:updates40")
                atomic_torch_save(sample_path, sample.state_dict())
            stem.with_suffix(".raw.txt").write_text(sample.text)
            parsed = None
            parse_error = None
            try:
                parsed = parse_action(environment.resolve_action(sample.text)).model_dump(mode="json")
            except ContractError as error:
                parse_error = {"code": error.code, "message": str(error)}
            before_events = len(renderer.events)
            outcome = environment.step(sample.text, seed)
            record = {"turn": turn, "raw_action": sample.text, "parsed_action": parsed, "parse_error": parse_error, "action_sample": str(sample_path), "snapshot": str(stem.with_suffix(".snapshot.pt")), "before_state": str(stem.with_suffix(".before.json")), "after_state": str(stem.with_suffix(".after.json")), "seed": seed, "result": outcome.model_dump(mode="json"), "render_events": renderer.events[before_events:], "verified": False, "repair_success": None, "judge_calibrated": False, "eligible_for_training": False}
            atomic_json(stem.with_suffix(".after.json"), environment.snapshot())
            atomic_json(step_path, record)
            actions.append(record)
            components.record_memory(f'continuation:{initial["index"]}:{turn}')
        record = {"task_id": initial["task_id"], "family_id": initial["family_id"], "original_initial_render": initial, "initial_execution": str(folder / "initial.step.json"), "actions": actions, "final_state": str(folder / "final.json"), "additional_renders": renderer.extra_used, "done": environment.state.done, "verified": False, "repair_success": None, "eligible_for_training": False, "judge_calibrated": False, "trajectory_ai_review": "pending"}
        atomic_json(folder / "final.json", environment.snapshot())
        atomic_json(result_path, record)
        records.append(record)
        atomic_json(output / "results.json", records)
        progress("collecting")
    atomic_json(output / "summary.json", {**registration, "status": "completed", "completed_families": len(records), "actual_sampled_actions": sum(len(row["actions"]) for row in records), "additional_renders": sum(row["additional_renders"] for row in records), "cached_initial_renders": len(records), "seconds": time.monotonic() - started})
    progress("complete")


if __name__ == "__main__":
    main()
