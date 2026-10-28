from dataclasses import asdict
from pathlib import Path

import torch

from agentic_image.actor.records import ActionSample
from agentic_image.checks.reward import dense_score
from agentic_image.core.protocol import ApplyAction, DraftRepairAction, PlanAction, parse_action
from agentic_image.core.types import ContractError, PatchOperation, PublicTask
from agentic_image.learning.sampling import stream_seed

from .io import append_jsonl, read_jsonl, write_jsonl, write_report
from .paired import actor_snapshot, public_environment, save_action_record
from .repairs import repair_is_positive
from .prerequisites import require_qualified_renderer


def collect_injections(
    components,
    tasks_path,
    specifications_path,
    output_dir,
    renderer_version,
    sampler_id,
    max_cases=256,
    repeated_cases=64,
    seed=20261027,
):
    if max_cases > 256 or repeated_cases > 64 or min(max_cases, repeated_cases) < 0:
        raise ValueError(
            "Injection budgets may not exceed the registered 256 plus 64 repeated cases"
        )
    require_qualified_renderer(components, renderer_version, sampler_id)
    components.load_all(private=True)
    components.render_bridge.behavior_version = renderer_version
    tasks = {row["task_id"]: PublicTask.model_validate(row) for row in read_jsonl(tasks_path)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "actor_records.jsonl").exists():
        raise FileExistsError("Injection collection needs a fresh output directory")
    records, bundles = [], []
    report = {
        "cases": 0,
        "states": 0,
        "normal_renders": 0,
        "bad_renders": 0,
        "recovery_renders": 0,
        "verified_recoveries": 0,
        "teacher_calls": 0,
        "failed_interventions": 0,
        "renderer_version": renderer_version,
        "sampler_id": sampler_id,
    }
    for specification in read_jsonl(specifications_path):
        if report["cases"] >= max_cases:
            break
        task = tasks[specification["task_id"]]
        if task.split_id != "train":
            continue
        clean_sample = ActionSample.from_state_dict(
            torch.load(specification["plan_action_path"], map_location="cpu", weights_only=True)
        )
        clean_action = parse_action(clean_sample.text)
        if not isinstance(clean_action, PlanAction):
            raise ValueError("A controlled intervention needs an actual public teacher PLAN record")
        if clean_sample.snapshot.metadata.get("task_id") != task.task_id:
            raise ValueError("Injection plan and task identity do not match")
        targeted = set(specification["private_target_eval_ids"])
        if not targeted:
            raise ValueError(
                "An intervention must declare its independent private evaluation targets"
            )
        case_index = report["cases"]
        report["cases"] += 1
        for repetition in range(2 if case_index < repeated_cases else 1):
            state_id = f"injection:{case_index:04d}:{repetition}"
            environment = public_environment(components, task, state_id)
            environment.view()
            result = environment.step(clean_sample.text)
            if result.error:
                raise ValueError(
                    "The saved public PLAN is no longer executable under this component version"
                )
            source = environment.state.manifest.selected_image_id
            clean_bundle = components.compiler.compile(environment.state.plan, source)
            shared_seed = stream_seed(seed, case_index, repetition, "intervention")
            normal = components.render_bridge.render(
                clean_bundle, shared_seed, f"{state_id}:normal"
            )
            report["normal_renders"] += 1
            normal_scores = components.evaluator.score_image(task, normal.image_id)
            normal_target = [score for score in normal_scores if score.atom_id in targeted]
            if len(normal_target) != len(targeted) or not all(
                score.known and score.value >= 0.8 for score in normal_target
            ):
                report["failed_interventions"] += 1
                continue
            operations = []
            for operation in specification["patch_operations"]:
                values = dict(operation)
                target = values["target_logical_node_id"]
                values["expected_old_version_id"] = (
                    None
                    if values["op"] == "add"
                    else environment.state.manifest.bindings.get(target)
                )
                operations.append(PatchOperation.model_validate(values))
            bad_plan, changed = environment._patch_plan(tuple(operations))
            intervention = environment._node(
                f"control_revision:{case_index}",
                "decision",
                "plan_revision",
                {},
                observation_parents=environment.state.last_view_node_ids,
            )
            environment.state.dag.complete(intervention)
            environment._materialize_plan(bad_plan, intervention.node_version_id)
            action = {
                "kind": "RENDER",
                "control_set_id": environment.state.manifest.active_control_set_id,
                "expected_revision": environment.state.manifest.revision_id,
            }
            result = environment.step(action, shared_seed)
            if result.error or result.render is None:
                report["failed_interventions"] += 1
                continue
            report["bad_renders"] += 1
            before_id = result.render.image_id
            before_scores = components.evaluator.score_image(task, before_id)
            bad_target = [score for score in before_scores if score.atom_id in targeted]
            if not all(score.known and score.value <= 0.6 for score in bad_target):
                report["failed_interventions"] += 1
                continue
            report["states"] += 1
            bundle = {
                "state_id": state_id,
                "family_id": task.family_id,
                "task_id": task.task_id,
                "renderer_version": renderer_version,
                "sampler_id": sampler_id,
                "normal_image_id": normal.image_id,
                "bad_image_id": before_id,
                "intervention_node_set": list(changed),
                "public_before_state": environment.snapshot(),
                "private_normal_scores": [asdict(score) for score in normal_scores],
                "private_bad_scores": [asdict(score) for score in before_scores],
                "shared_exogenous_seed": shared_seed,
                "verified": False,
            }
            bundles.append(bundle)
            draft_snapshot = actor_snapshot(components, environment)
            draft_sample = components.actor.sample(
                draft_snapshot, stream_seed(seed, state_id, "draft"), "offline-injection-recovery"
            )
            report["teacher_calls"] += 1
            try:
                draft_action = parse_action(draft_sample.text)
            except ContractError:
                continue
            if not isinstance(draft_action, DraftRepairAction) or draft_action.scope == "retry":
                continue
            result = environment.step(draft_sample.text)
            if result.error or not environment.state.candidates:
                continue
            apply_snapshot = actor_snapshot(components, environment)
            apply_sample = components.actor.sample(
                apply_snapshot, stream_seed(seed, state_id, "apply"), "offline-injection-recovery"
            )
            report["teacher_calls"] += 1
            try:
                apply_action = parse_action(apply_sample.text)
            except ContractError:
                continue
            if not isinstance(apply_action, ApplyAction):
                continue
            candidate = environment.state.candidates.get(
                environment.resolve_action(apply_sample.text)["candidate_id"]
            )
            if candidate is None or candidate.source_for_render != before_id:
                continue
            bundle["repair_node_set"] = list(candidate.changed_nodes)
            result = environment.step(apply_sample.text, stream_seed(seed, state_id, "recovery"))
            if result.error or result.render is None:
                continue
            report["recovery_renders"] += 1
            scores = components.evaluator.score_image(task, result.render.image_id)
            bundle["recovery_image_id"] = result.render.image_id
            bundle["private_recovery_scores"] = [asdict(score) for score in scores]
            target_scores = [score for score in scores if score.atom_id in targeted]
            if not all(
                score.known and score.value >= 0.8 for score in target_scores
            ) or not repair_is_positive(before_scores, scores):
                continue
            bundle["verified"] = True
            report["verified_recoveries"] += 1
            provenance = {
                "verification": "same_input_intervention_and_executed_recovery_from_bad_image",
                "intervention_node_set": list(changed),
                "repair_node_set": list(candidate.changed_nodes),
            }
            records.append(
                save_action_record(
                    components,
                    environment,
                    draft_sample.text,
                    output / "snapshots",
                    f"{state_id}:draft",
                    1,
                    state_id,
                    provenance,
                    draft_snapshot,
                    draft_sample,
                )
            )
            records.append(
                save_action_record(
                    components,
                    environment,
                    apply_sample.text,
                    output / "snapshots",
                    f"{state_id}:apply",
                    1,
                    state_id,
                    provenance,
                    apply_snapshot,
                    apply_sample,
                )
            )
            if all(score.passed for score in scores) and not environment.state.done:
                snapshot = actor_snapshot(components, environment)
                action = {"kind": "FINISH"}
                if not environment.step(action).error:
                    records.append(
                        save_action_record(
                            components,
                            environment,
                            action,
                            output / "snapshots",
                            f"{state_id}:finish",
                            2,
                            state_id,
                            {"verification": "all_requirements_pass_after_controlled_recovery"},
                            snapshot,
                        )
                    )
            write_jsonl(output / "actor_records.jsonl", records)
            write_jsonl(output / "state_bundles.jsonl", bundles)
    report["render_calls"] = sum(
        report[key] for key in ("normal_renders", "bad_renders", "recovery_renders")
    )
    report["action_records"] = len(records)
    write_jsonl(output / "actor_records.jsonl", records)
    write_jsonl(output / "state_bundles.jsonl", bundles)
    write_report(output / "report.json", report)
    return report


def collect_self_corrections(
    components,
    tasks_path,
    output_dir,
    renderer_version,
    sampler_id,
    round_index,
    max_tasks=256,
    seed=20261027,
):
    if round_index not in (0, 1) or not 1 <= max_tasks <= 256:
        raise ValueError("Self-correction uses exactly two registered rounds of at most 256 tasks")
    require_qualified_renderer(components, renderer_version, sampler_id)
    components.load_all(private=True)
    components.render_bridge.behavior_version = renderer_version
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "actor_records.jsonl").exists():
        raise FileExistsError("Self-correction collection needs a fresh round directory")
    records, trajectories = [], []
    written_records = 0
    report = {
        "round": round_index,
        "tasks": 0,
        "renders": 0,
        "actions": 0,
        "verified_records": 0,
        "renderer_version": renderer_version,
        "sampler_id": sampler_id,
    }
    for row in read_jsonl(tasks_path):
        if report["tasks"] >= max_tasks:
            break
        task = PublicTask.model_validate(row)
        if task.split_id != "train":
            continue
        if task.limits.max_renders > 3:
            raise ValueError(
                "Self-correction includes all candidate executions in a three-render task budget"
            )
        report["tasks"] += 1
        state_id = f"refresh:{round_index}:{report['tasks']:04d}"
        environment = public_environment(components, task, state_id)
        pending_draft = None
        history_scores = {}
        executed = []
        while not environment.state.done:
            before_id = environment.state.manifest.selected_image_id
            if before_id and before_id not in history_scores:
                history_scores[before_id] = components.evaluator.score_image(task, before_id)
            snapshot = actor_snapshot(components, environment)
            sample = components.actor.sample(
                snapshot,
                stream_seed(
                    seed,
                    round_index,
                    task.task_id,
                    environment.state.budget.actions,
                    "self_correction",
                ),
                f"self-correction:{round_index}",
            )
            turn = environment.state.budget.actions
            was_invalid = environment.state.budget.invalid > 0
            result = environment.step(
                sample.text, stream_seed(seed, round_index, task.task_id, turn, "render")
            )
            report["actions"] += 1
            action_path = output / "actions" / f"{report['tasks']:04d}_{turn:02d}.pt"
            action_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample.state_dict(), action_path)
            executed.append(
                {
                    "action_path": str(action_path.resolve()),
                    "result": result.model_dump(mode="json"),
                }
            )
            if result.error:
                continue
            action = parse_action(sample.text)
            selected = environment.state.manifest.selected_image_id
            if result.render:
                report["renders"] += 1
                history_scores[selected] = components.evaluator.score_image(task, selected)
            verified, phase, evidence = False, 3, ""
            if isinstance(action, DraftRepairAction):
                pending_draft = (snapshot, sample, turn, before_id)
            elif isinstance(action, ApplyAction) and before_id and selected:
                verified = repair_is_positive(history_scores[before_id], history_scores[selected])
                phase, evidence = 1, "executed_current_policy_repair_improved_fixed_before"
                if verified and pending_draft and pending_draft[3] == before_id:
                    draft_snapshot, draft_sample, draft_turn, _ = pending_draft
                    records.append(
                        save_action_record(
                            components,
                            environment,
                            draft_sample.text,
                            output / "snapshots",
                            f"{state_id}:{draft_turn}",
                            1,
                            state_id,
                            {"verification": evidence},
                            draft_snapshot,
                            draft_sample,
                        )
                    )
                pending_draft = None
            elif action.kind == "SELECT" and before_id and selected:
                if selected not in history_scores:
                    history_scores[selected] = components.evaluator.score_image(task, selected)
                after = history_scores[selected]
                before = history_scores[before_id]
                verified = all(score.known for score in (*before, *after)) and dense_score(
                    after
                ) > dense_score(before)
                phase, evidence = 2, "executed_rollback_to_higher_quality_historical_image"
            elif action.kind == "FINISH" and environment.state.delivered_image_id:
                delivered = environment.state.delivered_image_id
                if delivered not in history_scores:
                    history_scores[delivered] = components.evaluator.score_image(task, delivered)
                scores = history_scores[delivered]
                verified = (
                    all(score.passed for score in scores)
                    or environment.state.budget.renders >= task.limits.max_renders
                )
                phase, evidence = 2, "executed_success_or_render_budget_termination"
            elif (
                action.kind in {"INSPECT", "READ_NODES"}
                or isinstance(action, PlanAction)
                and was_invalid
            ):
                verified = True
                phase, evidence = 3, "executed_public_evidence_or_protocol_recovery"
            if verified:
                records.append(
                    save_action_record(
                        components,
                        environment,
                        sample.text,
                        output / "snapshots",
                        f"{state_id}:{turn}",
                        phase,
                        state_id,
                        {
                            "verification": evidence,
                            "renderer_version": renderer_version,
                            "sampler_id": sampler_id,
                        },
                        snapshot,
                        sample,
                    )
                )
        trajectories.append(
            {
                "state_id": state_id,
                "family_id": task.family_id,
                "task_id": task.task_id,
                "actions": executed,
                "final_state": environment.snapshot(),
                "private_image_scores": {
                    key: [asdict(score) for score in value] for key, value in history_scores.items()
                },
            }
        )
        append_jsonl(output / "actor_records.jsonl", records[written_records:])
        written_records = len(records)
        append_jsonl(output / "state_bundles.jsonl", trajectories[-1:])
    report["verified_records"] = len(records)
    report["unique_families"] = len({row["family_id"] for row in records})
    write_jsonl(output / "actor_records.jsonl", records)
    write_jsonl(output / "state_bundles.jsonl", trajectories)
    write_report(output / "report.json", report)
    return report
