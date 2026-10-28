from dataclasses import asdict
from collections import Counter
import json
from pathlib import Path

import torch

from agentic_image.actor.records import ActionSample
from agentic_image.checks.reward import dense_score
from agentic_image.core.protocol import ApplyAction, DraftRepairAction, PlanAction, parse_action
from agentic_image.core.types import ContractError, PatchOperation, PublicTask
from agentic_image.learning.sampling import stream_seed

from .io import append_jsonl, read_jsonl, write_jsonl, write_report
from .paired import actor_snapshot, public_environment, save_action_record
from .repairs import _restore, fixed_unknown_repair_requirements, public_rule_draft, public_rule_plan, repair_is_positive, require_calibrated_repair_evidence
from .prerequisites import require_qualified_renderer


def collect_public_rule_interventions(components, tasks_path, output_dir, renderer_version, sampler_id, max_tasks=128, seed=20261027):
    require_qualified_renderer(components, renderer_version, sampler_id)
    components.load_all(private=True)
    identity = require_calibrated_repair_evidence(components)
    components.render_bridge.behavior_version = renderer_version
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [PublicTask.model_validate(row) for row in read_jsonl(tasks_path)][:max_tasks]
    if not tasks or len({task.family_id for task in tasks}) != len(tasks) or any(task.split_id != "train" or not task.user_request.source_image_id for task in tasks):
        raise ValueError("Public rule interventions require distinct admitted training edit families")
    records, bundles = [], []
    report = {"tasks": 0, "renders": 0, "positive_repairs": 0, "observed_regressions": 0, "contract_rollbacks": 0, "budget_finishes": 0, "successful_finishes": 0, "unknown_recoveries": 0, "fixed_unknown_stops": 0, "skipped_unverifiable_renders": 0, "failed_tasks": 0, "teacher_origin": "deterministic_public_request_and_control_intervention", "actual_actor_samples": 0, "renderer_version": renderer_version, "sampler_id": sampler_id, "evidence_identity": identity}

    def persist():
        write_jsonl(output / "actor_records.jsonl", records)
        write_jsonl(output / "state_bundles.jsonl", bundles)
        report.update(action_records=len(records), phase_counts=dict(Counter(row["phase"] for row in records)), unique_families=len({row["family_id"] for row in records}))
        write_report(output / "report.json", report)
        print(json.dumps(report), flush=True)

    for index, task in enumerate(tasks):
        folder = output / "tasks" / f"{index:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        completed = folder / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["task_id"] != task.task_id or saved["family_id"] != task.family_id:
                raise ValueError("Resumed intervention task identity changed")
            if saved["counts"]["failed_tasks"]:
                raise RuntimeError("A preserved failed task cannot be silently skipped; repair it in an explicit new collection attempt")
            records.extend(read_jsonl(folder / "actor_records.jsonl"))
            bundles.append(json.loads((folder / "state_bundle.json").read_text()))
            for key, value in saved["counts"].items():
                report[key] += value
            persist()
            continue
        if (folder / "events.jsonl").exists():
            raise FileExistsError("An interrupted task has preserved evidence; use a fresh collection attempt rather than rerendering it silently")
        state_id = f"public_rule:{task.family_id.split(':', 1)[-1]}"
        fixed_unknown = fixed_unknown_repair_requirements(components.evaluator.evaluations[task.task_id], components.judge.thresholds)
        environment = public_environment(components, task, state_id)
        task_records, events = [], []
        counts = {key: 0 for key in ("tasks", "renders", "positive_repairs", "observed_regressions", "contract_rollbacks", "budget_finishes", "successful_finishes", "unknown_recoveries", "fixed_unknown_stops", "skipped_unverifiable_renders", "failed_tasks")}
        counts["tasks"] = 1
        bundle = {"state_id": state_id, "task_id": task.task_id, "family_id": task.family_id, "teacher_origin": report["teacher_origin"], "private_outcomes": [], "verified_record_ids": [], "evidence_identity": identity, "renderer_version": renderer_version, "sampler_id": sampler_id, "public_request": task.user_request.text, "public_source_image_id": task.user_request.source_image_id}

        def execute(env, action, name, rng=None):
            snapshot = actor_snapshot(components, env)
            before = env.snapshot()
            public_action = env.public_action(action)
            snapshot_path = folder / f"{name}.snapshot.pt"
            snapshot.save(snapshot_path)
            try:
                result = env.step(action, rng)
            except Exception as error:
                event = {"name": name, "public_action": public_action, "internal_action": action, "public_before_state": before, "public_after_state": env.snapshot(), "snapshot_path": str(snapshot_path.resolve()), "execution_exception": {"type": type(error).__name__, "message": str(error)}, "teacher_origin": report["teacher_origin"], "actual_actor_sample": False, "verified_for_sft": False}
                events.append(event)
                append_jsonl(folder / "events.jsonl", [event])
                raise
            event = {"name": name, "public_action": public_action, "internal_action": action, "public_before_state": before, "public_after_state": env.snapshot(), "snapshot_path": str(snapshot_path.resolve()), "result": result.model_dump(mode="json"), "teacher_origin": report["teacher_origin"], "actual_actor_sample": False, "verified_for_sft": False}
            events.append(event)
            append_jsonl(folder / "events.jsonl", [event])
            if result.render:
                counts["renders"] += 1
            return snapshot, result, event

        def admit(env, action, snapshot, name, phase, verification, **extra):
            provenance = {"verification": verification, "teacher_origin": report["teacher_origin"], "actual_actor_sample": False, "visual_success_label": False, "repair_success_label": False, "execution_evidence": str((folder / "events.jsonl").resolve()), **identity, **extra}
            record = save_action_record(components, env, action, folder / "snapshots", f"{state_id}:{name}", phase, state_id, provenance, snapshot)
            task_records.append(record)
            bundle["verified_record_ids"].append(record["record_id"])
            return record

        def require_step(result, render=False):
            if result.error or render and result.render is None:
                raise RuntimeError(f"Public teacher execution failed: {result.error.code if result.error else 'missing_render'}")

        try:
            action = public_rule_plan(task)
            snapshot, result, _ = execute(environment, action, "plan")
            require_step(result)
            admit(environment, action, snapshot, "plan", 0, "executed_original_public_request_plan")
            after_plan = environment.snapshot()
            recovery = _restore(components, after_plan, environment)
            _, invalid, _ = execute(recovery, action, "repeated_plan_error")
            if invalid.error is None or invalid.error.code != "PLAN_ALREADY_EXISTS" or recovery.state.budget.invalid != 1:
                raise RuntimeError("The intended recoverable repeated-PLAN error was not produced")
            read_action = {"kind": "READ_NODES", "node_ids": ["plan"]}
            snapshot, result, _ = execute(recovery, read_action, "error_recovery_read")
            require_step(result)
            admit(recovery, read_action, snapshot, "error_recovery_read", 3, "executed_read_after_actual_repeated_plan_error", root_error=invalid.error.model_dump(mode="json"))
            render_action = {"kind": "RENDER", "control_set_id": environment.state.manifest.active_control_set_id, "expected_revision": environment.state.manifest.revision_id}
            shared_seed = stream_seed(seed, task.task_id, "controlled_initial")
            snapshot, normal, _ = execute(environment, render_action, "initial_render", shared_seed)
            require_step(normal, True)
            normal_id = normal.render.image_id
            admit(environment, render_action, snapshot, "initial_render", 0, "executed_initial_render", render_trace=normal.render.trajectory_id, render_image_id=normal_id)
            normal_scores = components.evaluator.score_image(task, normal_id)
            bundle["private_outcomes"].append({"stage": "initial", "image_id": normal_id, "scores": [asdict(score) for score in normal_scores]})
            if fixed_unknown:
                actual_scores = {score.atom_id: score for score in normal_scores}
                if any(item["eval_id"] not in actual_scores or actual_scores[item["eval_id"]].known for item in fixed_unknown):
                    raise RuntimeError("Frozen unknown eligibility differs from the actual private evaluator output")
                if environment.state.done or counts["renders"] != 1 or len(task_records) != 3:
                    raise RuntimeError("Fixed-unknown collection must stop after exactly one render and three verified public actions")
                bundle["offline_collection_stop"] = {"reason": "fixed_private_requirements_cannot_produce_positive_repairs", "eligibility_scope": "frozen_route_confirmation_and_source_grounding_only", "requirements": fixed_unknown, "actual_initial_unknown_atom_ids": [score.atom_id for score in normal_scores if not score.known], "skipped_actions": ["injected_control_draft", "injected_control_render", "injected_contract_rollback", "recovery_draft", "recovery_render", "terminal_finish"], "skipped_render_calls": 2, "episode_done": False, "finish_executed": False, "rollback_executed": False, "family_retained": True, "private_specification_unchanged": True, "not_a_terminal_policy_action": True}
                counts["fixed_unknown_stops"] = 1
                counts["skipped_unverifiable_renders"] = 2
            else:
                if normal_scores and all(score.passed for score in normal_scores):
                    terminal = _restore(components, environment.snapshot(), environment)
                    finish = {"kind": "FINISH"}
                    snapshot, result, _ = execute(terminal, finish, "successful_initial_finish")
                    require_step(result)
                    admit(terminal, finish, snapshot, "successful_initial_finish", 2, "all_fixed_calibrated_requirements_pass", visual_success_label=True)
                    counts["successful_finishes"] += 1
                select_source = {"kind": "SELECT", "image_id": task.user_request.source_image_id}
                _, result, _ = execute(environment, select_source, "intervention_same_source")
                require_step(result)
                corrupt_text = "Keep the input image unchanged."
                if corrupt_text == task.user_request.text:
                    raise ValueError("The declared intervention would not change the public instruction")
                injected = public_rule_draft(environment, corrupt_text)
                _, result, _ = execute(environment, injected, "injected_control_draft")
                require_step(result)
                candidate = next(iter(environment.state.candidates.values()))
                bad_apply = {"kind": "APPLY", "candidate_id": candidate.candidate_id, "expected_revision": environment.state.manifest.revision_id}
                _, bad, _ = execute(environment, bad_apply, "injected_control_render", shared_seed)
                require_step(bad, True)
                bad_id = bad.render.image_id
                bad_scores = components.evaluator.score_image(task, bad_id)
                bundle["private_outcomes"].append({"stage": "injected", "image_id": bad_id, "scores": [asdict(score) for score in bad_scores], "same_exogenous_seed": shared_seed})
                regression = bool(normal_scores) and all(score.known for score in (*normal_scores, *bad_scores)) and any(score.passed and not {item.atom_id:item for item in bad_scores}[score.atom_id].passed for score in normal_scores)
                counts["observed_regressions"] += int(regression)
                rollback = _restore(components, environment.snapshot(), environment)
                rollback_action = {"kind": "SELECT", "image_id": normal_id, "reason": "Return to the image rendered with the original request after the injected control change."}
                snapshot, result, _ = execute(rollback, rollback_action, "injected_contract_rollback")
                require_step(result)
                if rollback.state.plan.global_text != task.user_request.text or rollback.state.manifest.selected_image_id != normal_id:
                    raise RuntimeError("The injected control rollback did not restore the original public plan and historical image")
                admit(rollback, rollback_action, snapshot, "injected_contract_rollback", 2, "executed_rollback_of_actual_injected_public_control", observed_visual_regression=regression, injection_action=injected)
                counts["contract_rollbacks"] += 1
                repaired_draft = public_rule_draft(environment)
                draft_snapshot, result, _ = execute(environment, repaired_draft, "recovery_draft")
                require_step(result)
                candidate = next(iter(environment.state.candidates.values()))
                if candidate.source_for_render != bad_id:
                    raise RuntimeError("The repair must use the actual bad rendered image")
                apply_action = {"kind": "APPLY", "candidate_id": candidate.candidate_id, "expected_revision": environment.state.manifest.revision_id}
                apply_snapshot, repaired, _ = execute(environment, apply_action, "recovery_render", stream_seed(seed, task.task_id, "controlled_recovery"))
                require_step(repaired, True)
                scores = components.evaluator.score_image(task, repaired.render.image_id)
                positive = repair_is_positive(bad_scores, scores)
                bundle["private_outcomes"].append({"stage": "recovery", "image_id": repaired.render.image_id, "scores": [asdict(score) for score in scores], "positive": positive, "before_image_id": bad_id})
                if positive:
                    proof = {"repair_success_label": True, "before_image_id": bad_id, "after_image_id": repaired.render.image_id, "minimum_dense_improvement": 0.1, "all_atoms_known": True, "no_previously_passed_requirement_regressed": True}
                    admit(environment, repaired_draft, draft_snapshot, "positive_draft", 1, "executed_calibrated_repair_improves_fixed_before", **proof)
                    admit(environment, apply_action, apply_snapshot, "positive_apply", 1, "executed_calibrated_repair_improves_fixed_before", **proof)
                    counts["positive_repairs"] += 1
                counts["unknown_recoveries"] += int(not all(score.known for score in (*bad_scores, *scores)))
                succeeded = bool(scores) and all(score.passed for score in scores)
                exhausted = environment.state.budget.renders >= task.limits.max_renders
                if not environment.state.done and (succeeded or exhausted):
                    finish = {"kind": "FINISH", "reason": "The render budget is exhausted; return the selected image."} if exhausted and not succeeded else {"kind": "FINISH"}
                    snapshot, result, _ = execute(environment, finish, "terminal_finish")
                    require_step(result)
                    admit(environment, finish, snapshot, "terminal_finish", 2, "all_fixed_calibrated_requirements_pass" if succeeded else "actual_render_budget_exhausted", visual_success_label=succeeded, render_budget_used=environment.state.budget.renders, render_budget_limit=task.limits.max_renders)
                    counts["successful_finishes" if succeeded else "budget_finishes"] += 1
        except Exception as error:
            counts["failed_tasks"] += 1
            bundle["failure"] = {"type": type(error).__name__, "message": str(error)}
        bundle["public_last_state"] = environment.snapshot()
        bundle["public_terminal_state"] = bundle["public_last_state"] if environment.state.done else None
        write_jsonl(folder / "actor_records.jsonl", task_records)
        write_report(folder / "state_bundle.json", bundle)
        write_report(completed, {"task_id": task.task_id, "family_id": task.family_id, "counts": counts})
        records.extend(task_records)
        bundles.append(bundle)
        for key, value in counts.items():
            report[key] += value
        persist()
        if "failure" in bundle:
            failure = bundle["failure"]
            raise RuntimeError(f"Intervention collection stopped after preserved task failure {task.task_id}: {failure['type']}: {failure['message']}")
    return report


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
            previous_render_count = environment.state.budget.renders
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
            elif isinstance(action, PlanAction) and not was_invalid:
                verified = True
                phase, evidence = 0, "executed_public_initial_plan"
            elif action.kind == "RENDER" and previous_render_count == 0 and result.render:
                verified = True
                phase, evidence = 0, "executed_public_initial_render"
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
