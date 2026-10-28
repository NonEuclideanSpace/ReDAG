from dataclasses import asdict
from pathlib import Path


from agentic_image.checks.reward import dense_score
from agentic_image.core.protocol import DraftRepairAction, PlanAction, parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import ContractError, PublicTask
from agentic_image.learning.sampling import stream_seed

from .io import append_jsonl, read_jsonl, write_jsonl, write_report
from .paired import actor_snapshot, public_environment, save_action_record
from .prerequisites import require_qualified_renderer


def repair_is_positive(before, after, minimum_improvement=0.1):
    if (
        not before
        or len(before) != len(after)
        or {score.atom_id for score in before} != {score.atom_id for score in after}
    ):
        raise ValueError("Repair comparisons must use the same complete evaluation atom set")
    if not all(score.known for score in (*before, *after)):
        return False
    after_by_id = {score.atom_id: score for score in after}
    if any(score.passed and not after_by_id[score.atom_id].passed for score in before):
        return False
    return dense_score(after) - dense_score(before) >= minimum_improvement


def _restore(components, snapshot, original):
    return EpisodeEnvironment.restore(
        snapshot,
        components.artifacts,
        components.compiler,
        components.render_bridge,
        components.observer,
        components.compiler.load,
        token_counter=original.token_counter,
        max_input_tokens=original.view_builder.max_input_tokens,
    )


def _rule_record(components, environment, action, output, phase, state_id, suffix, provenance):
    snapshot = actor_snapshot(components, environment)
    result = environment.step(action)
    if result.error:
        raise RuntimeError(f"A generated demonstration was not executable: {result.error.code}")
    return save_action_record(
        components,
        environment,
        action,
        output / "snapshots",
        f"{state_id}:{suffix}",
        phase,
        state_id,
        provenance,
        snapshot,
    )


def collect_repairs(
    components,
    tasks_path,
    output_dir,
    renderer_version,
    sampler_id,
    max_states=1024,
    second_seed_states=256,
    max_render_calls=3840,
    seed=20261027,
):
    if not renderer_version or not sampler_id or max_states > 1024 or second_seed_states > 256:
        raise ValueError(
            "Repair collection requires locked SFT and sampler versions and registered collection caps"
        )
    require_qualified_renderer(components, renderer_version, sampler_id)
    components.load_all(private=True)
    components.render_bridge.behavior_version = renderer_version
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "actor_records.jsonl").exists():
        raise FileExistsError("Repair outputs must be written to a fresh collection directory")
    records, bundles = [], []
    report = {
        "attempts": 0,
        "initial_renders": 0,
        "candidate_renders": 0,
        "second_seed_renders": 0,
        "retry_renders": 0,
        "positive_repairs": 0,
        "second_seed_states": 0,
        "successful_initial_states": 0,
        "teacher_calls": 0,
        "unknown_states": 0,
        "no_valid_proposal": 0,
        "renderer_version": renderer_version,
        "sampler_id": sampler_id,
    }

    def spent():
        return sum(
            report[key]
            for key in (
                "initial_renders",
                "candidate_renders",
                "second_seed_renders",
                "retry_renders",
            )
        )

    written = {"records": 0, "bundles": 0}

    def persist():
        append_jsonl(output / "actor_records.jsonl", records[written["records"] :])
        append_jsonl(output / "state_bundles.jsonl", bundles[written["bundles"] :])
        written.update(records=len(records), bundles=len(bundles))
        report["render_calls"] = spent()
        report["action_records"] = len(records)
        report["unique_families"] = len({bundle["family_id"] for bundle in bundles})
        write_report(output / "report.json", report)

    for task_row in read_jsonl(tasks_path):
        if report["attempts"] >= max_states or spent() >= max_render_calls:
            break
        task = PublicTask.model_validate(task_row)
        if task.split_id != "train":
            continue
        report["attempts"] += 1
        state_id = f"repair:{report['attempts']:06d}"
        environment = public_environment(components, task, state_id)
        for attempt in range(2):
            snapshot = actor_snapshot(components, environment)
            sample = components.actor.sample(
                snapshot,
                stream_seed(seed, task.task_id, attempt, "repair_root"),
                "offline-repair-teacher",
            )
            report["teacher_calls"] += 1
            result = environment.step(sample.text)
            if environment.state.plan is not None or environment.state.done:
                break
        if environment.state.plan is None or environment.state.done:
            report["no_valid_proposal"] += 1
            continue
        initial_action = {
            "kind": "RENDER",
            "control_set_id": environment.state.manifest.active_control_set_id,
            "expected_revision": environment.state.manifest.revision_id,
        }
        initial = environment.step(initial_action, stream_seed(seed, task.task_id, "initial"))
        if initial.error or initial.render is None:
            report["no_valid_proposal"] += 1
            continue
        report["initial_renders"] += 1
        before_id = initial.render.image_id
        before_scores = components.evaluator.score_image(task, before_id)
        before_snapshot = environment.snapshot()
        bundle = {
            "state_id": state_id,
            "family_id": task.family_id,
            "task_id": task.task_id,
            "renderer_version": renderer_version,
            "sampler_id": sampler_id,
            "compiler_version": "dag-agent-v3",
            "source_image_id": before_id,
            "public_before_state": before_snapshot,
            "private_before_scores": [asdict(score) for score in before_scores],
            "candidate_actions": [],
            "private_outcomes": [],
            "verified_record_ids": [],
        }
        bundles.append(bundle)
        if all(score.passed for score in before_scores):
            report["successful_initial_states"] += 1
            record = _rule_record(
                components,
                environment,
                {"kind": "FINISH"},
                output,
                2,
                state_id,
                "finish_initial",
                {"verification": "all_fixed_requirements_pass"},
            )
            records.append(record)
            bundle["verified_record_ids"].append(record["record_id"])
            persist()
            continue
        if not all(score.known for score in before_scores):
            report["unknown_states"] += 1
        draft_sample, draft_snapshot = None, None
        for proposal_index in range(2):
            if environment.state.done:
                break
            snapshot = actor_snapshot(components, environment)
            sample = components.actor.sample(
                snapshot,
                stream_seed(seed, task.task_id, proposal_index, "repair_proposal"),
                "offline-repair-teacher",
            )
            report["teacher_calls"] += 1
            try:
                action = parse_action(sample.text)
            except ContractError:
                environment.step(sample.text)
                continue
            if action.kind in {"RENDER", "APPLY"}:
                break
            result = environment.step(
                sample.text, stream_seed(seed, task.task_id, proposal_index, "inspect")
            )
            if (
                isinstance(action, DraftRepairAction)
                and not result.error
                and environment.state.candidates
            ):
                draft_sample, draft_snapshot = sample, snapshot
                break
        if draft_sample is None or environment.state.done:
            report["no_valid_proposal"] += 1
            persist()
            continue
        drafted_state = environment.snapshot()
        candidates = list(environment.state.candidates.values())[:2]
        outcomes = []
        for candidate_index, candidate in enumerate(candidates):
            if spent() >= max_render_calls:
                break
            branch = _restore(components, drafted_state, environment)
            apply_action = {
                "kind": "APPLY",
                "candidate_id": candidate.candidate_id,
                "expected_revision": branch.state.manifest.revision_id,
            }
            apply_snapshot = actor_snapshot(components, branch)
            result = branch.step(
                apply_action, stream_seed(seed, state_id, candidate_index, 0, "candidate")
            )
            if result.error or result.render is None:
                continue
            report["candidate_renders"] += 1
            scores = components.evaluator.score_image(task, result.render.image_id)
            delta = dense_score(scores) - dense_score(before_scores)
            outcome = {
                "candidate_id": candidate.candidate_id,
                "seed_index": 0,
                "image_id": result.render.image_id,
                "delta": delta,
                "scores": [asdict(score) for score in scores],
                "trace_path": result.render.trajectory_id,
                "positive": repair_is_positive(before_scores, scores),
            }
            bundle["candidate_actions"].append(apply_action)
            bundle["private_outcomes"].append(outcome)
            outcomes.append(
                {
                    "candidate": candidate,
                    "branch": branch,
                    "action": apply_action,
                    "snapshot": apply_snapshot,
                    "scores": scores,
                    "outcome": outcome,
                }
            )
        needs_second = any(
            0.05 <= outcome["outcome"]["delta"] <= 0.15
            or not all(score.known for score in outcome["scores"])
            for outcome in outcomes
        )
        second_available = (
            needs_second
            and report["second_seed_states"] < second_seed_states
            and spent() + len(outcomes) + 1 <= max_render_calls
        )
        if second_available:
            report["second_seed_states"] += 1
            for candidate_index, outcome in enumerate(outcomes):
                branch = _restore(components, drafted_state, environment)
                result = branch.step(
                    outcome["action"], stream_seed(seed, state_id, candidate_index, 1, "candidate")
                )
                if result.error or result.render is None:
                    outcome["outcome"]["positive"] = False
                    continue
                report["second_seed_renders"] += 1
                scores = components.evaluator.score_image(task, result.render.image_id)
                accepted = repair_is_positive(before_scores, scores)
                outcome["outcome"]["positive"] = outcome["outcome"]["positive"] and accepted
                bundle["private_outcomes"].append(
                    {
                        "candidate_id": outcome["candidate"].candidate_id,
                        "seed_index": 1,
                        "image_id": result.render.image_id,
                        "delta": dense_score(scores) - dense_score(before_scores),
                        "scores": [asdict(score) for score in scores],
                        "trace_path": result.render.trajectory_id,
                        "positive": accepted,
                    }
                )
            retry = _restore(components, before_snapshot, environment)
            retry.view()
            retry_action = {
                "kind": "DRAFT_REPAIR",
                "source_image_id": before_id,
                "evidence_ids": [],
                "suspect_node_ids": [],
                "patch_ops": [],
                "instruction": "Resample the same previous image operation with fresh noise.",
                "scope": "retry",
            }
            retry_result = retry.step(retry_action)
            if not retry_result.error and retry.state.candidates:
                retry_candidate = next(iter(retry.state.candidates.values()))
                result = retry.step(
                    {
                        "kind": "APPLY",
                        "candidate_id": retry_candidate.candidate_id,
                        "expected_revision": retry.state.manifest.revision_id,
                    },
                    stream_seed(seed, state_id, "retry"),
                )
                if result.render:
                    report["retry_renders"] += 1
                    retry_scores = components.evaluator.score_image(task, result.render.image_id)
                    bundle["private_outcomes"].append(
                        {
                            "candidate_id": retry_candidate.candidate_id,
                            "scope": "retry",
                            "image_id": result.render.image_id,
                            "delta": dense_score(retry_scores) - dense_score(before_scores),
                            "scores": [asdict(score) for score in retry_scores],
                            "trace_path": result.render.trajectory_id,
                        }
                    )
        elif needs_second:
            for outcome in outcomes:
                if 0.05 <= outcome["outcome"]["delta"] <= 0.15 or not all(
                    score.known for score in outcome["scores"]
                ):
                    outcome["outcome"]["positive"] = False
        positives = [outcome for outcome in outcomes if outcome["outcome"]["positive"]]
        if positives:
            record = save_action_record(
                components,
                environment,
                draft_sample.text,
                output / "snapshots",
                f"{state_id}:draft",
                1,
                state_id,
                {
                    "verification": "executed_repair_improves_fixed_before",
                    "renderer_version": renderer_version,
                    "sampler_id": sampler_id,
                },
                draft_snapshot,
                draft_sample,
            )
            records.append(record)
            bundle["verified_record_ids"].append(record["record_id"])
            for index, outcome in enumerate(positives):
                record = save_action_record(
                    components,
                    outcome["branch"],
                    outcome["action"],
                    output / "snapshots",
                    f"{state_id}:apply:{index}",
                    1,
                    state_id,
                    {"verification": "all_known_delta_at_least_0.1_without_regression"},
                    outcome["snapshot"],
                )
                records.append(record)
                bundle["verified_record_ids"].append(record["record_id"])
                report["positive_repairs"] += 1
                if (
                    all(score.passed for score in outcome["scores"])
                    and not outcome["branch"].state.done
                ):
                    record = _rule_record(
                        components,
                        outcome["branch"],
                        {"kind": "FINISH"},
                        output,
                        2,
                        state_id,
                        f"finish_repair:{index}",
                        {"verification": "all_fixed_requirements_pass_after_repair"},
                    )
                    records.append(record)
                    bundle["verified_record_ids"].append(record["record_id"])
        for index, outcome in enumerate(outcomes):
            previous = {score.atom_id: score for score in before_scores}
            regression = all(score.known for score in (*before_scores, *outcome["scores"])) and any(
                previous[score.atom_id].passed and not score.passed for score in outcome["scores"]
            )
            if regression and not outcome["branch"].state.done:
                record = _rule_record(
                    components,
                    outcome["branch"],
                    {"kind": "SELECT", "image_id": before_id},
                    output,
                    2,
                    state_id,
                    f"rollback:{index}",
                    {"verification": "observed_regression_of_previously_passed_requirement"},
                )
                records.append(record)
                bundle["verified_record_ids"].append(record["record_id"])
                if not positives and not outcome["branch"].state.done:
                    record = _rule_record(
                        components,
                        outcome["branch"],
                        {"kind": "FINISH"},
                        output,
                        2,
                        state_id,
                        f"finish_rollback:{index}",
                        {"verification": "rollback_after_tested_candidates_failed_to_improve"},
                    )
                    records.append(record)
                    bundle["verified_record_ids"].append(record["record_id"])
        persist()
    persist()
    return report


def protocol_records(components, tasks_path, output_dir, max_records=1024, seed=20261027):
    components.load_actor()
    components.load_encoders()
    components.load_judge(private=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "actor_records.jsonl").exists():
        raise FileExistsError("Protocol records must be written to a fresh directory")
    records = []
    attempted = 0
    for row in read_jsonl(tasks_path):
        if len(records) >= max_records:
            break
        task = PublicTask.model_validate(row)
        if task.split_id != "train":
            continue
        attempted += 1
        environment = public_environment(components, task, f"protocol:{attempted}")
        invalid = environment.step(
            {"kind": "RENDER", "control_set_id": "missing", "expected_revision": "missing"}
        )
        if invalid.error is None or environment.state.budget.invalid != 1:
            raise RuntimeError("Expected recoverable root protocol error was not produced")
        snapshot = actor_snapshot(components, environment)
        sample = components.actor.sample(
            snapshot,
            stream_seed(seed, task.task_id, "protocol_recovery"),
            "offline-protocol-teacher",
        )
        try:
            action = parse_action(sample.text)
        except ContractError:
            continue
        if not isinstance(action, PlanAction):
            continue
        result = environment.step(sample.text)
        if result.error:
            continue
        records.append(
            save_action_record(
                components,
                environment,
                sample.text,
                output / "snapshots",
                f"protocol:{attempted}:recover",
                3,
                f"protocol:{attempted}",
                {"verification": "executed_public_contract_recovery"},
                snapshot,
                sample,
            )
        )
        if len(records) < max_records:
            environment.view()
            action = {"kind": "READ_NODES", "node_ids": ["user_request"]}
            record = _rule_record(
                components,
                environment,
                action,
                output,
                3,
                f"protocol:{attempted}",
                "read",
                {"verification": "executed_public_node_read"},
            )
            records.append(record)
    write_jsonl(output / "actor_records.jsonl", records)
    report = {
        "attempted_tasks": attempted,
        "verified_records": len(records),
        "requested_cap": max_records,
        "unique_families": len({record["family_id"] for record in records}),
    }
    write_report(output / "report.json", report)
    return report


def complex_plan_records(components, tasks_path, output_dir, max_tasks=96, seed=20261027):
    if not 1 <= max_tasks <= 96:
        raise ValueError("The registered complex protocol proposal cap is 96 tasks")
    components.load_actor()
    components.load_encoders()
    components.load_judge(private=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "actor_records.jsonl").exists():
        raise FileExistsError("Complex protocol demonstrations need a fresh output directory")
    records = []
    report = {
        "attempted_tasks": 0,
        "teacher_calls": 0,
        "verified_plans": 0,
        "not_complex": 0,
        "invalid_plans": 0,
    }
    for row in read_jsonl(tasks_path):
        if report["attempted_tasks"] >= max_tasks:
            break
        task = PublicTask.model_validate(row)
        if task.split_id != "train":
            continue
        report["attempted_tasks"] += 1
        accepted = False
        for attempt in range(2):
            state_id = f"complex:{report['attempted_tasks']:04d}:{attempt}"
            environment = public_environment(components, task, state_id)
            snapshot = actor_snapshot(components, environment)
            sample = components.actor.sample(
                snapshot,
                stream_seed(seed, task.task_id, attempt, "complex_plan"),
                "offline-complex-plan-teacher",
            )
            report["teacher_calls"] += 1
            try:
                action = parse_action(sample.text)
                if not isinstance(action, PlanAction):
                    continue
                result = environment.step(sample.text)
                if result.error:
                    continue
                plan = environment.state.plan
                components.compiler.compile(plan, environment.state.manifest.selected_image_id)
                complex_plan = (
                    len(plan.entities) >= 4
                    or bool(plan.dependencies)
                    or len(plan.regions) == 2
                    and bool(plan.references)
                    or len(plan.requirements) >= 5
                )
                if not complex_plan:
                    report["not_complex"] += 1
                    break
                records.append(
                    save_action_record(
                        components,
                        environment,
                        sample.text,
                        output / "snapshots",
                        state_id,
                        0,
                        state_id,
                        {"verification": "public_only_complex_plan_schema_graph_and_compilation"},
                        snapshot,
                        sample,
                    )
                )
                report["verified_plans"] += 1
                accepted = True
                break
            except (ContractError, ValueError):
                continue
        if not accepted:
            report["invalid_plans"] += 1
    write_jsonl(output / "actor_records.jsonl", records)
    write_report(output / "report.json", report)
    return report
