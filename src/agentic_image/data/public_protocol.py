import json
from pathlib import Path

from agentic_image.core.types import PlanSpec, PublicTask

from .io import read_jsonl, write_jsonl, write_report
from .paired import actor_snapshot, public_environment, save_action_record


CASES = {
    "unknown_node": (
        {"kind": "READ_NODES", "node_ids": ["node:undisclosed"]},
        "UNKNOWN_ID",
    ),
    "unknown_cursor": (
        {"kind": "READ_NODES", "page_cursor": "page:undisclosed"},
        "UNKNOWN_CURSOR",
    ),
}


def prepare_recovery(environment, case):
    environment.view()
    setup = {
        "kind": "PLAN",
        "plan": PlanSpec(global_text=environment.task.user_request.text).model_dump(mode="json"),
    }
    planned = environment.step(setup)
    if planned.error or environment.state.plan is None:
        raise RuntimeError("The public protocol context could not execute its initial PLAN")
    environment.view()
    invalid, expected = CASES[case]
    failed = environment.step(invalid)
    if failed.error is None or failed.error.code != expected or failed.done:
        raise RuntimeError(f"Expected a recoverable {expected} contract error")
    return {
        "setup_action": setup,
        "setup_result": planned.model_dump(mode="json"),
        "failed_action": invalid,
        "failed_result": failed.model_dump(mode="json"),
    }


def recovery_action(case, payload):
    if "READ_NODES" not in payload["legal_actions"]:
        raise ValueError("The saved public state does not permit READ_NODES")
    if case == "unknown_node":
        if not any(row["id"] == "plan" for row in payload["node_directory"]):
            raise ValueError("The recovery target must name the disclosed plan")
        return {"kind": "READ_NODES", "node_ids": ["plan"], "fields": ["typed_parameters"]}
    return {"kind": "READ_NODES", "page_cursor": payload["page_cursor"]}


def verify_recovery(environment, result):
    if result.error or result.done or result.protocol_failure:
        raise RuntimeError("The proposed recovery failed the actual public contract")
    records = result.output.get("records", [])
    plans = [row for row in records if row["id"] == "plan"]
    if (
        not plans
        or plans[0]["value"]["typed_parameters"]["global_text"] != environment.task.user_request.text
    ):
        raise RuntimeError("The recovery did not retrieve the original public request's plan")
    if environment.state.budget.invalid != 1 or environment.state.budget.reads != 1:
        raise RuntimeError("The recovery changed the declared invalid/read budget unexpectedly")
    if environment.state.executed_renders:
        raise RuntimeError("Protocol warmup must not produce renderer training evidence")


def build_public_contract_records(components, tasks_path, output_dir, limit=32, resume=False):
    if not 16 <= limit <= 32:
        raise ValueError("Protocol warmup requires a declared limit of 16 to 32 real families")
    tasks = [PublicTask.model_validate(row) for row in read_jsonl(tasks_path)][:limit]
    families = {task.family_id for task in tasks}
    if len(tasks) < 16 or len(families) != len(tasks):
        raise ValueError("Provide at least 16 tasks with distinct real training families")
    for task in tasks:
        if task.split_id != "train" or task.source_dataset not in {"magicbrush", "omniedit"}:
            raise ValueError("Protocol warmup only accepts isolated official training-source families")
        source = task.user_request.source_image_id
        if (
            not source
            or components.artifacts.get_ref(source).metadata.get("purpose") == "deployment_probe"
        ):
            raise ValueError("Protocol warmup requires a real dataset source image")
        components.artifacts.get_image(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scope": "public_contract_recovery_only",
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "actor_model": components.config["models"]["actor"],
        "actor_revision": components.config["models"]["actor_revision"],
        "judge_model": components.config["models"]["judge"],
        "judge_revision": components.config["models"]["judge_revision"],
    }
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if not resume or json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Use a fresh output or resume the identical registered protocol dataset")
    else:
        write_report(metadata_path, metadata)
    record_path = output / "actor_records.jsonl"
    evidence_path = output / "state_bundles.jsonl"
    records = (
        {row["record_id"]: row for row in read_jsonl(record_path)} if record_path.exists() else {}
    )
    evidence = (
        {row["state_id"]: row for row in read_jsonl(evidence_path)} if evidence_path.exists() else {}
    )
    components.load_actor()
    components.load_judge(private=False)
    for task_index, task in enumerate(tasks):
        for case in CASES:
            state_id = f"public-protocol:{task_index:03d}:{case}"
            if state_id in records:
                if state_id not in evidence or not Path(records[state_id]["snapshot_path"]).is_file():
                    raise RuntimeError("An existing protocol record is missing its original evidence")
                continue
            environment = public_environment(components, task, state_id)
            setup = prepare_recovery(environment, case)
            snapshot = actor_snapshot(components, environment)
            action = recovery_action(case, snapshot.metadata["view"]["payload"])
            before = environment.snapshot()
            result = environment.step(action)
            verify_recovery(environment, result)
            provenance = {
                "verification": "executed_public_contract_read_recovery",
                "scope": "public_contract_recovery_only",
                "failure_code": CASES[case][1],
                "visual_success_label": False,
                "plan_semantic_label": False,
                "judge_model": components.judge.config.model,
                "judge_revision": components.judge.config.revision,
                "judge_calibrated": bool(components.judge.thresholds),
            }
            record = save_action_record(
                components, environment, action, output / "snapshots", state_id,
                3, state_id, provenance, snapshot,
            )
            record["split_id"] = "train"
            record["task_id"] = task.task_id
            record["source_dataset"] = task.source_dataset
            evidence[state_id] = {
                "state_id": state_id, "family_id": task.family_id, "task_id": task.task_id,
                "setup_is_supervised": False, **setup, "public_before_state": before,
                "recovery_action": action, "recovery_result": result.model_dump(mode="json"),
                "provenance": provenance,
            }
            records[state_id] = record
            write_jsonl(evidence_path, evidence.values())
            write_jsonl(record_path, records.values())
            report = {
                "scope": "public_contract_recovery_only", "phase": 3,
                "verified_records": len(records), "requested_records": len(tasks) * len(CASES),
                "families_with_records": len({row["family_id"] for row in records.values()}),
                "completed_families": sum(
                    all(f"public-protocol:{index:03d}:{name}" in records for name in CASES)
                    for index in range(len(tasks))
                ),
                "source_tasks": str(Path(tasks_path).resolve()),
                "visual_success_labels": 0, "plan_semantic_labels": 0,
                "renderer_training_pairs": 0, "private_rewards": 0,
                "recommended_protocol_phase_counts": [0, 0, 0, 4],
            }
            write_report(output / "report.json", report)
            print(json.dumps(report), flush=True)
    return str(record_path.resolve())
