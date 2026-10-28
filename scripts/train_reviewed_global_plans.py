import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.actor.config import adapter_directory, validate_actor_tokenizer
from agentic_image.actor.records import ActorSnapshot
from agentic_image.config import load_config
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.protocol import parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import PlanSpec
from agentic_image.data.audit import require_pair_audit
from agentic_image.data.io import read_jsonl, write_jsonl
from agentic_image.learning.transaction import atomic_json
from agentic_image.runtime.components import Components, actor_config
from agentic_image.runtime.prompt import SYSTEM_PROMPT
from agentic_image.runtime.rollout import read_tasks
from agentic_image.runtime.training import train_actor_sft


SCOPE = "original_request_global_pair_v1"
TEACHER = "deterministic_original_request_plan"


def prepare(arguments):
    from transformers import AutoProcessor

    audit = require_pair_audit(arguments.audit, review_mode="ai_exploratory")
    if audit.get("scope") != SCOPE:
        raise ValueError("The audit does not cover the original-request global PLAN scope")
    pairs_path = Path(arguments.pairs).resolve()
    if Path(audit["pair_source_path"]).resolve() != pairs_path.parent:
        raise ValueError("The admitted pairs are outside the audited dataset directory")
    pairs = list(read_jsonl(pairs_path))
    if len(pairs) != audit["reviewed"] or len({row["family_id"] for row in pairs}) < arguments.min_families:
        raise ValueError("The required audited independent families and exact audit coverage are missing")
    tasks = read_tasks(arguments.tasks)
    indexed = {task.task_id: (index, task) for index, task in enumerate(tasks)}
    config = load_config(arguments.config)
    config["review_mode"] = "ai_exploratory"
    config["run_dir"] = arguments.run_dir
    config["devices"]["actor"] = arguments.device
    config["models"]["actor_policy"] = str(adapter_directory(arguments.policy).resolve())
    config["sft"].update(actor_batch=4, protocol_phase_counts=[4, 0, 0, 0], protocol_updates=20)
    config["data"]["paired_audit"] = str(Path(arguments.audit).resolve())
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["data"]["actor_records"] = str(output / "actor_records.jsonl")
    actor_settings = actor_config(config)
    tokenizer = AutoProcessor.from_pretrained(config["models"]["actor_nf4"], local_files_only=True).tokenizer
    validate_actor_tokenizer(tokenizer, actor_settings)
    artifacts = ArtifactStore(config["data"]["artifacts"])
    reserve = len(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)) + 128
    records = []
    for pair in pairs:
        provenance = pair.get("verification_provenance", {})
        if not pair.get("verified") or provenance.get("review_type") != "ai" or provenance.get("human_reviewed") is not False:
            raise ValueError("Every admitted pair must carry its actual AI review provenance")
        index, task = indexed[pair["sample_id"]]
        if task.family_id != pair["family_id"] or task.split_id != "train":
            raise ValueError("The audited family does not match the public training task")
        expected = PlanSpec(global_text=task.user_request.text)
        plan = PlanSpec.model_validate(pair["condition_recipe"]["plan"])
        if plan.model_dump(mode="json") != expected.model_dump(mode="json") or plan.regions:
            raise ValueError("The label must be the exact deterministic original-request global PLAN")
        folder = Path(arguments.collection) / "tasks" / f"{index:03d}"
        snapshot_path = (folder / "attempt_00.snapshot.pt").resolve()
        before_path = folder / "attempt_00.before.json"
        before = json.loads(before_path.read_text())
        if before["task"]["task_id"] != task.task_id or before["budget"]["actions"] != 0 or before["plan"] is not None:
            raise ValueError("PLAN SFT must use the actual untouched initial action state")
        snapshot = ActorSnapshot.load(snapshot_path)
        if snapshot.metadata["task_id"] != task.task_id or snapshot.metadata["revision_id"] != before["manifest"]["revision_id"]:
            raise ValueError("The saved pre-action snapshot and environment state disagree")
        if snapshot.metadata.get("model_id") != actor_settings.model_id or snapshot.metadata.get("model_revision") != actor_settings.revision:
            raise ValueError("The saved initial snapshot belongs to another actor checkpoint")
        environment = EpisodeEnvironment.restore(
            before, artifacts, None, None, None, None,
            token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            max_input_tokens=config["actor"]["max_input_tokens"] - reserve,
        )
        environment._validate_plan(plan)
        internal_action = {"kind": "PLAN", "plan": plan.model_dump(mode="json")}
        public_action = environment.public_action(internal_action)
        restored = parse_action(environment.resolve_action(public_action))
        if restored.plan.model_dump(mode="json") != plan.model_dump(mode="json"):
            raise ValueError("Public action mapping changed the audited teacher PLAN")
        raw = json.dumps(public_action, ensure_ascii=True, separators=(",", ":"))
        target_ids = tokenizer.encode(raw, add_special_tokens=False) + [actor_settings.action_eos_token_id]
        if len(target_ids) > actor_settings.max_action_tokens:
            raise ValueError("The complete reviewed PLAN exceeds the action token budget")
        records.append({
            "record_id": f"original-request-plan:{task.task_id}", "task_id": task.task_id,
            "family_id": task.family_id, "state_id": f"{before['episode_id']}:initial", "phase": 0,
            "snapshot_path": str(snapshot_path), "target_ids": target_ids, "target_mask": [1] * len(target_ids),
            "target_action": public_action, "split_id": "train", "source_dataset": task.source_dataset,
            "verified": True, "public_reference_ids": list(snapshot.image_artifact_ids),
            "verifier_provenance": {
                "scope": SCOPE, "teacher_origin": TEACHER, "actor_sampled": False, "judge_generated": False,
                "review_type": "ai", "human_reviewed": False, "user_authorized_ai_review": True,
                "audit_path": str(Path(arguments.audit).resolve()), "paired_verification": provenance,
                "static_contract_validated": True, "environment_step_executed_for_teacher": False,
                "pre_action_environment_path": str(before_path.resolve()),
            },
        })
    identity = {"scope": SCOPE, "teacher_origin": TEACHER, "records": records, "initial_policy": config["models"]["actor_policy"]}
    metadata = output / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError("An existing reviewed PLAN dataset is immutable")
    atomic_json(metadata, identity)
    write_jsonl(output / "actor_records.jsonl", records)
    (output / "training.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(output / "report.json", {
        "scope": SCOPE, "teacher_origin": TEACHER, "records": len(records),
        "families": len({row["family_id"] for row in records}), "phase": 0, "updates": 20,
        "stage": "protocol", "full_four_phase_actor_sft": False,
        "review_type": "ai", "human_reviewed": False, "initial_policy": config["models"]["actor_policy"],
        "numerics_profile": actor_settings.numerics_profile,
    })
    return config


def wait_for_pairs(arguments):
    deadline = time.monotonic() + arguments.wait_seconds
    previous = None
    while True:
        families = 0
        if Path(arguments.pairs).is_file() and Path(arguments.audit).is_file():
            try:
                rows = list(read_jsonl(arguments.pairs))
                audit = json.loads(Path(arguments.audit).read_text())
                families = len({row["family_id"] for row in rows})
                ready = families >= arguments.min_families and audit.get("reviewed") == len(rows)
            except (ValueError, KeyError, FileNotFoundError):
                ready = False
        else:
            ready = False
        state = {"pid": os.getpid(), "stage": "reviewed_data_ready" if ready else "waiting_for_reviewed_data", "observed_families": families, "required_families": arguments.min_families}
        if state != previous:
            atomic_json(Path(arguments.pid_file).with_suffix(".wait.json"), state)
            print(json.dumps(state), flush=True)
            previous = state
        if ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("The reviewed global-pair dataset was not ready within the declared wait")
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--tasks", default="data/public/protocol_pilot_tasks.jsonl")
    parser.add_argument("--collection", default="runs/public-plan-candidates")
    parser.add_argument("--policy", default="runs/protocol-warmup/sft_protocol/policy/policy")
    parser.add_argument("--output", default="data/reviewed-global-plan-sft")
    parser.add_argument("--run-dir", default="runs/plan-global-ai")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-families", type=int, default=6)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--pid-file", default="deployment/plan-global-ai.pid")
    parser.add_argument("--exit-file", default="deployment/plan-global-ai.exit.json")
    arguments = parser.parse_args()
    Path(arguments.pid_file).write_text(str(os.getpid()) + "\n")
    status = {"pid": os.getpid(), "exit_code": 1}
    try:
        if arguments.min_families < 4 or arguments.wait_seconds < 0:
            raise ValueError("The reviewed PLAN stage needs at least four families and a bounded nonnegative wait")
        wait_for_pairs(arguments)
        config = prepare(arguments)
        if arguments.train:
            status["policy"] = train_actor_sft(Components(config), stage="protocol", updates=20, resume=arguments.resume)
        status["exit_code"] = 0
    except BaseException as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        raise
    finally:
        atomic_json(arguments.exit_file, status)


if __name__ == "__main__":
    main()
