import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from agentic_image.config import load_config
from agentic_image.core.types import PlanSpec, PublicTask
from agentic_image.data.io import append_jsonl, read_jsonl, write_report
from agentic_image.data.proposals import _evaluation
from agentic_image.runtime.components import Components


PLAN_PROMPT = """Propose a concrete image-editing PLAN using only the public source image and original request. You have no private target, mask or evaluation labels. Return JSON with global_text, entities and requirements. global_text is a concise complete description of the desired final image grounded in the source and edit request, preserving unrequested content. Do not guess illegible text or identity. entities is a list of {entity_id,name,category,attributes,request_span_ids}; requirements is a list of {requirement_id,text,request_span_ids,entity_ids,depends_on}. attributes MUST be a JSON object such as {"color":"blue"}, never a list. request_span_ids, entity_ids and depends_on are lists of strings. Use distinct IDs starting entity: or requirement:. Use only the supplied request span IDs. Do not invent coordinates, masks, references or additional changes. This pilot uses the supported zero-region global conditioning profile. Keep global_text below 180 words. Requirements must cover the explicit request. Return only these three top-level keys. Example structure: {"global_text":"A scene description.","entities":[{"entity_id":"entity:1","name":"object","category":"object","attributes":{},"request_span_ids":["span:request"]}],"requirements":[{"requirement_id":"requirement:1","text":"Requested change","request_span_ids":["span:request"],"entity_ids":["entity:1"],"depends_on":[]}]}.\n"""


class ReviewJudge:
    def __init__(self, judge, evidence, correction=""):
        self.judge = judge
        self.config = judge.config
        self.evidence = evidence
        self.correction = correction

    def generate_json(self, text, images=(), max_new_tokens=1024):
        constraint = "Use strictly valid JSON. For every evaluation atom checker_route must be exactly one of semantic, count, text, left_of, right_of, above, below, inside, contains. Never use presence, absence, attribute, preservation, identity or other route names: express those as semantic. For semantic use expected_value as a plain string. Example atom: {\"raw_span\":\"make it blue\",\"predicate\":\"The requested object is blue\",\"expected_value\":\"blue\",\"checker_route\":\"semantic\",\"mandatory\":true,\"preserved\":false}. "
        value = self.judge.generate_json(constraint + self.correction + "\n" + text,
                                         images, max_new_tokens=max_new_tokens)
        self.evidence.append(value)
        return value


def emit(output, event):
    event = {"time": datetime.now(timezone.utc).isoformat(), **event}
    append_jsonl(output / "progress.jsonl", [event])
    print(json.dumps(event), flush=True)


def propose_plan(judge, task, image, correction=""):
    public = {
        "request": task.user_request.text,
        "request_spans": [span.model_dump(mode="json") for span in task.user_request.spans],
    }
    raw = judge.generate_json(PLAN_PROMPT + correction + json.dumps(public), [image],
                             max_new_tokens=1536)
    if set(raw) != {"global_text", "entities", "requirements"}:
        raise ValueError("PLAN proposal must contain the three requested public fields")
    plan = PlanSpec.model_validate({**raw, "source_image_id": task.user_request.source_image_id})
    spans = {span.span_id for span in task.user_request.spans}
    if not plan.requirements or len(plan.entities) > task.limits.max_entities:
        raise ValueError("PLAN proposal has invalid public capacity or empty requirements")
    if any(not set(item.request_span_ids) <= spans for item in (*plan.entities, *plan.requirements)):
        raise ValueError("PLAN proposal refers to an unknown request span")
    return plan, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=32)
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise FileExistsError("Completed proposal sets must not be overwritten")
    config = load_config(arguments.config)
    config["run_dir"] = str(output / "runtime")
    config["judge"]["calibration"] = None
    config["devices"]["judge"] = "cuda:0"
    config["devices"]["encoders"] = "cuda:0"
    components = Components(config)
    tasks = [PublicTask.model_validate(row) for row in read_jsonl(arguments.tasks)][:arguments.limit]
    if not tasks or len({task.family_id for task in tasks}) != len(tasks):
        raise ValueError("Proposal inputs must contain distinct real families")
    if any(task.split_id != "train" or not task.user_request.source_image_id for task in tasks):
        raise ValueError("Proposal inputs must be public training edit tasks")
    free, total = torch.cuda.mem_get_info(0)
    if free < 12 * 2**30:
        raise RuntimeError("Insufficient free GPU memory for the independent proposal job")
    started = time.monotonic()
    emit(output, {"stage": "starting", "tasks": len(tasks), "free_gib": free / 2**30,
                  "total_gib": total / 2**30, "human_reviewed": False})
    existing = {row["sample_id"]: row for row in read_jsonl(output / "proposals.jsonl")} if (output / "proposals.jsonl").exists() else {}
    components.load_judge(private=False)
    components.record_memory("review_judge_loaded")
    proposals = []
    for index, task in enumerate(tasks):
        if task.task_id in existing and "plan" in existing[task.task_id] and "evaluation" in existing[task.task_id]:
            proposals.append(existing[task.task_id])
            continue
        image = components.artifacts.get_image(task.user_request.source_image_id)
        row = {"sample_id": task.task_id, "family_id": task.family_id,
               "source_image_id": task.user_request.source_image_id, "request": task.user_request.text,
               "verified": False, "human_reviewed": False, "calibrated": False,
               "proposal_scope": "source_only_zero_region_review_proposal",
               "teacher_model": config["models"]["judge"],
               "teacher_revision": config["models"]["judge_revision"],
               "actor_teacher_trajectory": False, "private_target_seen": False,
               "independent_calls": True, "errors": [], "evaluation_call_outputs": []}
        correction = ""
        for attempt in range(2):
            try:
                judge = ReviewJudge(components.judge, row["evaluation_call_outputs"], correction)
                evaluation, raw, covered = _evaluation(judge, task.user_request.text,
                                                       [image], task.task_id, 12)
                row.update(evaluation=evaluation.model_dump(mode="json"),
                           evaluation_proposal=raw, automated_coverage=covered)
                break
            except (ValueError, KeyError, TypeError) as error:
                correction = "Previous response was rejected; correct this schema error: " + str(error)
                row["errors"].append({"stage": "evaluation", "attempt": attempt + 1,
                                      "error": str(error)})
        correction = ""
        for attempt in range(2):
            try:
                plan, raw = propose_plan(components.judge, task, image, correction)
                row.update(plan=plan.model_dump(mode="json"), plan_proposal=raw,
                           plan_schema_valid=True)
                break
            except (ValueError, KeyError, TypeError) as error:
                correction = "Previous response was rejected; correct this schema error: " + str(error) + "\n"
                row["errors"].append({"stage": "plan", "attempt": attempt + 1,
                                      "error": str(error)})
        append_jsonl(output / "proposals.jsonl", [row])
        proposals.append(row)
        emit(output, {"stage": "proposed", "completed": index + 1, "total": len(tasks),
                      "sample_id": task.task_id, "has_plan": "plan" in row,
                      "has_evaluation": "evaluation" in row,
                      "elapsed_seconds": round(time.monotonic() - started, 2)})
    judge = None
    components.release_frozen()
    components.load_encoders()
    components.record_memory("review_paired_encoder_loaded")
    compiled = 0
    for row in proposals:
        if "plan" not in row:
            continue
        try:
            plan = PlanSpec.model_validate(row["plan"])
            components.compiler.validate_plan(plan, row["source_image_id"])
            bundle = components.compiler.compile(plan, row["source_image_id"])
            payload, recipe = components.compiler._paths(bundle.bundle_id)
            append_jsonl(output / "compiled_plans.jsonl", [{
                "sample_id": row["sample_id"], "family_id": row["family_id"],
                "plan": row["plan"], "bundle_id": bundle.bundle_id,
                "bundle_path": str(payload.resolve()), "recipe_path": str(recipe.resolve()),
                "source_image_id": row["source_image_id"], "human_reviewed": False,
                "actor_teacher_trajectory": False,
                "encoder_versions": components.encoders.version_manifest,
            }])
            compiled += 1
            emit(output, {"stage": "compiled", "completed": compiled,
                          "sample_id": row["sample_id"], "bundle_id": bundle.bundle_id})
        except (ValueError, RuntimeError) as error:
            append_jsonl(output / "compile_failures.jsonl", [{"sample_id": row["sample_id"],
                                                            "error": str(error)}])
    report = {"status": "awaiting_human_review", "tasks": len(tasks),
              "plans": sum("plan" in row for row in proposals),
              "independent_evaluations": sum("evaluation" in row for row in proposals),
              "compiled_plans": compiled, "human_reviewed": 0,
              "paired_training_admitted": 0, "calibrated": False,
              "actor_teacher_trajectories": 0,
              "elapsed_seconds": time.monotonic() - started,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
              "profile": "official-viewer-reencoded-v1"}
    components.release_frozen(encoders=True)
    write_report(output / "completed.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
