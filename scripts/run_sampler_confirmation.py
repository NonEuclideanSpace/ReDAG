import argparse
import copy
import json
import time
from pathlib import Path

from agentic_image.config import load_config
from agentic_image.learning.transaction import atomic_json
from agentic_image.core.types import PublicTask
from agentic_image.runtime.components import Components, judge_config
from agentic_image.runtime.sampler import (
    EVALUATION_SEEDS, cached_freeze_report, cached_render_report, comparison_tasks,
    complete_comparison_rows, freeze_comparison_plans, frozen_json, policy_identity,
    prepare_severe_review, render_comparison_shard, require_training,
    sampler_decision_with_protection, score_comparison_shard,
)


SAMPLERS = {"ode50": (50, 0.0), "sde50_eta005": (50, 0.05)}
CANDIDATE = "sde50_eta005"


def read(path):
    return json.loads(Path(path).read_text())


def register(config, tasks, output, original, selection_rule):
    output, original = Path(output), Path(original).resolve()
    if output.resolve() == original:
        raise ValueError("Confirmation cannot overwrite the original comparison")
    prior = read(original / "registration.json")
    families = {task.family_id for task in tasks}
    previous = {row["family_id"] for row in prior["tasks"]}
    if len(families) != 64 or len(previous) != 64 or families & previous:
        raise ValueError("Confirmation requires 64 new independent development families")
    rule = read(selection_rule)
    if (rule.get("status") not in ("preregistered_before_any_new_render_score",
            "preregistered_and_metadata_isolated_before_render")
            or rule.get("families") != 64 or set(rule.get("new_family_ids", [])) != families
            or set(rule.get("original_development_family_ids", [])) != previous
            or families & set(rule.get("excluded_family_ids", []))
            or rule.get("official_test_used") is not False):
        raise ValueError("Confirmation family exclusions differ from the fixed independent reservation")
    if prior["samplers"].get(CANDIDATE) != [50, 0.05]:
        raise ValueError("Confirmation must keep the original candidate unchanged")
    for key, value in prior["models"].items():
        if config["models"].get(key) != value:
            raise ValueError("Confirmation changed a frozen model: " + key)
    if (prior["training_seed"] != config["seed"] or prior["canvas"] != config["canvas"]
            or config["renderer"]["cfg_scale"] != 4):
        raise ValueError("Confirmation changed the sampling conditions")
    if (prior["eval_seeds"] != list(EVALUATION_SEEDS)
            or any(config["actor"].get(key) != value for key, value in prior["actor_sampling"].items())):
        raise ValueError("Confirmation changed the fixed controller sampling or evaluation seeds")
    identities = {role: read(original / f"{role}_policy_identity.json") for role in ("actor", "renderer")}
    for role, identity in identities.items():
        if identity != policy_identity(config["models"][role + "_policy"], actor=role == "actor"):
            raise ValueError("Confirmation policy changed after original sampling")
        frozen_json(output / f"{role}_policy_identity.json", identity)
    original_calibration = read(original / "calibration.json")
    identity = judge_config(config).calibration_identity
    if read(str(config["judge"]["calibration"]) + ".model.json") != identity:
        raise ValueError("Confirmation calibration identity differs from the configured judge")
    calibration = {"identity": identity, "routes": read(config["judge"]["calibration"])}
    if calibration != original_calibration:
        raise ValueError("Confirmation cannot recalibrate or change existing thresholds")
    frozen_json(output / "calibration.json", original_calibration)
    for method in SAMPLERS:
        (output / "scores" / method).mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    value = {"version": "independent-development64-confirmation-v1",
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "samplers": {key: list(value) for key, value in SAMPLERS.items()},
        "eval_seeds": list(EVALUATION_SEEDS), "training_seed": config["seed"],
        "expected_renders": 256, "canvas": config["canvas"], "cfg_scale": 4.0,
        "models": prior["models"], "actor_sampling": prior["actor_sampling"],
        "controller_seed": 101, "controller_attempts": 2,
        "original_registration": str(original / "registration.json"),
        "original_families": sorted(previous), "confirmation_families": sorted(families),
        "confirmation_budget_fixed_before_outputs": True, "adaptive_family_additions": False,
        "family_selection_rule": rule,
        "decision_population": "all_original64_plus_all_independent64_family_clusters",
        "independent_result_reported_separately": True,
        "success_margin": 0.05, "defect_margin": 0.05, "protection_margin": 0.05,
        "unknown_labels_retained": True, "human_reviewed": False}
    return frozen_json(output / "registration.json", value)


def select(config, tasks, output, original, review_paths, iterations):
    output, original = Path(output), Path(original)
    labels = {}
    for path in review_paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row["image_id"] in labels and labels[row["image_id"]] != row:
                raise ValueError("Conflicting reviews for one actual confirmation image")
            labels[row["image_id"]] = row
    reviews = output / "quality_reviews_merged.jsonl"
    reviews.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels.values()))
    new_rows, report = complete_comparison_rows(tasks, output, reviews, tuple(SAMPLERS), SAMPLERS)
    prior = read(original / "registration.json")
    old_tasks = [PublicTask.model_validate(row) for row in prior["tasks"]]
    old_rows, old_report = complete_comparison_rows(old_tasks, original,
        original / "quality_reviews_merged.jsonl", tuple(SAMPLERS), SAMPLERS)
    if report["status"] != "ready_for_decision" or old_report["status"] != "ready_for_decision":
        result = {"status": "incomplete", "accepted": False,
                  "confirmation": report, "original": old_report}
    else:
        if len(old_rows) != 256 or len(new_rows) != 256:
            raise ValueError("Confirmation selection requires all 512 actual scored observations")
        if {r["family_id"] for r in new_rows} & {r["family_id"] for r in old_rows}:
            raise ValueError("Confirmation families overlap the original comparison")
        original_decision = sampler_decision_with_protection(old_rows, CANDIDATE, "ode50", iterations)
        independent = sampler_decision_with_protection(new_rows, CANDIDATE, "ode50", iterations)
        combined = sampler_decision_with_protection(old_rows + new_rows, CANDIDATE, "ode50",
                                                   iterations, expected_families=128)
        result = {"status": combined["status"], "accepted": combined["accepted"],
            "candidate": CANDIDATE, "reference": "ode50", "steps": 50, "eta": 0.05,
            "cfg_scale": 4.0, "families": 128, "paired_records": len(old_rows) + len(new_rows),
            "eval_seeds": list(EVALUATION_SEEDS), "decision": combined,
            "decisions": {CANDIDATE: combined}, "original_decision": original_decision,
            "independent_confirmation_decision": independent,
            "qualified_on": {"families": 128, "original_families": 64, "confirmation_families": 64,
                "original_registration": str((original / "registration.json").resolve()),
                "confirmation_registration": str((output / "registration.json").resolve())},
            "renderer_policy": config["models"]["renderer_policy"],
            "renderer_policy_identity": read(output / "renderer_policy_identity.json"),
            "calibration": str((output / "calibration.json").resolve()),
            "severe_reviews": [str((original / "quality_reviews_merged.jsonl").resolve()), str(reviews.resolve())],
            "review_scope": "Actual AI blind visual review and calibrated private scores; unknowns retained",
            "human_reviewed": False, "fixed_confirmation_budget": 256,
            "final_sampler_lock_required_before_repair_sft_rl_and_main_test": True}
    atomic_json(output / "sampler_selection.json", result)
    return result



def render_availability(tasks, output, start, count):
    available = 0
    for index in range(start, start + count):
        task = tasks[index]
        for seed in EVALUATION_SEEDS:
            for method in SAMPLERS:
                metadata = Path(output) / "images" / method / f"{index:03d}_{seed}.json"
                pixels = metadata.with_suffix(".png")
                if not metadata.is_file() or not pixels.is_file() or not pixels.stat().st_size:
                    continue
                row = read(metadata)
                if (row.get("task_id") != task.task_id or row.get("family_id") != task.family_id
                        or row.get("method") != method or row.get("eval_seed") != seed
                        or row.get("status") != "rendered" or not row.get("image")
                        or Path(row["image"]).resolve() != pixels.resolve()):
                    raise ValueError("Waiting render metadata does not match the frozen task and pixels")
                available += 1
    return available


def wait_for_render_shard(tasks, output, start, count=8, timeout_seconds=1200, poll_seconds=5):
    if not 0 < timeout_seconds <= 1200 or not 0 < poll_seconds <= 30:
        raise ValueError("Render waiting requires a bounded twenty-minute deadline")
    required = count * len(EVALUATION_SEEDS) * len(SAMPLERS)
    started, previous, last_report = time.monotonic(), None, None
    path = Path(output) / "shards" / f"score_wait_{start:03d}_{start + count - 1:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        available = render_availability(tasks, output, start, count)
        now = time.monotonic()
        elapsed = now - started
        status = "ready" if available == required else "timed_out" if elapsed >= timeout_seconds else "waiting"
        if available != previous or last_report is None or now - last_report >= 30 or status != "waiting":
            report = {"stage": "score_wait", "status": status, "start": start, "count": count,
                "available": available, "required": required, "elapsed_seconds": round(elapsed, 3),
                "timeout_seconds": timeout_seconds, "accepted": False}
            atomic_json(path, report)
            print(json.dumps(report), flush=True)
            previous, last_report = available, now
        if status == "ready":
            return report
        if status == "timed_out":
            raise TimeoutError(f"Render shard {start:03d} has {available}/{required} images after {timeout_seconds}s")
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "freeze", "render", "score", "prepare-review", "select"))
    parser.add_argument("--config", default="configs/sampler-confirmation-200.yaml")
    parser.add_argument("--tasks", default="data/evaluation/sampler_confirm64/public_tasks.jsonl")
    parser.add_argument("--reservation", default="data/evaluation/sampler_confirm64/reserved.frozen.jsonl")
    parser.add_argument("--output", default="runs/sampler-confirmation-200")
    parser.add_argument("--selection-rule")
    parser.add_argument("--original", default="runs/sampler-fallback-200")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--wait-for-renders", action="store_true")
    parser.add_argument("--severe-reviews", action="append", default=[])
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    if args.wait_for_renders and args.stage != "score":
        parser.error("--wait-for-renders is available only for score")
    if args.start % 8 or args.count % 8 or args.start < 0 or args.count < 8 or args.start + args.count > 64:
        parser.error("Use complete eight-task shards within the 64 frozen confirmation families")
    config = copy.deepcopy(load_config(args.config))
    tasks = comparison_tasks(args.tasks, args.reservation, task_prefix="sampler-confirm64:", expected_split="official-dev")
    selection_rule = args.selection_rule or str(Path(args.reservation).with_name("selection_rule.json"))
    registration = register(config, tasks, args.output, args.original, selection_rule)
    if args.stage == "register":
        print(json.dumps(registration), flush=True)
        return
    if args.stage == "prepare-review":
        print(json.dumps(prepare_severe_review(tasks, args.output, SAMPLERS, "confirmation-quality")), flush=True)
        return
    if args.stage == "select":
        print(json.dumps(select(config, tasks, args.output, args.original, args.severe_reviews,
                                args.bootstrap_iterations)), flush=True)
        return
    if args.stage in ("freeze", "render"):
        training = config["sampler_comparison"][args.stage]
        require_training(training["exit"], training["metrics"], training["updates"])
    reports, pending = [], []
    for start in range(args.start, args.start + args.count, 8):
        cached = (cached_freeze_report(config, tasks, args.output, start, 8) if args.stage == "freeze"
                  else cached_render_report(config, tasks, args.output, start, 8, tuple(SAMPLERS), SAMPLERS)
                  if args.stage == "render" else None)
        if cached is None:
            pending.append(start)
        else:
            reports.append(cached)
    if pending:
        config["run_dir"] = str(Path(args.output) / "workers" / f"{args.stage}_{args.start:03d}")
        config["sampler_behavior_version"] = "sampler-independent-confirmation-v1"
        for name in config["devices"]:
            config["devices"][name] = args.device
        config["models"].update(actor_reference=None, renderer_reference=None)
        if args.stage != "score":
            config["judge"]["calibration"] = None
        components = None
        for start in pending:
            if args.wait_for_renders:
                wait_for_render_shard(tasks, args.output, start)
            if components is None:
                components = Components(config)
                if args.stage == "freeze":
                    components.load_actor()
                    components.load_encoders()
                    components.load_judge(private=False)
                elif args.stage == "render":
                    components.load_renderer()
                else:
                    components.load_judge(private=True)
                components.record_memory("confirmation_loaded")
            if args.stage == "freeze":
                row = freeze_comparison_plans(components, tasks, args.output, start, 8)
            else:
                operation = render_comparison_shard if args.stage == "render" else score_comparison_shard
                row = operation(components, tasks, args.output, start, 8, tuple(SAMPLERS), SAMPLERS)
            reports.append(row)
            if row["status"] != "completed":
                break
    result = {"status": "completed" if len(reports) == args.count // 8
              and all(row["status"] == "completed" for row in reports) else "blocked",
        "stage": args.stage, "start": args.start, "count": args.count,
        "completed": sum(row["completed"] for row in reports), "reports": reports, "accepted": False}
    atomic_json(Path(args.output) / "shards" / f"{args.stage}_{args.start:03d}_{args.start + args.count - 1:03d}.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
