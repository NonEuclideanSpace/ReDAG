import argparse
import copy
import json
import time
from pathlib import Path

from agentic_image.config import load_config
from agentic_image.learning.transaction import atomic_json
from agentic_image.runtime.components import Components
from agentic_image.runtime.sampler import EVALUATION_SEEDS, comparison_tasks, cached_render_report, render_comparison_shard, require_training, score_comparison_shard
from agentic_image.runtime.sampler_fallback import FALLBACK_CANDIDATES, FALLBACK_SAMPLERS, fallback_registration, select_fallback, prepare_fallback_review



def wait_for_render_shard(tasks, output, candidate, start, count=8, timeout_seconds=1200, poll_seconds=5):
    if candidate not in FALLBACK_CANDIDATES or not 0 < timeout_seconds <= 1200 or not 0 < poll_seconds <= 30:
        raise ValueError("Use a registered candidate and a bounded twenty-minute wait")
    required = count * len(EVALUATION_SEEDS)
    started, previous, last_report = time.monotonic(), None, None
    report_path = Path(output) / "shards" / f"score_wait_{candidate}_{start:03d}_{start + count - 1:03d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        available = 0
        for index in range(start, start + count):
            task = tasks[index]
            for seed in EVALUATION_SEEDS:
                metadata = Path(output) / "images" / candidate / f"{index:03d}_{seed}.json"
                pixels = metadata.with_suffix(".png")
                if not metadata.is_file() or not pixels.is_file() or not pixels.stat().st_size:
                    continue
                row = json.loads(metadata.read_text())
                steps, eta = FALLBACK_SAMPLERS[candidate]
                if (row.get("task_id") != task.task_id or row.get("family_id") != task.family_id
                        or row.get("method") != candidate or row.get("eval_seed") != seed
                        or row.get("steps") != steps or row.get("eta") != eta
                        or row.get("status") != "rendered" or not row.get("image")
                        or Path(row["image"]).resolve() != pixels.resolve()):
                    raise ValueError("Waiting render metadata differs from the registered candidate, task or pixels")
                available += 1
        now = time.monotonic()
        elapsed = now - started
        status = "ready" if available == required else "timed_out" if elapsed >= timeout_seconds else "waiting"
        if available != previous or last_report is None or now - last_report >= 30 or status != "waiting":
            report = {"stage": "score_wait", "status": status, "candidate": candidate,
                "start": start, "count": count, "available": available, "required": required,
                "elapsed_seconds": round(elapsed, 3), "timeout_seconds": timeout_seconds, "accepted": False}
            atomic_json(report_path, report)
            print(json.dumps(report), flush=True)
            previous, last_report = available, now
        if status == "ready":
            return report
        if status == "timed_out":
            raise TimeoutError(f"Render shard {start:03d} has {available}/{required} candidate images after {timeout_seconds}s")
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "render", "score", "prepare-review", "select"))
    parser.add_argument("--config", default="configs/sampler-expanded-200.yaml")
    parser.add_argument("--tasks", default="data/evaluation/development64/public_tasks.jsonl")
    parser.add_argument("--reservation", default="data/evaluation/development64/reserved.frozen.jsonl")
    parser.add_argument("--output", default="runs/sampler-fallback-200")
    parser.add_argument("--reference-output", default="runs/sampler-expanded-200")
    parser.add_argument("--reference-reviews", default="runs/sampler-expanded-200/ai_severe_reviews_merged.jsonl")
    parser.add_argument("--candidate", choices=FALLBACK_CANDIDATES, default=FALLBACK_CANDIDATES[0])
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
        parser.error("Use complete eight-task shards within the fixed 64 families")
    config = copy.deepcopy(load_config(args.config))
    tasks = comparison_tasks(args.tasks, args.reservation)
    fallback_registration(config, tasks, args.output, args.reference_output, args.reference_reviews)
    if args.stage == "prepare-review":
        print(json.dumps(prepare_fallback_review(tasks, args.output)), flush=True)
        return
    if args.stage == "select":
        print(json.dumps(select_fallback(config, tasks, args.output, args.candidate, args.severe_reviews, args.bootstrap_iterations)), flush=True)
        return
    if args.stage == "register":
        print(json.dumps({"status": "registered", "accepted": False, "new_renders": 256, "ode_reused": 128}), flush=True)
        return
    if args.stage == "render":
        training = config["sampler_comparison"]["render"]
        require_training(training["exit"], training["metrics"], training["updates"])
    config["sampler_behavior_version"] = "sampler-fallback-200-v1"
    config["run_dir"] = str(Path(args.output) / "workers" / f"{args.stage}_{args.candidate}_{args.start:03d}")
    for name in config["devices"]:
        config["devices"][name] = args.device
    config["models"]["actor_reference"] = None
    config["models"]["renderer_reference"] = None
    reports, pending = [], []
    for start in range(args.start, args.start + args.count, 8):
        cached = cached_render_report(config, tasks, args.output, start, 8, (args.candidate,), FALLBACK_SAMPLERS) if args.stage == "render" else None
        if cached is None:
            pending.append(start)
        else:
            reports.append(cached)
    if pending:
        if args.stage == "render":
            config["judge"]["calibration"] = None
        components = None
        for start in pending:
            if args.wait_for_renders:
                wait_for_render_shard(tasks, args.output, args.candidate, start)
            if components is None:
                components = Components(config)
                if args.stage == "render":
                    components.load_renderer()
                    operation = render_comparison_shard
                else:
                    components.load_judge(private=True)
                    operation = score_comparison_shard
                components.record_memory("sampler_fallback_loaded")
            reports.append(operation(components, tasks, args.output, start, 8, (args.candidate,), FALLBACK_SAMPLERS))
        components.record_memory("sampler_fallback_finished")
    result = {"status": "completed" if all(row["status"] == "completed" for row in reports) else "blocked",
        "stage": args.stage, "candidate": args.candidate, "start": args.start, "count": args.count,
        "images": sum(row["completed"] for row in reports), "required": args.count * 2,
        "reports": reports, "accepted": False, "ode_rerendered": 0}
    path = Path(args.output) / "shards" / f"{args.stage}_{args.candidate}_{args.start:03d}_{args.start + args.count - 1:03d}.json"
    atomic_json(path, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
