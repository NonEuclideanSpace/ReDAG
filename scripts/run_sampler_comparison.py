import argparse
import copy
import json
from pathlib import Path

from agentic_image.config import load_config
from agentic_image.runtime.components import Components
from agentic_image.runtime.sampler import (
    comparison_registration, comparison_tasks, freeze_comparison_plans,
    render_comparison_shard, require_training, shard_indices, score_comparison_shard,
    complete_comparison_rows, select_comparison_sampler, prepare_severe_review, cached_freeze_report, cached_render_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "render", "score", "assemble", "select", "prepare-review"))
    parser.add_argument("--config", default="configs/sampler-expanded-200.yaml")
    parser.add_argument("--tasks", default="data/evaluation/development64/public_tasks.jsonl")
    parser.add_argument("--reservation", default="data/evaluation/development64/reserved.frozen.jsonl")
    parser.add_argument("--output", default="runs/sampler-expanded-200")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--methods", help="Comma-separated registered methods; omitted means all four")
    parser.add_argument("--calibration")
    parser.add_argument("--severe-reviews")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    methods = args.methods.split(",") if args.methods else None
    shard_indices(args.start, args.count)
    config = copy.deepcopy(load_config(args.config))
    tasks = comparison_tasks(args.tasks, args.reservation)
    comparison_registration(config, tasks, args.output)
    if args.stage in {"assemble", "select", "prepare-review"}:
        if args.stage == "assemble":
            _, result = complete_comparison_rows(tasks, args.output, args.severe_reviews, methods)
        elif args.stage == "prepare-review":
            result = prepare_severe_review(tasks, args.output)
        else:
            if not args.severe_reviews:
                parser.error("Selection requires actual severe-defect reviews")
            result = select_comparison_sampler(config, tasks, args.output, args.severe_reviews, args.bootstrap_iterations)
        print(json.dumps(result), flush=True)
        return
    if args.stage in {"freeze", "render"}:
        training = config["sampler_comparison"][args.stage]
        require_training(training["exit"], training["metrics"], training["updates"])
    if args.stage == "freeze":
        cached = cached_freeze_report(config, tasks, args.output, args.start, args.count)
        if cached is not None:
            print(json.dumps(cached), flush=True)
            return
    if args.stage == "render":
        cached = cached_render_report(config, tasks, args.output, args.start, args.count, methods)
        if cached is not None:
            print(json.dumps(cached), flush=True)
            return
    config["run_dir"] = str(Path(args.output) / "workers" / f"{args.stage}_{args.start:03d}")
    for name in config["devices"]:
        config["devices"][name] = args.device
    config["models"]["actor_reference"] = None
    config["models"]["renderer_reference"] = None
    if args.stage == "score":
        config["judge"]["calibration"] = args.calibration or config["judge"]["calibration"]
        calibration = config["judge"]["calibration"]
        if not calibration or not Path(calibration).is_file() or not Path(str(calibration) + ".model.json").is_file():
            raise ValueError("Private scoring requires actual 8B calibration and its model identity")
    else:
        config["judge"]["calibration"] = None
    components = Components(config)
    if args.stage == "freeze":
        components.load_actor()
        components.load_encoders()
        components.load_judge(private=False)
        components.record_memory("sampler_freeze_loaded")
        result = freeze_comparison_plans(components, tasks, args.output, args.start, args.count)
    elif args.stage == "score":
        components.load_judge(private=True)
        components.record_memory("sampler_private_judge_loaded")
        result = score_comparison_shard(components, tasks, args.output, args.start, args.count, methods)
    else:
        components.load_renderer()
        components.record_memory("sampler_render_loaded")
        result = render_comparison_shard(components, tasks, args.output, args.start, args.count, methods)
    components.record_memory("sampler_shard_finished")
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
