import argparse
import json
from pathlib import Path

import yaml

from .config import load_config, validate_config


def parser():
    root = argparse.ArgumentParser(prog="agentic-image")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", default="configs/a6000.yaml")
    shared.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("hardware", parents=[shared])
    prepare = commands.add_parser("prepare-models", parents=[shared])
    prepare.add_argument(
        "--component", choices=["actor", "judge", "renderer", "all"], default="all"
    )
    probe = commands.add_parser("preflight", parents=[shared])
    probe.add_argument("--tasks", required=True)
    probe.add_argument("--component", choices=["actor", "renderer", "judge", "all"], default="all")
    probe.add_argument("--cycles", type=int, default=1)
    ingest = commands.add_parser("import-data", parents=[shared])
    ingest.add_argument("--source", choices=["magicbrush", "omniedit"], required=True)
    ingest.add_argument("--split", default="train")
    ingest.add_argument("--limit", type=int, default=12000)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--resume", action="store_true")
    families = commands.add_parser("split-data", parents=[shared])
    families.add_argument("--input", required=True)
    families.add_argument("--output", required=True)
    families.add_argument("--reviewed-pairs")
    families.add_argument("--no-visual-review", action="store_true")
    pairs = commands.add_parser("build-pairs", parents=[shared])
    pairs.add_argument("--candidates", required=True)
    pairs.add_argument("--annotations", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--limit", type=int, default=12000)
    annotation = commands.add_parser("annotation-packet", parents=[shared])
    annotation.add_argument("--candidates", required=True)
    annotation.add_argument("--output", required=True)
    annotation.add_argument("--limit", type=int, default=200)
    annotation.add_argument("--split", default="train")
    proposals = commands.add_parser("propose-annotations", parents=[shared])
    proposals.add_argument("--candidates", required=True)
    proposals.add_argument("--output", required=True)
    proposals.add_argument("--limit", type=int, default=12000)
    proposals.add_argument("--split", default="train")
    proposals.add_argument("--no-captions", action="store_true")
    proposals.add_argument("--no-geometry", action="store_true")
    pair_audit = commands.add_parser("pair-audit", parents=[shared])
    pair_audit.add_argument("--pairs", required=True)
    pair_audit.add_argument("--annotations", required=True)
    pair_audit.add_argument("--output", required=True)
    score_audit = commands.add_parser("score-pair-audit", parents=[shared])
    score_audit.add_argument("--packet", required=True)
    score_audit.add_argument("--reviews", required=True)
    captions = commands.add_parser("caption-candidates", parents=[shared])
    captions.add_argument("--input", required=True)
    captions.add_argument("--annotations", required=True)
    captions.add_argument("--output", required=True)
    captions.add_argument("--limit", type=int, default=1600)
    appearance = commands.add_parser("appearance-candidates", parents=[shared])
    appearance.add_argument("--input", required=True)
    appearance.add_argument("--annotations", required=True)
    appearance.add_argument("--output", required=True)
    appearance.add_argument("--fraction", type=float, default=0.1)
    repairs = commands.add_parser("build-repairs", parents=[shared])
    repairs.add_argument("--tasks", required=True)
    repairs.add_argument("--output", required=True)
    repairs.add_argument("--renderer-version", required=True)
    repairs.add_argument("--sampler-id", required=True)
    repairs.add_argument("--max-states", type=int, default=1024)
    injection = commands.add_parser("build-injections", parents=[shared])
    injection.add_argument("--tasks", required=True)
    injection.add_argument("--specifications", required=True)
    injection.add_argument("--output", required=True)
    injection.add_argument("--renderer-version", required=True)
    injection.add_argument("--sampler-id", required=True)
    injection.add_argument("--max-cases", type=int, default=256)
    refresh = commands.add_parser("build-refresh", parents=[shared])
    refresh.add_argument("--tasks", required=True)
    refresh.add_argument("--output", required=True)
    refresh.add_argument("--renderer-version", required=True)
    refresh.add_argument("--sampler-id", required=True)
    refresh.add_argument("--round-index", type=int, choices=[0, 1], required=True)
    refresh.add_argument("--max-tasks", type=int, default=256)
    merge = commands.add_parser("merge-records", parents=[shared])
    merge.add_argument("--input", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    protocol = commands.add_parser("build-protocol", parents=[shared])
    protocol.add_argument("--tasks", required=True)
    protocol.add_argument("--output", required=True)
    protocol.add_argument("--limit", type=int, default=1024)
    complex_plans = commands.add_parser("build-complex-plans", parents=[shared])
    complex_plans.add_argument("--tasks", required=True)
    complex_plans.add_argument("--output", required=True)
    complex_plans.add_argument("--limit", type=int, default=96)
    sft = commands.add_parser("train-sft", parents=[shared])
    sft.add_argument("--stage", choices=["renderer", "protocol", "actor", "refresh"], required=True)
    sft.add_argument("--updates", type=int)
    sft.add_argument("--resume", action="store_true")
    commands.add_parser("freeze-reference", parents=[shared])
    rl = commands.add_parser("train-rl", parents=[shared])
    rl.add_argument("--mode", choices=["agent", "renderer", "joint"], default="joint")
    rl.add_argument("--rounds", type=int)
    rl.add_argument("--resume", action="store_true")
    rl.add_argument("--initialize-from")
    rl.add_argument("--pilot", action="store_true")
    generate = commands.add_parser("generate", parents=[shared])
    generate.add_argument("--request", required=True)
    generate.add_argument("--source")
    generate.add_argument("--reference")
    generate.add_argument("--output", required=True)
    generate.add_argument("--seed", type=int, default=101)
    calibration = commands.add_parser("calibrate", parents=[shared])
    calibration.add_argument("--records", required=True)
    calibration.add_argument("--output")
    calibration_scores = commands.add_parser("score-calibration", parents=[shared])
    calibration_scores.add_argument("--tasks", required=True)
    calibration_scores.add_argument("--cases", required=True)
    calibration_scores.add_argument("--output", required=True)
    tasks = commands.add_parser("build-tasks", parents=[shared])
    tasks.add_argument("--candidates", required=True)
    tasks.add_argument("--annotations", required=True)
    tasks.add_argument("--output", required=True)
    tasks.add_argument("--split", default="development")
    tasks.add_argument("--limit", type=int)
    sampler = commands.add_parser("sample-comparison", parents=[shared])
    sampler.add_argument("--tasks", required=True)
    sampler.add_argument("--output", required=True)
    sampler.add_argument("--limit", type=int, default=64)
    sampler.add_argument("--fallbacks", action="store_true")
    decision = commands.add_parser("select-sampler", parents=[shared])
    decision.add_argument("--records", required=True)
    decision.add_argument("--candidate", default="sde50")
    decision.add_argument("--reference", default="ode50")
    statistics = commands.add_parser("compare", parents=[shared])
    statistics.add_argument("--records", required=True)
    statistics.add_argument("--output", required=True)
    statistics.add_argument("--metric", default="value")
    benchmark = commands.add_parser("benchmark", parents=[shared])
    benchmark.add_argument(
        "--benchmark", choices=["geneval", "geneval2", "magicbrush"], required=True
    )
    benchmark.add_argument("--input", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--seeds", type=int, nargs="+", default=[101])
    benchmark.add_argument("--method", choices=["B0", "B1", "B2", "B3", "B4", "B5"], default="B5")
    benchmark.add_argument("--renders", type=int, choices=[1, 3], default=1)
    benchmark.add_argument("--mask-provided", action="store_true")
    return root


def configure(arguments):
    config = load_config(arguments.config)
    for override in arguments.set:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Configuration overrides use KEY=VALUE")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown configuration key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return validate_config(config)


def generate(components, request, source, reference, seed, output):
    from .checks.public import PublicTaskBuilder
    from .core.geometry import letterbox, make_frame, normalize_image
    from .runtime.rollout import RolloutCollector
    from PIL import Image

    def import_image(path, role):
        if path is None:
            return None
        with Image.open(path) as image:
            image = normalize_image(image)
            frame = make_frame(
                (image.height, image.width), tuple(components.config["canvas"]), f"frame:{role}"
            )
            return components.artifacts.put_image(letterbox(image, frame), frame=frame).artifact_id

    components.load_all(private=False)
    source_id = import_image(source, "source")
    reference_id = import_image(reference, "reference")
    task = PublicTaskBuilder(components.judge, components.artifacts).build(
        task_id=f"inference:{seed}",
        text=request,
        source_image_id=source_id,
        appearance_image_ids=(reference_id,) if reference_id else (),
        split_id="inference",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = RolloutCollector(components).infer(
        task, seed, output.with_suffix("").with_name(output.stem + "_episode")
    )
    image_id = environment.state.delivered_image_id
    if image_id:
        components.artifacts.get_image(image_id).save(output)
    return {
        "image": str(output.resolve()) if image_id else None,
        "budget": environment.state.budget.model_dump(),
        "protocol_failure": environment.state.protocol_failure,
    }


def main():
    arguments = parser().parse_args()
    config = configure(arguments)
    command = arguments.command
    from .runtime.components import Components
    from .runtime.rollout import read_jsonl
    from .learning.transaction import atomic_json

    components = Components(config)
    result = None
    if command == "hardware":
        from .runtime.preflight import hardware_report

        result = hardware_report(config)
    elif command == "prepare-models":
        from .actor import prepare_nf4
        from .checks.judge import FrozenJudge
        from .runtime.components import actor_config, judge_config
        from huggingface_hub import snapshot_download

        if arguments.component in {"actor", "all"}:
            prepare_nf4(actor_config(config))
        if arguments.component in {"judge", "all"}:
            model = FrozenJudge.load(judge_config(config), quantize=True)
            del model
        if arguments.component in {"renderer", "all"}:
            snapshot_download(
                config["models"]["renderer"],
                revision=config["models"]["renderer_revision"],
                allow_patterns=[
                    "model_index.json",
                    "LICENSE.md",
                    "README.md",
                    "scheduler/*",
                    "text_encoder/*",
                    "tokenizer/*",
                    "transformer/*",
                    "vae/*",
                ],
            )
        result = {"prepared": arguments.component}
    elif command == "preflight":
        from .runtime.preflight import run_preflight

        result = run_preflight(components, arguments.tasks, arguments.component, arguments.cycles)
    elif command == "import-data":
        from .data.ingest import stream_import

        result = stream_import(
            arguments.source,
            arguments.output,
            components.artifacts,
            arguments.limit,
            arguments.split,
            resume=arguments.resume,
        )
    elif command == "split-data":
        from .data.families import partition_families
        from .checks.tools import ToolClient

        tools = ToolClient(config["judge"]["tool_url"], config["judge"]["timeout"])

        def features(image):
            reference = components.artifacts.put_image(image)
            result = tools.call("features", image_id=reference.artifact_id)
            if not result.get("known", False):
                raise RuntimeError(
                    "Near-duplicate review did not obtain valid frozen visual features"
                )
            return result["cls"]

        result = partition_families(
            arguments.input,
            arguments.output,
            components.artifacts,
            feature_fn=None if arguments.no_visual_review else features,
            reviewed_pairs_path=arguments.reviewed_pairs,
        )
    elif command == "build-pairs":
        from .data.paired import build_aligned_pairs

        components.load_actor()
        components.load_encoders()
        components.load_judge(private=False)
        result = build_aligned_pairs(
            components,
            arguments.candidates,
            arguments.annotations,
            arguments.output,
            max_candidates=arguments.limit,
        )
    elif command == "annotation-packet":
        from .data.annotation import prepare_annotation_packet

        result = prepare_annotation_packet(
            arguments.candidates,
            arguments.output,
            components.artifacts,
            arguments.limit,
            arguments.split,
        )
    elif command == "propose-annotations":
        from .data.proposals import propose_annotations

        result = propose_annotations(
            components,
            arguments.candidates,
            arguments.output,
            arguments.limit,
            arguments.split,
            not arguments.no_captions,
            not arguments.no_geometry,
        )
    elif command == "pair-audit":
        from .data.audit import prepare_pair_audit

        result = prepare_pair_audit(
            arguments.pairs,
            arguments.annotations,
            arguments.output,
            components.artifacts,
            seed=config["seed"],
        )
    elif command == "score-pair-audit":
        from .data.audit import evaluate_pair_audit

        result = evaluate_pair_audit(arguments.packet, arguments.reviews)
    elif command == "caption-candidates":
        from .data.captions import caption_candidates

        result = caption_candidates(
            arguments.input, arguments.annotations, arguments.output, arguments.limit
        )
    elif command == "appearance-candidates":
        from .data.appearance import appearance_candidates

        result = appearance_candidates(
            arguments.input,
            arguments.annotations,
            arguments.output,
            components.artifacts,
            arguments.fraction,
            config["seed"],
        )
    elif command == "merge-records":
        from .data.io import write_jsonl

        rows = [row for path in arguments.input for row in read_jsonl(path)]
        identifiers = [
            row.get("record_id", row.get("sample_id", row.get("task_id"))) for row in rows
        ]
        if any(identifier is None for identifier in identifiers) or len(identifiers) != len(
            set(identifiers)
        ):
            raise ValueError("Merged records must retain unique IDs")
        write_jsonl(arguments.output, rows)
        result = {"records": len(rows), "output": arguments.output}
    elif command == "build-protocol":
        from .data.repairs import protocol_records

        components.load_actor()
        components.load_encoders()
        components.load_judge(private=False)
        result = protocol_records(
            components, arguments.tasks, arguments.output, max_records=arguments.limit
        )
    elif command == "build-complex-plans":
        from .data.repairs import complex_plan_records

        components.load_actor()
        components.load_encoders()
        components.load_judge(private=False)
        result = complex_plan_records(
            components, arguments.tasks, arguments.output, max_tasks=arguments.limit
        )
    elif command == "build-repairs":
        from .data.repairs import collect_repairs

        components.load_all()
        result = collect_repairs(
            components,
            arguments.tasks,
            arguments.output,
            arguments.renderer_version,
            arguments.sampler_id,
            max_states=arguments.max_states,
        )
    elif command == "build-injections":
        from .data.interventions import collect_injections

        components.load_all()
        result = collect_injections(
            components,
            arguments.tasks,
            arguments.specifications,
            arguments.output,
            arguments.renderer_version,
            arguments.sampler_id,
            max_cases=arguments.max_cases,
        )
    elif command == "build-refresh":
        from .data.interventions import collect_self_corrections

        components.load_all()
        result = collect_self_corrections(
            components,
            arguments.tasks,
            arguments.output,
            arguments.renderer_version,
            arguments.sampler_id,
            arguments.round_index,
            max_tasks=arguments.max_tasks,
        )
    elif command == "train-sft":
        from .runtime.training import train_actor_sft, train_renderer_sft

        if arguments.stage == "renderer":
            result = train_renderer_sft(components, arguments.updates, arguments.resume)
        else:
            result = train_actor_sft(
                components, arguments.stage, arguments.updates, arguments.resume
            )
    elif command == "freeze-reference":
        components.load_actor()
        components.load_renderer()
        if not config["models"]["actor_policy"] or not config["models"]["renderer_policy"]:
            raise ValueError("Load the completed SFT policies before freezing references")
        folder = components.directory / "sft_reference"
        actor_path = components.actor.freeze_reference(folder / "actor")
        components.renderer.freeze_reference()
        components.renderer.save_adapters(folder / "renderer")
        result = {
            "actor_reference": str(actor_path),
            "renderer_reference": str((folder / "renderer" / "reference.pt").resolve()),
        }
        atomic_json(folder / "references.json", result)
        config["models"].update(result)
        prepared_config = components.directory / "rl_config.yaml"
        prepared_config.write_text(yaml.safe_dump(config, sort_keys=False))
        result["rl_config"] = str(prepared_config.resolve())
    elif command == "train-rl":
        from .runtime.training import train_rl

        result = train_rl(
            components,
            arguments.mode,
            arguments.rounds,
            arguments.resume,
            arguments.initialize_from,
            arguments.pilot,
        )
    elif command == "generate":
        result = generate(
            components,
            arguments.request,
            arguments.source,
            arguments.reference,
            arguments.seed,
            arguments.output,
        )
    elif command == "calibrate":
        from .checks.calibration import calibrate
        from .runtime.components import judge_config

        records = read_jsonl(arguments.records)
        identity = judge_config(config).calibration_identity
        identities = [row.get("judge_identity") for row in records]
        if identity["model_id"] == "Qwen/Qwen3-VL-8B-Instruct" and (
            not identities or any(value != identity for value in identities)
        ):
            raise ValueError("The 8B calibration requires scores from this exact frozen judge")
        review_mode = config.get("review_mode", "human")
        result = calibrate(records, review_mode=review_mode)
        path = Path(arguments.output or config["judge"]["calibration"])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
        atomic_json(Path(str(path) + ".review.json"), {
            "review_mode": review_mode,
            "human_reviewed": review_mode == "human",
            "record_count": len(records),
            "review_evidence": [row.get("review_evidence") for row in records],
        })
        if identities and all(value == identity for value in identities):
            atomic_json(Path(str(path) + ".model.json"), identity)
    elif command == "score-calibration":
        from .data.io import write_jsonl
        from .runtime.rollout import read_tasks

        components.load_judge()
        components.judge.thresholds = {}
        tasks = {task.task_id: task for task in read_tasks(arguments.tasks)}
        records = []
        for case in read_jsonl(arguments.cases):
            task = tasks[case["task_id"]]
            if task.split_id != "calibration":
                raise ValueError("Judge calibration must use reserved calibration families")
            scores = components.evaluator.score_image(task, case["image_id"])
            evaluation = components.evaluator.evaluations[task.task_id]
            kinds = {
                atom.eval_id: atom.expected_value.get("kind", atom.checker_route)
                if isinstance(atom.expected_value, dict)
                else atom.checker_route
                for atom in evaluation.eval_atoms
            }
            for score in scores:
                label = case.get("labels", {}).get(score.atom_id)
                records.append(
                    {
                        "task_id": task.task_id,
                        "family_id": task.family_id,
                        "atom_id": score.atom_id,
                        "kind": kinds[score.atom_id],
                        "partition": case["partition"],
                        "value": score.value,
                        "known": score.known,
                        "exact_pass": score.exact_pass,
                        "label": label,
                        "adjudicated": bool(case.get("adjudicated", False) and label is not None),
                        "review_type": case.get("review_type"),
                        "human_reviewed": case.get("human_reviewed"),
                        "review_evidence": case.get("review_evidence", {}).get(score.atom_id),
                        "judge_identity": components.judge.config.calibration_identity,
                    }
                )
        write_jsonl(arguments.output, records)
        result = {"atoms": len(records), "output": arguments.output}
    elif command == "build-tasks":
        from .data.paired import build_task_pool

        components.load_judge(private=False)
        result = build_task_pool(
            components,
            arguments.candidates,
            arguments.annotations,
            arguments.output,
            split=arguments.split,
            limit=arguments.limit,
        )
    elif command == "sample-comparison":
        from .runtime.sampler import sample_comparison

        components.load_all()
        result = sample_comparison(
            components, arguments.tasks, arguments.output, arguments.limit, arguments.fallbacks
        )
    elif command == "select-sampler":
        from .evaluation.statistics import sampler_quality_decision

        if arguments.candidate in {"sde20", "sde28"}:
            selection_path = components.directory / "sampler_selection.json"
            selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
            if (
                not selection.get("accepted")
                or selection.get("candidate") != arguments.reference
                or arguments.reference not in {"sde50", "sde50_eta015", "sde80_eta015"}
                or selection.get("renderer_policy") != config["models"]["renderer_policy"]
            ):
                raise ValueError(
                    "Shorter chains require an already qualified stochastic reference from the same renderer"
                )
        result = sampler_quality_decision(
            read_jsonl(arguments.records),
            arguments.candidate,
            arguments.reference,
            require_faster=arguments.candidate in {"sde20", "sde28"},
        )
        result["candidate"] = arguments.candidate
        result["reference"] = arguments.reference
        parameters = {
            "sde50": (50, 0.3),
            "sde28": (28, 0.3),
            "sde20": (20, 0.3),
            "sde50_eta015": (50, 0.15),
            "sde80_eta015": (80, 0.15),
        }
        if arguments.candidate not in parameters:
            raise ValueError("Select a registered stochastic sampler for training")
        steps, eta = parameters[arguments.candidate]
        result.update(
            steps=steps,
            eta=eta,
            cfg_scale=config["renderer"]["cfg_scale"],
            accepted=result["status"] == "accepted",
            renderer_policy=config["models"]["renderer_policy"],
        )
        atomic_json(components.directory / "sampler_selection.json", result)
    elif command == "compare":
        from .evaluation.statistics import compare_main_methods

        result = compare_main_methods(read_jsonl(arguments.records), metric=arguments.metric)
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(arguments.output, result)
    elif command == "benchmark":
        from .runtime.benchmarks import run_benchmark

        if arguments.method in {"B0", "B1"}:
            components.load_renderer()
            components.load_judge(private=False)
        else:
            components.load_all(private=False)
        result = run_benchmark(
            components,
            arguments.benchmark,
            arguments.input,
            arguments.output,
            arguments.seeds,
            arguments.method,
            arguments.renders,
            arguments.mask_provided,
        )
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
