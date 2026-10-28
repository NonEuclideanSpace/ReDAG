import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path


GENEVAL2_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DINO_REVISION = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
REVISIONS = {
    "geneval": "af4902f24d3ca90ebbb446dd9891a59e0f82725f",
    "geneval2": "a6e82d2289e8d418f27f0adee77908b07060eea3",
    "magicbrush": "d2d7f84d71ff23154e3f16f0c8f19392782bd201",
}


def run_official(
    benchmark, repository, arguments, python=None, model_snapshot=None, dino_repository=None
):
    if benchmark not in REVISIONS:
        raise ValueError(f"Unknown official benchmark: {benchmark}")
    command = [
        python or sys.executable,
        "-m",
        "agentic_image.evaluation.official",
        "--benchmark",
        benchmark,
        "--repository",
        str(Path(repository).resolve()),
    ]
    if model_snapshot:
        command.extend(["--model-snapshot", str(Path(model_snapshot).resolve())])
    if dino_repository:
        command.extend(["--dino-repository", str(Path(dino_repository).resolve())])
    command.extend(["--", *map(str, arguments)])
    environment = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = package_root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    subprocess.run(command, check=True, env=environment)
    return command


def score_geneval2(
    repository, benchmark_path, image_map_path, output_path, python=None, model_snapshot=None
):
    import numpy as np

    benchmark_path, image_map_path, output_path = map(
        lambda value: Path(value).resolve(), (benchmark_path, image_map_path, output_path)
    )
    with benchmark_path.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    images = json.loads(image_map_path.read_text())
    delivered = [record for record in records if record["prompt"] in images]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.stem + ".delivered.json")
    subset_path = output_path.with_name(output_path.stem + ".delivered.jsonl")
    with subset_path.open("w") as stream:
        for record in delivered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    partial = []
    if delivered:
        run_official(
            "geneval2",
            repository,
            [
                "--benchmark_data",
                subset_path,
                "--image_filepath_data",
                image_map_path,
                "--method",
                "soft_tifa_gm",
                "--output_file",
                partial_path,
            ],
            python=python,
            model_snapshot=model_snapshot,
        )
        partial = json.loads(partial_path.read_text())
        if len(partial) != len(delivered):
            raise ValueError("The official GenEval2 scorer did not return every delivered prompt")
    by_prompt = {record["prompt"]: scores for record, scores in zip(delivered, partial)}
    all_scores = [
        by_prompt.get(record["prompt"], [0.0] * len(record["vqa_list"])) for record in records
    ]
    per_prompt = []
    for scores in all_scores:
        values = np.asarray(scores, dtype=float)
        if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Official GenEval2 scores must be nonempty and nonnegative")
        per_prompt.append(0.0 if np.any(values == 0) else float(np.exp(np.log(values).mean())))
    output_path.write_text(json.dumps(all_scores))
    result = {
        "method": "soft_tifa_gm",
        "score": 100 * float(np.mean(per_prompt)),
        "prompts": len(records),
        "delivered": len(delivered),
        "no_image": len(records) - len(delivered),
        "per_prompt": per_prompt,
        "missing_image_score": 0.0,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(result, indent=2))
    return result


def score_geneval(repository, image_directory, output_path, python=None, detector_arguments=()):
    image_directory, output_path = map(
        lambda value: Path(value).resolve(), (image_directory, output_path)
    )
    manifest = [
        json.loads(line)
        for line in (image_directory / "generation_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if any(row["image_path"] for row in manifest):
        run_official(
            "geneval",
            repository,
            [image_directory, "--outfile", output_path, *detector_arguments],
            python=python,
        )
        records = [
            json.loads(line) for line in output_path.read_text().splitlines() if line.strip()
        ]
    else:
        records = []
    for row in manifest:
        if row["image_path"] is not None:
            continue
        index = int(row["task_id"].split(":")[-1])
        directory = image_directory / f"{index:05d}"
        metadata = json.loads((directory / "metadata.jsonl").read_text())
        records.append(
            {
                "filename": str(directory / "samples" / "00000.png"),
                "tag": metadata["tag"],
                "prompt": metadata["prompt"],
                "correct": False,
                "reason": "no_image",
                "metadata": json.dumps(metadata),
                "details": "{}",
            }
        )
    if len(records) != len(manifest):
        raise ValueError("GenEval scoring must account for every requested task")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    subprocess.run(
        [
            python or sys.executable,
            str(Path(repository).resolve() / "evaluation" / "summary_scores.py"),
            str(output_path),
        ],
        check=True,
    )
    return {
        "results": str(output_path),
        "prompts": len(manifest),
        "no_image": sum(row["image_path"] is None for row in manifest),
    }


def _argument(arguments, name):
    return arguments[arguments.index(name) + 1] if name in arguments else None


def _pin_geneval2(model_snapshot, arguments):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    original_processor = AutoProcessor.from_pretrained
    original_model = Qwen3VLForConditionalGeneration.from_pretrained

    def model_loader(name, *args, **kwargs):
        if name == "Qwen/Qwen3-VL-8B-Instruct":
            if model_snapshot:
                name = model_snapshot
                kwargs["local_files_only"] = True
            else:
                kwargs["revision"] = GENEVAL2_MODEL_REVISION
        return original_model(name, *args, **kwargs)

    def processor_loader(name, *args, **kwargs):
        if name == "Qwen/Qwen3-VL-8B-Instruct":
            if model_snapshot:
                name = model_snapshot
                kwargs["local_files_only"] = True
            else:
                kwargs["revision"] = GENEVAL2_MODEL_REVISION
        processor = original_processor(name, *args, **kwargs)
        benchmark_path = _argument(arguments, "--benchmark_data")
        output = _argument(arguments, "--output_file")
        if benchmark_path and output:
            with open(benchmark_path) as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            answer_sets = {"yes": ["Yes", "yes", " yes", " Yes"]}
            numbers = dict(
                zip(
                    "one two three four five six seven eight nine ten".split(),
                    map(str, range(1, 11)),
                )
            )
            for record in records:
                for question, answer in record["vqa_list"]:
                    if question.startswith("How many"):
                        numeric = numbers.get(answer, "other")
                        answer_sets[answer] = [
                            answer,
                            answer.capitalize(),
                            " " + answer,
                            " " + answer.capitalize(),
                            numeric,
                            " " + numeric,
                        ]
            audit = {}
            for key, answers in answer_sets.items():
                ids = [processor.tokenizer.encode(answer)[0] for answer in answers]
                audit[key] = {
                    "answers": answers,
                    "first_token_ids": ids,
                    "duplicate_ids": sorted({value for value in ids if ids.count(value) > 1}),
                    "official_duplicate_sum_preserved": True,
                }
            Path(str(output) + ".token_ids.json").write_text(json.dumps(audit, indent=2))
        return processor

    AutoProcessor.from_pretrained = staticmethod(processor_loader)
    Qwen3VLForConditionalGeneration.from_pretrained = staticmethod(model_loader)


def _pin_dino(repository):
    import torch

    original = torch.hub.load

    def loader(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir.startswith("facebookresearch/dino"):
            if repository:
                repo_or_dir = repository
                kwargs["source"] = "local"
            else:
                repo_or_dir = f"facebookresearch/dino:{DINO_REVISION}"
                kwargs["skip_validation"] = True
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(REVISIONS), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--model-snapshot")
    parser.add_argument("--dino-repository")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    root = Path(args.repository).resolve()
    if args.benchmark == "geneval":
        script = root / "evaluation" / "evaluate_images.py"
    elif args.benchmark == "geneval2":
        if _argument(arguments, "--method") != "soft_tifa_gm":
            raise ValueError("The main GenEval2 report requires the official soft_tifa_gm method")
        _pin_geneval2(args.model_snapshot, arguments)
        script = root / "evaluation.py"
    else:
        _pin_dino(args.dino_repository)
        script = root / "evaluation" / "image_eval.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    sys.path.insert(0, str(script.parent))
    os.chdir(root)
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
