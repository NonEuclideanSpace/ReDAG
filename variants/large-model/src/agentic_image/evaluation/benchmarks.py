import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    prompt: str
    family_id: str
    source_image: str | None = None
    user_controls: dict[str, Any] | None = None
    max_actions: int = 10
    max_renders: int = 3


@dataclass
class DeliveredImage:
    path: str | None
    costs: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_public_prompts(benchmark_path, output_path, benchmark="geneval2"):
    records = read_jsonl(benchmark_path)
    rows = [
        {
            "task_id": f"{benchmark}:{index}",
            "family_id": f"{benchmark}:{index}",
            "prompt": row["prompt"],
        }
        for index, row in enumerate(records)
    ]
    write_jsonl(output_path, rows)
    return str(Path(output_path).resolve())


def save_delivery(result, destination):
    if not isinstance(result, DeliveredImage):
        result = DeliveredImage(str(result) if result is not None else None)
    if result.path is None:
        return result
    from PIL import Image

    source = Path(result.path)
    if not source.is_file():
        raise FileNotFoundError(f"The delivered image is missing: {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination)
    return DeliveredImage(str(destination.resolve()), result.costs)


def run_geneval(benchmark_path, output_directory, generate: Callable, seeds=(101,)):
    records = read_jsonl(benchmark_path)
    output = Path(output_directory)
    manifests = []
    for seed in seeds:
        root = output / f"seed_{seed}"
        rows = []
        for index, record in enumerate(records):
            task = BenchmarkTask(f"geneval:{index}", record["prompt"], f"geneval:{index}")
            directory = root / f"{index:05d}"
            delivered = save_delivery(generate(task, seed), directory / "samples" / "00000.png")
            (directory / "samples").mkdir(parents=True, exist_ok=True)
            (directory / "metadata.jsonl").write_text(json.dumps(record, ensure_ascii=False))
            rows.append(
                {
                    "task_id": task.task_id,
                    "family_id": task.family_id,
                    "seed": seed,
                    "prompt": task.prompt,
                    "image_path": delivered.path,
                    "status": "delivered" if delivered.path else "no_image",
                    "costs": delivered.costs,
                }
            )
        write_jsonl(root / "generation_manifest.jsonl", rows)
        manifests.append(str(root / "generation_manifest.jsonl"))
    return manifests


def run_geneval2(benchmark_path, output_directory, generate: Callable, seeds=(101,)):
    records = read_jsonl(benchmark_path)
    if any(set(row) - {"task_id", "family_id", "prompt"} for row in records):
        raise ValueError(
            "Generate GenEval2 images from prepare_public_prompts output; private VQA fields belong only to the scorer"
        )
    if len({row["prompt"] for row in records}) != len(records):
        raise ValueError("GenEval2's official prompt map requires unique prompts")
    output = Path(output_directory)
    manifests = []
    for seed in seeds:
        root = output / f"seed_{seed}"
        image_map, rows = {}, []
        for index, record in enumerate(records):
            task = BenchmarkTask(
                record.get("task_id", f"geneval2:{index}"),
                record["prompt"],
                record.get("family_id", f"geneval2:{index}"),
            )
            delivered = save_delivery(generate(task, seed), root / "images" / f"{index:05d}.png")
            if delivered.path:
                image_map[task.prompt] = delivered.path
            rows.append(
                {
                    "task_id": task.task_id,
                    "family_id": task.family_id,
                    "seed": seed,
                    "prompt": task.prompt,
                    "image_path": delivered.path,
                    "status": "delivered" if delivered.path else "no_image",
                    "costs": delivered.costs,
                }
            )
        root.mkdir(parents=True, exist_ok=True)
        (root / "image_paths.json").write_text(json.dumps(image_map, ensure_ascii=False, indent=2))
        write_jsonl(root / "generation_manifest.jsonl", rows)
        manifests.append(str(root / "generation_manifest.jsonl"))
    return manifests


def load_magicbrush_sessions(records):
    sessions = {}
    for record in records:
        session_id = str(record["session_id"])
        sessions.setdefault(session_id, []).append(record)
    for session_id, turns in sessions.items():
        turns.sort(key=lambda row: int(row["turn_index"]))
        if [int(row["turn_index"]) for row in turns] != list(range(1, len(turns) + 1)):
            raise ValueError(
                f"MagicBrush session {session_id} must contain consecutive turns starting at one"
            )
        for turn in turns:
            if not Path(turn["source_image"]).is_file():
                raise FileNotFoundError(turn["source_image"])
    return sessions


def run_magicbrush(
    records, output_directory, generate: Callable, seeds=(101,), mask_provided=False
):
    if isinstance(records, (str, Path)):
        records = read_jsonl(records)
    sessions = load_magicbrush_sessions(records)
    output = Path(output_directory)
    manifests = []
    for seed in seeds:
        root = output / f"seed_{seed}"
        rows = []
        for session_id, turns in sessions.items():
            current = turns[0]["source_image"]
            for turn in turns:
                index = int(turn["turn_index"])
                controls = turn.get("user_controls") if mask_provided else None
                independent_task = BenchmarkTask(
                    f"magicbrush:{session_id}:{index}:independent",
                    turn["instruction"],
                    session_id,
                    turn["source_image"],
                    controls,
                )
                independent = save_delivery(
                    generate(independent_task, seed),
                    root / session_id / ("edit_1.png" if index == 1 else f"inde_{index}.png"),
                )
                rows.append(
                    {
                        **asdict(independent_task),
                        "track": "independent",
                        "turn_index": index,
                        "seed": seed,
                        "image_path": independent.path,
                        "costs": independent.costs,
                    }
                )
                if index == 1:
                    iterative = independent
                    iterative_source = current
                elif current is not None:
                    iterative_source = current
                    iterative_task = BenchmarkTask(
                        f"magicbrush:{session_id}:{index}:iterative",
                        turn["instruction"],
                        session_id,
                        current,
                        controls,
                    )
                    iterative = save_delivery(
                        generate(iterative_task, seed), root / session_id / f"iter_{index}.png"
                    )
                else:
                    iterative_source = None
                    iterative = DeliveredImage(None)
                rows.append(
                    {
                        "task_id": f"magicbrush:{session_id}:{index}:iterative",
                        "family_id": session_id,
                        "prompt": turn["instruction"],
                        "source_image": iterative_source,
                        "track": "iterative",
                        "turn_index": index,
                        "seed": seed,
                        "image_path": iterative.path,
                        "costs": iterative.costs,
                        "is_final_turn": index == len(turns),
                        "shared_first_turn": index == 1,
                    }
                )
                current = iterative.path
        write_jsonl(root / "generation_manifest.jsonl", rows)
        manifests.append(str(root / "generation_manifest.jsonl"))
    return manifests


def best_of_public(candidates):
    if not candidates:
        raise ValueError("Public best-of selection requires at least one delivered candidate")
    expected = candidates[0]["public_observations"]
    expected_ids = {item.get("check_id") for item in expected}
    for candidate in candidates:
        observations = candidate["public_observations"]
        if (
            len(observations) != len(expected)
            or {item.get("check_id") for item in observations} != expected_ids
        ):
            raise ValueError("Public selection must retain the same fixed CheckSpec denominator")

    def rank(candidate):
        observations = candidate["public_observations"]
        if not observations:
            return 0, 0.0, float(candidate.get("public_protection", 0.0))
        passes = sum(bool(item["known"] and item["passed"]) for item in observations)
        dense = sum(float(item["score"]) if item["known"] else 0.0 for item in observations) / len(
            observations
        )
        return passes, dense, float(candidate.get("public_protection", 0.0))

    return max(enumerate(candidates), key=lambda pair: (*rank(pair[1]), -pair[0]))[1]


def copy_official_targets(records, destination):
    destination = Path(destination)
    for session_id, turns in load_magicbrush_sessions(records).items():
        for turn in turns:
            target = Path(turn["target_image"])
            if not target.is_file():
                raise FileNotFoundError(target)
            output = destination / session_id / f"output{int(turn['turn_index'])}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            with Image.open(target) as image:
                image.convert("RGB").save(output)
    return destination
