import json
import time
from pathlib import Path

from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json

from .rollout import RolloutCollector, read_tasks


def sample_comparison(components, tasks_path, output_directory, limit=64, fallbacks=False):
    tasks = read_tasks(tasks_path)[:limit]
    if any(task.split_id == "train" for task in tasks):
        raise ValueError("Sampler selection requires held-out development tasks")
    if len(tasks) < limit:
        raise ValueError("Provide the complete registered development task count")
    configurations = [
        ("ode50", 50, 0.0),
        ("sde50", 50, 0.3),
        ("sde28", 28, 0.3),
        ("sde20", 20, 0.3),
    ]
    if fallbacks:
        configurations = [("sde50_eta015", 50, 0.15), ("sde80_eta015", 80, 0.15)]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    collector = RolloutCollector(components)
    rows = []
    for index, task in enumerate(tasks):
        environment = collector.environment(task, f"sampler:{index}")
        for attempt in range(2):
            snapshot = collector.snapshot(environment)
            sample = components.actor.sample(
                snapshot, stream_seed(101, index, attempt, "plan"), "sampler-teacher"
            )
            environment.step(sample.text)
            if environment.state.plan:
                break
        if not environment.state.plan:
            raise RuntimeError(
                f"Task {task.task_id} has no executable public plan; repair this input before the comparison"
            )
        plan = environment.state.plan
        bundle = components.compiler.compile(plan, task.user_request.source_image_id)
        atomic_json(output / f"plan_{index:03d}.json", plan.model_dump(mode="json"))
        for seed in (101, 211):
            for method, steps, eta in configurations:
                directory = output / method
                directory.mkdir(exist_ok=True)
                path = directory / f"{index:03d}_{seed}.png"
                record_path = path.with_suffix(".json")
                if record_path.is_file():
                    rows.append(json.loads(record_path.read_text()))
                    continue
                started = time.monotonic()
                if eta == 0:
                    latent = components.renderer.rollout_ode(bundle, seed, steps)
                else:
                    trace = components.renderer.rollout(
                        bundle, seed, steps=steps, eta=eta, behavior_version="sampler-comparison"
                    )
                    latent = trace.final_latent
                    trace.save(path.with_suffix(".pt"))
                image = components.encoders.decode_image(latent, bundle.canvas_hw)
                image.save(path)
                reference = components.artifacts.put_image(
                    image, producer_id=f"sampler:{method}:{index}:{seed}"
                )
                reward = components.evaluator.evaluate(
                    task, reference.artifact_id, renders=1, inspects=0, actions=2, invalids=0
                )
                row = {
                    "method": method,
                    "training_seed": components.config["seed"],
                    "task_id": task.task_id,
                    "family_id": task.family_id,
                    "eval_seed": seed,
                    "success": float(reward.success),
                    "protection": reward.preservation,
                    "dense": reward.dense,
                    "severe_defect": None,
                    "seconds": time.monotonic() - started,
                    "image": str(path.resolve()),
                    "steps": steps,
                    "eta": eta,
                    "cfg_scale": bundle.cfg_scale,
                    "bundle_id": bundle.bundle_id,
                }
                atomic_json(record_path, row)
                rows.append(row)
    with (output / "comparison.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return {
        "renders": len(rows),
        "records": str((output / "comparison.jsonl").resolve()),
        "remaining": "Fill severe_defect with blinded image-quality judgments before select-sampler",
    }
