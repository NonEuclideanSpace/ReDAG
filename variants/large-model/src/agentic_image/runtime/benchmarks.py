import json
from contextlib import ExitStack, nullcontext
from functools import wraps
from pathlib import Path
from time import perf_counter

import torch
import numpy as np
from PIL import Image

from agentic_image.checks.public import PublicTaskBuilder
from agentic_image.checks.reward import preservation_score
from agentic_image.core.geometry import letterbox, make_frame, normalize_image
from agentic_image.core.types import Canvas, PlanSpec, PublicTask, ReferenceBinding, TaskLimits
from agentic_image.evaluation.benchmarks import (
    DeliveredImage,
    best_of_public,
    prepare_public_prompts,
    run_geneval,
    run_geneval2,
    run_magicbrush,
)
from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json

from .rollout import RolloutCollector


class BenchmarkCosts:
    def __init__(self, components):
        self.components = components
        self.values = {
            "actual_renders": 0,
            "actual_render_steps": 0,
            "actual_dit_forwards": 0,
            "sampled_actions": 0,
            "sampled_output_tokens": 0,
            "encoder_cache_hits": 0,
            "encoder_cache_misses": 0,
            "public_check_cache_hits": 0,
            "tool_calls": 0,
        }
        self.gpu_seconds = {}
        self.stack = ExitStack()

    def wrap(self, owner, name, category=None, callback=None):
        original = getattr(owner, name)
        had_instance_value = name in vars(owner)
        instance_value = vars(owner).get(name)

        @wraps(original)
        def measured(*args, **kwargs):
            device = torch.device(self.components.config["devices"][category]) if category else None
            use_cuda = device is not None and device.type == "cuda"
            if use_cuda:
                with torch.cuda.device(device):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
            try:
                result = original(*args, **kwargs)
                if callback is not None:
                    callback(result)
                return result
            finally:
                if use_cuda:
                    with torch.cuda.device(device):
                        end.record()
                        end.synchronize()
                        self.gpu_seconds[category] = (
                            self.gpu_seconds.get(category, 0.0) + start.elapsed_time(end) / 1000
                        )

        setattr(owner, name, measured)
        self.stack.callback(
            setattr, owner, name, instance_value
        ) if had_instance_value else self.stack.callback(delattr, owner, name)

    def add(self, key, amount=1):
        self.values[key] += amount

    def record_sample(self, sample):
        self.add("sampled_actions")
        self.add("sampled_output_tokens", sample.token_ids.numel())

    def record_render(self, trace):
        self.add("actual_renders")
        self.add("actual_render_steps", trace.sigmas.numel() - 1)

    def __enter__(self):
        self.started = perf_counter()
        components = self.components
        self.initial_public_cache = len(components.observer.cache)
        self.initial_judge_forwards = components.judge.forwards
        if components.actor is not None:
            self.wrap(components.actor, "prepare", "actor")
            self.wrap(
                components.actor,
                "sample",
                "actor",
                self.record_sample,
            )
        self.wrap(components.renderer, "rollout", "renderer", self.record_render)
        self.wrap(
            components.renderer,
            "predict_branch",
            callback=lambda result: self.add("actual_dit_forwards"),
        )
        self.wrap(components.encoders, "encode_text", "encoders")
        self.wrap(components.encoders, "encode_image", "encoders")
        self.wrap(components.encoders, "decode_image", "encoders")
        self.wrap(
            components.encoders.cache,
            "get",
            callback=lambda result: self.add(
                "encoder_cache_hits" if result is not None else "encoder_cache_misses"
            ),
        )
        self.wrap(components.judge, "answer_probability", "judge")
        self.wrap(components.judge, "generate_json", "judge")
        self.wrap(
            components.observer,
            "check",
            callback=lambda result: self.add("public_checks", len(result)),
        )
        self.values["public_checks"] = 0
        if components.observer.tools is not None:
            self.wrap(
                components.observer.tools, "call", callback=lambda result: self.add("tool_calls")
            )
        return self

    def __exit__(self, *exception):
        self.stack.close()
        self.values["wall_seconds"] = perf_counter() - self.started
        self.values["public_check_cache_hits"] = max(
            0,
            self.values["public_checks"]
            - len(self.components.observer.cache)
            + self.initial_public_cache,
        )
        self.values["judge_forwards"] = self.components.judge.forwards - self.initial_judge_forwards
        self.values["gpu_seconds"] = sum(self.gpu_seconds.values())
        self.values["gpu_seconds_by_service"] = self.gpu_seconds
        self.values["gpu_seconds_scope"] = (
            "CUDA event elapsed time for local model services; external tool server excluded"
        )
        return False


def fixed_plan(task):
    request = task.user_request
    controls = request.controls
    references = controls.get("references")
    if references is None:
        references = [
            ReferenceBinding(binding_id=f"reference:{index}", image_id=image_id, scope="global")
            for index, image_id in enumerate(request.appearance_image_ids)
        ]
    plan = PlanSpec(
        global_text=request.text,
        entities=tuple(controls.get("entities", ())),
        regions=tuple(controls.get("regions", ())),
        references=tuple(references),
        source_image_id=request.source_image_id,
    )
    if any(binding.image_id not in request.appearance_image_ids for binding in plan.references):
        raise ValueError("Fixed-flow references must be supplied in the public request")
    if (
        len(plan.entities) > task.limits.max_entities
        or len(plan.regions) > task.limits.max_active_regions
    ):
        raise ValueError("Fixed-flow controls exceed the shared condition budget")
    image_count = len({binding.image_id for binding in plan.references}) + bool(
        request.source_image_id
    )
    if image_count > task.limits.max_references:
        raise ValueError("Fixed-flow references exceed the shared condition budget")
    return plan


def public_protection(components, task, image_id, observations):
    source_id = task.user_request.source_image_id
    mask_id = task.user_request.controls.get("protected_mask_id")
    if source_id and mask_id:
        mask = components.artifacts.get_array(mask_id).copy()
        frame = components.artifacts.get_ref(source_id).frame
        if frame is not None:
            valid = np.zeros(frame.canvas_hw, dtype=np.float32)
            px, py = frame.padding_xy
            height, width = frame.resized_hw
            valid[py : py + height, px : px + width] = 1
            if mask.shape != valid.shape:
                raise ValueError("Public protection must use the original source canvas frame")
            mask *= valid
        features = {}
        if mask.any():
            if components.observer.tools is None:
                raise RuntimeError("Public preservation requires the frozen DINO feature service")
            effective = components.artifacts.put_array(
                mask, frame=frame, metadata={"visibility": "public", "role": "effective_protection"}
            )
            features = components.observer.tools.call(
                "preservation",
                source_image_id=source_id,
                image_id=image_id,
                mask_id=effective.artifact_id,
            )
        return preservation_score(
            components.artifacts.get_image(source_id),
            components.artifacts.get_image(image_id),
            mask,
            **features,
        )
    preservation_ids = {
        atom.check_id
        for atom in task.check_spec.atoms
        if isinstance(atom.expected_value, dict)
        and atom.expected_value.get("kind") == "preservation"
    }
    protected = [
        observation for observation in observations if observation.check_id in preservation_ids
    ]
    return (
        sum(observation.score if observation.known else 0 for observation in protected)
        / len(protected)
        if protected
        else 0.0
    )


def run_fixed_flow(components, task, seed, folder, method, renders):
    if renders > task.limits.max_renders:
        raise ValueError("Fixed-flow sampling exceeds the shared render budget")
    folder.mkdir(parents=True, exist_ok=True)
    plan = fixed_plan(task)
    components.compiler.validate_plan(plan, task.user_request.source_image_id)
    bundle = components.compiler.compile(plan, task.user_request.source_image_id)
    candidates = []
    mode = components.renderer.base_policy() if method == "B0" else nullcontext()
    with mode:
        for index in range(renders):
            result = components.render_bridge.render(
                bundle,
                stream_seed(seed, task.task_id, "fixed", index),
                f"benchmark:{method}:{folder.resolve()}:{index}",
            )
            observations = components.observer.check(task, result.image_id) if renders > 1 else []
            candidates.append(
                {
                    "image_id": result.image_id,
                    "render": result.model_dump(mode="json"),
                    "public_observations": [
                        observation.model_dump(mode="json") for observation in observations
                    ],
                    "public_protection": public_protection(
                        components, task, result.image_id, observations
                    )
                    if renders > 1
                    else 0.0,
                }
            )
    selected = best_of_public(candidates) if renders > 1 else candidates[0]
    atomic_json(
        folder / "fixed_flow.json",
        {
            "method": method,
            "plan": plan.model_dump(mode="json"),
            "candidates": candidates,
            "delivered_image_id": selected["image_id"],
        },
    )
    steps = components.renderer.config.steps
    return selected["image_id"], {
        "actions": 0,
        "renders": renders,
        "inspects": 0,
        "reads": 0,
        "invalid": 0,
        "output_tokens": 0,
        "render_steps": renders * steps,
        "dit_forwards": renders * steps * (1 if bundle.cfg_scale == 1 else 2),
        "selection_checks": sum(len(candidate["public_observations"]) for candidate in candidates),
    }


def run_benchmark(
    components,
    benchmark,
    input_path,
    output_directory,
    seeds=(101,),
    method="B5",
    renders=1,
    mask_provided=False,
):
    if method not in {"B0", "B1", "B2", "B3", "B4", "B4-SFT", "B5"}:
        raise ValueError(f"Unknown benchmark method: {method}")
    fixed = method in {"B0", "B1"}
    if renders not in {1, 3} or (not fixed and renders != 1):
        raise ValueError(
            "Fixed B0/B1 permit one or three renders; agent methods run one budgeted trajectory"
        )
    if fixed and method == "B1" and not components.config["models"].get("renderer_policy"):
        raise ValueError("B1 requires an explicitly configured DiT SFT policy checkpoint")
    if benchmark not in {"geneval", "geneval2", "magicbrush"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "benchmark": benchmark,
        "method": method,
        "fixed_renders": renders if fixed else None,
        "mask_provided": mask_provided,
        "models": components.config["models"],
        "actor": components.config["actor"],
        "judge": components.config["judge"],
        "sampler": components.config["renderer"],
        "canvas": components.config["canvas"],
        "seeds": list(seeds),
    }
    protocol_path = output / "benchmark_protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(
            "The benchmark output directory belongs to a different protocol or checkpoint"
        )
    atomic_json(protocol_path, protocol)
    builder = PublicTaskBuilder(components.judge, components.artifacts)
    collector = None if fixed else RolloutCollector(components)
    prior_behavior = components.render_bridge.behavior_version
    components.render_bridge.behavior_version = (
        f"benchmark:{method}:{output.resolve()}:{prior_behavior}"
    )

    def generate(task, seed):
        folder = (
            output / "episodes" / f"seed_{seed}" / task.task_id.replace(":", "_").replace("/", "_")
        )
        folder.mkdir(parents=True, exist_ok=True)
        with BenchmarkCosts(components) as measured:
            source_id = None
            if task.source_image:
                with Image.open(task.source_image) as image:
                    image = normalize_image(image)
                    frame = make_frame(
                        (image.height, image.width),
                        tuple(components.config["canvas"]),
                        "benchmark-source",
                    )
                    source_id = components.artifacts.put_image(
                        letterbox(image, frame), frame=frame
                    ).artifact_id
            public_path = folder / "public_task.json"
            if public_path.is_file():
                public = PublicTask.model_validate_json(public_path.read_text())
                if (
                    public.user_request.text != task.prompt
                    or public.user_request.source_image_id != source_id
                    or public.user_request.controls != (task.user_controls or {})
                ):
                    raise ValueError("A resumed benchmark episode has different public inputs")
            else:
                public = builder.build(
                    task_id=task.task_id,
                    text=task.prompt,
                    family_id=task.family_id,
                    split_id="benchmark",
                    source_image_id=source_id,
                    controls=task.user_controls,
                    source_dataset=benchmark,
                )
                public = public.model_copy(
                    update={
                        "canvas": Canvas(
                            height=components.config["canvas"][0],
                            width=components.config["canvas"][1],
                        ),
                        "limits": TaskLimits(
                            max_actions=task.max_actions, max_renders=task.max_renders
                        ),
                    }
                )
                atomic_json(public_path, public.model_dump(mode="json"))
            if fixed:
                delivered, costs = run_fixed_flow(components, public, seed, folder, method, renders)
            else:
                environment = collector.infer(public, seed, folder)
                delivered = environment.state.delivered_image_id
                costs = environment.state.budget.model_dump()
                costs["output_tokens"] = sum(
                    torch.load(path, map_location="cpu", weights_only=True)["token_ids"].numel()
                    for path in folder.glob("action_*.pt")
                )
                costs["render_steps"] = (
                    len(environment.state.executed_renders) * components.renderer.config.steps
                )
                costs["dit_forwards"] = (
                    1 if components.renderer.config.cfg_scale == 1 else 2
                ) * costs["render_steps"]
        costs.update(measured.values)
        costs["render_cache_hits"] = max(0, costs["renders"] - costs["actual_renders"])
        costs["action_cache_hits"] = max(0, costs["actions"] - costs["sampled_actions"])
        costs["logical_action_cost"] = (
            costs["renders"] + 0.1 * costs["inspects"] + 0.01 * costs["actions"]
        ) / 3.3
        cost_path = folder / "costs.json"
        previous = json.loads(cost_path.read_text()) if cost_path.is_file() else {}
        for key in (*measured.values, "render_cache_hits", "action_cache_hits"):
            if isinstance(costs[key], (int, float)):
                costs[key] += previous.get(key, 0)
        previous_services = previous.get("gpu_seconds_by_service", {})
        costs["gpu_seconds_by_service"] = {
            key: measured.gpu_seconds.get(key, 0) + previous_services.get(key, 0)
            for key in measured.gpu_seconds.keys() | previous_services.keys()
        }
        costs["execution_attempts"] = previous.get("execution_attempts", 0) + 1
        atomic_json(cost_path, costs)
        return DeliveredImage(
            str(components.artifacts.path(delivered)) if delivered else None, costs
        )

    try:
        if benchmark == "geneval":
            return run_geneval(input_path, output, generate, seeds)
        if benchmark == "geneval2":
            public_path = prepare_public_prompts(input_path, output / "public_prompts.jsonl")
            return run_geneval2(public_path, output, generate, seeds)
        return run_magicbrush(input_path, output, generate, seeds, mask_provided)
    finally:
        components.render_bridge.behavior_version = prior_behavior
