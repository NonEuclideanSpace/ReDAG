import importlib.metadata

import torch

from agentic_image.core.types import BoundingBox, PlanSpec, RegionControl, Entity, ReferenceBinding
from agentic_image.learning.transaction import atomic_json
from agentic_image.render.training import FlowMatchingExample, backward_flow_matching


def hardware_report(config):
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "diffusers",
        "bitsandbytes",
        "flash-linear-attention",
        "causal-conv1d",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "memory_gib": torch.cuda.get_device_properties(index).total_memory / 2**30,
            "capability": torch.cuda.get_device_capability(index),
        }
        for index in range(torch.cuda.device_count())
    ]
    return {
        "packages": versions,
        "cuda": torch.version.cuda,
        "devices": devices,
        "configured_devices": config["devices"],
        "cuda_available": torch.cuda.is_available(),
    }


def run_preflight(components, task_path, component="all", cycles=1):
    from .rollout import RolloutCollector, read_tasks

    tasks = read_tasks(task_path)
    if not tasks:
        raise ValueError("Preflight requires a real public development task")
    task = tasks[0]
    report = {"hardware": hardware_report(components.config), "checks": {}}
    ceiling = components.config.get("resources", {}).get("max_reserved_gib", 42)
    if component in {"actor", "all"}:
        from agentic_image.core.runtime import EpisodeEnvironment
        from .prompt import SYSTEM_PROMPT

        components.load_actor()
        collector = RolloutCollector(components)
        tokenizer = components.actor.processor.tokenizer
        reserve = len(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)) + 128
        environment = EpisodeEnvironment(
            task,
            components.artifacts,
            None,
            None,
            None,
            episode_id="preflight",
            token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            max_input_tokens=components.config["actor"]["max_input_tokens"] - reserve,
        )
        snapshot = collector.snapshot(environment)
        torch.cuda.reset_peak_memory_stats(components.actor.device)
        report["checks"]["actor"] = components.actor.diagnose(snapshot)
        reserved = torch.cuda.max_memory_reserved(components.actor.device) / 2**30
        report["checks"]["actor"]["peak_reserved_gib"] = reserved
        if reserved > ceiling:
            raise RuntimeError(f"Controller exceeded the {ceiling} GiB profile: {reserved:.2f}")
    if component in {"renderer", "all"}:
        components.load_renderer()
        renderer = components.renderer
        image_ids = [task.user_request.source_image_id, *task.user_request.appearance_image_ids]
        image_ids = [image for image in image_ids if image]
        if len(image_ids) < 2:
            raise ValueError(
                "Maximum-layout renderer preflight needs two distinct public source/reference images"
            )
        regions = (
            RegionControl(
                region_id="region:left",
                text="the first object",
                entity_ids=("entity:0",),
                target_box=BoundingBox(x0=0.05, y0=0.1, x1=0.45, y1=0.9),
                source_box=BoundingBox(x0=0.05, y0=0.1, x1=0.45, y1=0.9),
                reference_ids=("binding:0",),
            ),
            RegionControl(
                region_id="region:right",
                text="the second object",
                entity_ids=("entity:1",),
                target_box=BoundingBox(x0=0.55, y0=0.1, x1=0.95, y1=0.9),
                source_box=BoundingBox(x0=0.55, y0=0.1, x1=0.95, y1=0.9),
            ),
        )
        plan = PlanSpec(
            global_text=task.user_request.text,
            entities=(
                Entity(entity_id="entity:0", name="first object"),
                Entity(entity_id="entity:1", name="second object"),
            ),
            regions=regions,
            references=(
                ReferenceBinding(
                    binding_id="binding:0", image_id=image_ids[1], entity_id="entity:0"
                ),
            ),
        )
        bundle = components.compiler.compile(plan, image_ids[0])
        if not renderer._reference_exists:
            renderer.freeze_reference()
        optimizer = torch.optim.AdamW(renderer.policy_parameters(), lr=5e-5)
        before = renderer.policy_state()
        target = components.encoders.encode_image(components.compiler.image(image_ids[0]))
        torch.cuda.reset_peak_memory_stats(renderer.device)
        try:
            for step in range(cycles):
                optimizer.zero_grad(set_to_none=True)
                metrics = backward_flow_matching(
                    renderer,
                    FlowMatchingExample("preflight", bundle, target, torch.ones(bundle.canvas_hw)),
                    0,
                    step,
                    1,
                )
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in renderer.policy_parameters()
                ):
                    raise FloatingPointError("Renderer preflight produced nonfinite gradients")
                optimizer.step()
            renderer.load_policy_state(before)
            optimizer.zero_grad(set_to_none=True)
            trace = renderer.rollout(bundle, 17, behavior_version="preflight")
            metrics = renderer.backward_replay(
                bundle, trace, 0.5, 1.0, expected_behavior_version="preflight"
            )
            image = components.encoders.decode_image(trace.final_latent, bundle.canvas_hw)
            output = components.directory / "preflight_render.png"
            image.save(output)
            report["checks"]["renderer"] = {
                **metrics,
                "cycles": cycles,
                "image": str(output.resolve()),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(renderer.device) / 2**30,
            }
            if report["checks"]["renderer"]["peak_reserved_gib"] > ceiling:
                raise RuntimeError(f"Renderer exceeded the {ceiling} GiB resource profile")
        finally:
            renderer.load_policy_state(before)
            optimizer.zero_grad(set_to_none=True)
    if component in {"judge", "all"}:
        components.load_judge(private=False)
        image_id = task.user_request.source_image_id or next(
            iter(task.user_request.appearance_image_ids), None
        )
        if image_id is None:
            raise ValueError("Judge preflight needs a public image")
        scores = components.observer.check(task, image_id)
        report["checks"]["judge"] = {
            "observations": [score.model_dump(mode="json") for score in scores],
            "queries": components.judge.queries,
            "forwards": components.judge.forwards,
        }
    atomic_json(components.directory / f"preflight_{component}.json", report)
    return report
