import json
from pathlib import Path

import torch

from agentic_image.actor import ActorConfig, ActorWorker
from agentic_image.checks.judge import FrozenJudge, JudgeConfig
from agentic_image.checks.private import PrivateEvaluator
from agentic_image.checks.public import PublicObserver
from agentic_image.checks.tools import ToolClient
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import box_mask, letterbox, make_frame, transform_mask
from agentic_image.core.types import ContractError, PlanSpec, RenderResult
from agentic_image.render.compiler import ConditionCompiler
from agentic_image.render.encoders import FrozenEncoders
from agentic_image.render.model import KleinRenderer
from agentic_image.render.types import (
    ConditionBundle,
    ConditionRequest,
    EncoderConfig,
    ReferenceSpec,
    RegionSpec,
    RendererConfig,
    RenderTrace,
)


class CompilerBridge:
    def __init__(self, artifacts, encoders, directory, config):
        self.artifacts = artifacts
        self.encoders = encoders
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.compiler = ConditionCompiler(encoders)
        self.canvas_hw = tuple(config["canvas"])
        self.cache_bytes = config.get("data", {}).get("bundle_cache_bytes", 64 * 1024**3)

    def validate_plan(self, plan, source_image_id=None):
        for text in (plan.global_text, *(region.text for region in plan.regions)):
            formatted = self.encoders.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = self.encoders.tokenizer(formatted, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]
            if len(tokens) > self.encoders.config.text_block_tokens:
                raise ContractError(
                    "TEXT_TOO_LONG",
                    "A complete templated text block exceeds the Klein token capacity",
                )

    def _paths(self, bundle_id):
        stem = bundle_id.replace(":", "_")
        return self.directory / f"{stem}.pt", self.directory / f"{stem}.json"

    def _evict(self):
        files = sorted(self.directory.glob("*.pt"), key=lambda path: path.stat().st_mtime)
        size = sum(path.stat().st_size for path in files)
        for path in files:
            if size <= self.cache_bytes:
                break
            size -= path.stat().st_size
            path.unlink(missing_ok=True)

    def frame(self, image_id):
        reference = self.artifacts.get_ref(image_id)
        if reference.frame is not None and reference.frame.canvas_hw == self.canvas_hw:
            return reference.frame
        image = self.artifacts.get_image(image_id)
        return make_frame((image.height, image.width), self.canvas_hw, f"frame:{image_id}")

    def image(self, image_id):
        image = self.artifacts.get_image(image_id)
        if (image.height, image.width) != self.canvas_hw:
            image = letterbox(image, self.frame(image_id))
        return image

    def reference_mask(self, mask_id, image_id):
        if mask_id is None:
            return torch.ones(self.canvas_hw, dtype=torch.float32)
        reference = self.artifacts.get_ref(mask_id)
        mask = self.artifacts.get_array(mask_id)
        frame = self.frame(image_id)
        if (
            reference.producer_id
            and reference.producer_id.startswith("artifact:")
            and reference.producer_id != image_id
        ):
            raise ContractError(
                "FRAME_MISMATCH", "Appearance mask was produced from a different image"
            )
        if reference.frame is not None and reference.frame.frame_id != frame.frame_id:
            raise ContractError(
                "FRAME_MISMATCH", "Appearance mask uses a different reference frame"
            )
        if mask.shape != self.canvas_hw:
            if mask.shape != frame.source_hw:
                raise ContractError(
                    "FRAME_MISMATCH", "Appearance mask does not match its reference image"
                )
            mask = transform_mask(
                mask, frame, "binary" if reference.metadata.get("kind") == "binary" else "soft"
            )
        return torch.from_numpy(mask.copy()).float()

    def mask(self, mask_id, box):
        if mask_id:
            mask = self.artifacts.get_array(mask_id)
            if tuple(mask.shape) != self.canvas_hw:
                raise ContractError(
                    "FRAME_MISMATCH", "Control mask must use the registered canvas frame"
                )
            return torch.from_numpy(mask.copy()).float()
        if box is not None:
            return torch.from_numpy(box_mask(box, self.canvas_hw))
        return None

    def compile(self, plan, source_image_id=None):
        regions = tuple(
            RegionSpec(
                region_id=region.region_id,
                entity_ids=region.entity_ids,
                text=region.text,
                target_mask=self.mask(region.target_mask_id, region.target_box),
                source_mask=self.mask(region.source_mask_id, region.source_box),
                appearance_bindings=tuple(
                    binding.image_id
                    for binding in plan.references
                    if binding.binding_id in region.reference_ids
                ),
            )
            for region in plan.regions
        )
        references = []
        if source_image_id:
            references.append(
                ReferenceSpec(
                    role="source",
                    image_id=source_image_id,
                    latent=self.encoders.encode_image(self.image(source_image_id)),
                    scope="global",
                    frame_id=self.frame(source_image_id).frame_id,
                )
            )
        grouped = {}
        for binding in plan.references:
            entry = grouped.setdefault(
                binding.image_id, {"role": binding.role, "scope": binding.scope, "memberships": {}}
            )
            if entry["role"] != binding.role:
                raise ContractError(
                    "REFERENCE_ROLE_CONFLICT", "One reference segment cannot have conflicting roles"
                )
            if binding.scope == "global":
                entry["scope"] = "global"
            if binding.entity_id:
                mask = self.reference_mask(binding.crop_mask_id, binding.image_id)
                entry["memberships"][binding.entity_id] = mask
        for image_id, values in grouped.items():
            references.append(
                ReferenceSpec(
                    image_id=image_id,
                    latent=self.encoders.encode_image(self.image(image_id)),
                    frame_id=self.frame(image_id).frame_id,
                    **values,
                )
            )
        try:
            bundle = self.compiler.compile(
                ConditionRequest(
                    global_text=plan.global_text,
                    regions=regions,
                    references=tuple(references),
                    canvas_hw=self.canvas_hw,
                    cfg_scale=self.config["renderer"]["cfg_scale"],
                    bindings={
                        "plan": plan.model_dump(mode="json"),
                        "source_image_id": source_image_id,
                        "frames": {
                            reference.image_id: self.frame(reference.image_id).model_dump(
                                mode="json"
                            )
                            for reference in references
                        },
                    },
                    model_versions={"renderer": self.config["models"]["renderer_revision"]},
                    sigma_table_id=f"flow-sde-{self.config['renderer']['steps']}-eta{self.config['renderer']['eta']}-v3",
                    attention_backend=self.config["renderer"]["attention_backend"],
                )
            )
        except ValueError as error:
            message = str(error)
            code = (
                "TEXT_TOO_LONG"
                if "text" in message.lower() or "token" in message.lower()
                else "INVALID_CONDITION"
            )
            raise ContractError(code, message) from error
        from agentic_image.learning.transaction import atomic_json

        payload, recipe = self._paths(bundle.bundle_id)
        atomic_json(
            recipe,
            {
                "bundle_id": bundle.bundle_id,
                "plan": plan.model_dump(mode="json"),
                "source_image_id": source_image_id,
                "compiler_version": bundle.compiler_version,
                "model_versions": bundle.model_versions,
            },
        )
        if not payload.is_file():
            bundle.save(payload)
        else:
            payload.touch()
        self._evict()
        return bundle

    def load(self, bundle_id):
        payload, recipe = self._paths(bundle_id)
        if payload.is_file():
            bundle = ConditionBundle.load(payload)
            payload.touch()
            return bundle
        if not recipe.is_file():
            raise FileNotFoundError(f"The condition recipe is unavailable: {bundle_id}")
        value = json.loads(recipe.read_text())
        bundle = self.compile(PlanSpec.model_validate(value["plan"]), value["source_image_id"])
        if bundle.bundle_id != bundle_id:
            raise RuntimeError(
                "The frozen condition recipe reconstructed a different renderer input"
            )
        return bundle


class RendererBridge:
    def __init__(self, renderer, compiler, artifacts, directory, behavior_version="initial"):
        self.renderer = renderer
        self.compiler = compiler
        self.artifacts = artifacts
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.behavior_version = behavior_version

    def render(self, bundle, seed, job_id):
        from agentic_image.learning.transaction import atomic_json

        safe_id = job_id.replace(":", "_").replace("/", "_")
        folder = self.directory / safe_id
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.is_file():
            result = RenderResult.model_validate_json(result_path.read_text())
            if result.bundle_id != bundle.bundle_id or result.metadata["seed"] != seed:
                raise RuntimeError("A render job ID was reused with different inputs")
            if result.metadata["behavior_version"] != self.behavior_version:
                raise RuntimeError("A render job ID was reused at a different behavior version")
            return result
        trace_path = folder / "trace.pt"
        if trace_path.is_file():
            trace = RenderTrace.load(trace_path)
            if (
                trace.bundle_id != bundle.bundle_id
                or trace.behavior_version != self.behavior_version
            ):
                raise RuntimeError("Interrupted render has incompatible conditions or behavior")
        else:
            trace = self.renderer.rollout(
                bundle, seed, behavior_version=self.behavior_version, render_id=job_id
            )
            trace.save(trace_path)
        image = self.compiler.encoders.decode_image(trace.final_latent, bundle.canvas_hw)
        frame = make_frame(bundle.canvas_hw, bundle.canvas_hw, "canvas")
        reference = self.artifacts.put_image(
            image,
            frame=frame,
            producer_id=job_id,
            metadata={"bundle_id": bundle.bundle_id, "seed": seed},
        )
        result = RenderResult(
            image_id=reference.artifact_id,
            bundle_id=bundle.bundle_id,
            trajectory_id=str(trace_path.resolve()),
            metadata={"seed": seed, "behavior_version": self.behavior_version},
        )
        atomic_json(result_path, result.model_dump(mode="json"))
        return result


def actor_config(config):
    values = config["actor"]
    models = config["models"]
    return ActorConfig(
        quantized_model=models["actor_nf4"],
        model_id=models["actor"],
        revision=models["actor_revision"],
        policy_adapter=models.get("actor_policy"),
        reference_adapter=models.get("actor_reference"),
        device=config["devices"]["actor"],
        max_input_tokens=values["max_input_tokens"],
        max_action_tokens=values["max_new_tokens"],
        max_images=values["max_images"],
        temperature=values["temperature"],
        head_chunk_size=values["head_chunk_size"],
        image_cache_directory=str(Path(config["run_dir"]) / "actor_images"),
    )


def judge_config(config):
    models = config["models"]
    values = config["judge"]
    return JudgeConfig(
        model=models["judge"],
        revision=models["judge_revision"],
        nf4_path=models["judge_nf4"],
        device=config["devices"]["judge"],
        max_input_tokens=values["max_input_tokens"],
        threshold=values["threshold"],
        calibration_path=values.get("calibration"),
    )


class Components:
    def __init__(self, config):
        self.config = config
        self.artifacts = ArtifactStore(config["data"]["artifacts"])
        self.directory = Path(config["run_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.actor = None
        self.encoders = None
        self.renderer = None
        self.judge = None
        self.compiler = None
        self.render_bridge = None
        self.observer = None
        self.evaluator = None

    def load_actor(self):
        if self.actor is None:
            self.actor = ActorWorker.load(actor_config(self.config))
        return self.actor

    def load_encoders(self):
        if self.encoders is None:
            models = self.config["models"]
            self.encoders = FrozenEncoders.load(
                EncoderConfig(
                    model_id=models["renderer"],
                    revision=models["renderer_revision"],
                    device=self.config["devices"]["encoders"],
                    cache_directory=str(self.directory / "encoding_cache"),
                )
            )
            self.compiler = CompilerBridge(
                self.artifacts, self.encoders, self.directory / "bundles", self.config
            )
        return self.encoders

    def load_renderer(self):
        if self.renderer is None:
            self.load_encoders()
            models = self.config["models"]
            values = self.config["renderer"]
            self.renderer = KleinRenderer.load(
                RendererConfig(
                    model_id=models["renderer"],
                    revision=models["renderer_revision"],
                    device=self.config["devices"]["renderer"],
                    attention_backend=values["attention_backend"],
                    saved_tensor_cpu_offload=values["saved_tensor_offload"],
                    policy_path=models.get("renderer_policy"),
                    reference_path=models.get("renderer_reference"),
                    steps=values["steps"],
                    eta=values["eta"],
                    cfg_scale=values["cfg_scale"],
                )
            )
            self.render_bridge = RendererBridge(
                self.renderer, self.compiler, self.artifacts, self.directory / "renders"
            )
        return self.renderer

    def load_judge(self, private=True):
        if self.judge is None:
            self.judge = FrozenJudge.load(judge_config(self.config))
            tools = ToolClient(self.config["judge"]["tool_url"], self.config["judge"]["timeout"])
            self.observer = PublicObserver(self.judge, self.artifacts, tools)
        if private and self.evaluator is None:
            self.evaluator = PrivateEvaluator.from_file(
                self.judge,
                self.artifacts,
                self.config["data"]["private_tasks"],
                self.observer.tools,
            )
        return self.judge

    def load_all(self, private=True):
        self.load_actor()
        self.load_renderer()
        self.load_judge(private)
        return self
