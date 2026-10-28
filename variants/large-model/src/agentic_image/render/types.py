from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import torch


MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
MODEL_REVISION = "32773329fbe7e81a90ef971740e8ba4b0364ecf3"


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    device: str = "cuda:0"
    local_files_only: bool = False
    text_block_tokens: int = 512
    text_hidden_size: int = 12288
    cache_directory: str | None = None
    cache_bytes: int = 64 * 1024**3


@dataclass(frozen=True)
class RendererConfig:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    device: str = "cuda:0"
    local_files_only: bool = False
    attention_backend: str = "_native_efficient"
    query_chunk_size: int = 128
    gradient_checkpointing: bool = True
    saved_tensor_cpu_offload: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    policy_path: str | None = None
    reference_path: str | None = None
    strict_architecture: bool = True
    steps: int = 50
    eta: float = 0.3
    cfg_scale: float = 4.0
    enable_controls: bool = False


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    entity_ids: tuple[str, ...]
    text: str
    target_mask: torch.Tensor | None
    source_mask: torch.Tensor | None = None
    appearance_bindings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceSpec:
    role: Literal["source", "appearance", "style"]
    image_id: str
    latent: torch.Tensor
    memberships: dict[str, torch.Tensor] = field(default_factory=dict)
    scope: Literal["regional", "global"] = "regional"
    frame_id: str | None = None


@dataclass(frozen=True)
class ConditionRequest:
    global_text: str
    regions: tuple[RegionSpec, ...] = ()
    references: tuple[ReferenceSpec, ...] = ()
    canvas_hw: tuple[int, int] = (512, 512)
    cfg_scale: float = 4.0
    bindings: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    sigma_table_id: str = "flow-sde-50-eta0.3-v3"
    attention_backend: str = "_native_efficient"


@dataclass(frozen=True)
class TokenSpan:
    kind: str
    key: str
    start: int
    end: int


@dataclass(frozen=True)
class ConditionBundle:
    bundle_id: str
    canvas_hw: tuple[int, int]
    positive_embeds: torch.Tensor
    negative_embeds: torch.Tensor
    text_ids: torch.Tensor
    image_ids: torch.Tensor
    reference_tokens: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    token_span_table: tuple[TokenSpan, ...]
    target_membership: torch.Tensor
    source_membership: torch.Tensor
    appearance_membership: dict[str, torch.Tensor]
    cfg_scale: float
    sigma_table_id: str
    attention_backend: str
    layout_signature: str
    artifact_bindings: dict[str, Any]
    model_versions: dict[str, str]
    compiler_version: str = "dag-agent-v3"
    raster_fallbacks: tuple[str, ...] = ()
    controls: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def num_target_tokens(self) -> int:
        return self.canvas_hw[0] * self.canvas_hw[1] // 256

    @property
    def region_count(self) -> int:
        return self.target_membership.shape[0]

    def to(self, device: str | torch.device) -> ConditionBundle:
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                values[item.name] = value.to(device)
            elif item.name in {"appearance_membership", "controls"}:
                values[item.name] = {key: tensor.to(device) for key, tensor in value.items()}
        return replace(self, **values)

    def state_dict(self) -> dict[str, Any]:
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            elif item.name == "token_span_table":
                value = [asdict(span) for span in value]
            elif item.name in {"appearance_membership", "controls"}:
                value = {key: tensor.detach().cpu() for key, tensor in value.items()}
            result[item.name] = value
        return result

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}")
        torch.save(self.state_dict(), temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> ConditionBundle:
        values = torch.load(path, map_location="cpu", weights_only=True)
        values["token_span_table"] = tuple(TokenSpan(**span) for span in values["token_span_table"])
        values["canvas_hw"] = tuple(values["canvas_hw"])
        values["raster_fallbacks"] = tuple(values["raster_fallbacks"])
        return cls(**values)


@dataclass(frozen=True)
class ReplayStep:
    stratum_id: int
    step_index: int
    sampling_probability: float
    weight: float


@dataclass(frozen=True)
class RenderTrace:
    render_id: str
    bundle_id: str
    behavior_version: str
    states: torch.Tensor
    sigmas: torch.Tensor
    transition_std: torch.Tensor
    stored_mean_logp: torch.Tensor
    seed_init: int
    seed_steps: tuple[int, ...]
    stochastic_indices: tuple[int, ...]
    eta: float
    cfg_scale: float
    replay_steps: tuple[ReplayStep, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_latent(self) -> torch.Tensor:
        return self.states[-1]

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values["replay_steps"] = [asdict(step) for step in self.replay_steps]
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.detach().cpu()
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}")
        torch.save(values, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> RenderTrace:
        values = torch.load(path, map_location="cpu", weights_only=True)
        values["replay_steps"] = tuple(ReplayStep(**step) for step in values["replay_steps"])
        values["seed_steps"] = tuple(values["seed_steps"])
        values["stochastic_indices"] = tuple(values["stochastic_indices"])
        return cls(**values)
