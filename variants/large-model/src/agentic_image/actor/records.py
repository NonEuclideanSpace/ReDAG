from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ActorSnapshot:
    tensors: dict[str, torch.Tensor]
    prefix_image_positions: torch.Tensor
    image_features: torch.Tensor | None
    position_ids: torch.Tensor
    rope_delta: torch.Tensor
    image_artifact_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    visual_files: tuple[str, ...] = ()

    @property
    def prefix_length(self):
        return self.tensors["input_ids"].shape[-1]

    def state_dict(self):
        return {
            "tensors": {
                name: value
                for name, value in self.tensors.items()
                if not self.visual_files or name != "pixel_values"
            },
            "prefix_image_positions": self.prefix_image_positions,
            "image_features": None if self.visual_files else self.image_features,
            "position_ids": self.position_ids,
            "rope_delta": self.rope_delta,
            "image_artifact_ids": list(self.image_artifact_ids),
            "metadata": self.metadata,
            "visual_files": list(self.visual_files),
        }

    @classmethod
    def from_state_dict(cls, state):
        state = dict(state)
        state["image_artifact_ids"] = tuple(state.get("image_artifact_ids", ()))
        state["visual_files"] = tuple(state.get("visual_files", ()))
        if state["visual_files"]:
            images = [
                torch.load(path, map_location="cpu", weights_only=True)
                for path in state["visual_files"]
            ]
            grid = torch.stack([image["grid_thw"] for image in images])
            if not torch.equal(grid, state["tensors"]["image_grid_thw"]):
                raise ValueError("Saved visual inputs do not match the action's image layout")
            state["tensors"] = dict(
                state["tensors"],
                pixel_values=torch.cat([image["pixel_values"] for image in images]),
            )
            state["image_features"] = torch.cat([image["image_features"] for image in images])
        return cls(**state)

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=True))


@dataclass
class ActionSample:
    snapshot: ActorSnapshot
    token_ids: torch.Tensor
    behavior_log_probs: torch.Tensor
    response_mask: torch.Tensor
    text: str
    finish_reason: str
    seed: int
    behavior_version: str
    probe_logits: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.token_ids.ndim != 1:
            raise ValueError("Action tokens must be a one-dimensional sequence")
        if (
            self.token_ids.shape != self.behavior_log_probs.shape
            or self.token_ids.shape != self.response_mask.shape
        ):
            raise ValueError("Action tokens, log probabilities and response mask must align")
        if not torch.isfinite(self.behavior_log_probs).all():
            raise ValueError("Action log probabilities must be finite")

    def state_dict(self):
        return {
            "snapshot": self.snapshot.state_dict(),
            "token_ids": self.token_ids,
            "behavior_log_probs": self.behavior_log_probs,
            "response_mask": self.response_mask,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "seed": self.seed,
            "behavior_version": self.behavior_version,
            "probe_logits": self.probe_logits,
        }

    @classmethod
    def from_state_dict(cls, state):
        state = dict(state)
        state["snapshot"] = ActorSnapshot.from_state_dict(state["snapshot"])
        return cls(**state)
