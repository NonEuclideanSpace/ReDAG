from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


CONTROL_DIMENSIONS = {"depth": 512, "pose": 4608}


def _to_canvas(channels: torch.Tensor, frame, valid_channel: int) -> torch.Tensor:
    if tuple(channels.shape[-2:]) != tuple(frame.source_hw):
        raise ValueError("Control input does not match its registered source frame")
    resized = F.interpolate(
        channels[None], size=frame.resized_hw, mode="bilinear", align_corners=False
    )[0]
    valid = F.interpolate(
        channels[None, valid_channel : valid_channel + 1], size=frame.resized_hw, mode="nearest"
    )[0, 0]
    resized[valid_channel] = valid
    canvas = torch.zeros(channels.shape[0], *frame.canvas_hw, dtype=torch.float32)
    x, y = frame.padding_xy
    height, width = frame.resized_hw
    canvas[:, y : y + height, x : x + width] = resized
    return canvas


def control_tokens(channels: torch.Tensor) -> torch.Tensor:
    if channels.ndim == 3:
        channels = channels[None]
    if channels.ndim != 4 or any(size % 16 for size in channels.shape[-2:]):
        raise ValueError("Control canvases must be BCHW with dimensions divisible by 16")
    return F.pixel_unshuffle(channels.float(), 16).flatten(2).transpose(1, 2).contiguous()


def depth_tokens(depth, valid, frame, already_inverse=False) -> torch.Tensor:
    depth = torch.as_tensor(np.asarray(depth), dtype=torch.float32)
    valid = torch.as_tensor(np.asarray(valid), dtype=torch.bool)
    if depth.shape != valid.shape or depth.ndim != 2:
        raise ValueError("Depth and validity must have the same HW shape")
    valid = valid & torch.isfinite(depth) & ((depth >= 0) if already_inverse else (depth > 0))
    if not valid.any():
        raise ValueError("A depth control needs at least one valid depth pixel")
    inverse = (
        depth
        if already_inverse
        else torch.where(valid, depth.reciprocal(), torch.zeros_like(depth))
    )
    lower, upper = torch.quantile(inverse[valid], torch.tensor([0.05, 0.95]))
    normalized = torch.zeros_like(depth)
    if upper <= lower:
        normalized[valid] = 0.5
    else:
        normalized[valid] = ((inverse[valid] - lower) / (upper - lower)).clamp(0, 1)
    channels = _to_canvas(torch.stack((normalized, valid.float())), frame, 1)
    channels[0] *= channels[1]
    return control_tokens(channels)


def pose_tokens(
    keypoints, frame, visibility=None, sigma_pixels=8.0, centers_at_zero=False
) -> torch.Tensor:
    points = torch.as_tensor(np.asarray(keypoints), dtype=torch.float32)
    if points.shape != (17, 3) or not torch.isfinite(points).all():
        raise ValueError("Pose input must contain 17 finite x, y, confidence rows")
    if torch.any((points[:, 2] < 0) | (points[:, 2] > 1)) or sigma_pixels <= 0:
        raise ValueError(
            "Pose confidence must lie within [0, 1] and heatmap sigma must be positive"
        )
    present = (
        points[:, 2] > 0
        if visibility is None
        else torch.as_tensor(visibility, dtype=torch.bool) & (points[:, 2] > 0)
    )
    if present.shape != (17,):
        raise ValueError("Pose visibility must have 17 entries")
    if torch.any(
        present
        & (
            (points[:, 0] < 0)
            | (points[:, 0] > frame.source_hw[1])
            | (points[:, 1] < 0)
            | (points[:, 1] > frame.source_hw[0])
        )
    ):
        raise ValueError("Visible pose joints must lie inside the registered source frame")
    height, width = frame.canvas_hw
    y, x = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    transformed = points[:, :2] + (0.5 if centers_at_zero else 0)
    transformed = transformed * torch.tensor(frame.scale_xy) + torch.tensor(frame.padding_xy)
    heatmaps = []
    for index in range(17):
        if present[index]:
            squared = (x - transformed[index, 0]).square() + (y - transformed[index, 1]).square()
            heatmaps.append(points[index, 2] * torch.exp(-squared / (2 * sigma_pixels**2)))
        else:
            heatmaps.append(torch.zeros(height, width))
    valid = torch.zeros(height, width)
    pad_x, pad_y = frame.padding_xy
    resized_h, resized_w = frame.resized_hw
    valid[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = 1
    return control_tokens(torch.stack((*(heatmap * valid for heatmap in heatmaps), valid)))


class ControlAdapters(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.policy = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, hidden_size))
                for name, width in CONTROL_DIMENSIONS.items()
            }
        )
        for network in self.policy.values():
            nn.init.zeros_(network[2].weight)
            nn.init.zeros_(network[2].bias)
        self.reference = None
        self.active = "policy"
        self._tokens = None
        self._target_tokens = 0
        self._hook = None

    def freeze_reference(self, state=None):
        self.reference = deepcopy(self.policy).eval().requires_grad_(False)
        if state is not None:
            self.reference.load_state_dict(state, strict=True)
        self.reference.float()

    def set_adapter(self, name):
        if name == "reference" and self.reference is None:
            raise RuntimeError("Control SFT reference has not been initialized")
        self.active = name
        self.policy.requires_grad_(name == "policy")
        if self.reference is not None:
            self.reference.requires_grad_(False)

    def forward(self, tokens, target_tokens):
        network = self.policy if self.active == "policy" else self.reference
        residual = None
        for name, value in tokens.items():
            if name not in CONTROL_DIMENSIONS or value.shape != (
                1,
                target_tokens,
                CONTROL_DIMENSIONS[name],
            ):
                raise ValueError("Control tensor does not match its type and target grid")
            if not torch.isfinite(value).all():
                raise ValueError("Control tokens must be finite")
            contribution = network[name](
                value.detach().to(next(network.parameters()).device, torch.float32)
            )
            residual = contribution if residual is None else residual + contribution
        return residual

    def attach(self, transformer):
        if self._hook is not None:
            self._hook.remove()

        def add_target_residual(module, arguments, output):
            if not self._tokens:
                return output
            residual = self(self._tokens, self._target_tokens)
            return torch.cat(
                (
                    output[:, : self._target_tokens] + residual.to(output.dtype),
                    output[:, self._target_tokens :],
                ),
                dim=1,
            )

        self._hook = transformer.x_embedder.register_forward_hook(add_target_residual)

    @contextmanager
    def condition(self, tokens, target_tokens):
        previous = self._tokens, self._target_tokens
        self._tokens, self._target_tokens = tokens, target_tokens
        try:
            yield
        finally:
            self._tokens, self._target_tokens = previous


def with_controls(bundle, controls, provenance):
    if not provenance or any(name not in CONTROL_DIMENSIONS for name in controls):
        raise ValueError("Controls need explicit public provenance and a registered control type")
    digest = sha256(bundle.bundle_id.encode())
    digest.update(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for name, value in sorted(controls.items()):
        if value.shape != (1, bundle.num_target_tokens, CONTROL_DIMENSIONS[name]):
            raise ValueError("Control tensor does not match the target grid")
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    bindings = {**bundle.artifact_bindings, "control_provenance": provenance}
    return replace(
        bundle,
        bundle_id=f"bundle:{digest.hexdigest()}",
        controls={name: value.detach().float().cpu() for name, value in controls.items()},
        artifact_bindings=bindings,
    )
