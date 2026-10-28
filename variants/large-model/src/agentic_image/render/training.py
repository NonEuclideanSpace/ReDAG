from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from .compiler import patch_coverage
from .sde import gaussian_noise, stream_seed
from .types import ConditionBundle

if TYPE_CHECKING:
    from .model import KleinRenderer


@dataclass(frozen=True)
class FlowMatchingExample:
    sample_id: str
    bundle: ConditionBundle
    target_latents: torch.Tensor
    valid_mask: torch.Tensor
    edit_mask: torch.Tensor | None = None
    sample_valid: bool = True


def flow_matching_errors(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    valid_coverage: torch.Tensor,
    edit_coverage: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target_velocity.shape or prediction.ndim != 3:
        raise ValueError("Predicted and target velocities must share shape BLC")
    batch, length, channels = prediction.shape
    valid = valid_coverage.to(prediction.device, torch.float32)
    edit = (
        torch.zeros_like(valid)
        if edit_coverage is None
        else edit_coverage.to(prediction.device, torch.float32)
    )
    if valid.shape != (batch, length) or edit.shape != (batch, length):
        raise ValueError("Supervision coverage must have shape BL")
    if (
        not torch.isfinite(valid).all()
        or not torch.isfinite(edit).all()
        or torch.any((valid < 0) | (valid > 1) | (edit < 0) | (edit > 1))
    ):
        raise ValueError("Supervision masks must contain finite coverage within [0, 1]")
    samples = (
        torch.ones(batch, dtype=torch.bool, device=prediction.device)
        if sample_mask is None
        else sample_mask.to(prediction.device, torch.bool)
    )
    if samples.shape != (batch,):
        raise ValueError("Sample validity must have shape B")
    weights = valid * (1 + 3 * edit)
    losses = []
    for index in range(batch):
        if not samples[index]:
            losses.append(prediction[index].sum() * 0)
            continue
        denominator = channels * weights[index].sum()
        if denominator <= 0:
            raise ValueError("A real sample must contain valid target supervision")
        residual = (prediction[index].float() - target_velocity[index].detach().float()).square()
        losses.append((weights[index, :, None] * residual).sum() / denominator)
    return torch.stack(losses)


def flow_matching_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    valid_coverage: torch.Tensor,
    edit_coverage: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    global_valid_count: int | None = None,
) -> torch.Tensor:
    errors = flow_matching_errors(
        prediction, target_velocity, valid_coverage, edit_coverage, sample_mask
    )
    count = prediction.shape[0] if sample_mask is None else int(sample_mask.sum())
    denominator = count if global_valid_count is None else global_valid_count
    if denominator < count or denominator < 0:
        raise ValueError(
            "The global sample denominator cannot be smaller than the local valid count"
        )
    return errors.sum() / denominator if denominator else prediction.sum() * 0


def backward_flow_matching(
    renderer: KleinRenderer,
    example: FlowMatchingExample,
    epoch: int,
    step: int,
    global_valid_count: int,
    text_dropout: float = 0.1,
    control_dropout: float = 0.1,
) -> dict[str, float]:
    if not 0 <= text_dropout <= 1 or not 0 <= control_dropout <= 1 or global_valid_count < 0:
        raise ValueError("Invalid flow matching batch or dropout configuration")
    if not example.sample_valid:
        return {"loss": 0.0, "samples": 0.0, "text_dropout": 0.0}
    if global_valid_count == 0:
        raise ValueError("A real sample requires a positive global denominator")
    target = example.target_latents.detach().to(renderer.device, torch.float32)
    if (
        target.shape != (1, example.bundle.num_target_tokens, 128)
        or not torch.isfinite(target).all()
    ):
        raise ValueError("Invalid target latent for flow matching")
    sigma_generator = torch.Generator().manual_seed(
        stream_seed(example.sample_id, epoch, step, "sigma")
    )
    sigma = (
        torch.randn(1, generator=sigma_generator)
        .sigmoid()
        .clamp(1e-4, 1 - 1e-4)
        .to(renderer.device)
    )
    noise = gaussian_noise(
        tuple(target.shape), stream_seed(example.sample_id, epoch, step, "noise"), renderer.device
    )
    state = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
    target_velocity = noise - target
    dropout_generator = torch.Generator().manual_seed(
        stream_seed(example.sample_id, epoch, step, "text_dropout")
    )
    dropped = bool(torch.rand((), generator=dropout_generator) < text_dropout)
    bundle = example.bundle.to(renderer.device)
    active_controls = {}
    for name, value in bundle.controls.items():
        control_generator = torch.Generator().manual_seed(
            stream_seed(example.sample_id, epoch, step, "control_dropout", name)
        )
        if float(torch.rand((), generator=control_generator)) >= control_dropout:
            active_controls[name] = value
    bundle = replace(
        bundle,
        cfg_scale=1.0,
        positive_embeds=bundle.negative_embeds if dropped else bundle.positive_embeds,
        controls=active_controls,
    )
    valid = patch_coverage(example.valid_mask, bundle.canvas_hw)[None].to(renderer.device)
    edit = (
        torch.zeros_like(valid)
        if example.edit_mask is None
        else patch_coverage(example.edit_mask, bundle.canvas_hw)[None].to(renderer.device)
    )
    with renderer.lock:
        renderer.transformer.train()
        renderer._set_adapter("policy")
        context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if renderer.config.saved_tensor_cpu_offload
            else nullcontext()
        )
        with context:
            prediction = renderer.predict_branch(bundle, state, sigma, "policy", negative=False)
            loss = flow_matching_loss(
                prediction, target_velocity, valid, edit, global_valid_count=global_valid_count
            )
            loss.backward()
    return {
        "loss": float(loss.detach()),
        "samples": 1.0,
        "text_dropout": float(dropped),
        "sigma": float(sigma),
    }
