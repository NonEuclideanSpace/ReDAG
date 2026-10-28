from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import json
import math

import torch

from .types import ReplayStep


def stream_seed(*parts: object) -> int:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(blake2b(encoded, digest_size=8).digest(), "little")


def gaussian_noise(
    shape: tuple[int, ...], seed: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def sigma_schedule(steps: int = 50, target_tokens: int = 1024) -> torch.Tensor:
    if steps < 4 or target_tokens < 1:
        raise ValueError(
            "The Flow-SDE schedule needs at least four steps and a positive target length"
        )
    m200 = 0.00016927 * target_tokens + 0.45666666
    m10 = 0.0000873809524 * target_tokens + 1.89833333
    shift = m200 if target_tokens > 4300 else m10 + (steps - 10) / 190 * (m200 - m10)
    times = 1 - torch.arange(steps, dtype=torch.float64) / steps
    exponential = math.exp(shift)
    sigmas = exponential / (exponential + 1 / times - 1)
    result = torch.cat((sigmas, torch.zeros(1, dtype=torch.float64))).float()
    if not torch.all(result[:-1] > result[1:]) or result[-1] != 0:
        raise ValueError("Invalid descending sigma schedule")
    return result


@dataclass(frozen=True)
class Transition:
    a: torch.Tensor
    b: torch.Tensor
    std: torch.Tensor
    sigma: torch.Tensor

    def mean(self, state: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return (
            self.a.to(state.device) * state.detach().float()
            + self.b.to(state.device) * velocity.float()
        )


class FlowSDE:
    def __init__(self, sigmas: torch.Tensor, eta: float = 0.3):
        self.sigmas = sigmas.detach().float().cpu().clone()
        if self.sigmas.ndim != 1 or len(self.sigmas) < 5:
            raise ValueError("A sigma schedule with at least four transitions is required")
        if not torch.isfinite(self.sigmas).all() or not torch.all(
            self.sigmas[:-1] > self.sigmas[1:]
        ):
            raise ValueError("Sigmas must be finite and strictly decreasing")
        if self.sigmas[0] != 1 or self.sigmas[-1] != 0 or eta <= 0 or not math.isfinite(eta):
            raise ValueError("Flow-SDE requires endpoints one and zero and positive finite eta")
        self.eta = eta
        sigma = self.sigmas[:-1]
        delta = self.sigmas[1:] - sigma
        safe_sigma = sigma.clone()
        safe_sigma[0] = sigma[1]
        diffusion_squared = eta**2 * sigma / (1 - safe_sigma)
        self.std = (diffusion_squared * -delta).sqrt()
        self.a = 1 + diffusion_squared * delta / (2 * sigma)
        self.b = (1 + diffusion_squared * (1 - sigma) / (2 * sigma)) * delta
        if not torch.isfinite(self.std).all() or not torch.all(self.std > 1e-8):
            raise ValueError(
                "Every trained transition must have positive finite standard deviation"
            )

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1

    def transition(self, index: int) -> Transition:
        if not 0 <= index < self.steps:
            raise IndexError(index)
        return Transition(self.a[index], self.b[index], self.std[index], self.sigmas[index])


def gaussian_mean_logp(
    next_state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor | float
) -> torch.Tensor:
    deviation = torch.as_tensor(std, dtype=torch.float32, device=mean.device)
    if not torch.isfinite(deviation).all() or not torch.all(deviation > 1e-8):
        raise ValueError("Gaussian log probability requires positive finite transition deviation")
    terms = (
        (next_state.detach().float() - mean.float()).square() / deviation.square()
        + 2 * deviation.log()
        + math.log(2 * math.pi)
    )
    return -0.5 * terms.flatten(1).mean(dim=1)


def gaussian_mean_kl(
    mean: torch.Tensor, reference_mean: torch.Tensor, std: torch.Tensor | float
) -> torch.Tensor:
    deviation = torch.as_tensor(std, dtype=torch.float32, device=mean.device)
    if not torch.isfinite(deviation).all() or not torch.all(deviation > 1e-8):
        raise ValueError("Gaussian KL requires positive finite transition deviation")
    return (mean.float() - reference_mean.detach().float()).square().flatten(1).mean(dim=1) / (
        2 * deviation.square()
    )


def stratified_steps(steps: int, count: int = 4, seed: int = 0) -> tuple[ReplayStep, ...]:
    if not 1 <= count <= steps:
        raise ValueError(
            "The number of replay strata must lie between one and the number of transitions"
        )
    quotient, remainder = divmod(steps, count)
    start = 0
    selected = []
    for index in range(count):
        size = quotient + int(index < remainder)
        generator = torch.Generator().manual_seed(stream_seed(seed, "step_subset", index))
        offset = int(torch.randint(size, (), generator=generator))
        selected.append(ReplayStep(index, start + offset, 1 / size, size / steps))
        start += size
    return tuple(selected)


def velocity_vjp(
    state: torch.Tensor,
    next_state: torch.Tensor,
    velocity: torch.Tensor,
    reference_velocity: torch.Tensor,
    transition: Transition,
    advantage: float,
    coefficient: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mean = transition.mean(state, velocity)
    reference_mean = transition.mean(state, reference_velocity).detach()
    logp = gaussian_mean_logp(next_state, mean, transition.std)
    kl = gaussian_mean_kl(mean, reference_mean, transition.std)
    if mean.shape[0] != 1:
        raise ValueError("Replay uses one image transition per microbatch")
    dimension = mean[0].numel()
    gradient = (
        coefficient
        * transition.b.to(mean.device)
        * (advantage * (mean - next_state.detach().float()) + beta * (mean - reference_mean))
        / (dimension * transition.std.to(mean.device).square())
    )
    metrics = {
        "loss": float(coefficient * (-advantage * logp + beta * kl)),
        "pg": float(-coefficient * advantage * logp),
        "kl": float(coefficient * kl),
        "mean_logp": float(logp),
    }
    return gradient.detach(), metrics
