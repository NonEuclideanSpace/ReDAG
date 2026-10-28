from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class Advantages:
    root: torch.Tensor
    suffix: torch.Tensor
    root_mean_reward: torch.Tensor


def hierarchical_rloo(rewards):
    rewards = torch.as_tensor(rewards)
    if rewards.ndim != 3 or rewards.shape[0] < 1 or min(rewards.shape[1:]) < 2:
        raise ValueError(
            "Hierarchical RLOO requires complete [P, N, M] groups with N and M at least two"
        )
    if not rewards.is_floating_point():
        rewards = rewards.float()
    if not torch.isfinite(rewards).all():
        raise ValueError("An incomplete or non-finite reward group cannot be updated")
    mean = rewards.mean(-1)
    root = mean - (mean.sum(-1, keepdim=True) - mean) / (rewards.shape[1] - 1)
    suffix = rewards - (rewards.sum(-1, keepdim=True) - rewards) / (rewards.shape[2] - 1)
    return Advantages(root.detach(), suffix.detach(), mean.detach())


@dataclass(frozen=True)
class ScoreWeights:
    tasks: int = 6
    roots: int = 2
    suffixes: int = 2
    max_actions: int = 10
    max_action_tokens: int = 1024
    max_renders: int = 3

    def __post_init__(self):
        if (
            min(self.tasks, self.max_actions, self.max_action_tokens, self.max_renders) < 1
            or min(self.roots, self.suffixes) < 2
        ):
            raise ValueError("Invalid hierarchical batch or trajectory budgets")

    @property
    def actor_root(self):
        return 1.0 / (self.tasks * self.roots * self.max_actions * self.max_action_tokens)

    @property
    def actor_suffix(self):
        return self.actor_root / self.suffixes

    @property
    def renderer(self):
        return 1.0 / (self.tasks * self.roots * self.suffixes * self.max_renders)


def phase_action_weights(
    phases: Sequence[int], global_counts=(5, 7, 5, 3), phase_weights=(0.25, 0.35, 0.25, 0.15)
):
    if len(global_counts) != len(phase_weights) or any(count <= 0 for count in global_counts):
        raise ValueError("Every SFT phase needs a positive global action count")
    if abs(sum(phase_weights) - 1.0) > 1e-8:
        raise ValueError("SFT phase weights must sum to one")
    if any(phase < 0 or phase >= len(global_counts) for phase in phases):
        raise ValueError("Unknown SFT phase")
    return [phase_weights[phase] / global_counts[phase] for phase in phases]


def phase_balanced_loss(
    action_losses, phases, global_counts=(5, 7, 5, 3), phase_weights=(0.25, 0.35, 0.25, 0.15)
):
    weights = torch.tensor(
        phase_action_weights(phases, global_counts, phase_weights),
        dtype=action_losses.dtype,
        device=action_losses.device,
    )
    if action_losses.ndim != 1 or action_losses.numel() != len(phases):
        raise ValueError("Each SFT action requires one normalized loss and one phase")
    return (action_losses * weights).sum()


def masked_action_ce(log_probs, mask):
    mask = mask.to(log_probs.dtype)
    if log_probs.shape != mask.shape or not bool(mask.sum() > 0):
        raise ValueError("A real SFT action must contain supervised tokens")
    return -(log_probs * mask).sum() / mask.sum()
