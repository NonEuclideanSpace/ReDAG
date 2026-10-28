from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HeadMetrics:
    loss: float
    policy_gradient: float
    kl: float
    token_count: int


def head_backward(
    hidden,
    head,
    targets,
    mask,
    coefficient,
    advantage=1.0,
    reference_hidden=None,
    beta=0.0,
    temperature=1.0,
    chunk_size=32,
):
    if hidden.ndim != 2 or targets.shape != mask.shape or hidden.shape[0] != targets.numel():
        raise ValueError("Head replay requires aligned hidden states, labels and masks")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("The vocabulary head must remain frozen")
    if beta and reference_hidden is None:
        raise ValueError("Reference hidden states are required for KL")
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("Invalid head replay configuration")
    detached = hidden.detach().requires_grad_(True)
    coefficient = torch.as_tensor(coefficient, device=hidden.device, dtype=torch.float32)
    advantage = torch.as_tensor(advantage, device=hidden.device, dtype=torch.float32).detach()
    metrics = torch.zeros(3, device=hidden.device, dtype=torch.float64)
    for start in range(0, targets.numel(), chunk_size):
        stop = min(start + chunk_size, targets.numel())
        logits = head(detached[start:stop]).float() / temperature
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, targets[start:stop, None]).squeeze(-1)
        weights = mask[start:stop].to(log_probs.dtype)
        local_coefficient = coefficient if coefficient.ndim == 0 else coefficient[start:stop]
        local_advantage = advantage if advantage.ndim == 0 else advantage[start:stop]
        pg = -(weights * local_coefficient * local_advantage * selected).sum()
        kl = selected.new_zeros(())
        if reference_hidden is not None and beta:
            with torch.no_grad():
                reference_log_probs = F.log_softmax(
                    head(reference_hidden[start:stop]).float() / temperature, dim=-1
                )
            per_token_kl = (log_probs.exp() * (log_probs - reference_log_probs)).sum(-1)
            kl = (weights * local_coefficient * per_token_kl).sum()
        loss = pg + beta * kl
        loss.backward()
        metrics += torch.stack((loss.detach(), pg.detach(), kl.detach())).double()
    hidden.backward(detached.grad)
    return HeadMetrics(*metrics.cpu().tolist(), int(mask.sum().item()))


@torch.no_grad()
def head_log_probs(hidden, head, targets, temperature=1.0, chunk_size=32):
    result = []
    for start in range(0, targets.numel(), chunk_size):
        stop = min(start + chunk_size, targets.numel())
        logits = head(hidden[start:stop]).float() / temperature
        result.append(
            F.log_softmax(logits, dim=-1).gather(-1, targets[start:stop, None]).squeeze(-1)
        )
    return torch.cat(result)
