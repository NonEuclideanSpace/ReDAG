import math

import torch
import torch.distributed as distributed


def make_optimizer(parameters, learning_rate, stage="rl", epsilon=1e-8):
    parameters = list(parameters)
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Optimizers must receive the FP32 policy adapter parameters")
    if stage not in {"sft", "rl"}:
        raise ValueError("Unknown optimization stage")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=epsilon,
        weight_decay=0.01 if stage == "sft" else 0.0,
    )


def make_sft_scheduler(optimizer, total_steps, warmup_fraction=0.05, minimum_ratio=0.1):
    if total_steps < 1:
        raise ValueError("SFT update count must be positive")
    warmup = max(1, int(total_steps * warmup_fraction))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup - 1))
        return minimum_ratio + (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def make_rl_scheduler(optimizer, warmup_steps=10):
    if warmup_steps < 0:
        raise ValueError("Warmup cannot be negative")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
    )


def sum_policy_gradients(parameters, group=None):
    parameters = list(parameters)
    active = distributed.is_available() and distributed.is_initialized()
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if active:
            distributed.all_reduce(parameter.grad, op=distributed.ReduceOp.SUM, group=group)


def finite_gradients(parameters, group=None):
    parameters = list(parameters)
    if not parameters:
        return True
    finite = torch.tensor(
        int(
            all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            )
        ),
        device=parameters[0].device,
        dtype=torch.int32,
    )
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(finite, op=distributed.ReduceOp.MIN, group=group)
    return bool(finite.item())


def clip_policy_gradients(parameters, max_norm=1.0):
    parameters = list(parameters)
    if not finite_gradients(parameters):
        raise FloatingPointError("A policy gradient is non-finite; no optimizer may step")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    return {
        "gradient_norm": float(norm),
        "clip_ratio": min(1.0, max_norm / max(float(norm), 1e-12)),
    }
