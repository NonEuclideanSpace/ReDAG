from types import SimpleNamespace

import pytest
import torch

from agentic_image.actor.config import ACTOR_NUMERICS_PROFILE
from agentic_image.actor.precision import actor_matmul_precision
from agentic_image.actor.worker import ActorWorker


def test_actor_precision_restores_after_nested_failure():
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        with pytest.raises(RuntimeError):
            with actor_matmul_precision():
                assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                with actor_matmul_precision():
                    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                    raise RuntimeError("Interrupted actor execution")
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


def test_old_behavior_profile_cannot_enter_rl_replay():
    actor = ActorWorker.__new__(ActorWorker)
    actor.config = SimpleNamespace(numerics_profile=ACTOR_NUMERICS_PROFILE)
    with pytest.raises(ValueError, match="different actor numerical profile"):
        actor._validate_sample_numerics(SimpleNamespace(numerics_profile=None))
    actor._validate_sample_numerics(SimpleNamespace(numerics_profile=ACTOR_NUMERICS_PROFILE))
