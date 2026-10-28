import itertools
import math

import pytest
import torch

from agentic_image.actor.head import head_backward
from agentic_image.learning import (
    JointTransaction,
    ScoreWeights,
    TrainingComponent,
    hierarchical_rloo,
    make_optimizer,
)


def test_frozen_head_chunking_preserves_full_prefix_gradient():
    torch.manual_seed(17)
    inputs = torch.randn(7, 4)
    parameter = torch.nn.Parameter(torch.randn(4, 3) * 0.2)
    head = torch.nn.Linear(3, 11, bias=False)
    head.requires_grad_(False)
    positions = torch.tensor([2, 3, 4, 5])
    targets = torch.tensor([0, 10, 8, 9])
    mask = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    advantages = torch.tensor([0.7, 0.7, -0.2, -0.2])
    coefficients = torch.tensor([1 / 120, 1 / 120, 1 / 240, 1 / 240])
    reference = torch.tanh(torch.cumsum(inputs @ (parameter.detach() + 0.03), dim=0))[positions]
    hidden = torch.tanh(torch.cumsum(inputs @ parameter, dim=0))[positions]
    logp = torch.log_softmax(head(hidden).float(), dim=-1)
    logq = torch.log_softmax(head(reference).float(), dim=-1)
    selected = logp.gather(-1, targets[:, None]).squeeze(-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)
    loss = (coefficients * mask * (-advantages * selected + 0.02 * kl)).sum()
    loss.backward()
    expected = parameter.grad.clone()
    parameter.grad = None
    hidden = torch.tanh(torch.cumsum(inputs @ parameter, dim=0))[positions]
    actual = head_backward(
        hidden, head, targets, mask, coefficients, advantages, reference, 0.02, chunk_size=2
    )
    torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-7)
    assert actual.loss == pytest.approx(loss.item(), abs=1e-7)
    assert parameter.grad.norm() > 0


def test_hierarchical_credit_matches_expected_reward_gradient():
    theta, phi = 0.31, -0.27

    def sigmoid(value):
        return 1 / (1 + math.exp(-value))

    root_probability = sigmoid(theta)
    roots = []
    direct = [0.0, 0.0]
    for root in (0, 1):
        root_prob = root_probability if root else 1 - root_probability
        suffixes = []
        for action, image in itertools.product((0, 1), repeat=2):
            action_prob = sigmoid(theta + 0.7 * root)
            image_prob = sigmoid(phi + 0.4 * root - 0.3 * action)
            probability = (action_prob if action else 1 - action_prob) * (
                image_prob if image else 1 - image_prob
            )
            reward = float(action == root) + 0.8 * float(image == root) - 0.15 * action
            suffixes.append((probability, reward, action - action_prob, image - image_prob))
            direct[0] += (
                root_prob * probability * reward * (root - root_probability + action - action_prob)
            )
            direct[1] += root_prob * probability * reward * (image - image_prob)
        for left, right in itertools.product(suffixes, repeat=2):
            roots.append((root_prob * left[0] * right[0], root - root_probability, left, right))
    estimated = torch.zeros(2, dtype=torch.float64)
    for first, second in itertools.product(roots, repeat=2):
        probability = first[0] * second[0]
        rewards = torch.tensor(
            [[[first[2][1], first[3][1]], [second[2][1], second[3][1]]]], dtype=torch.float64
        )
        advantages = hierarchical_rloo(rewards)
        theta_score = (advantages.root[0, 0] * first[1] + advantages.root[0, 1] * second[1]) / 2
        phi_score = 0.0
        for n, group in enumerate((first, second)):
            for m, suffix in enumerate(group[2:]):
                theta_score += advantages.suffix[0, n, m] * suffix[2] / 4
                phi_score += advantages.suffix[0, n, m] * suffix[3] / 4
        estimated += probability * torch.tensor([theta_score, phi_score], dtype=torch.float64)
    torch.testing.assert_close(
        estimated, torch.tensor(direct, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    weights = ScoreWeights()
    assert weights.actor_root == 1 / (6 * 2 * 10240)
    assert weights.actor_suffix == 1 / (6 * 2 * 2 * 10240)
    assert weights.renderer == 1 / (6 * 2 * 2 * 3)


def test_joint_transaction_rolls_back_both_optimizers_and_rng(tmp_path, monkeypatch):
    first = torch.nn.Linear(2, 1)
    second = torch.nn.Linear(2, 1)
    components = []
    for name, model in (("actor", first), ("renderer", second)):
        optimizer = make_optimizer(model.parameters(), 0.01)
        components.append(
            TrainingComponent(
                name, list(model.parameters()), optimizer, model.state_dict, model.load_state_dict
            )
        )
    transaction = JointTransaction(tmp_path, components)
    originals = [
        {name: value.clone() for name, value in model.state_dict().items()}
        for model in (first, second)
    ]
    transaction.begin("v0", {"step": 0})
    random_state = torch.get_rng_state().clone()
    for component in components:
        for parameter in component.parameters:
            parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="outside its bounds"):
        transaction.step_and_commit("v1", probe=lambda: {"actor_kl": 0.2}, max_retries=1)
    for model, original, component in zip((first, second), originals, components):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        assert not component.optimizer.state
    assert torch.equal(torch.get_rng_state(), random_state)
    recovered = JointTransaction(tmp_path, components).load_committed()
    assert recovered["data_cursor"] == {"step": 0}
    assert recovered["recovery_status"] == "ABORTED"
    from agentic_image.learning import transaction as transaction_module

    original_write = transaction_module.atomic_json

    def fail_latest_hint(path, value):
        if path.name == "LATEST.json":
            raise OSError("Unavailable hint path")
        original_write(path, value)

    monkeypatch.setattr(transaction_module, "atomic_json", fail_latest_hint)
    transaction.begin("v0", {"step": 0})
    committed = transaction.step_and_commit(
        "v1", probe=lambda: {"actor_kl": 0.001}, data_cursor={"step": 1}
    )
    assert committed["status"] == "COMMITTED"
    assert "latest_hint_error" in committed
    assert transaction.load_committed()["data_cursor"] == {"step": 1}
