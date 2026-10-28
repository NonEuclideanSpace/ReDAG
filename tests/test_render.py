from dataclasses import replace

import pytest
import torch

from agentic_image.render import (
    ConditionBundle,
    ConditionCompiler,
    ConditionRequest,
    FlowSDE,
    ReferenceSpec,
    RegionSpec,
    flow_matching_loss,
    gaussian_mean_kl,
    gaussian_mean_logp,
    pack,
    patchify,
    sigma_schedule,
    stratified_steps,
    unpack,
    unpatchify,
)
from agentic_image.render.attention import query_chunk_attention
from agentic_image.render.sde import velocity_vjp


class FixedTextEncoder:
    def encode_text(self, texts):
        return torch.stack(
            [torch.full((512, 12288), len(text) / 100, dtype=torch.bfloat16) for text in texts]
        )


def test_latent_codec_inverts_non_square_multichannel_layout():
    original = torch.arange(2 * 32 * 6 * 10).reshape(2, 32, 6, 10).float()
    tokens = pack(patchify(original))
    assert tokens.shape == (2, 15, 128)
    assert torch.equal(unpatchify(unpack(tokens, (3, 5))), original)
    assert torch.equal(tokens[0, 0, :4], original[0, 0, :2, :2].flatten())


def test_moving_entity_uses_independent_source_target_and_reference_masks(tmp_path):
    source_mask = torch.zeros(32, 32)
    source_mask[:16, :16] = 1
    target_mask = torch.zeros(32, 32)
    target_mask[16:, 16:] = 1
    appearance_mask = torch.zeros(32, 32)
    appearance_mask[:16, 16:] = 1
    region = RegionSpec(
        "region:cup", ("entity:cup",), "red cup", target_mask, source_mask, ("appearance:cup",)
    )
    references = (
        ReferenceSpec("source", "image:source", torch.zeros(1, 4, 128)),
        ReferenceSpec(
            "appearance", "appearance:cup", torch.ones(1, 4, 128), {"entity:cup": appearance_mask}
        ),
    )
    bundle = ConditionCompiler(FixedTextEncoder()).compile(
        ConditionRequest("Move the red cup", (region,), references, (32, 32))
    )
    mask = bundle.positive_mask[0, 0]
    region_text = 512
    target, source, appearance = 1024, 1028, 1032
    assert mask[region_text, target + 3] and not mask[region_text, target]
    assert mask[region_text, source] and not mask[region_text, source + 3]
    assert mask[region_text, appearance + 1] and not mask[region_text, appearance]
    assert mask[appearance + 1, region_text] and not mask[appearance, region_text]
    assert torch.equal(bundle.positive_mask, bundle.negative_mask)
    assert torch.equal(bundle.image_ids[:, 0], torch.tensor([0] * 4 + [10] * 4 + [20] * 4))
    assert torch.equal(bundle.text_ids[:512], bundle.text_ids[512:])
    path = tmp_path / "condition.pt"
    bundle.save(path)
    restored = ConditionBundle.load(path)
    assert torch.equal(restored.positive_mask, bundle.positive_mask)
    assert restored.token_span_table == bundle.token_span_table


def test_tiny_region_fallback_and_explicit_global_appearance():
    tiny = torch.zeros(32, 32)
    tiny[17, 18] = 1
    region = RegionSpec("small", ("entity",), "dot", tiny)
    bundle = ConditionCompiler(FixedTextEncoder()).compile(
        ConditionRequest("A dot", (region,), canvas_hw=(32, 32))
    )
    assert bundle.target_membership.tolist() == [[False, False, False, True]]
    assert bundle.raster_fallbacks == ("small:target",)
    reference = ReferenceSpec("appearance", "ref", torch.zeros(1, 4, 128), scope="global")
    global_bundle = ConditionCompiler(FixedTextEncoder()).compile(
        ConditionRequest("A cup", references=(reference,), canvas_hw=(32, 32))
    )
    assert global_bundle.positive_mask.all()
    with pytest.raises(ValueError, match="UNBOUND_REFERENCE"):
        ConditionCompiler(FixedTextEncoder()).compile(
            ConditionRequest(
                "A cup", references=(replace(reference, scope="regional"),), canvas_hw=(32, 32)
            )
        )


def test_flow_sde_schedule_and_time_strata_use_actual_transition_variance():
    kernel = FlowSDE(sigma_schedule())
    assert kernel.std[0].item() == pytest.approx(0.3, rel=1e-5)
    assert kernel.std[-1].item() == pytest.approx(0.0318446337, rel=1e-5)
    assert torch.all(kernel.std > 0)
    steps = stratified_steps(50, 4, 31)
    assert [step.weight for step in steps] == [0.26, 0.26, 0.24, 0.24]
    assert [step.sampling_probability for step in steps] == [1 / 13, 1 / 13, 1 / 12, 1 / 12]
    assert all(
        start <= step.step_index < stop
        for step, start, stop in zip(steps, [0, 13, 26, 38], [13, 26, 38, 50])
    )
    mean = torch.full((1, 4, 3), 0.025)
    zero = torch.zeros_like(mean)
    assert gaussian_mean_kl(mean, zero, 0.17).item() == pytest.approx(
        0.025**2 / (2 * 0.17**2), rel=1e-6
    )


def test_sequential_cfg_vjp_matches_joint_autograd_with_negative_branch():
    torch.manual_seed(12)
    state, next_state = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    inputs_positive, inputs_negative = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    initial = torch.randn(3, 3) * 0.1
    parameter = initial.clone().requires_grad_()
    reference_parameter = initial + 0.03
    kernel = FlowSDE(sigma_schedule(20, 4))
    transition = kernel.transition(13)
    positive = torch.tanh(inputs_positive @ parameter)
    negative = torch.tanh(inputs_negative @ parameter)
    velocity = 4 * positive - 3 * negative
    reference_velocity = 4 * torch.tanh(inputs_positive @ reference_parameter) - 3 * torch.tanh(
        inputs_negative @ reference_parameter
    )
    mean = transition.mean(state, velocity)
    reference_mean = transition.mean(state, reference_velocity)
    loss = 0.17 * (
        -0.7 * gaussian_mean_logp(next_state, mean, transition.std)
        + 0.01 * gaussian_mean_kl(mean, reference_mean, transition.std)
    )
    loss.backward()
    expected = parameter.grad.clone()
    parameter = initial.clone().requires_grad_()
    with torch.no_grad():
        current = 4 * torch.tanh(inputs_positive @ parameter) - 3 * torch.tanh(
            inputs_negative @ parameter
        )
        gradient, metrics = velocity_vjp(
            state, next_state, current, reference_velocity, transition, 0.7, 0.17, 0.01
        )
    (4 * gradient * torch.tanh(inputs_positive @ parameter)).sum().backward()
    (-3 * gradient * torch.tanh(inputs_negative @ parameter)).sum().backward()
    assert torch.allclose(parameter.grad, expected, rtol=2e-5, atol=2e-6)
    assert metrics["loss"] == pytest.approx(loss.item(), rel=1e-6)


def test_query_chunk_attention_matches_full_forward_and_gradients():
    torch.manual_seed(7)
    query, key, value = [torch.randn(1, 17, 2, 8).requires_grad_() for _ in range(3)]
    mask = torch.rand(1, 1, 17, 17) > 0.3
    mask[..., torch.arange(17), torch.arange(17)] = True
    full = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)
    expected = torch.autograd.grad(full.square().sum(), (query, key, value))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        chunked = query_chunk_attention(query, key, value, mask, chunk_size=5)
    actual = torch.autograd.grad(chunked.square().sum(), (query, key, value))
    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-5)
    assert all(
        torch.allclose(left, right, atol=2e-6, rtol=1e-5) for left, right in zip(actual, expected)
    )


def test_flow_matching_uses_global_real_sample_denominator_and_mask_weights():
    prediction = torch.arange(5 * 4 * 3).reshape(5, 4, 3).float().requires_grad_()
    target = torch.zeros_like(prediction)
    valid = torch.ones(5, 4)
    valid[4] = 0
    edit = torch.zeros_like(valid)
    edit[:, 0] = 1
    samples = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
    full = flow_matching_loss(prediction, target, valid, edit, samples, global_valid_count=4)
    expected = torch.autograd.grad(full, prediction)[0]
    actual = torch.zeros_like(prediction)
    for index in range(5):
        local = prediction[index : index + 1].detach().clone().requires_grad_()
        loss = flow_matching_loss(
            local,
            target[index : index + 1],
            valid[index : index + 1],
            edit[index : index + 1],
            samples[index : index + 1],
            global_valid_count=4,
        )
        actual[index : index + 1] = torch.autograd.grad(loss, local)[0]
    assert torch.equal(expected, actual)
    assert torch.count_nonzero(actual[4]) == 0
    with pytest.raises(ValueError, match="valid target supervision"):
        flow_matching_loss(prediction[4:], target[4:], valid[4:], edit[4:])


def test_optional_controls_start_at_zero_and_only_modify_target_tokens():
    from agentic_image.core.geometry import make_frame
    from agentic_image.render.controls import ControlAdapters, depth_tokens, pose_tokens

    frame = make_frame((16, 32), (32, 32), "control-frame")
    depth = depth_tokens(torch.ones(16, 32).numpy(), torch.ones(16, 32).numpy(), frame)
    points = torch.zeros(17, 3)
    points[0] = torch.tensor([16, 8, 0.9])
    pose = pose_tokens(points.numpy(), frame)
    assert depth.shape == (1, 4, 512)
    assert pose.shape == (1, 4, 4608)
    controls = ControlAdapters()
    assert sum(parameter.numel() for parameter in controls.policy.parameters()) == 860288
    transformer = torch.nn.Module()
    transformer.x_embedder = torch.nn.Linear(128, 4096)
    controls.attach(transformer)
    inputs = torch.randn(1, 8, 128)
    baseline = transformer.x_embedder(inputs)
    with controls.condition({"depth": depth, "pose": pose}, 4):
        assert torch.equal(transformer.x_embedder(inputs), baseline)
    with torch.no_grad():
        controls.policy["depth"][2].bias.fill_(0.5)
    with controls.condition({"depth": depth}, 4):
        changed = transformer.x_embedder(inputs)
    assert torch.allclose(changed[:, :4], baseline[:, :4] + 0.5)
    assert torch.equal(changed[:, 4:], baseline[:, 4:])
    controls.freeze_reference()
    with torch.no_grad():
        controls.policy["depth"][2].bias.add_(1)
    controls.set_adapter("reference")
    with controls.condition({"depth": depth}, 4):
        assert torch.equal(transformer.x_embedder(inputs), changed)


def test_real_tiny_flux2_sft_and_stochastic_replay_use_the_same_model():
    diffusers = pytest.importorskip("diffusers")
    from agentic_image.render import (
        FlowMatchingExample,
        KleinRenderer,
        RendererConfig,
        backward_flow_matching,
    )

    torch.manual_seed(23)
    transformer = diffusers.Flux2Transformer2DModel(
        in_channels=128,
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        joint_attention_dim=12288,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    ).to(torch.bfloat16)
    renderer = KleinRenderer(
        transformer,
        RendererConfig(
            device="cpu",
            attention_backend="query_chunk",
            query_chunk_size=128,
            strict_architecture=False,
            lora_rank=2,
            lora_alpha=2,
            steps=4,
        ),
    )
    bundle = ConditionCompiler(FixedTextEncoder()).compile(
        ConditionRequest("A red cup", canvas_hw=(32, 32), attention_backend="query_chunk")
    )
    optimizer = torch.optim.AdamW(renderer.policy_parameters(), lr=1e-3)
    example = FlowMatchingExample(
        "tiny-real-model", bundle, torch.randn(1, 4, 128), torch.ones(32, 32)
    )
    metrics = backward_flow_matching(renderer, example, epoch=0, step=0, global_valid_count=1)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert (
        sum(
            float(parameter.grad.norm())
            for parameter in renderer.policy_parameters()
            if parameter.grad is not None
        )
        > 0
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    renderer.freeze_reference()
    trace = renderer.rollout(bundle, 31, behavior_version="tiny-sft")
    assert trace.states.shape == (5, 1, 4, 128)
    assert trace.states.dtype == torch.float32
    replay = renderer.backward_replay(
        bundle, trace, advantage=0.5, coefficient=1 / 12, expected_behavior_version="tiny-sft"
    )
    assert replay["max_logp_error"] < 2e-3
    assert replay["kl"] == pytest.approx(0.0, abs=1e-7)
    assert (
        sum(
            float(parameter.grad.norm())
            for parameter in renderer.policy_parameters()
            if parameter.grad is not None
        )
        > 0
    )
    assert all(
        parameter.grad is None
        for name, parameter in transformer.named_parameters()
        if ".reference." in name
    )
