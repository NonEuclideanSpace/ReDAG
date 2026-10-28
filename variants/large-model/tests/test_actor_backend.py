from types import SimpleNamespace

import pytest
import torch

from agentic_image.actor import ActorConfig, ActorSnapshot, ActorWorker
from agentic_image.actor.config import ACTOR_TARGETS

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def test_real_qwen_hybrid_cache_visual_prefix_and_replay(tmp_path):
    torch.manual_seed(17)
    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 248320,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 2, 4],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 64,
        },
    )
    config._attn_implementation = "sdpa"
    base = transformers.Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    base.requires_grad_(False)
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0, target_modules=ACTOR_TARGETS),
        adapter_name="policy",
    )
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    base.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: str(ids)))
    worker = ActorWorker(
        model,
        processor,
        ActorConfig(
            device="cpu", max_action_tokens=4, image_cache_directory=str(tmp_path / "images")
        ),
    )
    ids = torch.tensor([[1, 248053, 248056, 248056, 248056, 248056, 248054, 2, 3]])
    types = torch.zeros_like(ids)
    types[0, 2:6] = 1
    attention = torch.ones_like(ids)
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 24)
    with torch.no_grad():
        features = torch.cat(
            base.model.get_image_features(pixels, grid, return_dict=True).pooler_output
        )
        positions, delta = worker._position_ids(ids, types, attention, grid)
    snapshot = ActorSnapshot(
        {
            "input_ids": ids,
            "attention_mask": attention,
            "mm_token_type_ids": types,
            "image_grid_thw": grid,
        },
        torch.arange(2, 6),
        features,
        positions,
        delta,
    )
    processor.apply_chat_template = lambda *args, **kwargs: dict(
        snapshot.tensors, pixel_values=pixels
    )
    snapshot = worker.prepare([], image_artifact_ids=("source-image",))
    torch.testing.assert_close(snapshot.tensors["pixel_values"], pixels, rtol=0, atol=0)
    assert snapshot.metadata["image_patch_offsets"] == [0, 16]
    assert snapshot.metadata["image_token_offsets"] == [0, 4]
    serialized = snapshot.state_dict()
    assert "pixel_values" not in serialized["tensors"]
    assert serialized["image_features"] is None
    restored = ActorSnapshot.from_state_dict(serialized)
    torch.testing.assert_close(restored.tensors["pixel_values"], pixels, rtol=0, atol=0)
    torch.testing.assert_close(restored.image_features, snapshot.image_features, rtol=0, atol=0)
    cached = worker.prepare([], image_artifact_ids=("source-image",))
    assert cached.visual_files == snapshot.visual_files
    completion = torch.tensor([7, 8, 9, 248046])
    with torch.no_grad():
        expected = base.model(
            input_ids=torch.cat((ids, completion[None]), dim=-1),
            attention_mask=torch.ones(1, ids.shape[-1] + completion.numel(), dtype=torch.long),
            mm_token_type_ids=torch.cat((types, torch.zeros_like(completion)[None]), dim=-1),
            pixel_values=pixels,
            image_grid_thw=grid,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0, ids.shape[-1] - 1 : -1]
        actual = worker._hidden(snapshot, completion)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sample = worker.sample(snapshot, 42, "tiny-behavior", probe_positions=(0, 1))
    replay = worker.replay_log_probs(sample)
    torch.testing.assert_close(replay, sample.behavior_log_probs, atol=0.015, rtol=0)
    special = torch.tensor([248056, 248057, 248053, 248046])
    metrics = worker.sft_backward(snapshot, special)
    assert metrics.token_count == 4
    assert (
        sum(
            parameter.grad.square().sum().item()
            for parameter in worker.policy_parameters()
            if parameter.grad is not None
        )
        > 0
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if ".policy." not in name
    )
    worker.freeze_reference(tmp_path / "reference")
    assert worker.has_reference
    worker.rl_backward(sample, 0.4, 0.01)
