from dataclasses import replace

import pytest
import torch

from agentic_image.render.compiler import ConditionCompiler
from agentic_image.render.encoders import FrozenEncoders
from agentic_image.render.model import KleinRenderer
from agentic_image.render.types import (
    ConditionRequest,
    EncoderConfig,
    RendererConfig,
    klein_profile,
)


@pytest.mark.parametrize("name", ["klein-base-9b", "klein-base-4b"])
def test_registered_renderer_structure_and_adapter_counts(name):
    from diffusers import Flux2Transformer2DModel

    profile = klein_profile(name)
    with torch.device("meta"):
        transformer = Flux2Transformer2DModel(**profile.transformer_config)
        renderer = KleinRenderer(
            transformer,
            RendererConfig(profile=name, device="meta", attention_backend="query_chunk"),
        )
    assert sum(p.numel() for p in renderer.policy_parameters()) == profile.adapter_parameters_rank16
    assert sum(p.numel() for p in transformer.parameters()) == (
        profile.base_parameters + profile.adapter_parameters_rank16
    )
    wrong = "klein-base-9b" if name == "klein-base-4b" else "klein-base-4b"
    with torch.device("meta"):
        transformer = Flux2Transformer2DModel(**profile.transformer_config)
        with pytest.raises(ValueError, match="architecture"):
            KleinRenderer(transformer, RendererConfig(profile=wrong, device="meta"))


def test_small_profile_rejects_distilled_source_and_wrong_hidden_layers():
    profile = klein_profile("klein-base-4b")
    with pytest.raises(ValueError, match="undistilled"):
        profile.validate_source("black-forest-labs/FLUX.2-klein-4B", profile.revision)
    with pytest.raises(ValueError, match="hidden layers"):
        EncoderConfig(profile="klein-base-4b", hidden_layers=(8, 16, 24))


class Tokenizer:
    chat_template = "test-user-template"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, return_tensors=None, max_length=None, **kwargs):
        from transformers import BatchEncoding

        tokens = [1, 2, 3, 4]
        if return_tensors is None:
            return {"input_ids": tokens}
        size = max_length or len(tokens)
        return BatchEncoding(
            {
                "input_ids": torch.tensor([tokens + [0] * (size - len(tokens))]),
                "attention_mask": torch.tensor([[1] * len(tokens) + [0] * (size - len(tokens))]),
            }
        )


def test_encoder_matches_native_pipeline_and_cache_tracks_hidden_selection(tmp_path):
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
    ).to(torch.bfloat16)
    vae = torch.nn.Module()
    vae.bn = torch.nn.BatchNorm2d(128)
    config = EncoderConfig(
        profile="klein-base-4b",
        device="cpu",
        text_block_tokens=8,
        text_hidden_size=24,
        hidden_layers=(1, 2, 3),
        strict_architecture=False,
        cache_directory=str(tmp_path),
    )
    encoder = FrozenEncoders(model, Tokenizer(), vae, config)
    actual = encoder.encode_text(["A red cup"])
    expected = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        model,
        encoder.tokenizer,
        "A red cup",
        device=torch.device("cpu"),
        max_sequence_length=8,
        hidden_states_layers=(1, 2, 3),
    )
    assert torch.equal(actual, expected)
    changed = FrozenEncoders(model, Tokenizer(), vae, replace(config, hidden_layers=(0, 1, 2)))
    assert encoder._key("text", b"A red cup") != changed._key("text", b"A red cup")
    assert not torch.equal(actual, changed.encode_text(["A red cup"]))


def test_small_compiler_records_encoder_provenance_and_rejects_mismatches():
    class Encoder:
        config = EncoderConfig(profile="klein-base-4b")
        version_manifest = {
            "encoder_revision": "fixed-small-snapshot",
            "encoder_hidden_layers": "9,18,27",
        }

        def encode_text(self, texts):
            return torch.zeros(len(texts), 512, 7680, dtype=torch.bfloat16)

    compiler = ConditionCompiler(Encoder())
    bundle = compiler.compile(ConditionRequest("A red cup", canvas_hw=(16, 16)))
    assert bundle.positive_embeds.shape == (1, 512, 7680)
    assert bundle.model_versions == Encoder.version_manifest
    with pytest.raises(ValueError, match="provenance"):
        compiler.compile(
            ConditionRequest(
                "A red cup", canvas_hw=(16, 16), model_versions={"encoder_revision": "old"}
            )
        )
