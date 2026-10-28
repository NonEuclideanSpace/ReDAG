from dataclasses import dataclass, field
from pathlib import Path


ACTOR_MODEL = "Qwen/Qwen3.8-27B"
ACTOR_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ACTOR_NUMERICS_PROFILE = "nf4-bf16-fp32-reduction-gdn-fp32-v1"
ACTOR_TARGETS = r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
ACTOR_PROFILES = {
    ACTOR_MODEL: {
        "revision": ACTOR_REVISION,
        "expected_targets": 496,
        "expected_adapter_parameters": 116727808,
        "expected_base_parameters": 27356728560,
        "expected_quantized_modules": 400,
        "expected_quantized_parameters": 24326963200,
    },
    "Qwen/Qwen3.5-4B": {
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "expected_targets": 248,
        "expected_adapter_parameters": 32464896,
        "expected_base_parameters": 4539265536,
        "expected_quantized_modules": 200,
        "expected_quantized_parameters": 3565158400,
    },
}


@dataclass(frozen=True)
class ActorConfig:
    quantized_model: str = "models/qwen3.8-27b-nf4"
    model_id: str = ACTOR_MODEL
    revision: str = ACTOR_REVISION
    source_path: str | None = None
    policy_adapter: str | None = None
    reference_adapter: str | None = None
    image_cache_directory: str | None = None
    device: str = "cuda:0"
    max_input_tokens: int = 6144
    max_action_tokens: int = 1024
    max_image_pixels: int = 262144
    max_images: int = 4
    temperature: float = 1.0
    head_chunk_size: int = 32
    gradient_checkpointing: bool = True
    numerics_profile: str = field(default=ACTOR_NUMERICS_PROFILE, init=False)
    expected_targets: int = field(init=False)
    expected_adapter_parameters: int = field(init=False)
    expected_base_parameters: int = field(init=False)
    expected_quantized_modules: int = field(init=False)
    expected_quantized_parameters: int = field(init=False)
    eos_token_ids: tuple[int, ...] = field(default=(248046, 248044), init=False)
    action_eos_token_id: int = field(default=248046, init=False)

    def __post_init__(self):
        profile = ACTOR_PROFILES.get(self.model_id)
        if profile is None or self.revision != profile["revision"]:
            raise ValueError("Actor model and revision must match a registered architecture profile")
        for name, value in profile.items():
            if name.startswith("expected_"):
                object.__setattr__(self, name, value)
        if min(self.max_input_tokens, self.max_action_tokens, self.head_chunk_size) < 1:
            raise ValueError("Actor token budgets must be positive")
        if self.temperature <= 0:
            raise ValueError("The categorical policy requires a positive temperature")
        if self.max_images > 4 or self.max_images < 0:
            raise ValueError("The actor supports at most four images per action")


def validate_actor_architecture(model, config):
    if config.model_id != "Qwen/Qwen3.5-4B":
        return
    text = model.config.get_text_config()
    expected = {
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "tie_word_embeddings": True,
    }
    mismatches = {
        key: (getattr(text, key, None), value)
        for key, value in expected.items()
        if getattr(text, key, None) != value
    }
    layers = ["linear_attention"] * 3 + ["full_attention"]
    if mismatches or list(text.layer_types) != layers * 8:
        raise ValueError(f"Unexpected Qwen3.5-4B architecture: {mismatches}")
    vision = model.config.vision_config
    expected_vision = {
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }
    if any(getattr(vision, key, None) != value for key, value in expected_vision.items()):
        raise ValueError("Unexpected Qwen3.5-4B vision architecture")


def validate_actor_tokenizer(tokenizer, config):
    expected = {"<|im_end|>": config.action_eos_token_id, "<|endoftext|>": 248044}
    if any(tokenizer.convert_tokens_to_ids(token) != value for token, value in expected.items()):
        raise ValueError("Actor tokenizer does not match the registered completion tokens")


def adapter_directory(path: str | Path, name: str = "policy") -> Path:
    path = Path(path)
    if (path / "adapter_config.json").is_file():
        return path
    if (path / name / "adapter_config.json").is_file():
        return path / name
    raise FileNotFoundError(f"No {name} adapter in {path}")
