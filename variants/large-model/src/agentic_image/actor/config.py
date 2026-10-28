from dataclasses import dataclass
from pathlib import Path


ACTOR_MODEL = "Qwen/Qwen3.8-27B"
ACTOR_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ACTOR_TARGETS = r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"


@dataclass(frozen=True)
class ActorConfig:
    quantized_model: str = "models/qwen3.8-27b-nf4"
    model_id: str = ACTOR_MODEL
    revision: str = ACTOR_REVISION
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
    expected_targets: int = 496
    expected_adapter_parameters: int = 116727808
    expected_base_parameters: int = 27356728560
    expected_quantized_modules: int = 400
    expected_quantized_parameters: int = 24326963200

    def __post_init__(self):
        if min(self.max_input_tokens, self.max_action_tokens, self.head_chunk_size) < 1:
            raise ValueError("Actor token budgets must be positive")
        if self.temperature <= 0:
            raise ValueError("The categorical policy requires a positive temperature")
        if self.max_images > 4 or self.max_images < 0:
            raise ValueError("The actor supports at most four images per action")


def adapter_directory(path: str | Path, name: str = "policy") -> Path:
    path = Path(path)
    if (path / "adapter_config.json").is_file():
        return path
    if (path / name / "adapter_config.json").is_file():
        return path / name
    raise FileNotFoundError(f"No {name} adapter in {path}")
