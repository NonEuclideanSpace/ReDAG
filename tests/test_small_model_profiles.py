import json
from pathlib import Path
import re

import pytest
import torch
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from agentic_image.actor.config import (
    ACTOR_TARGETS,
    ActorConfig,
    validate_actor_architecture,
)
from agentic_image.checks.judge import FrozenJudge, JUDGE_PROFILES, JudgeConfig


def test_pinned_actor_architecture_matches_registered_counts():
    root = Path(__file__).parents[1] / "deployment" / "model-profiles"
    raw = json.loads((root / "actor-config.json").read_text())
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(raw))
    config = ActorConfig(
        model_id="Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    validate_actor_architecture(model, config)
    targets = {
        name: module
        for name, module in model.named_modules()
        if re.fullmatch(ACTOR_TARGETS, name)
    }
    quantized = {
        name: module
        for name, module in targets.items()
        if not name.endswith(("in_proj_a", "in_proj_b"))
    }
    assert len(targets) == config.expected_targets
    assert len(quantized) == config.expected_quantized_modules
    assert sum(p.numel() for p in model.parameters()) == config.expected_base_parameters
    assert sum(m.in_features * m.out_features for m in quantized.values()) == config.expected_quantized_parameters
    assert sum(16 * (m.in_features + m.out_features) for m in targets.values()) == config.expected_adapter_parameters
    model.config.text_config.hidden_size += 1
    with pytest.raises(ValueError, match="architecture"):
        validate_actor_architecture(model, config)


def test_pinned_judge_architecture_matches_registered_counts():
    root = Path(__file__).parents[1] / "deployment" / "model-profiles"
    raw = json.loads((root / "judge-config.json").read_text())
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(raw))
    profile = JUDGE_PROFILES["Qwen/Qwen3-VL-8B-Instruct"]
    pattern = r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    targets = [module for name, module in model.named_modules() if re.fullmatch(pattern, name)]
    assert len(targets) == profile["quantized_modules"]
    assert sum(m.in_features * m.out_features for m in targets) == profile["quantized_parameters"]
    assert sum(p.numel() for p in model.parameters()) == profile["base_parameters"]


def test_profiles_reject_unregistered_revisions_and_precision():
    with pytest.raises(ValueError, match="registered"):
        ActorConfig(model_id="Qwen/Qwen3.5-4B")
    with pytest.raises(ValueError, match="registered"):
        JudgeConfig(model="Qwen/Qwen3-VL-8B-Instruct")
    with pytest.raises(ValueError, match="precision"):
        JudgeConfig(precision="float16")
    assert ActorConfig().expected_base_parameters == 27356728560
    assert JUDGE_PROFILES[JudgeConfig().model]["base_parameters"] == 33357390064


def test_small_judge_rejects_unidentified_or_wrong_model_calibration(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"semantic": {"threshold": 0.8, "confirmed": False}}))
    config = JudgeConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        calibration_path=str(path),
    )
    with pytest.raises(ValueError, match="identity"):
        FrozenJudge(torch.nn.Module(), None, config)
    identity = Path(str(path) + ".model.json")
    identity.write_text(json.dumps(JudgeConfig().calibration_identity))
    with pytest.raises(ValueError, match="different checkpoint"):
        FrozenJudge(torch.nn.Module(), None, config)
    identity.write_text(json.dumps(config.calibration_identity))
    judge = FrozenJudge(torch.nn.Module(), None, config)
    assert not judge.thresholds["semantic"]["confirmed"]
