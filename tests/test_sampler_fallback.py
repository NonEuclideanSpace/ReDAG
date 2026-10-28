import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_image.runtime import sampler, sampler_fallback


class FakeTask:
    def __init__(self, index):
        self.task_id = f"test-task:{index}"
        self.family_id = f"test-family:{index}"
        self.user_request = SimpleNamespace(source_image_id=f"test-source:{index}", text=f"test request {index}")

    def model_dump(self, mode=None):
        return {"task_id": self.task_id, "family_id": self.family_id, "source": self.user_request.source_image_id}


def fixture_reference(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    output = tmp_path / "fallback"
    reference.mkdir()
    tasks = [FakeTask(index) for index in range(64)]
    config = {"seed": 11, "canvas": [512, 512], "renderer": {"cfg_scale": 4},
              "models": {"actor_policy": "test-actor", "renderer_policy": "test-renderer"}}
    identity = {"path": "test-trained-policy", "bytes": 99, "mtime_ns": 1}
    monkeypatch.setattr(sampler_fallback, "policy_identity", lambda *args, **kwargs: identity)
    original = {"tasks": [task.model_dump() for task in tasks], "eval_seeds": [101, 211],
                "training_seed": 11, "canvas": [512, 512], "cfg_scale": 4, "models": config["models"]}
    (reference / "registration.json").write_text(json.dumps(original))
    (reference / "calibration.json").write_text('{}')
    for name in ("actor", "renderer"):
        (reference / f"{name}_policy_identity.json").write_text(json.dumps(identity))
    reviews = []
    for index, task in enumerate(tasks):
        folder = reference / "plans" / f"{index:03d}"
        folder.mkdir(parents=True)
        bundle = folder / "condition.pt"
        bundle.write_bytes(b"synthetic-unit-test-only")
        plan = {"task": task.model_dump(), "actual_sampled_plan": True, "actor_policy_identity": identity,
                "bundle_path": str(bundle), "bundle_id": f"test-bundle:{index}", "plan": {"test_only": True}}
        (folder / "frozen.json").write_text(json.dumps(plan))
        for seed in (101, 211):
            stem = f"{index:03d}_{seed}"
            image_dir = reference / "images" / "ode50"
            score_dir = reference / "scores" / "ode50"
            image_dir.mkdir(parents=True, exist_ok=True)
            score_dir.mkdir(parents=True, exist_ok=True)
            image = image_dir / f"{stem}.png"
            image.write_bytes(b"synthetic-unit-test-only")
            row = {"method": "ode50", "steps": 50, "eta": 0, "eval_seed": seed,
                "task_id": task.task_id, "family_id": task.family_id, "bundle_id": plan["bundle_id"],
                "source_image_id": task.user_request.source_image_id, "renderer_policy_identity": identity,
                "image": str(image), "image_id": f"test-image:{stem}", "image_value_id": f"test-value:{stem}"}
            (image_dir / f"{stem}.json").write_text(json.dumps(row))
            (score_dir / f"{stem}.json").write_text(json.dumps(row))
            reviews.append({"image_id": row["image_id"], "candidate_image_value_id": row["image_value_id"],
                "review_type": "ai", "human_reviewed": False, "known": True, "severe_defect": False,
                "review_evidence": {"synthetic_unit_test_only": True}, "reviewed_image_paths": [str(image)]})
    labels = reference / "reviews.jsonl"
    labels.write_text(''.join(json.dumps(row)+'\n' for row in reviews))
    return config, tasks, output, reference, labels


def test_fallback_links_original_ode_and_plans_without_mutation(tmp_path, monkeypatch):
    args = fixture_reference(tmp_path, monkeypatch)
    config, tasks, output, reference, labels = args
    before = (reference / "registration.json").read_bytes()
    registration = sampler_fallback.fallback_registration(*args)
    sampler_fallback.fallback_registration(*args)
    assert (reference / "registration.json").read_bytes() == before
    assert (output / "images" / "ode50").resolve() == reference / "images" / "ode50"
    assert (output / "plans").resolve() == reference / "plans"
    assert registration["reused_ode_renders"] == 128
    assert registration["ode_rerender_allowed"] is False
    assert len(json.loads((output / "reference_reviews.json").read_text())) == 128


def test_fallback_rejects_changed_source_condition(tmp_path, monkeypatch):
    args = fixture_reference(tmp_path, monkeypatch)
    path = args[3] / "images/ode50/000_101.json"
    row = json.loads(path.read_text())
    row["bundle_id"] = "different"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="identity changed"):
        sampler_fallback.fallback_registration(*args)


def test_fallback_rejects_missing_real_reference_review(tmp_path, monkeypatch):
    args = fixture_reference(tmp_path, monkeypatch)
    rows = args[4].read_text().splitlines()
    args[4].write_text('\n'.join(rows[1:])+'\n')
    with pytest.raises(ValueError, match="actual quality review"):
        sampler_fallback.fallback_registration(*args)


def test_fallback_registry_does_not_modify_original_experiment():
    expected = {"ode50": (50, 0), "sde50": (50, 0.3), "sde28": (28, 0.3), "sde20": (20, 0.3)}
    assert sampler.SAMPLERS == expected
    assert sampler.comparison_methods(("sde50_eta005",), sampler_fallback.FALLBACK_SAMPLERS) == ("sde50_eta005",)
    with pytest.raises(ValueError):
        sampler.comparison_methods(("sde50_eta005",))
    assert sampler.comparison_stage_suffix(("ode50", "sde50")) == "_base"
    assert sampler.comparison_stage_suffix(("sde50_eta005",), sampler_fallback.FALLBACK_SAMPLERS) == "_sde50_eta005"
    assert sampler.SAMPLERS == expected
