import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_image.config import load_config
from agentic_image.runtime import sampler


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sampler_confirmation_cli", ROOT / "scripts/run_sampler_confirmation.py")
confirmation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(confirmation)


def fixture_registration(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/sampler-confirmation-200.yaml")
    config["judge"]["calibration"] = str(ROOT / "data/calibration/thresholds.json")
    tasks = sampler.comparison_tasks(ROOT / "data/evaluation/sampler_confirm64/public_tasks.jsonl",
        ROOT / "data/evaluation/sampler_confirm64/reserved.frozen.jsonl",
        task_prefix="sampler-confirm64:", expected_split="official-dev")
    old = sampler.comparison_tasks(ROOT / "data/evaluation/development64/public_tasks.jsonl",
                                   ROOT / "data/evaluation/development64/reserved.frozen.jsonl")
    original, output = tmp_path / "original", tmp_path / "confirmation"
    original.mkdir()
    identity = {"path": "synthetic-test-policy", "bytes": 10, "mtime_ns": 1}
    monkeypatch.setattr(confirmation, "policy_identity", lambda *a, **k: identity)
    prior = {"tasks": [task.model_dump(mode="json") for task in old],
        "models": config["models"], "samplers": {"ode50": [50, 0], "sde50_eta005": [50, 0.05]},
        "training_seed": config["seed"], "canvas": config["canvas"], "eval_seeds": [101, 211],
        "actor_sampling": {key: config["actor"][key] for key in ("temperature", "max_new_tokens", "max_input_tokens")}}
    (original / "registration.json").write_text(json.dumps(prior))
    for role in ("actor", "renderer"):
        (original / f"{role}_policy_identity.json").write_text(json.dumps(identity))
    calibration = {"identity": confirmation.judge_config(config).calibration_identity,
                   "routes": json.loads(Path(config["judge"]["calibration"]).read_text())}
    (original / "calibration.json").write_text(json.dumps(calibration))
    rule = tmp_path / "selection_rule.json"
    rule.write_bytes((ROOT / "data/evaluation/sampler_confirm64/selection_rule.json").read_bytes())
    return config, tasks, output, original, rule


def test_real_confirmation_manifest_loads_without_relabeling_heldout_split():
    args = (ROOT / "data/evaluation/sampler_confirm64/public_tasks.jsonl",
            ROOT / "data/evaluation/sampler_confirm64/reserved.frozen.jsonl")
    with pytest.raises(ValueError, match="registration mismatch"):
        sampler.comparison_tasks(*args, task_prefix="sampler-confirm64:")
    tasks = sampler.comparison_tasks(*args, task_prefix="sampler-confirm64:", expected_split="official-dev")
    assert len(tasks) == len({task.family_id for task in tasks}) == 64
    assert {task.split_id for task in tasks} == {"official-dev"}


def test_confirmation_accepts_exact_existing_calibration_schema(tmp_path, monkeypatch):
    args = fixture_registration(tmp_path, monkeypatch)
    result = confirmation.register(*args)
    assert result["expected_renders"] == 256
    assert len(result["tasks"]) == 64
    assert set(result["confirmation_families"]).isdisjoint(result["original_families"])
    assert (args[2] / "calibration.json").read_text() == (args[3] / "calibration.json").read_text() or json.loads((args[2] / "calibration.json").read_text()) == json.loads((args[3] / "calibration.json").read_text())
    confirmation.register(*args)


def test_confirmation_rejects_admitted_family_or_changed_controller(tmp_path, monkeypatch):
    args = fixture_registration(tmp_path, monkeypatch)
    rule = json.loads(args[4].read_text())
    rule["excluded_family_ids"].append(args[1][0].family_id)
    args[4].write_text(json.dumps(rule))
    with pytest.raises(ValueError, match="family exclusions"):
        confirmation.register(*args)
    rule["excluded_family_ids"].pop()
    args[4].write_text(json.dumps(rule))
    args[0]["actor"]["temperature"] = 0.5
    with pytest.raises(ValueError, match="controller sampling"):
        confirmation.register(*args)


def test_confirmation_rejects_changed_thresholds(tmp_path, monkeypatch):
    args = fixture_registration(tmp_path, monkeypatch)
    path = args[3] / "calibration.json"
    record = json.loads(path.read_text())
    record["routes"]["semantic"]["threshold"] = 0.123
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="recalibrate"):
        confirmation.register(*args)


def paired_rows(prefix, count=64):
    rows = []
    for index in range(count):
        for seed in (101, 211):
            for method in confirmation.SAMPLERS:
                image_id = f"synthetic:{prefix}:{index}:{seed}:{method}"
                review = {"image_id": image_id, "candidate_image_value_id": image_id + ":value",
                    "known": True, "severe_defect": False, "review_type": "ai", "human_reviewed": False,
                    "review_evidence": {"synthetic_test_only": True}, "reviewed_image_paths": ["synthetic-unit-test"]}
                rows.append({"method": method, "training_seed": 11, "family_id": f"{prefix}:family:{index}",
                    "task_id": f"{prefix}:task:{index}", "eval_seed": seed, "source_image_id": f"{prefix}:source:{index}",
                    "bundle_id": f"{prefix}:bundle:{index}", "renderer_policy_identity": {"test": "fixed"},
                    "status": "scored", "calibrated_evaluation": True, "success": 1.0,
                    "protection": 0.8, "preservation_status": "measured", "severe_defect": 0.0,
                    "severe_defect_status": "reviewed", "severe_defect_review": review,
                    "image_id": image_id, "image_value_id": image_id + ":value", "seconds": 20.0})
    return rows


def test_pooled_128_requires_and_reports_all_512_records(tmp_path, monkeypatch):
    args = fixture_registration(tmp_path, monkeypatch)
    confirmation.register(*args)
    old, new = paired_rows("old"), paired_rows("new")
    monkeypatch.setattr(confirmation, "complete_comparison_rows", lambda tasks, output, *a, **k:
        (new if Path(output) == args[2] else old, {"status": "ready_for_decision"}))
    result = confirmation.select(*args[:4], [], 100)
    assert result["paired_records"] == 512
    assert result["decision"]["success"]["families"] == 128
    assert result["decision"]["success"]["observations"] == 256
    assert result["independent_confirmation_decision"]["success"]["families"] == 64
    assert result["original_decision"]["success"]["families"] == 64
    with pytest.raises(ValueError, match="all 128"):
        sampler.sampler_decision_with_protection((old + new)[:-1], confirmation.CANDIDATE, "ode50", 100, 128)
    changed = copy.deepcopy(old + new)
    for row in changed:
        if row["eval_seed"] == 211:
            row["eval_seed"] = 212
    with pytest.raises(ValueError, match="registered evaluation seeds"):
        sampler.sampler_decision_with_protection(changed, confirmation.CANDIDATE, "ode50", 100, 128)


def test_confirmation_blind_registration_has_256_unlabeled_slots(tmp_path):
    tasks = [SimpleNamespace(user_request=SimpleNamespace(text="synthetic", source_image_id="synthetic")) for _ in range(64)]
    report = sampler.prepare_severe_review(tasks, tmp_path, confirmation.SAMPLERS, "confirmation-quality")
    assert report["required"] == 256 and report["known_labels"] == 0
    assert len(json.loads((tmp_path / "quality_review_key.private.json").read_text())) == 256
    assert json.loads((tmp_path / "blinded_quality_review/records.json").read_text()) == []


def test_partial_freeze_cache_rejects_unsampled_saved_plan(tmp_path, monkeypatch):
    identity = {"path": "synthetic", "bytes": 1, "mtime_ns": 1}
    monkeypatch.setattr(sampler, "policy_identity", lambda *a, **k: identity)
    components = SimpleNamespace(config={"models": {"actor_policy": "synthetic"}},
        actor=SimpleNamespace(processor=SimpleNamespace(tokenizer=None)))
    tasks = [SimpleNamespace(model_dump=lambda mode, i=i: {"index": i}) for i in range(64)]
    folder = tmp_path / "plans/000"
    folder.mkdir(parents=True)
    bundle = folder / "condition.pt"
    bundle.write_bytes(b"synthetic-unit-test-only")
    (folder / "frozen.json").write_text(json.dumps({"task": {"index": 0}, "bundle_path": str(bundle),
        "actual_sampled_plan": False, "actor_policy_identity": identity}))
    with pytest.raises(ValueError, match="genuine sampled"):
        sampler.freeze_comparison_plans(components, tasks, tmp_path)


def write_wait_images(tmp_path, tasks, limit=32):
    count = 0
    for index in range(8):
        for seed in confirmation.EVALUATION_SEEDS:
            for method in confirmation.SAMPLERS:
                if count == limit:
                    return
                path = tmp_path / "images" / method / f"{index:03d}_{seed}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic-unit-test-pixels")
                path.with_suffix(".json").write_text(json.dumps({"task_id": tasks[index].task_id,
                    "family_id": tasks[index].family_id, "method": method, "eval_seed": seed,
                    "status": "rendered", "image": str(path.resolve())}))
                count += 1


def test_wait_records_real_metadata_and_pixel_availability_then_resumes(tmp_path, monkeypatch):
    tasks = [SimpleNamespace(task_id=f"task:{i}", family_id=f"family:{i}") for i in range(8)]
    write_wait_images(tmp_path, tasks, 31)
    clock = [0.0]
    monkeypatch.setattr(confirmation.time, "monotonic", lambda: clock[0])
    def advance(seconds):
        report = json.loads((tmp_path / "shards/score_wait_000_007.json").read_text())
        assert report["available"] == 31 and report["status"] == "waiting"
        clock[0] += seconds
        write_wait_images(tmp_path, tasks)
    monkeypatch.setattr(confirmation.time, "sleep", advance)
    report = confirmation.wait_for_render_shard(tasks, tmp_path, 0)
    assert report["status"] == "ready" and report["available"] == report["required"] == 32
    assert report["elapsed_seconds"] == 5
    bad = tmp_path / "images/ode50/000_101.json"
    row = json.loads(bad.read_text())
    row["family_id"] = "wrong-family"
    bad.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="frozen task"):
        confirmation.wait_for_render_shard(tasks, tmp_path, 0)


def test_wait_timeout_keeps_missing_images_explicit(tmp_path, monkeypatch):
    tasks = [SimpleNamespace(task_id=f"task:{i}", family_id=f"family:{i}") for i in range(8)]
    write_wait_images(tmp_path, tasks, 1)
    clock = [0.0]
    monkeypatch.setattr(confirmation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(confirmation.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(TimeoutError, match="1/32"):
        confirmation.wait_for_render_shard(tasks, tmp_path, 0, timeout_seconds=10)
    report = json.loads((tmp_path / "shards/score_wait_000_007.json").read_text())
    assert report["status"] == "timed_out" and report["elapsed_seconds"] == 10


@pytest.mark.parametrize("wait", [False, True])
def test_score_reuses_one_judge_and_waits_before_first_load(tmp_path, monkeypatch, wait):
    import sys
    events = []
    (tmp_path / "shards").mkdir()
    config = load_config(ROOT / "configs/sampler-confirmation-200.yaml")
    monkeypatch.setattr(confirmation, "load_config", lambda path: config)
    monkeypatch.setattr(confirmation, "comparison_tasks", lambda *a, **k: [])
    monkeypatch.setattr(confirmation, "register", lambda *a, **k: {})
    monkeypatch.setattr(confirmation, "wait_for_render_shard", lambda tasks, output, start:
        events.append(("wait", start)))
    class FakeComponents:
        def __init__(self, config):
            events.append(("construct",))
        def load_judge(self, private):
            assert private is True
            events.append(("judge",))
        def record_memory(self, label):
            events.append(("memory",))
    monkeypatch.setattr(confirmation, "Components", FakeComponents)
    def score(components, tasks, output, start, count, *args):
        events.append(("score", start))
        return {"status": "completed", "completed": 32}
    monkeypatch.setattr(confirmation, "score_comparison_shard", score)
    argv = ["run_sampler_confirmation.py", "score", "--output", str(tmp_path), "--count", "16"]
    if wait:
        argv.append("--wait-for-renders")
    monkeypatch.setattr(sys, "argv", argv)
    confirmation.main()
    assert events.count(("construct",)) == events.count(("judge",)) == 1
    if wait:
        assert events == [("wait", 0), ("construct",), ("judge",), ("memory",), ("score", 0),
                          ("wait", 8), ("score", 8)]
    else:
        assert events == [("construct",), ("judge",), ("memory",), ("score", 0), ("score", 8)]
    report = json.loads((tmp_path / "shards/score_000_015.json").read_text())
    assert report["status"] == "completed" and report["completed"] == 64
