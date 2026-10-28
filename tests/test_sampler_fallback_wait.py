import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("fallback_cli", Path(__file__).resolve().parents[1] / "scripts/run_sampler_fallback.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
CANDIDATE = "sde50_eta015"


def write_images(root, tasks, limit=16):
    folder = root / "images" / CANDIDATE
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(limit):
        index, seed = i // 2, (101, 211)[i % 2]
        path = folder / f"{index:03d}_{seed}.png"
        path.write_bytes(b"synthetic-test-only")
        path.with_suffix(".json").write_text(json.dumps({"method": CANDIDATE, "task_id": tasks[index].task_id,
            "family_id": tasks[index].family_id, "eval_seed": seed, "steps": 50, "eta": 0.15,
            "status": "rendered", "image": str(path.resolve())}))


def test_wait_requires_all_16_actual_candidate_pixels_and_metadata(tmp_path, monkeypatch):
    tasks = [SimpleNamespace(task_id=f"task:{i}", family_id=f"family:{i}") for i in range(8)]
    write_images(tmp_path, tasks, 15)
    now = [0.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: now[0])
    def advance(seconds):
        prior = json.loads((tmp_path / f"shards/score_wait_{CANDIDATE}_000_007.json").read_text())
        assert prior["available"] == 15 and prior["required"] == 16
        now[0] += seconds
        write_images(tmp_path, tasks)
    monkeypatch.setattr(cli.time, "sleep", advance)
    report = cli.wait_for_render_shard(tasks, tmp_path, CANDIDATE, 0)
    assert report["status"] == "ready" and report["elapsed_seconds"] == 5
    row_path = tmp_path / f"images/{CANDIDATE}/000_101.json"
    row = json.loads(row_path.read_text())
    row["eta"] = 0.05
    row_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="registered candidate"):
        cli.wait_for_render_shard(tasks, tmp_path, CANDIDATE, 0)


def test_wait_times_out_without_gpu_work(tmp_path, monkeypatch):
    tasks = [SimpleNamespace(task_id=f"task:{i}", family_id=f"family:{i}") for i in range(8)]
    now = [0.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    with pytest.raises(TimeoutError, match="0/16"):
        cli.wait_for_render_shard(tasks, tmp_path, CANDIDATE, 0, timeout_seconds=10)
    report = json.loads((tmp_path / f"shards/score_wait_{CANDIDATE}_000_007.json").read_text())
    assert report["status"] == "timed_out" and report["elapsed_seconds"] == 10


@pytest.mark.parametrize("wait", [False, True])
def test_one_persistent_judge_across_four_shards(tmp_path, monkeypatch, wait):
    events = []
    config = {"devices": {"judge": "cuda:0"}, "models": {}, "judge": {"calibration": "fixed"}}
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "comparison_tasks", lambda *a: [])
    monkeypatch.setattr(cli, "fallback_registration", lambda *a: (tmp_path / "shards").mkdir(exist_ok=True))
    monkeypatch.setattr(cli, "wait_for_render_shard", lambda tasks, output, candidate, start:
        events.append(("wait", start)))
    class FakeComponents:
        def __init__(self, config):
            events.append(("construct",))
        def load_judge(self, private):
            assert private is True
            events.append(("judge",))
        def record_memory(self, label):
            pass
    monkeypatch.setattr(cli, "Components", FakeComponents)
    def score(components, tasks, output, start, count, methods, registry):
        assert methods == (CANDIDATE,)
        events.append(("score", start))
        return {"status": "completed", "completed": 16}
    monkeypatch.setattr(cli, "score_comparison_shard", score)
    argv = ["run_sampler_fallback.py", "score", "--output", str(tmp_path), "--candidate", CANDIDATE]
    if wait:
        argv.append("--wait-for-renders")
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert events.count(("judge",)) == events.count(("construct",)) == 1
    expected = [("construct",), ("judge",)]
    for start in (0, 8, 16, 24):
        if wait:
            if start == 0:
                expected.insert(0, ("wait", start))
            else:
                expected.append(("wait", start))
        expected.append(("score", start))
    assert events == expected
    result = json.loads((tmp_path / f"shards/score_{CANDIDATE}_000_031.json").read_text())
    assert result["images"] == result["required"] == 64 and result["status"] == "completed"
