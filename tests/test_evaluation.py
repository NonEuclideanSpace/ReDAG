import json

import pytest
from PIL import Image

from agentic_image.evaluation.benchmarks import prepare_public_prompts, run_geneval2, run_magicbrush
from agentic_image.evaluation.statistics import paired_bootstrap


def test_geneval2_generation_cannot_receive_private_questions(tmp_path):
    private = tmp_path / "private.jsonl"
    private.write_text(
        json.dumps({"prompt": "A red cup", "vqa_list": [["Is the cup red?", "yes"]]}) + "\n"
    )
    public = tmp_path / "public.jsonl"
    prepare_public_prompts(private, public)
    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(source)
    requests = []

    def generate(task, seed):
        requests.append(task)
        assert not hasattr(task, "vqa_list")
        return source

    with pytest.raises(ValueError, match="private VQA"):
        run_geneval2(private, tmp_path / "bad", generate)
    run_geneval2(public, tmp_path / "good", generate, seeds=(101, 211))
    assert len(requests) == 2
    first = json.loads((tmp_path / "good" / "seed_101" / "image_paths.json").read_text())
    second = json.loads((tmp_path / "good" / "seed_211" / "image_paths.json").read_text())
    assert first["A red cup"] != second["A red cup"]


def test_magicbrush_iterative_edits_use_the_previous_delivered_image(tmp_path):
    source1, source2 = tmp_path / "source1.png", tmp_path / "source2.png"
    Image.new("RGB", (8, 8), "red").save(source1)
    Image.new("RGB", (8, 8), "blue").save(source2)
    turns = [
        {
            "session_id": "session-a",
            "turn_index": 1,
            "source_image": str(source1),
            "instruction": "Make the cup blue",
            "target_image": "hidden",
        },
        {
            "session_id": "session-a",
            "turn_index": 2,
            "source_image": str(source2),
            "instruction": "Move the cup left",
            "target_image": "hidden",
        },
    ]
    requests = []

    def generate(task, seed):
        requests.append(task)
        assert not hasattr(task, "target_image")
        return source1

    run_magicbrush(turns, tmp_path / "results", generate)
    assert len(requests) == 3
    assert requests[1].source_image == str(source2)
    assert requests[2].source_image.endswith("seed_101/session-a/edit_1.png")
    files = {path.name for path in (tmp_path / "results" / "seed_101" / "session-a").glob("*.png")}
    assert files == {"edit_1.png", "inde_2.png", "iter_2.png"}


def test_paired_comparison_preserves_category_macro_weights():
    rows = []
    for training_seed in (11, 23):
        for index in range(4):
            for method in ("B5", "B2"):
                rows.append(
                    {
                        "method": method,
                        "training_seed": training_seed,
                        "task_id": str(index),
                        "value": float(method == "B5" and index == 0),
                        "task_weight": 1.0 if index == 0 else 1 / 3,
                    }
                )
    result = paired_bootstrap(rows, "B5", "B2", iterations=100)
    assert result["difference"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same complete"):
        paired_bootstrap(rows[:-1], "B5", "B2", iterations=100)


def test_fixed_base_flow_keeps_public_denominator_and_earliest_tie(tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from agentic_image.core.types import (
        CheckSpec,
        Observation,
        PublicCheckAtom,
        PublicTask,
        RenderResult,
        UserRequest,
    )
    from agentic_image.evaluation.benchmarks import best_of_public
    from agentic_image.runtime.benchmarks import run_fixed_flow

    task = PublicTask(
        task_id="edit",
        family_id="family",
        task_mode="edit",
        user_request=UserRequest(text="Make the cup red", source_image_id="source"),
        check_spec=CheckSpec(
            extractor_version="test",
            atoms=(
                PublicCheckAtom(check_id="a", predicate="A red cup"),
                PublicCheckAtom(check_id="b", predicate="Original background"),
            ),
        ),
    )
    state = {"base": False, "renders": 0}

    @contextmanager
    def base_policy():
        state["base"] = True
        try:
            yield
        finally:
            state["base"] = False

    def compile_plan(plan, source):
        assert plan.global_text == task.user_request.text
        assert source == "source"
        assert not plan.regions
        return SimpleNamespace(bundle_id="bundle", cfg_scale=4)

    def render(bundle, seed, job_id):
        assert state["base"]
        index = state["renders"]
        state["renders"] += 1
        return RenderResult(image_id=str(index), bundle_id="bundle", trajectory_id=f"trace-{index}")

    def check(public, image_id):
        return [
            Observation(
                observation_id=f"a-{image_id}",
                image_id=image_id,
                check_id="a",
                known=True,
                passed=True,
                score=0.9,
            ),
            Observation(
                observation_id=f"b-{image_id}",
                image_id=image_id,
                check_id="b",
                known=image_id != "0",
                passed=False,
                score=1.0 if image_id == "0" else 0.2,
            ),
        ]

    components = SimpleNamespace(
        compiler=SimpleNamespace(validate_plan=lambda *args: None, compile=compile_plan),
        renderer=SimpleNamespace(base_policy=base_policy, config=SimpleNamespace(steps=50)),
        render_bridge=SimpleNamespace(render=render),
        observer=SimpleNamespace(check=check),
    )
    delivered, costs = run_fixed_flow(components, task, 101, tmp_path, "B0", 3)
    assert delivered == "1"
    assert not state["base"]
    assert costs["renders"] == 3
    assert costs["dit_forwards"] == 300
    records = json.loads((tmp_path / "fixed_flow.json").read_text())
    records["candidates"][0]["private_score"] = 100
    assert best_of_public(records["candidates"])["image_id"] == "1"
    records["candidates"][0]["public_observations"].pop()
    with pytest.raises(ValueError, match="fixed CheckSpec denominator"):
        best_of_public(records["candidates"])
