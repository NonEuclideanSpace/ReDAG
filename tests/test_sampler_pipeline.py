import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agentic_image.runtime.sampler import (
    EVALUATION_SEEDS, SAMPLERS, comparison_registration, comparison_tasks,
    complete_comparison_rows, frozen_json, prepare_severe_review,
    sampler_decision_with_protection, shard_indices,
)


def comparison_rows():
    rows = []
    for index in range(64):
        for seed in EVALUATION_SEEDS:
            for method in SAMPLERS:
                image_id = f"test:{method}:{index}:{seed}"
                rows.append({
                    "method": method, "training_seed": 11, "family_id": f"family:{index}",
                    "task_id": f"task:{index}", "eval_seed": seed,
                    "source_image_id": f"source:{index}", "bundle_id": f"bundle:{index}",
                    "renderer_policy_identity": {"path": "test-policy", "bytes": 100},
                    "status": "scored", "calibrated_evaluation": True,
                    "success": 1.0, "severe_defect": 0.0, "severe_defect_status": "reviewed",
                    "protection": None if index < 5 else 0.9,
                    "preservation_status": "not_applicable" if index < 5 else "measured",
                    "seconds": float(SAMPLERS[method][0]),
                    "image_id": image_id, "image_value_id": image_id + ":value",
                    "severe_defect_review": {"image_id": image_id,
                        "candidate_image_value_id": image_id + ":value",
                        "known": True, "severe_defect": False, "review_type": "ai",
                        "human_reviewed": False, "review_evidence": {"test_only": True},
                        "reviewed_image_paths": ["synthetic-unit-test-fixture"]},
                })
    return rows


def test_sampler_contract_keeps_two_seeds_and_all_four_methods():
    assert EVALUATION_SEEDS == (101, 211)
    assert SAMPLERS == {"ode50": (50, 0.0), "sde50": (50, 0.3), "sde28": (28, 0.3), "sde20": (20, 0.3)}
    assert len(comparison_rows()) == 512
    assert list(shard_indices(56, 8)) == list(range(56, 64))
    with pytest.raises(ValueError):
        shard_indices(1, 8)


def test_immutable_registration_has_one_writer_and_rejects_mutation(tmp_path):
    path = tmp_path / "shared" / "registration.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: frozen_json(path, {"fixed": 64}), range(16)))
    assert json.loads(path.read_text()) == {"fixed": 64}
    with pytest.raises(ValueError, match="changed"):
        frozen_json(path, {"fixed": 63})


def test_protection_is_only_skipped_when_explicitly_inapplicable():
    result = sampler_decision_with_protection(comparison_rows(), "sde50", "ode50", 100)
    assert result["accepted"]
    assert result["protection_not_applicable_observations"] == 10
    assert result["protection_applicable_observations"] == 118
    assert result["protection"]["families"] == 59
    assert result["success"]["families"] == 64
    assert result["severe_defects"]["observations"] == 128


def test_unknown_defect_prevents_a_decision():
    rows = comparison_rows()
    rows[0]["severe_defect"] = None
    rows[0]["severe_defect_status"] = "unknown"
    with pytest.raises(ValueError, match="Unknown image quality"):
        sampler_decision_with_protection(rows, "sde50", "ode50", 100)


def test_missing_local_protection_cannot_be_replaced_with_na():
    rows = comparison_rows()
    rows[0]["preservation_status"] = "not_evaluated"
    with pytest.raises(ValueError, match="Missing or inconsistent"):
        sampler_decision_with_protection(rows, "sde50", "ode50", 100)


def test_changed_condition_or_policy_rejects_paired_comparison():
    for field in ("bundle_id", "source_image_id", "renderer_policy_identity"):
        rows = comparison_rows()
        rows[0][field] = "changed"
        with pytest.raises(ValueError, match="same frozen"):
            sampler_decision_with_protection(rows, "sde50", "ode50", 100)


def test_short_chain_requires_real_timing_improvement():
    rows = comparison_rows()
    assert sampler_decision_with_protection(rows, "sde20", "sde50", 100)["accepted"]
    for row in rows:
        if row["method"] == "sde20":
            row["seconds"] = 51.0
    assert not sampler_decision_with_protection(rows, "sde20", "sde50", 100)["accepted"]


def test_missing_family_or_render_cannot_reduce_the_denominator():
    rows = [row for row in comparison_rows() if row["family_id"] != "family:63"]
    with pytest.raises(ValueError, match="all 64"):
        sampler_decision_with_protection(rows, "sde50", "ode50", 100)


def test_partial_blind_packet_leaves_no_false_review_files(tmp_path):
    with pytest.raises(ValueError, match="all 64"):
        prepare_severe_review([], tmp_path)
    assert not (tmp_path / "blinded_quality_review").exists()


def test_cached_freeze_completes_without_model_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import agentic_image.runtime.sampler as sampler

    identity = {"path": "fixed-actor", "bytes": 100, "mtime_ns": 1}
    monkeypatch.setattr(sampler, "policy_identity", lambda *args, **kwargs: identity)
    tasks = [SimpleNamespace(model_dump=lambda mode, i=i: {"index": i}) for i in range(64)]
    (tmp_path / "shards").mkdir()
    for index in range(8):
        folder = tmp_path / "plans" / f"{index:03d}"
        folder.mkdir(parents=True)
        bundle = folder / "condition.pt"
        bundle.write_bytes(b"test fixture only")
        (folder / "frozen.json").write_text(json.dumps({"task": {"index": index},
            "actual_sampled_plan": True, "actor_policy_identity": identity,
            "bundle_path": str(bundle)}))
    result = sampler.cached_freeze_report({"models": {"actor_policy": "fixed"}}, tasks, tmp_path)
    assert result["completed"] == 8
    assert result["all_frozen_reused"] is True
    assert result["models_loaded"] is False
    assert json.loads((tmp_path / "shards" / "freeze_000_007.json").read_text()) == result


def test_missing_reviews_overwrite_stale_acceptance(tmp_path):
    from agentic_image.runtime.sampler import select_comparison_sampler

    (tmp_path / "sampler_selection.json").write_text('{"accepted":true}')
    result = select_comparison_sampler({"models": {"renderer_policy": "fixed"}}, [], tmp_path, None, 100)
    assert result["status"] == "incomplete"
    assert json.loads((tmp_path / "sampler_selection.json").read_text())["accepted"] is False


def test_unknown_atoms_keep_the_full_family_denominator():
    rows = comparison_rows()
    for row in rows:
        if row["family_id"] == "family:0":
            row.update(calibrated_evaluation=False, evaluation_complete=True,
                       known_scores_calibrated=True, unknown_atom_ids=["unconfirmed-count"],
                       unconfirmed_routes=["count"], success=0.0)
    result = sampler_decision_with_protection(rows, "sde50", "ode50", 100)
    assert result["success"]["families"] == 64
    assert result["success"]["observations"] == 128
    assert result["success"]["means"] == pytest.approx([63 / 64, 63 / 64])


def test_complete_base_pair_can_select_before_short_chains_exist(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import agentic_image.runtime.sampler as sampler

    rows = [row for row in comparison_rows() if row["method"] in {"ode50", "sde50"}]
    tasks = [SimpleNamespace(task_id=f"task:{index}", family_id=f"family:{index}") for index in range(64)]
    identity = rows[0]["renderer_policy_identity"]
    monkeypatch.setattr(sampler, "policy_identity", lambda *args, **kwargs: identity)
    (tmp_path / "renderer_policy_identity.json").write_text(json.dumps(identity))
    reviews = []
    for row in rows:
        index = int(row["task_id"].split(":")[-1])
        image_id = f"test:{row['method']}:{index}:{row['eval_seed']}"
        row.update(image_id=image_id, image_value_id=image_id + ":value")
        image_path = tmp_path / f"{row['method']}_{index}_{row['eval_seed']}.png"
        image_path.write_bytes(b"synthetic-unit-test-image")
        row["image"] = str(image_path)
        for stage in ("images", "scores"):
            folder = tmp_path / stage / row["method"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{index:03d}_{row['eval_seed']}.json").write_text(json.dumps(row))
        reviews.append({"image_id": image_id, "candidate_image_value_id": image_id + ":value",
                        "known": True, "severe_defect": False, "review_type": "ai",
                        "human_reviewed": False, "review_evidence": {"test_only": True},
                        "reviewed_image_paths": ["synthetic-unit-test-fixture"]})
    path = tmp_path / "reviews.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in reviews))
    result = sampler.select_comparison_sampler({"models": {"renderer_policy": "fixed"}}, tasks, tmp_path, path, 100)
    assert result["accepted"] is True
    assert result["candidate"] == "sde50"
    assert result["base_pair_records"] == 256
    assert result["full_experiment_complete"] is False
    assert result["pending_shorter_samplers"] == ["sde28", "sde20"]
    assert (tmp_path / "sampler_selection.sde50.json").is_file()
