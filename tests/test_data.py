import json

import numpy as np
from PIL import Image
import pytest

from agentic_image.checks.reward import AtomScore
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.data.families import HammingIndex, partition_families
from agentic_image.data.io import write_jsonl
from agentic_image.data.repairs import repair_is_positive


def test_family_partition_keeps_official_overlap_out_of_training(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    rng = np.random.default_rng(7)
    image_a = artifacts.put_image(
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    ).artifact_id
    image_b = artifacts.put_image(
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    ).artifact_id
    rows = [
        {
            "sample_id": "train-turn-1",
            "official_split": "train",
            "session_id": "session-a",
            "source_image_id": image_a,
            "target_image_id": image_b,
        },
        {
            "sample_id": "test-turn-1",
            "official_split": "test",
            "session_id": "session-b",
            "source_image_id": image_b,
            "target_image_id": image_a,
        },
    ]
    source = tmp_path / "input.jsonl"
    destination = tmp_path / "partitioned.jsonl"
    write_jsonl(source, rows)
    report = partition_families(source, destination, artifacts, minimums={})
    partitioned = [json.loads(line) for line in destination.read_text().splitlines()]
    assert partitioned[0]["family_id"] == partitioned[1]["family_id"]
    assert partitioned[0]["split_id"] == "quarantine-official-overlap"
    assert partitioned[1]["split_id"] == "official-test"
    assert report["official_collision_families"] == 1


def test_duplicate_index_searches_all_neighbors_at_threshold():
    index = HammingIndex()
    values = [0, 3, 15, 255, 1023, 65535]
    for value in values:
        index.add(value, value)
    assert set(index.query(0, 6)) == {value for value in values if value.bit_count() <= 6}


def test_repair_labels_keep_fixed_denominator_unknown_and_regression():
    before = (AtomScore("a", 0.9, True), AtomScore("b", 0.2, True))
    improved = (AtomScore("a", 0.9, True), AtomScore("b", 0.8, True))
    regressed = (AtomScore("a", 0.7, True), AtomScore("b", 1.0, True))
    unknown = (AtomScore("a", 0.9, True), AtomScore("b", 1.0, False))
    assert repair_is_positive(before, improved)
    assert not repair_is_positive(before, regressed)
    assert not repair_is_positive(before, unknown)
    with pytest.raises(ValueError, match="same complete"):
        repair_is_positive(before, improved[:1])


def test_appearance_candidates_crop_only_the_public_source(tmp_path):
    from agentic_image.data.appearance import appearance_candidates

    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = artifacts.put_image(Image.new("RGB", (40, 20), (255, 0, 0))).artifact_id
    target = artifacts.put_image(Image.new("RGB", (40, 20), (0, 0, 255))).artifact_id
    rows = [
        {
            "sample_id": f"sample:{index}",
            "family_id": "shared-family",
            "split_id": "train",
            "source_dataset": "magicbrush",
            "source_image_id": source,
            "target_image_id": target,
            "appearance_image_ids": [],
        }
        for index in range(10)
    ]
    annotations = [
        {
            "sample_id": row["sample_id"],
            "appearance_reference": {
                "verified": True,
                "provenance": "independent-source-box",
                "source_box": [0.0, 0.0, 0.5, 1.0],
                "entity_name": "cup",
            },
        }
        for row in rows
    ]
    input_path, annotations_path, output_path = (
        tmp_path / name for name in ("input.jsonl", "annotations.jsonl", "output.jsonl")
    )
    write_jsonl(input_path, rows)
    write_jsonl(annotations_path, annotations)
    result = appearance_candidates(input_path, annotations_path, output_path, artifacts)
    output = [json.loads(line) for line in output_path.read_text().splitlines()]
    selected = [row for row in output if row["appearance_image_ids"]]
    assert result["selected_by_source"] == {"magicbrush": 1}
    assert len(selected) == 1
    reference_id = selected[0]["appearance_image_ids"][0]
    crop = artifacts.get_image(reference_id)
    assert crop.size == (20, 20)
    assert np.all(np.asarray(crop) == np.array([255, 0, 0]))
    assert artifacts.get_ref(reference_id).metadata["original_source_id"] == source


def test_registration_requires_distributed_real_correspondence():
    from agentic_image.data import registration_evidence

    rng = np.random.default_rng(17)
    source = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    edited = source.copy()
    edited[32:64, 32:64] = (255, 0, 0)
    assert registration_evidence(Image.fromarray(source), Image.fromarray(edited))["verified"]
    assert not registration_evidence(
        Image.fromarray(source), Image.fromarray(np.roll(source, 3, axis=0))
    )["verified"]
    assert not registration_evidence(Image.new("RGB", (128, 128)), Image.new("RGB", (128, 128)))[
        "verified"
    ]


def test_official_masks_decode_transparent_edits_without_using_image_brightness(tmp_path):
    from agentic_image.data.ingest import decode_edit_mask, store_edit_mask

    array = np.full((8, 12, 4), 255, dtype=np.uint8)
    array[..., :3] = [80, 120, 160]
    array[2:6, 4:8] = [0, 0, 0, 0]
    mask = Image.fromarray(array)
    decoded, encoding = decode_edit_mask(mask)
    assert encoding == "transparent_edit_region"
    assert decoded.sum() == 16
    assert decoded[2:6, 4:8].min() == 1
    artifacts = ArtifactStore(tmp_path / "artifacts")
    identifier = store_edit_mask(mask, artifacts, {"record": "official-edit"})
    assert artifacts.get_array(identifier).shape == (8, 12)
    assert artifacts.get_ref(identifier).metadata["original_size"] == [12, 8]
    white_mask = Image.fromarray((decoded * 255).astype(np.uint8))
    assert np.array_equal(decode_edit_mask(white_mask)[0], decoded)
    with pytest.raises(ValueError, match="opaque color image"):
        decode_edit_mask(mask.convert("RGB"))


def test_import_resumes_saved_records_after_a_network_failure(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from agentic_image.data.ingest import stream_import

    rows = [
        {
            "img_id": str(index),
            "turn_index": 1,
            "instruction": "Make the cup blue",
            "source_img": Image.new("RGB", (16, 16), "red"),
            "target_img": Image.new("RGB", (16, 16), "blue"),
            "mask_img": None,
        }
        for index in range(2)
    ]

    def interrupted():
        yield rows[0]
        raise OSError("Download interrupted")

    loader = SimpleNamespace(load_dataset=lambda *args, **kwargs: interrupted())
    monkeypatch.setitem(sys.modules, "datasets", loader)
    destination = tmp_path / "import.jsonl"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(OSError, match="Download interrupted"):
        stream_import("magicbrush", destination, artifacts, 2)
    assert not destination.exists()
    assert len(destination.with_suffix(".partial.jsonl").read_text().splitlines()) == 1
    loader.load_dataset = lambda *args, **kwargs: iter(rows)
    report = stream_import("magicbrush", destination, artifacts, 2, resume=True)
    imported = [json.loads(line) for line in destination.read_text().splitlines()]
    assert report["accepted"] == 2 and report["resumed"] == 1
    assert len({row["sample_id"] for row in imported}) == 2
    assert not destination.with_suffix(".partial.jsonl").exists()


def test_resolution_change_preserves_registered_masks_and_private_targets(tmp_path):
    from types import SimpleNamespace
    from PIL import ImageFilter

    from agentic_image.data.paired import _canvas_artifacts
    from agentic_image.data.proposals import _registered_mask, registration_evidence

    rng = np.random.default_rng(19)
    source = Image.fromarray(rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)).filter(
        ImageFilter.GaussianBlur(1)
    )
    target = source.resize((256, 128), Image.Resampling.NEAREST)
    assert registration_evidence(source, target)["verified"]
    assert not registration_evidence(source, target.resize((256, 130)))["verified"]
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source_id = artifacts.put_image(source).artifact_id
    target_id = artifacts.put_image(target).artifact_id
    mask = np.zeros((128, 256), dtype=np.float32)
    mask[16:48, 32:96] = 1
    mask_id = artifacts.put_array(mask).artifact_id
    registered = _registered_mask(artifacts, mask_id, source, target)
    assert registered.shape == (64, 128)
    assert registered[8:24, 16:48].sum() == 512
    components = SimpleNamespace(artifacts=artifacts, config={"canvas": (128, 128)})
    row = {
        "sample_id": "resized-pair",
        "source_image_id": source_id,
        "target_image_id": target_id,
        "edit_mask_id": mask_id,
    }
    with pytest.raises(ValueError, match="verified shared registration"):
        _canvas_artifacts(components, row, {"registration_verified": False})
    source_canvas, target_canvas, _, mask_canvas, frame = _canvas_artifacts(
        components, row, {"registration_verified": True}
    )
    assert artifacts.get_image(source_canvas).size == (128, 128)
    assert artifacts.get_image(target_canvas).size == (128, 128)
    projected = artifacts.get_array(mask_canvas)
    assert projected[44:52, 20:44].min() == 1
    assert projected[:32].max() == 0
    assert projected.sum() == 512
    assert frame.source_hw == (64, 128)
    assert artifacts.get_ref(target_canvas).metadata["original_frame"]["source_hw"] == [128, 256]


def test_independent_annotation_never_extracts_requirements_from_target(tmp_path):
    from types import SimpleNamespace

    from agentic_image.data import propose_annotations

    class Judge:
        config = SimpleNamespace(revision="test-revision")

        def generate_json(self, text, images, max_new_tokens):
            assert len(images) == 1
            assert images[0].getpixel((0, 0)) == (255, 0, 0)
            if text.startswith("Create independent"):
                return {
                    "atoms": [
                        {
                            "raw_span": "Make the cup blue",
                            "predicate": "The cup is blue",
                            "expected_value": "blue",
                            "checker_route": "semantic",
                        }
                    ],
                    "entities": [],
                    "global_edit": False,
                    "ambiguous": False,
                }
            return {"complete": True, "invented": False, "ambiguous": False}

        def check(self, *, atom_id, images, **kwargs):
            assert images[0].getpixel((0, 0)) == (0, 0, 255)
            return AtomScore(atom_id, 0.95, atom_id != "eval:0000")

    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = artifacts.put_image(Image.new("RGB", (32, 32), "red")).artifact_id
    target = artifacts.put_image(Image.new("RGB", (32, 32), "blue")).artifact_id
    components = SimpleNamespace(
        artifacts=artifacts,
        judge=Judge(),
        config={},
        observer=SimpleNamespace(tools=None),
        load_judge=lambda private: None,
    )
    candidate = tmp_path / "candidates.jsonl"
    output = tmp_path / "proposed.jsonl"
    write_jsonl(
        candidate,
        [
            {
                "sample_id": "sample",
                "split_id": "train",
                "instruction": "Make the cup blue",
                "source_image_id": source,
                "target_image_id": target,
                "registration": {"verified": True},
            }
        ],
    )
    report = propose_annotations(
        components, candidate, output, include_captions=False, geometry=False
    )
    row = json.loads(output.read_text())
    assert report["human_calibration_labels_generated"] == 0
    assert not row["verified"]
    assert row["review_reasons"] == ["target_requirement_verification"]
    assert row["evaluation"]["eval_atoms"][0]["raw_span"] == "Make the cup blue"
    heldout = tmp_path / "heldout.jsonl"
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    rows[0]["split_id"] = "calibration"
    write_jsonl(heldout, rows)
    heldout_output = tmp_path / "heldout_annotations.jsonl"
    propose_annotations(
        components,
        heldout,
        heldout_output,
        split="calibration",
        include_captions=False,
        geometry=False,
    )
    heldout_annotation = json.loads(heldout_output.read_text())
    assert heldout_annotation["verified"]
    assert not heldout_annotation["paired_target_verified"]
