import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.checks.judge import FrozenJudge, TEMPLATES
from agentic_image.config import load_config
from agentic_image.core.artifacts import ArtifactStore, canonical_json
from agentic_image.core.types import PrivateEvaluation
from agentic_image.learning.transaction import atomic_json
from agentic_image.runtime.components import judge_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid deterministic shard selection")
    payload = Path(args.manifest).read_bytes()
    manifest = json.loads(payload)
    if manifest["scope"] != "frozen_8b_raw_scores_without_labels_v1":
        raise ValueError("The frozen calibration input scope differs")
    rows = manifest["families"]
    if not 1 <= len(rows) <= 100 or len({row["family_id"] for row in rows}) != len(rows):
        raise ValueError("A registered batch requires 1 to 100 distinct calibration families")
    if len({row["calibration_index"] for row in rows}) != len(rows):
        raise ValueError("Calibration indices must be unique within a registered batch")
    selected = [row for row in rows if row["calibration_index"] % args.shard_count == args.shard_index]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "atoms").mkdir(exist_ok=True)
    config = load_config(args.config)
    config["devices"]["judge"] = args.device
    config["judge"]["calibration"] = None
    model_config = judge_config(config)
    identity = model_config.calibration_identity
    registration = {
        "scope": manifest["scope"], "input_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "draft_binding": manifest["draft_binding"], "judge_identity": identity,
        "prompt_templates_sha256": hashlib.sha256(canonical_json(TEMPLATES)).hexdigest(),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "family_ids": [row["family_id"] for row in selected],
        "candidate_count": sum(len(row["candidates"]) for row in selected),
        "atom_candidate_count": sum(len(row["candidates"]) * len(row["evaluation"]["eval_atoms"]) for row in selected),
        "labels": None, "calibrated": False, "eligible_for_training": False,
        "human_reviewed": False, "ai_review": "pending",
        "scoring_mode": "frozen_judge_templates_only_without_structured_tools",
        "known_semantics": "raw_template_agreement_only_not_reviewed_truth_or_structured_validation",
        "threshold_semantics": "existing_unfitted_template_agreement_cutoff_not_a_calibration_result",
    }
    registration_path = output / "registration.json"
    if registration_path.exists() and json.loads(registration_path.read_text()) != registration:
        raise ValueError("Resume inputs or judge identity differ from the frozen registration")
    atomic_json(registration_path, registration)
    results = []
    started = time.monotonic()

    def progress(stage, current=None):
        value = {
            "stage": stage, "pid": os.getpid(), "completed_atom_candidates": len(results),
            "total_atom_candidates": registration["atom_candidate_count"], "current": current,
            "labels": None, "calibrated": False, "updated_at": time.time(),
            "seconds": time.monotonic() - started,
        }
        atomic_json(output / "progress.json", value)
        print(json.dumps(value), flush=True)

    progress("loading_judge")
    judge = FrozenJudge.load(model_config)
    if judge.thresholds or any(parameter.requires_grad for parameter in judge.model.parameters()):
        raise RuntimeError("Raw calibration scoring requires the frozen uncalibrated judge")
    artifacts = ArtifactStore(config["data"]["artifacts"])
    structured_routes = {"count", "text", "identity", "left_of", "right_of", "above", "below", "inside", "contains", "layout", "geometry"}
    for row in selected:
        evaluation = PrivateEvaluation.model_validate(row["evaluation"])
        if not evaluation.eval_atoms or evaluation.original_protected_mask_id is not None:
            raise ValueError("Raw atom scoring requires fixed request atoms without private masks")
        draft_hash = hashlib.sha256(canonical_json(row["evaluation"])).hexdigest()
        source_id = row["source_image_id"]
        source = artifacts.get_image(source_id)
        for candidate in row["candidates"]:
            reference = artifacts.get_ref(candidate["artifact_id"])
            if reference.value_id != candidate["value_id"] or reference.kind != "rgb":
                raise ValueError("A registered calibration candidate identity changed")
            image = artifacts.get_image(candidate["artifact_id"])
            for atom_index, atom in enumerate(evaluation.eval_atoms):
                key = f'{row["calibration_index"]:03d}.{candidate["role"]}.{atom_index:03d}'
                atom_path = output / "atoms" / f"{key}.json"
                binding = {
                    "key": key, "sample_id": row["sample_id"], "family_id": row["family_id"],
                    "calibration_index": row["calibration_index"], "calibration_fold": row["calibration_fold"],
                    "candidate_role": candidate["role"], "candidate_artifact_id": candidate["artifact_id"],
                    "candidate_value_id": candidate["value_id"], "source_image_id": source_id,
                    "atom": atom.model_dump(mode="json"), "draft_evaluation_sha256": draft_hash,
                    "input_manifest_sha256": registration["input_manifest_sha256"], "judge_identity": identity,
                }
                if atom_path.exists():
                    saved = json.loads(atom_path.read_text())
                    if any(saved.get(name) != value for name, value in binding.items()):
                        raise ValueError("A resumed atom result has different frozen inputs")
                    results.append(saved)
                    continue
                expected = atom.expected_value
                reference_id = expected.get("reference_image_id") if isinstance(expected, dict) else None
                if reference_id not in (None, source_id):
                    raise ValueError("The request draft contains an unregistered reference image")
                kind = expected.get("kind", atom.checker_route) if isinstance(expected, dict) else atom.checker_route
                begin = time.monotonic()
                progress("scoring", key)
                score = judge.check(
                    atom_id=atom.eval_id, request=row["request"],
                    property_text=f"{atom.predicate}\nExpected: {json.dumps(expected)}",
                    images=[image, source], roles="candidate, original source or named identity reference",
                    kind=kind, structured=None,
                )
                result = {
                    **binding, "score": asdict(score), "labels": None, "calibrated": False,
                    "eligible_for_training": False, "ai_review": "pending", "human_reviewed": False,
                    "structured_evidence_pending": kind in structured_routes,
                    "seconds": time.monotonic() - begin, "judge_forwards": judge.forwards,
                }
                atomic_json(atom_path, result)
                results.append(result)
        atomic_json(output / "results.json", results)
        progress("scoring")
    atomic_json(output / "results.json", results)
    summary = {
        **registration, "status": "completed", "completed_atom_candidates": len(results),
        "raw_known": sum(row["score"]["known"] for row in results),
        "raw_unknown": sum(not row["score"]["known"] for row in results),
        "structured_evidence_pending": sum(row["structured_evidence_pending"] for row in results),
        "seconds": time.monotonic() - started,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(args.device) / 2**30,
    }
    atomic_json(output / "summary.json", summary)
    progress("complete")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
