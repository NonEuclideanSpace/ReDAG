import json

import numpy as np

from pathlib import Path

from agentic_image.core.geometry import letterbox, make_frame, transform_mask
from agentic_image.core.types import PrivateEvaluation

from .evidence import StructuredChecks
from .reward import AtomScore, preservation_score, terminal_reward


class PrivateEvaluator:
    def __init__(self, judge, artifacts, evaluations, tools=None, unconfirmed_route_policy="raise"):
        if unconfirmed_route_policy not in {"raise", "unknown"}:
            raise ValueError("Unconfirmed routes must raise or remain explicitly unknown")
        self.judge = judge
        self.artifacts = artifacts
        self.evaluations = evaluations
        self.tools = tools
        self.source_cache = {}
        self.structured = StructuredChecks(tools)
        self.unconfirmed_route_policy = unconfirmed_route_policy

    @classmethod
    def from_file(cls, judge, artifacts, path, tools=None, unconfirmed_route_policy="raise"):
        evaluations = {}
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                evaluation = PrivateEvaluation.model_validate(row["evaluation"])
                if not evaluation.eval_atoms:
                    raise ValueError("Private evaluation must retain all fixed requirements")
                evaluations[row["task_id"]] = evaluation
        return cls(judge, artifacts, evaluations, tools, unconfirmed_route_policy)

    def score_image(self, task, image_id):
        evaluation = self.evaluations[task.task_id]
        result = []
        unresolved = evaluation.annotation_provenance.get("forced_unknown_eval_ids", {})
        for atom in evaluation.eval_atoms:
            if atom.eval_id in unresolved:
                result.append(AtomScore(
                    atom.eval_id, 0.0, False, atom.threshold,
                    evidence=(f"source_grounding_unresolved:{unresolved[atom.eval_id]}",),
                ))
                continue
            expected = atom.expected_value
            reference_id = (
                expected.get("reference_image_id") if isinstance(expected, dict) else None
            )
            reference_id = reference_id or task.user_request.source_image_id
            images = [self.artifacts.get_image(image_id)]
            if reference_id:
                images.append(self.artifacts.get_image(reference_id))
            kind = (
                expected.get("kind", atom.checker_route)
                if isinstance(expected, dict)
                else atom.checker_route
            )
            calibration = getattr(self.judge, "thresholds", {}).get(kind, {})
            if self.unconfirmed_route_policy == "unknown" and not calibration.get("confirmed", False):
                result.append(AtomScore(
                    atom.eval_id, 0.0, False, atom.threshold, False,
                    (f"not_evaluated_calibration:{kind}",),
                ))
                continue
            structured = self.structured.evaluate(
                atom.eval_id, image_id, kind, expected, reference_id
            )
            audit = {}
            score = self.judge.check(
                atom_id=atom.eval_id,
                request=task.user_request.text,
                property_text=f"{atom.predicate}\nExpected: {json.dumps(expected)}",
                images=images,
                roles="candidate, original source or named identity reference",
                kind=kind,
                structured=structured,
                **({"audit": audit} if getattr(self.judge, "supports_probability_audit", False) else {}),
            )
            evidence = score.evidence
            if audit:
                audit.update(
                    candidate_image_id=image_id,
                    reference_image_id=reference_id,
                    effective_threshold=max(atom.threshold, score.threshold),
                )
                evidence = (*evidence, "judge_score_audit:" + json.dumps(
                    audit, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ))
            result.append(
                AtomScore(
                    score.atom_id,
                    score.value,
                    score.known,
                    max(atom.threshold, score.threshold),
                    score.exact_pass,
                    evidence,
                )
            )
        return tuple(result)

    def evaluate(
        self, task, image_id, *, renders, inspects, actions, invalids, terminal_reason="finished"
    ):
        arguments = dict(
            renders=renders,
            inspects=inspects,
            actions=actions,
            invalids=invalids,
            terminal_reason=terminal_reason,
            budgets=(task.limits.max_renders, task.limits.max_inspects, task.limits.max_actions),
        )
        if image_id is None or invalids >= 2:
            return terminal_reward((), image_present=False, **arguments)
        evaluation = self.evaluations[task.task_id]
        source_id = task.user_request.source_image_id
        mask_id = evaluation.original_protected_mask_id
        protection = evaluation.annotation_provenance.get("protection", {})
        if source_id and not mask_id:
            if not (
                protection.get("status") == "not_applicable"
                and protection.get("scope") == "global_edit"
                and protection.get("source_image_id") == source_id
                and protection.get("reason")
                and protection.get("review_evidence")
            ):
                raise ValueError(
                    "Editing evaluation requires source-grounded protection evidence; "
                    "missing masks are not successful preservation"
                )
        scores = self.score_image(task, image_id)
        baseline = ()
        keep = None
        preservation_status = "not_applicable"
        if source_id:
            cache_key = (task.task_id, source_id, evaluation.evaluator_version)
            if cache_key not in self.source_cache:
                self.source_cache[cache_key] = self.score_image(task, source_id)
            baseline = self.source_cache[cache_key]
            if mask_id:
                mask = self.artifacts.get_array(mask_id).copy()
                source_ref = self.artifacts.get_ref(source_id)
                source = self.artifacts.get_image(source_id)
                output = self.artifacts.get_image(image_id)
                canvas_hw = (task.canvas.height, task.canvas.width)
                aligned_source_id = source_id
                aligned_output_id = image_id
                aligned_frame = source_ref.frame
                if source_ref.frame is None:
                    aligned_frame = make_frame(
                        (source.height, source.width), canvas_hw,
                        f"private-evaluation:{task.task_id}",
                    )
                    mask_frame = self.artifacts.get_ref(mask_id).frame
                    if mask_frame is not None:
                        if mask_frame != aligned_frame or mask.shape != canvas_hw:
                            raise ValueError("Private mask has a different source-to-canvas registration")
                    else:
                        mask = transform_mask(mask, aligned_frame, "binary")
                    source = letterbox(source, aligned_frame)
                    aligned_source_id = self.artifacts.put_image(
                        source, frame=aligned_frame,
                        metadata={"visibility": "private", "role": "evaluation_source", "original_image_id": source_id},
                    ).artifact_id
                if image_id == source_id:
                    output = source
                    aligned_output_id = aligned_source_id
                if (output.height, output.width) != canvas_hw:
                    raise ValueError("Private evaluation output differs from the registered canvas")
                if source_ref.frame is not None:
                    frame = source_ref.frame
                    if frame.canvas_hw != canvas_hw:
                        raise ValueError("Private source frame differs from the evaluation canvas")
                    valid = np.zeros(frame.canvas_hw, dtype=np.float32)
                    px, py = frame.padding_xy
                    height, width = frame.resized_hw
                    valid[py : py + height, px : px + width] = 1
                    if valid.shape != mask.shape:
                        raise ValueError(
                            "Private protection mask is not in the original source canvas frame"
                        )
                    mask *= valid
                if not np.any(mask > 0):
                    raise ValueError("Private protection annotation has no valid source pixels")
                if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
                    raise ValueError("Private protection contains invalid pixel weights")
                feature_arguments = {}
                if self.tools is not None:
                    effective_mask = self.artifacts.put_array(
                        mask,
                        frame=aligned_frame,
                        metadata={"visibility": "private", "role": "effective_protection"},
                    )
                    features = self.tools.call(
                        "preservation",
                        source_image_id=aligned_source_id,
                        image_id=aligned_output_id,
                        mask_id=effective_mask.artifact_id,
                    )
                    feature_arguments = features
                elif mask.any():
                    raise RuntimeError(
                        "Protected-region evaluation requires the frozen DINO feature service"
                    )
                keep = preservation_score(
                    source,
                    output,
                    mask,
                    **feature_arguments,
                )
                preservation_status = "measured"
        protected = evaluation.annotation_provenance.get("preserved_eval_ids", ())
        return terminal_reward(
            scores,
            source_atoms=baseline,
            protected_atom_ids=protected,
            preservation=keep,
            preservation_status=preservation_status,
            **arguments,
        )
