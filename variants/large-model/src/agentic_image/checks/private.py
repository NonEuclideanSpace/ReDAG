import json

import numpy as np

from pathlib import Path

from agentic_image.core.types import PrivateEvaluation

from .evidence import StructuredChecks
from .reward import AtomScore, preservation_score, terminal_reward


class PrivateEvaluator:
    def __init__(self, judge, artifacts, evaluations, tools=None):
        self.judge = judge
        self.artifacts = artifacts
        self.evaluations = evaluations
        self.tools = tools
        self.source_cache = {}
        self.structured = StructuredChecks(tools)

    @classmethod
    def from_file(cls, judge, artifacts, path, tools=None):
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
        return cls(judge, artifacts, evaluations, tools)

    def score_image(self, task, image_id):
        evaluation = self.evaluations[task.task_id]
        result = []
        for atom in evaluation.eval_atoms:
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
            structured = self.structured.evaluate(
                atom.eval_id, image_id, kind, expected, reference_id
            )
            score = self.judge.check(
                atom_id=atom.eval_id,
                request=task.user_request.text,
                property_text=f"{atom.predicate}\nExpected: {json.dumps(expected)}",
                images=images,
                roles="candidate, original source or named identity reference",
                kind=kind,
                structured=structured,
            )
            result.append(
                AtomScore(
                    score.atom_id,
                    score.value,
                    score.known,
                    max(atom.threshold, score.threshold),
                    score.exact_pass,
                    score.evidence,
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
        scores = self.score_image(task, image_id)
        source_id = task.user_request.source_image_id
        baseline = ()
        keep = 1.0
        if source_id:
            cache_key = (task.task_id, source_id, evaluation.evaluator_version)
            if cache_key not in self.source_cache:
                self.source_cache[cache_key] = self.score_image(task, source_id)
            baseline = self.source_cache[cache_key]
            mask_id = evaluation.original_protected_mask_id
            if mask_id:
                mask = self.artifacts.get_array(mask_id).copy()
                source_ref = self.artifacts.get_ref(source_id)
                if source_ref.frame is not None:
                    frame = source_ref.frame
                    valid = np.zeros(frame.canvas_hw, dtype=np.float32)
                    px, py = frame.padding_xy
                    height, width = frame.resized_hw
                    valid[py : py + height, px : px + width] = 1
                    if valid.shape != mask.shape:
                        raise ValueError(
                            "Private protection mask is not in the original source canvas frame"
                        )
                    mask *= valid
                feature_arguments = {}
                if self.tools is not None:
                    effective_mask = self.artifacts.put_array(
                        mask,
                        frame=source_ref.frame,
                        metadata={"visibility": "private", "role": "effective_protection"},
                    )
                    features = self.tools.call(
                        "preservation",
                        source_image_id=source_id,
                        image_id=image_id,
                        mask_id=effective_mask.artifact_id,
                    )
                    feature_arguments = features
                elif mask.any():
                    raise RuntimeError(
                        "Protected-region evaluation requires the frozen DINO feature service"
                    )
                keep = preservation_score(
                    self.artifacts.get_image(source_id),
                    self.artifacts.get_image(image_id),
                    mask,
                    **feature_arguments,
                )
        protected = evaluation.annotation_provenance.get("preserved_eval_ids", ())
        return terminal_reward(
            scores,
            source_atoms=baseline,
            protected_atom_ids=protected,
            preservation=keep,
            **arguments,
        )
