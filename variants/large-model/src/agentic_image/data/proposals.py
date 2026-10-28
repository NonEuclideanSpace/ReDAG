from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from agentic_image.checks.evidence import StructuredChecks, box_value
from agentic_image.core.geometry import box_iou
from agentic_image.core.types import BoundingBox, PrivateEvaluation

from .io import append_jsonl, read_jsonl, write_report


ATOM_PROMPT = """Create independent private evaluation requirements from the original request and its public source only. Do not imagine a target image or inspect a controller plan. Return JSON with atoms, entities, global_edit, and ambiguous. Every atom has raw_span copied verbatim from the request, predicate, expected_value, checker_route, mandatory=true, and preserved. Preserve every requested object, count, attribute, relation, exact text, identity, and unchanged-content constraint. Split independent requirements without inventing requirements. Use checker_route semantic for ordinary visual attributes; count requires expected_value {kind:count,object:description,count:integer}; text requires {kind:text,text:exact string}; left_of/right_of/above/below/inside/contains require {kind:the same literal checker_route,subject:description,object:description}. Do not invent artifact IDs, identity thresholds, or target boxes. Entities have name, aliases, source_description, target_description, and changed. Only include entities grounded in the request or its public source, with no guessed coordinates. An absent source or target description is null. global_edit is true for whole-image style, lighting, background, or other nonlocal changes. ambiguous is true if the request cannot be interpreted reliably. Request: """

CAPTION_PROMPT = """Describe this complete image for text-to-image training. Return JSON {caption:string, uncertain:boolean}. Include the major entities, their visible number, distinguishing appearance, spatial relationships, setting, background, lighting, and visible written text when legible. Describe only what is visible. Do not write an editing instruction, refer to a source, infer identities, or omit the background. Mark uncertain if a faithful complete description is not possible."""


def _known_pass(score, minimum=0.9):
    return score.known and score.passed and score.value >= max(minimum, score.threshold)


def _evaluation(judge, request, images, task_key, max_atoms):
    request = unicodedata.normalize("NFC", request)
    proposal = judge.generate_json(ATOM_PROMPT + request, images, max_new_tokens=3072)
    items = proposal.get("atoms", [])
    if not isinstance(items, list) or not 1 <= len(items) <= max_atoms:
        raise ValueError("Independent evaluation has an invalid atom count")
    atoms, preserved = [], []
    for index, item in enumerate(items):
        span = item["raw_span"]
        if not isinstance(span, str) or not span or span not in request:
            raise ValueError("Independent evaluation must quote the original request")
        expected = item.get("expected_value")
        route = item.get("checker_route", "semantic")
        if route not in {
            "semantic",
            "count",
            "text",
            "left_of",
            "right_of",
            "above",
            "below",
            "inside",
            "contains",
        }:
            raise ValueError(
                "Independent automatic evaluation needs a supported explicit checking route"
            )
        if isinstance(expected, dict) and (
            expected.get("reference_image_id") is not None or expected.get("target_box") is not None
        ):
            raise ValueError(
                "Automatic evaluation must not invent a private reference or target box"
            )
        if isinstance(expected, dict) and expected.get("kind", route) != route:
            raise ValueError("Structured checking route and expected value disagree")
        identifier = f"eval:{index:04d}"
        atoms.append(
            {
                "eval_id": identifier,
                "raw_span": span,
                "predicate": item["predicate"],
                "expected_value": item.get("expected_value"),
                "checker_route": item.get("checker_route", "semantic"),
                "threshold": 0.8,
                "mandatory": True,
            }
        )
        if item.get("preserved"):
            preserved.append(identifier)
    verification = judge.generate_json(
        "Independently audit the following private evaluation against the original user request and public source. Return JSON {complete:boolean, invented:boolean, ambiguous:boolean, reason:string}. complete means every explicit requirement is represented, including quantities, relations, exact text and preservation. invented means any atom adds an unrequested requirement or replaces an ambiguous requirement with a guess. Do not use a hidden target or controller plan.\nOriginal request: "
        + request
        + "\nEvaluation: "
        + json.dumps(atoms),
        images,
        max_new_tokens=512,
    )
    covered = (
        verification.get("complete") is True
        and verification.get("invented") is False
        and verification.get("ambiguous") is False
        and not proposal.get("ambiguous", True)
    )
    evaluation = PrivateEvaluation.model_validate(
        {
            "private_task_key": task_key,
            "eval_atoms": atoms,
            "source_baseline_observations": [],
            "annotation_provenance": {
                "preserved_eval_ids": preserved,
                "origin": "independent_original_request_extraction",
                "coverage_audit": verification,
            },
            "evaluator_version": f"offline-independent:{judge.config.revision}",
        }
    )
    return evaluation, proposal, covered


def _score_target(components, evaluation, request, target_id, source_id=None):
    structured = StructuredChecks(components.observer.tools)
    images = [components.artifacts.get_image(target_id)]
    if source_id:
        images.append(components.artifacts.get_image(source_id))
    scores = []
    for atom in evaluation.eval_atoms:
        expected = atom.expected_value
        kind = (
            expected.get("kind", atom.checker_route)
            if isinstance(expected, dict)
            else atom.checker_route
        )
        evidence = structured.evaluate(atom.eval_id, target_id, kind, expected, source_id)
        score = components.judge.check(
            atom_id=atom.eval_id,
            request=request,
            property_text=atom.predicate + "\nExpected: " + json.dumps(expected),
            images=images,
            roles="private paired target, original public source"
            if source_id
            else "private generation target",
            kind=kind,
            structured=evidence,
        )
        scores.append(asdict(score))
    return scores


def registration_evidence(source, target, official=False):
    if source.width * target.height != target.width * source.height:
        return {"verified": False, "method": "aspect_ratio_mismatch"}
    if official and source.size == target.size:
        return {"verified": True, "method": "official_registered_pair"}
    width, height = source.size
    factor = min(1.0, 512 / max(width, height))
    shape = max(32, round(width * factor)), max(32, round(height * factor))
    left = (
        np.asarray(source.resize(shape, Image.Resampling.BILINEAR).convert("L"), dtype=np.float64)
        / 255
    )
    right = (
        np.asarray(target.resize(shape, Image.Resampling.BILINEAR).convert("L"), dtype=np.float64)
        / 255
    )
    matches, quadrants = [], set()
    for row in range(4):
        for column in range(4):
            y0, y1 = row * shape[1] // 4, (row + 1) * shape[1] // 4
            x0, x1 = column * shape[0] // 4, (column + 1) * shape[0] // 4
            a, b = left[y0:y1, x0:x1], right[y0:y1, x0:x1]
            if min(float(a.std()), float(b.std())) < 0.04:
                continue
            correlation = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
            error = float(np.median(np.abs(a - b)))
            gradient = np.hypot(*np.gradient(a))
            if correlation >= 0.97 and error <= 0.025 and np.quantile(gradient, 0.75) >= 0.015:
                matches.append(
                    {"tile": [row, column], "correlation": correlation, "median_error": error}
                )
                quadrants.add((row // 2, column // 2))
    return {
        "verified": len(matches) >= 4 and len(quadrants) >= 3,
        "method": "distributed_textured_pixel_correspondence",
        "source_size": list(source.size),
        "target_size": list(target.size),
        "matched_tiles": matches,
        "quadrants": len(quadrants),
    }


def _registered_mask(artifacts, identifier, source, target):
    mask = artifacts.get_array(identifier)
    if mask.shape == (source.height, source.width):
        return mask
    if (
        mask.shape != (target.height, target.width)
        or source.width * target.height != target.width * source.height
    ):
        raise ValueError("Mask does not match either registered image")
    return np.asarray(
        Image.fromarray(mask).resize(source.size, Image.Resampling.NEAREST), dtype=np.float32
    )


def _localized_entity(components, image_id, description, name):
    if not description:
        return None
    tools = components.observer.tools
    result = tools.call("detect", image_id=image_id, text=description)
    detections = result.get("detections", [])
    if not result.get("known") or len(detections) != 1 or float(detections[0]["score"]) < 0.5:
        return None
    box = BoundingBox.model_validate(box_value(detections[0]["box"]))
    segment = tools.call("segment", image_id=image_id, region=box.model_dump(), entity_id=name)
    if (
        not segment.get("known")
        or float(segment.get("score", 0)) < 0.9
        or len(segment.get("artifact_ids", [])) != 1
    ):
        return None
    mask_id = segment["artifact_ids"][0]
    mask = components.artifacts.get_array(mask_id)
    image = components.artifacts.get_image(image_id)
    if mask.shape != (image.height, image.width) or not np.isfinite(mask).all():
        return None
    mask = mask > 0.5
    if mask.sum() < 16 or mask.mean() > 0.85:
        return None
    ys, xs = np.where(mask)
    x0, x1 = int(box.x0 * image.width), int(np.ceil(box.x1 * image.width))
    y0, y1 = int(box.y0 * image.height), int(np.ceil(box.y1 * image.height))
    if mask[y0:y1, x0:x1].sum() / mask.sum() < 0.95:
        return None
    measured = BoundingBox(
        x0=float(xs.min() / image.width),
        y0=float(ys.min() / image.height),
        x1=float((xs.max() + 1) / image.width),
        y1=float((ys.max() + 1) / image.height),
    )
    if box_iou(box, measured) < 0.5:
        return None
    crop = image.crop((x0, y0, x1, y1))
    check = components.judge.check(
        atom_id="geometry_identity",
        request="Independently verify a detected entity crop.",
        property_text=f"The crop clearly contains exactly the intended entity: {description}. Its main body is visible and the crop is not an unrelated object.",
        images=[crop],
        roles="detected crop",
        kind="semantic",
    )
    if not _known_pass(check):
        return None
    return {
        "box": list(measured.as_tuple()),
        "mask_id": mask_id,
        "detector_score": float(detections[0]["score"]),
        "segmentation_score": float(segment["score"]),
        "review": asdict(check),
    }


def _geometry(components, proposal, source_id, target_id):
    entities, evidence = [], []
    for item in proposal.get("entities", [])[:8]:
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        source = (
            _localized_entity(components, source_id, item.get("source_description"), name)
            if source_id
            else None
        )
        target = _localized_entity(components, target_id, item.get("target_description"), name)
        if source is None and target is None:
            continue
        entity = {
            "name": name,
            "aliases": item.get("aliases", []),
            "changed": bool(item.get("changed")),
            "source_box": source["box"] if source else None,
            "target_box": target["box"] if target else None,
            "source_mask_id": source["mask_id"] if source else None,
            "target_mask_id": target["mask_id"] if target else None,
        }
        entities.append(entity)
        evidence.append({"name": name, "source": source, "target": target})
    return entities, evidence


def _caption(components, row, max_atoms):
    target_id = row["target_image_id"]
    image = components.artifacts.get_image(target_id)
    proposed = components.judge.generate_json(CAPTION_PROMPT, [image], max_new_tokens=1024)
    caption = proposed.get("caption", "").strip()
    if not caption or proposed.get("uncertain", True):
        return {
            "complete_caption_verified": False,
            "caption_review_reason": "uncertain_or_empty_caption",
        }
    review = components.judge.check(
        atom_id="complete_caption",
        request="Independently audit the completeness and correctness of a whole-image description.",
        property_text="Every assertion in this caption is visually true, the counts and spatial relations are correct, and the caption covers the main foreground entities and background without reducing the scene to an editing instruction: "
        + caption,
        images=[image],
        roles="private caption target",
        kind="semantic",
    )
    evaluation, proposal, covered = _evaluation(
        components.judge, caption, [], f"caption:{row['sample_id']}", max_atoms
    )
    scores = _score_target(components, evaluation, caption, target_id)
    verified = (
        covered
        and _known_pass(review)
        and all(
            item["known"]
            and item["value"] >= max(0.9, item["threshold"])
            and item["exact_pass"] is not False
            for item in scores
        )
    )
    result = {
        "complete_caption": caption,
        "caption_image_role": "target",
        "complete_caption_verified": _known_pass(review),
        "caption_evaluation": evaluation.model_dump(mode="json"),
        "caption_evaluation_verified": verified,
        "caption_verification": {"completeness": asdict(review), "target_atoms": scores},
        "caption_entity_proposal": proposal,
    }
    return result


def propose_annotations(
    components,
    candidates_path,
    output_path,
    limit=12000,
    split="train",
    include_captions=True,
    geometry=True,
):
    if limit < 1:
        raise ValueError("Annotation proposal budget must be positive")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Annotation proposals must not overwrite reviewed annotations")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=False)
    components.load_judge(private=False)
    reasons = Counter()
    report = {
        "attempted": 0,
        "verified": 0,
        "requires_review": 0,
        "geometry_verified": 0,
        "captions_verified": 0,
        "split": split,
        "limit": limit,
        "human_audit_required_per_bucket": 100,
        "human_calibration_labels_generated": 0,
    }
    max_atoms = components.config.get("train_max_eval_atoms", 12)
    for row in read_jsonl(candidates_path):
        if report["attempted"] >= limit:
            break
        if row.get("split_id") != split:
            continue
        report["attempted"] += 1
        source_id, target_id = row.get("source_image_id"), row["target_image_id"]
        provenance = {
            "source": "frozen_teacher_with_independent_request_and_target_verification",
            "model_revision": components.judge.config.revision,
            "annotation_date": datetime.now(timezone.utc).isoformat(),
            "human_reviewed": False,
            "requires_bucket_audit": True,
        }
        annotation = {
            "sample_id": row["sample_id"],
            "verified": False,
            "semantic_compatible": False,
            "registration_verified": False,
            "geometry_verified": False,
            "benchmark_derived": False,
            "provenance": provenance,
            "entities": [],
            "box_frame": "original",
            "reliable_edit_mask_id": row.get("edit_mask_id"),
            "protected_mask_id": None,
            "public_controls": {},
            "review_reasons": [],
        }
        try:
            images = [components.artifacts.get_image(source_id)] if source_id else []
            evaluation, proposal, covered = _evaluation(
                components.judge, row["instruction"], images, row["sample_id"], max_atoms
            )
            annotation["evaluation"] = evaluation.model_dump(mode="json")
            annotation["task_definition_verified"] = covered
            if not covered:
                annotation["review_reasons"].append("original_request_coverage_or_ambiguity")
            scores = _score_target(components, evaluation, row["instruction"], target_id, source_id)
            annotation["target_verification"] = scores
            annotation["semantic_compatible"] = covered and all(
                item["known"]
                and item["value"] >= max(0.9, item["threshold"])
                and item["exact_pass"] is not False
                for item in scores
            )
            if not annotation["semantic_compatible"]:
                annotation["review_reasons"].append("target_requirement_verification")
            target = components.artifacts.get_image(target_id)
            registration = (
                registration_evidence(
                    images[0], target, row.get("registration", {}).get("verified", False)
                )
                if source_id
                else {"verified": True, "method": "single_generation_target"}
            )
            annotation["registration_evidence"] = registration
            annotation["registration_verified"] = registration["verified"]
            if not registration["verified"]:
                annotation["review_reasons"].append("unverified_registration")
            if source_id:
                consistency = components.judge.check(
                    atom_id="pair_consistency",
                    request=row["instruction"],
                    property_text="The target correctly follows the requested edit. The identities, scene content, and layout not requested to change remain consistent with the original source, with no unrelated object changes or global corruption.",
                    images=[target, images[0]],
                    roles="private paired target, original public source",
                    kind="semantic",
                )
                annotation["pair_consistency"] = asdict(consistency)
                if not _known_pass(consistency):
                    annotation["review_reasons"].append("unverified_content_preservation")
            if geometry:
                entities, geometry_evidence = _geometry(components, proposal, source_id, target_id)
                annotation["entities"] = entities
                annotation["geometry_verification"] = geometry_evidence
                annotation["geometry_verified"] = bool(entities) and registration["verified"]
                changed = [item for item in entities if item["changed"]]
                annotation["two_changed_entities"] = len(changed) >= 2
                expected_changed = [
                    item for item in proposal.get("entities", []) if item.get("changed")
                ]
                if (
                    not annotation["reliable_edit_mask_id"]
                    and changed
                    and len(changed) == len(expected_changed)
                    and not proposal.get("global_edit", True)
                    and registration["verified"]
                ):
                    identifiers = [
                        item[key]
                        for item in changed
                        for key in ("source_mask_id", "target_mask_id")
                        if item[key]
                    ]
                    mask = np.maximum.reduce(
                        [
                            _registered_mask(
                                components.artifacts, identifier, images[0] if images else target, target
                            )
                            for identifier in identifiers
                        ]
                    )
                    annotation["reliable_edit_mask_id"] = components.artifacts.put_array(
                        mask.astype(np.float32),
                        producer_id=row["sample_id"],
                        metadata={
                            "role": "private_edit_annotation",
                            "origin": "verified_detector_sam_union",
                            "annotation_provenance": provenance,
                        },
                    ).artifact_id
                unchanged = [
                    item
                    for item in entities
                    if item["source_box"] and item["target_box"] and not item["changed"]
                ]
                if unchanged:
                    item = unchanged[0]
                    annotation["appearance_reference"] = {
                        "verified": True,
                        "origin": "source",
                        "source_box": item["source_box"],
                        "entity_name": item["name"],
                        "provenance": provenance,
                    }
            if row.get("edit_mask_id") and registration["verified"]:
                base_image = images[0] if images else target
                mask = _registered_mask(
                    components.artifacts, row["edit_mask_id"], base_image, target
                )
                if mask.shape == (base_image.height, base_image.width):
                    radius = max(1, round(min(mask.shape) * 0.01))
                    protected = ~binary_dilation(mask > 0.5, iterations=radius)
                    if protected.any():
                        annotation["protected_mask_id"] = components.artifacts.put_array(
                            protected.astype(np.float32),
                            producer_id=row["sample_id"],
                            metadata={
                                "role": "private_protected_annotation",
                                "origin": "official_edit_mask_complement",
                                "margin_pixels": radius,
                            },
                        ).artifact_id
            if include_captions:
                try:
                    caption = _caption(components, row, max_atoms)
                    caption_proposal = caption.pop("caption_entity_proposal", {})
                    if geometry and caption.get("caption_evaluation_verified"):
                        caption_entities, caption_geometry = _geometry(
                            components, caption_proposal, None, target_id
                        )
                        caption["caption_entities"] = caption_entities
                        caption["caption_geometry_verification"] = caption_geometry
                        caption["caption_geometry_verified"] = bool(caption_entities)
                    if annotation.get("appearance_reference"):
                        caption["caption_appearance_reference"] = annotation["appearance_reference"]
                    annotation.update(caption)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    annotation["caption_evaluation_verified"] = False
                    annotation["caption_review_reason"] = (
                        f"invalid_caption:{type(error).__name__}:{error}"
                    )
            annotation["verified"] = not annotation["review_reasons"]
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            annotation["review_reasons"].append(f"invalid_proposal:{type(error).__name__}:{error}")
        annotation["paired_target_verified"] = annotation["verified"]
        if split != "train":
            annotation["verified"] = bool(annotation.get("task_definition_verified", False))
        report["verified"] += int(annotation["verified"])
        report["requires_review"] += int(not annotation["verified"])
        report["geometry_verified"] += int(annotation["geometry_verified"])
        report["captions_verified"] += int(annotation.get("caption_evaluation_verified", False))
        reasons.update(annotation["review_reasons"])
        append_jsonl(output, [annotation])
    report["review_reasons"] = dict(reasons)
    report["annotations_path"] = str(output.resolve())
    write_report(output.with_suffix(".report.json"), report)
    return report
