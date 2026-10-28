from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np
import torch

from agentic_image.checks.private import PrivateEvaluator
from agentic_image.checks.public import PublicTaskBuilder
from agentic_image.core.geometry import (
    box_mask,
    letterbox,
    make_frame,
    mask_iou,
    transform_box,
    transform_mask,
)
from agentic_image.core.protocol import PlanAction, parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import BoundingBox, ContractError, PrivateEvaluation
from agentic_image.learning.sampling import stream_seed
from agentic_image.runtime.prompt import SYSTEM_PROMPT, actor_messages

from .io import append_jsonl, read_jsonl, write_jsonl, write_report


PAIR_QUOTAS = {
    "magicbrush": (2000, 1200, 800),
    "omniedit": (1400, 600, 400),
    "t2i": (600, 200, 800),
}


def public_environment(components, task, episode_id):
    tokenizer = components.actor.processor.tokenizer
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    reserve = count(SYSTEM_PROMPT) + 128
    return EpisodeEnvironment(
        task,
        components.artifacts,
        components.compiler,
        components.render_bridge,
        components.observer,
        episode_id=episode_id,
        versions={"compiler": "dag-agent-v3"},
        token_counter=count,
        max_input_tokens=components.config["actor"]["max_input_tokens"] - reserve,
    )


def actor_snapshot(components, environment):
    view = environment.view()
    return components.actor.prepare(
        actor_messages(view, components.artifacts),
        image_artifact_ids=view.image_ids,
        metadata={
            "task_id": view.task_id,
            "revision_id": view.revision_id,
            "view": view.model_dump(mode="json"),
        },
    )


def save_action_record(
    components,
    environment,
    action,
    directory,
    record_id,
    phase,
    state_id,
    provenance,
    snapshot=None,
    sampled=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = actor_snapshot(components, environment) if snapshot is None else snapshot
    if sampled is None:
        raw = (
            action
            if isinstance(action, str)
            else json.dumps(
                environment.public_action(action), ensure_ascii=True, separators=(",", ":")
            )
        )
        ids = components.actor.processor.tokenizer.encode(raw, add_special_tokens=False) + [248046]
    else:
        ids = sampled.token_ids.tolist()
        if sampled.finish_reason == "eos":
            ids = ids[:-1]
        ids.append(248046)
    if not ids or len(ids) > components.config["actor"]["max_new_tokens"]:
        raise ValueError("Verified SFT action exceeds the configured output token budget")
    safe = record_id.replace(":", "_").replace("/", "_")
    snapshot_path = directory / f"{safe}.snapshot.pt"
    snapshot.save(snapshot_path)
    return {
        "record_id": record_id,
        "family_id": environment.task.family_id,
        "state_id": state_id,
        "phase": phase,
        "snapshot_path": str(snapshot_path.resolve()),
        "target_ids": ids,
        "target_mask": [1] * len(ids),
        "verified": True,
        "public_reference_ids": list(snapshot.image_artifact_ids),
        "verifier_provenance": provenance,
    }


def _canvas_artifacts(components, row, annotation):
    artifacts = components.artifacts
    canvas = tuple(components.config["canvas"])
    source_id = row.get("source_image_id")
    target = artifacts.get_image(row["target_image_id"])
    if source_id:
        source = artifacts.get_image(source_id)
        registered = annotation.get(
            "registration_verified", row.get("registration", {}).get("verified", False)
        )
        if not registered or source.width * target.height != target.width * source.height:
            raise ValueError(
                "Paired pixel supervision requires independently verified shared registration"
            )
        frame = make_frame((source.height, source.width), canvas, f"frame:{row['sample_id']}")
        source_id = artifacts.put_image(
            letterbox(source, frame),
            frame=frame,
            producer_id=row["sample_id"],
            metadata={"role": "source", "original_image_id": source_id},
        ).artifact_id
    else:
        frame = make_frame((target.height, target.width), canvas, f"frame:{row['sample_id']}")
    target_frame = make_frame(
        (target.height, target.width), canvas, f"{frame.frame_id}:target"
    )
    target_id = artifacts.put_image(
        letterbox(target, target_frame),
        frame=frame,
        producer_id=row["sample_id"],
        metadata={
            "role": "private_target",
            "original_image_id": row["target_image_id"],
            "original_frame": target_frame.model_dump(mode="json"),
        },
    ).artifact_id
    valid = transform_mask(np.ones(frame.source_hw, dtype=np.float32), frame, "binary")
    valid_id = artifacts.put_array(
        valid, frame=frame, metadata={"role": "valid_supervision"}
    ).artifact_id
    edit_id = annotation.get("reliable_edit_mask_id", row.get("edit_mask_id"))
    if edit_id:
        mask = artifacts.get_array(edit_id)
        if mask.shape == frame.source_hw:
            mask = transform_mask(mask, frame, "soft")
        elif mask.shape == target_frame.source_hw:
            mask = transform_mask(mask, target_frame, "soft")
        elif mask.shape != frame.canvas_hw or annotation.get("mask_frame") != "canvas":
            raise ValueError("Edit supervision mask has no matching registered frame")
        edit_id = artifacts.put_array(
            mask, frame=frame, metadata={"role": "private_edit_supervision"}
        ).artifact_id
    return source_id, target_id, valid_id, edit_id, frame


def _geometry_compatible(components, plan, annotation, frame):
    if not plan.regions:
        return True
    if not annotation.get("geometry_verified"):
        return False
    entities = {entity.entity_id: entity for entity in plan.entities}
    annotations = annotation.get("entities", [])
    for region in plan.regions:
        target_boxes, source_boxes = [], []
        for entity_id in region.entity_ids:
            entity = entities[entity_id]
            matches = [
                item
                for item in annotations
                if item.get("entity_id") == entity_id
                or entity.name.casefold()
                in {name.casefold() for name in (item.get("name", ""), *item.get("aliases", []))}
            ]
            if len(matches) != 1:
                return False
            item = matches[0]
            for side, boxes in (("target", target_boxes), ("source", source_boxes)):
                value = item.get(f"{side}_box")
                if value is not None:
                    box = BoundingBox(x0=value[0], y0=value[1], x1=value[2], y1=value[3])
                    boxes.append(
                        transform_box(box, frame)
                        if annotation.get("box_frame", "original") == "original"
                        else box
                    )
        for side, boxes in (("target", target_boxes), ("source", source_boxes)):
            mask = components.compiler.mask(
                getattr(region, f"{side}_mask_id"), getattr(region, f"{side}_box")
            )
            if mask is None:
                continue
            if not boxes:
                return False
            expected = np.maximum.reduce([box_mask(box, frame.canvas_hw) for box in boxes])
            if mask_iou(mask.cpu().numpy(), expected) < 0.5:
                return False
    return True


def build_aligned_pairs(
    components,
    candidates_path,
    annotations_path,
    output_dir,
    max_candidates=12000,
    quotas=None,
    seed=20261027,
):
    if max_candidates < 1:
        raise ValueError("The paired-data proposal budget must be positive")
    components.load_actor()
    components.load_encoders()
    components.load_judge(private=False)
    annotations = {row["sample_id"]: row for row in read_jsonl(annotations_path)}
    quotas = PAIR_QUOTAS if quotas is None else quotas
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("actor_records.jsonl", "renderer_records.jsonl")):
        raise FileExistsError(
            "Use a new aligned-data directory so previous verified records are preserved"
        )
    builder = PublicTaskBuilder(components.judge, components.artifacts)
    accepted = defaultdict(int)
    appearance_accepted = defaultdict(int)
    report = {
        "attempted_candidates": 0,
        "teacher_calls": 0,
        "accepted": 0,
        "actor_records": 0,
        "rejections": {},
        "quotas": quotas,
    }
    reasons = Counter()
    tasks, private_tasks, actor_records, renderer_records = [], [], [], []
    public_layout_count = 0
    written = defaultdict(int)
    for row in read_jsonl(candidates_path):
        if report["attempted_candidates"] >= max_candidates:
            break
        if row.get("split_id") != "train":
            continue
        if row.get("source_dataset") not in quotas:
            reasons["unsupported_training_source"] += 1
            continue
        annotation = annotations.get(row["sample_id"])
        if (
            annotation is None
            or not annotation.get("verified")
            or not annotation.get("provenance")
            or not annotation.get("semantic_compatible")
        ):
            reasons["missing_independent_annotation"] += 1
            continue
        if annotation.get("benchmark_derived"):
            reasons["benchmark_derived"] += 1
            continue
        if row["source_dataset"] == "t2i" and not annotation.get("complete_caption_verified"):
            reasons["incomplete_generation_caption"] += 1
            continue
        report["attempted_candidates"] += 1
        try:
            source_id, target_id, valid_id, edit_id, frame = _canvas_artifacts(
                components, row, annotation
            )
            evaluation = PrivateEvaluation.model_validate(annotation["evaluation"])
            if not evaluation.eval_atoms or len(evaluation.eval_atoms) > components.config.get(
                "train_max_eval_atoms", 12
            ):
                raise ValueError(
                    "Private evaluation is empty or exceeds the registered task profile"
                )
            protected = annotation.get("protected_mask_id")
            if protected:
                mask = components.artifacts.get_array(protected)
                if mask.shape == frame.source_hw:
                    mask = transform_mask(mask, frame, "soft")
                if mask.shape != frame.canvas_hw:
                    raise ValueError("Protected mask does not match the registered frame")
                protected = components.artifacts.put_array(
                    mask, frame=frame, metadata={"role": "private_protected_mask"}
                ).artifact_id
                evaluation = evaluation.model_copy(update={"original_protected_mask_id": protected})
            appearance = tuple(row.get("appearance_image_ids", ()))
            target_value = components.artifacts.get_ref(row["target_image_id"]).value_id
            if any(
                components.artifacts.get_ref(image_id).value_id == target_value
                for image_id in appearance
            ):
                raise ValueError(
                    "Hidden target pixels cannot be supplied as an appearance reference"
                )
            public_controls = dict(annotation.get("public_controls", {}))
            if row.get("metadata", {}).get("public_appearance"):
                public_controls["appearance_reference"] = row["metadata"]["public_appearance"]
            task = builder.build(
                task_id=row["sample_id"],
                text=row["instruction"],
                source_image_id=source_id,
                appearance_image_ids=appearance,
                controls=public_controls,
                family_id=row["family_id"],
                split_id="train",
                source_dataset=row["source_dataset"],
                license_record=row.get("license_record", {}),
            )
            chosen = None
            for attempt in range(2):
                environment = public_environment(
                    components, task, f"aligned:{row['sample_id']}:{attempt}"
                )
                snapshot = actor_snapshot(components, environment)
                sample = components.actor.sample(
                    snapshot,
                    stream_seed(seed, row["sample_id"], attempt, "teacher_plan"),
                    "offline-plan-teacher",
                )
                report["teacher_calls"] += 1
                try:
                    action = parse_action(sample.text)
                    if not isinstance(action, PlanAction):
                        raise ContractError(
                            "PLAN_REQUIRED", "The paired teacher must propose a root PLAN"
                        )
                    result = environment.step(sample.text)
                    if result.error:
                        raise ContractError(result.error.code, result.error.message)
                    bundle = components.compiler.compile(environment.state.plan, source_id)
                    chosen = environment, snapshot, sample, bundle
                    break
                except (ContractError, ValueError):
                    continue
            if chosen is None:
                reasons["invalid_teacher_plan"] += 1
                continue
            environment, snapshot, sample, bundle = chosen
            regions = len(environment.state.plan.regions)
            key = row["source_dataset"], regions
            if regions > 2 or accepted[key] >= quotas[key[0]][regions]:
                reasons["quota_full"] += 1
                continue
            if appearance and not set(appearance) <= {
                binding.image_id for binding in environment.state.plan.references
            }:
                reasons["appearance_reference_not_consumed"] += 1
                continue
            appearance_target = round(quotas[key[0]][regions] * 0.1)
            if appearance and appearance_accepted[key] >= appearance_target:
                reasons["appearance_quota_full"] += 1
                continue
            if (
                not appearance
                and accepted[key] - appearance_accepted[key]
                >= quotas[key[0]][regions] - appearance_target
            ):
                reasons["nonappearance_quota_full"] += 1
                continue
            if (
                annotation.get("public_controls")
                and (public_layout_count + 1) / (report["accepted"] + 1) > 0.25
            ):
                reasons["public_layout_quota"] += 1
                continue
            if not _geometry_compatible(components, environment.state.plan, annotation, frame):
                reasons["geometry_incompatible"] += 1
                continue
            evaluator = PrivateEvaluator(
                components.judge,
                components.artifacts,
                {task.task_id: evaluation},
                components.observer.tools,
            )
            target_scores = evaluator.score_image(task, target_id)
            if not all(score.known and score.passed for score in target_scores):
                reasons["target_requirement_failure"] += 1
                continue
            target_image = components.artifacts.get_image(target_id)
            plan_score = components.judge.check(
                atom_id="private:plan_compatibility",
                request=task.user_request.text,
                property_text="The image is compatible with this proposed image plan: "
                + json.dumps(environment.state.plan.model_dump(mode="json")),
                images=[target_image],
                roles="paired supervision target",
                kind="semantic",
            )
            if not plan_score.known or plan_score.value < plan_score.threshold:
                reasons["plan_target_semantics"] += 1
                continue
            plan_path = output / "plans" / f"{report['accepted']:06d}.pt"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample.state_dict(), plan_path)
            renderer_records.append(
                {
                    "sample_id": row["sample_id"],
                    "family_id": row["family_id"],
                    "task_id": task.task_id,
                    "bundle_id": bundle.bundle_id,
                    "bundle_path": str(
                        (
                            components.compiler.directory
                            / f"{bundle.bundle_id.replace(':', '_')}.pt"
                        ).resolve()
                    ),
                    "condition_recipe": {
                        "plan": environment.state.plan.model_dump(mode="json"),
                        "source_image_id": source_id,
                    },
                    "target_image_id": target_id,
                    "valid_mask_id": valid_id,
                    "edit_mask_id": edit_id,
                    "plan_action_path": str(plan_path.resolve()),
                    "graph_revision": environment.state.manifest.revision_id,
                    "source_dataset": row["source_dataset"],
                    "region_count": regions,
                    "two_changed_entities": bool(annotation.get("two_changed_entities", False)),
                    "verification_provenance": annotation["provenance"],
                }
            )
            accepted[key] += 1
            appearance_accepted[key] += int(bool(appearance))
            report["accepted"] += 1
            public_layout_count += int(bool(annotation.get("public_controls")))
            tasks.append(task.model_dump(mode="json"))
            private_tasks.append(
                {"task_id": task.task_id, "evaluation": evaluation.model_dump(mode="json")}
            )
            if accepted[key] % 2 == 0:
                actor_records.append(
                    save_action_record(
                        components,
                        environment,
                        sample.text,
                        output / "snapshots",
                        f"aligned:{row['sample_id']}",
                        0,
                        f"root:{row['sample_id']}",
                        {
                            "source": "paired_target_postfilter",
                            "annotation": annotation["provenance"],
                        },
                        snapshot,
                        sample,
                    )
                )
            for name, records in (
                ("actor_records", actor_records),
                ("renderer_records", renderer_records),
                ("public_tasks", tasks),
                ("private_tasks", private_tasks),
            ):
                append_jsonl(output / f"{name}.jsonl", records[written[name] :])
                written[name] = len(records)
        except (ValueError, KeyError, ContractError) as error:
            reasons[f"contract:{str(error).split(':')[0][:80]}"] += 1
    report["actor_records"] = len(actor_records)
    report["rejections"] = dict(reasons)
    report["actual_by_bucket"] = {
        f"{source}:{regions}": accepted[(source, regions)]
        for source in quotas
        for regions in range(3)
    }
    report["shortfalls"] = {
        f"{source}:{regions}": max(0, quotas[source][regions] - accepted[(source, regions)])
        for source in quotas
        for regions in range(3)
    }
    report["unique_families"] = len({row["family_id"] for row in renderer_records})
    report["public_layout_samples"] = public_layout_count
    two_region_rows = [row for row in renderer_records if row["region_count"] == 2]
    report["two_changed_entities"] = {
        "count": sum(row["two_changed_entities"] for row in two_region_rows),
        "two_region_samples": len(two_region_rows),
    }
    report["appearance_by_bucket"] = {
        f"{source}:{regions}": appearance_accepted[(source, regions)]
        for source in quotas
        for regions in range(3)
    }
    report["desired_appearance_by_bucket"] = {
        f"{source}:{regions}": round(quotas[source][regions] * 0.1)
        for source in quotas
        for regions in range(3)
    }
    for name, records in (
        ("actor_records", actor_records),
        ("renderer_records", renderer_records),
        ("public_tasks", tasks),
        ("private_tasks", private_tasks),
    ):
        write_jsonl(output / f"{name}.jsonl", records)
    write_report(output / "report.json", report)
    return report
