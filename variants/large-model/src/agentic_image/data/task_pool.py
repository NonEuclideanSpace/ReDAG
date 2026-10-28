from collections import Counter
from pathlib import Path

from agentic_image.checks.public import PublicTaskBuilder
from agentic_image.core.geometry import letterbox, make_frame, transform_mask
from agentic_image.core.types import PrivateEvaluation

from .io import read_jsonl, write_jsonl, write_report


def build_task_pool(
    components, candidates_path, annotations_path, output_dir, split="development", limit=None
):
    if split not in {
        "train",
        "development",
        "calibration",
        "diagnostic-test",
        "official-dev",
        "official-test",
    }:
        raise ValueError("Task pools require a preassigned family split")
    if limit is not None and limit < 1:
        raise ValueError("Task pool limit must be positive")
    components.load_judge(private=False)
    builder = PublicTaskBuilder(components.judge, components.artifacts)
    annotations = {row["sample_id"]: row for row in read_jsonl(annotations_path)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "public_tasks.jsonl").exists():
        raise FileExistsError("Task pool output must be a new directory")
    public_tasks, private_tasks = [], []
    rejected = Counter()
    canvas_hw = tuple(components.config["canvas"])
    for row in read_jsonl(candidates_path):
        if limit is not None and len(public_tasks) >= limit:
            break
        if row.get("split_id") != split:
            continue
        annotation = annotations.get(row["sample_id"], {})
        if (
            not annotation.get("verified")
            or not annotation.get("provenance")
            or not annotation.get("evaluation")
        ):
            rejected["requires_independent_task_annotation"] += 1
            continue
        if split == "train" and annotation.get("benchmark_derived"):
            rejected["benchmark_derived_training_task"] += 1
            continue
        evaluation = PrivateEvaluation.model_validate(annotation["evaluation"])
        if not evaluation.eval_atoms:
            raise ValueError(
                "An annotated task must retain a nonempty fixed evaluation specification"
            )
        source_id = row.get("source_image_id")
        source_frame = None
        if source_id:
            source = components.artifacts.get_image(source_id)
            source_frame = make_frame(
                (source.height, source.width), canvas_hw, f"frame:{row['sample_id']}"
            )
            source_id = components.artifacts.put_image(
                letterbox(source, source_frame),
                frame=source_frame,
                producer_id=row["sample_id"],
                metadata={"role": "source", "original_image_id": source_id},
            ).artifact_id
        protected = annotation.get("protected_mask_id", evaluation.original_protected_mask_id)
        if protected:
            if source_frame is None:
                raise ValueError("A protected image region requires an original source image")
            mask = components.artifacts.get_array(protected)
            if (
                mask.shape == source_frame.source_hw
                and annotation.get("mask_frame", "original") == "original"
            ):
                mask = transform_mask(mask, source_frame, "soft")
            elif mask.shape != canvas_hw:
                raise ValueError("Protected mask is not registered to the source canvas")
            protected = components.artifacts.put_array(
                mask, frame=source_frame, metadata={"role": "private_protected_mask"}
            ).artifact_id
            evaluation = evaluation.model_copy(update={"original_protected_mask_id": protected})
        public_controls = dict(annotation.get("public_controls", {}))
        if row.get("metadata", {}).get("public_appearance"):
            public_controls["appearance_reference"] = row["metadata"]["public_appearance"]
        task = builder.build(
            task_id=row["sample_id"],
            text=row["instruction"],
            source_image_id=source_id,
            appearance_image_ids=tuple(row.get("appearance_image_ids", [])),
            controls=public_controls,
            family_id=row["family_id"],
            split_id=split,
            source_dataset=row.get("source_dataset", "curated"),
            license_record=row.get("license_record", {}),
        )
        public_tasks.append(task.model_dump(mode="json"))
        private_tasks.append(
            {"task_id": task.task_id, "evaluation": evaluation.model_dump(mode="json")}
        )
    write_jsonl(output / "public_tasks.jsonl", public_tasks)
    write_jsonl(output / "private_tasks.jsonl", private_tasks)
    report = {
        "split": split,
        "tasks": len(public_tasks),
        "unique_families": len({row["family_id"] for row in public_tasks}),
        "rejections": dict(rejected),
        "selection": "preassigned_family_partition_and_independent_annotations_only",
    }
    write_report(output / "report.json", report)
    return report
