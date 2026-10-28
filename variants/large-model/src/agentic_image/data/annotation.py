from html import escape
from pathlib import Path

from .io import read_jsonl, write_jsonl, write_report


def prepare_annotation_packet(candidates_path, output_dir, artifacts, limit=200, split="train"):
    if limit < 1:
        raise ValueError("Annotation packet size must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "annotations.jsonl").exists():
        raise FileExistsError("An existing annotation packet must not be overwritten")
    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    annotations, sections = [], []
    for row in read_jsonl(candidates_path):
        if len(annotations) >= limit:
            break
        if row.get("split_id") != split:
            continue
        index = len(annotations)
        images = []
        for role, image_id in (
            ("Source", row.get("source_image_id")),
            ("Private target", row["target_image_id"]),
        ):
            if image_id is None:
                continue
            image = artifacts.get_image(image_id)
            image.thumbnail((512, 512))
            name = f"{index:04d}_{role.lower().replace(' ', '_')}.png"
            image.save(previews / name)
            images.append(
                f'<figure><img src="previews/{name}" alt="{role}"><figcaption>{role}</figcaption></figure>'
            )
        sections.append(
            f"<section><h2>{escape(row['sample_id'])}</h2><p>{escape(row['instruction'])}</p><div>{''.join(images)}</div><p>Review completion, unchanged content, registration, entity identity, and reliable region annotations. Edit the corresponding JSONL row only after checking the images.</p></section>"
        )
        annotations.append(
            {
                "sample_id": row["sample_id"],
                "verified": False,
                "semantic_compatible": False,
                "registration_verified": False,
                "geometry_verified": False,
                "complete_caption_verified": False,
                "benchmark_derived": False,
                "provenance": {
                    "annotator": "",
                    "reviewer": "",
                    "annotation_date": "",
                    "source": "independent_manual_review",
                },
                "entities": [],
                "box_frame": "original",
                "reliable_edit_mask_id": row.get("edit_mask_id"),
                "protected_mask_id": None,
                "public_controls": {},
                "evaluation": {
                    "private_task_key": row["sample_id"],
                    "eval_atoms": [],
                    "original_protected_mask_id": None,
                    "source_baseline_observations": [],
                    "annotation_provenance": {"preserved_eval_ids": []},
                    "evaluator_version": "independent-annotations-v1",
                },
            }
        )
    style = "body{font:16px system-ui;margin:32px;background:#f4f5f6;color:#15202b}section{background:white;padding:24px;margin:24px 0;border-radius:12px}section>div{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0}img{max-width:45vw;max-height:512px}p{max-width:1000px;line-height:1.5}figcaption{padding:10px 0;font-weight:600}"
    html = f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent paired-data annotation</title><style>{style}</style><body><h1>Independent paired-data annotation</h1><p>These target images and evaluation labels are private supervision. They must never be passed to the controller or used to rewrite its proposed plan. Fill eval_atoms using the original user requirements. Each atom needs eval_id, raw_span, predicate, expected_value, checker_route, threshold, and mandatory. Leave uncertain samples unverified.</p>{"".join(sections)}</body></html>'
    (output / "index.html").write_text(html)
    write_jsonl(output / "annotations.jsonl", annotations)
    report = {
        "samples": len(annotations),
        "split": split,
        "preview": str((output / "index.html").resolve()),
        "annotations": str((output / "annotations.jsonl").resolve()),
        "status": "requires_independent_annotation",
    }
    write_report(output / "report.json", report)
    return report
