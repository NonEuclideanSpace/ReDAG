import argparse
import json
from pathlib import Path

from datasets import load_dataset

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.data.ingest import SOURCES, store_edit_mask
from agentic_image.data.io import write_jsonl, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--files", default="deployment/magicbrush-files.json")
    parser.add_argument("--artifacts", default="data/artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = SOURCES["magicbrush"]
    manifest = json.loads(Path(args.files).read_text())
    if manifest["revision"] != source["revision"]:
        raise ValueError("The mask manifest must match the imported dataset revision")
    files = [
        f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{name}"
        for name in manifest["files"]
        if name.startswith(f"data/{args.split}-")
    ]
    if not files:
        raise ValueError("No official parquet files were found for this split")
    artifacts = ArtifactStore(args.artifacts)
    print(f"Opening official {args.split} mask columns", flush=True)
    dataset = load_dataset(
        "parquet",
        data_files={args.split: files},
        split=args.split,
        streaming=True,
        columns=["img_id", "turn_index", "mask_img"],
    )
    report = {"split": args.split, "revision": source["revision"], "rows": 0, "empty": 0}
    print(f"Reading official {args.split} mask columns", flush=True)

    def records():
        for row in dataset:
            identifier = f"{row['img_id']}:{row['turn_index']}"
            provenance = {
                "dataset": source["dataset"],
                "revision": source["revision"],
                "record": identifier,
                "official_split": args.split,
            }
            mask = row["mask_img"]
            mask_id = store_edit_mask(mask, artifacts, provenance) if mask is not None else None
            report["rows"] += 1
            report["empty"] += mask_id is None
            if report["rows"] % 100 == 0:
                print(f"Prepared {report['rows']} {args.split} masks", flush=True)
            yield {"sample_id": f"magicbrush:{identifier}", "edit_mask_id": mask_id}

    write_jsonl(args.output, records())
    write_report(Path(args.output).with_suffix(".report.json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
