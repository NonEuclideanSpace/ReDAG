import argparse
import io
import json
from pathlib import Path
import tarfile
import time

from PIL import Image

from agentic_image.core.artifacts import ArtifactStore


def add_bytes(archive, name, payload):
    item = tarfile.TarInfo(name)
    item.size = len(payload)
    item.mtime = int(time.time())
    archive.addfile(item, io.BytesIO(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/sampler-expanded-200")
    parser.add_argument("--artifacts", default="data/artifacts")
    parser.add_argument("--group", choices=("base", "short", "all"), default="base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", help="Explicit registered methods for a separate comparison run")
    parser.add_argument("--packed-state", default="deployment/sampler-blind-packed-ids.json")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    run = Path(args.run)
    records = json.loads((run / "blinded_quality_review" / "records.json").read_text())
    private_key = {row["review_id"]: row for row in json.loads((run / "quality_review_key.private.json").read_text())}
    accepted = {"ode50", "sde50"} if args.group == "base" else {"sde28", "sde20"} if args.group == "short" else {"ode50", "sde50", "sde28", "sde20"}
    if args.methods:
        accepted = set(args.methods.split(","))
        if not accepted or accepted - {row["method"] for row in private_key.values()}:
            raise ValueError("Requested review methods are absent from this run registration")
    state = Path(args.packed_state)
    packed = set(json.loads(state.read_text())) if state.is_file() else set()
    selected = [row for row in records if row["review_id"] not in packed and private_key[row["review_id"]]["method"] in accepted][:args.limit]
    if not selected:
        print(json.dumps({"status": "no_new_images", "new_images": 0}), flush=True)
        return
    store = ArtifactStore(args.artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Use a new archive path; an already prepared transfer may still be needed")
    sources = set()
    transfer = []
    prefix = "sampler-blind-review"
    with tarfile.open(output, "w:gz") as archive:
        for row in selected:
            preview = Path(row["preview_path"])
            with Image.open(preview) as image:
                image.verify()
            candidate_name = f"candidates/{row['review_id']}.jpg"
            archive.add(preview, arcname=f"{prefix}/{candidate_name}", recursive=False)
            source_id = row["source_image_id"]
            source_name = f"sources/{source_id.split(':')[-1]}.jpg"
            if source_id not in sources:
                image = store.get_image(source_id)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90, optimize=True)
                add_bytes(archive, f"{prefix}/{source_name}", buffer.getvalue())
                sources.add(source_id)
            transfer.append({key: row[key] for key in ("review_id", "image_id", "candidate_image_value_id", "request", "source_image_id")})
            transfer[-1].update(candidate_path=candidate_name, source_path=source_name,
                severe_defect=None, known=False, review_type="ai", human_reviewed=False,
                review_evidence=None, reviewed_image_paths=[],
                display_provenance={"candidate":"JPEG quality85 preview of the actual512 render; original PNG retained remotely", "source":"JPEG quality90 from the actual public source artifact", "display_only":True})
        manifest_name = output.stem.replace('.tar', '') + '.records.jsonl'
        payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in transfer).encode()
        add_bytes(archive, f"{prefix}/manifests/{manifest_name}", payload)
    packed.update(row['review_id'] for row in selected)
    state.parent.mkdir(parents=True, exist_ok=True)
    temporary = state.with_suffix('.tmp')
    temporary.write_text(json.dumps(sorted(packed)) + '\n')
    temporary.replace(state)
    print(json.dumps({"status":"prepared", "archive":str(output.resolve()), "bytes":output.stat().st_size,
                      "new_images":len(transfer), "source_images":len(sources), "total_packed":len(packed),
                      "manifest":f"{prefix}/manifests/{manifest_name}", "known_labels":0,
                      "omitted":["method", "seed", "private_key", "judge_score", "training_loss", "PLAN"]}), flush=True)


if __name__ == "__main__":
    main()
