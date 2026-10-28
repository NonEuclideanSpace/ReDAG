import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode

import httpx

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.data.families import HammingIndex
from agentic_image.data.io import append_jsonl, read_jsonl, write_jsonl, write_report
from fetch_viewer_heldout import REPOSITORIES, download, request


def candidate_types(text):
    value = text.lower()
    types = ["semantic"]
    if re.search(r"\b(exactly|only|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", value):
        types.append("count")
    if re.search(r"\b(text|word|words|written|letter|letters|says|reads|spell)\b", value):
        types.append("text")
    for token, route in (("left of", "left_of"), ("right of", "right_of"),
                         ("above", "above"), ("below", "below"), ("inside", "inside")):
        if token in value:
            types.append(route)
    return types


def lookup(client, source, offset, rows, directory):
    path = directory / f"rows_{source}_{offset}.json"
    if path.exists() and time.time() - path.stat().st_mtime < 1800:
        payload = json.loads(path.read_text())
    else:
        url = "https://datasets-server.huggingface.co/rows?" + urlencode({
            "dataset": REPOSITORIES[source], "config": "default", "split": "train",
            "offset": offset, "length": 100,
        })
        payload = request(client, url).json()
        path.write_text(json.dumps(payload))
    result = {}
    requested = {row["sample_id"]: row for row in rows}
    for entry in payload.get("rows", []):
        row = entry["row"]
        sample = f"magicbrush:{row['img_id']}:{row['turn_index']}" if source == "magicbrush" else "omniedit:" + row["omni_edit_id"]
        if sample not in requested:
            continue
        instruction = row["instruction"] if source == "magicbrush" else max(
            (text for text in row["edited_prompt_list"] if isinstance(text, str) and text.strip()),
            key=len, default="")
        if entry.get("truncated_cells") or instruction != requested[sample]["instruction"]:
            raise ValueError("Official viewer row does not match the original request exactly")
        result[sample] = {**row, "_viewer_row_index": entry["row_idx"]}
    if set(result) != set(requested):
        raise ValueError("Exact official viewer lookup is missing requested sample identities")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/families.protocol.jsonl")
    parser.add_argument("--output", default="data/calibration/ai_reserved")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if not 1 <= arguments.limit <= 600:
        raise ValueError("The existing calibration reservation contains exactly 600 families")
    output = Path(arguments.output)
    (output / "api").mkdir(parents=True, exist_ok=True)
    manifest = list(read_jsonl(arguments.manifest))
    groups = defaultdict(list)
    for row in manifest:
        if row["split_id"] == "calibration":
            if row["official_split"] != "train":
                raise ValueError("Official dev/test cannot enter calibration")
            groups[row["family_id"]].append(row)
    if len(groups) != 600:
        raise ValueError("The saved reservation must contain exactly 600 distinct families")
    families = sorted(groups, key=lambda family: sha256(("ai-calibration-v1:" + family).encode()).hexdigest())
    reserved = []
    for index, family in enumerate(families):
        row = sorted(groups[family], key=lambda row: (row.get("metadata", {}).get("turn_index", 1), row["sample_id"]))[0]
        reserved.append({**row, "calibration_index": index,
                         "calibration_fold": "selection" if index < 400 else "confirmation"})
    frozen = output / "reserved_600.jsonl"
    if frozen.exists() and list(read_jsonl(frozen)) != reserved:
        raise ValueError("The saved 400/200 family assignment must not be changed")
    if not frozen.exists():
        write_jsonl(frozen, reserved)
    store = ArtifactStore("data/artifacts")
    images = {row["original_artifact_id"]: row for row in read_jsonl("data/viewer/images.jsonl")
              if row["status"] == "downloaded"}
    official = {row[field] for row in manifest if row["official_split"] in {"dev", "test"}
                for field in ("source_image_id", "target_image_id")}
    if official - images.keys():
        raise ValueError("Full official held-out fingerprint evidence must already be available")
    external = HammingIndex()
    for identifier in official:
        external.add(int(images[identifier]["phash"]), "official:" + identifier)
    train_sources = {row["user_request"]["source_image_id"] for row in read_jsonl("data/public/protocol_pilot_tasks.jsonl")}
    for row in images.values():
        if row["artifact_id"] in train_sources:
            external.add(int(row["phash"]), "exposed_train:" + row["artifact_id"])
    for row in read_jsonl("data/viewer/training_candidate_images.jsonl"):
        external.add(int(row["phash"]), "exposed_train:" + row["artifact_id"])
    ready_path = output / "prepared_families.jsonl"
    ready = {row["sample_id"]: row for row in read_jsonl(ready_path)} if ready_path.exists() else {}
    selected = reserved[:arguments.limit]
    pending = [row for row in selected if row["sample_id"] not in ready]
    started = time.monotonic()
    total_bytes, failures = 0, []
    with httpx.Client(follow_redirects=True, timeout=45, limits=httpx.Limits(max_connections=16)) as client:
        jobs = []
        for source in ("magicbrush", "omniedit"):
            source_rows = [row for row in manifest if row["source_dataset"] == source and row["official_split"] == "train"]
            positions = {row["sample_id"]: index for index, row in enumerate(source_rows)}
            pages = defaultdict(list)
            for row in pending:
                if row["source_dataset"] == source:
                    pages[positions[row["sample_id"]] // 100 * 100].append(row)
            jobs.extend((source, offset, items) for offset, items in sorted(pages.items()))
        viewers = {}
        with ThreadPoolExecutor(max_workers=min(4, arguments.workers)) as executor:
            futures = {executor.submit(lookup, client, source, offset, rows, output / "api"): rows for source, offset, rows in jobs}
            for future in as_completed(futures):
                try:
                    viewers.update(future.result())
                except Exception as error:
                    failures.append({"stage": "metadata", "samples": [row["sample_id"] for row in futures[future]], "error": str(error)})
                print(json.dumps({"stage": "metadata", "matched": len(viewers), "failed_batches": len(failures), "seconds": time.monotonic() - started}), flush=True)

        def prepare(row):
            viewer = viewers[row["sample_id"]]
            columns = ("source_img", "target_img") if row["source_dataset"] == "magicbrush" else ("src_img", "edited_img")
            restored = []
            for role, column, field in zip(("source", "target"), columns, ("source_image_id", "target_image_id")):
                original = store.get_ref(row[field])
                item = images.get(row[field])
                if item is not None:
                    try:
                        store.get_image(item["artifact_id"])
                    except (FileNotFoundError, OSError, ValueError, KeyError):
                        item = None
                if item is None:
                    metadata = {"sample_id": row["sample_id"], "artifact_id": original.artifact_id,
                                "value_id": original.value_id, "revision": row["license_record"]["revision"]}
                    item = download(client, store, [(row["source_dataset"], "train", viewer, column, metadata)])[0]
                else:
                    item = {**item, "bytes_downloaded": 0}
                restored.append({**item, "candidate_role": role, "review_image_url": viewer[column]["src"],
                                 "external_near_matches": external.query(int(item["phash"]), radius=6)})
            return {"sample_id": row["sample_id"], "family_id": row["family_id"],
                    "calibration_index": row["calibration_index"], "calibration_fold": row["calibration_fold"],
                    "source_dataset": row["source_dataset"], "request": row["instruction"],
                    "license_record": row["license_record"], "images": restored,
                    "verified_viewer_row_index": viewer["_viewer_row_index"],
                    "candidate_checker_types": candidate_types(row["instruction"]),
                    "independent_eval_atoms": [], "human_reviewed": False, "ai_review_complete": False,
                    "source_correct": None, "target_correct": None,
                    "quarantined": any(image["external_near_matches"] for image in restored)}

        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            futures = {executor.submit(prepare, row): row for row in pending if row["sample_id"] in viewers}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    prepared = future.result()
                    ready[row["sample_id"]] = prepared
                    total_bytes += sum(image["bytes_downloaded"] for image in prepared["images"])
                    append_jsonl(ready_path, [prepared])
                except Exception as error:
                    failures.append({"stage": "image", "sample_id": row["sample_id"], "error": str(error)})
                print(json.dumps({"stage": "images", "prepared_total": len(ready), "download_bytes_this_run": total_bytes,
                                  "seconds": round(time.monotonic() - started, 2), "failures": len(failures)}), flush=True)
    pairs = sorted((ready[row["sample_id"]] for row in selected if row["sample_id"] in ready), key=lambda row: row["calibration_index"])
    local = HammingIndex()
    conflicts = set()
    for row in pairs:
        for item in row["images"]:
            matches = set(local.query(int(item["phash"]), radius=6)) - {row["family_id"]}
            if matches:
                conflicts.update(matches | {row["family_id"]})
            local.add(int(item["phash"]), row["family_id"])
    records, sections = [], []
    for row in pairs:
        row["quarantined"] = row["quarantined"] or row["family_id"] in conflicts
        row["reserved_family_near_conflict"] = row["family_id"] in conflicts
        source = row["images"][0]["artifact_id"]
        for item in row["images"]:
            records.append({"candidate_id": row["sample_id"] + ":" + item["candidate_role"],
                            "sample_id": row["sample_id"], "family_id": row["family_id"],
                            "calibration_fold": row["calibration_fold"], "request": row["request"],
                            "source_image_id": source, "candidate_image_id": item["artifact_id"],
                            "candidate_role": item["candidate_role"], "correct": None,
                            "known": None, "ai_review_complete": False, "human_reviewed": False,
                            "independent_eval_atoms": [], "quarantined": row["quarantined"]})
        images_html = "".join(f'<figure><img loading="lazy" src="{escape(item["review_image_url"], quote=True)}"><figcaption>{escape(item["candidate_role"])}</figcaption></figure>' for item in row["images"])
        sections.append(f'<section><h2>{row["calibration_index"] + 1}. {escape(row["sample_id"])}</h2><p>{escape(row["request"])}</p><p>{row["calibration_fold"]} · {"隔离待复核" if row["quarantined"] else "等待实际 AI 视觉标注"}</p><div>{images_html}</div></section>')
    write_jsonl(output / f"batch_{arguments.limit:03d}.jsonl", pairs)
    write_jsonl(output / f"score_candidates_{arguments.limit:03d}.jsonl", records)
    report = {"status": "prepared_pending_ai_review" if len(pairs) == arguments.limit else "partial_preparation_failed",
              "reserved_families": 600, "selection": 400, "confirmation": 200,
              "requested": arguments.limit, "prepared": len(pairs), "candidate_images": len(records),
              "quarantined_families": sum(row["quarantined"] for row in pairs),
              "failures": failures, "download_bytes_this_run": total_bytes,
              "elapsed_seconds": time.monotonic() - started, "ai_labels_available": 0,
              "human_labels_available": 0, "calibration_complete": False,
              "checker_types_are_only_unconfirmed_hints": True,
              "candidate_type_counts": dict(Counter(t for row in pairs for t in row["candidate_checker_types"])),
              "unfetched_reserved_development_diagnostic_visual_isolation_complete": False,
              "manifest": arguments.manifest, "manifest_digest": sha256(Path(arguments.manifest).read_bytes()).hexdigest()}
    write_report(output / f"batch_{arguments.limit:03d}.report.json", report)
    (output / f"batch_{arguments.limit:03d}.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AI 校准候选</title><style>body{font:16px system-ui;margin:24px;background:#f4f5f7}section{background:white;padding:24px;margin:20px 0}section>div{display:flex;gap:20px}figure{margin:0;flex:1}img{width:100%;max-width:600px;max-height:600px;object-fit:contain}p{line-height:1.5}</style><h1>真实校准候选：未标注</h1><p>既有600个独立预留家族，固定400选择/200确认。源图和目标图都仅为待实际检查的候选；没有默认正负标签，没有8B自造真值。所有标签来源须明确为AI，human=false。图片引用固定版本官方查看器，实际像素已落盘；URL可能到期，届时使用保存的artifact。</p>' + ''.join(sections) + '</html>')
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}), flush=True)
    if len(pairs) != arguments.limit:
        raise RuntimeError("The requested calibration batch was not fully prepared; inspect its report")


if __name__ == "__main__":
    main()
