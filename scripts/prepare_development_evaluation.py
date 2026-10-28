import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.types import CheckSpec, PublicCheckAtom, PublicTask, RequestSpan, UserRequest


def read(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def lines(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/evaluation/development64')
    parser.add_argument('--publish-chunk', type=int, default=8)
    args = parser.parse_args()
    if args.publish_chunk < 1:
        parser.error('--publish-chunk must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read('data/families.protocol.jsonl')
    groups = defaultdict(list)
    for row in manifest:
        if row['split_id'] == 'development':
            groups[row['family_id']].append(row)
    if len(groups) != 64:
        raise ValueError('The original development reservation must contain exactly 64 families')
    families = sorted(groups, key=lambda family: sha256(('fixed-development64-v1:' + family).encode()).hexdigest())
    selected = [{**min(groups[family], key=lambda row: (row.get('metadata', {}).get('turn_index', 1), row['sample_id'])), 'evaluation_index': index} for index, family in enumerate(families)]
    frozen = output / 'reserved.frozen.jsonl'
    if frozen.exists() and read(frozen) != selected:
        raise ValueError('The fixed development task assignment must not change')
    lines(frozen, selected)
    outside = {row['family_id'] for row in manifest if row['split_id'] in {'train', 'calibration', 'diagnostic-test', 'official-dev', 'official-test'}}
    if set(families) & outside:
        raise ValueError('Development families overlap another registered split')
    store = ArtifactStore('data/artifacts')
    all_values = {store.get_ref(row[field]).value_id for family in families for row in groups[family] for field in ('source_image_id', 'target_image_id')}
    blocked_values = {store.get_ref(row[field]).value_id for row in manifest if row['family_id'] in outside for field in ('source_image_id', 'target_image_id')}
    if all_values & blocked_values:
        raise ValueError('Development has exact image-value leakage into another protected split')
    assets = {}
    for row in read('data/training_expansion/isolation/supplementary_images.jsonl'):
        if Path(row['local_image_path']).is_file():
            assets[row['original_value_id']] = row
    for row in read('data/calibration/ai_reserved/batch_600.jsonl'):
        for image in row['images']:
            path = image.get('local_review_path', f"data/calibration/ai_reserved/local_images/{row['calibration_index']:03d}_{image['candidate_role']}.jpg")
            if Path(path).is_file():
                assets.setdefault(image['original_value_id'], {**image, 'local_image_path': path})
    prepared = {row['sample_id']: row for row in read(output / 'prepared.jsonl')}
    for row in selected:
        if row['sample_id'] in prepared:
            continue
        original_refs = {role: store.get_ref(row[field]) for role, field in (('source', 'source_image_id'), ('target', 'target_image_id'))}
        if any(ref.value_id not in assets for ref in original_refs.values()):
            continue
        images = []
        for role, ref in original_refs.items():
            asset = assets[ref.value_id]
            with Image.open(asset['local_image_path']) as image:
                if image.size != (asset['width'], asset['height']):
                    raise ValueError('The cached official image dimensions changed')
                materialized = store.put_image(image, metadata={'dataset_profile': 'official-viewer-reencoded-v1', 'original_artifact_id': ref.artifact_id, 'sample_id': row['sample_id'], 'role': role if role == 'source' else 'private_target', 'viewer_revision': asset['viewer_revision'], 'original_revision': row['license_record']['revision']})
            images.append({'candidate_role': role, 'original_artifact_id': ref.artifact_id, 'original_value_id': ref.value_id, 'artifact_id': materialized.artifact_id, 'value_id': materialized.value_id, 'relative_path': materialized.relative_path, 'phash': int(asset['phash']), 'viewer_revision': asset['viewer_revision'], 'local_image_path': asset['local_image_path']})
        prepared[row['sample_id']] = {'evaluation_index': row['evaluation_index'], 'sample_id': row['sample_id'], 'family_id': row['family_id'], 'source_dataset': row['source_dataset'], 'request': row['instruction'], 'split_id': 'development', 'images': images, 'training_eligible': False, 'calibration_eligible': False, 'visual_success_labels': None}
    ready = sorted(prepared.values(), key=lambda row: row['evaluation_index'])
    lines(output / 'prepared.jsonl', ready)
    by_sample = {row['sample_id']: row for row in selected}
    tasks = {}
    for row in ready:
        original = by_sample[row['sample_id']]
        text = row['request']
        source = next(image for image in row['images'] if image['candidate_role'] == 'source')
        task = PublicTask(task_id='development64:' + row['sample_id'], family_id=row['family_id'], split_id='development', source_dataset=row['source_dataset'], license_record={**original['license_record'], 'dataset_profile': 'official-viewer-reencoded-v1', 'purpose': 'fixed_development_evaluation_only'}, user_request=UserRequest(text=text, source_image_id=source['artifact_id'], spans=(RequestSpan(span_id='span:request', start=0, end=len(text), text=text),)), task_mode='edit', check_spec=CheckSpec(extractor_version='public-protocol-only', fallback=True, atoms=(PublicCheckAtom(check_id='check:request', raw_span_ids=('span:request',), predicate=text, expected_value=text, confidence=0.0),)))
        tasks[row['sample_id']] = task.model_dump(mode='json')
    chunks = []
    for start in range(0, 64, args.publish_chunk):
        chunk = selected[start:start + args.publish_chunk]
        if not all(row['sample_id'] in tasks for row in chunk):
            continue
        path = output / f'public_tasks_{start:03d}_{start + len(chunk) - 1:03d}.jsonl'
        values = [tasks[row['sample_id']] for row in chunk]
        if path.exists() and read(path) != values:
            raise ValueError('A published evaluation chunk changed')
        lines(path, values)
        chunks.append(str(path))
    near_matches = []
    for i, left in enumerate(ready):
        for right in ready[i + 1:]:
            matches = [(a['candidate_role'], b['candidate_role'], (a['phash'] ^ b['phash']).bit_count()) for a in left['images'] for b in right['images'] if (a['phash'] ^ b['phash']).bit_count() <= 6]
            if matches:
                near_matches.append({'left_family': left['family_id'], 'right_family': right['family_id'], 'matches': matches, 'adjudicated': False})
    lines(output / 'within_cohort_near_matches.jsonl', near_matches)
    transfer = set()
    for row in ready:
        for image in row['images']:
            transfer.update([str(store.root / 'refs' / (image['artifact_id'].split(':')[1] + '.json')), str(store.root / image['relative_path'])])
    (output / 'transfer_files.txt').write_text('\n'.join(sorted(transfer)) + '\n')
    if len(tasks) == 64:
        lines(output / 'public_tasks.jsonl', [tasks[row['sample_id']] for row in selected])
    report = {'status': 'assets_complete' if len(tasks) == 64 else 'waiting_for_existing_downloads', 'fixed_families': 64, 'prepared_families': len(tasks), 'source_counts': dict(Counter(row['source_dataset'] for row in selected)), 'registered_seeds': [11, 23, 47, 73, 101], 'expected_outputs_per_configuration': 320, 'published_chunks': chunks, 'training_family_overlap': 0, 'calibration_family_overlap': 0, 'exact_value_overlap_with_protected_splits': 0, 'unadjudicated_within_cohort_near_matches': len(near_matches), 'cross_split_visual_audit': 'data/training_expansion/isolation', 'visual_evaluation_ready': False, 'visual_labels': None, 'official16_is_separate_cohort': True, 'new_download_requests': 0}
    (output / 'preparation.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
