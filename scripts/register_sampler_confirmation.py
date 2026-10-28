from collections import defaultdict, Counter
import json
from pathlib import Path

from agentic_image.core.types import CheckSpec, PublicCheckAtom, PublicTask, RequestSpan, UserRequest


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


root = Path('data/evaluation/sampler_confirm64')
root.mkdir(parents=True, exist_ok=True)
rows = read('data/families.protocol.jsonl')
old = read('data/evaluation/development64/reserved.frozen.jsonl')
protected = {'train', 'train-protocol-only', 'calibration', 'development', 'diagnostic-test', 'official-test'}
excluded = {row['family_id'] for row in rows if row['split_id'] in protected}
excluded.update(row['family_id'] for row in read('data/training_expansion/training_selection_v1.frozen.jsonl'))
for path in Path('data/training_expansion').glob('candidates*/candidates.frozen.jsonl'):
    excluded.update(row['family_id'] for row in read(path))
excluded.update(row['family_id'] for row in json.loads(Path('runs/protocol-evaluation/registration.json').read_text())['tasks'])
for path in Path('data/calibration/ai_reserved').glob('replacements*/reserved_replacements.frozen.jsonl'):
    excluded.update(row['family_id'] for row in read(path))
for path in Path('data/preflight').glob('*task*.jsonl'):
    excluded.update(row['family_id'] for row in read(path) if 'family_id' in row)
index_paths = ['data/training_expansion/isolation/guarded_value_index.jsonl',
    'data/training_expansion/isolation/expanded_family_value_index.jsonl']
by_value, by_family = {}, defaultdict(dict)
for path in index_paths:
    for row in read(path):
        value, fingerprint = row['original_value_id'], row['phash']
        if value in by_value and by_value[value]['phash'] != fingerprint:
            raise ValueError('Existing fingerprint profiles disagree for the same original value')
        by_value.setdefault(value, {'phash': fingerprint, 'families': set()})['families'].update(row['family_ids'])
        for family in row['family_ids']:
            by_family[family][value] = fingerprint
guard = {value: item for value, item in by_value.items() if item['families'] & excluded}
groups = defaultdict(list)
for row in rows:
    if row['split_id'] == 'official-dev' and row['family_id'] not in excluded:
        groups[row['family_id']].append(row)
pools = defaultdict(list)
for family in sorted(groups):
    row = min(groups[family], key=lambda row: (row.get('metadata', {}).get('turn_index', 0), row['sample_id']))
    pools[row['source_dataset'], row['metadata'].get('task_category', 'unspecified')].append(row)
selected, attempts = [], []
chosen_values = {}
for index, previous in enumerate(old):
    stratum = previous['source_dataset'], previous['metadata'].get('task_category', 'unspecified')
    while pools[stratum]:
        row = pools[stratum].pop(0)
        family = row['family_id']
        values = by_family.get(family, {})
        required_values = set()
        for member in groups[family]:
            for field in ('source_image_id', 'target_image_id'):
                ref = json.loads((Path('data/artifacts/refs') / (member[field].split(':')[1] + '.json')).read_text())
                required_values.add(ref['value_id'])
        missing = sorted(required_values - set(values))
        matches = []
        for value in sorted(required_values - set(missing)):
            fingerprint = values[value]
            for other, item in guard.items():
                distance = (fingerprint ^ item['phash']).bit_count()
                if value == other or distance <= 6:
                    matches.append({'candidate_original_value': value, 'guard_original_value': other,
                        'guard_families': sorted(item['families'] & excluded), 'hamming_distance': distance,
                        'exact_value': value == other, 'scope': 'excluded_family'})
            for other, item in chosen_values.items():
                distance = (fingerprint ^ item['phash']).bit_count()
                if value == other or distance <= 6:
                    matches.append({'candidate_original_value': value, 'guard_original_value': other,
                        'guard_families': [item['family']], 'hamming_distance': distance,
                        'exact_value': value == other, 'scope': 'earlier_confirmation_family'})
        attempts.append({'evaluation_index': index, 'family_id': family, 'sample_id': row['sample_id'],
            'stratum': list(stratum), 'family_turns': len(groups[family]), 'family_unique_values': len(required_values),
            'missing_fingerprints': missing, 'matches': matches, 'admitted': not matches and not missing})
        if missing or matches:
            continue
        selected.append(dict(row, evaluation_index=index, confirmation_split_id='sampler-confirmation-development'))
        for value in required_values:
            chosen_values[value] = {'phash': values[value], 'family': family}
        break
    else:
        raise ValueError(f'Insufficient independently isolated families in stratum {stratum}')
previous = read(root / 'reserved.frozen.jsonl') if (root / 'reserved.frozen.jsonl').is_file() else []
changes = [{'index': index, 'previous_family_id': left['family_id'], 'new_family_id': right['family_id']}
    for index, (left, right) in enumerate(zip(previous, selected)) if left['family_id'] != right['family_id']]
if changes:
    for name in ('reserved.frozen.jsonl', 'public_tasks.jsonl', 'source_inventory.jsonl', 'selection_rule.json'):
        path = root / name
        archive = root / ('initial_family_only.' + name)
        if path.is_file() and not archive.exists():
            archive.write_bytes(path.read_bytes())
write(root / 'reserved.frozen.jsonl', selected)
images = {row['original_artifact_id']: row for row in read('data/viewer/images.complete.remote.jsonl') if row['status'] == 'downloaded'}
tasks, inventory = [], []
for row in selected:
    image = images[row['source_image_id']]
    request = row['instruction']
    task = PublicTask(task_id='sampler-confirm64:' + row['sample_id'], family_id=row['family_id'], split_id='official-dev',
        source_dataset=row['source_dataset'], license_record={**row['license_record'], 'dataset_profile': 'official-viewer-reencoded-v1',
            'purpose': 'preregistered_independent_sampler_confirmation_only'},
        user_request=UserRequest(text=request, source_image_id=image['artifact_id'], spans=(RequestSpan(span_id='span:request', start=0, end=len(request), text=request),)),
        task_mode='edit', check_spec=CheckSpec(extractor_version='public-protocol-only', fallback=True,
            atoms=(PublicCheckAtom(check_id='check:request', raw_span_ids=('span:request',), predicate=request, expected_value=request, confidence=0.),)))
    tasks.append(task.model_dump(mode='json'))
    inventory.append({'evaluation_index': row['evaluation_index'], 'task_id': task.task_id, 'family_id': row['family_id'],
        'sample_id': row['sample_id'], 'request': request, 'source_dataset': row['source_dataset'], 'image': image,
        'source_artifact_path': str(Path('data/artifacts') / image['relative_path']),
        'source_ref_path': 'data/artifacts/refs/' + image['artifact_id'].split(':')[1] + '.json',
        'local_pixels_present': (Path('data/artifacts') / image['relative_path']).is_file()})
write(root / 'public_tasks.jsonl', tasks)
write(root / 'source_inventory.jsonl', inventory)
write(root / 'metadata_isolation_attempts.jsonl', attempts)
report = {'status': 'preregistered_and_metadata_isolated_before_render', 'revision': 'full_family_stored_fingerprints_v2',
    'families': 64, 'source_counts': dict(Counter(row['source_dataset'] for row in selected)),
    'task_strata': {'magicbrush:unspecified': 22, 'omniedit:addition': 42},
    'source_scope': 'official dev only', 'selection_order': 'ascending family_id; earliest official turn; same original slot dataset/declared category',
    'fixed_replacement_rule': 'next family in same prespecified stratum on existing exact-value/pHash conflict or missing fingerprint; before any new render',
    'selection_does_not_use': 'model results, quality outcomes, targets or local image availability',
    'fingerprint_sources': index_paths, 'new_image_hashes_computed': 0, 'phash_radius': 6,
    'full_family_value_coverage': len(chosen_values), 'excluded_family_count': len(excluded), 'excluded_value_count': len(guard),
    'excluded_family_ids': sorted(excluded), 'new_family_ids': [row['family_id'] for row in selected],
    'original_development_family_ids': [row['family_id'] for row in old],
    'overlap_original64': len({row['family_id'] for row in old} & {row['family_id'] for row in selected}),
    'overlap_excluded': len(excluded & {row['family_id'] for row in selected}),
    'accepted_exact_or_near_duplicate_matches': 0, 'excluded_candidates': sum(not row['admitted'] for row in attempts),
    'family_only_preregistration_changes': changes, 'source_images_locally_available': sum(row['local_pixels_present'] for row in inventory),
    'existing_remote_viewer_source_metadata': 64, 'human_reviewed': False, 'private_review_pending': True,
    'official_test_used': False, 'unused_quarantined_train_aliases': 'Not admitted to training; retained official-dev family identity'}
(root / 'selection_rule.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in {'excluded_family_ids', 'new_family_ids', 'original_development_family_ids'}}), flush=True)
