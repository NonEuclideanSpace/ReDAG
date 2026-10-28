from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from uuid import uuid4


LABELS = ('repair_positive', 'initial_success', 'recovery_success')
IMAGE_KEYS = ('SOURCE', 'INITIAL REQUEST', 'CONTROLLED WRONG REQUEST', 'RECOVERY')


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Visual review needs nonempty ' + name)


def bundle_images(bundle):
    outcomes = {}
    for outcome in bundle['private_outcomes']:
        stage = outcome['stage']
        if stage in outcomes:
            raise ValueError('Repeated private outcome stage for ' + bundle['task_id'])
        outcomes[stage] = outcome
    if not {'initial', 'injected', 'recovery'} <= outcomes.keys():
        raise ValueError('Visual review lacks the complete actual image sequence for ' + bundle['task_id'])
    return {'SOURCE': bundle['public_source_image_id'], 'INITIAL REQUEST': outcomes['initial']['image_id'],
            'CONTROLLED WRONG REQUEST': outcomes['injected']['image_id'], 'RECOVERY': outcomes['recovery']['image_id']}, outcomes


def validate_review(review, bundle):
    task_id = bundle['task_id']
    if review['task_id'] != task_id or review['family_id'] != bundle['family_id']:
        raise ValueError('Visual review task/family identity mismatch: ' + task_id)
    if (review.get('review_type') != 'ai' or review.get('human_reviewed') is not False
            or review.get('actual_visual_inspection') is not True):
        raise ValueError('Visual review must declare actual AI inspection without human review: ' + task_id)
    require_text(review.get('reviewer'), 'reviewer')
    paths = review.get('reviewed_image_paths')
    if not isinstance(paths, list) or not paths:
        raise ValueError('Visual review needs its actual reviewed image paths: ' + task_id)
    for path in paths:
        require_text(path, 'reviewed image path')
    for field in ('labels', 'evidence'):
        if not isinstance(review.get(field), dict) or set(review[field]) != set(LABELS):
            raise ValueError('Visual review requires all three ' + field + ': ' + task_id)
    for label in LABELS:
        if review['labels'][label] is not None and type(review['labels'][label]) is not bool:
            raise ValueError('Visual labels must be bool or null: ' + task_id + ':' + label)
        require_text(review['evidence'][label], 'specific evidence for ' + label)
    expected, outcomes = bundle_images(bundle)
    if not isinstance(review.get('images'), dict) or set(review['images']) != set(IMAGE_KEYS):
        raise ValueError('Visual review needs all four exact image identities: ' + task_id)
    if review['images'] != expected:
        raise ValueError('Visual review artifact identity mismatch: ' + task_id)
    if any(not isinstance(value, str) or not value.startswith('artifact:') for value in expected.values()):
        raise ValueError('Visual image identity is not an actual artifact ID: ' + task_id)
    if 'public_request' in review and review['public_request'] != bundle['public_request']:
        raise ValueError('Visual review original request mismatch: ' + task_id)
    return outcomes


def read_events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finish_label(record, bundle, review, event_rows):
    name = record['record_id'].rsplit(':', 1)[-1]
    choices = {'successful_initial_finish': ('initial_success', 'INITIAL REQUEST'),
               'terminal_finish': ('recovery_success', 'RECOVERY')}
    if record['phase'] != 2 or name not in choices:
        raise ValueError('Visual success record is not a recognized FINISH: ' + record['record_id'])
    matches = [event for event in event_rows if event.get('name') == name]
    if len(matches) != 1:
        raise ValueError('FINISH needs exactly one original execution event: ' + record['record_id'])
    event = matches[0]
    if event.get('internal_action', {}).get('kind') != 'FINISH':
        raise ValueError('The visual success event did not execute FINISH: ' + record['record_id'])
    result = event.get('result')
    if not isinstance(result, dict) or 'error' not in result or result['error'] is not None:
        raise ValueError('The original FINISH execution was not successful: ' + record['record_id'])
    label, role = choices[name]
    after = event['public_after_state']
    if after.get('done') is not True or after.get('delivered_image_id') != review['images'][role]:
        raise ValueError('FINISH did not deliver the exact visually reviewed image: ' + record['record_id'])
    for state in (event['public_before_state'], after):
        if (state['task']['task_id'] != bundle['task_id']
                or state['task']['family_id'] != bundle['family_id']):
            raise ValueError('FINISH execution task/family mismatch: ' + record['record_id'])
    return label, name


def filter_visual_sft_records(records, state_bundles, reviews, event_reader=read_events):
    by_state, by_task, family_tasks = {}, {}, {}
    for bundle in state_bundles:
        state_id, task_id, family_id = bundle['state_id'], bundle['task_id'], bundle['family_id']
        if state_id in by_state or task_id in by_task or family_id in family_tasks:
            raise ValueError('Repeated actual state/task/family bundle is ambiguous for visual admission')
        by_state[state_id], by_task[task_id], family_tasks[family_id] = bundle, bundle, task_id
    review_by_task, review_families, outcomes_by_task = {}, set(), {}
    for number, review in enumerate(reviews, 1):
        task_id, family_id = review['task_id'], review['family_id']
        if task_id in review_by_task or family_id in review_families:
            raise ValueError('Duplicate or conflicting visual review row: ' + task_id)
        if task_id not in by_task:
            raise ValueError('Visual review has no actual collection state bundle: ' + task_id)
        outcomes_by_task[task_id] = validate_review(review, by_task[task_id])
        review_by_task[task_id] = (number, review)
        review_families.add(family_id)
    included, audit, events, required_cases = [], [], {}, set()
    for record in records:
        proof = record['verifier_provenance']
        needed = record['phase'] == 1 or proof.get('visual_success_label') is True
        bundle = by_state.get(record['state_id'])
        item = {'record_id': record['record_id'], 'state_id': record['state_id'],
                'family_id': record['family_id'], 'task_id': bundle['task_id'] if bundle else None,
                'phase': record['phase'], 'decision': 'include', 'reason': 'not_a_visual_positive_or_success_label',
                'visual_label': None, 'review_value': None, 'original_record': record}
        if needed:
            if bundle is None or bundle['family_id'] != record['family_id']:
                raise ValueError('Visual training record has no exact actual state bundle: ' + record['record_id'])
            if record['record_id'] not in bundle['verified_record_ids']:
                raise ValueError('Visual training record was not admitted by its actual state bundle: ' + record['record_id'])
            task_id = bundle['task_id']
            required_cases.add(task_id)
            if task_id not in review_by_task:
                raise ValueError('Missing required actual visual review: ' + task_id)
            number, review = review_by_task[task_id]
            event_name = None
            if record['phase'] == 1:
                outcomes = outcomes_by_task[task_id]
                before, after = outcomes['injected']['image_id'], outcomes['recovery']['image_id']
                if (proof.get('before_image_id') != before or proof.get('after_image_id') != after
                        or outcomes['recovery'].get('before_image_id') != before
                        or outcomes['recovery'].get('positive') is not True):
                    raise ValueError('Repair record does not bind the actual injected/recovery pair: ' + record['record_id'])
                label = 'repair_positive'
            else:
                path = proof['execution_evidence']
                if path not in events:
                    events[path] = event_reader(path)
                label, event_name = finish_label(record, bundle, review, events[path])
            value = review['labels'][label]
            accepted = value is True
            item.update(decision='include' if accepted else 'exclude', visual_label=label,
                review_value=value, reason=f'ai_visual_{label}_' + ('true' if accepted else 'false' if value is False else 'unknown'),
                review_line=number, reviewer=review['reviewer'], review_type='ai', human_reviewed=False,
                actual_visual_inspection=True, reviewed_image_paths=review['reviewed_image_paths'],
                images=review['images'], evidence=review['evidence'][label],
                execution_evidence=proof['execution_evidence'], execution_event=event_name)
        if item['decision'] == 'include':
            included.append(record)
        audit.append(item)
    excluded = [item for item in audit if item['decision'] == 'exclude']
    summary = {'input_records': len(records), 'included_records': len(included), 'excluded_records': len(excluded),
        'input_families': len({row['family_id'] for row in records}),
        'included_families': len({row['family_id'] for row in included}),
        'input_phase_counts': {str(key): value for key, value in Counter(row['phase'] for row in records).items()},
        'included_phase_counts': {str(key): value for key, value in Counter(row['phase'] for row in included).items()},
        'excluded_reasons': dict(Counter(row['reason'] for row in excluded)),
        'review_rows': len(reviews), 'required_review_cases': len(required_cases),
        'included_positive_states': len({row['state_id'] for row in included if row['phase'] == 1}),
        'labels_or_records_added': False, 'raw_records_or_scores_modified': False,
        'unknown_labels_preserved': True, 'review_type': 'ai', 'human_reviewed': False}
    return included, audit, summary


def jsonl_identity(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return ([json.loads(line) for line in payload.splitlines() if line.strip()],
            {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()})


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False).encode()
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError('Existing immutable admission artifact has a different identity: ' + str(path))
            return False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def phase_sampler_class():
    path = Path(__file__).resolve().parents[1] / 'learning/sampling.py'
    spec = importlib.util.spec_from_file_location('full_actor_admission_phase_sampling', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PhaseBalancedSampler


def prepare_full_actor_admission(collections, output, visual_reviews=None, seed=11,
                                 selection_path='data/training_expansion/training_selection_v1.frozen.jsonl', updates=200):
    if type(updates) is not int or updates < 200:
        raise ValueError('The full actor stage requires at least 200 optimizer updates')
    selected, selection_identity = jsonl_identity(selection_path)
    families = {row['family_id'] for row in selected}
    records, sources, state_bundles = [], [], []
    if not collections:
        raise ValueError('Full actor admission requires completed collection inputs')
    for folder in map(Path, collections):
        completed_path = folder / 'completed.json'
        completed_bytes = completed_path.read_bytes()
        if json.loads(completed_bytes).get('status') != 'completed':
            raise ValueError('A collection has not completed its actual execution')
        rows, source = jsonl_identity(folder / 'actor_records.jsonl')
        source['records'] = len(rows)
        source['completed'] = {'path': str(completed_path.resolve()),
                               'sha256': hashlib.sha256(completed_bytes).hexdigest()}
        records.extend(rows)
        if visual_reviews:
            bundles, source['state_bundles'] = jsonl_identity(folder / 'state_bundles.jsonl')
            state_bundles.extend(bundles)
        sources.append(source)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    visual_admission = None
    if visual_reviews:
        reviews, review_identity = jsonl_identity(visual_reviews)
        records, decisions, summary = filter_visual_sft_records(records, state_bundles, reviews)
        audit = {'schema': 'visual_sft_admission_v1', 'reviews': review_identity,
                 'collection_sources': sources, 'summary': summary, 'decisions': decisions}
        audit_sha256 = hashlib.sha256(json.dumps(audit, indent=2, allow_nan=False).encode()).hexdigest()
        audit_path = output / 'visual-admission' / (audit_sha256 + '.json')
        immutable_json(audit_path, audit)
        visual_admission = {'schema': audit['schema'], 'reviews': review_identity,
            'audit_path': str(audit_path), 'audit_sha256': audit_sha256, 'summary': summary}
    if (len(records) < 512 or {row['family_id'] for row in records} != families
            or len(families) != 128 or len(selected) != 128):
        raise ValueError('Full actor SFT needs at least 512 actual verified actions across the fixed 128 training families')
    if len({row['record_id'] for row in records}) != len(records):
        raise ValueError('Duplicated action records cannot inflate the independent action count')
    for row in records:
        proof = row['verifier_provenance']
        if (row.get('verified') is not True or not Path(row['snapshot_path']).is_file()
                or not row['target_ids'] or row['target_ids'][-1] != 248046
                or not Path(proof['execution_evidence']).is_file()):
            raise ValueError('A training action lacks its actual pre-action snapshot or execution evidence')
        if row['phase'] == 1 and not (proof.get('repair_success_label') is True and proof.get('all_atoms_known') is True
                and proof.get('no_previously_passed_requirement_regressed') is True
                and proof.get('minimum_dense_improvement') == 0.1):
            raise ValueError('Repair phase actions require independently calibrated positive repair evidence')
    positive_states = {row['state_id'] for row in records if row['phase'] == 1}
    if len(positive_states) < 4:
        raise ValueError('Full actor SFT requires at least four actually verified positive repair states')
    sampler = phase_sampler_class()(records, quotas=(5, 7, 5, 3), seed=seed)
    batch = sampler.next_batch()
    if Counter(row['phase'] for row in batch) != Counter({0: 5, 1: 7, 2: 5, 3: 3}):
        raise ValueError('The real state pools cannot fill the original phase-balanced batch')
    counts = {'records': len(records), 'independent_families': len(families),
        'positive_states': len(positive_states),
        'phase_records': {str(key): value for key, value in Counter(row['phase'] for row in records).items()},
        'phase_state_bundles': {str(phase): len({row['state_id'] for row in records if row['phase'] == phase}) for phase in range(4)},
        'phase_counts': [5, 7, 5, 3], 'phase_weights': [0.25, 0.35, 0.25, 0.15]}
    identity = {'schema': 'full_actor_cpu_admission_v1', 'selection': selection_identity,
        'sources': sources, 'visual_admission': visual_admission, 'counts': counts,
        'seed': seed, 'planned_optimizer_updates': updates, 'maximum_actions_per_state_in_batch': 2,
        'retained_records_sha256': hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()}
    identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {'records': records, 'sources': sources, 'visual_admission': visual_admission,
            'counts': counts, 'identity': identity, 'identity_sha256': identity_sha256}


def publish_visual_admission_ready(result, path):
    if result['visual_admission'] is None:
        raise ValueError('The visual ready marker requires actual visual review admission')
    ready = {'status': 'completed', 'scope': 'full_actor_visual_admission_cpu_only',
             'identity_sha256': result['identity_sha256'], 'identity': result['identity'],
             'counts': result['counts'], 'models_loaded': False, 'training_started': False,
             'raw_scores_modified': False, 'labels_added': False}
    created = immutable_json(path, ready)
    return ready, created
