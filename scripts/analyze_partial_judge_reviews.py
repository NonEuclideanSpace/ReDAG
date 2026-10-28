import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    reviews = root / 'data/calibration/ai_reserved'
    scores = {}
    for path in sorted((root / 'runs').glob('calibration-raw-8b*/shard-*/results.json')):
        for row in json.loads(path.read_text()):
            key = (row['calibration_index'], row['candidate_role'], row['atom']['eval_id'])
            if key in scores:
                raise ValueError(f'Duplicate actual score: {key}')
            scores[key] = row
    labels = {}
    revisions = []
    for pattern in ('ai_reviews_*.jsonl', 'ai_labels_*.jsonl'):
        for path in sorted(reviews.rglob(pattern)):
            for row in map(json.loads, path.read_text().splitlines()):
                key = (row['calibration_index'], row['candidate_role'], row['eval_id'])
                if row.get('supersedes_review_path'):
                    revisions.append((key, row))
                    continue
                if key in labels:
                    raise ValueError(f'Duplicate visual judgment: {key}')
                labels[key] = row
    for key, row in revisions:
        prior = labels[key]
        if prior['review_evidence']['record_path'] != row['supersedes_review_path'] or prior['correct'] != row['prior_correct']:
            raise ValueError(f'Visual reassessment does not identify its prior judgment: {key}')
        for name in ('family_id', 'candidate_image_id', 'atom'):
            if row[name] != prior[name]:
                raise ValueError(f'Reassessment changed frozen identity: {key}, {name}')
        if 'candidate_value_id' in prior and row['candidate_value_id'] != prior['candidate_value_id']:
            raise ValueError(f'Reassessment changed candidate value: {key}')
        labels[key] = row
    joined, unknown = [], []
    for key, label in labels.items():
        score = scores[key]
        for name in ('sample_id', 'family_id', 'calibration_fold', 'source_image_id', 'atom'):
            if label[name] != score[name]:
                raise ValueError(f'Frozen review binding mismatch: {key}, {name}')
        if label['candidate_image_id'] != score['candidate_artifact_id']:
            raise ValueError(f'Candidate image changed: {key}')
        for name in ('candidate_value_id', 'draft_evaluation_sha256'):
            if name in label and label[name] != score[name]:
                raise ValueError(f'Candidate value/draft changed: {key}')
        evidence = label['review_evidence']
        if label['review_type'] != 'ai' or label['human_reviewed'] is not False or any(not evidence.get(k) for k in ('reviewer', 'observation', 'record_path')):
            raise ValueError(f'Visual review provenance missing: {key}')
        if label['correct'] is None:
            unknown.append(key)
            continue
        if type(label['correct']) is not bool:
            raise ValueError(f'Non-boolean label: {key}')
        expected = score['atom']['expected_value']
        kind = expected.get('kind', score['atom']['checker_route']) if isinstance(expected, dict) else score['atom']['checker_route']
        joined.append({
            'family_id': score['family_id'], 'partition': score['calibration_fold'],
            'calibration_index': score['calibration_index'], 'candidate_role': score['candidate_role'],
            'candidate_image_id': score['candidate_artifact_id'], 'candidate_value_id': score['candidate_value_id'],
            'draft_evaluation_sha256': score['draft_evaluation_sha256'], 'eval_id': score['atom']['eval_id'],
            'kind': kind, 'value': score['score']['value'], 'label': label['correct'],
            'known': score['score']['known'] and not score['structured_evidence_pending'],
            'raw_template_known': score['score']['known'], 'structured_evidence_pending': score['structured_evidence_pending'],
            'exact_pass': score['score']['exact_pass'], 'adjudicated': True,
            'review_type': 'ai', 'human_reviewed': False, 'review_evidence': evidence,
            'judge_identity': score['judge_identity'], 'calibrated': False,
        })
    groups = defaultdict(list)
    for row in joined:
        groups[(row['partition'], row['kind'])].append(row)
    diagnostics = []
    for (partition, kind), rows in sorted(groups.items()):
        positives = [r for r in rows if r['known'] and r['value'] >= .8 and r['exact_pass'] is not False]
        true_positive = sum(r['label'] for r in positives)
        n = len(positives)
        p = true_positive / n if n else 0
        z = 1.959963984540054
        lower = (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n) if n else 0
        diagnostics.append(dict(partition=partition, kind=kind, labels=len(rows), families=len({r['family_id'] for r in rows}), positive_labels=sum(r['label'] for r in rows), raw_unknown=sum(not r['raw_template_known'] for r in rows), structured_pending=sum(r['structured_evidence_pending'] for r in rows), fixed_threshold=.8, predicted_positive=n, true_positive=true_positive, false_positive=n-true_positive, precision=p if n else None, precision_wilson_lower=lower, brier=sum((r['value']-r['label'])**2 for r in rows)/len(rows)))
    report = dict(scope='partial_fixed_threshold_diagnostics_no_calibration', raw_scores=len(scores), raw_families=len({r['family_id'] for r in scores.values()}), reviewed_atom_candidates=len(labels), reviewed_families=len({r['family_id'] for r in labels.values()}), unknown_review_labels=len(unknown), merged_boolean_labels=len(joined), reviewed_family_counts=dict(Counter({fold:len({r['family_id'] for r in labels.values() if r['calibration_fold']==fold}) for fold in ['selection','confirmation']})), diagnostics=diagnostics, thresholds_fitted=False, calibration_complete=False, review_type='ai', human_reviewed=False)
    (reviews/'joined_actual_scores_and_ai_labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in joined))
    (reviews/'partial_judge_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
