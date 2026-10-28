import argparse
from collections import Counter
import json
from pathlib import Path

from agentic_image.config import load_config
from agentic_image.core.geometry import box_mask, transform_box
from agentic_image.core.types import BoundingBox, PlanSpec
from agentic_image.data.io import read_jsonl, write_jsonl, write_report
from agentic_image.data.paired import _canvas_artifacts
from agentic_image.runtime.components import Components


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--candidates', required=True)
parser.add_argument('--packet', required=True)
parser.add_argument('--reviews', nargs='+', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
config = load_config(args.config)
if config.get('review_mode') != 'ai_exploratory':
    raise ValueError('The user-authorized AI review profile must be explicit')
components = Components(config)
output = Path(args.output)
if output.exists():
    raise FileExistsError('Reviewed training datasets are immutable')
packet = Path(args.packet)
keys = {row['sample_id']: row for row in read_jsonl(packet / 'keys.jsonl')}
annotations = {row['sample_id']: row for row in read_jsonl(packet / 'annotations.pending.jsonl')}
candidates = {row['sample_id']: row for row in read_jsonl(args.candidates)}
dimensions = ('task_completion', 'mask', 'identity', 'registration', 'original_request_coverage', 'plan_compatible')
accepted, rejected, reviews = [], [], []
seen = set()
for path in args.reviews:
    for review in read_jsonl(path):
        sample = review['sample_id']
        if sample in seen:
            raise ValueError('Duplicate review entries require explicit adjudication')
        seen.add(sample)
        if review.get('review_type') != 'ai' or review.get('human_reviewed') is not False:
            raise ValueError('Only explicitly nonhuman AI reviews belong to this pilot')
        key = keys[sample]
        if review.get('scope') != 'original_request_global_pair_v1':
            raise ValueError('This stage requires review of the original-request global plan')
        if review.get('review_revision') != key['review_revision']:
            raise ValueError('Review does not match the exact displayed candidate')
        if not review.get('reviewer') or not review.get('evidence'):
            raise ValueError('Reviews require an identified AI reviewer and observed evidence')
        reviews.append(review)
        ratings = review['ratings']
        if any(ratings.get(name) is not True for name in dimensions):
            rejected.append({'sample_id': sample, 'ratings': ratings})
            continue
        if key['isolation']['quarantined']:
            rejected.append({'sample_id': sample, 'reason': 'quarantined'})
            continue
        plan = PlanSpec(global_text=key['request'])
        if PlanSpec.model_validate(review['global_plan']) != plan:
            raise ValueError('The reviewed plan must exactly retain the original public request')
        annotation = dict(annotations[sample])
        annotation.update(verified=True, semantic_compatible=True, registration_verified=True,
                          human_reviewed=False, review_type='ai')
        row = candidates[sample]
        source, target, valid, edit, frame = _canvas_artifacts(components, row, annotation)
        exclusions = review.get('target_exclusion_boxes', [item['bbox_xyxy_normalized'] for item in review.get('private_loss_exclusions', [])])
        if exclusions:
            valid_mask = components.artifacts.get_array(valid).copy()
            for exclusion in exclusions:
                box = BoundingBox.model_validate(exclusion) if isinstance(exclusion, dict) else BoundingBox(x0=exclusion[0], y0=exclusion[1], x1=exclusion[2], y1=exclusion[3])
                if (box.x1 - box.x0) * (box.y1 - box.y0) > 0.02:
                    raise ValueError('A small dataset-mark exclusion cannot remove a large target region')
                valid_mask *= 1 - box_mask(transform_box(box, frame), frame.canvas_hw)
            valid = components.artifacts.put_array(valid_mask, frame=frame, metadata={'role': 'valid_supervision', 'target_artifact_exclusions': exclusions, 'reviewer': review['reviewer']}).artifact_id
        accepted.append({'sample_id': sample, 'family_id': key['family_id'], 'task_id': sample,
                         'condition_recipe': {'plan': plan.model_dump(mode='json'), 'source_image_id': source},
                         'target_image_id': target, 'valid_mask_id': valid, 'edit_mask_id': edit,
                         'source_dataset': key['source_dataset'], 'region_count': 0,
                         'split_id': 'train', 'verified': True,
                         'verification_provenance': {'review_type': 'ai', 'human_reviewed': False,
                             'reviewer': review['reviewer'], 'review_revision': key['review_revision'],
                             'target_teacher': 'deterministic_original_request_plan',
                             'target_artifact_exclusions': exclusions, 'actor_sampled_plan': False}})
if len({row['family_id'] for row in accepted}) < 4:
    raise ValueError(f'At least four accepted independent families are required; found {len(accepted)}')
output.mkdir(parents=True)
write_jsonl(output / 'renderer.jsonl', accepted)
write_jsonl(output / 'ai_reviews.jsonl', reviews)
write_jsonl(output / 'rejected.jsonl', rejected)
counts = Counter(f"{row['source_dataset']}:0" for row in accepted)
report = {'status': 'accepted', 'review_mode': 'ai_exploratory', 'human_reviewed': False,
          'user_authorized_ai_review': True, 'ai_review_path': str((output / 'ai_reviews.jsonl').resolve()),
          'pair_source_path': str(output.resolve()), 'paused_buckets': [], 'reviewed': len(accepted),
          'selected_by_bucket': dict(counts), 'population_by_bucket': dict(counts),
          'screened_candidates': len(reviews), 'rejected_candidates': len(rejected),
          'scope': 'original_request_global_pair_v1',
          'screening_is_independent_human_audit': False,
          'buckets': {bucket: {name: {'count': count, 'correct': count, 'error_rate': 0.0}
                               for name in dimensions} for bucket, count in counts.items()}}
write_report(output / 'audit.json', report)
print(json.dumps(report), flush=True)
