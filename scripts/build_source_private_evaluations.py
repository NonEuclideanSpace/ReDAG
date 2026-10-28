import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import make_frame, transform_mask
from agentic_image.core.types import BoundingBox, PrivateEvaluation


def write_rows(path, rows):
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    if path.exists() and path.read_text() != payload:
        raise ValueError('Frozen private evaluations may not change')
    path.write_text(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['development64', 'training128', 'sampler_confirm64'], required=True)
    args = parser.parse_args()
    root = Path('data/private_eval_v1') / args.cohort
    annotations_path = root / 'annotations.frozen.jsonl'
    annotations = [json.loads(line) for line in annotations_path.read_text().splitlines()]
    required = 128 if args.cohort == 'training128' else 64
    if len(annotations) != required or len({row['family_id'] for row in annotations}) != required:
        raise ValueError('Private evaluations must preserve the complete frozen cohort')
    store = ArtifactStore('data/artifacts')
    output, inventory, reports = [], [], []
    for row in annotations:
        source_ref = store.get_ref(row['source_image_id'])
        if source_ref.value_id != row['source_value_id']:
            raise ValueError('Reviewed source identity changed')
        evaluation = PrivateEvaluation.model_validate(row['evaluation'])
        if any(not atom.raw_span or atom.raw_span not in row['request'] for atom in evaluation.eval_atoms):
            raise ValueError('An evaluation requirement has no exact original request binding')
        annotation = row['protection_annotation']
        reviewed_paths = {row['source_path']}
        if row.get('review_preview_path') and row.get('preview_provenance'):
            reviewed_paths.add(row['review_preview_path'])
        if annotation['reviewed_source_path'] not in reviewed_paths or not annotation['observation']:
            raise ValueError('Protection review is not bound to the actual source or its registered preview')
        if not Path(annotation['reviewed_contact_path']).is_file():
            raise ValueError('The actual source review evidence file is missing')
        if annotation['status'] == 'not_applicable':
            if annotation.get('scope') != 'global_edit' or not annotation.get('reason') or annotation.get('normalized_boxes'):
                raise ValueError('Global protection inapplicability requires actual source review and an explicit reason')
            protection = {'status': 'not_applicable', 'scope': 'global_edit',
                'source_image_id': source_ref.artifact_id, 'source_value_id': source_ref.value_id,
                'reason': annotation['reason'], 'review_type': 'ai', 'human_reviewed': False,
                'review_evidence': {'reviewer': annotation['reviewer'], 'observation': annotation['observation'],
                    'record_path': str(annotations_path), 'reviewed_contact_path': annotation['reviewed_contact_path']}}
            provenance = {**evaluation.annotation_provenance, 'protection': protection}
            evaluation = evaluation.model_copy(update={'original_protected_mask_id': None, 'annotation_provenance': provenance})
            output.append({'task_id': row['task_id'], 'family_id': row['family_id'], 'source_image_id': source_ref.artifact_id,
                'evaluation': evaluation.model_dump(mode='json')})
            reports.append({'index': row['index'], 'task_id': row['task_id'], 'mask_id': None,
                'protected_fraction': None, 'preservation_status': 'not_applicable',
                'forced_unknown_eval_ids': provenance.get('forced_unknown_eval_ids', {}), 'evaluability': 'global_protection_not_applicable'})
            continue
        if annotation['status'] != 'annotated' or annotation['mode'] not in {'include', 'exclude'}:
            raise ValueError('This builder requires actual source-reviewed nonempty regions')
        height, width = source_ref.shape[:2]
        mask = np.ones((height, width), np.float32) if annotation['mode'] == 'exclude' else np.zeros((height, width), np.float32)
        for coordinates in annotation['normalized_boxes']:
            box = BoundingBox.model_validate(dict(zip(('x0','y0','x1','y1'), coordinates)))
            x0, y0 = int(np.floor(box.x0*width)), int(np.floor(box.y0*height))
            x1, y1 = int(np.ceil(box.x1*width)), int(np.ceil(box.y1*height))
            mask[y0:y1,x0:x1] = 0 if annotation['mode'] == 'exclude' else 1
        frame = source_ref.frame
        if frame is None:
            frame = make_frame((height,width),(512,512),f"private-evaluation:{row['task_id']}")
            mask = transform_mask(mask, frame, 'binary')
        valid = np.zeros(frame.canvas_hw, np.float32)
        px,py=frame.padding_xy
        rh,rw=frame.resized_hw
        valid[py:py+rh,px:px+rw]=1
        mask *= valid
        coverage = float(mask.sum()/valid.sum())
        if not 0 < coverage < 1:
            raise ValueError('A reviewed local-edit protection mask cannot be empty or the entire valid image')
        metadata = {'visibility':'private','role':'private_protected_mask','source_image_id':source_ref.artifact_id,'source_value_id':source_ref.value_id,'task_id':row['task_id'],'annotation_origin':'ai_source_only_conservative_regions','review_type':'ai','human_reviewed':False,'annotation_path':str(annotations_path),'mode':annotation['mode'],'normalized_boxes':annotation['normalized_boxes']}
        ref=store.put_array(mask,frame=frame,producer_id=row['task_id'],metadata=metadata)
        protection={'status':'annotated','scope':'local_edit','source_image_id':source_ref.artifact_id,'source_value_id':source_ref.value_id,'mask_id':ref.artifact_id,'mask_value_id':ref.value_id,'coverage_fraction':coverage,'coverage_scope':annotation['coverage_scope'],'review_type':'ai','human_reviewed':False,'review_evidence':{'reviewer':annotation['reviewer'],'observation':annotation['observation'],'record_path':str(annotations_path),'reviewed_contact_path':annotation['reviewed_contact_path']},'pixel_frame':frame.model_dump(mode='json'),'not_a_target_edit_mask':True}
        provenance={**evaluation.annotation_provenance,'protection':protection}
        evaluation=evaluation.model_copy(update={'original_protected_mask_id':ref.artifact_id,'annotation_provenance':provenance})
        output.append({'task_id':row['task_id'],'family_id':row['family_id'],'source_image_id':source_ref.artifact_id,'evaluation':evaluation.model_dump(mode='json')})
        inventory.extend([str(Path('data/artifacts/refs')/f"{ref.artifact_id.split(':')[1]}.json"),str(Path('data/artifacts')/ref.relative_path)])
        reports.append({'index':row['index'],'task_id':row['task_id'],'mask_id':ref.artifact_id,'protected_fraction':coverage,'forced_unknown_eval_ids':provenance.get('forced_unknown_eval_ids',{}),'evaluability':'fixed_unknown_atoms' if provenance.get('forced_unknown_eval_ids') else 'requires_confirmed_checker_evidence'})
    write_rows(root/'private_tasks.jsonl',output)
    write_rows(root/'protection_report.jsonl',reports)
    routes=Counter(atom['checker_route'] for row in output for atom in row['evaluation']['eval_atoms'])
    fractions=[row['protected_fraction'] for row in reports if row['protected_fraction'] is not None]
    report={'status':'ready','tasks_required':required,'specifications_frozen':required,'protection_ready_count':required,'actual_mask_count':len(fractions),'protection_not_applicable_count':required-len(fractions),'unevaluable_task_ids':[],'fixed_unknown_task_ids':[row['task_id'] for row in reports if row['forced_unknown_eval_ids']],'checker_routes_required':dict(routes),'calibration_confirmed':False,'calibration_binding':'runtime_confirmed_model_specific_route_thresholds_required','source_images_actually_reviewed':required,'target_images_used':0,'human_reviewed':False,'review_type':'ai','mask_scope':'conservative_source_regions_not_full_target_difference_masks','threshold_floor':.8,'private_tasks':str(root/'private_tasks.jsonl'),'protected_fraction_min':min(fractions) if fractions else None,'protected_fraction_max':max(fractions) if fractions else None}
    ready=Path('deployment')/f'{args.cohort}-private-eval.ready.json'
    ready.write_text(json.dumps(report,indent=2)+'\n')
    inventory.extend([str(root/'private_tasks.jsonl'),str(root/'protection_report.jsonl'),str(ready)])
    (root/'deployment_files.txt').write_text('\n'.join(inventory)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
