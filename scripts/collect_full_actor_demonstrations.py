import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentic_image.config import load_config
from agentic_image.data.audit import require_pair_audit
from agentic_image.data.interventions import collect_public_rule_interventions
from agentic_image.data.repairs import fixed_unknown_repair_requirements
from agentic_image.data.io import read_jsonl, write_jsonl
from agentic_image.learning.transaction import atomic_json
from agentic_image.runtime.components import Components


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/actor-full-200.yaml')
    parser.add_argument('--tasks', default='data/public/expanded_training128.tasks.jsonl')
    parser.add_argument('--selection', default='data/training_expansion/training_selection_v1.frozen.jsonl')
    parser.add_argument('--sampler-selection', default='runs/sampler-expanded-200/sampler_selection.json')
    parser.add_argument('--private-ready', default='deployment/training128-private-eval.ready.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--renderer-version', default='renderer-expanded-200')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid collection shard')
    config = load_config(args.config)
    require_pair_audit(config['data']['paired_audit'], config.get('review_mode', 'human'))
    selection = list(read_jsonl(args.selection))
    tasks = list(read_jsonl(args.tasks))
    if len(selection) != 128 or len(tasks) != 128 or {(r['sample_id'], r['family_id']) for r in selection} != {(r['task_id'], r['family_id']) for r in tasks}:
        raise ValueError('The full controller pool must equal the frozen 128-family training selection')
    if any(r['split_id'] != 'train' or r['limits']['max_renders'] != 3 for r in tasks):
        raise ValueError('Collection must preserve the original train split and three-render budget')
    private_ready = json.loads(Path(args.private_ready).read_text())
    if private_ready.get('status') not in {'ready', 'completed'}:
        raise ValueError('Independent training private evaluations are not ready')
    private = list(read_jsonl(config['data']['private_tasks']))
    if {r['task_id'] for r in private} != {r['task_id'] for r in tasks}:
        raise ValueError('Every fixed training task needs its independent private evaluation')
    sampler = json.loads(Path(args.sampler_selection).read_text())
    if sampler.get('status') != 'accepted' and sampler.get('accepted') is not True:
        raise ValueError('The actual sampler quality decision has not passed')
    sampler_id = sampler.get('sampler_id') or sampler.get('candidate')
    if not sampler_id or any(key not in sampler for key in ('steps', 'eta', 'cfg_scale')):
        raise ValueError('The accepted sampler must declare its complete numerical configuration')
    if sampler.get('renderer_policy') != config['models']['renderer_policy']:
        raise ValueError('The accepted sampler belongs to a different renderer policy')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [r for index, r in enumerate(tasks) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('Collection limit must be positive')
        selected = selected[:args.limit]
    registration = {'scope': 'actual_public_rule_intervention_trajectory_v1', 'task_manifest_sha256': digest(args.tasks), 'training_selection_sha256': digest(args.selection), 'private_manifest_sha256': digest(config['data']['private_tasks']), 'private_ready_sha256': digest(args.private_ready), 'calibration_sha256': digest(config['judge']['calibration']), 'sampler_selection_sha256': digest(args.sampler_selection), 'renderer_version': args.renderer_version, 'actor_policy': config['models']['actor_policy'], 'renderer_policy': config['models']['renderer_policy'], 'shard_index': args.shard_index, 'shard_count': args.shard_count, 'selected_tasks': [r['task_id'] for r in selected], 'teacher_origin': 'deterministic_public_request_and_control_intervention', 'maximum_renders_per_family': 3, 'positive_repair_rule': 'complete_known_atoms_dense_delta_at_least_0.1_without_passed_atom_regression', 'rollback_rule': 'actual_injected_public_control_restored_with_legal_historical_SELECT', 'finish_rule': 'calibrated_all_requirements_pass_or_actual_render_budget_exhausted', 'human_reviewed': False, 'visual_labels_generated': False}
    thresholds = json.loads(Path(config['judge']['calibration']).read_text())
    fixed_unknown = {row['task_id']: reasons for row in private if (reasons := fixed_unknown_repair_requirements(row['evaluation'], thresholds))}
    registration['fixed_unknown_early_stop'] = {'scope': 'frozen_private_route_or_source_grounding_only', 'task_reasons': fixed_unknown, 'independent_families': len(fixed_unknown), 'renders_per_affected_family': 1, 'verified_public_actions_per_affected_family': 3, 'finish_or_rollback_labels_generated': False, 'full_training_pool_retained': True}
    for role in ('actor', 'renderer'):
        weight = Path(config['models'][role + '_policy'])
        if weight.is_dir():
            weight = weight / 'adapter_model.safetensors'
        stat = weight.stat()
        registration[role + '_weight_identity'] = {'path': str(weight.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    path = output / 'registration.json'
    if path.exists() and json.loads(path.read_text()) != registration:
        raise ValueError('Collection resume inputs changed')
    atomic_json(path, registration)
    write_jsonl(output / 'tasks.jsonl', selected)
    atomic_json(output / 'sampler_selection.json', sampler)
    config['run_dir'] = str(output)
    config['renderer'].update({key: sampler[key] for key in ('steps', 'eta', 'cfg_scale')})
    for name in ('actor', 'renderer', 'encoders', 'judge'):
        config['devices'][name] = args.device
    (output / 'collection.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = collect_public_rule_interventions(Components(config), output / 'tasks.jsonl', output, args.renderer_version, sampler_id, max_tasks=len(selected), seed=config['seed'])
    atomic_json(output / 'completed.json', {'status': 'completed', 'report': report, 'training_admission_pending': True})


if __name__ == '__main__':
    main()
