import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentic_image.config import load_config
from agentic_image.data.io import read_jsonl, write_jsonl
from agentic_image.data.visual_admission import prepare_full_actor_admission
from agentic_image.learning.transaction import atomic_json
from agentic_image.runtime.components import Components
from agentic_image.runtime.training import train_actor_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/actor-full-200.yaml')
    parser.add_argument('--collection', action='append', required=True)
    parser.add_argument('--output', default='data/actor-full-sft')
    parser.add_argument('--run-dir', default='runs/actor-full-200')
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--visual-reviews')
    parser.add_argument('--sft-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    downstream_devices = dict(config['devices'])
    config['run_dir'] = args.run_dir
    config['devices']['actor'] = args.device
    config['sft'].update(actor_batch=20, phase_counts=[5, 7, 5, 3], phase_weights=[0.25, 0.35, 0.25, 0.15], actor_updates=args.updates)
    output = Path(args.output).resolve()
    admission = prepare_full_actor_admission(args.collection, output, args.visual_reviews, seed=config['seed'], updates=args.updates)
    records, sources, counts = admission['records'], admission['sources'], admission['counts']
    visual_admission = admission['visual_admission']
    registration = {'scope': 'full_four_phase_actor_sft_v1', 'records': counts['records'], 'independent_families': counts['independent_families'], 'phase_records': counts['phase_records'], 'phase_state_bundles': counts['phase_state_bundles'], 'phase_counts': [5,7,5,3], 'phase_weights': [0.25,0.35,0.25,0.15], 'updates': args.updates, 'planned_action_exposures': 20 * args.updates, 'initial_actor_policy': config['models']['actor_policy'], 'sources': sources, 'teacher_origin': 'executed_public_rules_with_independent_private_positive_repair_filter', 'review_type': 'ai', 'human_reviewed': False, 'admission_identity_sha256': admission['identity_sha256']}
    if args.sft_only:
        registration.update(scope='independent_actor_sft_seed_replication_v1',
            experiment_scope='same_data_training_order_and_seed_replication',
            training_seed=config['seed'], independent_family_expansion=False,
            downstream_rl_planned=False)
    if visual_admission is not None:
        registration['visual_admission'] = visual_admission
    registration = json.loads(json.dumps(registration, allow_nan=False))
    path = output / 'registration.json'
    if path.exists() and json.loads(path.read_text()) != registration:
        raise ValueError('Full actor training resume registration changed')
    atomic_json(path, registration)
    write_jsonl(output / 'actor_records.jsonl', records)
    config['data']['actor_records'] = str(output / 'actor_records.jsonl')
    (output / 'training.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    policy = train_actor_sft(Components(config), stage='actor', updates=args.updates, resume=args.resume)
    metrics_path = Path(args.run_dir) / 'sft_actor' / 'metrics.jsonl'
    metrics = list(read_jsonl(metrics_path))
    if [r['step'] for r in metrics] != list(range(1, args.updates + 1)):
        raise ValueError('Requested full actor updates lack complete committed metrics')
    completed = {'status': 'completed', 'scope': registration['scope'], 'updates': len(metrics), 'policy': policy, 'metrics': str(metrics_path.resolve()), 'actual_action_exposures': sum(len(r['record_ids']) for r in metrics), 'actual_unique_actions_exposed': len({x for r in metrics for x in r['record_ids']}), 'actual_unique_families_exposed': len({x for r in metrics for x in r['family_ids']}), 'phase_counts': [5,7,5,3], 'reference_freeze_pending': True}
    if args.sft_only:
        completed.update(reference_freeze_pending=False,
            experiment_scope=registration['experiment_scope'], training_seed=config['seed'],
            independent_family_expansion=False, downstream_rl_planned=False)
    atomic_json(output / 'training_completed.json', completed)
    if args.sft_only:
        return
    config['devices'] = downstream_devices
    config['run_dir'] = 'runs/full-actor-rl'
    config['models'].update(actor_policy=policy, actor_reference=None, renderer_reference=None)
    selection = json.loads((Path(args.collection[0]) / 'sampler_selection.json').read_text())
    config['renderer'].update({key: selection[key] for key in ('steps', 'eta', 'cfg_scale')})
    Path(config['run_dir']).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(config['run_dir']) / 'sampler_selection.json', selection)
    (output / 'rl_unfrozen.yaml').write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == '__main__':
    main()
