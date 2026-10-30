import argparse
import gc
import json
import random
import sys
import shutil
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.harness.environment import add_runtime_arguments, load_runtime, run_record, validate_dataset
from redag.learn.credit import credit_experiment, validate_logs
from redag.learn.sft_trainer import read_records
from redag.models.planner import Planner, preserve_previous_conditions
from redag.verify.verifier import load_vlm_scorer, export_view, apply_annotations, apply_ratings

STAGES = ('zero_shot', 'sft', 'ctrl_rl', 'edit_rl', 'alternating_rl')


def export_case(state, dag, trace, folder, identifier, group, budget, config):
    views, steps = {}, []
    final = export_view(state.image, dag, folder, identifier + '_final')
    views[final['id']] = final
    for index, step in enumerate(trace):
        graph = step['dag']
        before = export_view(step['parent_state'].image, graph, folder, f'{identifier}_{index}_before')
        candidate = export_view(step['candidate_state'].image, graph, folder, f'{identifier}_{index}_candidate')
        views.update({before['id']: before, candidate['id']: candidate})
        returned = step.get('returned_state', step['delivered_state'])
        delivered = candidate if step['accepted'] else before
        if returned is not step['delivered_state']:
            delivered = export_view(returned.image, graph, folder, f'{identifier}_{index}_delivered')
            views[delivered['id']] = delivered
        steps.append({'delivered': delivered['id'], 'before': before['id'], 'candidate': candidate['id'], 'target': step['target_node_id'], 'protected': sorted(step['parent_state'].protected_set), 'accepted': step['accepted'], 'stage_scores': step['stage_scores'], 'affected': sorted(step['affected']), 'coverage_uncertain_nodes': step.get('coverage_uncertain_nodes', [])})
    return {'id': identifier, 'group': group, 'views': views, 'final': final['id'], 'steps': steps, 'execution': dict(budget), 'min_gain': config.min_target_gain, 'max_drop': config.max_protected_drop}


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.add_argument('--exp_id', type=int, choices=(1, 2, 3))
    for flag in ('data', 'audit_model', 'audit_device', 'stage_checkpoints', 'training_logs', 'annotations', 'human_ratings', 'resume_results', 'blind_key'):
        parser.add_argument('--' + flag)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--verification_budget', type=int, default=1000)
    parser.add_argument('--generation_budget', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-small-dataset', action='store_true')
    parser.add_argument('--exp1_modes', nargs='+', choices=('flat_full', 'tree_rollback_only', 'spatial_local', 'spatial_no_gates', 'redag_full'), default=('flat_full', 'tree_rollback_only', 'spatial_local', 'spatial_no_gates', 'redag_full'))
    parser.add_argument('--stages', nargs='+', choices=STAGES, help='experiment 3: run only these stages (merge parts with scripts/merge_eval_results.py)')
    parser.add_argument('--shard', type=int, default=0, help='run records with index %% num_shards == shard')
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error('--shard must be in [0, num_shards)')
    partial = args.num_shards > 1 or bool(args.stages)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.audit_model and args.audit_model == args.verifier_model:
        parser.error('Proxy auditor and training verifier must use different checkpoints')
    annotations = json.loads(Path(args.annotations).read_text()) if args.annotations else None
    if args.resume_results:
        result = json.loads(Path(args.resume_results).read_text())
    else:
        if args.exp_id is None or not args.data or min(args.max_steps, args.verification_budget, args.generation_budget) < 1:
            parser.error('New evaluations require --exp_id, --data and positive budgets')
        records = read_records(args.data)
        validate_dataset(records, args.exp_id, args.allow_small_dataset)
        stages = {'fixed_weights': {'controller': args.controller_adapter, 'editor': args.editor_adapter}}
        if args.exp_id == 3:
            if not args.stage_checkpoints:
                parser.error('Experiment 3 requires --stage_checkpoints')
            stages = json.loads(Path(args.stage_checkpoints).read_text())
            if set(stages) != {'zero_shot', 'sft', 'ctrl_rl', 'edit_rl', 'alternating_rl'} or any(set(v) != {'controller', 'editor'} for v in stages.values()):
                raise ValueError('Five named stage checkpoint pairs are required')
            for checkpoints in stages.values():
                for model, path in checkpoints.items():
                    if path:
                        checkpoints[model] = str((Path(args.stage_checkpoints).parent / path).resolve())
                        if not Path(checkpoints[model]).is_dir():
                            raise ValueError('Missing stage adapter directory')
            if any(stages['zero_shot'].values()) or any(not all(v.values()) for k, v in stages.items() if k != 'zero_shot') or len({json.dumps(v, sort_keys=True) for v in stages.values()}) != 5:
                raise ValueError('Zero shot must be base-only and all five stage pairs distinct')
            if args.stages:
                stages = {k: v for k, v in stages.items() if k in args.stages}
        result = {'exp_id': args.exp_id, 'diagnostic_only': args.allow_small_dataset, 'evaluation_cases': [], 'human_ratings': None, 'human_evaluation_status': 'pending', 'budgets': {'verification': args.verification_budget, 'generation': args.generation_budget}}
        if partial:
            result['part'] = {'shard': args.shard, 'num_shards': args.num_shards, 'stages': list(stages), 'seed': args.seed}
        if args.exp_id == 2:
            if not args.training_logs and not args.allow_small_dataset:
                parser.error('Experiment 2 requires actual --training_logs')
            result['training_convergence'], result['full_audits'] = validate_logs(args.training_logs) if args.training_logs else (None, [])
        folder = Path(args.output).parent / (Path(args.output).stem + '_images')
        planner = None
        if any(isinstance(turn, str) or ('graph' not in turn and 'dag' not in turn) for r in records for turn in r.get('turns', [])):
            device = args.planner_device or args.device
            planner = Planner.from_pretrained(args.planner_model, device, torch.float32 if device == 'cpu' else torch.bfloat16, args.planner_adapter)
            for record in records:
                previous = None
                for index, turn in enumerate(record.get('turns', [])):
                    turn = {'user_prompt': turn} if isinstance(turn, str) else dict(turn)
                    explicit = turn.get('superseded_ids', turn.get('supersedes'))
                    if 'graph' in turn or 'dag' in turn:
                        raw = turn.get('graph', turn.get('dag'))
                        graph = preserve_previous_conditions(raw, previous, explicit)
                        turn['superseded_ids'] = explicit if explicit is not None else raw.get('superseded_ids', [])
                    else:
                        graph = planner.plan(turn['user_prompt'], previous, superseded_ids=explicit)
                        turn['superseded_ids'] = list(planner.last_superseded_ids)
                    turn['graph'], previous = graph, graph
                    record['turns'][index] = turn
            del planner
            planner = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        validate_dataset(records, args.exp_id, args.allow_small_dataset, require_graphs=True)
        saved, descriptions, serial = defaultdict(dict), {}, 0
        for stage, checkpoints in stages.items():
            args.controller_adapter, args.editor_adapter = checkpoints['controller'], checkpoints['editor']
            harness, controller, editor = load_runtime(args)
            harness.verification_budget, harness.generation_budget = args.verification_budget, args.generation_budget
            if args.exp_id == 2:
                credit_experiment(args, records, harness, controller, editor, planner, result, folder)
            else:
                for mode in (args.exp1_modes if args.exp_id == 1 else ('redag_full',)):
                    harness.mode, counts = mode, defaultdict(int)
                    for index, record in enumerate(records):
                        if index % args.num_shards != args.shard:
                            continue
                        for seed in (range(3) if args.exp_id == 3 else (0,)):
                            state, dag, trace = run_record(record, harness, controller, editor, args.max_steps, args.seed + seed, planner=planner)
                            k = len(dag.nodes)
                            counts[k] += 1
                            if not args.allow_small_dataset and k not in (10, 20, 30):
                                raise ValueError('Main evaluations require final K=10/20/30')
                            identifier = f'case_s{args.shard}_{stage}_{serial:06d}' if partial else f'case_{serial:06d}'
                            serial += 1
                            case = export_case(state, dag, trace, folder, identifier, f'{stage}/{mode}/{k}', harness.last_run, harness.config)
                            result['evaluation_cases'].append(case)
                            saved[index, seed][stage] = case['views'][case['final']]['image']
                            descriptions[index, seed] = {'requirements': [node.question for node in dag.nodes.values()], 'source': str(record['image']) if isinstance(record['image'], (str, Path)) else None}
                    if not args.allow_small_dataset and not partial and any(counts[k] < (300 if args.exp_id == 3 else 100) for k in (10, 20, 30)):
                        raise ValueError('Each K tier requires 100 sessions, with three seeds for experiment 3')
            del harness, controller, editor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del planner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.exp_id == 3 and partial:  # blind pairs are drawn after merging every part
            result['final_views'] = {f'{i}:{s}': images for (i, s), images in saved.items()}
            result['descriptions'] = {f'{i}:{s}': value for (i, s), value in descriptions.items()}
        elif args.exp_id == 3:
            rng, pairs, key = random.Random(args.seed), [], []
            for identity, images in rng.sample(list(saved.items()), min(200, len(saved))):
                names = rng.sample(list(stages), 2)
                pair = dict(descriptions[identity], id=len(pairs))
                for side, name in zip(('a', 'b'), names):
                    path = folder / f'blind_{len(pairs)}_{side}.png'
                    shutil.copyfile(images[name], path)
                    pair[side] = str(path.resolve())
                pairs.append(pair)
                key.append({'id': len(key), 'session_seed': identity, 'a': names[0], 'b': names[1]})
            result['blind_pairs'] = pairs
            Path(args.output).with_suffix('.blind_pairs.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2))
            Path(args.output).with_suffix('.blind_key.json').write_text(json.dumps(key, ensure_ascii=False, indent=2))
    if args.audit_model:
        device = args.audit_device or args.verifier_device or args.device
        scorer = load_vlm_scorer(args.audit_model, device, torch.float32 if device == 'cpu' else torch.bfloat16)
        apply_annotations(result, scorer=scorer, source='model_proxy')
        result['proxy_auditor'] = args.audit_model
    if annotations is not None:
        apply_annotations(result, annotations=annotations)
    if args.human_ratings:
        key = args.blind_key or Path(args.resume_results or args.output).with_suffix('.blind_key.json')
        apply_ratings(result, args.human_ratings, key if Path(key).is_file() else None)
    groups, audits = result.get('human_results', {}), result.get('full_audits', [])
    if annotations is not None or args.human_ratings:
        complete = bool(groups or audits) and all(v['annotated_sessions'] == v['expected_sessions'] for v in groups.values()) and all(a.get('human_E_leak') is not None for a in audits)
        if result['exp_id'] == 3:
            complete = complete and bool((result.get('human_ratings') or {}).get('complete'))
        result['human_evaluation_status'] = 'complete' if complete else 'partial'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
