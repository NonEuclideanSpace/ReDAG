import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics


METHODS = ('base', 'sft', 'agent_rl', 'joint_rl')
SEEDS = (11, 23, 47, 73, 101)
COMPARISONS = (('base', 'sft'), ('sft', 'agent_rl'), ('agent_rl', 'joint_rl'), ('sft', 'joint_rl'))
COST_FIELDS = ('observed_gpu_seconds', 'observed_wall_seconds', 'actual_dit_forwards_observed',
               'sampled_actions_observed', 'output_tokens_in_saved_actions')


def read_json(path, evidence):
    path = Path(path)
    before = path.stat()
    value = json.loads(path.read_text())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed during reading: ' + str(path))
    evidence.append({'path': str(path.resolve()), 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns})
    return value


def read_lines(path, evidence):
    path = Path(path)
    before = path.stat()
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed during reading: ' + str(path))
    evidence.append({'path': str(path.resolve()), 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns})
    return values


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def registered_tasks(tasks, reserved, training_sets):
    if len(tasks) != 64 or len(reserved) != 64 or len({t['family_id'] for t in tasks}) != 64:
        raise ValueError('Exactly64 distinct registered development families are required')
    if len({t['task_id'] for t in tasks}) != 64:
        raise ValueError('Development task IDs must be unique')
    for index, (task, registration) in enumerate(zip(tasks, reserved)):
        if (task['split_id'] != 'development' or registration['split_id'] != 'development'
                or type(registration['evaluation_index']) is not int or registration['evaluation_index'] != index
                or task['task_id'] != 'development64:' + registration['sample_id']
                or task['family_id'] != registration['family_id']
                or task['user_request']['text'] != registration['instruction']):
            raise ValueError('Development registration differs at index ' + str(index))
    training = [row for rows in training_sets for row in rows]
    families, identities = {t['family_id'] for t in tasks}, {t['task_id'] for t in tasks}
    if any(t['family_id'] in families or t['task_id'] in identities for t in training):
        raise ValueError('Supplied training records overlap registered evaluation families or task IDs')
    return {'status': 'checked' if training_sets else 'not_checked', 'supplied_training_records': len(training),
            'scope': 'Family and task IDs in supplied training manifests only; not a fresh image-duplication or all-source leakage audit'}


def validate_episode(row, method, relative, tasks, parameters):
    index, seed = row.get('index'), row.get('seed')
    if type(index) is not int or index not in range(64) or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Episode index or seed is outside the fixed registration')
    task = tasks[index]
    expected = Path(f'{index:03d}') / f'seed{seed}' / 'episode.json'
    if (relative != expected or row.get('method') != method or row.get('task_id') != task['task_id']
            or row.get('family_id') != task['family_id'] or row.get('scope') != 'actual_controller_episode'):
        raise ValueError('Episode path or identity differs from the fixed registration: ' + str(relative))
    if row.get('private_target_used_for_actions') is not False or type(row.get('training_updates')) is not int or row['training_updates'] != 0:
        raise ValueError('Episode does not declare target isolation and zero evaluation updates')
    if row.get('sampler_parameters') != parameters:
        raise ValueError('Episode sampler differs from its frozen common protocol')
    terminal = row['terminal']
    if type(terminal.get('success')) is not bool or type(row.get('protocol_failure')) is not bool:
        raise ValueError('Success and protocol failure must be recorded booleans')
    if 'delivered_image_id' not in row or (row['delivered_image_id'] is not None and not isinstance(row['delivered_image_id'], str)):
        raise ValueError('Delivered image identity must be explicitly recorded')
    if any(type(row.get('budget', {}).get(key)) is not int or row['budget'][key] < 0 for key in ('actions', 'renders', 'invalid')):
        raise ValueError('Actions, renders and invalid counts must be recorded nonnegative integers')
    if any(type(atom.get('known')) is not bool for atom in terminal['atoms']):
        raise ValueError('Every atom must preserve its known/unknown flag')
    preservation = terminal.get('preservation_status')
    if preservation not in ('measured', 'not_evaluated', 'not_applicable'):
        raise ValueError('Preservation availability must be explicit')
    if preservation == 'measured' and not finite_number(terminal.get('preservation')):
        raise ValueError('Measured preservation must be finite')
    loaded = row.get('loaded_state_verification', {})
    if method == 'base':
        valid = loaded.get('source') == 'untrained_base' and loaded.get('zero_output_projections_verified') is True
    else:
        source = 'completed_sft_exports' if method == 'sft' else 'committed_rl_pair'
        valid = loaded.get('source') == source and all(loaded.get(role, {}).get('exact_tensor_equality') is True for role in ('actor', 'renderer'))
    if not valid:
        raise ValueError('Episode lacks the expected loaded-policy verification')
    costs = row.get('costs') or {}
    if 'historical_measurements_complete' in costs and type(costs['historical_measurements_complete']) is not bool:
        raise ValueError('Cost completeness must be a recorded boolean')
    for key in COST_FIELDS:
        value = costs.get(key)
        if value is not None and (not finite_number(value) or value < 0):
            raise ValueError('Invalid observed cost: ' + key)
    value = row.get('private_scoring_seconds')
    if value is not None and (not finite_number(value) or value < 0):
        raise ValueError('Invalid separately measured scoring time')
    return row['family_id'], seed


def descriptive(values, total):
    values = list(values)
    return {'observed_count': len(values), 'missing_count': total - len(values),
            'sum': sum(values) if values else None, 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None}


def configuration_summary(rows, tasks):
    observed = list(rows.values())
    missing = [{'index': i, 'family_id': task['family_id'], 'seed': seed}
               for i, task in enumerate(tasks) for seed in SEEDS if (task['family_id'], seed) not in rows]
    successes = sum(row['terminal']['success'] for row in observed)
    costs = {key: descriptive(((r.get('costs') or {}).get(key) for r in observed
                              if (r.get('costs') or {}).get(key) is not None), len(observed)) for key in COST_FIELDS}
    costs['private_scoring_seconds'] = descriptive((r['private_scoring_seconds'] for r in observed
                                                   if r.get('private_scoring_seconds') is not None), len(observed))
    costs['complete_ledger_episodes'] = sum((r.get('costs') or {}).get('historical_measurements_complete') is True for r in observed)
    costs['scope'] = 'Observed local-service GPU time is a lower bound. Summed per-action wall time is not elapsed benchmark wall time. Private scoring is separate; startup and external tool GPU time are not included.'
    return {'observed_episodes': len(observed), 'required_episodes': 320, 'complete': not missing,
            'observed_families': len({family for family, _ in rows}), 'missing': missing,
            'success_count': successes, 'observed_success_rate': successes / len(observed) if observed else None,
            'full_registered_success_rate': successes / 320 if not missing else None,
            'failure_counts_overlap': {'automatic_unsuccessful': len(observed) - successes,
                'protocol_failure': sum(r['protocol_failure'] for r in observed),
                'no_delivered_image': sum(not r.get('delivered_image_id') for r in observed),
                'with_invalid_actions': sum(r['budget'].get('invalid', 0) > 0 for r in observed),
                'with_unknown_atoms': sum(any(not a['known'] for a in r['terminal']['atoms']) for r in observed)},
            'unknown_atoms': sum(not a['known'] for r in observed for a in r['terminal']['atoms']),
            'terminal_reasons': dict(Counter(r['terminal'].get('terminal_reason', 'missing') for r in observed)),
            'preservation_status': dict(Counter(r['terminal']['preservation_status'] for r in observed)),
            'preservation': descriptive((r['terminal']['preservation'] for r in observed
                                         if r['terminal']['preservation_status'] == 'measured'), len(observed)),
            'costs': costs,
            'logical_budget': {key: descriptive((r['budget'][key] for r in observed if key in r['budget']), len(observed))
                               for key in ('actions', 'renders', 'inspects', 'reads', 'invalid')},
            'by_seed': {str(seed): {'observed': sum(s == seed for _, s in rows),
                                  'successes': sum(r['terminal']['success'] for (f, s), r in rows.items() if s == seed)} for seed in SEEDS}}


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_statistics(values, draws, random_seed):
    grouped = defaultdict(list)
    for family, before, after in values:
        grouped[family].append((before, after))
    if not values:
        return {'pairs': 0, 'families': 0, 'before_mean': None, 'after_mean': None,
                'after_minus_before': None, 'equal_family_difference': None,
                'family_cluster_95ci': None, 'equal_family_95ci': None, 'family_paired_seed_counts': {}}
    clusters = [(sum(b - a for a, b in grouped[family]), len(grouped[family])) for family in sorted(grouped)]
    weighted, equal = [], []
    if len(clusters) >= 2:
        rng = random.Random(random_seed)
        for _ in range(draws):
            sample = rng.choices(clusters, k=len(clusters))
            weighted.append(sum(s for s, n in sample) / sum(n for s, n in sample))
            equal.append(statistics.mean(s / n for s, n in sample))
    return {'pairs': len(values), 'families': len(grouped),
            'before_mean': statistics.mean(a for f, a, b in values),
            'after_mean': statistics.mean(b for f, a, b in values),
            'after_minus_before': statistics.mean(b - a for f, a, b in values),
            'equal_family_difference': statistics.mean(s / n for s, n in clusters),
            'family_cluster_95ci': [percentile(weighted, p) for p in (.025, .975)] if weighted else None,
            'equal_family_95ci': [percentile(equal, p) for p in (.025, .975)] if equal else None,
            'family_paired_seed_counts': dict(Counter(str(n) for s, n in clusters))}


def comparison(before, after, draws, random_seed):
    common = sorted(set(before) & set(after))
    full = {family for family, seed in common if all((family, s) in before and (family, s) in after for s in SEEDS)}
    def subset(keys):
        success = [(f, float(before[f, s]['terminal']['success']), float(after[f, s]['terminal']['success'])) for f, s in keys]
        preservation = [(f, before[f, s]['terminal']['preservation'], after[f, s]['terminal']['preservation'])
                        for f, s in keys if all(rows[f, s]['terminal']['preservation_status'] == 'measured' for rows in (before, after))]
        return {'success': paired_statistics(success, draws, random_seed),
                'preservation': paired_statistics(preservation, draws, random_seed),
                'preservation_missing_pairs': len(keys) - len(preservation)}
    result = subset(common)
    result.update(paired_episodes=len(common), paired_families=len({f for f, s in common}),
                  before_only=len(set(before) - set(after)), after_only=len(set(after) - set(before)),
                  complete_five_seed_families=len(full), complete_five_seed_sensitivity=subset([(f, s) for f, s in common if f in full]),
                  success_transitions=dict(Counter(f"{int(before[key]['terminal']['success'])}->{int(after[key]['terminal']['success'])}" for key in common)),
                  after_only_success_with_before_protocol_failure=sum(not before[k]['terminal']['success'] and after[k]['terminal']['success'] and before[k]['protocol_failure'] for k in common))
    result['matched_keys'] = [{'family_id': f, 'seed': s, 'task_id': before[f, s]['task_id']} for f, s in common]
    return result


def analyze(root, tasks_path, reservation_path, training_paths=(), draws=20000, random_seed=20261028):
    if type(draws) is not int or not 100 <= draws <= 100000:
        raise ValueError('Bootstrap draws must be between100 and100000')
    started = datetime.now(timezone.utc).isoformat()
    root, evidence = Path(root), []
    tasks, reserved = read_lines(tasks_path, evidence), read_lines(reservation_path, evidence)
    isolation = registered_tasks(tasks, reserved, [read_lines(path, evidence) for path in training_paths])
    common = read_json(root / 'common_protocol.json', evidence)
    if (common.get('scope') != 'actual_controller_episodes' or common.get('tasks') != tasks
            or common.get('image_seeds') != list(SEEDS) or common.get('private_target_used_for_actor_actions') is not False
            or type(common.get('training_updates_during_evaluation')) is not int or common['training_updates_during_evaluation'] != 0):
        raise ValueError('Common protocol differs from registered isolated evaluation')
    observed, registrations = {}, {}
    for method in METHODS:
        folder = root / method / 'episodes'
        paths = sorted(folder.rglob('episode.json')) if folder.exists() else []
        registration_path = root / method / 'registration.json'
        registration = read_json(registration_path, evidence) if registration_path.is_file() else None
        if registration is not None:
            policy = registration.get('evaluated_policy', {})
            source = {'base': 'untrained_base', 'sft': 'completed_sft_exports'}.get(method, 'committed_rl_pair')
            if (registration.get('method') != method or registration.get('scope') != 'actual_controller_episodes'
                    or registration.get('common_protocol') != common or policy.get('method') != method
                    or policy.get('source') != source):
                raise ValueError('Policy registration differs from the frozen common protocol: ' + method)
            registrations[method] = policy
        if paths and registration is None:
            raise ValueError('Observed episodes have no policy registration: ' + method)
        rows = {}
        for path in paths:
            row = read_json(path, evidence)
            key = validate_episode(row, method, path.relative_to(folder), tasks, common['sampler_protocol']['parameters'])
            if key in rows:
                raise ValueError('Duplicate family and seed within a configuration: ' + str(key))
            rows[key] = row
        observed[method] = rows
    configurations = {method: configuration_summary(observed[method], tasks) for method in METHODS}
    complete = all(value['complete'] for value in configurations.values()) and len(registrations) == 4
    return {'schema': 'formal_controller_family_analysis_v1', 'status': 'complete' if complete else 'incomplete',
            'read_started_at': started, 'read_finished_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'Automatic formal episode results; current completion-order subset until all1280 registered episodes are present. No visual review is performed.',
            'required_total_episodes': 1280, 'observed_total_episodes': sum(len(rows) for rows in observed.values()),
            'registered_families': 64, 'registered_seeds': list(SEEDS), 'training_manifest_isolation': isolation,
            'configurations': configurations, 'evaluated_policies': registrations,
            'comparisons': {after + '_minus_' + before: comparison(observed[before], observed[after], draws, random_seed)
                            for before, after in COMPARISONS},
            'bootstrap': {'unit': 'family_id', 'draws': draws, 'random_seed': random_seed,
                          'method': 'Python random.Random.choices over sorted family IDs, reset per metric/subset; resample families with replacement carrying every eligible paired seed; percentile intervals with linear interpolation',
                          'minimum_families_for_interval': 2, 'multiple_comparison_adjustment': None},
            'limitations': ['Intervals describe variation among observed families, not uncertainty from unobserved or selectively completed tasks.',
                            'Success is the stored automatic judgment; it is not an independent human or visual review.',
                            'Preservation uses only pairs measured on both sides; missing scores are never imputed.',
                            'Success gains can include protocol-completion gains; preservation and failure categories must be interpreted together.',
                            'Different device placement and reused actions can affect observed cost. No total elapsed-time or full GPU-cost claim is made.',
                            'Saved policy-loading proof is checked structurally; weights are not reloaded and image scoring is not repeated.',
                            'Family/task overlap checks cover only supplied training manifests and do not replace the original data audit.'],
            'source_files': evidence, 'source_weights_loaded': False, 'gpu_work_started': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/development64-controller-episodes'))
    parser.add_argument('--tasks', type=Path, default=Path('data/evaluation/development64/public_tasks.jsonl'))
    parser.add_argument('--reservation', type=Path, default=Path('data/evaluation/development64/reserved.frozen.jsonl'))
    parser.add_argument('--training-tasks', type=Path, action='append')
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--random-seed', type=int, default=20261028)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output to preserve previous analysis: ' + str(args.output))
    report = analyze(args.root, args.tasks, args.reservation,
                     args.training_tasks or [Path('data/public/expanded_training128.tasks.jsonl')], args.draws, args.random_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'observed_episodes': report['observed_total_episodes'], 'report': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
