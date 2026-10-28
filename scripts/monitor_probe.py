import csv
import importlib.metadata
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace/agentic-image-pilot-small')
ROLES = ('actor', 'renderer', 'judge')
LOGS = (
    'rental-actor-download.log', 'rental-renderer-copy.log', 'rental-judge-copy.log',
    'rental-actor-metadata.log', 'rental-renderer-metadata.log', 'rental-judge-metadata.log',
    'rental-install-wheel.log', 'rental-kernels.log', 'rental-protocol-warmup.log',
    'rental-training.log', 'protocol-warmup.log', 'actor-nf4-export.log', 'judge-nf4-export.log',
    'protocol-build.log', 'protocol-evaluation.log', 'actor-replay-diagnostic.log',
    'renderer-baseline.log', 'viewer-proposals.log', 'actor-layer-diagnostic.log',
    'renderer-global-ai.log', 'plan-global-ai.log', 'renderer-global-ai-evaluation.log',
    'plan-global-ai-evaluation.log',
)
PACKAGES = ('torch', 'transformers', 'diffusers', 'peft', 'bitsandbytes', 'causal-conv1d', 'flash-linear-attention')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def tail(path, limit=24000):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read(limit).decode('utf-8', errors='replace').splitlines()
    except OSError:
        return []


def safe_error(line):
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    line = re.sub(r'https?://\S+', '[link]', line)
    line = re.sub(r'(?i)(token|password|secret|authorization|api[_-]?key)\s*[:=]\s*\S+', r'\1=[hidden]', line)
    line = re.sub(r'\b(?:hf_|sk-)[A-Za-z0-9_-]+', '[hidden]', line)
    return line.strip()[:260]


def gpu_report():
    fields = 'index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit'
    try:
        result = subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4, check=True)
        return [dict(index=int(row[0]), name=row[1].strip(), utilization=number(row[2]), memory_used_mib=number(row[3]), memory_total_mib=number(row[4]), temperature_c=number(row[5]), power_w=number(row[6]), power_limit_w=number(row[7])) for row in csv.reader(result.stdout.splitlines())]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def model_report(role):
    metadata = read_json(ROOT / f'deployment/rental-{role}-hf.json')
    if not metadata:
        return {'role': role, 'status': 'unknown'}
    source = read_json(ROOT / f'deployment/rental-{role}-sources.json')
    rows = source.get('files', []) if source else metadata.get('siblings', [])
    rows = [row for row in rows if row['rfilename'] != '.gitattributes' and not Path(row['rfilename']).name.lower().startswith(('readme', 'license')) and not row['rfilename'].lower().endswith(('.jpg', '.png', '.mp4')) and not (role == 'renderer' and '/' not in row['rfilename'] and row['rfilename'].endswith('.safetensors'))]
    destination = ROOT / 'models' / role / metadata['sha']
    total = complete = received = 0
    changed = None
    pending = []
    for row in rows:
        name = row['rfilename']
        if Path(name).is_absolute() or '..' in Path(name).parts:
            continue
        expected = row.get('size')
        if not isinstance(expected, int):
            continue
        total += expected
        target = destination / name
        partial = target.with_name(target.name + '.incomplete')
        present = target if target.is_file() else partial
        try:
            stat = present.stat()
            size = min(stat.st_size, expected)
            changed = max(changed or 0, stat.st_mtime)
        except OSError:
            size = 0
        received += size
        if target.is_file() and target.stat().st_size == expected:
            complete += 1
        else:
            pending.append({'name': name, 'bytes': size, 'total': expected})
    return {'role': role, 'model': metadata['id'], 'revision': metadata['sha'], 'bytes': received, 'total': total, 'files_complete': complete, 'files_total': len(rows), 'status': 'complete' if complete == len(rows) and rows else 'incomplete', 'changed_at': changed, 'runtime_complete': not any(not Path(row['name']).name.lower().startswith(('readme', 'license')) for row in pending), 'pending': pending[:5]}


def dependencies():
    paths = [Path('/usr/local/miniconda3/envs/py312/lib/python3.12/site-packages'), *ROOT.glob('.venv/lib/python*/site-packages')]
    versions = {}
    for distribution in importlib.metadata.distributions(path=[str(path) for path in paths]):
        name = distribution.metadata.get('Name', '').lower().replace('_', '-')
        if name in PACKAGES:
            versions[name] = distribution.version
    return [{'name': name, 'version': versions.get(name), 'gpu_verified': None} for name in PACKAGES]


def export_report():
    exports = []
    for role, directory in (('actor', 'actor-4b-nf4'), ('judge', 'judge-8b-nf4')):
        process_present = None
        try:
            pid = int((ROOT / f'deployment/{role}-nf4-export.pid').read_text().strip())
            stat = Path(f'/proc/{pid}/stat')
            process_present = stat.is_file() and stat.read_text().rsplit(')', 1)[1].strip().split()[0] != 'Z'
        except (OSError, ValueError, IndexError):
            pass
        manifest = read_json(ROOT / 'models' / directory / 'base_manifest.json')
        exports.append({'role': role, 'process_present': process_present, 'manifest_present': bool(manifest)})
    return exports


def training_report(path=None):
    paths = []
    for pattern in ('*/metrics.jsonl', '*/*/metrics.jsonl', '*/*/*/metrics.jsonl'):
        paths.extend((ROOT / 'runs').glob(pattern))
    if not paths:
        return {'status': 'no_updates', 'stage': None, 'update': None, 'loss': None, 'gradients': None, 'checkpoint': None, 'changed_at': None}
    path = path or max(paths, key=lambda item: item.stat().st_mtime)
    rows = []
    for line in tail(path, 64000):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except ValueError:
            pass
    if not rows:
        return {'status': 'unknown', 'stage': path.parent.name}
    row = rows[-1]
    checkpoints = list((path.parent / 'checkpoints').glob('*/COMMITTED.json'))
    checkpoint = max(checkpoints, key=lambda item: item.stat().st_mtime) if checkpoints else None
    selected = {key: row.get(key) for key in ('loss', 'actor_loss', 'renderer_loss', 'gradients', 'parameter_changes', 'seconds', 'reward', 'mean_reward')}
    history = []
    for metric in rows[-100:]:
        commit = read_json(Path(metric.get('checkpoint', '')) / 'COMMITTED.json')
        if commit and metric.get('loss') is not None:
            history.append({'x': commit['committed_ns'] / 1e9, 'y': metric['loss'], 'step': metric.get('step')})
    exits = {'protocol-warmup': 'protocol-warmup', 'paired-global-ai': 'renderer-global-ai', 'plan-global-ai': 'plan-global-ai', 'public-stage-transitions': 'public-stage-transitions-v2', 'public-initialization': 'public-initialization-sft80-resume4t', 'renderer-expanded-200': 'renderer-expanded-200-resume4t'}
    exit_name = exits.get(path.parent.parent.name)
    outcome = read_json(ROOT / f'deployment/{exit_name}.exit.json') if exit_name else None
    status = 'completed' if outcome and outcome.get('returncode', outcome.get('exit_code')) == 0 else 'failed' if outcome else 'updates_recorded'
    return {'status': status, 'stage': row.get('stage', path.parent.name), 'run': str(path.parent.relative_to(ROOT)), 'update': row.get('step', row.get('round')), 'changed_at': path.stat().st_mtime, 'checkpoint': str(checkpoint.relative_to(ROOT)) if checkpoint else None, 'loss_history': history, **selected}


def jsonl_rows(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return None


def file_count(folder, pattern):
    return sum(path.is_file() for path in folder.glob(pattern)) if folder.is_dir() else None


def formal_evaluation_progress():
    folder = ROOT / 'runs/development64-controller-episodes'
    available = folder.is_dir()
    seeds = (11, 23, 47, 73, 101)
    methods = []
    for method in ('base', 'sft', 'agent_rl', 'joint_rl'):
        count = sum((folder / method / 'episodes' / f'{index:03d}' / f'seed{seed}' / 'episode.json').is_file()
            for index in range(64) for seed in seeds) if available else None
        methods.append({'method': method, 'episode_files': count, 'expected_episodes': 320})
    total = sum(row['episode_files'] for row in methods) if available else None
    return {'episode_files': total, 'expected_episodes': 1280, 'families': 64, 'seeds': list(seeds),
        'all_episode_files_present': total == 1280 if available else None, 'methods': methods,
        'measurement': 'saved_episode_file_count', 'success_scores_included': False, 'training_updates_included': False}


def sampler_progress(name, methods, review_name):
    folder = ROOT / 'runs' / name
    registration = read_json(folder / 'registration.json')
    selection = read_json(folder / 'sampler_selection.json')
    review_paths = [*folder.glob('ai_severe_reviews_*.jsonl')]
    if folder.name in {'sampler-fallback-200', 'sampler-confirmation-200'}:
        review_paths += list(folder.glob('quality_reviews_*.jsonl'))
    review_paths = [path for path in review_paths if path.name != review_name] + [folder / review_name]
    labels = {}
    for path in review_paths:
        for row in jsonl_rows(path) or []:
            if row.get('image_id') and row.get('review_evidence'):
                labels[row['image_id']] = row
    for row in read_json(folder / 'reference_reviews.json') or []:
        if row.get('image_id') and row.get('review_evidence'):
            labels[row['image_id']] = row
    rows = []
    for method in methods:
        images = folder / 'images' / method
        scores = folder / 'scores' / method
        image_rows = [value for path in images.glob('*.json') if isinstance(value := read_json(path), dict)]
        score_rows = [value for path in scores.glob('*.json') if isinstance(value := read_json(path), dict)]
        reviewed = [labels[row['image_id']] for row in image_rows if row.get('image_id') in labels]
        decision = read_json(folder / f'sampler_selection.{method}.json')
        if not decision and selection:
            decision = selection.get('decisions', {}).get(method)
        rows.append({'method': method, 'expected': 128,
            'images': file_count(images, '*.png'), 'score_files': file_count(scores, '*.json'),
            'scored': sum(row.get('status') == 'scored' for row in score_rows) if scores.is_dir() else None,
            'score_blocked': sum(row.get('status') == 'score_blocked' for row in score_rows) if scores.is_dir() else None,
            'scores_with_unknown_atoms': sum(bool(row.get('unknown_atom_ids')) for row in score_rows) if scores.is_dir() else None,
            'reviewed': len(reviewed) if labels and images.is_dir() else None,
            'review_known': sum(row.get('known') is True and type(row.get('severe_defect')) is bool for row in reviewed) if labels and images.is_dir() else None,
            'review_unknown': sum(row.get('known') is False for row in reviewed) if labels and images.is_dir() else None,
            'reused': name == 'sampler-fallback-200' and method == 'ode50',
            'accepted': decision.get('accepted') if decision else None,
            'decision_status': decision.get('status') if decision else None})
    return {'run': name, 'registered': registration is not None, 'methods': rows,
        'accepted': selection.get('accepted') if selection else None,
        'decision_status': selection.get('status') if selection else None,
        'candidate': selection.get('candidate') if selection else None,
        'plans': file_count(folder / 'plans', '*/frozen.json'), 'expected_plans': 64,
        'selection_families': selection.get('families') if selection else None,
        'selection_paired_records': selection.get('paired_records') if selection else None}


def public_task_manifest():
    path = ROOT / 'data/public/expanded_training128.tasks.jsonl'
    report = {'path': str(path.relative_to(ROOT)), 'file_present': path.is_file(), 'bytes': None, 'tasks': None, 'error': None}
    try:
        report['bytes'] = path.stat().st_size
        content = path.read_text(encoding='utf-8-sig')
        if content.lstrip().startswith('['):
            rows = json.loads(content)
            report['format'] = 'json_array'
        else:
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            report['format'] = 'jsonl'
        if not isinstance(rows, list) or any(not isinstance(row, dict) or 'task_id' not in row or 'family_id' not in row for row in rows):
            raise ValueError('Public manifest does not contain task records')
        report['tasks'] = len(rows)
    except (OSError, ValueError) as error:
        report['error'] = f'{type(error).__name__}: {error}'[:240]
    return report


def pipeline_report(queue):
    development = []
    for name in ('development64-baseline', 'development64-sft200'):
        folder = ROOT / 'runs' / name
        development.append({'run': name, 'images': file_count(folder, '*/*/[0-9][0-9][0-9].png'),
            'expected': 320, 'families': 64, 'seeds': 5,
            'completed_shards': sum((read_json(path) or {}).get('status') == 'completed' for path in folder.glob('*/*/summary.json')) if folder.is_dir() else None})
    full = []
    keys = ('tasks', 'renders', 'positive_repairs', 'contract_rollbacks', 'budget_finishes', 'successful_finishes', 'unknown_recoveries', 'fixed_unknown_stops', 'skipped_unverifiable_renders', 'failed_tasks', 'action_records', 'phase_counts', 'unique_families')
    for index in range(2):
        report = read_json(ROOT / f'runs/full-actor-demonstrations/shard-{index}/report.json')
        full.append({'shard': index, 'status': 'report_available' if report else 'unknown', **{key: report.get(key) if report else None for key in keys}})
    rl = []
    for name in ('rl_agent', 'rl_joint'):
        folder = ROOT / 'runs/full-actor-rl' / name
        completion = read_json(folder / 'training_completed.json')
        metrics = jsonl_rows(folder / 'metrics.jsonl')
        last = metrics[-1] if metrics else {}
        target = (completion or {}).get('required_effective_updates', 32)
        commit = read_json(Path(last['checkpoint']) / 'COMMITTED.json') if last.get('checkpoint') else None
        stage_development = folder / 'development' / str(last.get('round', 0))
        states = [read_json(path) for path in stage_development.glob('*/state.json')]
        results = read_json(stage_development / 'results.json')
        rl.append({'stage': name, 'status': completion.get('status') if completion else 'updates_recorded' if metrics else 'unknown',
            'round': last.get('round'), 'effective_update_counts': (completion or last).get('effective_update_counts'),
            'effective_target_met': completion.get('effective_target_met') if completion else None,
            'required_effective_updates': target,
            'training_target_committed': (last.get('effective_update_counts') or {}).get('required_components', 0) >= target and (commit or {}).get('status') == 'COMMITTED',
            'development': {'started_trajectories': len(states), 'completed_trajectories': sum(bool(state and state.get('done')) for state in states),
                'scored_results': len(results) if isinstance(results, list) else None, 'expected_tasks': 64},
            'optimizer_steps': last.get('optimizer_steps'), 'parameter_changes': last.get('parameter_changes')})
    actor_completion = read_json(ROOT / 'data/actor-full-sft/training_completed.json')
    actor_metrics = jsonl_rows(ROOT / 'runs/actor-full-200/sft_actor/metrics.jsonl')
    actor_training = actor_completion or ({'status': 'updates_recorded', 'updates': actor_metrics[-1].get('step')} if actor_metrics else None)
    replications = []
    for seed in (23, 47):
        replicate_metrics = jsonl_rows(ROOT / f'runs/actor-full-seed{seed}/sft_actor/metrics.jsonl')
        replicate_completion = read_json(ROOT / f'data/actor-full-sft-seed{seed}/training_completed.json')
        replications.append({'seed': seed, 'run': f'runs/actor-full-seed{seed}', 'target_updates': 200,
            'updates': replicate_completion.get('updates') if replicate_completion else replicate_metrics[-1].get('step') if replicate_metrics else None,
            'status': 'completed' if replicate_completion else 'updates_recorded' if replicate_metrics else 'not_recorded',
            'new_independent_families': 0})
    review_root = ROOT / 'data/actor-full-sft-visual'
    review_path = review_root / 'visual-reviews-complete.jsonl'
    if not review_path.is_file():
        review_path = review_root / 'visual-reviews-progress.jsonl'
    reviews = jsonl_rows(review_path)
    visual = {'reviewed_families': len(reviews) if reviews is not None else None,
        'labels': {key: {name: sum(row.get('labels', {}).get(key) is value for row in reviews)
            for name, value in [('true', True), ('false', False), ('unknown', None)]}
            for key in ('repair_positive', 'initial_success', 'recovery_success')} if reviews is not None else None,
        'admission': read_json(ROOT / 'deployment/full-actor-visual-admission.ready.json')}
    selection = jsonl_rows(ROOT / 'data/training_expansion/training_selection_v1.frozen.jsonl')
    public = public_task_manifest()
    private = read_json(ROOT / 'deployment/training128-private-eval.ready.json') or {}
    calibration = read_json(ROOT / 'data/calibration/calibration600.completed.json') or {}
    jobs = list((queue or {}).get('jobs', {}).values())
    running = [{'title': job.get('title'), 'gpus': job.get('gpus'), 'status': job.get('status')} for job in jobs if job.get('status') == 'running']
    return {'current': running, 'queue_available': queue is not None,
        'dataset': {'training_families': len({row['family_id'] for row in selection}) if selection is not None else None,
            'public_tasks': public['tasks'], 'public_manifest': public, 'private_specifications': private.get('specifications_frozen'),
            'calibration_families': calibration.get('family_counts'), 'calibration_known_labels': calibration.get('boolean_records'),
            'calibration_unknown_labels': calibration.get('unknown_labels_excluded'), 'confirmed_routes': calibration.get('confirmed_routes')},
        'development': development,
        'formal_evaluation': formal_evaluation_progress(),
        'sampler': sampler_progress('sampler-expanded-200', ('ode50', 'sde50', 'sde28', 'sde20'), 'ai_severe_reviews_merged.jsonl'),
        'fallback': sampler_progress('sampler-fallback-200', ('ode50', 'sde50_eta005', 'sde50_eta015'), 'quality_reviews_merged.jsonl'),
        'confirmation': sampler_progress('sampler-confirmation-200', ('ode50', 'sde50_eta005'), 'quality_reviews_merged.jsonl'),
        'controller_collection': full,
        'visual_admission': visual,
        'renderer_provisional': read_json(ROOT / 'outputs/renderer-fixed240-provisional.json'),
        'renderer_fixed': read_json(ROOT / 'runs/development64-fixed-private/aggregate.json'),
        'controller_training': actor_training,
        'controller_replications': replications,
        'controller_admission': read_json(ROOT / 'data/actor-full-sft/registration.json'), 'rl': rl}


def main():
    memory = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable'):
                memory[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(ROOT)
    environment = read_json(ROOT / 'deployment/rental-environment.json') or {}
    checks = environment.get('checks', {})
    nf4 = {name: bool(value.get('nf4_forward_finite') and value.get('nf4_backward_finite')) for name, value in checks.items() if isinstance(value, dict) and name in ('cuda:0', 'cuda:1')}
    logs = []
    errors = []
    models = [model_report(role) for role in ROLES]
    ready = {model['role']: model.get('runtime_complete', False) for model in models}
    for name in LOGS:
        path = ROOT / 'deployment' / name
        if not path.is_file():
            continue
        stat = path.stat()
        logs.append({'name': name, 'bytes': stat.st_size, 'changed_at': stat.st_mtime})
        lines = tail(path)[-80:]
        recovered = any('Successfully installed' in line for line in lines)
        recovered = recovered or (name == 'protocol-warmup.log' and (read_json(ROOT / 'deployment/protocol-warmup.exit.json') or {}).get('returncode') == 0)
        recovered = recovered or any(name.startswith(f'rental-{role}-') and ready[role] for role in ROLES)
        for line in lines:
            if re.search(r'(?i)(\berror\b|exception|traceback|\bfailed\b|fatal|timed out)', line):
                errors.append({'source': name, 'at': stat.st_mtime, 'message': safe_error(line), 'recovered': recovered})
    queue = read_json(ROOT / 'deployment/gpu-queue-state.json')
    baseline = read_json(ROOT / 'runs/renderer-baseline/progress.json')
    print(json.dumps({'sampled_at': time.time(), 'gpus': gpu_report(), 'memory': {'total': memory.get('MemTotal'), 'available': memory.get('MemAvailable')}, 'disk': {'total': disk.total, 'used': disk.used, 'free': disk.free}, 'models': models, 'dependencies': dependencies(), 'nf4_operator_checks': nf4, 'training': training_report(), 'training_runs': [training_report(path) for path in sorted((ROOT / 'runs').glob('*/*/metrics.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)[:5]], 'exports': export_report(), 'queue': queue, 'pipeline': pipeline_report(queue), 'efficiency': read_json(ROOT / 'deployment/gpu-efficiency.json'), 'baseline': baseline, 'logs': logs, 'errors': errors[-8:]}, allow_nan=False))


if __name__ == '__main__':
    main()
