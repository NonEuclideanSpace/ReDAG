import argparse
import gzip
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tarfile
import time

from sample_delivery_compression import DigestSink, low_priority, write_json

progress_path = Path(__file__).resolve().parents[1] / 'src/agentic_image/learning/progress.py'
progress_spec = importlib.util.spec_from_file_location('rl_export_progress', progress_path)
progress = importlib.util.module_from_spec(progress_spec)
progress_spec.loader.exec_module(progress)
accumulate_effective_updates = progress.accumulate_effective_updates
committed_round_metrics = progress.committed_round_metrics
required_rl_components = progress.required_rl_components


def read_json(path):
    return json.loads(path.read_text())


def require_file(path, root):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if '..' in path.parts or not path.is_relative_to(root):
        raise ValueError('Export sources must be inside the project')
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Export sources must not traverse symlinks')
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError('A nonempty regular export source is required: ' + str(path))
    return path


def snapshot(files):
    result = {}
    for path in files:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Export source became nonregular')
        result[str(path)] = {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'device': info.st_dev, 'inode': info.st_ino}
    return result


def validate_history(stage, mode, expected_updates=32):
    stage = Path(stage).resolve()
    if mode not in ('agent', 'joint') or stage.name != 'rl_' + mode:
        raise ValueError('The explicit stage and mode must match')
    completed = read_json(stage / 'training_completed.json')
    if (completed.get('status') != 'completed' or completed.get('effective_target_met') is not True
            or completed.get('mode') != mode or completed.get('required_effective_updates') != expected_updates
            or completed.get('required_components') != sorted(required_rl_components(mode))):
        raise ValueError('A completed stage with the exact effective target is required')
    rounds = completed.get('rounds')
    if type(rounds) is not int or rounds < 1:
        raise ValueError('A positive completed round cursor is required')
    metrics = [json.loads(line) for line in (stage / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if [row.get('round') for row in metrics] != list(range(1, rounds + 1)):
        raise ValueError('Metrics must cover every completed round exactly once')
    commits = {}
    for marker in (stage / 'checkpoints').glob('*/COMMITTED.json'):
        commit = read_json(marker)
        cursor = commit.get('next_data_cursor', {}).get('round')
        if (type(cursor) is not int or cursor in commits or commit.get('status') != 'COMMITTED'
                or commit.get('transaction_id') != marker.parent.name
                or commit.get('metadata', {}).get('mode') != mode
                or commit.get('new_version') != f'{mode}:{cursor}'
                or commit.get('components') != {'actor': 'actor.pt', 'renderer': 'renderer.pt'}):
            raise ValueError('Commit identity, components, mode or cursor is inconsistent')
        commits[cursor] = (marker.parent, commit)
    if sorted(commits) != list(range(1, rounds + 1)):
        raise ValueError('All historical COMMITTED markers are required for recovery')
    latest = max(commits.values(), key=lambda pair: (
        pair[1].get('committed_ns', (pair[0] / 'COMMITTED.json').stat().st_mtime_ns), pair[0].name))
    checkpoint, final = commits[rounds]
    if latest[0] != checkpoint:
        raise ValueError('Actual latest COMMITTED differs from the final metric checkpoint')
    counts = {'actor': 0, 'renderer': 0, 'required_components': 0}
    zeros = 0
    for number, prior in enumerate(metrics, 1):
        directory, commit = commits[number]
        accumulate_effective_updates(counts, mode, commit)
        expected = committed_round_metrics(commit, directory, counts)
        for key, value in expected.items():
            if key == 'checkpoint':
                if Path(prior.get(key, '')).resolve() != directory.resolve():
                    raise ValueError('Metric checkpoint differs from its actual COMMITTED')
            elif prior.get(key) != value:
                raise ValueError('Metric evidence differs from COMMITTED: ' + key)
        fraction = prior.get('zero_advantage_fraction')
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Invalid zero-advantage evidence')
        zeros = zeros + 1 if fraction > .8 else 0
    if (counts != completed.get('effective_update_counts') or counts['required_components'] < expected_updates
            or completed.get('consecutive_zero_advantage_rounds') != zeros):
        raise ValueError('Completion counts differ from actual committed effective updates')
    for name in ('actor.pt', 'renderer.pt', 'training_state.pt'):
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError('Final checkpoint is missing recoverable state: ' + name)
    config = final['metadata']['config']
    if config.get('rl', {}).get(mode + '_effective_updates') != expected_updates:
        raise ValueError('Committed configuration has a different effective target')
    return {'completed': completed, 'commits': commits, 'metrics': metrics,
        'checkpoint': checkpoint, 'config': config}


def adapter_file(directory):
    directory = Path(directory)
    choices = [directory / 'adapter_model.safetensors', directory / 'policy/adapter_model.safetensors']
    present = [path for path in choices if path.is_file()]
    if len(present) != 1:
        raise ValueError('Exactly one actual actor adapter must be present: ' + str(directory))
    return present[0]


def validate_weights(plan):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    root, stage, checkpoint = plan['root'], plan['stage'], plan['checkpoint']

    def load(path):
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)

    def equal(actual, expected, name):
        if not isinstance(actual, dict) or not actual or actual.keys() != expected.keys():
            raise ValueError(name + ' tensor keys differ')
        for key in actual:
            a, b = actual[key], expected[key]
            if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                    or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b)):
                raise ValueError(name + ' tensors differ: ' + key)

    models = plan['config']['models']
    state = load(checkpoint / 'training_state.pt')
    rounds = plan['completed']['rounds']
    if state.get('behavior_version') != f'{plan["mode"]}:{rounds}' or state.get('data_cursor') != {'round': rounds}:
        raise ValueError('Training state behavior and cursor differ from COMMITTED')
    if not {'python', 'torch', 'cuda', 'numpy'} <= state.get('rng', {}).keys():
        raise ValueError('Training state lacks complete RNG recovery data')
    for name in ('actor', 'renderer'):
        component = load(checkpoint / (name + '.pt'))
        if (not isinstance(component.get('optimizer'), dict)
                or not component['optimizer'].get('param_groups')
                or not isinstance(component['optimizer'].get('state'), dict)
                or not isinstance(component.get('scheduler'), dict)
                or type(component.get('active')) is not bool):
            raise ValueError(name + ' checkpoint lacks optimizer/scheduler/active state')
        if name == 'actor':
            if set(component.get('model', {})) != {'policy', 'reference'}:
                raise ValueError('Actor checkpoint must contain policy and frozen reference')
            equal(component['model']['policy'], load_file(str(adapter_file(stage / 'actor_policy')), device='cpu'), 'Final actor policy')
            equal(component['model']['reference'], load_file(str(adapter_file(root / models['actor_reference'])), device='cpu'), 'Frozen actor reference')
        else:
            equal(component['model'], load(stage / 'renderer_adapters/policy.pt'), 'Final renderer policy')
            equal(load(stage / 'renderer_adapters/reference.pt'), load(root / models['renderer_reference']), 'Frozen renderer reference')
        del component
    return {'actual_checkpoint_tensors_checked': True, 'final_policies_match_checkpoint': True,
        'frozen_reference_tensors_match': True, 'optimizer_scheduler_rng_present': True}


def collect_export(root, run, mode, expected_updates=32):
    root = Path(root).resolve()
    run = Path(run)
    run = run if run.is_absolute() else root / run
    run = run.resolve()
    if not run.is_relative_to(root):
        raise ValueError('Training run must belong to this project')
    stage = run / ('rl_' + mode)
    plan = validate_history(stage, mode, expected_updates)
    plan.update(root=root, run=run, stage=stage, mode=mode)
    models = plan['config']['models']
    references_path = run / 'sft_reference/references.json'
    references = read_json(references_path)
    for key in ('actor_reference', 'renderer_reference'):
        if (root / references.get(key, '')).resolve() != (root / models.get(key, '')).resolve():
            raise ValueError('Frozen reference identity differs from committed configuration')
    sampler_path = run / 'sampler_selection.json'
    sampler = read_json(sampler_path)
    if sampler.get('accepted') is not True or sampler.get('renderer_policy') != models.get('renderer_policy'):
        raise ValueError('Sampler selection differs from the initial renderer policy')
    if any(sampler.get(key) != plan['config']['renderer'][key] for key in ('steps', 'eta', 'cfg_scale')):
        raise ValueError('Sampler settings differ from the committed configuration')
    files = {stage / 'training_completed.json', stage / 'metrics.jsonl', sampler_path, run / 'rl_config.yaml'}
    files.update(path / 'COMMITTED.json' for path, _ in plan['commits'].values())
    files.update(plan['checkpoint'] / name for name in ('actor.pt', 'renderer.pt', 'training_state.pt'))
    files.update(path for path in stage.glob('*.json*') if path.is_file())
    latest_hint = stage / 'checkpoints/LATEST.json'
    if latest_hint.is_file():
        files.add(latest_hint)
    directories = [stage / 'actor_policy', stage / 'renderer_adapters', run / 'sft_reference']
    for key in ('actor_policy', 'actor_reference'):
        directory = root / models[key]
        adapter_file(directory)
        directories.append(directory)
    for key in ('renderer_policy', 'renderer_reference'):
        files.add(root / models[key])
    for directory in directories:
        if not directory.is_dir():
            raise ValueError('Required adapter directory is missing: ' + str(directory))
        files.update(path for path in directory.rglob('*') if path.is_file() or path.is_symlink())
    files = sorted({require_file(path, root) for path in files})
    for path in files:
        relative = path.relative_to(root)
        if any(part in {'renders', 'bundles', 'encoding_cache'} for part in relative.parts):
            raise ValueError('Shared mutable render data must not enter this recovery export')
        if relative.parts[0] == 'models':
            raise ValueError('Base model directories are separate backup dependencies')
    plan.update(files=files, references=references, before=snapshot(files))
    plan['weight_validation'] = validate_weights(plan)
    if snapshot(files) != plan['before']:
        raise RuntimeError('Completed sources changed during validation')
    return plan


class RateReader:
    def __init__(self, stream, clock):
        self.stream = stream
        self.clock = clock

    def read(self, size=-1):
        data = self.stream.read(size)
        self.clock['bytes'] += len(data)
        time.sleep(max(0, self.clock['bytes'] / self.clock['rate'] - (time.monotonic() - self.clock['start'])))
        return data


def export_completed(root, run, mode, output, manifest, expected_updates=32, read_mib_per_second=16, set_priority=True):
    root = Path(root).resolve()
    output, manifest = root / output, root / manifest
    if (not output.is_relative_to(root) or not manifest.is_relative_to(root)
            or '..' in output.parts or '..' in manifest.parts or not math.isfinite(read_mib_per_second)
            or read_mib_per_second <= 0):
        raise ValueError('Outputs must belong to the project and read rate must be positive')
    if set_priority:
        low_priority()
    started = time.time()
    plan = collect_export(root, run, mode, expected_updates)
    if output.exists():
        prior = read_json(manifest)
        if (prior.get('status') == 'completed' and prior.get('stage') == mode
                and prior.get('checkpoint') == str(plan['checkpoint']) and prior.get('files') == plan['before']
                and prior.get('source') == str(output) and output.stat().st_size == prior.get('bytes')
                and output.stat().st_mtime <= prior.get('finished_at', 0)):
            return prior
        raise FileExistsError('An existing archive lacks matching completed export evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.part')
    sink = DigestSink(temporary)
    clock = {'bytes': 0, 'rate': read_mib_per_second * 2**20, 'start': time.monotonic()}
    try:
        with gzip.GzipFile(filename='', mode='wb', fileobj=sink, compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for path in plan['files']:
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                    with path.open('rb') as stream:
                        archive.addfile(info, RateReader(stream, clock))
                config = json.dumps(plan['config'], indent=2).encode()
                info = tarfile.TarInfo(str(plan['stage'].relative_to(root) / 'committed_restore_config.json'))
                info.size = len(config)
                archive.addfile(info, io.BytesIO(config))
        sink.finish()
        if snapshot(plan['files']) != plan['before']:
            raise RuntimeError('Completed training sources changed during export')
        after = validate_history(plan['stage'], mode, expected_updates)
        if after['checkpoint'] != plan['checkpoint'] or after['completed'] != plan['completed']:
            raise RuntimeError('Completed stage identity changed during export')
        temporary.replace(output)
    except BaseException:
        if not sink.stream.closed:
            sink.stream.close()
        raise
    result = {'schema': 'rl_recovery_export_v1', 'status': 'completed', 'stage': mode, 'source': str(output), 'bytes': sink.size,
        'mtime': int(output.stat().st_mtime), 'sha256': sink.digest.hexdigest(),
        'rounds': plan['completed']['rounds'], 'required_effective_updates': expected_updates,
        'effective_target_met': True, 'effective_update_counts': plan['completed']['effective_update_counts'],
        'checkpoint': str(plan['checkpoint']), 'files': plan['before'], 'frozen_references': plan['references'],
        'committed_config': plan['config'], 'weight_validation': plan['weight_validation'],
        'historical_committed_markers': len(plan['commits']), 'includes_optimizer_scheduler_rng_and_sampler': True,
        'includes_final_actor_policy_and_renderer_adapters': True, 'includes_initial_adapter_dependencies': True,
        'shared_renders_included': False, 'separate_backup_dependencies': ['base models', 'task and calibration data', 'full rollout and shared render records'],
        'source_files_unchanged': True, 'checkpoints_deleted': False,
        'read_mib_per_second_limit': read_mib_per_second, 'started_at': started, 'finished_at': time.time()}
    write_json(manifest, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/full-actor-rl'))
    parser.add_argument('--mode', choices=('agent', 'joint'), required=True)
    parser.add_argument('--expected-effective-updates', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--read-mib-per-second', type=float, default=16)
    args = parser.parse_args()
    result = export_completed(Path.cwd(), args.run, args.mode, args.output, args.manifest,
        args.expected_effective_updates, args.read_mib_per_second)
    print(json.dumps({key: result[key] for key in ('status', 'stage', 'source', 'bytes', 'sha256', 'checkpoint')}))


if __name__ == '__main__':
    main()
