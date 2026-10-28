import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from export_completed_rl import read_json, require_file, validate_history


def prepare(project, start, count=16):
    if type(start) is not int or start not in (0, 16) or type(count) is not int or count != 16:
        raise ValueError('Select exactly 16 checkpoints starting at 0 or 16')
    project = Path(project).resolve(strict=True)
    stage = project / 'runs/full-actor-rl/rl_joint'
    completed = read_json(require_file(stage / 'training_completed.json', project))
    expected_counts = {'actor': 32, 'renderer': 32, 'required_components': 32}
    if (completed.get('status') != 'completed' or completed.get('rounds') != 32
            or completed.get('effective_update_counts') != expected_counts):
        raise ValueError('Joint training must have completed exactly 32 effective updates of both components')
    metrics_path = require_file(stage / 'metrics.jsonl', project)
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(metrics) != 32 or [row.get('round') for row in metrics] != list(range(1, 33)):
        raise ValueError('Metrics must contain exactly 32 unique ordered rounds')
    checkpoints = []
    for row in metrics:
        source = row.get('checkpoint')
        if not isinstance(source, str) or not source:
            raise ValueError('Every metric must identify its actual checkpoint directory')
        directory = Path(source)
        if not directory.is_absolute():
            directory = project / directory
        marker = require_file(directory / 'COMMITTED.json', project)
        if marker.parent.parent != stage / 'checkpoints':
            raise ValueError('Checkpoint must be directly inside the completed joint stage')
        if marker.parent in checkpoints:
            raise ValueError('Distinct effective updates must identify distinct COMMITTED directories')
        checkpoints.append(marker.parent)
    history = validate_history(stage, 'joint', expected_updates=32)
    for _, commit in history['commits'].values():
        for role in ('actor', 'renderer'):
            gradient = commit.get('gradients', {}).get(role, {}).get('gradient_norm')
            change = commit.get('parameter_changes', {}).get(role, {})
            values = (gradient, change.get('l2'), change.get('max_abs'))
            if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values)
                    or type(change.get('changed_tensors')) is not int or change['changed_tensors'] <= 0):
                raise ValueError('Every selected training round requires finite effective updates of both components')
    selected = []
    for offset, directory in enumerate(checkpoints[start:start + count], start + 1):
        sizes = {name: require_file(directory / name, project).stat().st_size
            for name in ('actor.pt', 'renderer.pt', 'training_state.pt')}
        selected.append({'round': offset, 'group': directory.relative_to(project).as_posix(),
            'payload_bytes': sizes})
    output = project / 'delivery/project-results'
    command = [sys.executable, str(Path(__file__).with_name('package_rental_results.py')),
        '--project', str(project), '--output', str(output), '--read-mib-per-second', '16',
        '--chunk-mib', '256', '--compression', 'gzip', '--lock-wait-seconds', '120']
    for item in selected:
        command.extend(('--only-group', item['group']))
    report = {'schema': 'completed_joint_checkpoint_archive_scope_v1',
        'status': 'selected_scope_validated', 'project': str(project), 'start': start,
        'count': count, 'effective_update_counts': expected_counts, 'selected': selected,
        'selected_payload_bytes': sum(sum(item['payload_bytes'].values()) for item in selected),
        'output': str(output), 'metadata_checks_only': True,
        'payload_contents_loaded': False, 'full_project_delivery_complete': False,
        'command': command}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--start', type=int, choices=(0, 16), required=True)
    parser.add_argument('--count', type=int, choices=(16,), default=16)
    args = parser.parse_args(argv)
    report = prepare(args.project, args.start, args.count)
    report['free_bytes_at_execution'] = shutil.disk_usage(report['project']).free
    report['required_free_bytes'] = 42_000_000_000
    print(json.dumps(report, allow_nan=False), flush=True)
    if report['free_bytes_at_execution'] < report['required_free_bytes']:
        raise RuntimeError('At least42GB free space is required before archiving16 joint checkpoints')
    result = subprocess.run(report['command'], cwd=report['project'], check=False)
    print(json.dumps({'status': 'selected_scope_packer_exited', 'start': args.start,
        'count': args.count, 'returncode': result.returncode,
        'full_project_delivery_complete': False}), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
