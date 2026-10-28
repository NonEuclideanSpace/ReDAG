import argparse
import json
import os
from pathlib import Path
import random
import select
import signal
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve(strict=True)
    value = path.stat()
    if not path.is_file() or value.st_size == 0:
        raise ValueError(f'Missing nonempty checkpoint file: {path}')
    return {'path': str(path), 'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'inode': value.st_ino, 'device': value.st_dev}


def primary_complete(project):
    path = project / 'data/actor-full-sft/training_completed.json'
    value = read_json(path)
    if (value.get('status') != 'completed' or value.get('updates') != 200
            or value.get('scope') != 'full_four_phase_actor_sft_v1'):
        raise ValueError('Primary full actor SFT has not completed 200 updates')
    return identity(path)


def checkpoint_snapshot(project, load_state, seed=23):
    run = project / f'runs/actor-full-seed{seed}'
    output = project / f'data/actor-full-sft-seed{seed}'
    registration_path = output / 'registration.json'
    registration = read_json(registration_path)
    expected = {'scope': 'independent_actor_sft_seed_replication_v1', 'updates': 200,
                'training_seed': seed, 'independent_family_expansion': False,
                'downstream_rl_planned': False, 'phase_counts': [5, 7, 5, 3]}
    if any(registration.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Independent seed {seed} registration does not match the 200-update experiment')
    if not registration.get('admission_identity_sha256'):
        raise ValueError('Independent admission identity is absent')
    if (output / 'training_completed.json').exists():
        raise ValueError('Independent training has already completed')
    metrics_path = run / 'sft_actor/metrics.jsonl'
    raw = metrics_path.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('Metrics have an incomplete trailing line')
    metrics = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (not metrics or [row['step'] for row in metrics] != list(range(1, len(metrics) + 1))
            or any(row.get('stage') != 'actor' or len(row.get('record_ids', [])) != 20 for row in metrics)):
        raise ValueError('Metrics are not complete contiguous actor updates')
    step = metrics[-1]['step']
    if not 0 < step < 200:
        raise ValueError('Only an unfinished committed step can be paused')
    checkpoints = run / 'sft_actor/checkpoints'
    commits = []
    for marker in checkpoints.glob('*/COMMITTED.json'):
        value = read_json(marker)
        commits.append((value.get('committed_ns', marker.stat().st_mtime_ns), marker.parent.name, marker, value))
    if not commits:
        raise ValueError('No committed checkpoint is available')
    _, transaction_id, marker, commit = max(commits, key=lambda value: (value[0], value[1]))
    checkpoint = marker.parent.resolve()
    if (commit.get('status') != 'COMMITTED' or commit.get('transaction_id') != transaction_id
            or commit.get('new_version') != f'actor:{step}'
            or commit.get('components') != {'actor': 'actor.pt'}
            or commit.get('optimizer_steps') != ['actor']
            or Path(metrics[-1]['checkpoint']).resolve() != checkpoint):
        raise ValueError('Latest committed checkpoint and last complete metric do not match')
    state_path = checkpoint / 'training_state.pt'
    state = load_state(state_path)
    cursor = state.get('data_cursor', {})
    if cursor.get('step') != step or state.get('behavior_version') != f'actor:{step}':
        raise ValueError('Checkpoint data cursor does not match the committed metric')
    random.Random().setstate(cursor['sampler']['rng'])
    rng = state.get('rng', {})
    if not {'python', 'torch', 'cuda', 'numpy'} <= rng.keys():
        raise ValueError('Checkpoint RNG state is incomplete')
    random.Random().setstate(rng['python'])
    actor_path = checkpoint / 'actor.pt'
    actor = load_state(actor_path)
    if not actor.get('model') or actor.get('optimizer') is None or actor.get('scheduler') is None:
        raise ValueError('Committed actor model, optimizer or scheduler is incomplete')
    paths = [registration_path, metrics_path, marker, state_path, actor_path]
    latest = checkpoints / 'LATEST.json'
    if latest.exists():
        hint = read_json(latest)
        if hint != {'transaction_id': transaction_id, 'version': f'actor:{step}'}:
            raise ValueError('LATEST hint and actual latest commit disagree')
        paths.append(latest)
    return {'step': step, 'target_updates': 200, 'transaction_id': transaction_id,
            'checkpoint': str(checkpoint), 'admission_identity_sha256': registration['admission_identity_sha256'],
            'sampler_cursor_valid': True, 'optimizer_scheduler_rng_preserved': True,
            'files': [identity(path) for path in paths]}


def process_identity(project, pid, seed=23):
    proc = Path('/proc') / str(pid)
    args = [value.decode() for value in (proc / 'cmdline').read_bytes().split(b'\0') if value]
    executable = (proc / 'exe').resolve(strict=True)
    cwd = (proc / 'cwd').resolve(strict=True)
    if not executable.name.startswith('python') or cwd != project:
        raise ValueError('PID is not a Python trainer in the expected project')
    position = 2 if len(args) > 1 and args[1] == '-u' else 1
    if len(args) <= position or (cwd / args[position]).resolve() != project / 'scripts/train_full_actor_sft.py':
        raise ValueError('PID is not the direct full actor trainer')
    options = args[position + 1:]
    expected = {'--config': f'configs/actor-full-seed{seed}.yaml', '--output': f'data/actor-full-sft-seed{seed}',
                '--run-dir': f'runs/actor-full-seed{seed}', '--updates': '200',
                '--device': {23: 'cuda:1', 47: 'cuda:0'}[seed]}
    for option, value in expected.items():
        if options.count(option) != 1:
            raise ValueError(f'Trainer must explicitly identify {option}')
        actual = options[options.index(option) + 1]
        equal = actual == value if option in {'--updates', '--device'} else (cwd / actual).resolve() == project / value
        if not equal:
            raise ValueError(f'Trainer identity differs at {option}')
    if options.count('--sft-only') != 1:
        raise ValueError('Only the independent SFT-only trainer can be paused')
    status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
    term_mask = 1 << (signal.SIGTERM - 1)
    if int(status['TracerPid']) or any(int(status[key].strip(), 16) & term_mask for key in ('SigCgt', 'SigIgn', 'SigBlk')):
        raise ValueError('SIGTERM must have unblocked default disposition without a tracer')
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19], 'executable': str(executable),
            'cwd': str(cwd), 'argv': args, 'state': status['State'].strip().split()[0]}


def stopped(pid):
    tasks = list((Path('/proc') / str(pid) / 'task').glob('*/status'))
    return bool(tasks) and all(next(line for line in path.read_text().splitlines()
                                   if line.startswith('State:')).split()[1] == 'T' for path in tasks)


def write_proof(path, proof):
    with path.open('x') as stream:
        json.dump(proof, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def pause_timeout(signum, frame):
    raise TimeoutError('Stopped-state inspection exceeded 15 seconds')


def same_process(first, second):
    return {key: value for key, value in first.items() if key != 'state'} == {
        key: value for key, value in second.items() if key != 'state'}


def guarded_signal(handle, project, expected, kind, seed=23):
    current = (process_identity(project, expected['pid']) if seed == 23
               else process_identity(project, expected['pid'], seed=seed))
    if not same_process(current, expected):
        raise ValueError('Trainer process identity changed before signal')
    if handle is not None:
        signal.pidfd_send_signal(handle, kind)
    else:
        os.kill(expected['pid'], kind)


def process_exited(expected):
    try:
        fields = (Path('/proc') / str(expected['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return True
    return fields[19] != expected['start_ticks'] or fields[0] == 'Z'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--seed', type=int, choices=(23, 47), default=23)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    import torch
    load_state = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    before = process_identity(project, args.pid, seed=args.seed)
    if before['state'] in {'T', 't', 'Z'}:
        raise ValueError('Trainer must be live and not already stopped')
    if not args.pause:
        print(json.dumps({'status': 'inspect_only', 'process': before,
                          'checkpoint': checkpoint_snapshot(project, load_state, seed=args.seed)}, indent=2))
        return
    primary = primary_complete(project)
    supports_pidfd = hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal')
    handle = os.pidfd_open(args.pid, 0) if supports_pidfd else None
    previous_alarm = signal.signal(signal.SIGALRM, pause_timeout)
    sent_stop = False
    termination_requested = False
    try:
        current = process_identity(project, args.pid, seed=args.seed)
        if not same_process(current, before):
            raise ValueError('Trainer process identity changed')
        signal.setitimer(signal.ITIMER_REAL, 15)
        guarded_signal(handle, project, before, signal.SIGSTOP, seed=args.seed)
        sent_stop = True
        deadline = time.monotonic() + 1
        while not stopped(args.pid):
            if time.monotonic() >= deadline:
                raise TimeoutError('Trainer did not enter a complete group stop')
            time.sleep(0.02)
        frozen = process_identity(project, args.pid, seed=args.seed)
        snapshot = checkpoint_snapshot(project, load_state, seed=args.seed)
        proof = {'status': 'validated_before_termination', 'created_at': time.time(),
                 'process': frozen, 'primary_completion': primary, 'checkpoint': snapshot,
                 'resume_required': True, 'resume_arguments': ['--resume', '--sft-only', '--updates', '200']}
        path = project / f'runs/actor-full-seed{args.seed}' / f'pause-proof-{args.pid}-{time.time_ns()}.json'
        write_proof(path, proof)
        guarded_signal(handle, project, before, signal.SIGTERM, seed=args.seed)
        termination_requested = True
        if not process_exited(before):
            guarded_signal(handle, project, before, signal.SIGCONT, seed=args.seed)
        sent_stop = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        if handle is not None:
            poller = select.poll()
            poller.register(handle, select.POLLIN)
            exited = bool(poller.poll(5000))
        else:
            deadline = time.monotonic() + 5
            while not process_exited(before) and time.monotonic() < deadline:
                time.sleep(0.02)
            exited = process_exited(before)
        result = {'status': 'paused_for_resume' if exited else 'termination_unconfirmed',
                  'step': snapshot['step'], 'proof': str(path), 'process_exited': exited}
        print(json.dumps(result, indent=2))
        if not exited:
            raise RuntimeError('Termination is unconfirmed; do not assume GPU memory was released')
    except BaseException as error:
        if sent_stop:
            try:
                guarded_signal(handle, project, before, signal.SIGCONT, seed=args.seed)
            except (ProcessLookupError, FileNotFoundError):
                pass
        print(json.dumps({'status': 'termination_requested' if termination_requested else 'not_paused',
                          'error': f'{type(error).__name__}: {error}'}), file=sys.stderr)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if handle is not None:
            os.close(handle)


if __name__ == '__main__':
    main()
