import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from continue_gpu_queue_once import process_identity, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--owner', default='full-joint-rl32')
    parser.add_argument('--minimum-free-gib', type=float, default=3)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if not args.job.startswith('development64-controller-') or not 0 < args.seconds <= 3600:
        raise ValueError('Only a bounded existing controller evaluation can be guarded')
    if not 3 <= args.minimum_free_gib <= 10:
        raise ValueError('The free-memory guard must be between3 and10 GiB')
    root = Path.cwd()
    output = root / 'deployment' / (args.job + '.overlap-guard.json')
    if output.exists():
        raise FileExistsError('Inspect the existing guard evidence before starting another guard')
    jobs = json.loads((root / 'deployment/gpu-jobs.json').read_text())['jobs']
    job = next(row for row in jobs if row['id'] == args.job)
    owner = next(row for row in jobs if row['id'] == args.owner)
    if 'scripts/evaluate_development_controller.py' not in job['command']:
        raise ValueError('The guarded command must be the existing evaluation entry point')
    state = {'status': 'waiting_for_worker', 'guard_pid': os.getpid(), 'job': args.job,
        'owner': args.owner, 'started_at': time.time(), 'samples': 0, 'minimum_free_gib': None,
        'threshold_gib': args.minimum_free_gib}
    deadline = time.monotonic() + args.seconds
    identity = None
    wrapper = None
    low = 0
    save(output, state)
    while time.monotonic() < deadline:
        exit_file = root / job['exit_file']
        if exit_file.exists():
            outcome = json.loads(exit_file.read_text())
            code = outcome.get('returncode') if isinstance(outcome, dict) else None
            guard_code = 0 if type(code) is int and code == 0 else 1
            state.update(status='worker_exited', exit=outcome, guard_exit_code=guard_code, finished_at=time.time())
            save(output, state)
            return guard_code
        if identity is None:
            pid_file = root / job['pid_file']
            if pid_file.exists():
                wrapper = int(pid_file.read_text())
                children = []
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():
                        continue
                    try:
                        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                        if int(fields[1]) == wrapper:
                            current = process_identity(int(proc.name))
                            if current and current.get('cmdline') == job['command']:
                                children.append(current)
                    except (OSError, ValueError):
                        continue
                if len(children) == 1:
                    identity = children[0]
                    state.update(status='monitoring', worker=identity, wrapper_pid=wrapper)
            save(output, state)
            time.sleep(1)
            continue
        reason = None
        owner_exit = root / owner['exit_file']
        if owner_exit.exists() and json.loads(owner_exit.read_text())['returncode'] != 0:
            reason = 'primary_training_failed'
        try:
            value = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True, timeout=5)
            free = float(value.strip()) / 1024
            state.update(updated_at=time.time(), latest_free_gib=free, samples=state['samples'] + 1)
            state['minimum_free_gib'] = free if state['minimum_free_gib'] is None else min(free, state['minimum_free_gib'])
            low = low + 1 if free < args.minimum_free_gib else 0
            if low >= 2:
                reason = 'gpu1_free_memory_below_guard_on_two_samples'
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            state.update(last_observation_error=str(error)[-300:], last_observation_error_at=time.time())
        if reason:
            current = process_identity(identity['pid'])
            if current is None or any(current.get(key) != identity.get(key) for key in ('start_ticks', 'cmdline', 'cwd', 'exe')):
                raise RuntimeError('Evaluation identity changed; no signal was sent')
            fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) != wrapper:
                raise RuntimeError('Evaluation parent changed; no signal was sent')
            os.kill(identity['pid'], signal.SIGTERM)
            state.update(status='withdrawn_for_primary', reason=reason, signal='SIGTERM', guard_exit_code=1, finished_at=time.time())
            save(output, state)
            return 1
        save(output, state)
        time.sleep(5)
    state.update(status='bounded_watch_ended', guard_exit_code=1, finished_at=time.time())
    save(output, state)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
