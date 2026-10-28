import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_identity(pid):
    directory = Path('/proc') / str(pid)
    try:
        before = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    identity = {'pid': pid, 'start_ticks': int(before[19]), 'state': before[0]}
    if before[0] == 'Z':
        return identity
    try:
        identity.update(cmdline=(directory / 'cmdline').read_bytes().decode().strip('\0').split('\0'),
            cwd=os.readlink(directory / 'cwd'), exe=os.readlink(directory / 'exe'))
        after = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        if not directory.exists():
            return None
        raise RuntimeError('Process identity became unavailable while its PID still exists')
    if before[19] != after[19]:
        raise RuntimeError('PID changed identity during inspection')
    identity['state'] = after[0]
    return identity


def old_process_state(expected, current, current_boot_id):
    if current_boot_id != expected['boot_id']:
        return 'refused_boot_identity'
    if current is None:
        return 'exited'
    if current['pid'] != expected['pid'] or current['start_ticks'] != expected['start_ticks']:
        return 'refused_pid_identity'
    if current['state'] == 'Z':
        return 'exited'
    if any(current.get(key) != expected[key] for key in ('cmdline', 'cwd', 'exe')):
        return 'refused_process_identity'
    return 'waiting_for_original_supervisor'


def advance(expected, current, current_boot_id, state, persist, launch):
    if state.get('launch_attempted'):
        return state['status']
    status = old_process_state(expected, current, current_boot_id)
    state.update(status=status, updated_at=time.time(), observed_old_process=current)
    if status != 'exited':
        persist(state)
        return status
    state.update(status='launch_committed', launch_attempted=True, old_exit_observed_at=time.time())
    persist(state)
    try:
        pid = launch()
    except Exception as error:
        state.update(status='launch_refused_or_failed', error=f'{type(error).__name__}: {error}')
    else:
        state.update(status='continued', new_supervisor_pid=pid, launched_at=time.time())
    persist(state)
    return state['status']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=16)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.hours <= 16 or not 5 <= args.poll_seconds <= 60:
        raise ValueError('Continuation is limited to 16 hours and polling to 5..60 seconds')
    expected = json.loads(args.identity.read_text())
    root = Path(expected['cwd'])
    command = expected['cmdline'].copy()
    if ('scripts/gpu_task_queue.py' not in command or '--run-job' in command
            or command.count('--hours') != 1 or command.count('--jobs') != 1
            or command[command.index('--jobs') + 1] != 'deployment/gpu-jobs.json'
            or Path(expected['lock_path']) != root / 'deployment/gpu-queue.lock'):
        raise ValueError('The recorded process is not the intended queue supervisor')
    command[command.index('--hours') + 1] = str(args.hours)
    status_path = args.status.resolve()
    if status_path.parent != root / 'deployment':
        raise ValueError('Continuation evidence must stay in this deployment directory')
    with status_path.with_suffix('.lock').open('a+') as watcher_lock:
        fcntl.flock(watcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous.get('launch_attempted'):
            raise RuntimeError('This continuation already committed its only launch attempt')
        if previous and previous.get('old_identity') != expected:
            raise RuntimeError('Existing continuation is bound to another process identity')
        if previous.get('status', '').startswith('refused_') or previous.get('status') == 'bounded_wait_expired':
            raise RuntimeError('This continuation previously refused its identity or exhausted its wait')
        wait_until = min(expected['observed_at'] + expected['deadline_seconds'] + 900, time.time() + 16 * 3600)
        state = {'watcher_pid': os.getpid(), 'started_at': time.time(), 'old_identity': expected,
            'new_command': command, 'new_hours': args.hours, 'wait_deadline_at': wait_until,
            'launch_attempted': False, 'training_jobs_modified': False, 'signals_sent': []}
        children = []

        def launch():
            with Path(expected['lock_path']).open('r+') as queue_lock:
                fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if old_process_state(expected, process_identity(expected['pid']), boot_id()) != 'exited':
                    raise RuntimeError('Original supervisor identity is no longer safely exited')
                fcntl.flock(queue_lock, fcntl.LOCK_UN)
                log = root / 'deployment/gpu-queue-continuation.supervisor.log'
                with log.open('ab') as output:
                    child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                return child.pid

        while time.time() < wait_until:
            result = advance(expected, process_identity(expected['pid']), boot_id(), state,
                lambda value: save(status_path, value), launch)
            if result == 'continued':
                time.sleep(3)
                child = children[0]
                state['new_supervisor_identity'] = process_identity(child.pid)
                state['startup_returncode'] = child.poll()
                if state['startup_returncode'] is not None:
                    state['status'] = 'new_supervisor_exited_during_startup'
                save(status_path, state)
                return
            if result != 'waiting_for_original_supervisor':
                return
            time.sleep(min(args.poll_seconds, max(0, wait_until - time.time())))
        state.update(status='bounded_wait_expired', updated_at=time.time())
        save(status_path, state)


if __name__ == '__main__':
    main()
