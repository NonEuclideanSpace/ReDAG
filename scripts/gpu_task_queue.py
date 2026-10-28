import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_efficiency import choose_reserve_groups, efficiency, read_samples


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, ValueError, IndexError, TypeError):
        return False


def job_pid(job):
    try:
        return int(Path(job['pid_file']).read_text().strip())
    except (OSError, ValueError, KeyError):
        return None


def outcome(job):
    value = read(job.get('exit_file', ''))
    if isinstance(value, dict):
        return value.get('returncode', value.get('exit_code'))
    return value if isinstance(value, int) else None


def ancestors(pid):
    found = set()
    while pid and pid not in found:
        found.add(pid)
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return found


def gpu_state(jobs):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=5)
    rows = {}
    for line in text.splitlines():
        index, uuid, free, utilization = (part.strip() for part in line.split(','))
        rows[int(index)] = {'uuid': uuid, 'free_gib': float(free) / 1024, 'utilization': float(utilization), 'busy': False}
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=5)
    owners = [(job_pid(job), set(job.get('gpus', []))) for job in jobs]
    for line in text.splitlines():
        pid, uuid, memory = (part.strip() for part in line.split(','))
        chain = ancestors(int(pid))
        expected = set().union(*(devices for parent, devices in owners if parent in chain))
        for index, row in rows.items():
            if row['uuid'] == uuid:
                context_only = bool(expected) and index not in expected and float(memory) < 1024
                row['busy'] = row['busy'] or not context_only
    for job in jobs:
        if outcome(job) is None and alive(job_pid(job)):
            for device in job.get('gpus', []):
                if device in rows:
                    rows[device]['busy'] = True
    return rows


def running_reservation_rule(job):
    rule = job.get('allow_with_running_reservation')
    if not isinstance(rule, dict) or not isinstance(rule.get('job_id'), str) or not rule['job_id']:
        return None
    devices = rule.get('gpus')
    if (not isinstance(devices, list) or not devices
            or any(type(device) is not int or device < 0 for device in devices)):
        return None
    return rule


def prerequisite_reasons(job, statuses):
    blocked = []
    if 'allow_with_running_reservation' in job:
        rule = running_reservation_rule(job)
        if rule is None:
            blocked.append('invalid allow_with_running_reservation rule')
        elif statuses.get(rule['job_id'], {}).get('status') not in {'running', 'completed'}:
            blocked.append(f'awaiting running or completed reservation owner {rule["job_id"]}')
    for dependency in job.get('after', []):
        if statuses.get(dependency, {}).get('status') != 'completed':
            blocked.append(f'awaiting {dependency}')
    for path in job.get('requires', []):
        if not Path(path).is_file():
            blocked.append(f'missing {path}')
    for gate in job.get('gates', []):
        value = read(gate['path'], {})
        for key in gate['key'].split('.'):
            value = value.get(key) if isinstance(value, dict) else None
        if value != gate['equals']:
            blocked.append(gate.get('reason', f'gate pending: {gate["path"]}'))
    if not job.get('command'):
        blocked.append(job.get('reason', 'command not prepared'))
    return blocked


def resource_reasons(job, statuses, gpu):
    blocked = []
    group = job.get('exclusive_group')
    if group and any(key != job['id'] and value.get('exclusive_group') == group and value['status'] == 'running' for key, value in statuses.items()):
        blocked.append(f'active worker in {group}; startup memory is not yet visible')
    for device in job.get('gpus', []):
        state = gpu.get(device)
        if state is None:
            blocked.append(f'GPU {device} unavailable')
        elif state['free_gib'] < job.get('minimum_free_gib', 20):
            blocked.append(f'GPU {device} needs more free memory')
        elif state['busy'] and not job.get('allow_overlap', False):
            blocked.append(f'GPU {device} occupied')
    return blocked


def ready_gpu_reservations(jobs, statuses, gpu):
    reservations = {}
    reserved_devices = set()
    for job in sorted(jobs, key=lambda item: item.get('priority', 100)):
        devices = set(job.get('gpus', []))
        status = statuses[job['id']]['status']
        if (job.get('reserve_gpus_when_ready') is not True or job.get('external')
                or status not in {'pending', 'running'}
                or (status == 'pending' and prerequisite_reasons(job, statuses)) or not devices
                or devices & reserved_devices):
            continue
        reservations[job['id']] = {
            'job_id': job['id'], 'gpus': sorted(devices),
            'priority': job.get('priority', 100),
            'resource_reasons': resource_reasons(job, statuses, gpu) if status == 'pending' else [],
            'status': status,
        }
        reserved_devices.update(devices)
    return reservations


def reasons(job, statuses, gpu, reservations=None):
    blocked = prerequisite_reasons(job, statuses) + resource_reasons(job, statuses, gpu)
    rule = running_reservation_rule(job)
    for reservation in (reservations or {}).values():
        if (reservation['job_id'] != job['id']
                and job.get('priority', 100) >= reservation['priority']):
            devices = job.get('gpus', [])
            if (rule is not None and rule['job_id'] == reservation['job_id']
                    and reservation.get('status') == 'running'
                    and statuses.get(rule['job_id'], {}).get('status') == 'running'
                    and isinstance(devices, list) and devices
                    and all(type(device) is int for device in devices)
                    and set(devices) <= set(rule['gpus'])
                    and set(devices) <= set(reservation['gpus'])):
                continue
            for device in sorted(set(job.get('gpus', [])) & set(reservation['gpus'])):
                blocked.append(f'GPU {device} reserved for ready job {reservation["job_id"]} (priority {reservation["priority"]})')
    return blocked


def run_job(job):
    save(job['started_file'], {'started_at': time.time(), 'command': job['command']})
    try:
        result = subprocess.run(job['command'], env={**os.environ, 'PYTHONPATH': 'src', **job.get('env', {})})
        value = {'returncode': result.returncode, 'finished_at': time.time()}
    except Exception as error:
        value = {'returncode': 127, 'finished_at': time.time(), 'error': str(error)}
    save(job['exit_file'], value)


def supervise(path, hours):
    lock = open('deployment/gpu-queue.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + hours * 3600
    previous = None
    probe_failures = 0
    consecutive_failures = 0
    last_success = None
    while time.monotonic() < deadline:
        configuration = read(path, {})
        jobs = configuration['jobs']
        try:
            gpu = gpu_state(jobs)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            probe_failures += 1
            consecutive_failures += 1
            health = {'status': 'gpu_probe_failed', 'supervisor_pid': os.getpid(),
                'updated_at': time.time(), 'last_success_at': last_success,
                'probe_failures': probe_failures, 'consecutive_failures': consecutive_failures,
                'error': f'{type(error).__name__}: {error}'[:1500], 'new_jobs_admitted': False}
            save('deployment/gpu-queue-health.json', health)
            with open('deployment/gpu-queue-probe-errors.jsonl', 'a') as stream:
                stream.write(json.dumps(health) + '\n')
            print(json.dumps(health), flush=True)
            time.sleep(min(5, max(0, deadline - time.monotonic())))
            continue
        last_success = time.time()
        consecutive_failures = 0
        save('deployment/gpu-queue-health.json', {'status': 'healthy', 'supervisor_pid': os.getpid(),
            'updated_at': last_success, 'last_success_at': last_success,
            'probe_failures': probe_failures, 'consecutive_failures': 0})
        with open('deployment/gpu-utilization.jsonl', 'a') as usage:
            usage.write(json.dumps({'time': time.time(), 'gpus': gpu}) + '\n')
        now = time.time()
        metrics = efficiency(read_samples('deployment/gpu-utilization.jsonl', now), now)
        reserve = read('deployment/gpu-reserve.json', {'enabled': False, 'groups': []})
        refill = read('deployment/gpu-auto-fill-state.json', {'last_admission': {}, 'actions': []})
        basic = {}
        for job in jobs:
            code = outcome(job)
            basic[job['id']] = 'completed' if code == 0 else 'failed' if code is not None else 'running' if alive(job_pid(job)) else 'interrupted' if job.get('started_file') and Path(job['started_file']).exists() else 'pending'
        for job in jobs:
            if basic[job['id']] == 'pending' and any(not Path(required).is_file() for required in job.get('requires', [])):
                basic[job['id']] = 'blocked'
        added = choose_reserve_groups(reserve.get('groups', []), jobs, basic, metrics, now, refill['last_admission']) if reserve.get('enabled') else []
        for group in added:
            jobs.extend(group['jobs'])
            for device in group['gpus']:
                refill['last_admission'][str(device)] = now
            refill['actions'].append({'time': now, 'group': group['id'], 'title': group['title'], 'jobs': [job['id'] for job in group['jobs']], 'reason': 'ten_minute_mean_below_80', 'means': {str(device): metrics['gpus'][str(device)]['600']['mean_utilization'] for device in group['gpus']}})
        if added:
            save(path, configuration)
        known_ids = {job['id'] for job in jobs}
        remaining = [group['id'] for group in reserve.get('groups', []) if not any(job['id'] in known_ids for job in group['jobs'])]
        refill.update(enabled=bool(reserve.get('enabled')), updated_at=now, threshold=80, window_seconds=600, remaining_groups=remaining, actions=refill['actions'][-20:])
        save('deployment/gpu-auto-fill-state.json', refill)
        metrics['auto_fill'] = refill
        save('deployment/gpu-efficiency.json', metrics)
        statuses = {}
        for job in jobs:
            code = outcome(job)
            pid = job_pid(job)
            status = 'completed' if code == 0 else 'failed' if code is not None else 'running' if alive(pid) else 'pending'
            if status == 'pending' and job.get('started_file') and Path(job['started_file']).exists():
                status = 'interrupted'
            statuses[job['id']] = {'status': status, 'pid': pid, 'returncode': code, 'gpus': job.get('gpus', []), 'title': job['title'], 'priority': job.get('priority', 100), 'after': job.get('after', []), 'budget': job.get('budget'), 'eta_note': job.get('eta_note'), 'exclusive_group': job.get('exclusive_group')}
        reservations = ready_gpu_reservations(jobs, statuses, gpu)
        for job in sorted(jobs, key=lambda item: item.get('priority', 100)):
            row = statuses[job['id']]
            if row['status'] != 'pending':
                continue
            blocked = reasons(job, statuses, gpu, reservations)
            row['reasons'] = blocked
            if job.get('external'):
                row['status'] = 'awaiting_external_launch'
            elif blocked:
                row['status'] = 'blocked'
            else:
                with open(job['log_file'], 'ab', buffering=0) as log:
                    process = subprocess.Popen([sys.executable, __file__, '--jobs', str(path), '--run-job', job['id']], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                Path(job['pid_file']).write_text(str(process.pid))
                row.update(status='running', pid=process.pid, started_at=time.time())
                if job['id'] in reservations:
                    reservations[job['id']].update(status='running', resource_reasons=[])
                for device in job.get('gpus', []):
                    gpu[device]['busy'] = True
                    gpu[device]['free_gib'] -= job.get('minimum_free_gib', 20)
        state = {'updated_at': time.time(), 'jobs': statuses, 'gpus': gpu, 'reservations': reservations, 'deadline_seconds': max(0, deadline - time.monotonic())}
        save('deployment/gpu-queue-state.json', state)
        fingerprint = json.dumps(statuses, sort_keys=True)
        if fingerprint != previous:
            print(json.dumps(state), flush=True)
            previous = fingerprint
        if all(row['status'] in {'completed', 'failed', 'interrupted'} for row in statuses.values()):
            break
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', default='deployment/gpu-jobs.json')
    parser.add_argument('--hours', type=float, default=4)
    parser.add_argument('--run-job')
    args = parser.parse_args()
    if args.run_job:
        run_job(next(job for job in read(args.jobs)['jobs'] if job['id'] == args.run_job))
    else:
        supervise(Path(args.jobs), args.hours)
