import json
import math
from pathlib import Path
import time


def read_samples(path, now=None):
    now = time.time() if now is None else now
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, Path(path).stat().st_size - 1000000))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if now - 630 <= row['time'] <= now:
                rows.append(row)
        except (ValueError, KeyError, TypeError):
            pass
    return sorted(rows, key=lambda row: row['time'])


def efficiency(samples, now=None, windows=(300, 600), maximum_gap=15):
    now = time.time() if now is None else now
    devices = sorted({str(index) for row in samples for index in row['gpus']})
    report = {}
    for device in devices:
        report[device] = {}
        for window in windows:
            observed = weighted = high_load = idle = 0.0
            for index, row in enumerate(samples):
                value = row['gpus'].get(device, row['gpus'].get(int(device), {})).get('utilization')
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                following = samples[index + 1]['time'] if index + 1 < len(samples) else now
                if following - row['time'] > maximum_gap:
                    continue
                start = max(now - window, row['time'])
                end = min(now, following)
                duration = max(0, end - start)
                observed += duration
                weighted += duration * min(100, max(0, value))
                high_load += duration * (value >= 80)
                idle += duration * (value <= 5)
            report[device][str(window)] = {
                'mean_utilization': weighted / observed if observed else None,
                'high_load_time_percent': high_load / observed * 100 if observed else None,
                'idle_time_percent': idle / observed * 100 if observed else None,
                'observed_seconds': round(observed, 2),
                'coverage_percent': min(100, observed / window * 100),
                'unknown_seconds': round(max(0, window - observed), 2),
                'unknown_time_percent': max(0, 100 - observed / window * 100),
            }
    return {'updated_at': now, 'threshold': 80, 'windows_seconds': list(windows),
            'maximum_sample_gap_seconds': maximum_gap, 'gpus': report}


def choose_reserve_groups(groups, jobs, statuses, metrics, now, last_admission,
                          threshold=80, cooldown=120, maximum_backlog=2):
    known = {job['id'] for job in jobs}
    selected = []
    selected_devices = set()
    for group in groups:
        devices = group['gpus']
        if any(job['id'] in known for job in group['jobs']):
            continue
        if any(device in selected_devices for device in devices):
            continue
        eligible = True
        for device in devices:
            row = metrics.get('gpus', {}).get(str(device), {}).get('600', {})
            mean = row.get('mean_utilization')
            backlog = sum(device in job.get('gpus', []) and bool(job.get('command'))
                          and statuses.get(job['id']) in {'pending', 'running'}
                          for job in jobs)
            eligible = eligible and mean is not None and mean < threshold
            eligible = eligible and row.get('observed_seconds', 0) >= 60
            eligible = eligible and now - last_admission.get(str(device), 0) >= cooldown
            eligible = eligible and backlog < maximum_backlog
        if eligible:
            selected.append(group)
            selected_devices.update(devices)
            known.update(job['id'] for job in group['jobs'])
    return selected
