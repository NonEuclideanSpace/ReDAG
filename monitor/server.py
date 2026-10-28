import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'last-state.json'
CONTROL_HOSTS = {
    '/tmp/dag-pilot-rental-direct1-ssh': '117.50.181.105',
    '/tmp/dag-pilot-rental-control-ssh': '117.50.181.105',
}
COOLDOWNS = {}
COMMAND = 'python3 /workspace/agentic-image-pilot-small/scripts/monitor_probe.py'
LOCK = threading.Lock()
STATE = {'sample': None, 'received_at': None, 'last_attempt': None, 'error': None, 'progress_at': None, 'control': None}
HISTORY = deque(maxlen=100)
if CACHE.is_file():
    try:
        saved = json.loads(CACHE.read_text())
        STATE.update({key: saved[key] for key in STATE if key in saved})
        HISTORY.extend(row for row in saved.get('history', []) if time.time() - row['sampled_at'] < 600)
    except (OSError, ValueError, KeyError, TypeError):
        pass


def signature(sample):
    return json.dumps({'models': [(row.get('bytes'), row.get('files_complete')) for row in sample['models']], 'logs': [(row['name'], row['bytes']) for row in sample['logs']], 'training': [sample['training'].get('update'), sample['training'].get('stage')], 'dependencies': sample['dependencies'], 'pipeline': sample.get('pipeline')}, sort_keys=True)


def read_sample():
    deadline = time.monotonic() + 12
    for control in CONTROLS:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or COOLDOWNS.get(control, 0) > time.monotonic():
            continue
        if not Path(control).is_socket():
            continue
        command = [
            'ssh', '-S', control, '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=2', '-p', '29410',
        ]
        target = 'root@' + CONTROL_HOSTS[control]
        try:
            subprocess.run([*command, '-O', 'check', target], capture_output=True,
                           text=True, timeout=min(1, remaining), check=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                continue
            result = subprocess.run([*command, target, COMMAND], capture_output=True,
                                    text=True, timeout=remaining, check=True)
            return json.loads(result.stdout), Path(control).name
        except (OSError, ValueError, subprocess.SubprocessError):
            COOLDOWNS[control] = time.monotonic() + 15
    raise ConnectionError('No existing SSH control connection returned a valid probe')


def collect(interval):
    fingerprint = signature(STATE['sample']) if STATE['sample'] else None
    while True:
        started = time.time()
        try:
            sample, control = read_sample()
            progress = signature(sample)
            with LOCK:
                if progress != fingerprint:
                    STATE['progress_at'] = started
                    fingerprint = progress
                STATE.update(sample=sample, received_at=time.time(), last_attempt=started, error=None, control=control)
                HISTORY.append(sample)
                while HISTORY and started - HISTORY[0]['sampled_at'] > 600:
                    HISTORY.popleft()
                temporary = CACHE.with_suffix('.tmp')
                temporary.write_text(json.dumps({**STATE, 'history': list(HISTORY)}))
                temporary.replace(CACHE)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            with LOCK:
                STATE.update(last_attempt=started, error=f'SSH probe unavailable ({type(error).__name__})')
        time.sleep(max(1, interval - (time.time() - started)))


def archive_transfer(status_path, transfer_reports, label, kind):
    try:
        row = json.loads(status_path.read_text())
    except (OSError, ValueError):
        row = {}
    completed = row.get('completed', {})
    verified = {name for name, item in completed.items() if item.get('sha256')
        and (item.get('verified_at') is not None or item.get('reused_verified_download') is True)}
    failures = row.get('failures', [])
    recovered = [item for item in failures if item.get('archive') in verified]
    unresolved = {item.get('archive', str(index)) for index, item in enumerate(failures)
        if item.get('archive') not in verified}
    active = row.get('active', [])
    in_flight = []
    for name in active:
        if not isinstance(name, str) or Path(name).name != name:
            continue
        try:
            part = json.loads((transfer_reports / (name + '.json')).read_text())
            if part.get('received_bytes') is not None:
                in_flight.append(part['received_bytes'])
        except (OSError, ValueError):
            pass
    snapshot = row.get('remote_snapshot') or {}
    return {'name': label, 'kind': kind,
        'status': row.get('status', 'unknown'),
        'bytes': row.get('verified_bytes', sum(item.get('bytes', 0) for item in completed.values()) if row else None),
        'expected_bytes': row.get('known_archive_bytes'), 'bytes_per_second': None,
        'updated_at': row.get('updated_at'), 'files_verified': row.get('verified_archives', len(completed)) if row else None,
        'files_total': len(snapshot['archives']) if isinstance(snapshot.get('archives'), list) else None,
        'files_failed': len(unresolved) if row else None,
        'historical_failures': len(failures) if row else None,
        'recovered_failures': len(recovered) if row else None,
        'active_archives': len(active) if row else None, 'active_received_bytes': sum(in_flight) if in_flight else None,
        'packing_status': snapshot.get('status'), 'full_delivery_complete': row.get('full_delivery_complete'),
        'stale': bool(row) and row.get('status') == 'running' and time.time() - row.get('updated_at', 0) > 120}


def local_transfers():
    deployment = ROOT.parent / 'deployment'
    workspace_deployment = ROOT.parent.parent / 'deployment'
    reports, environment_rows = [], []
    sources = [('public-model-backup-status.json', '公开 BF16 基础权重', 'downloaded_bytes', 'recent_bytes_per_second'),
        ('chunk-env-py312-status.json', 'Linux 主环境归档', 'received_bytes', 'average_bytes_per_second'),
        ('chunk-env-tools-status.json', 'Linux 工具环境归档', 'received_bytes', 'average_bytes_per_second'),
        ('chunk-env-support-status.json', 'Linux 支持文件归档', 'received_bytes', 'average_bytes_per_second'),
        ('chunk-checkpoint-backup-status.json', '渲染器训练权重', 'received_bytes', 'average_bytes_per_second')]
    for filename, label, size_key, rate_key in sources:
        try:
            row = json.loads((deployment / filename).read_text())
        except (OSError, ValueError):
            row = {}
        if filename.startswith('chunk-env-'):
            environment_rows.append(row)
        reports.append({'name': label, 'status': row.get('status', 'unknown'), 'bytes': row.get(size_key),
            'expected_bytes': row.get('expected_bytes'), 'bytes_per_second': row.get(rate_key),
            'updated_at': row.get('updated_at'), 'files_verified': row.get('files_verified'),
            'files_total': row.get('files_total'), 'files_failed': row.get('files_failed'),
            'source_checksum_verified': row.get('source_checksum_verified'),
            'stale': bool(row) and row.get('status') == 'running' and time.time() - row.get('updated_at', 0) > 120})
    reports.append(archive_transfer(deployment / 'result-archive-backup-status.json',
        deployment / 'result-archive-transfers', 'data / runs 结果压缩包', 'result_archives'))
    reports.append(archive_transfer(workspace_deployment / 'unique-model-archive-backup-status.json',
        workspace_deployment / 'unique-model-archive-backup-transfers', '量化与工具模型', 'unique_model_archives'))
    try:
        proof = json.loads((deployment / 'isolated-restore-verification.json').read_text())
    except (OSError, ValueError):
        proof = {}
    results = proof.get('results', {})
    environments = [results.get(name, {}) for name in ['py312', 'tools']]
    payloads = [item.get('payload', {}) for item in environments]
    restored = proof.get('status') == 'verified' and proof.get('runtime_verified') is True
    imports_verified = restored and all(item.get('verified') is True and item.get('returncode') == 0
        and item.get('payload', {}).get('modules')
        and all(module.get('status') == 'imported' for module in item['payload']['modules'].values())
        for item in environments)
    return {'sampled_at': time.time(), 'reports': reports, 'restore': {
        'status': proof.get('status', 'unknown'),
        'archives_completed': all(row.get('status') == 'completed' and row.get('source_checksum_verified') is True
            for row in environment_rows),
        'imports_verified': imports_verified if proof else None,
        'cpu_verified': restored and all(item.get('cpu_tensor', {}).get('status') == 'passed'
            for item in payloads) if proof else None,
        'ocr_verified': restored and payloads[1].get('ocr', {}).get('status') == 'passed' if proof else None,
        'gpu_verified': proof.get('gpu_verified'),
        'host_system_libraries_reused': proof.get('host_kernel_and_system_libraries_reused')}}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            with LOCK:
                value = {**STATE, 'history': list(HISTORY), 'now': time.time()}
            value['stale'] = value['received_at'] is None or value['now'] - value['received_at'] > STALE_AFTER or value['error'] is not None
            value['transfers'] = local_transfers()
            value['stalled'] = value['progress_at'] is not None and value['now'] - value['progress_at'] >= 360
            payload = json.dumps(value, allow_nan=False).encode()
            content_type = 'application/json; charset=utf-8'
        elif self.path in ('/', '/index.html'):
            payload = (ROOT / 'index.html').read_bytes()
            content_type = 'text/html; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8877)
parser.add_argument('--interval', type=int, default=30)
parser.add_argument('--control-path', choices=list(CONTROL_HOSTS), default='/tmp/dag-pilot-rental-control-ssh')
parser.add_argument('--host', choices=list(CONTROL_HOSTS.values()))
arguments = parser.parse_args()
STALE_AFTER = max(20, arguments.interval + 12)
CONTROLS = [arguments.control_path, *[path for path in CONTROL_HOSTS if path != arguments.control_path]]
server = None
for port in range(arguments.port, arguments.port + 10):
    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        break
    except OSError:
        continue
if server is None:
    raise RuntimeError('No available loopback monitor port')
threading.Thread(target=collect, args=(max(5, arguments.interval),), daemon=True).start()
print(f'Local monitor: http://127.0.0.1:{server.server_port}', flush=True)
server.serve_forever()
