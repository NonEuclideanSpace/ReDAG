"""Bounded SSD prefetch while the external disk drains; no content hashing."""
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'deployment/archive-staging'
STATUS = ROOT / 'deployment/archive-prefetch-status.json'
BASE = ROOT / 'deployment/result-archive-backup-status.json'
REMOTE = '/workspace/agentic-image-pilot-small/delivery/project-results'
SSH = ['ssh', '-S', '/tmp/dag-pilot-rental-background-ssh', '-o', 'ProxyCommand=false',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p', '29410', 'root@117.50.181.105']

def remote(code):
    r = subprocess.run(SSH + ['/usr/local/miniconda3/envs/py312/bin/python -c ' + shlex.quote(code)],
                       capture_output=True, text=True, check=True, timeout=45)
    return json.loads(r.stdout)

def fetch(item):
    name = item['archive']
    receipt = CACHE / (name + '.json')
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('status') == 'completed' and previous.get('received_bytes') == item['archive_bytes']:
            return item['archive_bytes']
    proof = remote('import pathlib,json;p=pathlib.Path(' + repr(REMOTE) + ');'
                   'print(json.dumps({"mtime":int((p/' + repr(name) + ').stat().st_mtime)}))')
    (CACHE / (name + '.manifest.json')).write_text(json.dumps(item))
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/direct_rental_archive.py'), '--staging',
           '--source', REMOTE + '/' + name, '--destination', str(CACHE / name),
           '--bytes', str(item['archive_bytes']), '--mtime', str(proof['mtime']), '--sha256', item['sha256'],
           '--control', '/tmp/dag-pilot-rental-background-ssh', '--status', str(receipt)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return item['archive_bytes']

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASE.read_text())
    excluded = set(baseline['completed']) | set(baseline.get('active', []))
    state = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'running',
             'completed': {}, 'active': [], 'failures': [], 'excluded_existing': sorted(excluded),
             'destination': str(CACHE), 'content_hashing': False}
    deadline = time.monotonic() + 4 * 3600
    futures, known, attempts = {}, {}, {}
    next_read = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while time.monotonic() < deadline:
            if time.monotonic() >= next_read:
                try:
                    snapshot = remote('import json,pathlib;s=json.loads(pathlib.Path(' + repr(REMOTE + '/status.json') + ').read_text());print(json.dumps({"status":s["status"],"archives":s["archives"]}))')
                    known = {x['archive']: x for x in snapshot['archives'] if x['archive'] not in excluded}
                    state['catalog_status'] = snapshot['status']
                    state['target_bytes'] = sum(x['archive_bytes'] for x in known.values())
                    state['known_archive_bytes'] = sum(x['archive_bytes'] for x in snapshot['archives'])
                    state.pop('observation_error', None)
                except Exception as error:
                    state['observation_error'] = str(error)
                next_read = time.monotonic() + 45
            for future, name in list(futures.items()):
                if future.done():
                    del futures[future]
                    try:
                        state['completed'][name] = future.result()
                    except Exception as error:
                        state['failures'].append({'name': name, 'error': str(error), 'at': time.time()})
            for name, item in known.items():
                if len(futures) == 3:
                    break
                if name in state['completed'] or name in futures.values() or attempts.get(name, 0) >= 3:
                    continue
                if shutil.disk_usage(CACHE).free < item['archive_bytes'] + 15 * 1024**3:
                    state['status'] = 'waiting_for_local_space'
                    break
                attempts[name] = attempts.get(name, 0) + 1
                futures[pool.submit(fetch, item)] = name
            state['active'] = list(futures.values())
            state['received_bytes'] = sum(state['completed'].values())
            state['updated_at'] = time.time()
            state['catalog_caught_up'] = bool(known) and set(known) <= set(state['completed'])
            temp = STATUS.with_suffix('.tmp')
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(STATUS)
            time.sleep(3)

if __name__ == '__main__':
    main()
