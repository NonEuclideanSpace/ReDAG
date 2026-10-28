import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
directory = root / 'deployment'
lock = (directory / 'final-sft-seed-transfer.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
volume = Path('/Volumes/TOSHIBA EXT')
if not volume.is_mount():
    raise RuntimeError('The designated external disk is not mounted')
destination = volume / 'dag-pilot-rental-backup-20261027T130933Z-9769b4/remote/priority-checkpoints'
plan = json.loads((directory / 'final-sft-seed-transfer-plan.json').read_text())
state = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'running', 'items': []}
status_path = directory / 'final-sft-seed-transfer-status.json'

def save():
    state['updated_at'] = time.time()
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(status_path)

save()
for item in plan['items']:
    if not volume.is_mount():
        raise RuntimeError('The external disk was disconnected')
    name = Path(item['source']).name
    target = destination / name
    report = directory / (name + '.transfer.json')
    record = {'source': item['source'], 'destination': str(target), 'report': str(report)}
    state['items'].append(record)
    if report.is_file():
        prior = json.loads(report.read_text())
        if (prior.get('status') == 'completed' and prior.get('source_checksum_verified') is True
                and prior.get('sha256') == item['sha256'] and target.is_file()
                and target.stat().st_size == item['bytes']):
            record['status'] = 'already_verified'
            save()
            continue
    command = [sys.executable, '-u', str(root / 'scripts/download_rental_chunks.py'),
        '--source', item['source'], '--destination', str(target), '--bytes', str(item['bytes']),
        '--mtime', str(item['mtime']), '--sha256', item['sha256'], '--workers', '1',
        '--status', str(report), '--control', '/tmp/dag-pilot-rental-direct1-ssh']
    with (directory / (name + '.transfer.log')).open('ab') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        record.update(status='transferring', pid=process.pid, started_at=time.time())
        save()
        result = process.wait()
    proof = json.loads(report.read_text()) if report.is_file() else {}
    record.update(returncode=result, finished_at=time.time(),
        status='completed' if result == 0 and proof.get('source_checksum_verified') is True else 'failed')
    save()
state['status'] = 'completed' if all(x['status'] in ['completed', 'already_verified'] for x in state['items']) else 'completed_with_failures'
state['finished_at'] = time.time()
save()
