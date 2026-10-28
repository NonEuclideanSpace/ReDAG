import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from archive_reclamation import archive_name, manifest_name, read_regular_json, regular_file, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/workspace/agentic-image-pilot-small/delivery/project-results'
DESTINATION = Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4/remote/project-results-archives')
CONTROLS = ['/tmp/dag-pilot-rental-direct1-ssh']
STATUS = ROOT / 'deployment/result-archive-backup-status.json'
TRANSFER_REPORTS = ROOT / 'deployment/result-archive-transfers'


def remote(code):
    command = '/usr/local/miniconda3/envs/py312/bin/python -c ' + shlex.quote(code)
    result = subprocess.run(['ssh', '-S', '/tmp/dag-pilot-rental-direct1-ssh', '-o',
        'ProxyCommand=false', '-o', 'BatchMode=yes', '-p', '29410', 'root@117.50.181.105', command],
        capture_output=True, text=True, timeout=45, check=True)
    return json.loads(result.stdout)


def save(state):
    state['updated_at'] = time.time()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(STATUS)


def verified_local_archive(item):
    name = archive_name(item['archive'])
    destination = DESTINATION / name
    manifest = DESTINATION / manifest_name(name)
    report = TRANSFER_REPORTS / (name + '.json')
    if not all(regular_file(path) for path in (destination, manifest, report)):
        return False
    try:
        proof, old = read_regular_json(manifest), read_regular_json(report)
        validate_manifest(proof, item)
        info = destination.stat()
        content_checked = old.get('source_checksum_verified') is True and old.get('sha256') == item['sha256']
        direct_received = (old.get('transport_mode') == 'direct_no_content_hash'
            and old.get('verification') == 'authenticated_ssh_and_exact_length_only'
            and old.get('transfer_returncode') == 0 and old.get('received_bytes') == item['archive_bytes']
            and old.get('expected_sha256_from_manifest') == item['sha256'])
        return (old.get('status') == 'completed' and (content_checked or direct_received)
            and old.get('expected_bytes') == item['archive_bytes']
            and old.get('destination') == str(destination) and old.get('source') == REMOTE + '/' + name
            and info.st_size == item['archive_bytes']
            and isinstance(old.get('finished_at'), (int, float)) and info.st_mtime <= old['finished_at'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def reuse_verified_archives(known, completed, active):
    for name, item in known.items():
        if name not in completed and name not in active and verified_local_archive(item):
            receipt = read_regular_json(TRANSFER_REPORTS / (name + '.json'))
            completed[name] = {'archive': name, 'bytes': item['archive_bytes'],
                'sha256': item['sha256'], 'reused_existing_download': True,
                'reused_verified_download': receipt.get('source_checksum_verified') is True,
                'content_checksum_verified': receipt.get('source_checksum_verified') is True}


def download(item, workers_per_archive):
    name = archive_name(item['archive'])
    validate_manifest(item)
    proof_name = manifest_name(name)
    if verified_local_archive(item):
        receipt = read_regular_json(TRANSFER_REPORTS / (name + '.json'))
        return {'archive': name, 'bytes': item['archive_bytes'], 'sha256': item['sha256'],
            'reused_existing_download': True,
            'reused_verified_download': receipt.get('source_checksum_verified') is True,
            'content_checksum_verified':receipt.get('source_checksum_verified') is True}
    proof = remote('import pathlib,json; p=pathlib.Path(' + repr(REMOTE) + '); '
        'r=json.loads((p/' + repr(proof_name) + ').read_text()); '
        'print(json.dumps({"proof":r,"mtime":int((p/' + repr(name) + ').stat().st_mtime)}))')
    validate_manifest(proof['proof'], item)
    (DESTINATION / proof_name).write_text(json.dumps(proof['proof'], indent=2))
    report = TRANSFER_REPORTS / (name + '.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'scripts/direct_rental_archive.py'),
        '--source', REMOTE + '/' + name, '--destination', str(DESTINATION / name),
        '--bytes', str(item['archive_bytes']), '--mtime', str(proof['mtime']),
        '--sha256', item['sha256'], '--workers', str(workers_per_archive), '--status', str(report)]
    for control in CONTROLS:
        command += ['--control', control]
    with (DESTINATION / (name + '.transfer.log')).open('ab') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not verified_local_archive(item):
        raise ValueError('Archive transfer did not finish verified')
    return {'archive': name, 'bytes': item['archive_bytes'], 'sha256': item['sha256'],
        'received_at': time.time(), 'content_checksum_verified': False, 'verification':'authenticated_ssh_and_exact_length_only'}


def main():
    global REMOTE, DESTINATION, STATUS, TRANSFER_REPORTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=6)
    parser.add_argument('--parallel-archives', type=int, default=3)
    parser.add_argument('--workers-per-archive', type=int, choices=range(1, 5), default=2,
        help='Chunk download workers per archive (default: 2; maximum total: 24)')
    parser.add_argument('--follow-updates', action='store_true',
        help='Wait for later catalog archives until the bounded duration expires')
    parser.add_argument('--remote', default=REMOTE)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--status', type=Path, default=STATUS)
    args = parser.parse_args()
    if not 1 <= args.parallel_archives <= 6:
        raise ValueError('Archive concurrency must be between one and six')
    source = Path(args.remote)
    if not source.is_absolute() or '..' in source.parts:
        raise ValueError('Remote archive directory must be an absolute path without parent traversal')
    REMOTE = source.as_posix()
    DESTINATION = args.destination.resolve()
    STATUS = args.status.resolve()
    if STATUS != ROOT / 'deployment/result-archive-backup-status.json':
        TRANSFER_REPORTS = STATUS.parent / (STATUS.stem.removesuffix('-status') + '-transfers')
    if not DESTINATION.is_relative_to(Path('/Volumes/TOSHIBA EXT')):
        raise ValueError('Archive destination must be on the designated external disk')
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.ismount('/Volumes/TOSHIBA EXT') or not 0 < args.hours <= 16:
        raise ValueError('A mounted destination and bounded duration are required')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(),
        'destination': str(DESTINATION), 'remote': REMOTE, 'transfer_reports': str(TRANSFER_REPORTS), 'completed': {}, 'failures': [], 'active': [],
        'parallel_archives': args.parallel_archives, 'workers_per_archive': args.workers_per_archive,
        'max_chunk_workers': args.parallel_archives * args.workers_per_archive, 'full_delivery_complete': False}
    state.update(hours=args.hours, follow_updates=args.follow_updates,
        accept_new_archives_until=state['started_at'] + args.hours * 3600)
    futures, attempted, known = {}, {}, {}
    deadline = time.monotonic() + args.hours * 3600
    next_read = 0
    with ThreadPoolExecutor(max_workers=args.parallel_archives) as pool:
        while time.monotonic() < deadline:
            if time.monotonic() >= next_read:
                try:
                    snapshot = remote('import json,pathlib; r=json.loads(pathlib.Path(' + repr(REMOTE + '/status.json') + ').read_text()); print(json.dumps({k:r.get(k) for k in ["status","archives","errors","pending_changed_or_new_paths","updated_at"]}))')
                    known = {r['archive']: r for r in snapshot['archives']}
                    state['remote_snapshot'] = snapshot
                    state['known_archive_bytes'] = sum(r['archive_bytes'] for r in known.values())
                    reuse_verified_archives(known, state['completed'], set(futures.values()))
                    state.pop('observation_error', None)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    state['observation_error'] = str(error)
                next_read = time.monotonic() + 45
            for future, name in list(futures.items()):
                if not future.done():
                    continue
                del futures[future]
                try:
                    state['completed'][name] = future.result()
                except Exception as error:
                    state['failures'].append({'archive': name, 'at': time.time(), 'error': str(error)})
            active = set(futures.values())
            for name, item in known.items():
                if len(futures) >= args.parallel_archives:
                    break
                if name in active or name in state['completed'] or attempted.get(name, 0) >= 3:
                    continue
                attempted[name] = attempted.get(name, 0) + 1
                futures[pool.submit(download, item, args.workers_per_archive)] = name
            state['active'] = list(futures.values())
            state['received_archive_bytes'] = sum(row['bytes'] for row in state['completed'].values())
            state['received_archives'] = len(state['completed'])
            state['verified_bytes'] = sum(row['bytes'] for row in state['completed'].values() if row.get('content_checksum_verified'))
            state['verified_archives'] = sum(bool(row.get('content_checksum_verified')) for row in state['completed'].values())
            state['verification_policy'] = 'User requested direct transfer without content hashing; verified counts include historical checks only'
            finished = state.get('remote_snapshot', {}).get('status') in {'snapshot_catalog_caught_up', 'delta_required'}
            caught_up = bool(finished and known and not futures and set(known) <= set(state['completed']))
            state['catalog_caught_up'] = caught_up
            state['status'] = 'waiting_for_new_archives' if caught_up and args.follow_updates else 'running'
            if caught_up and not args.follow_updates:
                state['status'] = 'snapshot_archives_downloaded'
            save(state)
            if caught_up and not args.follow_updates:
                break
            time.sleep(10)
    if state['status'] in {'running', 'waiting_for_new_archives'}:
        state['status'] = 'bounded_pass_ended'
    state['finished_at'] = time.time()
    save(state)


if __name__ == '__main__':
    main()
