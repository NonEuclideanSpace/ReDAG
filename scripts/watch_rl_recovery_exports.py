import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time


CONTROL = '/tmp/dag-pilot-rental-direct1-ssh'
REMOTE_ROOT = '/workspace/agentic-image-pilot-small'
EXTERNAL = Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_manifest(value, item):
    if (value.get('schema') != 'rl_recovery_export_v1' or value.get('status') != 'completed'
            or value.get('stage') != item['mode'] or value.get('source') != item['source']
            or value.get('required_effective_updates') != 32 or value.get('effective_target_met') is not True
            or value.get('source_files_unchanged') is not True or value.get('shared_renders_included') is not False
            or type(value.get('bytes')) is not int or value['bytes'] <= 0
            or type(value.get('mtime')) is not int or not isinstance(value.get('sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', value['sha256'])
            or value.get('weight_validation', {}).get('actual_checkpoint_tensors_checked') is not True):
        raise ValueError('Completed RL recovery manifest identity or evidence is invalid')


def verified_local(manifest, target, report):
    if target.is_symlink() or report.is_symlink() or not target.is_file() or not report.is_file():
        return False
    previous = json.loads(report.read_text())
    info = target.stat()
    return (previous.get('status') == 'completed' and previous.get('source_checksum_verified') is True
        and previous.get('source') == manifest['source'] and previous.get('destination') == str(target)
        and previous.get('expected_bytes') == manifest['bytes'] and previous.get('sha256') == manifest['sha256']
        and info.st_size == manifest['bytes'] and info.st_mtime <= previous.get('finished_at', 0))


def active_direct_downloads():
    rows = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,stat=,command='], text=True).splitlines():
        fields = line.strip().split(None, 2)
        if (len(fields) != 3 or 'Z' in fields[1] or 'download_rental_chunks.py --source ' not in fields[2]
                or '--control ' + CONTROL not in fields[2]):
            continue
        count = re.search(r'--workers (\d+)(?: |$)', fields[2])
        source = re.search(r'--source (\S+)', fields[2])
        if count is None or source is None:
            raise ValueError('Cannot safely count an existing direct download')
        rows.append({'pid': int(fields[0]), 'workers': int(count[1]), 'source': source[1]})
    return rows


def transfer_capacity_available(item, active):
    return sum(x['workers'] for x in active) <= 6 and all(x['source'] != item['source'] for x in active)


def read_remote(item, timeout):
    code = "import json,pathlib; p=pathlib.Path(" + repr(item['manifest']) + "); print(json.dumps(None if not p.is_file() else json.loads(p.read_text())))"
    command = ['ssh', '-S', CONTROL, '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false',
        '-o', 'BatchMode=yes', '-p', '29410', 'root@117.50.181.105',
        '/usr/local/miniconda3/envs/py312/bin/python -c ' + shlex.quote(code)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=16)
    args = parser.parse_args()
    if not 0 < args.hours <= 16:
        raise ValueError('Recovery export watching is bounded to at most16 hours')
    root = Path(__file__).resolve().parents[1]
    directory = root / 'deployment'
    status_path = directory / 'rl-recovery-export-transfer-status.json'
    with (directory / 'rl-recovery-export-transfer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads(args.plan.read_text())
        if [x['mode'] for x in plan['items']] != ['agent', 'joint']:
            raise ValueError('The explicit export order must be agent then joint')
        deadline = time.monotonic() + args.hours * 3600
        state = {'pid': os.getpid(), 'started_at': time.time(), 'hours': args.hours, 'status': 'waiting',
            'control': CONTROL, 'workers': 1, 'maximum_combined_transfer_workers': 7, 'items': []}
        save(status_path, state)
        for item in plan['items']:
            if (item['source'] != REMOTE_ROOT + '/delivery/priority-checkpoints/full-' + item['mode'] + '-rl32-complete.tar.gz'
                    or item['manifest'] != REMOTE_ROOT + '/deployment/full-' + item['mode'] + '-rl32-export.json'):
                raise ValueError('Export paths differ from the explicit stage delivery contract')
            record = {'mode': item['mode'], 'source': item['source'], 'status': 'waiting_for_completed_export'}
            state['items'].append(record)
            while time.monotonic() < deadline:
                try:
                    if not Path(CONTROL).is_socket():
                        raise RuntimeError('The existing direct master socket is unavailable')
                    active = active_direct_downloads()
                    if not transfer_capacity_available(item, active):
                        record.update(status='waiting_for_transfer_capacity', active_direct_downloads=active)
                        save(status_path, state)
                        time.sleep(min(30, max(0, deadline - time.monotonic())))
                        continue
                    manifest = read_remote(item, min(30, max(1, deadline - time.monotonic())))
                    if manifest is None or manifest.get('status') != 'completed':
                        record['status'] = 'waiting_for_completed_export'
                        save(status_path, state)
                        time.sleep(min(30, max(0, deadline - time.monotonic())))
                        continue
                    validate_manifest(manifest, item)
                    if not os.path.ismount('/Volumes/TOSHIBA EXT') or not EXTERNAL.is_dir():
                        raise RuntimeError('The designated external backup is not mounted')
                    name = Path(item['source']).name
                    target = EXTERNAL / 'remote/priority-checkpoints' / name
                    report = directory / (name + '.transfer.json')
                    manifest_copy = directory / ('full-' + item['mode'] + '-rl32-export.received.json')
                    save(manifest_copy, manifest)
                    record.update(export_manifest=manifest, report=str(report), destination=str(target))
                    if verified_local(manifest, target, report):
                        record['status'] = 'already_verified'
                    else:
                        command = [sys.executable, '-u', str(root / 'scripts/download_rental_chunks.py'),
                            '--source', item['source'], '--destination', str(target), '--bytes', str(manifest['bytes']),
                            '--mtime', str(manifest['mtime']), '--sha256', manifest['sha256'], '--workers', '1',
                            '--status', str(report), '--control', CONTROL]
                        with (directory / (name + '.transfer.log')).open('ab') as output:
                            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
                        record.update(status='transferring', pid=child.pid, command=command, started_at=time.time())
                        state['status'] = 'transferring'
                        save(status_path, state)
                        while child.poll() is None and time.monotonic() < deadline:
                            if report.is_file():
                                record['transfer_progress'] = json.loads(report.read_text())
                                save(status_path, state)
                            time.sleep(min(15, max(0, deadline - time.monotonic())))
                        if child.poll() is None:
                            child.send_signal(signal.SIGINT)
                            record['status'] = 'bounded_watch_ended_download_draining'
                            state['status'] = 'bounded_watch_ended'
                            save(status_path, state)
                            return
                        if child.returncode != 0 or not verified_local(manifest, target, report):
                            raise RuntimeError('Recovery archive download did not finish checksum verified')
                        record.update(status='completed', finished_at=time.time())
                    shutil.copy2(manifest_copy, EXTERNAL / 'reports' / manifest_copy.name)
                    shutil.copy2(report, EXTERNAL / 'reports' / report.name)
                    save(status_path, state)
                    shutil.copy2(status_path, EXTERNAL / 'reports' / status_path.name)
                    state['status'] = 'waiting'
                    break
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    record.update(last_error=f'{type(error).__name__}: {error}', last_error_at=time.time())
                    save(status_path, state)
                    time.sleep(min(60, max(0, deadline - time.monotonic())))
            else:
                state['status'] = 'bounded_watch_ended'
                save(status_path, state)
                return
        state.update(status='completed', finished_at=time.time())
        save(status_path, state)
        shutil.copy2(status_path, EXTERNAL / 'reports' / status_path.name)


if __name__ == '__main__':
    main()
