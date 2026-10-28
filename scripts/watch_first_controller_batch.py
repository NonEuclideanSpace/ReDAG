import argparse
import fcntl
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from watch_rl_recovery_exports import CONTROL, EXTERNAL, REMOTE_ROOT, active_direct_downloads, read_remote, save, verified_local


VOLUME = Path('/Volumes/TOSHIBA EXT')
ITEM = {'manifest': REMOTE_ROOT + '/deployment/first-controller-batch-8x4.json',
    'source': REMOTE_ROOT + '/delivery/first-controller-batch-8x4.tar.gz'}
MAXIMUM_BYTES = 2 * 2**30 + 32 * 2**20


def validate_manifest(value):
    if (value.get('schema') != 'first_batch_archive_receipt_v1' or value.get('status') != 'completed'
            or value.get('source') != ITEM['source'] or value.get('archive') != ITEM['source']
            or value.get('source_files_unchanged') is not True
            or type(value.get('bytes')) is not int or not 0 < value['bytes'] <= MAXIMUM_BYTES
            or type(value.get('mtime')) is not int or value['mtime'] <= 0
            or not isinstance(value.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', value['sha256'])
            or value.get('full1280_complete') is not False or value.get('gpu_work_started') is not False):
        raise ValueError('First-batch archive identity, size, timestamp or completion evidence differs')


def transfer_capacity_available(active):
    if any(type(row.get('workers')) is not int or row['workers'] < 1 for row in active):
        raise ValueError('Existing direct transfer worker counts are invalid')
    return sum(row['workers'] for row in active) + 1 <= 7 and all(row['source'] != ITEM['source'] for row in active)


def check_disk(expected=None):
    if not os.path.ismount(VOLUME) or not EXTERNAL.is_dir():
        raise RuntimeError('The designated external backup is not mounted')
    device = VOLUME.stat().st_dev
    if EXTERNAL.is_symlink() or EXTERNAL.stat().st_dev != device or (expected is not None and device != expected):
        raise RuntimeError('The designated external disk identity changed')
    for relative in ('remote', 'remote/first-controller-batch', 'reports'):
        path = EXTERNAL / relative
        if path.is_symlink() or (path.exists() and (not path.is_dir() or path.stat().st_dev != device)):
            raise RuntimeError('A delivery directory is not on the designated external disk')
    partials = EXTERNAL / 'remote/first-controller-batch/first-controller-batch-8x4.tar.gz.parts'
    if partials.is_symlink() or (partials.exists() and (not partials.is_dir() or partials.stat().st_dev != device)):
        raise RuntimeError('Download partials are not on the designated external disk')
    return device


def command(root, manifest, target, report):
    return [sys.executable, '-u', str(root / 'scripts/download_rental_chunks.py'),
        '--source', ITEM['source'], '--destination', str(target), '--bytes', str(manifest['bytes']),
        '--mtime', str(manifest['mtime']), '--sha256', manifest['sha256'], '--workers', '1',
        '--status', str(report), '--control', CONTROL]


def verified(manifest, target, report):
    return verified_local(manifest, target, report) and int(target.stat().st_mtime) == manifest['mtime']


def stop_child(child, deadline):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=max(0, min(3, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)


def watch(root, hours):
    if not math.isfinite(hours) or not 0 < hours <= 3:
        raise ValueError('First-batch watching is bounded to at most3 hours')
    directory = root / 'deployment'
    status_path = directory / 'first-controller-batch-transfer-status.json'
    with (directory / 'first-controller-batch-transfer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + hours * 3600
        device = check_disk()
        target = EXTERNAL / 'remote/first-controller-batch/first-controller-batch-8x4.tar.gz'
        report = directory / 'first-controller-batch-8x4.transfer.json'
        manifest_copy = directory / 'first-controller-batch-8x4.received.json'
        state = {'status': 'waiting_for_completed_export', 'pid': os.getpid(), 'started_at': time.time(),
            'hours': hours, 'workers': 1, 'maximum_combined_transfer_workers': 7, 'source': ITEM['source'],
            'remote_manifest': ITEM['manifest'], 'destination': str(target), 'external_device': device}
        save(status_path, state)
        child = None
        try:
            while time.monotonic() < deadline - 5:
                check_disk(device)
                if not Path(CONTROL).is_socket():
                    state['status'] = 'waiting_for_existing_control_connection'
                else:
                    try:
                        manifest = read_remote(ITEM, min(30, deadline - time.monotonic() - 5))
                    except (OSError, subprocess.SubprocessError) as error:
                        state.update(status='waiting_after_connection_error', last_error=str(error), last_error_at=time.time())
                    else:
                        if manifest is None or manifest.get('status') != 'completed':
                            state['status'] = 'waiting_for_completed_export'
                        else:
                            validate_manifest(manifest)
                            check_disk(device)
                            save(manifest_copy, manifest)
                            if verified(manifest, target, report):
                                state['status'] = 'already_verified'
                                break
                            active = active_direct_downloads()
                            if not transfer_capacity_available(active):
                                state.update(status='waiting_for_transfer_capacity', active_direct_downloads=active)
                            else:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                check_disk(device)
                                arguments = command(root, manifest, target, report)
                                with (directory / 'first-controller-batch-8x4.transfer.log').open('ab') as output:
                                    child = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=output,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                                state.update(status='transferring', transfer_pid=child.pid, report=str(report))
                                save(status_path, state)
                                while child.poll() is None and time.monotonic() < deadline - 5:
                                    check_disk(device)
                                    active = active_direct_downloads()
                                    if sum(row['workers'] for row in active) > 7:
                                        raise RuntimeError('The combined direct transfer worker limit was exceeded')
                                    time.sleep(min(10, max(0, deadline - time.monotonic() - 5)))
                                if child.poll() is None:
                                    state['status'] = 'bounded_watch_ended'
                                    break
                                if child.returncode != 0 or not verified(manifest, target, report):
                                    raise RuntimeError('First-batch download did not finish checksum verified')
                                state['status'] = 'completed'
                                break
                save(status_path, state)
                time.sleep(min(30, max(0, deadline - time.monotonic() - 5)))
            else:
                state['status'] = 'bounded_watch_ended'
            if state['status'] in ('completed', 'already_verified'):
                check_disk(device)
                (EXTERNAL / 'reports').mkdir(exist_ok=True)
                for source in (manifest_copy, report):
                    shutil.copy2(source, EXTERNAL / 'reports' / source.name)
                state['finished_at'] = time.time()
                save(status_path, state)
                shutil.copy2(status_path, EXTERNAL / 'reports' / status_path.name)
            return state
        except BaseException as error:
            state.update(status='failed', last_error=f'{type(error).__name__}: {error}', finished_at=time.time())
            raise
        finally:
            stop_child(child, deadline)
            save(status_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=3)
    args = parser.parse_args()
    watch(Path(__file__).resolve().parents[1], args.hours)


if __name__ == '__main__':
    main()
