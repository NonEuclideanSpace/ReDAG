import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
from itertools import cycle
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time


parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--destination', required=True)
parser.add_argument('--bytes', type=int, required=True)
parser.add_argument('--mtime', type=int, required=True)
parser.add_argument('--workers', type=int, default=6)
parser.add_argument('--chunk-bytes', type=int, default=4194304)
parser.add_argument('--control', action='append')
parser.add_argument('--status', default='pilot-small/deployment/chunk-checkpoint-backup-status.json')
parser.add_argument('--sha256')
args = parser.parse_args()
destination = Path(args.destination)
volume = Path('/Volumes/TOSHIBA EXT')
if not os.path.ismount(volume) or not destination.is_relative_to(volume):
    raise RuntimeError('The requested external disk is not available')
device = volume.stat().st_dev
parts = destination.parent / (destination.name + '.parts')
parts.mkdir(parents=True, exist_ok=True)
status_path = Path(args.status)
status_path.parent.mkdir(parents=True, exist_ok=True)
started = time.time()
controls = args.control or ['/tmp/dag-pilot-rental-backup-ssh']
stop = threading.Event()
worker_connection = threading.local()
control_offset = int.from_bytes(hashlib.sha256(args.source.encode()).digest()[:4], 'big') % len(controls)
initial_bytes = sum(p.stat().st_size for p in parts.glob('*.part') if p.is_file() and not p.name.startswith('._'))
state = {'status': 'running', 'started_at': started, 'source': args.source, 'destination': str(destination), 'expected_bytes': args.bytes, 'workers': args.workers, 'controls': controls, 'initial_bytes': initial_bytes, 'pid': os.getpid(), 'chunk_failures': [], 'completed_chunks': 0}
state.update(control_order=controls[control_offset:] + controls[:control_offset], existing_master_only=True)


def ssh_for(index, attempt=0):
    preferred = getattr(worker_connection, 'control', controls[(control_offset + index) % len(controls)])
    first = (controls.index(preferred) + attempt) % len(controls)
    failures = []
    for offset in range(len(controls)):
        control = controls[(first + offset) % len(controls)]
        if not Path(control).is_socket():
            failures.append(control + ': missing control socket')
            continue
        command = ['ssh', '-S', control, '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false', '-o', 'BatchMode=yes', '-p', '29410', 'root@117.50.181.105']
        try:
            check = subprocess.run(command[:-1] + ['-O', 'check', command[-1]], capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            failures.append(control + ': local master check timed out')
            continue
        if check.returncode == 0:
            return command
        failures.append(control + ': ' + check.stderr.strip()[-300:])
    raise RuntimeError('No existing transfer master is available; ' + '; '.join(failures))


def save():
    sizes = {}
    for path in parts.glob('[0-9]*.part*'):
        if path.suffix not in {'.part', '.tmp'}:
            continue
        try:
            index = int(path.name.split('.', 1)[0])
            size = path.stat().st_size
        except (OSError, ValueError):
            continue
        if 0 <= index < (args.bytes + args.chunk_bytes - 1) // args.chunk_bytes:
            sizes[index] = max(sizes.get(index, 0), size)
    state.update(updated_at=time.time(), received_bytes=sum(sizes.values()))
    state['average_bytes_per_second'] = (state['received_bytes'] - initial_bytes) / max(1, time.time() - started)
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(status_path)


def transfer(index):
    offset = index * args.chunk_bytes
    size = min(args.chunk_bytes, args.bytes - offset)
    path = parts / f'{index:06d}.part'
    if path.is_file() and path.stat().st_size == size:
        return index
    command = f'nice -n 15 ionice -c 3 dd if={shlex.quote(args.source)} iflag=skip_bytes,count_bytes skip={offset} count={size} bs=1048576 status=none'
    last_returncode = None
    for attempt in range(4):
        if stop.is_set():
            raise RuntimeError(f'Chunk {index} cancelled before another transfer attempt')
        temporary = path.with_suffix('.part.tmp')
        attempt_error = 'timeout'
        with temporary.open('wb') as output, (parts / f'{index:06d}.log').open('ab') as error:
            try:
                ssh = ssh_for(index, attempt)
                result = subprocess.run(ssh + [command], stdout=output, stderr=error, timeout=600)
            except subprocess.TimeoutExpired:
                result = None
                error.write(f'Attempt {attempt + 1}: transfer timed out\n'.encode())
            except RuntimeError as failure:
                result = None
                attempt_error = 'no_existing_master'
                error.write(f'Attempt {attempt + 1}: {failure}\n'.encode())
            else:
                if result.returncode != 0:
                    error.write(f'Attempt {attempt + 1}: control={ssh[2]} returncode={result.returncode}\n'.encode())
        last_returncode = result.returncode if result is not None else attempt_error
        if result is not None and result.returncode == 0 and temporary.stat().st_size == size:
            temporary.replace(path)
            return index
        if attempt < 3 and stop.wait(15):
            break
    raise RuntimeError(f'Chunk {index} did not transfer its exact {size} bytes after bounded retries; last_returncode={last_returncode}')


def download_chunks(count):
    assignments = cycle(controls[control_offset:] + controls[:control_offset])
    assignment_lock = threading.Lock()
    def initialize_worker():
        with assignment_lock:
            worker_connection.control = next(assignments)
    executor = ThreadPoolExecutor(max_workers=args.workers, initializer=initialize_worker)
    pending = {}
    try:
        pending = {executor.submit(transfer, index): index for index in range(count)}
        while pending:
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    future.result()
                    state['completed_chunks'] += 1
                except Exception as error:
                    state['chunk_failures'].append({'index': index, 'at': time.time(), 'error': f'{type(error).__name__}: {error}'})
                    state.update(status='running_with_chunk_failures', error=state['chunk_failures'][-1]['error'])
                    save()
            state['pending_chunks'] = len(pending)
            if not os.path.ismount(volume) or volume.stat().st_dev != device:
                raise RuntimeError('The external disk identity changed')
            save()
        if state['chunk_failures']:
            raise RuntimeError(f"{len(state['chunk_failures'])} chunks exhausted retries; completed chunks are retained for resume")
    except BaseException as error:
        stop.set()
        state.update(status='failed', error=f'{type(error).__name__}: {error}',
            cleanup_running_chunks=sum(future.running() for future in pending),
            cleanup='queued_chunks_cancelled_active_attempts_keep_their_existing_timeout')
        for future in pending:
            future.cancel()
        save()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


try:
    count = (args.bytes + args.chunk_bytes - 1) // args.chunk_bytes
    save()
    download_chunks(count)
    state.update(status='verifying_source', pending_chunks=0)
    save()
    probe = subprocess.run(ssh_for(0) + [f'stat -c "%s %Y" {shlex.quote(args.source)}'], capture_output=True, text=True, timeout=60, check=True)
    if list(map(int, probe.stdout.split())) != [args.bytes, args.mtime]:
        raise RuntimeError('The frozen source checkpoint changed during transfer')
    temporary = destination.with_suffix(destination.suffix + '.assembled.tmp')
    digest = hashlib.sha256()
    state.update(status='assembling', assembled_bytes=0)
    save()
    last_save = time.monotonic()
    with temporary.open('wb') as output:
        for index in range(count):
            with (parts / f'{index:06d}.part').open('rb') as source:
                while block := source.read(1048576):
                    output.write(block)
                    digest.update(block)
                    state['assembled_bytes'] += len(block)
                    if time.monotonic() - last_save >= 5:
                        save()
                        last_save = time.monotonic()
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size != args.bytes:
        raise RuntimeError('Assembled checkpoint size differs from its frozen source')
    if args.sha256 and digest.hexdigest() != args.sha256:
        raise RuntimeError('Assembled file differs from the source streaming checksum')
    os.utime(temporary, (args.mtime, args.mtime))
    temporary.replace(destination)
    state.update(status='completed', finished_at=time.time(), chunks=count, sha256=digest.hexdigest(), source_checksum_verified=bool(args.sha256), verification='SSH authenticated transfer, exact chunk lengths, stable source size/mtime, exact assembled length; checksum streamed while assembling')
    save()
except BaseException as error:
    state.update(status='failed', error=f'{type(error).__name__}: {error}')
    save()
    raise
