import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock, stopped = threading.Lock(), threading.Event()
    states = {row['path']: {'path': row['path'], 'bytes': 0, 'expected_bytes': row['size'], 'status': 'pending', 'last_byte_at': None} for row in manifest['files']}
    started = time.time()
    history = []

    def update(path, **fields):
        with lock:
            states[path].update(fields)

    def publish():
        now = time.time()
        with lock:
            files = [dict(value) for value in states.values()]
        current = sum(row['bytes'] for row in files)
        history.append((now, current))
        while len(history) > 2 and history[0][0] < now - 60:
            history.pop(0)
        speed = (current - history[0][1]) / max(now - history[0][0], 0.001)
        finished = sum(row['status'] == 'verified' for row in files)
        failed = sum(row['status'] == 'failed' for row in files)
        state = {'pid': os.getpid(), 'started_at': started, 'updated_at': now, 'status': 'completed' if finished == len(files) else 'failed' if stopped.is_set() and failed else 'running', 'destination': str(destination), 'files_verified': finished, 'files_total': len(files), 'files_failed': failed, 'downloaded_bytes': current, 'expected_bytes': sum(row['size'] for row in manifest['files']), 'recent_bytes_per_second': speed, 'proxy_mode': 'direct' if args.direct else 'system_proxy', 'canonical_revisions': manifest['canonical_revisions'], 'scope': 'public_original_model_snapshots_not_remote_nf4_or_trained_adapters', 'files': files}
        write(args.status, state)
        write(destination / 'status.json', state)

    def monitor():
        while not stopped.wait(5):
            publish()

    def download(row):
        relative = row['path']
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        proof_path = target.with_name(target.name + '.verified.json')
        if target.is_file() and proof_path.is_file():
            proof = json.loads(proof_path.read_text())
            if proof.get('native_hash') == row['native_hash'] and target.stat().st_size == row['size']:
                update(relative, status='verified', bytes=row['size'], verification='existing_download_proof')
                return
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.direct else urllib.request.build_opener()
        last_error = None
        for attempt in range(5):
            source = row['sources'][attempt % len(row['sources'])]
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset > row['size']:
                raise ValueError('Owned partial exceeds its frozen expected length')
            digest = hashlib.sha256() if row['hash_algorithm'] == 'sha256' else hashlib.sha1()
            if row['hash_algorithm'] == 'git_blob_sha1':
                digest.update(f"blob {row['size']}\0".encode())
            if offset:
                with partial.open('rb') as stream:
                    while block := stream.read(4 * 1024**2):
                        digest.update(block)
            update(relative, status='connecting', bytes=offset, source=source['name'], attempt=attempt + 1)
            try:
                if offset < row['size']:
                    headers = {'Range': f'bytes={offset}-'} if offset else {}
                    with opener.open(urllib.request.Request(source['url'], headers=headers), timeout=35) as response:
                        content_range = response.headers.get('Content-Range', '')
                        if offset and (response.status != 206 or not content_range.startswith(f'bytes {offset}-') or not content_range.endswith('/' + str(row['size']))):
                            raise RuntimeError('Server did not honor the exact resume range')
                        with partial.open('ab' if offset else 'wb') as stream:
                            while block := response.read(65536):
                                if offset + len(block) > row['size']:
                                    raise RuntimeError('Downloaded response exceeds its frozen expected size')
                                stream.write(block)
                                digest.update(block)
                                offset += len(block)
                                update(relative, status='downloading', bytes=offset, last_byte_at=time.time())
                if offset != row['size'] or digest.hexdigest() != row['native_hash']:
                    raise RuntimeError('Downloaded native LFS or Git blob identity differs')
                partial.replace(target)
                write(proof_path, {'canonical_repo': row['repo'], 'canonical_revision': row['revision'], 'file': row['filename'], 'size': offset, 'hash_algorithm': row['hash_algorithm'], 'native_hash': digest.hexdigest(), 'source': source['name'], 'verified_at': time.time()})
                update(relative, status='verified', bytes=offset, verification='native_hash_streamed_during_download')
                return
            except Exception as error:
                last_error = {'error_type': type(error).__name__, 'http_status': getattr(error, 'code', None)}
                update(relative, status='retrying', **last_error)
                time.sleep(min(2 ** attempt, 20))
        update(relative, status='failed', **last_error)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    publish()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download, row): row for row in sorted(manifest['files'], key=lambda row: row['size'])}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    update(futures[future]['path'], status='failed', error_type=type(error).__name__)
    finally:
        stopped.set()
        watcher.join()
        publish()


if __name__ == '__main__':
    main()
