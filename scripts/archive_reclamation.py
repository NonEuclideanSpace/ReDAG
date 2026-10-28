import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from uuid import uuid4


SCHEMA = 'rental_archive_reclamation_v1'
REQUEST_SCHEMA = 'rental_archive_reclamation_requests_v1'
EXTERNAL_ROOT = Path('/Volumes/TOSHIBA EXT')


def archive_name(value, gzip_only=True):
    suffix = r'\.tar\.gz' if gzip_only else r'\.tar(?:\.gz)?'
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*' + suffix, value):
        raise ValueError('Archive must be an explicit safe basename')
    return value


def manifest_name(name):
    name = archive_name(name, gzip_only=False)
    return (name[:-7] if name.endswith('.tar.gz') else name[:-4]) + '.manifest.json'


def receipt_name(name):
    return archive_name(name).removesuffix('.tar.gz') + '.reclaimed.json'


def regular_file(path):
    try:
        return stat.S_ISREG(Path(path).lstat().st_mode)
    except FileNotFoundError:
        return False


def read_regular_json(path):
    if not regular_file(path):
        raise ValueError('Expected a regular, non-symlink JSON file: ' + str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('JSON input changed to a nonregular file')
        return json.load(stream)


def manifest_digest(proof):
    payload = json.dumps(proof, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(proof, expected=None):
    archive_name(proof['archive'], gzip_only=False)
    if type(proof['archive_bytes']) is not int or proof['archive_bytes'] <= 0:
        raise ValueError('Archive byte count must be a positive integer')
    if not isinstance(proof['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', proof['sha256']):
        raise ValueError('Archive SHA256 must be exact lowercase hexadecimal')
    if expected is not None and any(proof[key] != expected[key] for key in ('archive', 'archive_bytes', 'sha256')):
        raise ValueError('Archive name, bytes or SHA256 differ from the manifest')


def validate_receipt(value, proof, delivery, states=None):
    validate_manifest(value)
    validate_manifest(proof, value)
    name = archive_name(value['archive'])
    if value['source'] != str(Path(delivery) / name):
        raise ValueError('Receipt source differs from the explicit delivery directory')
    destination = Path(value['destination'])
    if (not destination.is_absolute() or '..' in destination.parts
            or not destination.is_relative_to(EXTERNAL_ROOT) or destination.name != name):
        raise ValueError('Receipt destination must be the exact archive on the designated external disk')
    if value['manifest_sha256'] != manifest_digest(proof):
        raise ValueError('Receipt belongs to a different retained manifest')
    verified = value['external_verification']
    if verified['status'] != 'completed' or verified['method'] != 'stream_sha256':
        raise ValueError('External copy lacks completed streaming SHA256 verification')
    if type(verified.get('bytes')) is not int:
        raise ValueError('External verification bytes must be an exact integer')
    for key, expected in {'bytes': proof['archive_bytes'], 'sha256': proof['sha256'],
                          'source': value['source'], 'destination': value['destination']}.items():
        if verified.get(key) != expected:
            raise ValueError('External verification differs: ' + key)
    timestamp = verified.get('verified_at')
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError('External verification needs its actual timestamp')
    if states is not None and (value.get('schema') != SCHEMA or value.get('state') not in states):
        raise ValueError('No completed remote reclamation receipt exists')


def require_reclaimed_receipt(delivery, proof):
    path = Path(delivery) / receipt_name(proof['archive'])
    value = read_regular_json(path)
    validate_receipt(value, proof, delivery, {'remote_copy_reclaimed'})
    return value


def validate_delivery(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path.parent.name != 'delivery':
        raise ValueError('Supply one explicit absolute directory immediately under delivery')
    for item in reversed([path, *path.parents]):
        if item.is_symlink():
            raise ValueError('Symlinks are forbidden in the delivery path')
    if not path.is_dir():
        raise ValueError('The observed delivery directory must already exist')
    return path


def atomic_receipt(directory, name, value):
    temporary = '.' + name + '.' + uuid4().hex + '.tmp'
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def reclaim(delivery, request):
    delivery = validate_delivery(delivery)
    if request.get('schema') != REQUEST_SCHEMA or request.get('delivery_directory') != str(delivery):
        raise ValueError('Explicit receipt list is bound to a different delivery directory')
    items = request.get('receipts')
    if not isinstance(items, list) or not items:
        raise ValueError('A nonempty explicit receipt list is required')
    names = [archive_name(item['archive']) for item in items]
    if len(names) != len(set(names)):
        raise ValueError('The explicit receipt list contains duplicate archives')
    descriptor = os.open(delivery / '.pack.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'r+') as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError('The pack lock is not a regular file')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'lock_busy', 'deleted': [], 'deleted_bytes': 0}
        prepared = []
        for item in items:
            name = item['archive']
            manifest = delivery / manifest_name(name)
            proof = read_regular_json(manifest)
            validate_receipt(item, proof, delivery)
            archive = delivery / name
            saved_path = delivery / receipt_name(name)
            saved = None
            if saved_path.exists() or saved_path.is_symlink():
                saved = read_regular_json(saved_path)
                validate_receipt(saved, proof, delivery, {'delete_pending', 'remote_copy_reclaimed'})
                if any(saved[key] != item[key] for key in ('destination', 'external_verification', 'manifest_sha256')):
                    raise ValueError('Existing receipt differs from this explicit authorization')
            info = archive.lstat() if archive.exists() or archive.is_symlink() else None
            if info is not None:
                if not stat.S_ISREG(info.st_mode) or info.st_size != proof['archive_bytes']:
                    raise ValueError('Remote archive is not the matching regular file')
            elif saved is None:
                raise ValueError('Missing archive has no previously persisted reclamation receipt')
            prepared.append((item, proof, info, saved))
        deleted, finalized = [], []
        for item, proof, info, saved in prepared:
            name = item['archive']
            archive = delivery / name
            value = dict(item, schema=SCHEMA, state='delete_pending',
                         requested_at=saved.get('requested_at', time.time()) if saved else time.time(),
                         original_sources_deleted=False, manifest_retained=True)
            if info is not None:
                current = archive.lstat()
                if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
                    raise ValueError('Archive changed after validation; nothing removed for this item')
                atomic_receipt(delivery, receipt_name(name), value)
                current = archive.lstat()
                if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
                    raise ValueError('Archive changed before removal')
                archive.unlink()
                deleted.append(name)
            else:
                finalized.append(name)
            value.update(state='remote_copy_reclaimed', reclaimed_at=time.time())
            atomic_receipt(delivery, receipt_name(name), value)
        return {'status': 'reclaimed', 'deleted': deleted, 'already_absent_finalized': finalized,
                'deleted_bytes': sum(item['archive_bytes'] for item in items if item['archive'] in deleted),
                'original_sources_deleted': False, 'manifests_retained': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    args = parser.parse_args()
    delivery = validate_delivery(args.delivery)
    if not args.receipts.is_absolute() or args.receipts.parent != delivery or '..' in args.receipts.parts:
        raise ValueError('The explicit request file must be directly inside the observed delivery directory')
    result = reclaim(delivery, read_regular_json(args.receipts))
    print(json.dumps(result, indent=2))
    if result['status'] == 'lock_busy':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
