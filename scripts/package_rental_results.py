import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import time

from sample_delivery_compression import DigestSink, low_priority, write_json
from archive_reclamation import archive_name, manifest_name, read_regular_json, regular_file, require_reclaimed_receipt, validate_manifest


UNIQUE_MODELS = ('actor-4b-nf4', 'judge-8b-nf4', 'tools')


def acquire_lock(lock, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError as error:
            if wait_seconds == 0:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'Timed out after {wait_seconds:g} seconds waiting for archive lock: {lock.name}') from error
            time.sleep(min(0.2, remaining))


def metadata(path):
    value = path.lstat()
    kind = 'file' if stat.S_ISREG(value.st_mode) else 'directory' if stat.S_ISDIR(value.st_mode) else 'symlink' if stat.S_ISLNK(value.st_mode) else 'unsupported'
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns,
        'mode': value.st_mode, 'inode': value.st_ino, 'device': value.st_dev, 'type': kind,
        'link_target': os.readlink(path) if kind == 'symlink' else None}


def inventory(project, excluded, include_unique_models=False, only_groups=()):
    rows = {}
    errors = []
    for group in only_groups:
        if not isinstance(group, str) or not group or any(part in {'', '.', '..'} for part in group.split('/')):
            raise ValueError('Only-group must be a normalized project-relative group')
    selected = lambda group: not only_groups or group in only_groups
    unique_roots = [project / 'models' / name for name in UNIQUE_MODELS if selected('models/' + name)]
    if include_unique_models and not (project / 'models').exists():
        errors.append({'path': 'models', 'error': 'requested_unique_model_parent_missing'})
    def walk(path, group):
        if not selected(group):
            return
        if any(path == root or root in path.parents for root in excluded):
            return
        relative = path.relative_to(project).as_posix()
        try:
            info = metadata(path)
            if info['type'] == 'unsupported':
                errors.append({'path': relative, 'error': 'unsupported_filesystem_type'})
                return
            rows[relative] = {'path': relative, 'group': group, 'stat': info}
            if group.startswith('models/') and info['type'] == 'symlink':
                target = path.resolve(strict=False)
                if not target.exists() or not any(target == root or root in target.parents for root in unique_roots):
                    errors.append({'path': relative, 'error': 'unique_model_link_payload_not_archived', 'target': str(target)})
            if info['type'] == 'directory':
                for child in sorted(path.iterdir()):
                    walk(child, group)
        except OSError as error:
            errors.append({'path': relative, 'error': f'{type(error).__name__}: {error}'})
    for path in sorted(project.iterdir()):
        if path in excluded:
            continue
        if path.name == 'models':
            if include_unique_models:
                for name in UNIQUE_MODELS:
                    child = path / name
                    group = 'models/' + name
                    if not selected(group):
                        continue
                    if child.exists() or child.is_symlink():
                        walk(child, group)
                    else:
                        errors.append({'path': str(child.relative_to(project)), 'error': 'requested_unique_model_root_missing'})
            continue
        if only_groups and all(group.startswith('models/') for group in only_groups):
            continue
        if path.name in {'data', 'runs'} and path.is_dir() and not path.is_symlink():
            if only_groups and not any(group.startswith(path.name + '/') for group in only_groups):
                continue
            if selected(path.name + '/metadata'):
                rows[path.name] = {'path': path.name, 'group': path.name + '/metadata', 'stat': metadata(path)}
            for child in sorted(path.iterdir()):
                group = path.name + '/' + child.name
                if selected(group):
                    walk(child, group)
                else:
                    for scope in only_groups:
                        if not scope.startswith(group + '/'):
                            continue
                        target = project / scope
                        if any(parent.is_symlink() for parent in target.parents if parent != project and project in parent.parents):
                            errors.append({'path': scope, 'error': 'selected_group_has_symlink_parent'})
                        elif target.exists() or target.is_symlink():
                            walk(target, scope)
        elif path.name in {'configs', 'src', 'scripts', 'tests', 'monitor', 'deployment', 'outputs', 'docs', 'examples', 'notebooks'} or path.is_file() or path.is_symlink():
            walk(path, 'configs' if path.name == 'configs' else 'code')
    matched = {row['group'] for row in rows.values()}
    errors.extend({'path': group, 'error': 'selected_group_missing_or_empty'} for group in only_groups if group not in matched)
    return rows, errors


def load_catalog(output):
    catalog = {}
    archives = []
    for path in sorted(output.glob('*.manifest.json')):
        proof = read_regular_json(path)
        validate_manifest(proof)
        archive_name(proof['archive'], gzip_only=False)
        if path.name != manifest_name(proof['archive']):
            raise ValueError('Archive manifest filename differs from its archive')
        archive = output / proof['archive']
        reclaimed = False
        if not archive.exists() and not archive.is_symlink():
            require_reclaimed_receipt(output, proof)
            reclaimed = True
        elif not regular_file(archive) or archive.stat().st_size != proof['archive_bytes']:
            raise ValueError('Completed archive missing or size changed: ' + str(archive))
        item = {key: proof[key] for key in ('archive', 'archive_bytes', 'sha256', 'group', 'source_bytes', 'completed_at')}
        if reclaimed:
            item['remote_copy_reclaimed'] = True
        archives.append(item)
        for row in proof['members']:
            if row['stable']:
                catalog[row['path']] = row['before']
    return catalog, archives


class Reader:
    def __init__(self, source, state, publish):
        self.source, self.state, self.publish = source, state, publish
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size):
        data = self.source.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        self.state['source_bytes_read'] += len(data)
        self.state['next_read_at'] = max(time.monotonic(), self.state.get('next_read_at', 0)) + len(data) / self.state['rate']
        time.sleep(max(0, self.state['next_read_at'] - time.monotonic()))
        self.publish()
        return data


def pack_chunk(project, output, group, rows, name, compression, state, publish):
    final = output / (name + ('.tar.gz' if compression == 'gzip' else '.tar'))
    partial = final.with_suffix(final.suffix + '.part')
    if final.exists() or partial.exists():
        raise FileExistsError(final)
    if shutil.disk_usage(output).free < max(64 * 2**20, sum(r['stat']['size'] for r in rows if r['stat']['type'] == 'file') * 1.02):
        raise RuntimeError('Insufficient free disk for this archive chunk; source files remain untouched')
    sink = DigestSink(partial)
    members = []
    source_bytes = 0
    state['active'] = {'group': group, 'archive': final.name, 'files_total': len(rows), 'members_written': 0}
    def update():
        state['active']['archive_bytes'] = sink.size
        publish()
    gzip_level = state.get('gzip_level', 1)
    compressed = gzip.GzipFile(filename='', mode='wb', fileobj=sink, compresslevel=gzip_level, mtime=0) if compression == 'gzip' else sink
    try:
        with tarfile.open(fileobj=compressed, mode='w|', dereference=False, copybufsize=1024 * 1024) as archive:
            for row in rows:
                path = project / row['path']
                before = metadata(path)
                info = archive.gettarinfo(str(path), arcname=row['path'])
                if info is None:
                    raise ValueError('Unsupported archive member: ' + row['path'])
                digest, read_bytes = None, 0
                if info.isfile():
                    with path.open('rb') as source:
                        descriptor = os.fstat(source.fileno())
                        if (descriptor.st_ino, descriptor.st_size, descriptor.st_mtime_ns) != (before['inode'], before['size'], before['mtime_ns']):
                            raise RuntimeError('Source replaced before read: ' + row['path'])
                        reader = Reader(source, state, update)
                        archive.addfile(info, reader)
                        digest, read_bytes = reader.digest.hexdigest(), reader.bytes
                        source_bytes += read_bytes
                else:
                    archive.addfile(info)
                try:
                    after = metadata(path)
                except OSError:
                    after = None
                members.append({'path': row['path'], 'before': before, 'after': after, 'stable': before == after,
                    'changed_since_inventory': before != row['stat'], 'sha256_of_archived_file_bytes': digest,
                    'archived_file_bytes': read_bytes, 'tar_type': info.type.decode('ascii'),
                    'tar_link_target': info.linkname if info.issym() or info.islnk() else None})
                state['active']['members_written'] += 1
                update()
        if compression == 'gzip':
            compressed.close()
        sink.finish()
    except BaseException:
        sink.stream.close()
        raise
    partial.replace(final)
    proof = {'archive': final.name, 'group': group, 'project': str(project), 'archive_bytes': sink.size,
        'sha256': sink.digest.hexdigest(), 'hash_scope': 'archive_stream_and_each_regular_file_stream_once',
        'compression': f'gzip_level_{gzip_level}_single_thread' if compression == 'gzip' else 'uncompressed_tar',
        'source_bytes': source_bytes, 'members': members, 'live_changes': [r['path'] for r in members if not r['stable']],
        'completed_at': time.time(), 'atomic_filesystem_snapshot': False, 'source_deleted': False}
    write_json(output / (name + '.manifest.json'), proof)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/workspace/agentic-image-pilot-small'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--completed-run', action='append', default=[])
    parser.add_argument('--only-group', action='append', default=[])
    parser.add_argument('--include-unique-models', action='store_true')
    parser.add_argument('--read-mib-per-second', type=float, default=16)
    parser.add_argument('--chunk-mib', type=int, default=256)
    parser.add_argument('--compression', choices=('gzip', 'none'), default='gzip')
    parser.add_argument('--gzip-level', type=int, choices=(0, 1), default=1)
    parser.add_argument('--lock-wait-seconds', type=float, default=0)
    args = parser.parse_args()
    model_groups = {f'models/{name}' for name in UNIQUE_MODELS}
    if any(group.startswith('models/') and (not args.include_unique_models or group not in model_groups) for group in args.only_group):
        raise ValueError('Model groups require --include-unique-models and one of the three unique model roots')
    if args.read_mib_per_second <= 0 or args.chunk_mib < 1:
        raise ValueError('Read rate and chunk size must be positive')
    if not math.isfinite(args.lock_wait_seconds) or args.lock_wait_seconds < 0:
        raise ValueError('Lock wait seconds must be finite and nonnegative')
    project = args.project.resolve(strict=True)
    output = (args.output or project / 'delivery/project-results').resolve()
    if output == project or project.is_relative_to(output):
        raise ValueError('Archive output cannot contain the project root')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.pack.lock').open('a')
    acquire_lock(lock, args.lock_wait_seconds)
    low_priority()
    excluded = [output, *[project / name for name in ('models', 'delivery', '.venv', '.venv-tools', '.runtime', '.cache', '.git')]]
    if args.include_unique_models:
        excluded.remove(project / 'models')
        excluded.extend(project / 'models' / name for name in ('actor', 'renderer', 'judge'))
    catalog, archives = load_catalog(output)
    generation = str(time.time_ns())
    state = {'status': 'inventory', 'pid': os.getpid(), 'started_at': time.time(), 'project': str(project),
        'output': str(output), 'source_bytes_read': 0, 'archives': archives, 'errors': [],
        'excluded_exact_roots': [str(p) for p in excluded], 'completed_runs_priority_declared': args.completed_run,
        'scope': 'data_runs_configs_project_code_and_deployment_records', 'only_groups': args.only_group,
        'include_unique_models': args.include_unique_models, 'unique_model_roots': list(UNIQUE_MODELS) if args.include_unique_models else [],
        'original_bf16_model_roots_included': False,
        'read_mib_per_second': args.read_mib_per_second, 'chunk_mib': args.chunk_mib, 'gzip_level': args.gzip_level,
        'atomic_filesystem_snapshot': False, 'source_deleted': False, 'full_return_completed': False,
        'clock': time.monotonic(), 'rate': args.read_mib_per_second * 2**20}
    last_publish = [0.0]
    def publish(force=False):
        if force or time.monotonic() - last_publish[0] >= 5:
            state['updated_at'] = time.time()
            write_json(output / 'status.json', {k: v for k, v in state.items() if k not in {'clock', 'rate', 'next_read_at'}})
            last_publish[0] = time.monotonic()
    publish(True)
    rows, errors = inventory(project, excluded, args.include_unique_models, args.only_group)
    state['errors'].extend(errors)
    if args.only_group and errors:
        state.update(status='inventory_failed', completed_at=time.time())
        publish(True)
        raise ValueError('The selected archive scope could not be read: ' + json.dumps(errors))
    wanted = lambda row: not args.only_group or row['group'] in args.only_group
    pending = [row for row in rows.values() if wanted(row) and catalog.get(row['path']) != row['stat']]
    groups = {}
    for row in pending:
        groups.setdefault(row['group'], []).append(row)
    completed = {'runs/' + name for name in args.completed_run}
    priority = lambda group: (0 if group in completed else 1 if group in {'code', 'configs'} else 2 if group.startswith('data/') else 3, group)
    state.update(status='packing', inventory_members=len(rows), pending_members=len(pending), groups_order=sorted(groups, key=priority))
    publish(True)
    part = 0
    for group in sorted(groups, key=priority):
        chunks, chunk, size = [], [], 0
        for row in sorted(groups[group], key=lambda r: r['path']):
            count = row['stat']['size'] if row['stat']['type'] == 'file' else 0
            if chunk and size + count > args.chunk_mib * 2**20:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append(row)
            size += count
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            part += 1
            name = f'{generation}-{part:05d}-' + re.sub('[^A-Za-z0-9_.-]', '-', group)
            try:
                proof = pack_chunk(project, output, group, chunk, name, args.compression, state, publish)
                state['archives'].append({key: proof[key] for key in ('archive', 'archive_bytes', 'sha256', 'group', 'source_bytes', 'completed_at')})
                for row in proof['members']:
                    if row['stable']:
                        catalog[row['path']] = row['before']
            except Exception as error:
                state['errors'].append({'archive': name, 'group': group, 'error': f'{type(error).__name__}: {error}'})
            publish(True)
    after, errors = inventory(project, excluded, args.include_unique_models, args.only_group)
    state['errors'].extend(errors)
    changed = [row['path'] for row in after.values() if wanted(row) and catalog.get(row['path']) != row['stat']]
    missing = [path for path in catalog if path not in after]
    state.update(status='delta_required' if changed or state['errors'] else 'snapshot_catalog_caught_up',
        completed_at=time.time(), active=None, pending_changed_or_new_paths=changed,
        disappeared_paths_retained_in_previous_archives=missing,
        archive_bytes=sum(r['archive_bytes'] for r in state['archives']),
        snapshot_scope='metadata_observation_at_end_of_pass_not_a_frozen_live_filesystem')
    publish(True)
    write_json(output / (generation + '.pass.json'), {k: v for k, v in state.items() if k not in {'clock', 'rate', 'next_read_at'}})
    print(json.dumps({'status': state['status'], 'new_archives': part, 'pending_paths': len(changed),
        'errors': len(state['errors']), 'archive_bytes': state['archive_bytes'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
