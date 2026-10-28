import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import tarfile
import time


PROJECT = '/workspace/agentic-image-pilot-small'


def relative_name(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or str(path) != value or value == '.':
        raise ValueError('Unsafe project-relative path: ' + str(value))
    return str(path)


def internal_link(path, target, project):
    if not isinstance(target, str) or not target:
        raise ValueError('A symbolic link needs its recorded target')
    if target.startswith('/'):
        full = PurePosixPath(target)
        base = PurePosixPath(project)
        if '..' in full.parts or not full.is_relative_to(base):
            raise ValueError('External symbolic link requires a separate dependency: ' + path)
        resolved = str(full.relative_to(base))
    else:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
    if resolved == '..' or resolved.startswith('../') or resolved.startswith('/'):
        raise ValueError('Symbolic link escapes the restored project: ' + path)
    return posixpath.relpath(resolved, posixpath.dirname(path) or '.')


def code_metadata(backup, selected):
    directory = backup / 'code-revisions'
    expected_digest = None
    if selected is None:
        index = backup / 'DELIVERY_INDEX.json'
        if index.is_file():
            recommendation = json.loads(index.read_text())['code']
            name = relative_name(recommendation['archive'])
            selected = backup / name
            expected_digest = recommendation.get('sha256')
        else:
            pointer = directory / 'code-update-order.json'
            if not pointer.is_file():
                raise ValueError('Select an integrated code archive explicitly; no code pointer is present')
            recommendation = json.loads(pointer.read_text())
            selected = recommendation['recommended_integrated_source_archive']
            expected_digest = recommendation.get('recommended_integrated_source_sha256')
    path = Path(selected)
    path = path if path.is_absolute() else directory / path
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 2**20:
        raise ValueError('An existing small integrated code archive is required')
    records = [path.with_name(path.name.removesuffix('.tar.gz') + '.manifest.json'),
        backup / 'reports' / ('code-snapshot-' + path.name.removeprefix('pilot-small-code-').removesuffix('.tar.gz') + '.json')]
    for record in records:
        if record.is_file():
            value = json.loads(record.read_text()).get('sha256')
            if value:
                if expected_digest and expected_digest != value:
                    raise ValueError('Integrated code checksum records disagree')
                expected_digest = value
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_digest and digest != expected_digest:
        raise ValueError('Integrated code checksum differs from its delivery record')
    members = []
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = relative_name(member.name)
            parts = PurePosixPath(name).parts
            if parts[0] != 'pilot-small':
                raise ValueError('Integrated code archive must use the pilot-small prefix')
            if len(parts) == 1 and member.isdir():
                continue
            stripped = relative_name(str(PurePosixPath(*parts[1:])))
            if not member.isfile() and not member.isdir():
                raise ValueError('Integrated code contains an unsupported link or special file')
            if stripped in members:
                raise ValueError('Integrated code contains a duplicate member: ' + stripped)
            members.append(stripped)
    if not any(name.startswith('src/') for name in members):
        raise ValueError('Integrated code archive does not contain project source')
    return {'archive': str(path), 'bytes': path.stat().st_size,
        'sha256': digest, 'expected_checksum_record_present': bool(expected_digest), 'members': members,
        'prefix_removed': 'pilot-small/'}


def build_plan(backup_root, code_archive=None):
    backup = Path(backup_root).resolve(strict=True)
    directory = backup / 'remote/project-results-archives'
    errors, warnings, archives, latest, selected, versions = [], [], [], {}, {}, {}
    manifests = sorted(path for path in directory.glob('*.manifest.json') if not path.name.startswith('._'))
    if not manifests:
        errors.append('No project archive manifests are present')
    project = None
    for path in manifests:
        proof = json.loads(path.read_text())
        name = proof['archive']
        if not re.fullmatch(r'\d+-\d+-[A-Za-z0-9_.-]+\.tar(?:\.gz)?', name):
            raise ValueError('Unexpected generated archive name: ' + name)
        expected_manifest = name.removesuffix('.gz').removesuffix('.tar') + '.manifest.json'
        if path.name != expected_manifest:
            raise ValueError('Archive manifest name differs')
        if project is None:
            project = proof['project']
        if proof['project'] != project:
            raise ValueError('Project archive manifests disagree on the original project')
        archive = directory / name
        valid = archive.is_file() and not archive.is_symlink() and archive.stat().st_size == proof['archive_bytes']
        if not valid:
            errors.append('Missing or incomplete archive: ' + name)
        archives.append({'archive': str(archive), 'name': name, 'bytes': proof['archive_bytes'],
            'sha256': proof['sha256'], 'manifest': str(path), 'size_verified': valid})
        for row in proof['members']:
            member = relative_name(row['path'])
            kind = row['tar_type']
            entry = {'archive': name, 'path': member, 'type': kind,
                'size': row.get('archived_file_bytes', row['before']['size']),
                'mode': row['before']['mode'] & 0o777,
                'mtime_ns': row['before'].get('mtime_ns'),
                'source_identity': {key: row['before'].get(key) for key in ('size', 'mtime_ns', 'inode', 'device')},
                'sha256': row.get('sha256_of_archived_file_bytes'), 'stable': row.get('stable') is True}
            if kind in ('1', '2'):
                entry['recorded_link_target'] = row.get('tar_link_target')
            latest[member] = entry
            versions[member] = versions.get(member, 0) + 1
            if entry['stable']:
                selected[member] = entry
    for name, row in latest.items():
        if not row['stable']:
            errors.append('Latest observed member is unstable: ' + name)
    for name in selected:
        row = selected[name]
        if row['type'] not in ('0', '\x00', '5', '2', '1'):
            errors.append('Unsupported archive member type ' + repr(row['type']) + ': ' + name)
        if row['type'] in ('1', '2'):
            try:
                row['link_target'] = (internal_link(name, row['recorded_link_target'], project) if row['type'] == '2'
                    else relative_name(row['recorded_link_target']))
            except (TypeError, ValueError) as error:
                errors.append(str(error))
                continue
        if row['type'] == '1':
            target = selected.get(row['link_target'])
            if not target or target['type'] not in ('0', '\x00', '1') or target['source_identity'] != row['source_identity']:
                errors.append('Hardlink target version is unavailable or changed: ' + name)
        for parent in PurePosixPath(name).parents:
            ancestor = selected.get(str(parent))
            if ancestor and ancestor['type'] != '5':
                errors.append('Retained member has a non-directory ancestor: ' + name)
                break
    code = code_metadata(backup, code_archive)
    newest = sorted(path for path in (backup / 'code-revisions').glob('pilot-small-code-*.tar.gz') if not path.name.startswith('._'))
    if newest and Path(code['archive']).name != newest[-1].name:
        warnings.append('A newer integrated code archive is present: ' + newest[-1].name)
    if not (directory / 'status.json').is_file() and not (backup / 'manifests/project-results/status.json').is_file():
        warnings.append('Final project catalog status and writer-completion evidence are not present at the archive entry point')
    if not (backup / 'manifests/transfer-receipts/project-results').is_dir() and not (backup / 'reports/result-archive-transfers').is_dir():
        warnings.append('Per-archive transfer receipts are not copied to the external entry point')
    wanted = {row['archive'] for row in selected.values()}
    return {'schema': 'rental_bundle_restore_plan_v1', 'created_at': time.time(), 'backup_root': str(backup),
        'original_project': project or PROJECT, 'ready': not errors, 'errors': errors, 'warnings': warnings,
        'archives': archives, 'selected_archives': [row['name'] for row in archives if row['name'] in wanted],
        'selected_members': selected, 'archive_order': 'lexicographic generation then chunk sequence',
        'manifest_count': len(manifests), 'selected_path_count': len(selected),
        'multiple_version_paths': sum(value > 1 for value in versions.values()), 'code': code,
        'deleted_paths_replayed': False, 'selected_restore_is_final_delivery': False,
        'scope': 'Latest stable retained project members plus selected integrated source; models and Linux environments are separate',
        'large_archive_hashes_verified_during_restore': True}


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def write_member(archive, member, target, expected=None):
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return
    if not member.isfile():
        raise ValueError('Only regular files may be written directly')
    if expected and member.size != expected['size']:
        raise ValueError('Archive member size differs: ' + member.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_dir():
        raise ValueError('A file would overwrite a symbolic link or directory: ' + str(target))
    temporary = target.with_name(target.name + '.restoring')
    digest = hashlib.sha256()
    with archive.extractfile(member) as source, temporary.open('xb') as output:
        while data := source.read(2**20):
            output.write(data)
            digest.update(data)
    if expected and expected.get('sha256') and digest.hexdigest() != expected['sha256']:
        raise ValueError('Archived member checksum differs: ' + member.name)
    os.replace(temporary, target)
    os.chmod(target, member.mode & 0o777)
    if expected and expected.get('mtime_ns') is not None:
        os.utime(target, ns=(expected['mtime_ns'], expected['mtime_ns']))
    else:
        os.utime(target, (member.mtime, member.mtime))


def restore_plan(plan, destination):
    if plan.get('schema') != 'rental_bundle_restore_plan_v1' or not plan.get('ready') or plan.get('errors'):
        raise ValueError('A complete selected restore plan is required')
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Restore destination must be new and unused')
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError('Restore destination must not traverse symbolic links')
    report_path = destination.parent / (destination.name + '.restore-report.json')
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError('Restore report must be new and unused')
    destination.mkdir(parents=True)
    selected = plan['selected_members']
    completed, restored, links, hardlinks, directories = [], set(), {}, {}, {}
    report = {'status': 'restoring', 'started_at': time.time(), 'destination': str(destination),
        'original_project': plan['original_project'], 'source_backup': plan['backup_root'],
        'deleted_paths_replayed': False, 'model_or_environment_restored': False, 'gpu_verified': False,
        'full_offline_restore_verified': False, 'archive_checksums_verified': completed}
    def publish():
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(report_path)
    publish()
    try:
        for info in plan['archives']:
            if info['name'] not in plan['selected_archives']:
                continue
            source = Path(info['archive'])
            if source.is_symlink() or source.stat().st_size != info['bytes']:
                raise ValueError('Archive size changed after planning: ' + info['name'])
            with source.open('rb') as incoming:
                reader = DigestReader(incoming)
                with tarfile.open(fileobj=reader, mode='r|gz' if source.name.endswith('.gz') else 'r|') as archive:
                    for member in archive:
                        name = relative_name(member.name)
                        row = selected.get(name)
                        if row is None or row['archive'] != info['name']:
                            continue
                        if name in restored or member.type.decode('ascii') != row['type']:
                            raise ValueError('Duplicate or mismatched selected archive member: ' + name)
                        target = destination / name
                        if member.issym():
                            link = internal_link(name, member.linkname, plan['original_project'])
                            if link != row['link_target']:
                                raise ValueError('Symbolic link target differs from its manifest')
                            links[name] = link
                        elif member.islnk():
                            if relative_name(member.linkname) != row['link_target']:
                                raise ValueError('Hardlink target differs from its manifest')
                            hardlinks[name] = row['link_target']
                        else:
                            write_member(archive, member, target, row)
                            if member.isdir():
                                directories[name] = row
                        restored.add(name)
                while reader.read(2**20):
                    pass
            if reader.digest.hexdigest() != info['sha256']:
                raise ValueError('Archive checksum differs: ' + info['name'])
            completed.append(info['name'])
            report['restored_project_paths'] = len(restored)
            publish()
        if restored != set(selected):
            raise ValueError('One or more selected members were absent from their archives')
        code = plan['code']
        if Path(code['archive']).stat().st_size != code['bytes']:
            raise ValueError('Integrated code archive changed after planning')
        with Path(code['archive']).open('rb') as incoming:
            reader = DigestReader(incoming)
            with tarfile.open(fileobj=reader, mode='r|gz') as archive:
                for member in archive:
                    parts = PurePosixPath(relative_name(member.name)).parts
                    if parts[0] != 'pilot-small':
                        raise ValueError('Unexpected code archive prefix')
                    if len(parts) == 1 and member.isdir():
                        continue
                    name = relative_name(str(PurePosixPath(*parts[1:])))
                    write_member(archive, member, destination / name)
                    links.pop(name, None)
                    directories.pop(name, None)
            while reader.read(2**20):
                pass
        if reader.digest.hexdigest() != code['sha256']:
            raise ValueError('Integrated code checksum differs')
        for name in hardlinks:
            if name in code['members'] or hardlinks[name] in code['members']:
                raise ValueError('Code overlay conflicts with a retained hardlink')
        while hardlinks:
            ready = [name for name, target in hardlinks.items() if (destination / target).is_file() and not (destination / target).is_symlink()]
            if not ready:
                raise ValueError('Hardlink targets form a cycle or are missing')
            for name in ready:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(destination / hardlinks.pop(name), target)
        for name, link in links.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise ValueError('A symbolic link would overwrite another restored member')
            target.symlink_to(link)
        for name, row in sorted(directories.items(), key=lambda item: len(PurePosixPath(item[0]).parts), reverse=True):
            target = destination / name
            os.chmod(target, row['mode'])
            if row['mtime_ns'] is not None:
                os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
        report.update(status='selected_project_restored', completed_at=time.time(),
            restored_project_paths=len(restored), code_archive=code['archive'], code_sha256=code['sha256'],
            symbolic_links_restored=len(links), old_runtime_state_must_not_be_launched=True,
            code_overlay_exact_file_set=False)
    except BaseException as error:
        report.update(status='failed', error=str(error), failed_at=time.time())
        raise
    finally:
        publish()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup-root', type=Path)
    parser.add_argument('--code-archive', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.execute:
        if not args.destination:
            raise ValueError('Execution requires a new destination')
        result = restore_plan(json.loads(args.plan.read_text()), args.destination)
    else:
        if not args.backup_root:
            raise ValueError('Planning requires the external backup root')
        result = build_plan(args.backup_root, args.code_archive)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(result, indent=2) + '\n')
    summary = {key: result[key] for key in ('ready', 'warnings', 'status', 'selected_path_count', 'destination') if key in result}
    if 'errors' in result:
        summary.update(error_count=len(result['errors']), first_errors=result['errors'][:20])
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
