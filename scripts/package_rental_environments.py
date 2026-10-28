import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def probe_python(executable):
    code = """import importlib.metadata as m,json,os,platform,sys,sysconfig
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
def public_url(value):
 parts=urlsplit(value)
 host=parts.hostname or ''
 if parts.port: host+=':'+str(parts.port)
 return urlunsplit((parts.scheme,host,parts.path,'','')) if parts.scheme in ('http','https') else value
packages=[]
editable=[]
for d in m.distributions():
 value=d.read_text('direct_url.json')
 value=json.loads(value) if value else None
 if value and 'url' in value: value['url']=public_url(value['url'])
 files=d.files or []
 native=[str(item) for item in files if str(item).endswith(('.so','.pyd','.dll','.dylib'))]
 packages.append({'name':d.metadata.get('Name'),'version':d.version,'installer':d.read_text('INSTALLER'),'wheel':d.read_text('WHEEL'),'direct_url':value,'location':str(d.locate_file('')),'native_files':native,'origin_known':bool(value)})
 if value and value.get('dir_info',{}).get('editable'): editable.append(value)
conda=[]
for path in sorted((Path(sys.prefix)/'conda-meta').glob('*.json')):
 value=json.loads(path.read_text())
 entry={key:value.get(key) for key in ('name','version','build','build_number','subdir','fn','sha256','md5','link')}
 for key in ('url','channel'):
  if value.get(key): entry[key]=public_url(value[key])
 conda.append(entry)
print(json.dumps({'executable':sys.executable,'real_executable':os.path.realpath(sys.executable),'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':sys.version,'machine':platform.machine(),'platform':platform.platform(),'site_packages':sysconfig.get_paths().get('purelib'),'packages':packages,'editable':editable,'conda_packages':conda,'package_cache_scope':'recorded_conda_link_and_installed_direct_url_only_no_cache_scan'}))"""
    result = subprocess.run([str(executable), '-c', code], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout)


class ArchiveSink:
    def __init__(self, path):
        self.file = path.open('wb')
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data):
        self.digest.update(data)
        self.size += len(data)
        return self.file.write(data)

    def flush(self):
        self.file.flush()

    def finish(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


class PacedReader:
    def __init__(self, stream, progress):
        self.stream = stream
        self.progress = progress

    def read(self, size):
        data = self.stream.read(size)
        state = self.progress
        state['raw_bytes'] += len(data)
        delay = state['raw_bytes'] / state['rate'] - (time.monotonic() - state['clock'])
        if delay > 0:
            time.sleep(delay)
        state['publish']()
        return data


def archive(name, roots, output, rate, status, publish):
    destination = output / (name + '.tar.gz')
    proof_path = output / (name + '.manifest.json')
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if proof['roots'] != [str(path) for path in roots]:
            raise ValueError('Completed archive roots differ; use a new output directory')
        if destination.is_file() and destination.stat().st_size == proof['archive_bytes']:
            status['archives'][name] = proof
            publish()
            return proof
        raise ValueError('A completed archive is missing or has changed size')
    partial = destination.with_suffix(destination.suffix + '.part')
    if partial.exists():
        partial.replace(partial.with_name(partial.name + '.interrupted-' + str(time.time_ns())))
    sink = ArchiveSink(partial)
    start = time.monotonic()
    entry = {'status': 'packing', 'file': destination.name, 'raw_bytes': 0, 'archive_bytes': 0, 'members': 0, 'roots': [str(path) for path in roots]}
    status['archives'][name] = entry
    progress = {'raw_bytes': 0, 'rate': rate, 'clock': start}
    last_publish = [0.0]
    def update():
        now = time.monotonic()
        if now - last_publish[0] >= 5:
            entry.update(raw_bytes=progress['raw_bytes'], archive_bytes=sink.size, elapsed_seconds=now-start)
            publish()
            last_publish[0] = now
    progress['publish'] = update
    links = []
    excluded = []
    try:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=sink, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', dereference=False, copybufsize=1024*1024) as stream:
                def add(path):
                    if path.name in {'__pycache__', '.pytest_cache', '.git'} or path.suffix == '.pyc':
                        return
                    info = stream.gettarinfo(str(path), arcname=str(path).lstrip('/'))
                    if info is None:
                        raise ValueError('Unsupported environment filesystem member: ' + str(path))
                    if info.issym():
                        resolved = path.resolve(strict=False)
                        links.append({'path': str(path), 'target': info.linkname, 'resolved': str(resolved), 'exists': resolved.exists()})
                    if info.isfile():
                        before = path.stat()
                        with path.open('rb') as source:
                            stream.addfile(info, PacedReader(source, progress))
                        after = path.stat()
                        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                            raise RuntimeError('Environment file changed while packing: ' + str(path))
                    else:
                        stream.addfile(info)
                    entry['members'] += 1
                    update()
                    if info.isdir():
                        for child in sorted(path.iterdir()):
                            add(child)
                for path in roots:
                    add(path)
        sink.finish()
    except BaseException:
        sink.file.close()
        raise
    partial.replace(destination)
    proof = {**entry, 'status': 'complete', 'archive_bytes': sink.size, 'raw_bytes': progress['raw_bytes'], 'sha256': sink.digest.hexdigest(), 'hash_scope': 'compressed_archive_stream_once', 'compression': 'gzip_level_1_single_thread', 'elapsed_seconds': time.monotonic()-start, 'symlinks': links, 'excluded_paths': excluded, 'completed_at': time.time()}
    write_json(proof_path, proof)
    status['archives'][name] = proof
    publish()
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/workspace/agentic-image-pilot-small'))
    parser.add_argument('--main-prefix', type=Path, default=Path('/usr/local/miniconda3/envs/py312'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--read-mib-per-second', type=float, default=16)
    parser.add_argument('--include', type=Path, action='append', default=[])
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Package the actual Linux rental environments on the rental instance')
    if args.read_mib_per_second <= 0:
        raise ValueError('Read rate must be positive')
    project = args.project.absolute()
    output = (args.output or project / 'delivery/linux-environments').absolute()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.pack.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(19)
    if Path('/usr/bin/ionice').exists():
        subprocess.run(['/usr/bin/ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    inventory = {}
    groups = {}
    for name, prefix in [('py312', args.main_prefix.absolute()), ('tools', project / '.venv-tools')]:
        inventory[name] = probe_python(prefix / 'bin/python')
        roots = [prefix]
        resolved = prefix.resolve()
        if resolved != prefix:
            roots.append(resolved)
        base = Path(inventory[name]['base_prefix'])
        if not inside(base, roots) and not inside(base, [Path('/usr'), Path('/opt')]):
            roots.append(base)
        elif base != Path('/usr') and not inside(base, roots):
            roots.append(base)
        groups[name] = roots
    extras = [project / item for item in ('.runtime', 'third_party', 'src', 'pyproject.toml', 'scripts/serve_tools.py')]
    extras += args.include
    extras = [path.absolute() for path in extras if path.exists() or path.is_symlink()]
    for path in list(extras):
        resolved = path.resolve()
        if resolved != path and not inside(resolved, extras):
            extras.append(resolved)
    covered = groups['py312'] + groups['tools']
    groups['support'] = [path for path in extras if not inside(path, covered)]
    for roots in groups.values():
        for path in roots:
            if path in {Path('/'), Path('/usr'), Path('/usr/local'), Path('/opt'), project}:
                raise ValueError('Refusing an unbounded environment root: ' + str(path))
            if inside(path, [project/'models', output]):
                raise ValueError('Models and archive outputs cannot be environment roots')
            if not path.exists():
                raise FileNotFoundError(path)
    status = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(), 'project': str(project), 'platform': platform.platform(), 'machine': platform.machine(), 'read_mib_per_second': args.read_mib_per_second, 'inventory': inventory, 'archives': {}, 'public_model_weights_included': False, 'restoration_verified': False}
    def publish():
        status['updated_at'] = time.time()
        write_json(output / 'status.json', status)
    write_json(output / 'installed-package-origins.json', inventory)
    publish()
    try:
        for name, roots in groups.items():
            archive(name, roots, output, args.read_mib_per_second*1024*1024, status, publish)
        all_roots = [path for roots in groups.values() for path in roots]
        external = []
        for result in status['archives'].values():
            for link in result['symlinks']:
                target = Path(link['resolved'])
                if not inside(target, all_roots):
                    external.append({**link, 'system_dependency': inside(target, [Path('/lib'), Path('/lib64'), Path('/usr/lib'), Path('/usr/bin'), Path('/bin')])})
        status.update(status='complete', completed_at=time.time(), external_symlinks=external, unresolved_non_system_symlinks=[item for item in external if not item['system_dependency']], archive_bytes=sum(item['archive_bytes'] for item in status['archives'].values()))
        write_json(output / 'manifest.json', status)
        publish()
        print(json.dumps({'status': status['status'], 'archive_bytes': status['archive_bytes'], 'unresolved_non_system_symlinks': status['unresolved_non_system_symlinks'], 'output': str(output)}))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        publish()
        raise


if __name__ == '__main__':
    main()
