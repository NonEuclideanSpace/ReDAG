import concurrent.futures
import io
import json
from pathlib import Path
import pickle
import sys
import time
import urllib.request


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Only plain metadata is accepted')


project = Path('/workspace/agentic-image-pilot-small')
role = sys.argv[1]
metadata = json.loads((project / f'deployment/rental-{role}-hf.json').read_text())
repo = metadata['id']
source = Path('/model/ModelScope') / repo
registered = MetadataReader(io.BytesIO((source / '.msc').read_bytes())).load()
revisions = {row['Revision'] for row in registered if row['Path'].endswith('.safetensors')}
published = {}
for revision in revisions:
    url = f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision={revision}&Recursive=true'
    with urllib.request.urlopen(url, timeout=25) as response:
        payload = json.load(response)
    published.update({row['Path']: row for row in payload['Data']['Files']})
destination = project / 'models' / role / metadata['sha']
destination.mkdir(parents=True, exist_ok=True)
records = []
for row in metadata['siblings']:
    name = row['rfilename']
    if not name.endswith('.safetensors') or role == 'renderer' and '/' not in name:
        continue
    actual = published.get(name, {})
    if actual.get('Sha256') != row['lfs']['sha256'] or actual.get('Size') != row['size']:
        raise ValueError(f'Published weight metadata mismatch: {name}')
    if (source / name).stat().st_size != row['size']:
        raise ValueError(f'Cached weight size mismatch: {name}')
    records.append({'name': name, 'size': row['size'], 'sha256': row['lfs']['sha256'], 'source_revision': actual['Revision']})
(project / f'deployment/rental-{role}-cache-evidence.json').write_text(json.dumps({'repo': repo, 'hf_revision': metadata['sha'], 'metadata_matches': True, 'local_full_hash_scan': False, 'files': records}, indent=2))
print('CACHE_METADATA_MATCHED', role, sum(row['size'] for row in records), flush=True)


def copy(row):
    name = row['name']
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == row['size']:
        return
    partial = target.with_name(target.name + '.incomplete')
    offset = partial.stat().st_size if partial.exists() else 0
    started = time.monotonic()
    initial = offset
    with (source / name).open('rb') as reader, partial.open('ab') as writer:
        reader.seek(offset)
        reported = time.monotonic()
        while chunk := reader.read(8 * 1024**2):
            writer.write(chunk)
            offset += len(chunk)
            if time.monotonic() - reported >= 10:
                print(json.dumps({'role': role, 'file': name, 'bytes': offset, 'total': row['size'], 'MiB_s': round((offset-initial)/(time.monotonic()-started)/1024**2,2)}), flush=True)
                reported = time.monotonic()
    if partial.stat().st_size != row['size']:
        raise ValueError(f'Copied weight size mismatch: {name}')
    partial.replace(target)
    print('COPIED', role, name, row['size'], flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(copy, records))
print('CACHE_WEIGHTS_READY', role, destination, flush=True)
