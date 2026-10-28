import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request


project = Path('/workspace/agentic-image-pilot-small')
role = sys.argv[1]
metadata = json.loads((project / f'deployment/rental-{role}-hf.json').read_text())
repo = metadata['id']
with urllib.request.urlopen(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true', timeout=20) as response:
    source = {row['Path']: row for row in json.load(response)['Data']['Files']}
destination = project / 'models' / role / metadata['sha']
platform = Path('/model/ModelScope') / repo


def matches(data, row):
    if len(data) != row['size']:
        return False
    if row.get('lfs'):
        return hashlib.sha256(data).hexdigest() == row['lfs']['sha256']
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == row['blobId']


def get(row):
    name = row['rfilename']
    target = destination / name
    cached = platform / name
    if target.is_file() and matches(target.read_bytes(), row):
        return
    data = cached.read_bytes() if cached.is_file() else None
    origin = str(cached)
    if data is None or not matches(data, row):
        candidate = source[name]
        origin = f"https://modelscope.cn/models/{repo}/resolve/{candidate['Revision']}/{name}"
        with urllib.request.urlopen(origin, timeout=30) as response:
            data = response.read()
    if not matches(data, row):
        raise ValueError(f'Pinned metadata content mismatch: {name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(json.dumps({'role': role, 'file': name, 'bytes': len(data), 'matches_pinned_hf_blob': True, 'source': origin}), flush=True)


rows = [row for row in metadata['siblings'] if row['rfilename'].endswith(('.json', '.jinja', '.txt', '.model'))]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(get, rows))
print('PINNED_METADATA_READY', role, flush=True)
