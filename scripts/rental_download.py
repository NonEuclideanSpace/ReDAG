import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request


project = Path('/workspace/agentic-image-pilot-small')
role = sys.argv[1]
metadata = json.loads((project / f'deployment/rental-{role}-hf.json').read_text())
repo = metadata['id']
revision = metadata['sha']
destination = project / 'models' / role / revision
destination.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, headers=None):
    return opener.open(urllib.request.Request(url, headers=headers or {}), timeout=35)


with request(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true') as response:
    source = json.load(response)
files = {row['Path']: row for row in source['Data']['Files']}
(project / f'deployment/rental-{role}-ms.json').write_text(json.dumps(source, indent=2))
selected = []
for row in metadata['siblings']:
    name = row['rfilename']
    if name == '.gitattributes' or name.lower().endswith(('.jpg', '.png', '.mp4')):
        continue
    if role == 'renderer' and '/' not in name and name.endswith('.safetensors'):
        continue
    alternate = files.get(name)
    matched = bool(alternate and row.get('lfs') and alternate.get('Sha256') == row['lfs']['sha256'] and alternate.get('Size') == row['size'])
    if matched:
        url = f"https://modelscope.cn/models/{repo}/resolve/{alternate['Revision']}/{urllib.parse.quote(name)}"
    else:
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}'
    selected.append({**row, 'url': url, 'mirror_weight_metadata_matches': matched})
(project / f'deployment/rental-{role}-sources.json').write_text(json.dumps({'repo': repo, 'revision': revision, 'files': selected}, indent=2))


def download(row):
    name, total = row['rfilename'], row['size']
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.incomplete')
    if target.exists() and target.stat().st_size == total:
        return
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == total:
            partial.replace(target)
            break
        try:
            started = time.monotonic()
            received = 0
            with request(row['url'], {'Range': f'bytes={offset}-'} if offset else None) as response:
                if offset and response.status != 206:
                    raise RuntimeError('Resume requires HTTP 206')
                with partial.open('ab' if offset else 'wb') as stream:
                    reported = time.monotonic()
                    while chunk := response.read(4 * 1024**2):
                        stream.write(chunk)
                        offset += len(chunk)
                        received += len(chunk)
                        if time.monotonic() - reported >= 10:
                            print(json.dumps({'role': role, 'file': name, 'bytes': offset, 'total': total, 'MiB_s': round(received / (time.monotonic()-started) / 1024**2, 2)}), flush=True)
                            reported = time.monotonic()
            if partial.stat().st_size != total:
                raise RuntimeError(f'Length mismatch: {partial.stat().st_size} != {total}')
            partial.replace(target)
            break
        except Exception as error:
            print(json.dumps({'role': role, 'file': name, 'attempt': attempt + 1, 'error': str(error)}), flush=True)
            if attempt == 2:
                raise
    print(json.dumps({'role': role, 'complete': name, 'bytes': total}), flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, sorted(selected, key=lambda row: -row['size'])))
(project / f'deployment/rental-{role}-download.json').write_text(json.dumps({'complete': True, 'path': str(destination), 'revision': revision, 'bytes': sum(row['size'] for row in selected)}, indent=2))
print('DOWNLOAD_COMPLETE', role, destination, flush=True)
