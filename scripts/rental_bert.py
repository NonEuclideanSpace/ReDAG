import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


root = Path('/workspace/agentic-image-pilot-small')
metadata = json.loads((root / 'deployment/rental-bert-hf.json').read_text())
names = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
official = {row['rfilename']: row for row in metadata['siblings']}
with urllib.request.urlopen('https://modelscope.cn/api/v1/models/google-bert/bert-base-uncased/repo/files?Revision=master&Recursive=true', timeout=20) as response:
    alternate = {row['Path']: row for row in json.load(response)['Data']['Files']}
destination = root / 'models/tools/bert-base-uncased'
destination.mkdir(parents=True, exist_ok=True)
records = []
for name in names:
    primary, source = official[name], alternate[name]
    if primary.get('lfs') and (source['Sha256'] != primary['lfs']['sha256'] or source['Size'] != primary['size']):
        raise ValueError(f'BERT pinned weight metadata mismatch: {name}')
    url = f"https://modelscope.cn/models/google-bert/bert-base-uncased/resolve/{source['Revision']}/{name}"
    path = destination / name
    subprocess.run(['curl', '--fail', '--location', '--continue-at', '-', '--connect-timeout', '15', '--max-time', '180', '--retry', '2', url, '-o', str(path)], check=True)
    if path.stat().st_size != primary['size']:
        raise ValueError(f'BERT file size mismatch: {name}')
    if not primary.get('lfs'):
        data = path.read_bytes()
        if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != primary['blobId']:
            raise ValueError(f'BERT pinned metadata mismatch: {name}')
    records.append({'name': name, 'bytes': path.stat().st_size, 'source': url})
    print('BERT_FILE_COMPLETE', name, path.stat().st_size, flush=True)
(root / 'deployment/rental-bert-download.json').write_text(json.dumps({'revision': metadata['sha'], 'complete': True, 'files': records}, indent=2))
