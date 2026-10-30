"""Download the six model checkpoints from ModelScope (fast inside China) into the Hugging Face cache layout.

  $HF_HOME/hub/models--<org>--<name>/snapshots/<sha>/...   the files
  $MODELS/<org>/<name> -> that snapshot                      what scripts/launch/env.sh points to

Resumable: complete files are skipped, partial ones (*.incomplete) continue with HTTP range requests, so the same
command can be rerun after any interruption (or on a cloned instance).  --check only verifies (sizes and that every
shard named in a *.safetensors.index.json exists).  ~120 GiB in total.
  python scripts/launch/download_models.py [--models Qwen/Qwen3.5-9B ...] [--workers 16] [--check]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODELS = ['IDEA-Research/grounding-dino-base', 'facebook/sam2.1-hiera-large', 'Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3.5-9B',
          'black-forest-labs/FLUX.2-klein-base-9B', 'Qwen/Qwen3.8-27B']
# duplicates of the safetensors weights and sample images that nothing loads
SKIP_NAMES = {'.gitattributes', 'LICENSE', 'LICENSE.md', 'README.md', 'editing.jpg', 'others.jpg', 'realism.jpg',
              'flux-2-klein-base-9b.safetensors', 'pytorch_model.bin', 'sam2.1_hiera_large.pt'}
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def get_json(url, attempts=6):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception as error:
            if attempt == attempts - 1:
                raise
            print(f'retry {url}: {error}', flush=True)
            time.sleep(5 * (attempt + 1))


def snapshot_dir(hub, repo_id):
    """Snapshot folder named after the Hugging Face commit (via hf-mirror) so the cache layout matches the hub's."""
    folder = hub / ('models--' + repo_id.replace('/', '--'))
    ref = folder / 'refs' / 'main'
    if ref.exists():
        sha = ref.read_text().strip()
    else:
        try:
            sha = get_json(f'https://hf-mirror.com/api/models/{repo_id}', attempts=3)['sha']
        except Exception:
            sha = 'modelscope-master'
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(sha)
    snap = folder / 'snapshots' / sha
    snap.mkdir(parents=True, exist_ok=True)
    return snap


def file_list(repo_id):
    query = chr(63) + 'Recursive=1'
    info = get_json(f'https://www.modelscope.cn/api/v1/models/{repo_id}/repo/files{query}')
    files = info.get('Data', {}).get('Files', [])
    return [(f['Path'], int(f.get('Size') or 0)) for f in files
            if f.get('Type') != 'tree' and Path(f['Path']).name not in SKIP_NAMES and f['Path'] not in SKIP_NAMES]


def complete(path, size):
    return path.exists() and (size == 0 or path.stat().st_size == size)


def download(repo_id, snap, rel, size):
    dest = snap / rel
    if complete(dest, size):
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    part = dest.with_name(dest.name + '.incomplete')
    have = part.stat().st_size if part.exists() else 0
    url = f'https://www.modelscope.cn/models/{repo_id}/resolve/master/{rel}'
    failures = 0
    while size == 0 or have < size:
        try:
            headers = dict(HEADERS, **({'Range': f'bytes={have}-'} if have else {}))
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                mode = 'ab' if have and response.status == 206 else 'wb'
                if mode == 'wb':
                    have = 0
                with open(part, mode) as out:
                    while True:
                        chunk = response.read(4 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
            if size == 0 or have >= size:
                break
        except Exception as error:
            failures += 1
            if failures > 200:
                raise RuntimeError(f'{repo_id}/{rel}: giving up after {failures} errors ({error})')
            print(f'retry {repo_id}/{rel} at {have / 2**20:.0f}/{size / 2**20:.0f} MiB: {error}', flush=True)
            time.sleep(min(60, 2 * failures))
    if size and have != size:
        raise RuntimeError(f'{repo_id}/{rel}: got {have} bytes, expected {size}')
    part.replace(dest)
    print(f'completed {repo_id}/{rel} ({have / 2**20:.0f} MiB)', flush=True)
    return have


def check_shards(root):
    missing = []
    for index in root.rglob('*.safetensors.index.json'):
        for shard in set(json.loads(index.read_text())['weight_map'].values()):
            if not (index.parent / shard).exists():
                missing.append(str((index.parent / shard).relative_to(root)))
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--check', action='store_true', help='verify only; exit status 1 if anything is missing')
    args = parser.parse_args()
    hub = Path(os.environ.get('HF_HOME', '/root/autodl-tmp/hf_cache')) / 'hub'
    models = Path(os.environ.get('MODELS', '/root/autodl-tmp/models'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from redag.progress import report
    tasks, todo_bytes = [], 0
    for repo_id in args.models:
        snap = snapshot_dir(hub, repo_id)
        link = models / repo_id
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() != snap.resolve():
            link.unlink()
        if not link.exists() and not link.is_symlink():
            link.symlink_to(snap)
        files = file_list(repo_id)
        missing = [(rel, size) for rel, size in files if not complete(snap / rel, size)]
        print(f'{repo_id}: {len(files) - len(missing)}/{len(files)} files complete, '
              f'{sum(s for _, s in missing) / 2**30:.1f} GiB to fetch', flush=True)
        tasks += [(repo_id, snap, rel, size) for rel, size in missing]
        todo_bytes += sum(s for _, s in missing)
    if args.check:
        bad = {m: check_shards(models / m) for m in args.models}
        bad = {m: v for m, v in bad.items() if v}
        print('missing shards:', bad if bad else 'none')
        sys.exit(1 if tasks or bad else 0)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download, *task) for task in tasks]
        for future in as_completed(futures):
            done += future.result()
            report(done / 2**30, todo_bytes / 2**30, 'GiB')
    bad = {m: check_shards(models / m) for m in args.models}
    bad = {m: v for m, v in bad.items() if v}
    if bad:
        raise SystemExit(f'missing shards after download: {bad}')
    print(f'all {len(args.models)} models complete in {models}', flush=True)


if __name__ == '__main__':
    main()
