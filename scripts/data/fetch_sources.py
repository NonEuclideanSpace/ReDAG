"""Download raw benchmark sources into data/raw (resumable, low memory).

  MagicBrush      osunlp/MagicBrush (parquet; train ~23.9 GiB, dev ~1.4 GiB, CC-BY-4.0)
  T2I-CoReBench   lioooox/T2I-CoReBench (12 x 90 prompts with yes/no checklists, Apache-2.0)
  LGBench         TruemanV5/LGBench (t2i_1000.json, i2i_1000.json, 1000 I2I source images ~1.35 GiB)
  LongBench-T2I   YCZhou/LongBench-T2I (instruction.jsonl)
  Files that already exist with the expected size are kept (e.g. copies uploaded by hand).

Examples
  python scripts/data/fetch_sources.py --what corebench
  python scripts/data/fetch_sources.py --what magicbrush --splits dev
  python scripts/data/fetch_sources.py --what magicbrush --splits train --wait_pid 3206   # start after another job ends
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _common import RAW, log

# env.sh puts every job in offline mode (weights are local); downloading is this script's whole purpose
for _flag in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ.pop(_flag, None)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

SOURCES = {
    'magicbrush': ('osunlp/MagicBrush', 'data'),
    'corebench': ('lioooox/T2I-CoReBench', 'splits'),
    # 4 images per CoReBench prompt from one T2I model; used as zero-shot initial canvases (no GPU rendering)
    'corebench_images': ('lioooox/T2I-CoReBench-Images', ''),
    'lgbench': ('TruemanV5/LGBench', ''),
    'longbench': ('YCZhou/LongBench-T2I', ''),
}
TARGETS = {'lgbench': RAW / 'benchmarks' / 'lgbench', 'longbench': RAW / 'benchmarks' / 'longbench_t2i'}
KEEP = {'lgbench': lambda f: f.endswith('_1000.json') or f.startswith('i2i_source_images/'),
        'longbench': lambda f: f == 'instruction.jsonl'}


def link_source_folders(target):
    """i2i_1000.json names its images under a dated folder (e.g. i2i_source_images_20251104_153239/...); the repository
    stores them under i2i_source_images/, so the dated name becomes a relative symlink."""
    index = target / 'i2i_1000.json'
    if not index.exists():
        return
    records = json.loads(index.read_text())
    records = records if isinstance(records, list) else list(records.values())
    for folder in {str(r.get('source_image_path', '')).split('/')[0] for r in records if isinstance(r, dict)} - {'', 'i2i_source_images'}:
        link = target / folder
        if not link.exists() and not link.is_symlink():
            link.symlink_to('i2i_source_images')
            log(f'linked {folder} -> i2i_source_images')


def extract_archive(path, target):
    import tarfile
    marker = target / f'.{path.name}.extracted'
    if marker.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'r|gz') as archive:  # streaming: constant memory
        archive.extractall(target, filter='data')
    marker.touch()
    log(f'extracted {path.name} -> {target}')


def wait_for_pid(pid, poll=60):
    while Path(f'/proc/{pid}').exists():
        log(f'waiting for PID {pid} to finish before downloading ...')
        time.sleep(poll)


def listing(repo_id, folder):
    from huggingface_hub import HfApi
    for attempt in range(10):
        try:
            entries = HfApi().list_repo_tree(repo_id, path_in_repo=folder, repo_type='dataset', recursive=True)
            return [(e.path, getattr(e, 'size', None)) for e in entries if getattr(e, 'size', None) is not None]
        except Exception as error:  # network hiccups through the mirror are common
            log(f'listing {repo_id} failed ({error!r}); retry {attempt + 1}/10')
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'Could not list {repo_id}')


def fetch(repo_id, filename, size, target):
    from huggingface_hub import hf_hub_download
    path = target / filename
    if path.exists() and (size is None or path.stat().st_size == size):
        return filename, 'cached'
    for attempt in range(30):
        try:
            hf_hub_download(repo_id, filename, repo_type='dataset', local_dir=target)
            if size is None or path.stat().st_size == size:
                return filename, 'ok'
            log(f'{filename}: size mismatch, retrying')
        except Exception as error:
            log(f'{filename}: {type(error).__name__}; retry {attempt + 1}/30')
        time.sleep(min(120, 5 * (attempt + 1)))
    return filename, 'failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--what', nargs='+', choices=sorted(SOURCES), default=['corebench'])
    parser.add_argument('--splits', nargs='+', default=['dev'], help='MagicBrush splits: dev and/or train')
    parser.add_argument('--max_files', type=int, help='only the first N matching files per split (smoke tests)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--wait_pid', type=int)
    parser.add_argument('--archives', nargs='+', default=['FLUX.2-klein-9B'], help='corebench_images model archives')
    args = parser.parse_args()
    if args.wait_pid:
        wait_for_pid(args.wait_pid)
    report = {}
    for name in args.what:
        repo_id, folder = SOURCES[name]
        target = TARGETS.get(name, RAW / name)
        target.mkdir(parents=True, exist_ok=True)
        if name == 'corebench_images':
            files = [(f'{model}.tar.gz', None) for model in args.archives]
        elif name == 'lgbench':  # fixed layout; a recursive listing of 3,000 files pages through huggingface.co itself
            files = [('t2i_1000.json', None), ('i2i_1000.json', None)] + [
                (f'i2i_source_images/task_{i:04d}/{f}', None) for i in range(1000) for f in ('source_image.png', 'metadata.json', 'source_prompt.txt')]
        else:
            files = listing(repo_id, folder)
        if name in KEEP:
            files = [f for f in files if KEEP[name](f[0])]
        if name == 'magicbrush':
            picked = []
            for split in args.splits:
                chosen = sorted(f for f in files if Path(f[0]).name.startswith(split + '-'))
                picked += chosen[:args.max_files] if args.max_files else chosen
            files = picked
        log(f'{name}: {len(files)} files, {sum(s or 0 for _, s in files) / 2**30:.2f} GiB from {repo_id}')
        results = {}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(fetch, repo_id, f, s, target) for f, s in files]
            for future in as_completed(futures):
                filename, status = future.result()
                results[filename] = status
                log(f'{name}: {filename} -> {status}')
        report[name] = {'repo_id': repo_id, 'files': results, 'endpoint': os.environ['HF_ENDPOINT']}
        (target / 'fetch_manifest.json').write_text(json.dumps(report[name], indent=2))
        if name == 'corebench_images':
            for filename, status in results.items():
                if status in ('ok', 'cached'):
                    extract_archive(target / filename, target / filename.replace('.tar.gz', ''))
        if name == 'lgbench':
            link_source_folders(target)
    failed = [f for r in report.values() for f, s in r['files'].items() if s == 'failed']
    log('done' if not failed else f'{len(failed)} files failed; rerun the same command to resume')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
