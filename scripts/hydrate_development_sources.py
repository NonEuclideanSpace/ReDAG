import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentic_image.core.artifacts import ArtifactStore


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/evaluation/development64/source_hydration.jsonl')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ready', default='deployment/development64-assets.ready.json')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    if len(rows) != 64 or len({row['family_id'] for row in rows}) != 64 or {row['evaluation_index'] for row in rows} != set(range(64)):
        raise ValueError('Hydration requires the exact 64 registered development sources')
    store = ArtifactStore('data/artifacts')
    started = time.monotonic()

    def hydrate(row):
        expected = row['expected_ref']
        if expected['metadata']['role'] != 'source' or expected['metadata']['sample_id'] != row['sample_id'] or f"/--/{row['viewer_revision']}/--/" not in row['source_url']:
            raise ValueError('Only the registered public source and fixed viewer revision may be hydrated')
        path = store.root / expected['relative_path']
        try:
            existing = store.get_ref(expected['artifact_id']).model_dump(mode='json')
        except (FileNotFoundError, KeyError):
            existing = None
        if existing == expected and path.is_file() and path.stat().st_size:
            return {'evaluation_index': row['evaluation_index'], 'sample_id': row['sample_id'], 'artifact_id': expected['artifact_id'], 'value_id': expected['value_id'], 'status': 'existing_exact_ref', 'bytes_downloaded': 0}
        for attempt in range(3):
            try:
                with urlopen(Request(row['source_url'], headers={'User-Agent': 'dag-development-hydration/1'}), timeout=45) as response:
                    payload = response.read()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        with Image.open(io.BytesIO(payload)) as image:
            if image.size != (row['width'], row['height']):
                raise ValueError('Official JPEG dimensions differ from the frozen source')
            actual = store.put_image(image, metadata=expected['metadata']).model_dump(mode='json')
        if actual != expected:
            raise ValueError('Hydrated source differs from its exact frozen artifact/value identity')
        return {'evaluation_index': row['evaluation_index'], 'sample_id': row['sample_id'], 'artifact_id': actual['artifact_id'], 'value_id': actual['value_id'], 'status': 'downloaded_exact_identity', 'bytes_downloaded': len(payload)}

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(hydrate, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({'evaluation_index': row['evaluation_index'], 'sample_id': row['sample_id'], 'error_type': type(error).__name__, 'http_status': getattr(error, 'code', None)})
            progress = {'completed': len(completed), 'required': 64, 'failures': len(failures), 'bytes_downloaded': sum(item['bytes_downloaded'] for item in completed), 'seconds': time.monotonic() - started}
            write('deployment/development64-hydration.progress.json', progress)
            print(json.dumps(progress), flush=True)
    report = {**progress, 'status': 'completed' if len(completed) == 64 and not failures else 'failed', 'sources': sorted(completed, key=lambda row: row['evaluation_index']), 'errors': failures, 'private_targets_downloaded': False}
    write('deployment/development64-hydration.report.json', report)
    if failures or len(completed) != 64:
        raise RuntimeError('Development source hydration is incomplete; exact failures were recorded')
    write(args.ready, {'status': 'ready', 'tasks': 64, 'families': 64, 'public_source_identity_verified': True, 'report': 'deployment/development64-hydration.report.json', 'visual_evaluation_labels_ready': False})


if __name__ == '__main__':
    main()
