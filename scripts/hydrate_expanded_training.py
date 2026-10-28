import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from PIL import Image
from agentic_image.core.artifacts import ArtifactStore


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/training_expansion/qualified_hydration.jsonl')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--store', default='data/artifacts')
    parser.add_argument('--report-prefix', default='deployment/expanded-training-hydration')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    families = defaultdict(list)
    for row in rows:
        families[row['family_id']].append(row)
    if not rows or any(len(group) != 2 or {row['role'] for row in group} != {'source', 'private_target'} or len({row['sample_id'] for row in group}) != 1 for group in families.values()):
        raise ValueError('Every independent training family must have exactly its source and private target')
    if len({(row['sample_id'], row['role']) for row in rows}) != len(rows):
        raise ValueError('Duplicate sample/role identity')
    store = ArtifactStore(args.store)
    started = time.monotonic()

    def hydrate(row):
        expected = row['expected_ref']
        if expected['metadata']['role'] != row['role'] or expected['metadata']['sample_id'] != row['sample_id'] or f"/--/{row['viewer_revision']}/--/" not in row['source_url']:
            raise ValueError('Only the exact qualified image and fixed official viewer revision may be hydrated')
        path = store.root / expected['relative_path']
        identity = {key: row[key] for key in ('expansion_index', 'sample_id', 'family_id', 'role')}
        try:
            existing = store.get_ref(expected['artifact_id']).model_dump(mode='json')
        except (FileNotFoundError, KeyError):
            existing = None
        if existing == expected and path.is_file() and path.stat().st_size:
            return {**identity, 'artifact_id': expected['artifact_id'], 'value_id': expected['value_id'], 'status': 'existing_exact_ref', 'bytes_downloaded': 0}
        for attempt in range(3):
            try:
                with urlopen(Request(row['source_url'], headers={'User-Agent': 'dag-training-hydration/1'}), timeout=45) as response:
                    payload = response.read()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        with Image.open(io.BytesIO(payload)) as image:
            if image.size != (row['width'], row['height']):
                raise ValueError('Official JPEG dimensions differ from the frozen pair')
            actual = store.put_image(image, metadata=expected['metadata']).model_dump(mode='json')
        if actual != expected:
            raise ValueError('Hydrated pixels differ from the exact frozen artifact/value identity')
        return {**identity, 'artifact_id': actual['artifact_id'], 'value_id': actual['value_id'], 'status': 'downloaded_exact_identity', 'bytes_downloaded': len(payload)}

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(hydrate, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({'sample_id': row['sample_id'], 'role': row['role'], 'error_type': type(error).__name__, 'http_status': getattr(error, 'code', None)})
            progress = {'completed': len(completed), 'required': len(rows), 'failures': len(failures), 'bytes_downloaded': sum(item['bytes_downloaded'] for item in completed), 'seconds': time.monotonic() - started}
            write(args.report_prefix + '.progress.json', progress)
            print(json.dumps(progress), flush=True)
    report = {**progress, 'status': 'completed' if len(completed) == len(rows) and not failures else 'failed', 'images': sorted(completed, key=lambda row: (row['expansion_index'], row['role'])), 'errors': failures, 'private_targets_for_renderer_loss_only': True}
    write(args.report_prefix + '.report.json', report)
    if failures or len(completed) != len(rows):
        raise RuntimeError('Training image hydration incomplete; exact failures were recorded')
    write(args.report_prefix + '.ready.json', {'status': 'ready', 'families': len(families), 'images': len(rows), 'exact_image_identity_verified': True, 'private_targets_for_renderer_loss_only': True})


if __name__ == '__main__':
    main()
