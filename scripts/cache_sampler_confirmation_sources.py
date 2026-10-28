from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import textwrap
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont

from agentic_image.core.artifacts import ArtifactStore


root = Path('data/evaluation/sampler_confirm64')
private = Path('data/private_eval_v1/sampler_confirm64')
private.mkdir(parents=True, exist_ok=True)
(private / 'source_contacts').mkdir(exist_ok=True)
rows = [json.loads(line) for line in (root / 'reserved.frozen.jsonl').read_text().splitlines()]
inventory = [json.loads(line) for line in (root / 'source_inventory.jsonl').read_text().splitlines()]
store = ArtifactStore('data/artifacts')
by_sample = {row['sample_id']: row for row in rows}
api = {}
pages = set()
repositories = {'magicbrush': 'osunlp/MagicBrush', 'omniedit': 'TIGER-Lab/OmniEdit-Filtered-1.2M'}
started = time.monotonic()
for source in repositories:
    for path in Path('data/viewer/api').glob(f'{source}_dev_*.json'):
        for item in json.loads(path.read_text())['rows']:
            row = item['row']
            sample = f"magicbrush:{row['img_id']}:{row['turn_index']}" if source == 'magicbrush' else 'omniedit:' + row['omni_edit_id']
            if sample in by_sample:
                api[sample] = item
for record in inventory:
    if not Path(record['source_artifact_path']).is_file():
        pages.add((record['source_dataset'], api[record['sample_id']]['row_idx'] // 100 * 100))
for source, offset in sorted(pages):
    url = 'https://datasets-server.huggingface.co/rows?' + urlencode({'dataset': repositories[source], 'config': 'default', 'split': 'dev', 'offset': offset, 'length': 100})
    with urlopen(url, timeout=20) as response:
        payload = json.load(response)
    for item in payload['rows']:
        row = item['row']
        sample = f"magicbrush:{row['img_id']}:{row['turn_index']}" if source == 'magicbrush' else 'omniedit:' + row['omni_edit_id']
        if sample in by_sample:
            api[sample] = item
    print(json.dumps({'refreshed_page': [source, offset], 'seconds': time.monotonic() - started}), flush=True)


def prepare(record):
    saved = record['image']
    path = Path(record['source_artifact_path'])
    downloaded = 0
    if not path.is_file():
        item = api[record['sample_id']]
        row = item['row']
        request = row['instruction'] if record['source_dataset'] == 'magicbrush' else max((text for text in row['edited_prompt_list'] if isinstance(text, str) and text.strip()), key=len)
        if request != record['request'] or item.get('truncated_cells'):
            raise ValueError('Official development request identity changed')
        image = row['source_img' if record['source_dataset'] == 'magicbrush' else 'src_img']
        if f"/--/{saved['viewer_revision']}/--/" not in image['src']:
            raise ValueError('Official source revision changed')
        with urlopen(image['src'], timeout=20) as response:
            payload = response.read()
        downloaded = len(payload)
        with Image.open(io.BytesIO(payload)) as decoded:
            if decoded.size != (image['width'], image['height']):
                raise ValueError('Official source dimensions changed')
            ref = store.put_image(decoded, metadata={'dataset_profile': 'official-viewer-reencoded-v1',
                'original_artifact_id': saved['original_artifact_id'], 'sample_id': saved['sample_id'],
                'role': 'source', 'viewer_revision': saved['viewer_revision'],
                'original_revision': by_sample[record['sample_id']]['license_record']['revision']})
        if ref.value_id != saved['value_id'] or ref.artifact_id != saved['artifact_id']:
            raise ValueError('Official decoded pixels differ from the existing registered source')
    ref = store.get_ref(saved['artifact_id'])
    return {'index': record['evaluation_index'], 'task_id': record['task_id'], 'family_id': record['family_id'],
        'split_id': 'official-dev', 'request': record['request'], 'source_image_id': ref.artifact_id,
        'source_value_id': ref.value_id, 'source_path': str(path.resolve()), 'source_hw': list(ref.shape[:2]),
        'source_frame': ref.frame.model_dump(mode='json') if ref.frame else None,
        'input_scope': 'original_public_request_and_source_only'}, downloaded


prepared, failures, total_bytes = [], [], 0
with ThreadPoolExecutor(max_workers=4) as pool:
    pending = {pool.submit(prepare, record): record for record in inventory}
    for future in as_completed(pending):
        record = pending[future]
        try:
            result, size = future.result()
            prepared.append(result)
            total_bytes += size
        except Exception as error:
            failures.append({'index': record['evaluation_index'], 'error': str(error)})
        if len(prepared) % 8 == 0 or failures:
            print(json.dumps({'source_images': len(prepared), 'download_bytes': total_bytes, 'failures': failures,
                'seconds': time.monotonic() - started}), flush=True)
prepared.sort(key=lambda row: row['index'])
(private / 'sources.frozen.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in prepared))
report = {'actual_sources': len(prepared), 'required': 64, 'download_bytes': total_bytes,
    'seconds': time.monotonic() - started, 'failures': failures, 'private_targets_opened': 0,
    'review_labels_created': 0, 'human_reviewed': False}
(root / 'local_sources.report.json').write_text(json.dumps(report, indent=2) + '\n')
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 16)
small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
by_index = {row['index']: row for row in prepared}
for start in range(0, 64, 4):
    if any(index not in by_index for index in range(start, start + 4)):
        continue
    sheet = Image.new('RGB', (1040, 1200), 'white')
    draw = ImageDraw.Draw(sheet)
    for offset in range(4):
        row = by_index[start + offset]
        x, y = offset % 2 * 520, offset // 2 * 600
        draw.text((x + 8, y + 6), f"confirmation {row['index']:03d} | {row['source_hw']}", fill='black', font=font)
        draw.multiline_text((x + 8, y + 32), '\n'.join(textwrap.wrap(row['request'], width=64)), fill='black', font=small, spacing=3)
        with Image.open(row['source_path']) as image:
            image = image.convert('RGB')
            image.thumbnail((500, 470))
            sheet.paste(image, (x + 8, y + 120))
    sheet.save(private / 'source_contacts' / f'{start:03d}-{start+3:03d}.jpg', quality=93)
print(json.dumps(report), flush=True)
if failures:
    raise SystemExit(2)
