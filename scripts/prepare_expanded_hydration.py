import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlencode
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
import time

from agentic_image.core.artifacts import ArtifactStore


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/training_expansion')
    parser.add_argument('--output', default='data/training_expansion/qualified_hydration.jsonl')
    parser.add_argument('--selection', default='data/training_expansion/training_selection_v1.frozen.jsonl')
    args = parser.parse_args()
    root = Path(args.root)
    selected = {row['sample_id']: row for row in read(args.selection)}
    store = ArtifactStore('data/artifacts')
    prepared, viewers = {}, {}
    for folder in sorted(root.glob('candidates*')):
        if not (folder / 'prepared.jsonl').is_file():
            continue
        prepared.update({row['sample_id']: row for row in read(folder / 'prepared.jsonl')})
        for page in (folder / 'api').glob('rows_*.json'):
            for item in json.loads(page.read_text())['rows']:
                row = item['row']
                if 'img_id' in row:
                    sample = f"magicbrush:{row['img_id']}:{row['turn_index']}"
                    columns = ('source_img', 'target_img')
                else:
                    sample = 'omniedit:' + row['omni_edit_id']
                    columns = ('src_img', 'edited_img')
                if sample not in selected:
                    continue
                expiry = int(parse_qs(urlparse(row[columns[0]]['src']).query)['Expires'][0])
                if sample not in viewers or viewers[sample][0] < expiry:
                    viewers[sample] = (expiry, row, columns, item['row_idx'])
    def refresh(sample):
        pair = prepared[sample]
        dataset = 'osunlp/MagicBrush' if pair['source_dataset'] == 'magicbrush' else 'TIGER-Lab/OmniEdit-Filtered-1.2M'
        url = 'https://datasets-server.huggingface.co/rows?' + urlencode({'dataset': dataset, 'config': 'default', 'split': 'train', 'offset': pair['verified_viewer_row_index'], 'length': 1})
        with urlopen(Request(url, headers={'User-Agent': 'dag-training-hydration/1', 'Cache-Control': 'no-cache'}), timeout=40) as response:
            entry = json.load(response)['rows'][0]
        row = entry['row']
        actual_sample = f"magicbrush:{row['img_id']}:{row['turn_index']}" if pair['source_dataset'] == 'magicbrush' else 'omniedit:' + row['omni_edit_id']
        if actual_sample != sample or entry.get('truncated_cells'):
            raise ValueError('Refreshed official row identity mismatch')
        columns = viewers[sample][2]
        expiry = int(parse_qs(urlparse(row[columns[0]]['src']).query)['Expires'][0])
        if expiry < time.time() + 300:
            raise ValueError('Refreshed official URL is already expired')
        return sample, (expiry, row, columns, entry['row_idx'])
    expiring = [sample for sample, entry in viewers.items() if entry[0] < time.time() + 900]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for sample, refreshed in executor.map(refresh, expiring):
            viewers[sample] = refreshed
    output, payload_bytes = [], 0
    for sample, selection in sorted(selected.items(), key=lambda item: item[1]['expansion_index']):
        pair = prepared[sample]
        _, viewer, columns, index = viewers[sample]
        instruction = viewer['instruction'] if pair['source_dataset'] == 'magicbrush' else max(viewer['edited_prompt_list'], key=len)
        if instruction != pair['request'] or index != pair['verified_viewer_row_index']:
            raise ValueError('Official viewer sample/request identity differs from the frozen pair')
        for image, column, role in zip(pair['images'], columns, ('source', 'private_target')):
            expected = store.get_ref(image['artifact_id']).model_dump(mode='json')
            if expected['metadata']['role'] != role or expected['metadata']['sample_id'] != sample:
                raise ValueError('Artifact role/sample identity differs from the qualified pair')
            if (image['width'], image['height']) != (viewer[column]['width'], viewer[column]['height']):
                raise ValueError('Viewer dimensions changed')
            payload_bytes += image['bytes_downloaded']
            output.append({**selection, 'role': role, 'request': pair['request'],
                           'viewer_revision': image['viewer_revision'], 'viewer_row_index': index,
                           'source_url': viewer[column]['src'], 'width': image['width'], 'height': image['height'],
                           'expected_ref': expected})
    target = Path(args.output)
    target.write_text(''.join(json.dumps(row) + '\n' for row in output))
    status = {'families': len(selected), 'images': len(output), 'expected_download_bytes': payload_bytes,
              'manifest_bytes': target.stat().st_size, 'private_targets_for_renderer_loss_only': True,
              'human_reviewed': False, 'signed_url_minimum_expiry': min(item[0] for item in viewers.values())}
    target.with_suffix('.report.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
