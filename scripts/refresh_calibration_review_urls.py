import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlparse, parse_qs
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--batch', required=True)
arguments = parser.parse_args()
path = Path(arguments.batch)
rows = list(map(json.loads, path.read_text().splitlines()))
pages = defaultdict(list)
for row in rows:
    pages[row['source_dataset'], row['verified_viewer_row_index'] // 100 * 100].append(row)
repositories = {'magicbrush': 'osunlp/MagicBrush', 'omniedit': 'TIGER-Lab/OmniEdit-Filtered-1.2M'}
started = time.monotonic()


def refresh(key, selected):
    source, offset = key
    url = 'https://datasets-server.huggingface.co/rows?' + urlencode({'dataset': repositories[source], 'config': 'default', 'split': 'train', 'offset': offset, 'length': 100})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                payload = json.load(response)
            found = {entry['row_idx']: entry for entry in payload['rows']}
            output = []
            for saved in selected:
                entry = found[saved['verified_viewer_row_index']]
                item = entry['row']
                sample = f"magicbrush:{item['img_id']}:{item['turn_index']}" if source == 'magicbrush' else 'omniedit:' + item['omni_edit_id']
                request = item['instruction'] if source == 'magicbrush' else max((x for x in item['edited_prompt_list'] if isinstance(x, str) and x.strip()), key=len)
                if sample != saved['sample_id'] or request != saved['request'] or entry.get('truncated_cells'):
                    raise ValueError('Refreshed viewer identity/request differs')
                columns = {'source': 'source_img', 'target': 'target_img'} if source == 'magicbrush' else {'source': 'src_img', 'target': 'edited_img'}
                for image in saved['images']:
                    fresh = item[columns[image['candidate_role']]]['src']
                    if '/--/' + image['viewer_revision'] + '/--/' not in fresh:
                        raise ValueError('Refreshed viewer revision differs')
                    if int(parse_qs(urlparse(fresh).query)['Expires'][0]) < time.time() + 300:
                        raise ValueError('Refreshed image signature is already stale')
                    image['review_image_url'] = fresh
                output.append(saved)
            return output
        except Exception:
            if attempt:
                raise
            time.sleep(1)


updated, failures = {}, []
with ThreadPoolExecutor(max_workers=4) as executor:
    jobs = {executor.submit(refresh, key, group): key for key, group in pages.items()}
    for future in as_completed(jobs):
        try:
            for row in future.result():
                updated[row['calibration_index']] = row
        except Exception as error:
            failures.append({'page': jobs[future], 'error': str(error)})
        print(json.dumps({'updated': len(updated), 'failures': failures, 'seconds': time.monotonic()-started}), flush=True)
if failures or len(updated) != len(rows):
    raise RuntimeError('URL refresh incomplete; original batch retained')
path.with_suffix('.expired_urls.jsonl').write_text(path.read_text())
path.write_text(''.join(json.dumps(updated[row['calibration_index']], ensure_ascii=False)+'\n' for row in rows))
print(json.dumps({'complete': True, 'families': len(rows), 'seconds': time.monotonic()-started}), flush=True)
