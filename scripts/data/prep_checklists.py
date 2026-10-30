"""Benchmarks with per-prompt requirements -> unified raw sessions for graph annotation.

  corebench : data/raw/corebench/splits/*.jsonl      T2I-CoReBench, yes/no checklists (kept verbatim)
  lgbench   : data/raw/benchmarks/lgbench/{t2i,i2i}_1000.json   LGBench goal lists (rephrased into yes/no questions)
  longbench : data/raw/benchmarks/longbench_t2i/instruction.jsonl   LongBench-T2I long prompts (+ per-aspect labels)

LGBench I2I base images are not distributed; when data/raw/benchmarks/lgbench/<source_image_path> is missing the
session keeps image=null and canvas_prompt=simple_source_prompt, so build_sets.py queues a FLUX.2-klein render.
--canvas_dir: an extracted T2I-CoReBench-Images model folder; its first sample per prompt becomes the canvas.

Output rows (data/processed/sessions/<name>.jsonl):
  {id, source, kind: t2i|i2i, split, prompt, image|null, canvas_prompt?, checklist?: [{qid, question, tags, fixed}], hints?}
"""
import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from _common import PROC, RAW, log, write_jsonl

COMPOSITION = ('C-MI', 'C-MA', 'C-MR', 'C-TR')
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.webp')
ID_PATTERN = re.compile(r'([CR])[-_]([A-Z]{2})[-_/](\d{1,3})')
BENCH = RAW / 'benchmarks'


def as_question(text):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text if text.endswith('?') else f'Does the image satisfy this requirement: {text.rstrip(".")}?'


def canvas_index(folder):
    index = defaultdict(list)
    for path in sorted(Path(folder).rglob('*')) if folder else []:
        if path.suffix.lower() in IMAGE_EXT:
            match = ID_PATTERN.search(str(path.relative_to(folder)))
            if match:
                index[f'{match.group(1)}-{match.group(2)}-{int(match.group(3)):03d}'].append(str(path.resolve()))
    return index


def corebench(splits, canvases, sample):
    files = sorted((RAW / 'corebench' / 'splits').glob('*.jsonl'))
    if not files:
        raise SystemExit('No CoReBench splits; run fetch_sources.py --what corebench')
    rows = []
    for path in files:
        if splits and path.stem not in splits:
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            items = [{'qid': f'q{i + 1:02d}', 'question': as_question(c['question']), 'tags': list(c.get('tags') or []),
                      'fixed': True} for i, c in enumerate(raw['Checklist'])]
            found = canvases.get(raw['ID'], [])
            rows.append({'id': f"cb:{raw['ID']}", 'source': 'corebench', 'kind': 't2i', 'split': path.stem,
                         'sub_class': raw.get('Sub Class'), 'prompt': raw['Prompt'].strip(),
                         'image': found[min(sample, len(found) - 1)] if found else None,
                         'canvas_origin': 'corebench_images' if found else None, 'checklist': items})
    return rows


def lgbench(folder):
    rows = []
    for name, kind in (('t2i_1000.json', 't2i'), ('i2i_1000.json', 'i2i')):
        path = folder / name
        if not path.exists():
            continue
        for record in json.loads(path.read_text(encoding='utf-8')):
            items = [{'qid': f'q{i + 1:02d}', 'question': as_question(g.get('description', '')), 'tags': [],
                      'goal_type': g.get('type'), 'goal': g.get('description', ''), 'fixed': False}
                     for i, g in enumerate(record.get('goals', [])) if g.get('description')]
            row = {'id': f"lg:{kind}:{record['id']}", 'source': 'lgbench', 'kind': kind,
                   'split': f"{kind}/{record.get('category', '')}", 'prompt': record['prompt'].strip(), 'image': None,
                   'checklist': items}
            if kind == 'i2i':
                local = folder / record.get('source_image_path', '')
                row['image'] = str(local.resolve()) if record.get('source_image_path') and local.is_file() else None
                row['canvas_prompt'] = record.get('simple_source_prompt') or ''
            rows.append(row)
    return rows


def longbench(path):
    rows = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        labels = record.get('label') or []
        if isinstance(labels, str):
            try:
                labels = ast.literal_eval(labels)
            except (ValueError, SyntaxError):
                labels = [labels]
        hints = []
        for item in labels if isinstance(labels, list) else [labels]:
            if isinstance(item, dict):
                hints += [f'{k}: {v}' for k, v in item.items()]
            elif isinstance(item, str):
                hints.append(item)
        rows.append({'id': f'lb:{index:03d}', 'source': 'longbench', 'kind': 't2i', 'split': 'longbench_t2i',
                     'prompt': record['instruction'].strip(), 'image': None, 'hints': hints})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--what', nargs='+', choices=('corebench', 'lgbench', 'longbench'),
                        default=['corebench', 'lgbench', 'longbench'])
    parser.add_argument('--corebench_splits', nargs='*', default=list(COMPOSITION), help='pass nothing to keep all 12')
    parser.add_argument('--canvas_dir', help='extracted T2I-CoReBench-Images model folder used as initial canvases')
    parser.add_argument('--canvas_sample', type=int, default=0)
    parser.add_argument('--lgbench_dir', default=str(BENCH / 'lgbench'))
    parser.add_argument('--longbench_file', default=str(BENCH / 'longbench_t2i' / 'instruction.jsonl'))
    args = parser.parse_args()
    out = PROC / 'sessions'
    if 'corebench' in args.what:
        rows = corebench(set(args.corebench_splits or []), canvas_index(args.canvas_dir), args.canvas_sample)
        write_jsonl(out / 'corebench.jsonl', rows)
        log(f'corebench: {len(rows)} sessions; {dict(Counter((r["split"], bool(r["image"])) for r in rows))}')
    if 'lgbench' in args.what:
        rows = lgbench(Path(args.lgbench_dir))
        write_jsonl(out / 'lgbench.jsonl', rows)
        missing = sum(1 for r in rows if r['kind'] == 'i2i' and not r['image'])
        log(f'lgbench: {len(rows)} sessions ({dict(Counter(r["kind"] for r in rows))}); I2I without base image: {missing}')
    if 'longbench' in args.what and Path(args.longbench_file).exists():
        rows = longbench(Path(args.longbench_file))
        write_jsonl(out / 'longbench.jsonl', rows)
        log(f'longbench: {len(rows)} sessions')


if __name__ == '__main__':
    main()
