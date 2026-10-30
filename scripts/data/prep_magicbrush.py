"""MagicBrush parquet -> aligned turn images, binary edit masks, editor-SFT crops and I2I sessions.

Phase 1 (extract) streams one image column at a time out of each parquet file, so the peak
memory is one column chunk (~200 MB) instead of a whole shard; it only writes the original
bytes to disk.  Phase 2 (build) works turn by turn:
  * aligns source to target size (identity / resize / center-crop+resize, lowest error wins),
  * decides the editable mask from mask_img (alpha / white / black convention, whichever best
    matches the source->target difference) and falls back to a dilated difference mask,
  * writes full-size aligned images + a binary mask (255 = editable),
  * crops a window around the edit (mask box x1.25, >= 25% side) for editor SFT (eq. 21-24 use
    crop windows I_src, I_tgt and mask M).
Outputs
  data/processed/magicbrush/turns_<split>.jsonl
  data/processed/sessions/magicbrush_<split>.jsonl          (I2I sessions for graph annotation)
  data/final/editor_sft_<split>.jsonl                        (input of scripts/run_sft_editor.py)
"""
import argparse
import io
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from _common import FINAL, PROC, RAW, bbox_of, expand_box, log, relpath, write_jsonl

ROLES = (('source_img', 'src'), ('target_img', 'tgt'), ('mask_img', 'mask'))
OUT = PROC / 'magicbrush'


def image_bytes(value):
    if isinstance(value, dict):
        return value.get('bytes')
    return value


def extract(split):
    """Phase 1: parquet -> raw image files + turns_raw_<split>.jsonl (idempotent per shard)."""
    import pyarrow.parquet as pq
    raw_dir = OUT / 'raw_images' / split
    raw_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted((RAW / 'magicbrush' / 'data').glob(f'{split}-*.parquet'))
    if not shards:
        raise SystemExit(f'No {split} parquet under {RAW / "magicbrush" / "data"}; run fetch_sources.py first')
    meta_path = OUT / f'turns_raw_{split}.jsonl'
    for shard in shards:
        marker = raw_dir / f'.{shard.name}.done'
        if marker.exists():
            continue
        parquet = pq.ParquetFile(shard)
        rows = []
        for batch in parquet.iter_batches(batch_size=16, columns=['img_id', 'turn_index', 'instruction']):
            for img_id, turn, text in zip(*(batch.column(i).to_pylist() for i in range(3))):
                rows.append({'img_id': str(img_id), 'turn': int(turn), 'instruction': str(text).strip(), 'shard': shard.name})
        for column, role in ROLES:
            for batch in parquet.iter_batches(batch_size=4, columns=['img_id', 'turn_index', column]):
                ids, turns, images = (batch.column(i).to_pylist() for i in range(3))
                for img_id, turn, value in zip(ids, turns, images):
                    data = image_bytes(value)
                    if not data:
                        continue
                    ext = 'png' if data[:4] == b'\x89PNG' else 'jpg'
                    (raw_dir / f'{img_id}_{turn}_{role}.{ext}').write_bytes(data)
                del batch, images
        with meta_path.open('a', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        marker.touch()
        log(f'{split}: extracted {shard.name} ({parquet.metadata.num_rows} rows, {parquet.metadata.num_row_groups} row groups)')
    return meta_path


def load_raw(raw_dir, img_id, turn, role):
    for ext in ('png', 'jpg'):
        path = raw_dir / f'{img_id}_{turn}_{role}.{ext}'
        if path.exists():
            with Image.open(path) as image:
                image.load()
                return image.copy()
    return None


def to_array(image):
    return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0


def align_source(source, target):
    """Return (aligned_source, rule, mean_abs_error) mapping source onto the target canvas."""
    if source.size == target.size:
        return source.convert('RGB'), 'identity', float(np.abs(to_array(source) - to_array(target)).mean())
    tgt = to_array(target)
    candidates = [('resize', source.convert('RGB').resize(target.size, Image.BICUBIC))]
    w, h = source.size
    side = min(w, h)
    if abs(target.size[0] - target.size[1]) <= 1 and w != h:
        box = ((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)
        candidates.append(('center_crop', source.convert('RGB').crop(box).resize(target.size, Image.BICUBIC)))
    scored = [(float(np.abs(to_array(img) - tgt).mean()), rule, img) for rule, img in candidates]
    error, rule, image = min(scored, key=lambda item: item[0])
    return image, rule, error


def decide_mask(mask_image, diff):
    """Binary editable mask (H, W) on the target grid and the rule that produced it."""
    h, w = diff.shape
    candidates = {}
    if mask_image is not None:
        mask_image = mask_image.resize((w, h), Image.NEAREST)
        if mask_image.mode in ('RGBA', 'LA') or (mask_image.mode == 'P' and 'transparency' in mask_image.info):
            alpha = np.asarray(mask_image.convert('RGBA'))[..., 3]
            candidates['alpha'] = alpha < 128
        luma = np.asarray(mask_image.convert('L'))
        candidates['white'] = luma > 200
        candidates['black'] = luma < 55
    best, best_score = None, -1.0
    for rule, region in candidates.items():
        fraction = float(region.mean())
        if not 0.002 < fraction < 0.95:
            continue
        score = float(diff[region].mean() - diff[~region].mean())
        if score > best_score:
            best, best_score = rule, score
    if best is not None and best_score > 0.01:
        return candidates[best], best
    changed = Image.fromarray(((diff > 0.08) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(15))
    region = np.asarray(changed) > 127
    if 0.002 < float(region.mean()) < 0.95:
        return region, 'difference'
    return None, 'none'


def build_turn(task):
    split, row, previous_target, size = task
    raw_dir = OUT / 'raw_images' / split
    img_dir = OUT / 'images' / split
    crop_dir = PROC / 'editor_sft' / split
    img_id, turn = row['img_id'], row['turn']
    key = f'{img_id}_{turn}'
    target = load_raw(raw_dir, img_id, turn, 'tgt')
    source = load_raw(raw_dir, img_id, turn, 'src')
    if target is None or source is None:
        return None, 'missing_image', img_id
    target = target.convert('RGB')
    if size and max(target.size) > size:  # MagicBrush targets are 1024 px; the content is ~512 px COCO
        scale = size / max(target.size)
        target = target.resize((round(target.size[0] * scale), round(target.size[1] * scale)), Image.BICUBIC)
    if previous_target is not None and Path(previous_target).exists():
        with Image.open(previous_target) as chained:  # later turns start from the previous (aligned) target
            source = chained.convert('RGB').copy()
    source, rule, error = align_source(source, target)
    diff = np.abs(to_array(source) - to_array(target)).mean(axis=2)
    region, mask_rule = decide_mask(load_raw(raw_dir, img_id, turn, 'mask'), diff)
    if region is None:
        return None, 'no_mask', img_id
    outside = float(diff[~region].mean()) if (~region).any() else 0.0
    if outside > 0.12:
        return None, 'misaligned', img_id
    img_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: img_dir / f'{key}_{name}.png' for name in ('src', 'tgt', 'mask')}
    mask = Image.fromarray((region * 255).astype(np.uint8), 'L')
    source.save(paths['src'], compress_level=1)
    target.save(paths['tgt'], compress_level=1)
    mask.save(paths['mask'], compress_level=1)
    box = bbox_of(region)
    window = expand_box(box, 1.25, 0.25)
    w, h = target.size
    pixel = (int(window[0] * w), int(window[1] * h), max(int(window[0] * w) + 16, int(round(window[2] * w))),
             max(int(window[1] * h) + 16, int(round(window[3] * h))))
    crops = {}
    for name, image in (('src', source), ('tgt', target), ('mask', mask)):
        crops[name] = crop_dir / f'{key}_{name}.png'
        image.crop(pixel).save(crops[name], compress_level=1)
    record = {'id': f'mb:{split}:{key}', 'img_id': img_id, 'turn': turn, 'split': split, 'instruction': row['instruction'],
              'source': str(paths['src']), 'target': str(paths['tgt']), 'mask': str(paths['mask']), 'size': [w, h],
              'edit_box': [round(v, 6) for v in box], 'window': window, 'mask_rule': mask_rule, 'align': rule,
              'align_err': round(error, 4), 'outside_err': round(outside, 4), 'mask_fraction': round(float(region.mean()), 4),
              'crops': {k: str(v) for k, v in crops.items()}, 'crop_size': [pixel[2] - pixel[0], pixel[3] - pixel[1]]}
    return record, 'ok', img_id


def build(split, workers, size):
    meta = OUT / f'turns_raw_{split}.jsonl'
    rows = {}
    for line in meta.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row['img_id'], row['turn'])] = row
    sessions = defaultdict(list)
    for (img_id, turn), row in sorted(rows.items(), key=lambda item: (item[0][0], item[0][1])):
        sessions[img_id].append(row)
    records, status = [], Counter()
    # turns of a session must run in order (turn t>1 starts from the aligned target of turn t-1)
    max_turns = max(len(v) for v in sessions.values())
    previous = {}
    broken = set()  # sessions whose chain is interrupted: later turns would start from the wrong image
    with Pool(max(1, workers)) as pool:
        for position in range(max_turns):
            tasks = [(split, turns[position], previous.get(img_id), size) for img_id, turns in sessions.items()
                     if position < len(turns) and img_id not in broken]
            for result, state, img_id in pool.imap(build_turn, tasks, chunksize=4):
                status[state] += 1
                if result:
                    records.append(result)
                    previous[img_id] = result['target']
                else:
                    broken.add(img_id)
            log(f'{split}: turn position {position + 1}/{max_turns} done, status so far {dict(status)}')
    records.sort(key=lambda r: (r['img_id'], r['turn']))
    write_jsonl(OUT / f'turns_{split}.jsonl', records)
    by_session = defaultdict(list)
    for record in records:
        by_session[record['img_id']].append(record)
    session_rows = []
    for img_id, turns in sorted(by_session.items()):
        turns = sorted(turns, key=lambda t: t['turn'])
        turns = [t for i, t in enumerate(turns) if t['turn'] == i + 1]  # gap-free prefix 1..T
        if not turns:
            continue
        session_rows.append({'id': f'mb:{split}:{img_id}', 'source': 'magicbrush', 'kind': 'i2i', 'split': split,
                             'image': turns[0]['source'], 'turns': [{'turn': t['turn'], 'instruction': t['instruction'],
                             'edit_box': t['edit_box'], 'target': t['target']} for t in turns]})
    write_jsonl(PROC / 'sessions' / f'magicbrush_{split}.jsonl', session_rows)
    final_rows = [{'id': r['id'], 'source': relpath(r['crops']['src'], FINAL), 'target': relpath(r['crops']['tgt'], FINAL),
                   'mask': relpath(r['crops']['mask'], FINAL), 'instruction': r['instruction'], 'origin': 'magicbrush',
                   'split': split} for r in records]
    FINAL.mkdir(parents=True, exist_ok=True)
    write_jsonl(FINAL / f'editor_sft_{split}.jsonl', final_rows)
    rules = Counter(r['mask_rule'] for r in records)
    aligns = Counter(r['align'] for r in records)
    log(f'{split}: {len(records)} turns, {len(session_rows)} sessions; status {dict(status)}; mask rules {dict(rules)}; align {dict(aligns)}')
    return {'split': split, 'turns': len(records), 'sessions': len(session_rows), 'status': dict(status),
            'mask_rules': dict(rules), 'align': dict(aligns)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=['dev'])
    parser.add_argument('--phase', choices=('extract', 'build', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=1, help='processes for phase 2 (use 1 in no-GPU mode)')
    parser.add_argument('--size', type=int, default=512, help='long side of the stored canvases (0 keeps the original)')
    args = parser.parse_args()
    summary = []
    for split in args.splits:
        if args.phase in ('extract', 'all'):
            extract(split)
        if args.phase in ('build', 'all'):
            summary.append(build(split, args.workers, args.size))
    if summary:
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
