"""Merge run_eval_benchmarks.py parts (--shard / --stages) into one result file.

Checks that the parts belong to the same experiment and budgets, re-applies the K-tier session counts
that single-process runs enforce, and for experiment 3 draws the 200 blind pairs exactly as
run_eval_benchmarks.py does (same seed, same canonical stage order).  The merged file can then be
annotated with run_eval_benchmarks.py --resume_results <merged> --annotations/--human_ratings.
  python scripts/merge_eval_results.py --inputs runs/eval/exp3_*.json --output runs/eval/exp3.json
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

STAGES = ('zero_shot', 'sft', 'ctrl_rl', 'edit_rl', 'alternating_rl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-small-dataset', action='store_true')
    args = parser.parse_args()
    parts = [json.loads(Path(p).read_text()) for p in args.inputs]
    head = parts[0]
    if any(p['exp_id'] != head['exp_id'] or p['budgets'] != head['budgets'] for p in parts):
        raise ValueError('All parts must come from the same experiment and budgets')
    merged = {k: v for k, v in head.items() if k not in ('evaluation_cases', 'part', 'final_views', 'descriptions')}
    merged['evaluation_cases'] = [case for p in parts for case in p['evaluation_cases']]
    merged['merged_from'] = [str(Path(p).resolve()) for p in args.inputs]
    ids = Counter(case['id'] for case in merged['evaluation_cases'])
    if any(n > 1 for n in ids.values()):
        raise ValueError('Duplicate case ids: the same shard/stage was merged twice')
    tiers = defaultdict(Counter)
    for case in merged['evaluation_cases']:
        stage, mode, k = case['group'].rsplit('/', 2)
        tiers[f'{stage}/{mode}'][int(k)] += 1
    need = 300 if head['exp_id'] == 3 else 100
    short = {group: dict(c) for group, c in tiers.items() if any(c[k] < need for k in (10, 20, 30))}
    if short and not args.allow_small_dataset and head['exp_id'] != 2:
        raise ValueError(f'Incomplete K tiers after merging (need {need} per tier): {short}')
    merged['tier_counts'] = {group: dict(c) for group, c in tiers.items()}
    if head['exp_id'] == 3:
        saved, descriptions = defaultdict(dict), {}
        for part in parts:
            for key, images in part.get('final_views', {}).items():
                saved[key].update(images)
            descriptions.update(part.get('descriptions', {}))
        complete = {key: images for key, images in saved.items() if set(images) == set(STAGES)}
        folder = Path(args.output).parent / (Path(args.output).stem + '_images')
        folder.mkdir(parents=True, exist_ok=True)
        rng, pairs, key_rows = random.Random(head.get('part', {}).get('seed', 42)), [], []
        ordered = sorted(complete.items(), key=lambda item: tuple(int(x) for x in item[0].split(':')))
        for identity, images in rng.sample(ordered, min(200, len(ordered))):
            names = rng.sample(list(STAGES), 2)
            pair = dict(descriptions[identity], id=len(pairs))
            for side, name in zip(('a', 'b'), names):
                path = folder / f'blind_{len(pairs)}_{side}.png'
                shutil.copyfile(images[name], path)
                pair[side] = str(path.resolve())
            pairs.append(pair)
            key_rows.append({'id': len(key_rows), 'session_seed': identity, 'a': names[0], 'b': names[1]})
        merged['blind_pairs'] = pairs
        Path(args.output).with_suffix('.blind_pairs.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2))
        Path(args.output).with_suffix('.blind_key.json').write_text(json.dumps(key_rows, ensure_ascii=False, indent=2))
        merged['incomplete_stage_sessions'] = len(saved) - len(complete)
    Path(args.output).write_text(json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'cases': len(merged['evaluation_cases']), 'tiers': merged['tier_counts'],
                      'blind_pairs': len(merged.get('blind_pairs', []))}))


if __name__ == '__main__':
    main()
