"""Choose which raw sessions are sent to annotate_graphs.py (bounds its GPU cost to what the splits need).

Evaluation needs 3 x 100 sessions with exact K, training needs ~1,150 failure states (1,000 rejection-
sampling states + RL train/dev) plus planner pairs.  Annotating every benchmark prompt (~8k sessions)
would cost ~15 GPU-h; these quotas (~2k sessions, about 2x what is consumed) keep it near 5 GPU-h.
Pools come from build_sets.pool_of, so the held-out evaluation pool never leaks into training.
"""
import argparse
from collections import Counter
from pathlib import Path

from _common import PROC, iter_jsonl, log, stable_unit
from build_sets import pool_of

QUOTA = {('eval', 'magicbrush'): 200, ('eval', 'corebench'): 400, ('eval', 'lgbench'): 160, ('eval', 'longbench'): 120,
         ('train', 'magicbrush'): 800, ('train', 'corebench'): 400, ('train', 'lgbench'): 300, ('train', 'longbench'): 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', nargs='+', default=sorted(str(p) for p in (PROC / 'sessions').glob('*.jsonl')))
    parser.add_argument('--eval_fraction', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=1.0, help='multiply every quota (e.g. 0.05 for a smoke test)')
    parser.add_argument('--output', default=str(PROC / 'sessions' / 'annotate_ids.txt'))
    args = parser.parse_args()
    groups = {}
    for path in args.sessions:
        for row in iter_jsonl(path):
            if row.get('kind') == 'i2i' and row['source'] == 'magicbrush' and not row.get('turns'):
                continue
            groups.setdefault((pool_of(row, args.eval_fraction), row['source']), []).append(row['id'])
    chosen = []
    for key, ids in sorted(groups.items()):
        ids.sort(key=lambda i: stable_unit(i, 'annotate'))
        chosen += ids[:max(1, int(QUOTA.get(key, 0) * args.scale))] if QUOTA.get(key) else []
    Path(args.output).write_text('\n'.join(chosen) + '\n')
    picked = set(chosen)
    log(f'{len(chosen)} sessions selected -> {args.output}: '
        f'{dict(Counter(f"{p}/{s}" for (p, s), ids in groups.items() for i in ids if i in picked))}')


if __name__ == '__main__':
    main()
