import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FINAL, PROC, log, write_jsonl
from build_sets import TIERS, TURNS, edit_ids, load_graphs, merged, pool_of, relative, select_exact, session_turns, superseded_ids, table


def build(rows, per_tier, seed, exclude):
    used, records = set(exclude), []
    for k in TIERS:
        eligible = defaultdict(list)
        for row in rows:
            if row['id'] in used:
                continue
            nodes, parents = table(row)
            chosen = select_exact(nodes, parents, k + len(superseded_ids(row)), edit_ids(row), random.Random(f'{seed}:train:{k}:{row["id"]}'))
            turns = chosen and session_turns(row, chosen, TURNS[k])
            if not turns:
                continue
            try:
                if len(merged(turns).nodes) == k:
                    eligible[row['source']].append((row, turns))
            except (ValueError, TypeError):
                continue
        rng = random.Random(f'{seed}:train:{k}')
        pools = [items for _, items in sorted(eligible.items())]
        for items in pools:
            rng.shuffle(items)
        picked = []
        while len(picked) < per_tier and any(pools):
            for items in pools:
                if items and len(picked) < per_tier:
                    picked.append(items.pop())
        for row, turns in picked:
            used.add(row['id'])
            records.append({'id': f'train:K{k}:{row["id"]}', 'session_id': row['id'], 'source': row['source'], 'kind': row['kind'],
                            'k': k, 'image': row['image'], 'turns': turns})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs', default=str(PROC / 'graphs'))
    parser.add_argument('--per_tier', type=int, default=60)
    parser.add_argument('--out', default=str(FINAL / 'train_sessions.jsonl'))
    args = parser.parse_args()
    manifest = json.loads((FINAL / 'manifest.json').read_text())['args']
    sessions = load_graphs(args.graphs, set(manifest.get('statuses', ['ok', 'repaired'])))
    exclude = set()
    for name in ('rl_dev.jsonl', 'eval_main.jsonl', 'eval_credit.jsonl'):
        path = FINAL / name
        if path.exists():
            exclude |= {json.loads(line).get('session_id') for line in path.read_text().splitlines() if line.strip()}
    rows = [row for row in sorted(sessions.values(), key=lambda r: r['id'])
            if pool_of(row, manifest['eval_fraction']) == 'train' and row.get('image') and Path(row['image']).exists()]
    records = relative(build(rows, args.per_tier, manifest['seed'], exclude))
    write_jsonl(Path(args.out), records)
    log(f'{len(records)} training sessions from {len(rows)} train-pool rows with images', dict(Counter((r['kind'], r['k']) for r in records)))


if __name__ == '__main__':
    main()
