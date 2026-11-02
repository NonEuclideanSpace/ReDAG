import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(path):
    states = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            states[(row['source'], row['index'])].append(row)
    return {key: dict(node_type=rows[0]['node_type'], pass1=sum(r['accepted'] for r in rows) / len(rows),
                      target1=sum(r['target_ok'] for r in rows) / len(rows), passk=float(any(r['accepted'] for r in rows)),
                      samples=len(rows)) for key, rows in states.items()}


def bootstrap(diffs, rounds=10000, seed=0):
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(rounds))
    return (means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    runs = {}
    for spec in args.runs:
        name, path = spec.split('=', 1)
        if Path(path).exists():
            data = load(path)
            if data:
                runs[name] = data
    if args.reference not in runs:
        raise SystemExit(f'reference run {args.reference} has no rows')
    common = sorted(set.intersection(*(set(v) for v in runs.values())))
    ref = runs[args.reference]
    report, lines = {}, [f'common states {len(common)}', 'run pass1 target1 passk samples diff_vs_ref ci95 wins losses']
    for name, data in runs.items():
        rows = [data[k] for k in common]
        n = max(1, len(rows))
        diffs = [data[k]['pass1'] - ref[k]['pass1'] for k in common]
        low, high = bootstrap(diffs)
        types = defaultdict(list)
        for k in common:
            types[data[k]['node_type']].append(data[k]['pass1'])
        entry = dict(states=len(rows), pass1=sum(r['pass1'] for r in rows) / n, target1=sum(r['target1'] for r in rows) / n,
                     passk=sum(r['passk'] for r in rows) / n, samples=sum(r['samples'] for r in rows) / n,
                     diff=sum(diffs) / max(1, len(diffs)), ci95=[low, high], wins=sum(d > 0 for d in diffs), losses=sum(d < 0 for d in diffs),
                     by_type={t: dict(states=len(v), pass1=sum(v) / len(v)) for t, v in sorted(types.items())})
        report[name] = entry
        lines.append(f"{name} {entry['pass1']:.3f} {entry['target1']:.3f} {entry['passk']:.3f} {entry['samples']:.1f} {entry['diff']:+.3f} "
                     f"[{low:+.3f},{high:+.3f}] {entry['wins']} {entry['losses']}")
    Path(args.out).write_text(json.dumps(dict(common_states=len(common), reference=args.reference, runs=report), indent=2))
    Path(args.out).with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
