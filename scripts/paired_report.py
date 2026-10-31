import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def run_label(file_name, mode):
    name = file_name.split('.s')[0].removeprefix('exp1').removeprefix('_')
    return mode if name in ('', mode) else f'{name}/{mode}'


def dependency_parents(record):
    parents = defaultdict(set)
    for turn in record['turns']:
        graph = turn['graph']
        for node in graph['nodes']:
            parents[node['node_id']].update(node.get('depends_on', []))
        for key in ('comp_edges', 'supp_edges'):
            for parent, child in graph.get(key, []):
                parents[child].add(parent)
    return parents


def step_score(record):
    return record['stage2'] if record['stage2'] is not None else record['stage1']


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float('nan')


def sign_p(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else float('nan')


def compare(a, b):
    wins, losses = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    return '%.2f vs %.2f (better %d worse %d p %.3f)' % (mean(a), mean(b), wins, losses, sign_p(wins, losses))


def load_sessions(paths, skip):
    sessions, components = {}, Counter()
    for path in paths:
        name = Path(path).name
        if any(token in name for token in skip):
            continue
        for line in Path(path).read_text(encoding='utf-8').splitlines()[1:]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            _, mode, index, seed = row['key'].split('|')
            label, case = run_label(name, mode), row['case']
            damage = 0
            for step in case['steps']:
                nodes = case['views'][step['candidate']]['nodes']
                scores = {r['node_id']: step_score(r) for r in step['stage_scores']}
                damage += sum(1 for v in step['protected'] if v in scores and scores[v] < nodes[v]['threshold'])
                closure, refuted = step.get('closure_nodes', []), step.get('refuted_drops', [])
                if closure or refuted:
                    components[label, 'steps_with_closure'] += bool(closure)
                    if step['accepted'] and closure:
                        components[label, 'closure_requirements_in_accepted_edits'] += len(closure)
                        components[label, 'closure_requirements_met_in_the_same_edit'] += sum(1 for v in closure if v in scores and scores[v] >= nodes[v]['threshold'])
                    components[label, 'steps_with_refuted_drop'] += bool(refuted)
                    components[label, 'acceptances_rescued_by_confirmation'] += bool(refuted) and step['accepted']
            execution = case.get('execution', {})
            sessions[label, int(index), int(seed)] = dict(k=len(case['views'][case['final']]['nodes']), accepted=sum(s['accepted'] for s in case['steps']),
                                                          damage=damage, verifier=execution.get('verifier_calls') or 0, stop=execution.get('stop_reason'))
    return sessions, components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', default='runs/eval')
    parser.add_argument('--data', default='data/final/eval_main.jsonl')
    parser.add_argument('--judges', nargs='*', default=[])
    parser.add_argument('--base', default='redag_edge/redag_full')
    parser.add_argument('--skip', nargs='*', default=['turnbudget'])
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.data, encoding='utf-8')]
    parents = [dependency_parents(r) for r in records]
    turns, edges = [len(r['turns']) for r in records], [sum(len(p) for p in ps.values()) for ps in parents]
    sessions, components = load_sessions(sorted(glob.glob(str(Path(args.eval_dir) / 'exp1*.cases.jsonl'))), args.skip)
    labels = sorted({label for label, _, _ in sessions})
    for label in labels:
        rows = [v for (l, _, _), v in sessions.items() if l == label]
        print('RUN %-28s n %3d accepted %.2f damage %.2f verifier %.1f stops %s' % (label, len(rows), mean([r['accepted'] for r in rows]),
              mean([r['damage'] for r in rows]), mean([r['verifier'] for r in rows]), dict(Counter(r['stop'] for r in rows))))
    for (label, key), value in sorted(components.items()):
        print('COMPONENT %s %s %d' % (label, key, value))
    base = args.base
    density = sorted(edges[i] / max(1, v['k']) for (l, i, _), v in sessions.items() if l == base)
    cut = density[len(density) // 2] if density else 0
    for other in labels:
        keys = sorted((i, s) for (l, i, s) in sessions if l == base and (other, i, s) in sessions) if other != base else []
        if not keys:
            continue
        get = lambda label, field, ks: [sessions[label, i, s][field] for i, s in ks]
        print('PAIR %s vs %s n %d accepted %s damage %.2f vs %.2f verifier %.1f vs %.1f' % (base, other, len(keys), compare(get(base, 'accepted', keys), get(other, 'accepted', keys)),
              mean(get(base, 'damage', keys)), mean(get(other, 'damage', keys)), mean(get(base, 'verifier', keys)), mean(get(other, 'verifier', keys))))
        subsets = [('K=%d' % k, lambda i, s, k=k: sessions[base, i, s]['k'] == k) for k in (10, 20, 30)]
        subsets += [('turns<=2', lambda i, s: turns[i] <= 2), ('turns>=3', lambda i, s: turns[i] >= 3),
                    ('sparse deps', lambda i, s: edges[i] / max(1, sessions[base, i, s]['k']) < cut),
                    ('dense deps', lambda i, s: edges[i] / max(1, sessions[base, i, s]['k']) >= cut)]
        for tag, test in subsets:
            subset = [(i, s) for i, s in keys if test(i, s)]
            if subset:
                print('    %-11s n %3d accepted %s' % (tag, len(subset), compare(get(base, 'accepted', subset), get(other, 'accepted', subset))))
    judged = {}
    for path in args.judges:
        for line in open(path, encoding='utf-8'):
            row = json.loads(line)
            _, mode, index, _ = row['key'].split('|')
            judged[run_label(row.get('file', ''), mode), int(index)] = row
    if not judged:
        return

    def consistent(index, row):
        scores, memo = row['scores'], {}

        def holds(node):
            if node not in memo:
                memo[node] = False
                memo[node] = scores.get(node, 0) >= 0.7 and all(holds(p) for p in parents[index].get(node, ()) if p in scores)
            return memo[node]
        return sum(holds(node) for node in scores)
    for label in sorted({l for l, _ in judged}):
        rows = [(i, r) for (l, i), r in judged.items() if l == label]
        print('JUDGE %-28s n %3d satisfied/K %.3f consistent/K %.3f all-satisfied %d' % (label, len(rows), mean([r['satisfied'] / r['k'] for _, r in rows]),
              mean([consistent(i, r) / r['k'] for i, r in rows]), sum(r['all_satisfied'] for _, r in rows)))
    for other in sorted({l for l, _ in judged}):
        keys = sorted(i for (l, i) in judged if l == base and (other, i) in judged) if other != base else []
        if keys:
            satisfied = [judged[base, i]['satisfied'] - judged[other, i]['satisfied'] for i in keys]
            chained = [consistent(i, judged[base, i]) - consistent(i, judged[other, i]) for i in keys]
            print('JPAIR %s vs %s n %d satisfied %+.2f (%d/%d p %.3f) consistent %+.2f (%d/%d p %.3f)' % (
                base, other, len(keys), mean(satisfied), sum(x > 0 for x in satisfied), sum(x < 0 for x in satisfied),
                sign_p(sum(x > 0 for x in satisfied), sum(x < 0 for x in satisfied)), mean(chained), sum(x > 0 for x in chained),
                sum(x < 0 for x in chained), sign_p(sum(x > 0 for x in chained), sum(x < 0 for x in chained))))


if __name__ == '__main__':
    main()
