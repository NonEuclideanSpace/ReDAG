import argparse
import json
import math
from collections import Counter, defaultdict


def graphs(path):
    out = []
    for record in map(json.loads, open(path, encoding='utf-8')):
        types, children, parents = {}, defaultdict(set), defaultdict(set)
        for turn in record['turns']:
            graph = turn['graph']
            for node in graph['nodes']:
                types[node['node_id']] = node['node_type']
                for parent in node.get('depends_on', []):
                    children[parent].add(node['node_id'])
                    parents[node['node_id']].add(parent)
            for key in ('comp_edges', 'supp_edges'):
                for parent, child in graph.get(key, []):
                    children[parent].add(child)
                    parents[child].add(parent)
        out.append((types, children, parents, record.get('parent')))
    return out


def descendants(children, roots):
    found, stack = set(), list(roots)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    p = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= p0 + 1e-12)


def label(name, mode):
    prefix = name.split('.')[0]
    return f'{prefix}/{mode}'


def missed_damage(paths, main, stress, drop):
    counts = defaultdict(Counter)
    for path in paths:
        for row in map(json.loads, open(path)):
            _, mode, index, _ = row['key'].split('|')
            types, children, parents, _ = (stress if row['file'].startswith('stress_') else main)[int(index)]
            target = row['target']
            anchors = {target} | {p for p in parents.get(target, ()) if types.get(p) == 'entity'}
            group = 'stress' if row['file'].startswith('stress_') else 'main'
            tag = row['file'].split('.')[0]
            c = counts[(group, tag if group == 'stress' else 'exp1', mode)]
            c['accepted_steps'] += 1
            damaged_step = False
            for node in descendants(children, anchors) - {target}:
                before, after, threshold = row['before'].get(node), row['after'].get(node), row['thresholds'].get(node, 0.7)
                if before is None or after is None or before < threshold:
                    continue
                c['satisfied_dependents'] += 1
                if after < threshold or before - after > drop:
                    where = 'checked' if node in row['affected'] else 'unchecked'
                    c[where] += 1
                    c[f'{where}_{types.get(node, "?")}'] += 1
                    damaged_step |= where == 'unchecked'
            c['steps_with_unchecked'] += damaged_step
    for key, c in sorted(counts.items()):
        kinds = {k.split('_', 1)[1]: v for k, v in c.items() if k.startswith('unchecked_')}
        print('MISSED %-28s accepted steps %4d satisfied dependents %4d unchecked damaged %3d (steps %3d) checked damaged %3d by type %s' % (
            '/'.join(key), c['accepted_steps'], c['satisfied_dependents'], c['unchecked'], c['steps_with_unchecked'], c['checked'], kinds))
    for group in sorted({g for g, _, _ in counts}):
        rows = {(t, m): c for (g, t, m), c in counts.items() if g == group}
        for (tag_a, mode_a), a in rows.items():
            for (tag_b, mode_b), b in rows.items():
                if mode_a == 'spatial_no_gates' and mode_b == 'redag_full' and tag_a == tag_b:
                    print('FISHER %s %s unchecked dependents %d/%d vs %d/%d p %.2g | steps %d/%d vs %d/%d p %.2g | all damage %d/%d vs %d/%d p %.2g' % (
                        group, tag_a, a['unchecked'], a['satisfied_dependents'], b['unchecked'], b['satisfied_dependents'],
                        fisher(a['unchecked'], a['satisfied_dependents'] - a['unchecked'], b['unchecked'], b['satisfied_dependents'] - b['unchecked']),
                        a['steps_with_unchecked'], a['accepted_steps'], b['steps_with_unchecked'], b['accepted_steps'],
                        fisher(a['steps_with_unchecked'], a['accepted_steps'] - a['steps_with_unchecked'], b['steps_with_unchecked'], b['accepted_steps'] - b['steps_with_unchecked']),
                        a['unchecked'] + a['checked'], a['satisfied_dependents'], b['unchecked'] + b['checked'], b['satisfied_dependents'],
                        fisher(a['unchecked'] + a['checked'], a['satisfied_dependents'] - a['unchecked'] - a['checked'],
                               b['unchecked'] + b['checked'], b['satisfied_dependents'] - b['unchecked'] - b['checked'])))


def stress_outcomes(paths, stress):
    results = defaultdict(dict)
    for path in paths:
        for row in map(json.loads, open(path)):
            _, mode, index, _ = row['key'].split('|')
            types, children, parents, parent = stress[int(index)]
            related = descendants(children, {parent}) - {'s1'}
            ok = lambda scores, node: scores.get(node, 0) >= row['thresholds'].get(node, 0.7)
            kept = [node for node in related if ok(row['initial'], node)]
            results[label(row['file'], mode)][int(index)] = dict(target=ok(row['final'], 's1'), kept=sum(ok(row['final'], n) for n in kept), total=len(kept),
                                                                  relations=sum(types.get(n) == 'relation' for n in kept),
                                                                  relations_kept=sum(ok(row['final'], n) for n in kept if types.get(n) == 'relation'))
    for name, rows in sorted(results.items()):
        values = list(rows.values())
        clean = sum(r['target'] and r['kept'] == r['total'] for r in values)
        print('STRESS %-34s n %3d target %.0f%% clean %.0f%% dependents kept %d/%d relations kept %d/%d' % (
            name, len(values), 100 * sum(r['target'] for r in values) / len(values), 100 * clean / len(values), sum(r['kept'] for r in values),
            sum(r['total'] for r in values), sum(r['relations_kept'] for r in values), sum(r['relations'] for r in values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/final/eval_main.jsonl')
    parser.add_argument('--stress_data', default='data/final/eval_stress.jsonl')
    parser.add_argument('--steps', nargs='*', default=[])
    parser.add_argument('--sessions', nargs='*', default=[])
    parser.add_argument('--drop', type=float, default=0.08)
    args = parser.parse_args()
    main_graphs, stress_graphs = graphs(args.data), graphs(args.stress_data)
    if args.steps:
        missed_damage(args.steps, main_graphs, stress_graphs, args.drop)
    if args.sessions:
        stress_outcomes(args.sessions, stress_graphs)


if __name__ == '__main__':
    main()
