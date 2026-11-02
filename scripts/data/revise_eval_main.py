import argparse
import json
from pathlib import Path


REVISIONS = {
    'eval:K20:mb:dev:580685': dict(turn=0, retire=['i12', 'i13', 'i14', 'i15', 'i16'],
                                   reason='The request puts a chocolate cake on the plate and t1_2 requires an image free of hot dogs; '
                                          'the hot dog count, topping and bun conditions describe the replaced food.'),
    'eval:K20:mb:dev:134290': dict(turn=0, retire=['i8'],
                                   reason='The request turns the hair white (t1_1); the reddish-brown hair condition describes the replaced colour.'),
}


def retire(graph, ids):
    ids = set(ids)
    present = {n['node_id'] for n in graph['nodes']}
    if not ids <= present:
        raise ValueError(f'Retired conditions are missing from the graph: {sorted(ids - present)}')
    graph['nodes'] = [n for n in graph['nodes'] if n['node_id'] not in ids]
    for node in graph['nodes']:
        if set(node.get('depends_on') or []) & ids:
            raise ValueError(f"{node['node_id']} depends on a retired condition")
    for key in ('comp_edges', 'supp_edges', 'int_edges'):
        graph[key] = [edge for edge in graph.get(key) or [] if not set(edge) & ids]
    return graph


def revise(record, revisions=REVISIONS):
    revision = revisions.get(record['id'])
    if revision is None:
        return record
    record = json.loads(json.dumps(record))
    retire(record['turns'][revision['turn']]['graph'], revision['retire'])
    record['revisions'] = record.get('revisions', []) + [dict(version='v2', turn=revision['turn'], retired=list(revision['retire']), reason=revision['reason'])]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/final/eval_main.jsonl')
    parser.add_argument('--out', default='data/final/eval_main_v2.jsonl')
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.src).read_text(encoding='utf-8').splitlines() if line.strip()]
    missing = set(REVISIONS) - {r['id'] for r in records}
    if missing:
        raise SystemExit(f'Revised sessions not found: {sorted(missing)}')
    revised = [revise(r) for r in records]
    Path(args.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in revised), encoding='utf-8')
    print(json.dumps(dict(records=len(revised), revised=sorted(REVISIONS), out=args.out)))


if __name__ == '__main__':
    main()
