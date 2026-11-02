import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.harness.environment import add_runtime_arguments, load_runtime
from redag.learn.rl_trainer import select_failed_parent
from redag.learn.sft_trainer import read_records


def summarize(rows):
    states = defaultdict(list)
    for row in rows:
        states[(row['source'], row['index'])].append(row)
    out = defaultdict(lambda: [0, 0.0, 0, 0.0, 0.0])
    for items in states.values():
        for key in ('ALL', 'type=' + items[0]['node_type']):
            entry = out[key]
            entry[0] += 1
            entry[1] += sum(r['accepted'] for r in items) / len(items)
            entry[2] += any(r['accepted'] for r in items)
            entry[3] += sum(r['target_ok'] for r in items) / len(items)
            entry[4] += any(r['target_ok'] for r in items)
    return {k: dict(states=v[0], pass1=v[1] / v[0], passk=v[2] / v[0], target1=v[3] / v[0], targetk=v[4] / v[0]) for k, v in out.items()}


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.add_argument('--data', nargs='+', default=['data/final/rl_train.jsonl', 'data/final/rl_dev.jsonl'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--vary', choices=('editor', 'controller'), default='editor')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    harness, controller, editor = load_runtime(args)
    out = Path(args.out)
    done = set()
    rows = []
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            rows.append(row)
            done.add((row['source'], row['index'], row['sample']))
    work = []
    for path in args.data:
        records = read_records(path)
        if args.limit:
            records = records[:args.limit]
        work.extend((Path(path).name, i, r) for i, r in enumerate(records))
    with out.open('a') as stream:
        for source, index, record in work:
            if all((source, index, k) in done for k in range(args.samples)):
                continue
            harness.history, harness.last_run, harness.unknown_nodes = [], {}, set()
            harness.verifier.call_limit, harness.render_limit = None, None
            state, dag, target = select_failed_parent(record, harness)
            if target is None:
                continue
            fixed = None
            if args.vary == 'editor':
                fixed = controller.sample(state, dag, target, seed=args.seed, temperature=0)['action']
            for k in range(args.samples):
                if (source, index, k) in done:
                    continue
                action = fixed if fixed is not None else controller.sample(state, dag, target, seed=args.seed + k, temperature=0.8)['action']
                seed = args.seed + 1000 + k if args.vary == 'editor' else args.seed + 1000
                crop, editable = harness.render(editor, state, action, seed, None)
                harness.evaluate_candidate_action(state, dag, target, action, crop, editable)
                last = harness.last_candidate
                row = dict(source=source, index=index, sample=k, target=target, node_type=dag.nodes[target].node_type, accepted=bool(last['accepted']),
                           target_ok=last['raw_scores'][target] >= dag.nodes[target].threshold, before=state.scores[target], after=last['raw_scores'][target],
                           reward=last['reward'], affected=len(last['affected']), verifier_calls=last['verifier_calls'], box=list(action.box),
                           instruction=action.instruction if k == 0 or args.vary == 'controller' else None)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                rows.append(row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = summarize(rows)
    Path(str(out) + '.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary.get('ALL', {})))


if __name__ == '__main__':
    main()
