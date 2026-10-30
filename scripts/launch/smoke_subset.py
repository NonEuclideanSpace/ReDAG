"""Tiny absolute-path subsets of the prepared files for the first-hour smoke test / throughput calibration."""
import json
import sys
from pathlib import Path

PATH_KEYS = ('image', 'source', 'target', 'mask')


def absolute(row, folder):
    for key in PATH_KEYS:
        if isinstance(row.get(key), str):
            row[key] = str((folder / row[key]).resolve())
    return row


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    plan = {'editor_sft.jsonl': ('editor_sft_train.jsonl', 8), 'rs_states.jsonl': ('controller_rs_states.jsonl', 4),
            'rl_train.jsonl': ('rl_train.jsonl', 4), 'rl_dev.jsonl': ('rl_dev.jsonl', 1)}
    for name, (source, n) in plan.items():
        path = data / source
        if not path.exists() and source == 'editor_sft_train.jsonl':
            path = data / 'editor_sft_dev.jsonl'
        rows = [absolute(json.loads(l), path.parent) for l in path.read_text().splitlines()[:n] if l.strip()]
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    rows = [json.loads(l) for l in (data / 'eval_main.jsonl').read_text().splitlines() if l.strip()]
    picked = [absolute(next(r for r in rows if r['k'] == k), data) for k in (10, 20, 30) if any(r['k'] == k for r in rows)]
    (out / 'eval.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in picked))
    state = absolute(json.loads((data / 'controller_rs_states.jsonl').read_text().splitlines()[0]), data)
    nodes = state['graph']['nodes']
    synthetic = dict(state, scores={n['node_id']: 0.5 for n in nodes}, target_node_id=nodes[0]['node_id'],
                     action={'op': 'modify_attr', 'box': [0.25, 0.25, 0.75, 0.75],
                             'instruction': 'Edit the selected region so that the requested condition is satisfied.'})
    (out / 'controller_sft_synthetic.jsonl').write_text(json.dumps(synthetic, ensure_ascii=False) + '\n')
    print('smoke subsets written to', out)


if __name__ == '__main__':
    main()
