"""Validate prepared JSONL files against what scripts/*.py will accept (CPU only, no models).

Uses the repository's own loaders (read_records resolves relative image paths; ConditionDAG /
preserve_previous_conditions / validate_dataset / ControllerAction) so a file that passes here will
not fail on schema or K-tier checks after the GPU job has started.
  python scripts/data/check_data.py                 # every known file present in data/final
  python scripts/data/check_data.py --images all    # also open every image (slow in no-GPU mode)
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from _common import FINAL, log


def open_image(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        return image.size, image.mode, image


def check_editor(rows, images):
    import numpy as np
    from PIL import Image
    errors = []
    for row in rows if images == 'all' else random.Random(0).sample(rows, min(len(rows), 50)):
        try:
            sizes = [open_image(row[k])[0] for k in ('source', 'target', 'mask')]
            if len(set(sizes)) != 1:
                errors.append(f"{row['id']}: source/target/mask sizes differ {sizes}")
            with Image.open(row['mask']) as mask:
                if not (np.asarray(mask.convert('L')) > 127).any():
                    errors.append(f"{row['id']}: empty mask")
            if not str(row.get('instruction', '')).strip():
                errors.append(f"{row['id']}: empty instruction")
        except (OSError, KeyError) as error:
            errors.append(f"{row.get('id')}: {error}")
    return errors


def check_planner(rows, _):
    from redag.models.planner import parse_json_object, preserve_previous_conditions
    errors = []
    for row in rows:
        try:
            response = parse_json_object(row.get('response', row.get('dag', row.get('graph'))))
            preserve_previous_conditions(response, row.get('previous_dag'))
            if not str(row.get('user_prompt', '')).strip():
                errors.append(f"{row.get('id')}: empty user_prompt")
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"{row.get('id')}: {error}")
    return errors


def check_states(rows, images, need_frontier=True):
    from redag.graph.dag import compute_active_frontier
    from redag.models.planner import dag_from_dict
    errors, ids = [], Counter(r.get('id') for r in rows)
    errors += [f'duplicate id {i}' for i, n in ids.items() if n > 1]
    for row in rows:
        try:
            dag = dag_from_dict(row.get('graph', row.get('dag')))
            if not Path(row['image']).exists():
                errors.append(f"{row['id']}: missing image {row['image']}")
            elif images == 'all':
                open_image(row['image'])
            if 'scores' in row:
                scores = row['scores']
                if set(scores) != set(dag.nodes) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
                    errors.append(f"{row['id']}: scores must cover every node with values in [0, 1]")
                elif need_frontier and not compute_active_frontier(dag, scores):
                    errors.append(f"{row['id']}: no failing node to repair")
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"{row.get('id')}: {error}")
    return errors


def check_controller_sft(rows, images):
    from redag.graph.schema import ControllerAction
    errors = check_states(rows, images, need_frontier=False)
    for row in rows:
        try:
            if row['target_node_id'] not in {n['node_id'] for n in row['graph']['nodes']}:
                errors.append(f"{row['id']}: target not in graph")
            action = row['action']
            ControllerAction(action['op'], action['box'], action['instruction'])
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"{row.get('id')}: {error}")
    return errors


def check_eval(rows, images, main):
    from redag.harness.environment import validate_dataset
    errors = []
    try:
        validate_dataset(rows, 1, not main)
    except (ValueError, TypeError) as error:
        errors.append(f'validate_dataset: {error}')
    missing = [r['id'] for r in rows if not Path(r['image']).exists()]
    if missing:
        errors.append(f'{len(missing)} sessions have no canvas yet (render_canvases.py), e.g. {missing[:3]}')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=str(FINAL))
    parser.add_argument('--images', choices=('sample', 'all'), default='sample')
    parser.add_argument('--allow_missing_canvases', action='store_true')
    args = parser.parse_args()
    from redag.learn.sft_trainer import read_records
    folder, report, failed = Path(args.dir), {}, False
    checks = [('editor_sft*.jsonl', check_editor), ('planner_sft*.jsonl', check_planner),
              ('controller_rs_states.jsonl', check_states), ('rl_train.jsonl', check_states), ('rl_dev.jsonl', check_states),
              ('controller_sft*.jsonl', check_controller_sft),
              ('eval_main.jsonl', lambda r, i: check_eval(r, i, True)), ('eval_credit.jsonl', lambda r, i: check_eval(r, i, False))]
    for pattern, check in checks:
        for path in sorted(folder.glob(pattern)):
            rows = read_records(path)
            errors = check(rows, args.images)
            if args.allow_missing_canvases:
                errors = [e for e in errors if 'no canvas yet' not in e and 'missing image' not in e]
            extra = {'k': dict(Counter(r.get('k') for r in rows))} if path.name.startswith('eval_') else {}
            report[path.name] = {'rows': len(rows), 'errors': len(errors), 'first_errors': errors[:5], **extra}
            failed |= bool(errors)
            log(f"{path.name}: {len(rows)} rows, {len(errors)} problems" + (f' e.g. {errors[0]}' if errors else ''))
    train, dev = folder / 'rl_train.jsonl', folder / 'rl_dev.jsonl'
    if train.exists() and dev.exists():
        a, b = read_records(train), read_records(dev)
        overlap = {r['id'] for r in a} & {r['id'] for r in b} | {r['image'] for r in a} & {r['image'] for r in b}
        report['rl_train_dev_disjoint'] = not overlap
        failed |= bool(overlap)
        log(f'rl train/dev disjoint: {not overlap}')
    (folder / 'check_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
