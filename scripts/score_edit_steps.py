import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_finals import load_scorer
from redag.verify.verifier import score_view


def rows(paths, modes, done, unit):
    for path in map(Path, paths):
        for line in path.read_text(encoding='utf-8').splitlines()[1:]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row['key'].split('|')[1] not in modes:
                continue
            case = row['case']
            if unit == 'sessions':
                if (path.name, row['key'], -1) not in done:
                    yield path.name, row['key'], -1, case, None
                continue
            for index, step in enumerate(case['steps']):
                if step['accepted'] and (path.name, row['key'], index) not in done:
                    yield path.name, row['key'], index, case, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--unit', choices=('steps', 'sessions'), default='steps')
    parser.add_argument('--modes', nargs='+', default=['flat_full', 'tree_rollback_only', 'spatial_no_gates', 'redag_full'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = Path(args.out)
    done = {(r['file'], r['key'], r.get('step', -1)) for r in map(json.loads, out.open())} if out.exists() else set()
    work = list(rows(args.cases, set(args.modes), done, args.unit))
    print('to score', len(work), flush=True)
    if not work:
        return
    scorer = load_scorer(args.model, args.device)
    with out.open('a') as stream:
        for name, key, index, case, step in work:
            if step is None:
                final = case['views'][case['final']]
                initial = case['views'][case['steps'][0]['before']] if case['steps'] else final
                row = dict(file=name, key=key, step=-1, initial=score_view(initial, scorer), final=score_view(final, scorer),
                           thresholds={nid: n['threshold'] for nid, n in final['nodes'].items()}, accepted=sum(s['accepted'] for s in case['steps']),
                           steps=len(case['steps']), execution=case.get('execution', {}))
            else:
                row = dict(file=name, key=key, step=index, target=step['target'], affected=step['affected'], protected=step['protected'],
                           thresholds={nid: n['threshold'] for nid, n in case['views'][step['candidate']]['nodes'].items()},
                           before=score_view(case['views'][step['before']], scorer), after=score_view(case['views'][step['candidate']], scorer))
            stream.write(json.dumps(row) + '\n')
            stream.flush()


if __name__ == '__main__':
    main()
