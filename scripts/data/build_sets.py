"""Annotated condition graphs -> the JSONL files consumed by scripts/*.py (CPU only).

Pools are fixed before anything is sampled: MagicBrush dev and a hash-selected share of every other
benchmark are held out for evaluation; the rest feeds training.  Outputs in data/final:
  eval_main.jsonl              K = 10/20/30 multi-turn sessions with exact final node counts
  eval_credit.jsonl            small diagnostic subset for experiment 2
  planner_sft.jsonl / _dev     (user_prompt, previous_dag, response) pairs, text-derivable nodes only
  controller_rs_states.jsonl   N_state failure states for collect_controller_rs.py (eq. 19)
  rl_train.jsonl / rl_dev.jsonl  failure states for run_rl_alternating.py (session-disjoint)
  data/processed/canvases/requests.jsonl  T2I sessions that still need an initial canvas
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _common import (EDGE_KEYS, FINAL, PROC, build_dag, closure, iter_jsonl, log, parents_of, relpath,
                     sanitize_graph, stable_unit, topo_order, write_jsonl)

TIERS = (30, 20, 10)
TURNS = {10: 2, 20: 3, 30: 4}
QUOTA = {30: {'corebench': 0.40, 'lgbench': 0.40, 'longbench': 0.10, 'magicbrush': 0.10},
         20: {'corebench': 0.30, 'lgbench': 0.30, 'longbench': 0.10, 'magicbrush': 0.30},
         10: {'corebench': 0.25, 'lgbench': 0.25, 'longbench': 0.10, 'magicbrush': 0.40}}


def load_graphs(folder, statuses, sessions_dir=None):
    """Annotated sessions; canvas fields come from the current session files, because a session can be annotated
    before its initial canvas is attached (the checklist benchmarks are annotated while CoReBench canvases download)."""
    current = {}
    for path in sorted(Path(sessions_dir or PROC / 'sessions').glob('*.jsonl')):
        for row in iter_jsonl(path):
            current[row['id']] = row
    sessions = {}
    for path in sorted(Path(folder).glob('*.jsonl')):
        for row in iter_jsonl(path):
            if row.get('annotation', {}).get('status') in statuses:
                fresh = current.get(row['id'], {})
                row.update({key: fresh[key] for key in ('image', 'canvas_origin', 'canvas_prompt') if key in fresh})
                sessions[row['id']] = row
    return sessions


def pool_of(row, eval_fraction):
    if row['source'] == 'magicbrush':
        return 'eval' if row.get('split') == 'dev' else 'train'
    if row['source'] == 'corebench' and not str(row.get('split', '')).startswith('C-'):
        return 'train'  # reasoning splits: few, unusual conditions -> training only
    return 'eval' if stable_unit(row['id']) < eval_fraction else 'train'


def table(row):
    nodes = {n['node_id']: n for n in row['graph']['nodes']}
    return nodes, parents_of(nodes, row['graph']['comp_edges'], row['graph']['supp_edges'])


def closure_all(ids, parents):
    out = set()
    for nid in ids:
        out |= closure(nid, parents)
    return out


def superseded_ids(row, upto=None):
    return {i for t, ids in (row.get('superseded') or {}).items() if upto is None or int(t) <= upto for i in ids}


def edit_ids(row, upto=None):
    if row['mode'] != 'instruction':
        return []
    return [n for n, t in row['node_turn'].items() if t >= 1 and (upto is None or t <= upto)]


def select_exact(nodes, parents, size, required, rng, exclude=frozenset(), tries=64):
    """Dependency-closed node set of exactly `size` nodes that contains `required`, else None."""
    base = closure_all(required, parents)
    if len(base) > size or base & exclude:
        return None
    free = [n for n in nodes if n not in base and n not in exclude]
    for _ in range(tries):
        rng.shuffle(free)
        chosen = set(base)
        for nid in sorted(free, key=lambda n: nodes[n]['node_type'] == 'entity'):  # structured nodes pull in entities
            extra = closure(nid, parents) - chosen
            if not extra & exclude and len(chosen) + len(extra) <= size:
                chosen |= extra
            if len(chosen) == size:
                return chosen
    return None


def statement(row, node):
    text = (row.get('statements') or {}).get(node['node_id'])
    return text or f'Make sure the answer to "{node["question"]}" is yes.'


def turn_prompt(row, index, items):
    if row['mode'] == 'instruction':
        return row['turns'][index - 1]['instruction']
    body = ' '.join(statement(row, n) for n in items)
    if index == 1:
        return ('Create an image where: ' if row['kind'] == 't2i' else 'Edit the image so that: ') + body
    return 'Also make sure that: ' + body


def session_turns(row, chosen, turns):
    nodes, parents = table(row)
    order = [n for n in topo_order(nodes, row['graph']['comp_edges'], row['graph']['supp_edges']) if n in chosen]
    if row['mode'] == 'instruction':
        turn_of = {n: max(1, row['node_turn'].get(n, 1)) for n in order}
    else:
        bounds = [round(len(order) * (i + 1) / turns) for i in range(turns)]
        turn_of = {n: next(t + 1 for t, b in enumerate(bounds) if i < b) for i, n in enumerate(order)}
    for n in order:  # never before its dependencies
        turn_of[n] = max([turn_of[n]] + [turn_of[d] for d in parents[n] if d in turn_of])
    out = []
    for t in range(1, max(turn_of.values()) + 1):
        items = [dict(nodes[n], bbox=None, depends_on=[d for d in nodes[n]['depends_on'] if d in chosen])
                 for n in order if turn_of[n] == t]
        if not items:
            return None
        edges = {key: [list(e) for e in row['graph'][key] if e[0] in chosen and e[1] in chosen
                       and max(turn_of[e[0]], turn_of[e[1]]) == t] for key in EDGE_KEYS}
        dropped = sorted(set((row.get('superseded') or {}).get(str(t), [])) & chosen)
        out.append({'user_prompt': turn_prompt(row, t, items), 'graph': {'nodes': items, **edges}, 'superseded_ids': dropped})
    return out


def merged(turns):
    graph = None
    for turn in turns:
        graph = build_dag(turn['graph'], graph, turn['superseded_ids'])
    return graph


def canvas_for(row, requests, seed):
    """Initial canvas path; T2I sessions without one get a render request (prompt = the full original prompt)."""
    if row.get('image'):
        return row['image']
    digest = hashlib.sha1(row['id'].encode()).hexdigest()[:16]
    path = PROC / 'canvases' / f'{digest}.png'
    requests[row['id']] = {'id': row['id'], 'prompt': row.get('canvas_prompt') or row.get('prompt') or '', 'path': str(path),
                           'seed': seed + int(digest[:6], 16) % 100000, 'kind': row['kind']}
    return str(path)


def build_eval(sessions, per_tier, rng, requests, seed):
    used, records, short = set(), [], {}
    for k in TIERS:
        eligible = defaultdict(list)
        for row in sessions:
            if row['id'] in used:
                continue
            nodes, parents = table(row)
            target = k + len(superseded_ids(row))
            chosen = select_exact(nodes, parents, target, edit_ids(row), random.Random(f'{seed}:{k}:{row["id"]}'))
            turns = chosen and session_turns(row, chosen, TURNS[k])
            if not turns:
                continue
            try:
                if len(merged(turns).nodes) == k:
                    eligible[row['source']].append((row, turns))
            except (ValueError, TypeError):
                continue
        for items in eligible.values():
            rng.shuffle(items)
        want = {src: int(round(per_tier * share)) for src, share in QUOTA[k].items()}
        picked = [item for src in sorted(want, key=lambda s: -want[s]) for item in eligible.get(src, [])[:want[src]]]
        spare = [item for src, items in eligible.items() for item in items[want.get(src, 0):]]
        rng.shuffle(spare)
        picked = (picked + spare)[:per_tier]
        if len(picked) < per_tier:
            short[k] = per_tier - len(picked)
        for row, turns in picked:
            used.add(row['id'])
            records.append({'id': f'eval:K{k}:{row["id"]}', 'session_id': row['id'], 'source': row['source'],
                            'kind': row['kind'], 'k': k, 'image': canvas_for(row, requests, seed), 'turns': turns})
    return records, short


def failure_state(row, rng, k_range):
    """One failure state (image + merged graph) per training session."""
    nodes, parents = table(row)
    if row['mode'] == 'instruction':
        turn = rng.randint(1, len(row['turns']))
        image = row['image'] if turn == 1 else row['turns'][turn - 2]['target']
        future = {n for n, t in row['node_turn'].items() if t > turn}  # later requests are not part of this state
        removed = frozenset(superseded_ids(row, turn)) | future
        edits = [n for n in edit_ids(row, turn) if n not in removed]
        inventory = [n for n in nodes if row['node_turn'].get(n) == 0]
        size = min(len(closure_all(edits, parents)) + rng.randint(2, 8), len(edits) + len(inventory))
        chosen = select_exact(nodes, parents, size, edits, rng, exclude=removed)
    else:
        turn, image = 1, row.get('image')
        chosen = select_exact(nodes, parents, min(rng.randint(*k_range), len(nodes)), [], rng)
    if not chosen or not image:
        return None
    order = topo_order(nodes, row['graph']['comp_edges'], row['graph']['supp_edges'])
    graph, _ = sanitize_graph({n: dict(nodes[n], bbox=None) for n in order if n in chosen},
                              row['graph']['comp_edges'], row['graph']['supp_edges'], row['graph']['int_edges'])
    build_dag(graph)
    return {'id': f'state:{row["id"]}:t{turn}', 'session_id': row['id'], 'origin': row['source'], 'turn': turn,
            'image': image, 'graph': graph}


def planner_pairs(row, rng):
    """Pairs for the text-only planner; image-only inventory facts are left out unless an edit refers to them."""
    from redag.models.planner import dag_to_dict
    nodes, parents = table(row)
    order = topo_order(nodes, row['graph']['comp_edges'], row['graph']['supp_edges'])
    pairs, previous = [], None
    if row['mode'] == 'instruction':
        included = set()
        for turn in range(1, len(row['turns']) + 1):
            new = closure_all([n for n, t in row['node_turn'].items() if t == turn], parents) - included
            items = [dict(nodes[n], bbox=None) for n in order if n in new]
            if not items:
                break
            scope = included | new
            edges = {key: [list(e) for e in row['graph'][key] if e[0] in scope and e[1] in scope and (e[0] in new or e[1] in new)]
                     for key in EDGE_KEYS}
            dropped = sorted(set((row.get('superseded') or {}).get(str(turn), [])) & included)
            response = {'nodes': items, **edges, 'superseded_ids': dropped}
            pairs.append({'id': f'plan:{row["id"]}:t{turn}', 'user_prompt': row['turns'][turn - 1]['instruction'],
                          'previous_dag': previous, 'response': response})
            previous = dag_to_dict(build_dag(response, previous, dropped))
            included = (included | new) - set(dropped)
        return pairs
    full = {'nodes': [dict(nodes[n], bbox=None) for n in order], **{k: row['graph'][k] for k in EDGE_KEYS}}
    pairs.append({'id': f'plan:{row["id"]}:full', 'user_prompt': row.get('prompt') or '', 'previous_dag': None, 'response': full})
    chosen = select_exact(nodes, parents, min(len(nodes), rng.randint(8, 24)), [], rng)
    for index, turn in enumerate((chosen and session_turns(row, chosen, rng.randint(2, 3))) or []):
        pairs.append({'id': f'plan:{row["id"]}:inc{index + 1}', 'user_prompt': turn['user_prompt'], 'previous_dag': previous,
                      'response': dict(turn['graph'], superseded_ids=[])})
        previous = dag_to_dict(build_dag(turn['graph'], previous))
    return [p for p in pairs if p['user_prompt']]


def take(states, n, share, used):
    """n states, `share` of them from MagicBrush when available, never reusing a session."""
    mb = [s for s in states if s['origin'] == 'magicbrush' and s['session_id'] not in used]
    other = [s for s in states if s['origin'] != 'magicbrush' and s['session_id'] not in used]
    picked = mb[:int(round(n * share))]
    picked += other[:n - len(picked)]
    picked += [s for s in mb if s not in picked][:n - len(picked)]
    used |= {s['session_id'] for s in picked}
    return picked


def relative(rows):
    for row in rows:
        if row.get('image'):
            row['image'] = relpath(row['image'], FINAL)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs', default=str(PROC / 'graphs'))
    parser.add_argument('--per_tier', type=int, default=100)
    parser.add_argument('--credit_sessions', type=int, default=24)
    parser.add_argument('--eval_fraction', type=float, default=0.5, help='share of non-MagicBrush sessions held out for eval')
    parser.add_argument('--n_rs_states', type=int, default=1000)
    parser.add_argument('--n_rl_train', type=int, default=128)
    parser.add_argument('--n_rl_dev', type=int, default=16)
    parser.add_argument('--state_k', type=int, nargs=2, default=(6, 16))
    parser.add_argument('--mb_state_share', type=float, default=0.6)
    parser.add_argument('--statuses', nargs='+', default=['ok', 'repaired'], help='add "fallback" only for smoke tests')
    parser.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sessions = load_graphs(args.graphs, set(args.statuses))
    pools = defaultdict(list)
    for row in sorted(sessions.values(), key=lambda r: r['id']):
        pools[pool_of(row, args.eval_fraction)].append(row)
    log(f'{len(sessions)} usable sessions; eval pool {len(pools["eval"])}, train pool {len(pools["train"])}; '
        f'{dict(Counter(r["source"] for r in sessions.values()))}')
    requests = {}
    eval_rows, short = build_eval(pools['eval'], args.per_tier, rng, requests, args.seed)
    credit = [r for k in (10, 20, 30) for r in random.Random(f'{args.seed}:credit:{k}').sample(
        [x for x in eval_rows if x['k'] == k], min(args.credit_sessions // 3, sum(x['k'] == k for x in eval_rows)))]
    train = pools['train'][:]
    rng.shuffle(train)
    states, pending = [], {}
    for row in train:
        if not row.get('image'):
            planned = {}
            row = dict(row, image=canvas_for(row, planned, args.seed))
            pending.update(planned)
        state = failure_state(row, random.Random(f'{args.seed}:state:{row["id"]}'), tuple(args.state_k))
        if state:
            states.append(state)
    used = set()
    dev = take(states, args.n_rl_dev, args.mb_state_share, used)
    rl = take(states, args.n_rl_train, args.mb_state_share, used)
    rs = take(states, args.n_rs_states, args.mb_state_share, used)
    requests.update({sid: pending[sid] for sid in used if sid in pending})  # render only canvases that are used
    pairs = []
    for row in train:
        try:
            pairs += planner_pairs(row, random.Random(f'{args.seed}:plan:{row["id"]}'))
        except (ValueError, TypeError) as error:
            log(f'skip planner pairs for {row["id"]}: {error}')
    held = {p['id'] for p in pairs if stable_unit(p['id'].rsplit(':', 1)[0], 'plan-dev') < 0.05}
    outputs = {'eval_main.jsonl': relative(eval_rows), 'eval_credit.jsonl': credit, 'rl_dev.jsonl': relative(dev),
               'rl_train.jsonl': relative(rl), 'controller_rs_states.jsonl': relative(rs),
               'planner_sft.jsonl': [p for p in pairs if p['id'] not in held],
               'planner_sft_dev.jsonl': [p for p in pairs if p['id'] in held]}
    manifest = {'args': vars(args), 'short_tiers': short, 'eval_k': dict(Counter(r['k'] for r in eval_rows)),
                'canvas_requests': len(requests), 'files': {}}
    for name, rows in outputs.items():
        manifest['files'][name] = {'rows': write_jsonl(FINAL / name, rows),
                                   'by_source': dict(Counter(r.get('source') or r.get('origin') or r['id'].split(':')[1]
                                                             for r in rows))}
    write_jsonl(PROC / 'canvases' / 'requests.jsonl', sorted(requests.values(), key=lambda r: r['id']))
    (FINAL / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log(json.dumps({k: v['rows'] for k, v in manifest['files'].items()}), 'eval K', manifest['eval_k'],
        'short', short, 'canvas requests', len(requests))


if __name__ == '__main__':
    main()
