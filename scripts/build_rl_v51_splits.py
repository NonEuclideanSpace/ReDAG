import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / 'scripts' / 'data') not in sys.path:
    sys.path.insert(0, str(REPO / 'scripts' / 'data'))

from redag.harness.environment import graph_ancestors, is_source_invariant_id
from _common import FINAL, PROC, build_dag, iter_jsonl, relpath, sanitize_graph, topo_order, write_jsonl
from build_sets import (
    closure_all, edit_ids, load_graphs, pool_of, select_exact, superseded_ids, table, turn_prompt,
)

STALE_KEYS = ('detected_counts', 'scores', 'frontier', 'protected', 'verifier_mode')


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def base_id_of(session_id):
    return str(session_id or '').split(':')[-1]


def enrich_metadata(record, session_row):
    out = dict(record)
    for stale_key in STALE_KEYS:
        out.pop(stale_key, None)
    dag_obj = build_dag(out['graph'])
    chosen_ids = [n['node_id'] for n in out['graph']['nodes']]
    turn = int(out.get('turn') or 1)
    if session_row is None:
        draft_t1 = bool(out.get('kind') == 't2i') if 'kind' in out else bool(out.get('origin') != 'magicbrush')
        inv_set = (
            set() if draft_t1 else {
                nid for nid in chosen_ids
                if is_source_invariant_id(nid) and all(is_source_invariant_id(a) for a in graph_ancestors(dag_obj, nid))
            }
        )
        prompt = str(out.get('user_prompt') or '')
        out['user_prompt'] = prompt
        out['turn_nodes'] = [nid for nid in chosen_ids if nid not in inv_set] or list(chosen_ids)
        out['source_invariants'] = [nid for nid in chosen_ids if nid in inv_set]
        out['node_turn_request'] = {nid: ('' if nid in inv_set else prompt) for nid in chosen_ids}
        out['draft_first_turn'] = draft_t1
        return out

    nodes, _ = table(session_row)
    items = [nodes[nid] for nid in chosen_ids if nid in nodes]
    draft_t1 = bool(session_row.get('kind') == 't2i')
    inv_set = (
        set() if draft_t1 else {
            nid for nid in chosen_ids
            if is_source_invariant_id(nid) and all(is_source_invariant_id(a) for a in graph_ancestors(dag_obj, nid))
        }
    )
    if session_row.get('mode') == 'instruction':
        turns = session_row.get('turns') or []
        idx = max(1, min(turn, len(turns))) if turns else 1
        cur_prompt = turns[idx - 1]['instruction'] if turns else ''
        nt = session_row.get('node_turn') or {}
        if idx == 1:
            t_nodes = [nid for nid in chosen_ids if nid not in inv_set and int(nt.get(nid) or 0) <= 1]
        else:
            t_nodes = [nid for nid in chosen_ids if nid not in inv_set and int(nt.get(nid) or 0) == idx]
        if not t_nodes:
            t_nodes = [nid for nid in chosen_ids if nid not in inv_set]
        req_map = {}
        for nid in chosen_ids:
            if nid in inv_set:
                req_map[nid] = ''
            else:
                n_turn = max(1, min(int(nt.get(nid) or 1), len(turns))) if turns else 1
                req_map[nid] = turns[n_turn - 1]['instruction'] if turns else ''
        out['user_prompt'] = cur_prompt
        out['turn_nodes'] = t_nodes or list(chosen_ids)
        out['source_invariants'] = [nid for nid in chosen_ids if nid in inv_set]
        out['node_turn_request'] = req_map
        out['draft_first_turn'] = draft_t1
    else:
        cur_prompt = turn_prompt(session_row, 1, items) if items else str(session_row.get('prompt') or '')
        out['user_prompt'] = cur_prompt
        out['turn_nodes'] = [nid for nid in chosen_ids if nid not in inv_set] or list(chosen_ids)
        out['source_invariants'] = [nid for nid in chosen_ids if nid in inv_set]
        out['node_turn_request'] = {nid: ('' if nid in inv_set else cur_prompt) for nid in chosen_ids}
        out['draft_first_turn'] = draft_t1
    return out


def upgrade_dense_graph(record, session_row, rng, checklist_k=(20, 26), mb_max_k=20):
    if session_row is None:
        return enrich_metadata(record, None)
    nodes, parents = table(session_row)
    turn = int(record.get('turn') or 1)
    existing = [n['node_id'] for n in record['graph']['nodes'] if n['node_id'] in nodes]
    chosen = None
    if session_row.get('mode') == 'instruction':
        future = {n for n, t in (session_row.get('node_turn') or {}).items() if t > turn}
        removed = frozenset(superseded_ids(session_row, turn)) | future
        edits = [n for n in edit_ids(session_row, turn) if n not in removed]
        inventory = [n for n in nodes if (session_row.get('node_turn') or {}).get(n) == 0 and n not in removed]
        target_size = min(max(len(closure_all(edits, parents)) + rng.randint(5, 10), len(existing)),
                          min(mb_max_k, len(edits) + len(inventory)))
        req = [n for n in existing if n not in removed]
        if len(closure_all(req, parents)) <= target_size:
            chosen = select_exact(nodes, parents, target_size, req, rng, exclude=removed)
        if chosen is None:
            chosen = select_exact(nodes, parents, target_size, edits, rng, exclude=removed)
    else:
        target_size = min(len(nodes), rng.randint(checklist_k[0], checklist_k[1]))
        if target_size > len(existing):
            rel_seeds = [n for n in nodes if nodes[n].get('node_type') == 'relation']
            rng.shuffle(rel_seeds)
            seed_req = []
            for r_id in rel_seeds[:2]:
                cand_req = seed_req + [r_id]
                if len(closure_all(cand_req, parents)) <= target_size:
                    seed_req = cand_req
            chosen = select_exact(nodes, parents, target_size, seed_req, rng)
            if chosen is None:
                chosen = select_exact(nodes, parents, target_size, [], rng)
    out = dict(record)
    if chosen and len(chosen) >= len(existing):
        order = topo_order(nodes, session_row['graph']['comp_edges'], session_row['graph']['supp_edges'])
        graph, _ = sanitize_graph(
            {n: dict(nodes[n], bbox=None) for n in order if n in chosen},
            session_row['graph']['comp_edges'],
            session_row['graph']['supp_edges'],
            session_row['graph']['int_edges'],
        )
        try:
            build_dag(graph)
            out['graph'] = graph
        except (ValueError, TypeError):
            pass
    return enrich_metadata(out, session_row)


def pick_stratified(pool, quotas, used_sids, rng):
    by_origin = {}
    for row in pool:
        if row['session_id'] in used_sids:
            continue
        by_origin.setdefault(row['origin'], []).append(row)
    for origin, items in by_origin.items():
        items.sort(key=lambda r: (
            -(1 if int(r.get('turn') or 1) >= 2 else 0),
            -len(r['graph']['nodes']),
            r['id'],
        ))
        top_half = items[:max(1, len(items) * 3 // 4)]
        rest = items[len(top_half):]
        rng.shuffle(top_half)
        rng.shuffle(rest)
        by_origin[origin] = top_half + rest
    picked = []
    for origin, want in quotas.items():
        avail = [r for r in by_origin.get(origin, []) if r['session_id'] not in used_sids]
        take_rows = avail[:want]
        picked.extend(take_rows)
        used_sids.update(r['session_id'] for r in take_rows)
    return picked


def resolve_img(row, base_dir):
    return (Path(base_dir) / row['image']).resolve()


def build_splits(final_dir, graphs_dir, sessions_dir, out_dir, seed=20261103):
    final_dir, out_dir = Path(final_dir).resolve(), Path(out_dir).resolve()
    sessions = load_graphs(graphs_dir, {'ok', 'repaired'}, sessions_dir=sessions_dir)

    eval_main = list(iter_jsonl(final_dir / 'eval_main.jsonl'))
    eval_edival = list(iter_jsonl(final_dir / 'eval_edival100.jsonl'))
    sft_rows = list(iter_jsonl(final_dir / 'controller_sft.jsonl'))
    rl_dev_orig = list(iter_jsonl(final_dir / 'rl_dev.jsonl'))
    rl_train_orig = list(iter_jsonl(final_dir / 'rl_train.jsonl'))
    rs_rows = list(iter_jsonl(final_dir / 'controller_rs_states.jsonl'))

    eval_sids = {r.get('session_id') or r['id'] for r in eval_main + eval_edival}
    eval_ids = {r['id'] for r in eval_main + eval_edival} | eval_sids
    eval_base_ids = {base_id_of(s) for s in eval_sids} | {str(r.get('edival_sid')) for r in eval_edival if 'edival_sid' in r}
    eval_img_sha1 = set()
    for r in eval_main + eval_edival:
        img_p = resolve_img(r, final_dir)
        if img_p.exists() and img_p.name != 'blank.png':
            eval_img_sha1.add(file_sha1(img_p))

    sft_ids = {r['id'] for r in sft_rows}
    sft_sids = {':'.join(r['id'].split(':')[1:-1]) for r in sft_rows if str(r.get('id', '')).startswith('state:')}
    sft_img_sha1 = set()
    for r in rs_rows:
        if r['id'] in sft_ids or r['session_id'] in sft_sids:
            img_p = resolve_img(r, final_dir)
            if img_p.exists():
                sft_img_sha1.add(file_sha1(img_p))

    orig_dev_sids = {r['session_id'] for r in rl_dev_orig}
    orig_train_sids = {r['session_id'] for r in rl_train_orig}
    orig_img_sha1 = set()
    for r in rl_dev_orig + rl_train_orig:
        img_p = resolve_img(r, final_dir)
        if img_p.exists():
            orig_img_sha1.add(file_sha1(img_p))

    forbidden_sids = eval_ids | sft_sids | orig_dev_sids | orig_train_sids
    forbidden_sha1 = eval_img_sha1 | sft_img_sha1 | orig_img_sha1

    free_rs = []
    seen_sha1 = set()
    for r in rs_rows:
        sid = r['session_id']
        if r['id'] in sft_ids or sid in forbidden_sids or base_id_of(sid) in eval_base_ids:
            continue
        s_row = sessions.get(sid)
        if s_row is not None and pool_of(s_row, 0.5) != 'train':
            continue
        img_p = resolve_img(r, final_dir)
        if not img_p.exists() or img_p.name == 'blank.png':
            continue
        digest = file_sha1(img_p)
        if digest in forbidden_sha1 or digest in seen_sha1:
            continue
        seen_sha1.add(digest)
        enriched = upgrade_dense_graph(
            dict(r, image=str(img_p)),
            s_row,
            random.Random(f'{seed}:dense:{r["id"]}'),
        )
        free_rs.append(enriched)

    used_sids = set(forbidden_sids)
    dev_extra = pick_stratified(
        free_rs,
        {'lgbench': 22, 'corebench': 15, 'longbench': 8, 'magicbrush': 25},
        used_sids,
        random.Random(f'{seed}:dev'),
    )
    train_extra = pick_stratified(
        free_rs,
        {'lgbench': 40, 'corebench': 25, 'longbench': 12, 'magicbrush': 28},
        used_sids,
        random.Random(f'{seed}:train'),
    )

    dev_rows = [
        upgrade_dense_graph(
            dict(r, image=str(resolve_img(r, final_dir))),
            sessions.get(r['session_id']),
            random.Random(f'{seed}:dense:{r["id"]}'),
        )
        for r in rl_dev_orig
    ] + dev_extra
    train_rows = [
        upgrade_dense_graph(
            dict(r, image=str(resolve_img(r, final_dir))),
            sessions.get(r['session_id']),
            random.Random(f'{seed}:dense:{r["id"]}'),
        )
        for r in rl_train_orig
    ] + train_extra

    out_dir.mkdir(parents=True, exist_ok=True)
    for rows in (dev_rows, train_rows):
        for r in rows:
            build_dag(r['graph'])
            r['image'] = relpath(r['image'], out_dir)
            for bad_key in ('source', 'target', 'mask', *STALE_KEYS):
                if bad_key in r:
                    raise ValueError(f'Disallowed or stale key {bad_key} present in split row')

    dev_sids = {r['session_id'] for r in dev_rows}
    train_sids = {r['session_id'] for r in train_rows}
    dev_sha1 = {file_sha1(resolve_img(r, out_dir)) for r in dev_rows}
    train_sha1 = {file_sha1(resolve_img(r, out_dir)) for r in train_rows}

    if dev_sids & train_sids:
        raise AssertionError('Dev and train session_ids overlap')
    if dev_sha1 & train_sha1:
        raise AssertionError('Dev and train image SHA1 digests overlap')
    if (dev_sids | train_sids) & (eval_ids | sft_sids):
        raise AssertionError('RL split session_ids overlap with eval or SFT')
    if ({base_id_of(s) for s in (dev_sids | train_sids)} & eval_base_ids):
        raise AssertionError('RL split base IDs overlap with eval')
    if (dev_sha1 | train_sha1) & (eval_img_sha1 | sft_img_sha1):
        raise AssertionError('RL split image SHA1 digests overlap with eval or SFT')
    if len(dev_rows) < 75:
        raise AssertionError(f'Expected at least 75 candidate dev rows for >=64 local frontier guarantee, got {len(dev_rows)}')

    dev_path = out_dir / 'rl_dev_v51.jsonl'
    train_path = out_dir / 'rl_train_v51.jsonl'
    write_jsonl(dev_path, dev_rows)
    write_jsonl(train_path, train_rows)

    manifest = {
        'seed': seed,
        'dev': {
            'path': str(dev_path),
            'rows': len(dev_rows),
            'pre_scored_rows': sum(1 for r in dev_rows if 'scores' in r),
            'by_origin': dict(Counter(r['origin'] for r in dev_rows)),
            'mean_k': round(sum(len(r['graph']['nodes']) for r in dev_rows) / len(dev_rows), 2),
            'k_ge_18': sum(1 for r in dev_rows if len(r['graph']['nodes']) >= 18),
            'sha256': file_sha256(dev_path),
        },
        'train': {
            'path': str(train_path),
            'rows': len(train_rows),
            'pre_scored_rows': sum(1 for r in train_rows if 'scores' in r),
            'by_origin': dict(Counter(r['origin'] for r in train_rows)),
            'mean_k': round(sum(len(r['graph']['nodes']) for r in train_rows) / len(train_rows), 2),
            'k_ge_18': sum(1 for r in train_rows if len(r['graph']['nodes']) >= 18),
            'sha256': file_sha256(train_path),
        },
        'disjointness_verified': {
            'dev_vs_train_sids': len(dev_sids & train_sids),
            'dev_vs_train_sha1': len(dev_sha1 & train_sha1),
            'rl_vs_eval_sids': len((dev_sids | train_sids) & eval_ids),
            'rl_vs_eval_base_ids': len({base_id_of(s) for s in (dev_sids | train_sids)} & eval_base_ids),
            'rl_vs_eval_sha1': len((dev_sha1 | train_sha1) & eval_img_sha1),
            'rl_vs_sft_sids': len((dev_sids | train_sids) & sft_sids),
            'rl_vs_sft_sha1': len((dev_sha1 | train_sha1) & sft_img_sha1),
        },
    }
    manifest_path = out_dir / 'rl_v51_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final_dir', default=str(FINAL))
    parser.add_argument('--graphs_dir', default=str(PROC / 'graphs'))
    parser.add_argument('--sessions_dir', default=str(PROC / 'sessions'))
    parser.add_argument('--out_dir', default=str(FINAL))
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args()
    manifest = build_splits(args.final_dir, args.graphs_dir, args.sessions_dir, args.out_dir, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
