"""Shared helpers for the ReDAG data pipeline.

Everything here is CPU-only and streams data so it stays inside the 2 GiB / 0.5-core
limits of AutoDL's no-GPU mode.  Graph validation reuses the repository's own
ConditionDAG / preserve_previous_conditions so prepared files match what the
training and evaluation scripts accept.
"""
import hashlib
import json
import os
import re
import sys
import time
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get('REDAG_DATA', REPO / 'data'))
RAW, PROC, FINAL = DATA / 'raw', DATA / 'processed', DATA / 'final'

NODE_TYPES = ('background', 'entity', 'attribute', 'relation', 'count', 'prohibition')
SCOPES = ('local', 'global', 'unknown')
EDGE_KEYS = ('comp_edges', 'supp_edges', 'int_edges')


def log(*parts):
    print(time.strftime('[%H:%M:%S]'), *parts, flush=True)


def iter_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path):
    return list(iter_jsonl(path))


def write_jsonl(path, rows):
    """Atomic write: readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    count = 0
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    tmp.replace(path)
    return count


def append_jsonl(path, row):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()


def done_ids(path, key='id'):
    """IDs already present in an (append-only) output file, for resumable jobs."""
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                ids.add(json.loads(line)[key])
            except (ValueError, KeyError, TypeError):
                continue  # tolerate a truncated last line after a crash
    return ids


def stable_unit(key, salt='redag-v2'):
    """Deterministic pseudo-random number in [0, 1) used for reproducible splits."""
    digest = hashlib.sha1(f'{salt}:{key}'.encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def relpath(path, start):
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def bbox_of(mask_rows_cols):
    """Normalized [x1, y1, x2, y2] of the True pixels of a 2D boolean numpy array."""
    import numpy as np
    ys, xs = np.nonzero(mask_rows_cols)
    if len(xs) == 0:
        return None
    h, w = mask_rows_cols.shape
    return [float(xs.min()) / w, float(ys.min()) / h, float(xs.max() + 1) / w, float(ys.max() + 1) / h]


def expand_box(box, scale=1.25, min_side=0.25):
    """Enlarge a normalized box around its center; keeps it inside [0, 1]."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half_w = max((box[2] - box[0]) * scale, min_side) / 2
    half_h = max((box[3] - box[1]) * scale, min_side) / 2
    x1, x2 = max(0.0, cx - half_w), min(1.0, cx + half_w)
    y1, y2 = max(0.0, cy - half_h), min(1.0, cy + half_h)
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


# ----------------------------------------------------------------------------- graphs

def _question(text):
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    return text if not text or text.endswith('?') else text + '?'


def clean_node(raw, default_scope='unknown'):
    """Coerce a loosely formatted node dict into ConditionNode keyword arguments."""
    node_type = str(raw.get('node_type', 'attribute')).strip().lower()
    node_type = node_type if node_type in NODE_TYPES else 'attribute'
    scope = str(raw.get('scope', default_scope)).strip().lower()
    scope = scope if scope in SCOPES else default_scope
    try:
        weight = float(raw.get('weight', 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    try:
        threshold = float(raw.get('threshold', 0.7))
    except (TypeError, ValueError):
        threshold = 0.7
    count = raw.get('expected_count')
    if node_type != 'count' or isinstance(count, bool):
        count = None
    elif count is not None:
        try:
            count = max(0, int(count))
        except (TypeError, ValueError):
            count = None
    deps = raw.get('depends_on') or []
    deps = [str(d) for d in deps if isinstance(d, (str, int))] if isinstance(deps, (list, tuple)) else []
    return {'node_id': str(raw['node_id']), 'node_type': node_type, 'question': _question(raw.get('question')),
            'entity_phrase': re.sub(r'\s+', ' ', str(raw.get('entity_phrase') or '')).strip(),
            'bbox': None, 'weight': weight if weight > 0 and weight == weight and weight < 1e6 else 1.0,
            'threshold': min(max(threshold, 0.0), 1.0), 'expected_count': count, 'scope': scope,
            'depends_on': list(dict.fromkeys(deps))}


def parents_of(nodes, comp=(), supp=()):
    parents = {nid: set(node.get('depends_on', [])) for nid, node in nodes.items()}
    for u, v in list(comp) + list(supp):
        if v in parents:
            parents[v].add(u)
    return parents


def _reaches(parents, start, goal):
    """True when `goal` is an ancestor-or-self of `start` following parent links."""
    stack, seen = [start], set()
    while stack:
        cur = stack.pop()
        if cur == goal:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(parents.get(cur, ()))
    return False


def sanitize_graph(nodes, comp=(), supp=(), inter=()):
    """Drop dangling/self/duplicate references and any dependency that would close a cycle.

    `nodes` is an ordered dict node_id -> cleaned node.  Returns (graph_dict, dropped_count).
    """
    dropped = 0
    ids = set(nodes)
    parents = {nid: set() for nid in nodes}
    for nid, node in nodes.items():
        kept = []
        for dep in node['depends_on']:
            if dep in ids and dep != nid and not _reaches(parents, dep, nid):
                kept.append(dep)
                parents[nid].add(dep)
            else:
                dropped += 1
        node['depends_on'] = kept
    edges = {}
    for key, raw_edges in (('comp_edges', comp), ('supp_edges', supp)):
        clean = []
        for edge in raw_edges or ():
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                dropped += 1
                continue
            u, v = str(edge[0]), str(edge[1])
            if u not in ids or v not in ids or u == v or [u, v] in clean or _reaches(parents, u, v):
                dropped += 1
                continue
            clean.append([u, v])
            parents[v].add(u)
        edges[key] = clean
    seen, clean = set(), []
    for edge in inter or ():
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            dropped += 1
            continue
        u, v = str(edge[0]), str(edge[1])
        if u not in ids or v not in ids or u == v or frozenset((u, v)) in seen:
            dropped += 1
            continue
        seen.add(frozenset((u, v)))
        clean.append([u, v])
    edges['int_edges'] = clean
    return {'nodes': list(nodes.values()), **edges}, dropped


def tighten_scopes(graph):
    """Downgrade 'local' claims the harness could not support (see redag.graph.dag.semantic_support)."""
    nodes = {n['node_id']: n for n in graph['nodes']}
    parents = parents_of(nodes, graph['comp_edges'], graph['supp_edges'])
    for nid, node in nodes.items():
        if node['node_type'] in ('background', 'count', 'prohibition') and node['scope'] == 'local':
            node['scope'] = 'global'
        if node['scope'] != 'local':
            continue
        if node['node_type'] == 'relation' and len(parents[nid]) < 2:
            node['scope'] = 'unknown'
        elif node['node_type'] == 'attribute' and not parents[nid]:
            node['scope'] = 'unknown'
        elif node['node_type'] == 'entity' and not node['entity_phrase']:
            node['scope'] = 'unknown'
    return graph


def build_dag(graph, previous=None, superseded=()):
    """Validate with the repository's own code path; raises ValueError/TypeError on bad graphs."""
    from redag.models.planner import preserve_previous_conditions
    return preserve_previous_conditions(graph, previous, list(superseded))


def topo_order(nodes, comp=(), supp=()):
    """Stable topological order (dependencies first, otherwise original order)."""
    parents = parents_of(nodes, comp, supp)
    order, placed = [], set()
    pending = list(nodes)
    while pending:
        progressed = False
        for nid in list(pending):
            if parents[nid] <= placed:
                order.append(nid)
                placed.add(nid)
                pending.remove(nid)
                progressed = True
        if not progressed:
            raise CycleError('dependency cycle', pending)
    return order


def closure(nid, parents):
    out, stack = set(), [nid]
    while stack:
        cur = stack.pop()
        if cur not in out:
            out.add(cur)
            stack.extend(parents.get(cur, ()))
    return out


def check_acyclic(nodes, comp=(), supp=()):
    TopologicalSorter({k: v for k, v in parents_of(nodes, comp, supp).items()}).prepare()
