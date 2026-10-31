from typing import Dict, List, Set
from redag.graph.schema import BBox, ConditionDAG, ControllerAction, FailureMode

def boxes_intersect(b1: BBox, b2: BBox) -> bool:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    return x2 > x1 and y2 > y1

def compute_box_iou(b1: BBox, b2: BBox) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max((b1[2] - b1[0]) * (b1[3] - b1[1]), 1e-08)
    area2 = max((b2[2] - b2[0]) * (b2[3] - b2[1]), 1e-08)
    return float(inter / (area1 + area2 - inter))

def get_comp_descendants(dag: ConditionDAG, seed_nodes: Set[str]) -> Set[str]:
    adj: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for u, v in dag.comp_edges:
        if u in adj:
            adj[u].append(v)
    visited: Set[str] = set()
    queue = list(seed_nodes)
    while queue:
        curr = queue.pop(0)
        for nxt in adj.get(curr, []):
            if nxt not in visited and nxt not in seed_nodes:
                visited.add(nxt)
                queue.append(nxt)
    return visited

def get_int_neighbors(dag: ConditionDAG, seed_nodes: Set[str]) -> Set[str]:
    neighbors: Set[str] = set()
    for u, v in dag.int_edges:
        if u in seed_nodes and v in dag.nodes:
            neighbors.add(v)
        if v in seed_nodes and u in dag.nodes:
            neighbors.add(u)
    return neighbors

def condition_dependencies(dag: ConditionDAG, node_id: str) -> Set[str]:
    return set(dag.nodes[node_id].depends_on) | {p for p, c in dag.comp_edges + dag.supp_edges if c == node_id}


def semantic_support(dag: ConditionDAG, node_id: str) -> BBox | None:
    pending, seen, boxes = [node_id], set(), []
    while pending:
        nid = pending.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = dag.nodes[nid]
        dependencies = condition_dependencies(dag, nid)
        if node.scope != 'local' or node.node_type in ('background', 'count', 'prohibition') or node.bbox is None:
            return None
        if node.node_type == 'relation' and len(dependencies) < 2:
            return None
        boxes.append(node.bbox)
        pending.extend(dependencies - seen)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def get_dependency_descendants(dag: ConditionDAG, seed_nodes: Set[str]) -> Set[str]:
    affected = set(seed_nodes)
    while True:
        expanded = affected | {nid for nid in dag.nodes if condition_dependencies(dag, nid) & affected}
        if expanded == affected:
            return affected - seed_nodes
        affected = expanded


CANVAS: BBox = (0.0, 0.0, 1.0, 1.0)
WHOLE_CANVAS_SHARE = 0.9
MIN_FOOTPRINT_OVERLAP = 0.10


def box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def footprint_overlap(footprint: BBox, box: BBox) -> float:
    width = min(footprint[2], box[2]) - max(footprint[0], box[0])
    height = min(footprint[3], box[3]) - max(footprint[1], box[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height / max(box_area(footprint), 1e-08)


def condition_parents(dag: ConditionDAG, node_id: str) -> List[str]:
    return list(dict.fromkeys([p for p, c in dag.comp_edges if c == node_id] + [p for p, c in dag.supp_edges if c == node_id]
                              + list(dag.nodes[node_id].depends_on)))


def edit_footprint(dag: ConditionDAG, node_id: str) -> BBox | None:
    node = dag.nodes[node_id]
    if node.node_type in ('background', 'prohibition'):
        return CANVAS
    if node.node_type not in ('entity', 'attribute'):
        return None
    if node.bbox is not None:
        return CANVAS if box_area(node.bbox) >= WHOLE_CANVAS_SHARE else node.bbox
    if node.node_type == 'entity':
        return None
    owners = [p for p in condition_parents(dag, node_id) if dag.nodes[p].node_type in ('entity', 'attribute')]
    if not owners:
        return CANVAS
    for owner in owners:
        footprint = edit_footprint(dag, owner)
        if footprint is not None:
            return footprint
    return None


def compute_spatial_subgraph(dag: ConditionDAG, action: ControllerAction, target_node_id: str) -> Set[str]:
    if target_node_id not in dag.nodes:
        raise ValueError('Target condition must exist')
    touched = {target_node_id}
    for nid in dag.nodes:
        footprint = edit_footprint(dag, nid)
        if footprint is not None and footprint_overlap(footprint, action.box) >= MIN_FOOTPRINT_OVERLAP:
            touched.add(nid)
    return touched


def compute_affected_subgraph(dag: ConditionDAG, action: ControllerAction, target_node_id: str, protected: Set[str] = frozenset()) -> Set[str]:
    touched = compute_spatial_subgraph(dag, action, target_node_id)
    affected = touched | get_dependency_descendants(dag, touched)
    neighbors = get_int_neighbors(dag, affected) - affected
    affected |= neighbors | get_dependency_descendants(dag, neighbors)
    phrases = {dag.nodes[nid].entity_phrase for nid in touched if dag.nodes[nid].node_type in ('entity', 'count')}
    return affected | {nid for nid, node in dag.nodes.items()
                       if (node.node_type == 'count' and node.entity_phrase in phrases)
                       or (node.node_type == 'prohibition' and nid in protected)}


def compute_active_frontier(dag: ConditionDAG, scores: Dict[str, float], gate_tau: float=0.5, gated: bool=True) -> List[str]:
    ready_candidates: List[str] = []
    for nid, node in dag.nodes.items():
        if scores.get(nid, 0.0) >= node.threshold:
            continue
        if gated:
            parents = {p for p, c in dag.comp_edges if c == nid} | set(node.depends_on)
            if any(scores.get(p, 0.0) < gate_tau for p in parents):
                continue
            if any(scores.get(b, 0.0) < dag.nodes[b].threshold for b, c in dag.supp_edges if c == nid):
                continue
        ready_candidates.append(nid)
    ready_candidates.sort(key=lambda nid: dag.nodes[nid].weight * (1.0 - scores.get(nid, 0.0)), reverse=True)
    return ready_candidates

def classify_failure_mode(dag: ConditionDAG, node_id: str, scores: Dict[str, float], detected_count: int=1) -> FailureMode:
    node = dag.nodes[node_id]
    if node.node_type == 'count' and node.expected_count is not None and (detected_count > node.expected_count):
        return 'over_count'
    if node.node_type == 'entity' and (node.bbox is None or scores.get(node_id, 0.0) < 0.35):
        return 'missing'
    if node.bbox is not None:
        for u, v in dag.int_edges:
            other_id = v if u == node_id else u if v == node_id else None
            if other_id and dag.nodes[other_id].bbox is not None:
                if compute_box_iou(node.bbox, dag.nodes[other_id].bbox) > 0.5:
                    return 'entangled'
    if node.node_type == 'relation':
        return 'relation_error'
    return 'attribute_error'

def topological_nodes(dag: ConditionDAG) -> List[str]:
    from graphlib import TopologicalSorter
    parents = {nid: condition_dependencies(dag, nid) for nid in dag.nodes}
    return list(TopologicalSorter(parents).static_order())
