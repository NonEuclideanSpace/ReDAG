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

def compute_affected_subgraph(dag: ConditionDAG, action: ControllerAction, target_node_id: str) -> Set[str]:
    window: BBox = action.box
    B_a: Set[str] = {target_node_id}
    for nid, node in dag.nodes.items():
        if node.bbox is not None and boxes_intersect(node.bbox, window):
            B_a.add(nid)
    tilde_B_a: Set[str] = B_a | get_comp_descendants(dag, B_a)
    int_nb: Set[str] = get_int_neighbors(dag, tilde_B_a)
    for nid in tilde_B_a:
        b_u = dag.nodes[nid].bbox
        if b_u is None:
            continue
        for other_id, other_node in dag.nodes.items():
            if other_node.bbox is not None and boxes_intersect(b_u, other_node.bbox):
                int_nb.add(other_id)
    touched_phrases = {dag.nodes[nid].entity_phrase for nid in tilde_B_a if dag.nodes[nid].entity_phrase}
    q_global: Set[str] = set()
    for nid, node in dag.nodes.items():
        if node.node_type in ('background', 'prohibition'):
            q_global.add(nid)
        elif node.node_type == 'count' and node.entity_phrase in touched_phrases:
            q_global.add(nid)
    return tilde_B_a | int_nb | q_global

def compute_active_frontier(dag: ConditionDAG, scores: Dict[str, float], gate_tau: float=0.5) -> List[str]:
    comp_parents: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for p, c in dag.comp_edges:
        if c in comp_parents:
            comp_parents[c].append(p)
    supp_parents: Dict[str, List[str]] = {nid: [] for nid in dag.nodes}
    for b, c in dag.supp_edges:
        if c in supp_parents:
            supp_parents[c].append(b)
    ready_candidates: List[str] = []
    for nid, node in dag.nodes.items():
        if scores.get(nid, 0.0) >= node.threshold:
            continue
        if any((scores.get(p, 0.0) < gate_tau for p in comp_parents[nid])):
            continue
        if any((scores.get(b, 0.0) < dag.nodes[b].threshold for b in supp_parents[nid])):
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
    parents = {nid: set() for nid in dag.nodes}
    for parent, child in dag.comp_edges:
        parents[child].add(parent)
    return list(TopologicalSorter(parents).static_order())
