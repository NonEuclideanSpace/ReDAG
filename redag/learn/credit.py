import torch
from typing import Dict, List, Set
from redag.graph.schema import ConditionDAG

def phi_asymmetric(delta_q: float, is_protected: bool, lam: float=1.5) -> float:
    if delta_q >= 0.0 or not is_protected:
        return float(delta_q)
    return float((1.0 + lam) * delta_q)

def compute_subgraph_reward(dag: ConditionDAG, prev_scores: Dict[str, float], delivered_scores: Dict[str, float], protected_set: Set[str], affected_subgraph: Set[str], step_cost: float=0.01, lam: float=1.5) -> float:
    Z = dag.total_weight_Z
    assert Z > 0.0, 'Global weight sum Z must be strictly positive.'
    total_utility = 0.0
    for v in affected_subgraph:
        node = dag.nodes[v]
        dq = delivered_scores[v] - prev_scores[v]
        util_v = phi_asymmetric(dq, is_protected=v in protected_set, lam=lam)
        total_utility += node.weight / Z * util_v
    return float(total_utility - step_cost)

def compute_pairwise_advantages(rewards: List[float], delta_noise: float=0.015, s_0: float=0.08, a_max: float=2.0) -> torch.Tensor:
    if s_0 <= 0 or delta_noise < 0 or a_max <= 0:
        raise ValueError("Invalid pairwise advantage scales")
    G = len(rewards)
    assert G >= 2, 'Group size G must be at least 2.'
    r = torch.tensor(rewards, dtype=torch.float32)
    if not torch.isfinite(r).all():
        raise ValueError("Rewards must be finite")
    diff = r.unsqueeze(1) - r.unsqueeze(0)
    omega = (torch.abs(diff) > delta_noise).float()
    omega.fill_diagonal_(0.0)
    clipped = torch.clamp(diff / s_0, min=-a_max, max=a_max)
    advantages = (omega * clipped).sum(dim=1) / float(G - 1)
    assert torch.abs(advantages.sum()).item() < 1e-05, f'Zero-sum violated: {advantages.sum().item()}'
    return advantages
