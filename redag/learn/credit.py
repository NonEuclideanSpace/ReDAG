import torch
import json
import random
from pathlib import Path
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


def credit_experiment(args, records, harness, controller, editor, planner, result, folder):
    from redag.harness.environment import run_record
    from redag.learn.rl_trainer import ablation_reward
    from redag.verify.verifier import export_view
    rng, curves, audits, skipped = random.Random(args.seed), [], [], 0
    methods = ('terminal_broadcast', 'step_full_table', 'subgraph_credit_ours')
    for index, record in enumerate(records):
        _, _, trace = run_record(record, harness, controller, editor, args.max_steps, args.seed + index, 6, planner)
        for step_index, step in enumerate(trace):
            if len(step['candidate_group']) != 6:
                skipped += 1
                continue
            groups = {method: [] for method in methods}
            for candidate_index, candidate in enumerate(step['candidate_group']):
                before, delivered, graph = candidate['parent_state'], candidate['delivered_state'], candidate['dag']
                for method in methods:
                    groups[method].append(ablation_reward(method, before, delivered, graph, candidate['reward'], harness, controller, editor, args.max_steps))
                if rng.random() < 0.05:
                    identifier = f'audit_{index}_{step_index}_{candidate_index}'
                    audits.append({'source': 'evaluation_diagnostic', 'protection_lambda': harness.lam, 'before': export_view(before.image, graph, folder, identifier + '_before'), 'after': export_view(delivered.image, graph, folder, identifier + '_after'), 'affected': sorted(candidate['affected']), 'protected': sorted(before.protected_set)})
            for method, rewards in groups.items():
                values = torch.tensor(rewards)
                advantages = compute_pairwise_advantages(rewards) if method == 'subgraph_credit_ours' else (values - values.mean()) * 6 / 5
                curves.append({'session': index, 'step': step_index, 'method': method, 'rewards': rewards, 'advantages': advantages.tolist(), 'reward_variance': float(values.var(unbiased=False)), 'advantage_variance': float(advantages.var(unbiased=False))})
    result['diagnostic_reward_advantage_curves'] = curves
    result['diagnostic_incomplete_groups'] = skipped
    result.setdefault('full_audits', []).extend(audits)


def validate_logs(path):
    from redag.learn.sft_trainer import read_records
    paths = json.loads(Path(path).read_text())
    methods = {'terminal_broadcast', 'step_full_table', 'subgraph_credit_ours'}
    if set(paths) != methods:
        raise ValueError('Training logs must cover all three reward methods')
    files = {name: Path(path).parent / file for name, file in paths.items()}
    logs = {name: read_records(file) for name, file in files.items()}
    settings = []
    for name, rows in logs.items():
        required = {'round', 'step', 'stage', 'optimizer_updates', 'reward_variance', 'advantage_variance', 'dev_acceptance_p'}
        if not rows or any(row.get('method') != name or not required <= row.keys() for row in rows):
            raise ValueError('Training logs must contain actual method-specific training measurements')
        config = json.loads((files[name].parent / 'training_config.json').read_text())
        if config.pop('method') != name:
            raise ValueError('Training configuration method does not match its measurements')
        settings.append(config)
    signatures = [[(r['round'], r['stage'], r['step']) for r in rows] for rows in logs.values()]
    if any(s != signatures[0] for s in signatures[1:]) or any(s != settings[0] for s in settings[1:]):
        raise ValueError('Reward comparisons require the same configuration, seeds and training budgets')
    audits = []
    for file in files.values():
        audit_file = file.parent / 'training_audits.jsonl'
        if audit_file.exists() and audit_file.stat().st_size:
            audits.extend(dict(row, source='actual_training') for row in read_records(audit_file))
    return logs, audits
