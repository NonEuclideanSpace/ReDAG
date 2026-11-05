import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from redag.config import ReDAGConfig
from redag.graph.dag import box_area
from redag.graph.schema import CanvasState, ConditionDAG
from redag.harness.environment import (
    GLOBAL_ROUTE_TYPES, add_runtime_arguments, consistent_count, graph_ancestors, load_runtime,
    unlocked_satisfied_descendants,
)
from redag.learn.credit import compute_pairwise_advantages, compute_subgraph_reward
from redag.learn.rl_trainer import compute_controller_rl_loss, relative_lora_displacement
from redag.learn.sft_trainer import read_records
from redag.models.planner import dag_from_dict, dag_to_dict
from redag.verify.grounder import load_image

V51_DEFAULTS = dict(
    editor_context_pad=0.25,
    num_denoise_steps=28,
    protect_masks=False,
    closure_instructions=False,
    confirm_drops=False,
    soft_gating=False,
    preserve_instructions=False,
    dag_box=False,
    edge_policy='full',
    scope_routing=True,
    global_types=','.join(GLOBAL_ROUTE_TYPES),
    global_gate='net2',
    local_gate='net2',
    global_retries=1,
    user_text=True,
    turn_pass=True,
    turn_gate='net2',
    turn_retries=0,
    turn_priority=True,
    turn_localize=True,
    turn_protect=True,
    clean_instruction=True,
    single_stage_verify=True,
    focused_turn=True,
    smart_turn_pass=True,
    cascade_priority=True,
    repair_gate=True,
    monotonic_turn=True,
    candidate_k=3,
)


@dataclass
class PreparedParent:
    record_id: str
    session_id: str
    origin: str
    turn: int
    image_path: str
    dag: ConditionDAG
    scores: dict
    grounding: object
    user_prompt: str
    node_turn_request: dict
    turn_nodes: set
    source_invariants: set
    draft_first_turn: bool
    target_node_id: str
    fallback_global: bool


def select_v51_local_target(harness, dag, scores, turn_nodes, source_invariants, draft_first_turn, allow_global_fallback=True):
    harness.mode = 'redag_full'
    harness.draft_first_turn = bool(draft_first_turn)
    harness.source_invariants = set(source_invariants) if source_invariants is not None else set()
    harness.turn_nodes = set(turn_nodes) if turn_nodes is not None else set(dag.nodes)
    harness.global_failures, harness.local_failures = {}, {}
    frontier = harness.active_frontier(dag, scores)
    if harness.focused_turn and not harness.draft_first_turn:
        inv = harness.active_source_invariants(dag)
        if inv:
            eff_g = harness.effective_graph(dag)
            frontier_set = set(frontier)
            blocked_edits = [
                u for u in dag.nodes
                if u not in inv and scores.get(u, 0.0) < dag.nodes[u].threshold and u not in frontier_set
            ]
            unblocking = (
                set().union(*(graph_ancestors(eff_g, u) for u in blocked_edits))
                if blocked_edits else set()
            )
            frontier = [nid for nid in frontier if nid not in inv or nid in unblocking]
    if not frontier:
        return None, False
    eff_dag = harness.effective_graph(dag)

    def sort_key(nid):
        return (
            -(unlocked_satisfied_descendants(eff_dag, scores, nid) if harness.cascade_priority else 0),
            harness.turn_priority and nid not in harness.turn_nodes,
            -dag.nodes[nid].weight * (1.0 - scores.get(nid, 0.0)),
        )

    local_fr = [nid for nid in frontier if not harness.routes_globally(dag, nid)]
    if local_fr:
        local_fr.sort(key=sort_key)
        return local_fr[0], False
    if allow_global_fallback and harness.global_retries > 0:
        rel_fr = [nid for nid in frontier if dag.nodes[nid].node_type == 'relation']
        if rel_fr:
            rel_fr.sort(key=sort_key)
            return rel_fr[0], True
    return None, False


def configure_parent_context(harness, parent):
    harness.mode = 'redag_full'
    harness.turn_request = str(parent.user_prompt or '')
    harness.node_turn_request = dict(parent.node_turn_request) if parent.node_turn_request else {}
    harness.turn_nodes = set(parent.turn_nodes)
    harness.source_invariants = set(parent.source_invariants)
    harness.draft_first_turn = bool(parent.draft_first_turn)
    harness.turn_index = max(0, int(parent.turn) - 1)
    harness.turn_count = max(1, int(parent.turn))
    harness.steps_remaining = 0
    harness.repair_queue, harness.repaired_in_turn = [], set()
    harness.unknown_nodes = set()
    harness.global_failures = {(harness.turn_index, parent.target_node_id): harness.global_retries} if parent.fallback_global else {}
    harness.local_failures = {}
    if harness.grounder is not None and parent.grounding is not None:
        harness.restore_grounding(parent.grounding)


def cpu_grounding_snapshot(harness):
    if harness.grounder is None:
        return None
    counts = dict(harness.grounder.counts)
    masks = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in harness.grounder.masks.items()}
    return (counts, masks)


def infer_draft_first_turn(rec):
    if 'draft_first_turn' in rec:
        return bool(rec['draft_first_turn'])
    if 'kind' in rec:
        return bool(rec['kind'] == 't2i')
    return bool(rec.get('origin') != 'magicbrush')


@torch.no_grad()
def prepare_parents(records, harness, cache_path=None, allow_global_fallback=True):
    mode_tag = 'single' if harness.single_stage_verify else 'two_stage'
    cached_by_id, cached_order = {}, []
    if cache_path is not None and Path(cache_path).exists():
        for line in Path(cache_path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                rid = row['id']
                if rid not in cached_by_id:
                    cached_order.append(rid)
                cached_by_id[rid] = row
    merged_cache = dict(cached_by_id)
    merged_order = list(cached_order)
    parents, updated_rows, dirty, last_flushed = [], [], False, 0

    for idx, rec in enumerate(records, 1):
        rid = rec['id']
        if rid not in merged_cache:
            merged_order.append(rid)
        cached = cached_by_id.get(rid)
        img_tensor = rec['image'] if torch.is_tensor(rec['image']) else load_image(rec['image'])
        img_path = '' if torch.is_tensor(rec['image']) else str(rec['image'])
        if 'scores' in rec and rec.get('verifier_mode') == mode_tag:
            dag = dag_from_dict(rec.get('graph', rec.get('dag')))
            if harness.grounder is not None:
                dag = harness.grounder.ground(img_tensor, dag)
            scores = {k: float(v) for k, v in rec['scores'].items()}
            row_to_save = dict(rec, graph=json.loads(json.dumps(dag_to_dict(dag))), scores=scores, verifier_mode=mode_tag)
            if torch.is_tensor(row_to_save.get('image')):
                row_to_save.pop('image', None)
        elif cached is not None and 'scores' in cached and cached.get('verifier_mode') == mode_tag:
            dag = dag_from_dict(cached['graph'])
            if harness.grounder is not None:
                dag = harness.grounder.ground(img_tensor, dag)
            scores = {k: float(v) for k, v in cached['scores'].items()}
            row_to_save = cached
        else:
            state, dag = harness.initialize(img_tensor, dag_from_dict(rec.get('graph', rec.get('dag'))))
            scores = {k: float(v) for k, v in state.scores.items()}
            row_to_save = dict(rec, graph=json.loads(json.dumps(dag_to_dict(dag))), scores=scores, verifier_mode=mode_tag)
            if torch.is_tensor(row_to_save.get('image')):
                row_to_save.pop('image', None)
            dirty = True
        harness.verifier.records.clear()
        grounding = cpu_grounding_snapshot(harness)
        all_ids = list(dag.nodes)
        turn_nodes = rec.get('turn_nodes') or all_ids
        src_inv = rec.get('source_invariants') if rec.get('source_invariants') is not None else []
        draft_t1 = infer_draft_first_turn(rec)
        req_map = dict(rec['node_turn_request']) if isinstance(rec.get('node_turn_request'), dict) else {}
        target, fallback = select_v51_local_target(
            harness, dag, scores, turn_nodes, src_inv, draft_t1, allow_global_fallback=allow_global_fallback,
        )
        updated_rows.append(row_to_save)
        merged_cache[rid] = row_to_save
        if target is not None:
            parents.append(PreparedParent(
                record_id=rid,
                session_id=str(rec.get('session_id') or rid),
                origin=str(rec.get('origin') or 'unknown'),
                turn=int(rec.get('turn') or 1),
                image_path=img_path,
                dag=dag,
                scores=scores,
                grounding=grounding,
                user_prompt=str(rec.get('user_prompt') or ''),
                node_turn_request=req_map,
                turn_nodes=set(turn_nodes),
                source_invariants=set(src_inv),
                draft_first_turn=draft_t1,
                target_node_id=target,
                fallback_global=fallback,
            ))
        if cache_path is not None and dirty and idx % 20 == 0:
            save_jsonl_atomic(cache_path, [merged_cache[k] for k in merged_order if k in merged_cache])
            last_flushed = idx
            dirty = False
    if cache_path is not None and (dirty or last_flushed != len(records) or not Path(cache_path).exists()):
        save_jsonl_atomic(cache_path, updated_rows)
    return parents


def save_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            clean = {k: v for k, v in row.items() if not torch.is_tensor(v)}
            stream.write(json.dumps(clean, ensure_ascii=False) + '\n')
    tmp.replace(path)


def parent_canvas_state(parent, inline_image=None):
    image = inline_image if inline_image is not None else load_image(parent.image_path)
    scores = dict(parent.scores)
    protected = {nid for nid, q in scores.items() if q >= parent.dag.nodes[nid].threshold}
    return CanvasState(image, scores, protected, 0)


def compute_v51_step_reward(harness, parent_state, dag, cand):
    eff = harness.effective_graph(dag)
    before_con = consistent_count(eff, parent_state.scores)
    after_con = consistent_count(eff, cand['delivered_state'].scores)
    con_gain = after_con - before_con
    dense = compute_subgraph_reward(
        eff, parent_state.scores, cand['raw_scores'], parent_state.protected_set,
        cand['affected'], harness.step_cost, harness.lam,
    )
    sample = cand.get('sample') or {}
    raw_box = sample.get('controller_box', cand['action'].box)
    box_pen = 0.03 if box_area(raw_box) >= 0.92 else 0.0
    fmt_pen = 0.03 if sample.get('instruction_repaired', False) else 0.0
    reward = dense + 0.10 * float(cand['accepted']) + 0.08 * float(con_gain) - box_pen - fmt_pen
    return float(reward), int(con_gain)


def detach_sample_to_cpu(sample):
    return {
        'inputs': {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sample['inputs'].items()},
        'token_ids': sample['token_ids'].detach().cpu(),
        'action_mask': sample['action_mask'].detach().cpu(),
        'temperature': float(sample.get('temperature', 1.0)),
        'instruction_repaired': bool(sample.get('instruction_repaired', False)),
    }


@torch.no_grad()
def rollout_parent_group(harness, controller, editor, parent, group_size, noise_seed, inline_image=None):
    configure_parent_context(harness, parent)
    state = parent_canvas_state(parent, inline_image=inline_image)
    saved_cf, saved_bonus = harness.counterfactual_credit, harness.acceptance_bonus
    harness.counterfactual_credit = True
    harness.acceptance_bonus = 0.0
    try:
        cands = harness.sample_group(
            state, parent.dag, controller, editor, parent.target_node_id, group_size=group_size, seed=noise_seed,
        )
    finally:
        harness.counterfactual_credit = saved_cf
        harness.acceptance_bonus = saved_bonus
        harness.verifier.records.clear()
        harness.last_candidate = None
        harness.global_failures, harness.local_failures = {}, {}
    if len(cands) < 2:
        return None
    items, rewards, con_gains, accepts, areas, ops, repairs = [], [], [], [], [], {}, []
    for cand in cands:
        sample = cand.get('sample') or {}
        if 'token_ids' not in sample:
            raise RuntimeError(f'Expected local controller sample with token_ids for {parent.record_id}:{parent.target_node_id}')
        rew, c_gain = compute_v51_step_reward(harness, state, parent.dag, cand)
        raw_box = sample.get('controller_box', cand['action'].box)
        items.append(detach_sample_to_cpu(sample))
        rewards.append(rew)
        con_gains.append(c_gain)
        accepts.append(int(cand['accepted']))
        areas.append(box_area(raw_box))
        ops[cand['action'].op] = ops.get(cand['action'].op, 0) + 1
        repairs.append(int(bool(sample.get('instruction_repaired', False))))
    return {
        'items': items,
        'rewards': rewards,
        'con_gains': con_gains,
        'accepts': accepts,
        'areas': areas,
        'ops': ops,
        'repairs': repairs,
    }


def optimize_ctrl_groups(groups, controller, optimizer, ctrl_old_snap, ctrl_ref_snap, cfg):
    params = controller.lora_parameters()
    total_tokens = sum(int(item['action_mask'].sum()) for items, _ in groups for item in items)
    active = total_tokens > 0 and any(bool(adv.abs().any()) for _, adv in groups)
    if not active:
        return {
            'optimizer_updates': 0, 'updated': False, 'loss': 0.0,
            'max_old_policy_kl': 0.0, 'mean_sft_kl': 0.0, 'kl_early_stop': False, 'actual_action_tokens': total_tokens,
        }
    flat_items = [item for items, _ in groups for item in items]
    current_snap = controller.snapshot()
    with torch.no_grad():
        try:
            controller.load_snapshot(ctrl_old_snap)
            old_logits = [controller.forward_sample(item, action_only=True).detach().cpu() for item in flat_items]
            controller.load_snapshot(ctrl_ref_snap)
            ref_logits = [controller.forward_sample(item, action_only=True).detach().cpu() for item in flat_items]
        finally:
            controller.load_snapshot(current_snap)
    updates, last_loss, max_kl, sft_kl_weighted, stopped = 0, 0.0, 0.0, 0.0, False
    for _ in range(cfg.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, epoch_sft_kl, cursor = 0.0, 0.0, 0
        for items, adv in groups:
            for index, item in enumerate(items):
                new = controller.forward_sample(item, action_only=True)
                old = old_logits[cursor].to(new.device)
                ref = ref_logits[cursor].to(new.device)
                cursor += 1
                length = int(item['action_mask'].sum())
                ids = item['token_ids'][:, -(length + 1):].to(new.device)
                mask = torch.ones((1, length), device=new.device)
                loss, metrics = compute_controller_rl_loss(
                    new, old, ref, ids, mask, adv[index:index + 1], cfg.clip_eps_ctrl, cfg.beta_kl_ctrl,
                )
                scaled = loss * length / total_tokens
                policy_kl = metrics['ctrl_policy_kl']
                max_kl = max(max_kl, policy_kl)
                if metrics['ratio_guard_hit'] or policy_kl > cfg.target_kl_ctrl:
                    stopped = True
                    break
                if not torch.isfinite(scaled):
                    raise FloatingPointError('Non-finite controller RL loss')
                scaled.backward()
                epoch_loss += float(scaled.detach().item())
                epoch_sft_kl += float(metrics['ctrl_kl_sft']) * length / total_tokens
            if stopped:
                break
        if stopped:
            optimizer.zero_grad(set_to_none=True)
            break
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        last_loss = epoch_loss
        sft_kl_weighted = epoch_sft_kl
    return {
        'optimizer_updates': updates,
        'updated': updates > 0,
        'loss': last_loss,
        'max_old_policy_kl': max_kl,
        'mean_sft_kl': sft_kl_weighted,
        'kl_early_stop': stopped,
        'actual_action_tokens': total_tokens,
    }


@torch.no_grad()
def evaluate_dev_v51(parents, harness, controller, editor, ctrl_ref_snap, seed=42, candidate_k=1, inline_images=None):
    if not parents:
        raise ValueError('Dev parents cannot be empty')
    saved_k = harness.candidate_k
    current_snap = controller.snapshot()
    harness.candidate_k = max(1, int(candidate_k))
    accepted_sum, con_gain_sum, target_gain_sum, repair_sum, full_box_sum = 0, 0, 0.0, 0, 0
    detached_samples = []
    kl_num, tok_den = 0.0, 0
    try:
        for idx, parent in enumerate(parents):
            configure_parent_context(harness, parent)
            img = inline_images.get(parent.record_id) if inline_images else None
            state = parent_canvas_state(parent, inline_image=img)
            cands = harness.sample_group(
                state, parent.dag, controller, editor, parent.target_node_id, group_size=1, seed=seed + idx,
            )
            harness.verifier.records.clear()
            harness.last_candidate = None
            harness.global_failures, harness.local_failures = {}, {}
            if not cands:
                continue
            cand = cands[0]
            sample = cand.get('sample') or {}
            eff = harness.effective_graph(parent.dag)
            before_con = consistent_count(eff, state.scores)
            after_con = consistent_count(eff, cand['delivered_state'].scores)
            accepted_sum += int(cand['accepted'])
            con_gain_sum += int(after_con - before_con)
            target_gain_sum += float(cand['raw_scores'][parent.target_node_id] - state.scores[parent.target_node_id])
            repair_sum += int(bool(sample.get('instruction_repaired', False)))
            raw_box = sample.get('controller_box', cand['action'].box)
            full_box_sum += int(box_area(raw_box) >= 0.92)
            if 'token_ids' in sample:
                detached_samples.append(detach_sample_to_cpu(sample))
        chunk_size = 16
        for start in range(0, len(detached_samples), chunk_size):
            chunk = detached_samples[start:start + chunk_size]
            controller.load_snapshot(current_snap)
            new_chunk = [controller.forward_sample(s, action_only=True).detach().cpu() for s in chunk]
            controller.load_snapshot(ctrl_ref_snap)
            ref_chunk = [controller.forward_sample(s, action_only=True).detach().cpu() for s in chunk]
            for s, new_l, ref_l in zip(chunk, new_chunk, ref_chunk):
                length = int(s['action_mask'].sum())
                ids = s['token_ids'][:, -(length + 1):]
                _, m = compute_controller_rl_loss(
                    new_l, new_l, ref_l, ids, torch.ones((1, length)), torch.zeros(1),
                )
                kl_num += float(m['ctrl_kl_sft']) * length
                tok_den += length
    finally:
        controller.load_snapshot(current_snap)
        harness.candidate_k = saved_k
    n = len(parents)
    return {
        'candidate_k': int(candidate_k),
        'dev_states': n,
        'dev_acceptance_p': accepted_sum / n,
        'dev_accepted_count': accepted_sum,
        'dev_mean_consistent_gain': con_gain_sum / n,
        'dev_total_consistent_gain': con_gain_sum,
        'dev_mean_raw_target_gain': target_gain_sum / n,
        'dev_repair_rate': repair_sum / n,
        'dev_full_box_share': full_box_sum / n,
        'ctrl_kl_sft': kl_num / max(tok_den, 1),
    }


def train_ctrl_rl_v51(
    harness, controller, editor, train_parents, dev_parents, output,
    max_steps=40, batch_size=4, group_size=4, eval_every=8, final_k=3,
    lr=1.0e-5, ppo_epochs=2, clip_eps=0.2, beta_kl=0.04, target_kl=0.05,
    min_dev_states=64, seed=42, inline_images=None, cli_args=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if max_steps < 1:
        raise ValueError(f'max_steps must be >= 1, got {max_steps}')
    if harness.strategy is not None:
        raise ValueError('Controller RL requires harness.strategy to be None during rollout')
    if len(dev_parents) < min_dev_states:
        raise ValueError(f'Hard dev gate requires >= {min_dev_states} local-frontier dev states, got {len(dev_parents)}')
    if len(train_parents) < batch_size:
        raise ValueError(f'Need at least {batch_size} train parents, got {len(train_parents)}')
    train_sids = {p.session_id for p in train_parents}
    dev_sids = {p.session_id for p in dev_parents}
    if train_sids & dev_sids:
        raise ValueError('Train and dev session_ids must be strictly disjoint')

    ctrl_params = controller.lora_parameters()
    if not ctrl_params:
        raise ValueError('Controller requires trainable LoRA adapter parameters')
    for p in ctrl_params:
        p.requires_grad_(True)
    for p in editor.lora_parameters():
        p.requires_grad_(False)
    editor.eval()

    cfg = ReDAGConfig(
        lr_rl_controller=lr,
        ppo_epochs=ppo_epochs,
        clip_eps_ctrl=clip_eps,
        beta_kl_ctrl=beta_kl,
        target_kl_ctrl=target_kl,
    )
    ctrl_ref = controller.snapshot()
    controller.save_adapter(output / 'ctrl_sft_init')

    baseline_k1 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=1, inline_images=inline_images)
    baseline_k3 = (
        evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=final_k, inline_images=inline_images)
        if final_k > 1 else dict(baseline_k1)
    )
    (output / 'baseline_dev.json').write_text(json.dumps({'k1': baseline_k1, 'k_final': baseline_k3}, indent=2))

    optimizer = torch.optim.AdamW(ctrl_params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order = []
    metrics_path = output / 'metrics.jsonl'
    metrics_path.write_text('', encoding='utf-8')
    candidates_pool = []
    history = []

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        if len(order) < batch_size:
            extra = list(range(len(train_parents)))
            rng.shuffle(extra)
            order.extend(extra)
        batch_indices, order = order[:batch_size], order[batch_size:]
        old_snap = controller.snapshot()
        groups, rewards_all, adv_all, con_all, acc_all, area_all, ops_all, rep_all = [], [], [], [], [], [], {}, []
        for p_idx, idx in enumerate(batch_indices):
            parent = train_parents[idx]
            noise_seed = seed + 1000 * step + 17 * p_idx
            img = inline_images.get(parent.record_id) if inline_images else None
            rolled = rollout_parent_group(harness, controller, editor, parent, group_size, noise_seed, inline_image=img)
            if rolled is None:
                continue
            adv = compute_pairwise_advantages(rolled['rewards'], cfg.delta_noise, cfg.reward_scale_s0, cfg.adv_clip_max)
            groups.append((rolled['items'], adv))
            rewards_all.extend(rolled['rewards'])
            adv_all.extend(adv.tolist())
            con_all.extend(rolled['con_gains'])
            acc_all.extend(rolled['accepts'])
            area_all.extend(rolled['areas'])
            rep_all.extend(rolled['repairs'])
            for k_op, cnt in rolled['ops'].items():
                ops_all[k_op] = ops_all.get(k_op, 0) + cnt

        opt_metrics = optimize_ctrl_groups(groups, controller, optimizer, old_snap, ctrl_ref, cfg)
        row = {
            'step': step,
            'stage': 'ctrl_rl_v51',
            'sampled_parents': len(groups),
            'sampled_candidates': len(rewards_all),
            'mean_reward': sum(rewards_all) / max(len(rewards_all), 1),
            'reward_variance': float(torch.tensor(rewards_all).var(unbiased=False)) if rewards_all else 0.0,
            'advantage_variance': float(torch.tensor(adv_all).var(unbiased=False)) if adv_all else 0.0,
            'train_acceptance_p': sum(acc_all) / max(len(acc_all), 1),
            'train_mean_consistent_gain': sum(con_all) / max(len(con_all), 1),
            'action_box_area_mean': sum(area_all) / max(len(area_all), 1),
            'action_full_box_share': sum(a >= 0.92 for a in area_all) / max(len(area_all), 1),
            'instruction_repair_rate': sum(rep_all) / max(len(rep_all), 1),
            'action_ops': ops_all,
            'ctrl_lora_relative_displacement': relative_lora_displacement(ctrl_params, list(ctrl_ref.values())),
            **opt_metrics,
        }
        if step % eval_every == 0 or step == max_steps:
            dev_k1 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=1, inline_images=inline_images)
            row.update({
                'dev_acceptance_p': dev_k1['dev_acceptance_p'],
                'dev_acceptance_gain': dev_k1['dev_acceptance_p'] - baseline_k1['dev_acceptance_p'],
                'dev_mean_consistent_gain': dev_k1['dev_mean_consistent_gain'],
                'dev_consistent_delta_vs_sft': dev_k1['dev_mean_consistent_gain'] - baseline_k1['dev_mean_consistent_gain'],
                'dev_repair_rate': dev_k1['dev_repair_rate'],
                'dev_full_box_share': dev_k1['dev_full_box_share'],
                'dev_ctrl_kl_sft': dev_k1['ctrl_kl_sft'],
            })
            ckpt_dir = output / f'ckpt_step_{step:03d}'
            controller.save_adapter(ckpt_dir)
            candidates_pool.append((step, dev_k1, controller.snapshot(), str(ckpt_dir)))
        row['step_wall_seconds'] = time.perf_counter() - t0
        with metrics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        history.append(row)
        print(json.dumps({k: row[k] for k in row if k != 'action_ops'}), flush=True)

    candidates_pool.sort(
        key=lambda item: (
            item[1]['dev_mean_consistent_gain'],
            item[1]['dev_acceptance_p'],
            -item[1]['ctrl_kl_sft'],
        ),
        reverse=True,
    )
    top_to_verify = candidates_pool[:2]
    verified_k3 = []
    for step_id, dev_k1, snap, ckpt_dir in top_to_verify:
        controller.load_snapshot(snap)
        dev_k3 = (
            evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=final_k, inline_images=inline_images)
            if final_k > 1 else dict(dev_k1)
        )
        verified_k3.append((step_id, dev_k1, dev_k3, snap, ckpt_dir))

    verified_k3.sort(
        key=lambda item: (
            item[2]['dev_mean_consistent_gain'],
            item[2]['dev_acceptance_p'],
            item[1]['dev_mean_consistent_gain'],
            -item[2]['ctrl_kl_sft'],
        ),
        reverse=True,
    )
    best_step, best_k1, best_k3, best_snap, _ = verified_k3[0]
    gate_passed = bool(
        best_k3['dev_mean_consistent_gain'] > baseline_k3['dev_mean_consistent_gain']
        and best_k3['dev_acceptance_p'] >= baseline_k3['dev_acceptance_p']
        and (
            best_k3['dev_acceptance_p'] > baseline_k3['dev_acceptance_p']
            or best_k1['dev_acceptance_p'] > baseline_k1['dev_acceptance_p']
        )
        and best_k3['dev_repair_rate'] <= baseline_k3['dev_repair_rate'] + 0.05
        and best_k3['dev_full_box_share'] <= 0.25
    )
    if gate_passed:
        controller.load_snapshot(best_snap)
        controller.save_adapter(output / 'ctrl_best')
    else:
        controller.load_snapshot(ctrl_ref)

    summary = {
        'gate_passed': gate_passed,
        'selected_step': best_step if gate_passed else 0,
        'dev_states': len(dev_parents),
        'train_states': len(train_parents),
        'baseline_k1': baseline_k1,
        'baseline_k_final': baseline_k3,
        'best_k1': best_k1,
        'best_k_final': best_k3,
        'gain_k_final_consistent': best_k3['dev_mean_consistent_gain'] - baseline_k3['dev_mean_consistent_gain'],
        'gain_k_final_acceptance_p': best_k3['dev_acceptance_p'] - baseline_k3['dev_acceptance_p'],
        'gain_k1_consistent': best_k1['dev_mean_consistent_gain'] - baseline_k1['dev_mean_consistent_gain'],
        'gain_k1_acceptance_p': best_k1['dev_acceptance_p'] - baseline_k1['dev_acceptance_p'],
        'cli_args': cli_args or {},
    }
    (output / 'rl_v51_summary.json').write_text(json.dumps(summary, indent=2))
    return summary, history


def cache_fingerprint(data_path, harness, args):
    salt = json.dumps({
        'single_stage_verify': bool(harness.single_stage_verify),
        'edge_policy': str(harness.edge_policy),
        'protect_masks': bool(harness.protect_masks),
        'soft_gating': bool(harness.soft_gating),
        'verifier_model': str(args.verifier_model),
        'detector_model': str(args.detector_model),
        'sam_model': str(args.sam_model),
    }, sort_keys=True).encode('utf-8')
    h = hashlib.sha1()
    h.update(salt)
    h.update(Path(data_path).read_bytes())
    return h.hexdigest()[:16]


def assert_v51_alignment(harness, controller, editor=None):
    required = {
        'clean_instruction': bool(getattr(controller, 'clean_instruction', False)),
        'single_stage_verify': bool(harness.single_stage_verify),
        'scope_routing': bool(harness.scope_routing),
        'user_text': bool(harness.user_text),
        'focused_turn': bool(harness.focused_turn),
        'smart_turn_pass': bool(harness.smart_turn_pass),
        'cascade_priority': bool(harness.cascade_priority),
        'turn_pass': bool(harness.turn_pass),
        'turn_priority': bool(harness.turn_priority),
        'turn_localize': bool(harness.turn_localize),
        'turn_protect': bool(harness.turn_protect),
        'repair_gate': bool(harness.repair_gate),
        'monotonic_turn': bool(harness.monotonic_turn),
        'local_gate_net2': harness.local_gate == 'net2',
        'global_gate_net2': harness.global_gate == 'net2',
        'turn_gate_net2': harness.turn_gate == 'net2',
        'global_retries_1': int(harness.global_retries) == 1,
        'turn_retries_0': int(harness.turn_retries) == 0,
        'edge_policy_full': harness.edge_policy == 'full',
        'no_protect_masks': not bool(harness.protect_masks),
        'no_closure_instructions': not bool(harness.closure_instructions),
        'no_confirm_drops': not bool(harness.confirm_drops),
        'no_soft_gating': not bool(harness.soft_gating),
        'no_preserve_instructions': not bool(harness.preserve_instructions),
        'no_dag_box': not bool(harness.dag_box),
    }
    if editor is not None and hasattr(editor, 'context_pad'):
        required['editor_context_pad_0.25'] = abs(float(editor.context_pad) - 0.25) < 1e-6
    if editor is not None and hasattr(editor, 'num_steps'):
        required['num_denoise_steps_28'] = int(editor.num_steps) == 28
    missing = [k for k, ok in required.items() if not ok]
    if missing:
        raise ValueError(f'Mismatched v5.1 runtime settings: {missing}')
    if harness.strategy is not None:
        raise ValueError('strategy_policy must not be set during controller RL training')


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.set_defaults(**V51_DEFAULTS)
    parser.add_argument('--data', required=True)
    parser.add_argument('--dev_data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_rl_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=4)
    parser.add_argument('--eval_every', type=int, default=8)
    parser.add_argument('--final_k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1.0e-5)
    parser.add_argument('--ppo_epochs', type=int, default=2)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--beta_kl', type=float, default=0.04)
    parser.add_argument('--target_kl', type=float, default=0.05)
    parser.add_argument('--min_dev_states', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    if (out / 'rl_v51_summary.json').exists():
        print('already complete:', out / 'rl_v51_summary.json')
        return
    if not args.controller_adapter or not args.editor_adapter:
        parser.error('Both --controller_adapter and --editor_adapter are required')
    if args.max_rl_steps < 1:
        parser.error('--max_rl_steps must be >= 1')

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    train_records = read_records(args.data)
    dev_records = read_records(args.dev_data)
    harness, controller, editor = load_runtime(args, trainable_controller=True, trainable_editor=False)
    assert_v51_alignment(harness, controller, editor)

    cache_dir = out / 'cache'
    dev_hash = cache_fingerprint(args.dev_data, harness, args)
    train_hash = cache_fingerprint(args.data, harness, args)
    dev_parents = prepare_parents(dev_records, harness, cache_path=cache_dir / f'dev_{dev_hash}.jsonl')
    train_parents = prepare_parents(train_records, harness, cache_path=cache_dir / f'train_{train_hash}.jsonl')
    print(f'Prepared {len(dev_parents)}/{len(dev_records)} dev parents and {len(train_parents)}/{len(train_records)} train parents', flush=True)

    summary, _ = train_ctrl_rl_v51(
        harness, controller, editor, train_parents, dev_parents, out,
        max_steps=args.max_rl_steps, batch_size=args.batch_size, group_size=args.group_size,
        eval_every=args.eval_every, final_k=args.final_k, lr=args.lr, ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps, beta_kl=args.beta_kl, target_kl=args.target_kl,
        min_dev_states=args.min_dev_states, seed=args.seed, cli_args=vars(args),
    )
    summary['wall_seconds_total'] = time.perf_counter() - started
    (out / 'rl_v51_summary.json').write_text(json.dumps(summary, indent=2))
    print('FINAL SUMMARY:', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
