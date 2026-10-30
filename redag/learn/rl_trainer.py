import math
import random
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

def compute_controller_rl_loss(logits_new: torch.Tensor, logits_old: torch.Tensor, logits_sft: torch.Tensor, token_ids: torch.Tensor, action_mask: torch.Tensor, advantages: torch.Tensor, clip_eps: float=0.2, beta_kl: float=0.02) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid = action_mask.bool()
    if valid.shape != token_ids[:, 1:].shape:
        raise ValueError('Action mask must match shifted token IDs')
    total = valid.sum().clamp(min=1)
    if not valid.any():
        return logits_new.sum() * 0, {'ctrl_loss': 0.0, 'ctrl_kl_sft': 0.0, 'ctrl_mean_ratio': 0.0, 'ctrl_active_tokens': 0.0}
    new = F.log_softmax(logits_new[:, :-1][valid].float(), -1)
    old = F.log_softmax(logits_old.detach()[:, :-1][valid].float(), -1)
    ref = F.log_softmax(logits_sft.detach()[:, :-1][valid].float(), -1)
    targets = token_ids[:, 1:][valid, None]
    ratio = (new.gather(-1, targets) - old.gather(-1, targets)).squeeze(-1).exp()
    adv = advantages.detach().to(new).view(-1, 1).expand_as(action_mask)[valid]
    objective = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    kl = (new.exp() * (new - ref)).sum(-1).clamp_min(0)
    loss = -(objective - beta_kl * kl).sum() / total
    return loss, {'ctrl_loss': loss.item(), 'ctrl_kl_sft': kl.mean().item(), 'ctrl_mean_ratio': ratio.mean().item(), 'ctrl_active_tokens': float(valid.sum())}

def compute_flow_sde_sigma(t: torch.Tensor, eta: float=0.35) -> torch.Tensor:
    t = t.float()
    inside = (t >= 0.02) & (t <= 0.98)
    t_safe = torch.clamp(t, min=0.02, max=0.98)
    sigma = eta * torch.sqrt(t_safe / (1.0 - t_safe))
    return torch.where(inside, sigma, torch.zeros_like(sigma))

def predict_cfg_velocity(dit_model, z_t: torch.Tensor, z_src: torch.Tensor, text_embeds: torch.Tensor, null_text_embeds: torch.Tensor, t: torch.Tensor, cfg_scale: float=3.5) -> torch.Tensor:
    v_uncond = dit_model(z_t=z_t, z_src=z_src, text_embeds=null_text_embeds, timesteps=t)
    v_cond = dit_model(z_t=z_t, z_src=z_src, text_embeds=text_embeds, timesteps=t)
    return (1.0 - cfg_scale) * v_uncond + cfg_scale * v_cond

def compute_flow_sde_drift_mean(z_t: torch.Tensor, v_cfg: torch.Tensor, t: torch.Tensor, dt: float, eta: float=0.35) -> Tuple[torch.Tensor, torch.Tensor]:
    z_t, v_cfg, t = z_t.float(), v_cfg.float(), t.float()
    if dt <= 0:
        raise ValueError("Time increment must be positive")
    B = z_t.size(0)
    sigma_t = compute_flow_sde_sigma(t, eta=eta)
    sigma_view = sigma_t.view(B, 1, 1, 1)
    t_view = torch.clamp(t.view(B, 1, 1, 1), min=0.02)
    score_correction = sigma_view ** 2 / (2.0 * t_view) * (z_t + (1.0 - t_view) * v_cfg)
    mu = z_t - (v_cfg + score_correction) * dt
    return (mu, sigma_t)

def compute_masked_gaussian_logprob(z_next: torch.Tensor, mu: torch.Tensor, sigma_t: torch.Tensor, mask_latent: torch.Tensor, dt: float) -> torch.Tensor:
    B, C, h, w = z_next.shape
    z_next, mu, sigma_t = z_next.float(), mu.float(), sigma_t.float()
    M_E = mask_latent.to(device=z_next.device, dtype=torch.float32)
    if M_E.shape != (B, 1, h, w) or not ((M_E == 0) | (M_E == 1)).all() or (M_E.sum((1, 2, 3)) == 0).any():
        raise ValueError("Mask must be binary, nonempty, and single-channel")
    if dt <= 0:
        raise ValueError("Time increment must be positive")
    num_editable = (float(C) * M_E.sum(dim=(1, 2, 3))).clamp(min=1.0)
    var = torch.clamp(sigma_t ** 2 * dt, min=1e-08)
    sq_diff = ((z_next - mu) * M_E) ** 2
    quad_term = sq_diff.sum(dim=(1, 2, 3)) / (2.0 * num_editable * var)
    const_term = 0.5 * torch.log(2.0 * math.pi * var)
    return -(quad_term + const_term)

def sample_4_strata_indices(num_steps: int=50, rng=None) -> List[int]:
    rng = rng or random
    valid_indices = [k for k in range(num_steps) if 0.02 <= round(1.0 - k / float(num_steps), 12) <= 0.98]
    n_valid = len(valid_indices)
    if n_valid < 4:
        raise ValueError("Four stochastic strata require at least four valid steps")
    strata_size = n_valid // 4
    chosen = []
    for s in range(4):
        start = s * strata_size
        end = n_valid if s == 3 else (s + 1) * strata_size
        chosen.append(rng.choice(valid_indices[start:end]))
    return chosen

def compute_editor_rl_loss(dit_model, trajectories: List[Dict[str, torch.Tensor]], z_src: torch.Tensor, text_embeds: torch.Tensor, null_text_embeds: torch.Tensor, mask_latent: torch.Tensor, advantages: torch.Tensor, lora_params_current: List[torch.nn.Parameter], lora_params_sft_snapshot: List[torch.Tensor], dt: float=1.0 / 50.0, eta: float=0.35, cfg_scale: float=3.5, clip_eps: float=0.2, gamma_anchor: float=0.001) -> Tuple[torch.Tensor, Dict[str, float]]:
    if len(lora_params_current) != len(lora_params_sft_snapshot) or not lora_params_current:
        raise ValueError("LoRA snapshot must match current parameters")
    G_E = len(trajectories)
    surrogate_terms = []
    ratios_logged = []
    for i in range(G_E):
        adv_i = advantages[i].detach().to(z_src.device)
        traj = trajectories[i]
        for k in traj['strata_indices']:
            t_k = traj['t_steps'][k].detach().to(z_src.device).unsqueeze(0)
            z_t = traj['z_t_list'][k].detach().to(z_src.device).unsqueeze(0)
            z_next = traj['z_next_list'][k].detach().to(z_src.device).unsqueeze(0)
            logp_old = traj['logp_old_list'][k].detach().to(z_src.device)
            v_cfg = predict_cfg_velocity(dit_model, z_t, z_src, text_embeds, null_text_embeds, t_k, cfg_scale=cfg_scale)
            mu_new, sigma_t = compute_flow_sde_drift_mean(z_t, v_cfg, t_k, dt=dt, eta=eta)
            logp_new = compute_masked_gaussian_logprob(z_next, mu_new, sigma_t, mask_latent, dt=dt).squeeze(0)
            ratio = torch.exp(torch.clamp(logp_new - logp_old, min=-10.0, max=10.0))
            ratios_logged.append(float(ratio.detach().item()))
            s1 = ratio * adv_i
            s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_i
            surrogate_terms.append(torch.minimum(s1, s2))
    policy_loss = -torch.stack(surrogate_terms).mean()
    anchor_sq = sum((torch.sum((p.float() - p_sft.detach().to(p.device).float()) ** 2) for p, p_sft in zip(lora_params_current, lora_params_sft_snapshot)))
    total_loss = policy_loss + gamma_anchor * anchor_sq
    metrics = {'edit_rl_loss': float(total_loss.item()), 'edit_policy_loss': float(policy_loss.item()), 'edit_anchor_l2': float(anchor_sq.item()), 'edit_mean_ratio': float(sum(ratios_logged) / max(len(ratios_logged), 1))}
    return (total_loss, metrics)


def relative_lora_displacement(parameters, snapshot):
    numerator = sum(float((p.detach().float().cpu() - q.detach().float().cpu()).square().sum())
                    for p, q in zip(parameters, snapshot))
    denominator = sum(float(q.detach().float().cpu().square().sum()) for q in snapshot)
    return math.sqrt(numerator / max(denominator, 1e-30))


def set_trainable(parameters, enabled):
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def select_failed_parent(record, harness):
    from redag.harness.environment import state_from_record
    from redag.graph.dag import compute_active_frontier
    state, dag = state_from_record(record, harness)
    frontier = compute_active_frontier(dag, state.scores, harness.verifier.gate_tau)
    target = record.get("target_node_id")
    if target not in frontier:
        target = frontier[0] if frontier else None
    return state, dag, target


def sample_parent_action(controller, harness, state, dag, target, seed, temperature):
    counts = getattr(harness.grounder, "counts", {})
    count = counts.get(target, counts.get(dag.nodes[target].entity_phrase, 1))
    return controller.sample(state, dag, target, seed=seed, temperature=temperature, detected_count=count)


@torch.no_grad()
def ablation_reward(mode, parent, delivered, dag, local_reward, harness, controller, editor, rollout_steps):
    from copy import deepcopy
    from redag.learn.credit import compute_subgraph_reward
    if mode == "subgraph_credit_ours":
        return local_reward
    if mode == "step_full_table":
        scores = harness.verifier.verify_subgraph_after_edit(delivered.image, dag, parent.scores, set(dag.nodes))
        return compute_subgraph_reward(dag, parent.scores, scores, parent.protected_set, set(dag.nodes),
                                       harness.step_cost, harness.lam)
    history, candidate = list(harness.history), harness.last_candidate
    counts = dict(harness.grounder.counts) if harness.grounder else None
    masks = dict(harness.grounder.masks) if harness.grounder else None
    try:
        terminal, terminal_dag, steps = harness.run_session(delivered, deepcopy(dag), controller, editor,
                                      max_steps=rollout_steps - 1, seed=42, group_size=1)
        scores = harness.verifier.verify_subgraph_after_edit(terminal.image, terminal_dag, terminal.scores,
                                                            set(terminal_dag.nodes))
        endpoint = sum(terminal_dag.nodes[n].weight * score for n, score in scores.items()) / terminal_dag.total_weight_Z
        return endpoint / (parent.step_idx + len(steps) + 1)
    finally:
        harness.history, harness.last_candidate = history, candidate
        if harness.grounder:
            harness.grounder.counts, harness.grounder.masks = counts, masks


@torch.no_grad()
def evaluate_rl_development(records, harness, controller, editor, reference):
    current = controller.snapshot()
    accepted, attempts, kl_sum, tokens = 0, 0, 0.0, 0
    for record in records:
        state, dag, target = select_failed_parent(record, harness)
        if target is None:
            continue
        sample = sample_parent_action(controller, harness, state, dag, target, 42, 0)
        crop = editor.render(state.image, sample["action"], seed=42)
        _, _, accept, _ = harness.evaluate_candidate_action(state, dag, target, sample["action"], crop)
        accepted += int(accept)
        attempts += 1
        new = controller.forward_sample(sample, action_only=True)
        try:
            controller.load_snapshot(reference)
            ref = controller.forward_sample(sample, action_only=True)
        finally:
            controller.load_snapshot(current)
        length = int(sample["action_mask"].sum())
        ids = sample["token_ids"][:, -(length + 1):]
        mask = torch.ones((1, length), device=ids.device)
        _, metrics = compute_controller_rl_loss(new, new, ref, ids, mask, torch.zeros(1, device=ids.device))
        tokens += length
        kl_sum += metrics["ctrl_kl_sft"] * length
    return {"dev_acceptance_p": accepted / attempts if attempts else None, "dev_attempts": attempts,
            "ctrl_kl_sft": kl_sum / tokens if tokens else None}


def train_alternating(harness, controller, editor, records, dev_records, output,
                      config=None, rounds=2, epochs=1, batch_size=4,
                      reward_mode="subgraph_credit_ours", rollout_steps=5):
    import json
    from pathlib import Path
    from redag.config import ReDAGConfig
    from redag.learn.credit import compute_pairwise_advantages
    cfg = config or ReDAGConfig()
    if rounds != 2 or epochs < 1 or batch_size != 4 or not records or not dev_records:
        raise ValueError("Alternating RL requires two rounds, batches of four, and nonempty train/development data")
    if reward_mode not in ("terminal_broadcast", "step_full_table", "subgraph_credit_ours") or rollout_steps < 1:
        raise ValueError("Invalid reward ablation or rollout length")
    train_ids = {str(r.get("id", r.get("image"))) for r in records if "id" in r or isinstance(r.get("image"), str)}
    dev_ids = {str(r.get("id", r.get("image"))) for r in dev_records if "id" in r or isinstance(r.get("image"), str)}
    if train_ids & dev_ids:
        raise ValueError("Training and development sample IDs or image paths must be disjoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise ValueError("Output already contains RL metrics; choose a fresh output directory")
    ctrl_params, edit_params = controller.lora_parameters(), editor.lora_parameters()
    if not ctrl_params or not edit_params:
        raise ValueError("Both components require SFT LoRA adapters")
    ctrl_reference = controller.snapshot()
    edit_reference = [p.detach().cpu().clone() for p in edit_params]
    ctrl_anchor = list(ctrl_reference.values())
    history, global_step = [], 0

    def advantages(rewards):
        if reward_mode == "subgraph_credit_ours":
            return compute_pairwise_advantages(rewards, cfg.delta_noise, cfg.reward_scale_s0, cfg.adv_clip_max)
        values = torch.tensor(rewards, dtype=torch.float32)
        return values - (values.sum() - values) / (len(values) - 1)

    def record_step(stage, round_index, updated, extra):
        nonlocal global_step
        editor.eval()
        metrics = evaluate_rl_development(dev_records, harness, controller, editor, ctrl_reference)
        metrics.update(extra)
        metrics.update(stage=stage, method=reward_mode, round=round_index + 1, step=global_step, updated=updated,
                       ctrl_lora_relative_displacement=relative_lora_displacement(ctrl_params, ctrl_anchor),
                       edit_lora_relative_displacement=relative_lora_displacement(edit_params, edit_reference))
        history.append(metrics)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        global_step += 1
        return metrics

    for round_index in range(rounds):
        set_trainable(ctrl_params, True)
        set_trainable(edit_params, False)
        editor.eval()
        optimizer = torch.optim.AdamW(ctrl_params, lr=cfg.lr_rl_controller, weight_decay=0.0)
        baseline = evaluate_rl_development(dev_records, harness, controller, editor, ctrl_reference)
        best_p, best_controller = baseline["dev_acceptance_p"], controller.snapshot()
        for _ in range(epochs):
            for start in range(0, len(records), batch_size):
                active, reward_values, advantage_values = [], [], []
                for record in records[start:start + batch_size]:
                    state, dag, target = select_failed_parent(record, harness)
                    if target is None:
                        continue
                    samples, rewards = [], []
                    for i in range(cfg.group_size_ctrl):
                        sample = sample_parent_action(controller, harness, state, dag, target, global_step * 100 + i, 0.8)
                        crop = editor.render(state.image, sample["action"], seed=42)
                        delivered, reward, _, _ = harness.evaluate_candidate_action(state, dag, target, sample["action"], crop)
                        reward = ablation_reward(reward_mode, state, delivered, dag, reward, harness,
                                                 controller, editor, rollout_steps)
                        samples.append(sample)
                        rewards.append(reward)
                    adv = advantages(rewards)
                    advantage_values.extend(adv.tolist())
                    reward_values.extend(rewards)
                    if adv.abs().any():
                        active.extend(zip(samples, adv))
                optimizer.zero_grad(set_to_none=True)
                total_tokens = sum(int(sample["action_mask"].sum()) for sample, _ in active)
                current, weighted_loss = controller.snapshot(), 0.0
                for sample, advantage in active:
                    with torch.no_grad():
                        old = controller.forward_sample(sample, action_only=True)
                        try:
                            controller.load_snapshot(ctrl_reference)
                            ref = controller.forward_sample(sample, action_only=True)
                        finally:
                            controller.load_snapshot(current)
                    new = controller.forward_sample(sample, action_only=True)
                    length = int(sample["action_mask"].sum())
                    ids = sample["token_ids"][:, -(length + 1):]
                    mask = torch.ones((1, length), device=ids.device)
                    loss, _ = compute_controller_rl_loss(new, old, ref, ids, mask, advantage.reshape(1),
                              cfg.clip_eps_ctrl, cfg.beta_kl_ctrl)
                    loss = loss * length / total_tokens
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite controller RL loss")
                    loss.backward()
                    weighted_loss += loss.detach().item()
                    del new, old, ref, loss
                if active:
                    torch.nn.utils.clip_grad_norm_(ctrl_params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                metrics = record_step("ctrl_rl", round_index, bool(active),
                          {"ctrl_loss": weighted_loss, "actual_action_tokens": total_tokens,
                           "reward_variance": float(torch.tensor(reward_values).var(unbiased=False)) if reward_values else 0.0,
                           "advantage_variance": float(torch.tensor(advantage_values).var(unbiased=False)) if advantage_values else 0.0,
                           "mean_reward": sum(reward_values) / max(len(reward_values), 1)})
                p = metrics["dev_acceptance_p"]
                if p is not None and (best_p is None or p >= best_p):
                    best_p, best_controller = p, controller.snapshot()
        controller.load_snapshot(best_controller)
        stage_dir = output / f"round_{round_index + 1}"
        controller.save_adapter(stage_dir / "ctrl_rl")
        set_trainable(ctrl_params, False)
        set_trainable(edit_params, True)
        optimizer = torch.optim.AdamW(edit_params, lr=cfg.lr_rl_editor, weight_decay=0.0)
        for _ in range(epochs):
            for start in range(0, len(records), batch_size):
                groups, reward_values, advantage_values = [], [], []
                editor.eval()
                for record in records[start:start + batch_size]:
                    state, dag, target = select_failed_parent(record, harness)
                    if target is None:
                        continue
                    sample = sample_parent_action(controller, harness, state, dag, target, 42, 0)
                    conditioning = editor.prepare(state.image, sample["action"])
                    trajectories, rewards = [], []
                    for seed in range(cfg.group_size_edit):
                        trajectory = editor.sample(state.image, sample["action"], seed=seed, conditioning=conditioning)
                        delivered, reward, _, _ = harness.evaluate_candidate_action(state, dag, target, sample["action"],
                                          trajectory["edited_crop"].to(state.image.device))
                        reward = ablation_reward(reward_mode, state, delivered, dag, reward, harness,
                                                 controller, editor, rollout_steps)
                        for key in ("t_steps", "z_t_list", "z_next_list", "logp_old_list"):
                            trajectory[key] = [tensor.detach().cpu() for tensor in trajectory[key]]
                        trajectory.pop("edited_crop")
                        trajectories.append(trajectory)
                        rewards.append(reward)
                    adv = advantages(rewards)
                    advantage_values.extend(adv.tolist())
                    reward_values.extend(rewards)
                    if adv.abs().any():
                        groups.append((conditioning, trajectories, adv))
                optimizer.zero_grad(set_to_none=True)
                editor.train()
                weighted_loss = 0.0
                for conditioning, trajectories, adv in groups:
                    for index, trajectory in enumerate(trajectories):
                        loss, _ = compute_editor_rl_loss(editor.dit_model, [trajectory], **conditioning,
                                  advantages=adv[index:index + 1], lora_params_current=edit_params,
                                  lora_params_sft_snapshot=edit_reference, dt=1 / cfg.num_denoise_steps,
                                  eta=cfg.sde_eta, cfg_scale=cfg.cfg_scale, clip_eps=cfg.clip_eps_edit,
                                  gamma_anchor=cfg.gamma_anchor)
                        loss = loss / (len(groups) * len(trajectories))
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite editor RL loss")
                        loss.backward()
                        weighted_loss += loss.detach().item()
                if groups:
                    torch.nn.utils.clip_grad_norm_(edit_params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                record_step("edit_rl", round_index, bool(groups), {"edit_rl_loss": weighted_loss,
                            "reward_variance": float(torch.tensor(reward_values).var(unbiased=False)) if reward_values else 0.0,
                            "advantage_variance": float(torch.tensor(advantage_values).var(unbiased=False)) if advantage_values else 0.0,
                            "mean_reward": sum(reward_values) / max(len(reward_values), 1)})
        editor.save_adapter(stage_dir / "edit_rl")
    controller.save_adapter(output / "alternating_rl" / "controller")
    editor.save_adapter(output / "alternating_rl" / "editor")
    return history
