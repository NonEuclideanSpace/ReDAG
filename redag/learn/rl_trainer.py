import math
import random
import torch
import torch.nn.functional as F


def compute_controller_rl_loss(logits_new, logits_old, logits_sft, token_ids, action_mask, advantages, clip_eps=0.2, beta_kl=0.02):
    valid = action_mask.bool()
    if valid.shape != token_ids[:, 1:].shape or not ((action_mask == 0) | (action_mask == 1)).all():
        raise ValueError('Action mask must be binary and match shifted token IDs')
    if not valid.any():
        return logits_new.sum() * 0, dict(ctrl_loss=0.0, ctrl_kl_sft=0.0, ctrl_policy_kl=0.0, ctrl_mean_ratio=0.0, ctrl_active_tokens=0.0, ratio_guard_hit=False)
    new = F.log_softmax(logits_new[:, :-1][valid].float(), -1)
    old = F.log_softmax(logits_old.detach()[:, :-1][valid].float(), -1)
    ref = F.log_softmax(logits_sft.detach()[:, :-1][valid].float(), -1)
    targets = token_ids[:, 1:][valid, None]
    log_ratio = (new.gather(-1, targets) - old.gather(-1, targets)).squeeze(-1)
    ratio = log_ratio.clamp(-60, 60).exp()
    adv = advantages.detach().to(new).view(-1, 1).expand_as(action_mask)[valid]
    objective = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    kl = (new.exp() * (new - ref)).sum(-1).clamp_min(0)
    policy_kl = (old.exp() * (old - new)).sum(-1).clamp_min(0)
    loss = -(objective - beta_kl * kl).mean()
    return loss, dict(ctrl_loss=loss.item(), ctrl_kl_sft=kl.mean().item(), ctrl_policy_kl=policy_kl.mean().item(),
                      ctrl_mean_ratio=ratio.mean().item(), ctrl_active_tokens=float(valid.sum()), ratio_guard_hit=bool((log_ratio.abs() > 60).any()))


def compute_flow_sde_sigma(t, eta=0.35):
    t = t.float()
    if not torch.isfinite(t).all() or not ((t >= 0) & (t <= 1)).all() or eta <= 0:
        raise ValueError('Invalid diffusion time or noise strength')
    safe = t.clamp(0.02, 0.98)
    return torch.where((t >= 0.02) & (t <= 0.98), eta * (safe / (1 - safe)).sqrt(), torch.zeros_like(t))


def predict_cfg_velocity(dit_model, z_t, z_src, text_embeds, null_text_embeds, t, cfg_scale=3.5):
    uncond = dit_model(z_t=z_t, z_src=z_src, text_embeds=null_text_embeds, timesteps=t)
    cond = dit_model(z_t=z_t, z_src=z_src, text_embeds=text_embeds, timesteps=t)
    return (1 - cfg_scale) * uncond + cfg_scale * cond


def compute_flow_sde_drift_mean(z_t, v_cfg, t, dt, eta=0.35):
    if dt <= 0:
        raise ValueError('Time increment must be positive')
    z_t, v_cfg, t = z_t.float(), v_cfg.float(), t.float()
    sigma = compute_flow_sde_sigma(t, eta)
    time = t.reshape(-1, 1, 1, 1).clamp_min(0.02)
    correction = sigma.reshape(-1, 1, 1, 1).square() / (2 * time) * (z_t + (1 - time) * v_cfg)
    return z_t - (v_cfg + correction) * dt, sigma


def gaussian_mask_terms(z_next, mu, sigma_t, mask_latent, dt):
    mask = mask_latent.to(device=z_next.device, dtype=torch.float64)
    if z_next.ndim != 4 or mu.shape != z_next.shape or mask.shape != (z_next.shape[0], 1, *z_next.shape[2:]):
        raise ValueError('Transition tensors and single-channel mask shapes do not match')
    if not ((mask == 0) | (mask == 1)).all() or (mask.sum((1, 2, 3)) == 0).any():
        raise ValueError('Mask must be binary and nonempty')
    if sigma_t.shape != (z_next.shape[0],) or not torch.isfinite(z_next).all() or not torch.isfinite(mu).all():
        raise ValueError('Transition means, targets and batch noise scales must be finite and aligned')
    if not math.isfinite(dt) or dt <= 0 or not torch.isfinite(sigma_t).all() or (sigma_t <= 0).any():
        raise ValueError('Gaussian likelihood requires a stochastic transition')
    count = z_next.shape[1] * mask.sum((1, 2, 3))
    return mask, count, sigma_t.double().square() * dt


def compute_masked_gaussian_logprob(z_next, mu, sigma_t, mask_latent, dt, reduction='mean'):
    mask, count, var = gaussian_mask_terms(z_next, mu, sigma_t, mask_latent, dt)
    joint = -((z_next.double() - mu.double()).square() * mask).sum((1, 2, 3)) / (2 * var) - count * torch.log(2 * math.pi * var) / 2
    if reduction not in ('mean', 'sum'):
        raise ValueError('Likelihood reduction must be mean or sum')
    return joint / count if reduction == 'mean' else joint


def compute_transition_log_ratio(z_next, mu_new, mu_old, sigma_t, mask_latent, dt, reduction='mean'):
    mask, count, var = gaussian_mask_terms(z_next, mu_new, sigma_t, mask_latent, dt)
    if mu_old.shape != mu_new.shape or not torch.isfinite(mu_old).all():
        raise ValueError('Old transition means must be finite and match current means')
    if reduction not in ('mean', 'sum'):
        raise ValueError('Likelihood reduction must be mean or sum')
    scale = 2 * count * var if reduction == 'mean' else 2 * var
    old_residual = z_next.double() - mu_old.detach().double()
    new_residual = z_next.double() - mu_new.double()
    log_ratio = ((old_residual.square() - new_residual.square()) * mask).sum((1, 2, 3)) / scale
    kl = ((mu_new.double() - mu_old.detach().double()).square() * mask).sum((1, 2, 3)) / scale
    return log_ratio, kl


def get_stochastic_strata(num_steps=50):
    valid = [k for k in range(num_steps) if 0.02 <= round((num_steps - k) / num_steps, 12) <= 0.98]
    if len(valid) < 4:
        raise ValueError('Four strata require at least four stochastic steps')
    size = len(valid) // 4
    return [valid[s * size:(s + 1) * size if s < 3 else len(valid)] for s in range(4)]


def strata_weights(num_steps=50):
    strata = get_stochastic_strata(num_steps)
    return [len(part) / sum(map(len, strata)) for part in strata]


def sample_4_strata_indices(num_steps=50, rng=None):
    return [(rng or random).choice(part) for part in get_stochastic_strata(num_steps)]


def compute_editor_rl_loss(dit_model, trajectories, z_src, text_embeds, null_text_embeds, mask_latent, advantages,
                           lora_params_current, lora_params_sft_snapshot, dt=0.02, eta=0.35, cfg_scale=3.5, clip_eps=0.2, gamma_anchor=0.001):
    if not trajectories or len(trajectories) != len(advantages) or not lora_params_current or len(lora_params_current) != len(lora_params_sft_snapshot):
        raise ValueError('Trajectory advantages and LoRA snapshots must match')
    terms, ratios, kls, guarded = [], [], [], False
    for traj, advantage in zip(trajectories, advantages):
        strata = get_stochastic_strata(len(traj['t_steps']))
        weights = strata_weights(len(traj['t_steps']))
        if not math.isclose(dt, 1 / len(traj['t_steps']), rel_tol=1e-6):
            raise ValueError('Replay time increment must match the sampled trajectory')
        if len(traj['strata_indices']) != 4 or any(k not in part for k, part in zip(traj['strata_indices'], strata)):
            raise ValueError('Replay must contain one transition from each stratum')
        for index, k in enumerate(traj['strata_indices']):
            t = traj['t_steps'][k].detach().to(z_src.device).reshape(1)
            z_t, z_next, mu_old = [traj[name][k].detach().to(z_src.device).unsqueeze(0) for name in ('z_t_list', 'z_next_list', 'mu_old_list')]
            velocity = predict_cfg_velocity(dit_model, z_t, z_src, text_embeds, null_text_embeds, t, cfg_scale)
            mu, sigma = compute_flow_sde_drift_mean(z_t, velocity, t, dt, eta)
            log_ratio, kl = compute_transition_log_ratio(z_next, mu, mu_old, sigma, mask_latent, dt)
            guarded = guarded or bool((log_ratio.abs() > 60).any())
            ratio = log_ratio.clamp(-60, 60).exp().squeeze()
            adv = advantage.detach().to(ratio)
            weight = weights[index] / len(trajectories)
            terms.append(weight * torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv))
            ratios.append(weight * ratio.detach().item())
            kls.append(weight * kl.detach().item())
    policy_loss = -torch.stack(terms).sum()
    anchor = sum((p.float() - q.detach().to(p.device).float()).square().sum() for p, q in zip(lora_params_current, lora_params_sft_snapshot))
    loss = policy_loss + gamma_anchor * anchor
    return loss, dict(edit_rl_loss=loss.item(), edit_policy_loss=policy_loss.item(), edit_anchor_l2=anchor.item(),
                      edit_mean_ratio=sum(ratios), edit_policy_kl=sum(kls), ratio_guard_hit=guarded)


def relative_lora_displacement(parameters, snapshot):
    num = sum((p.detach().float().cpu() - q.float().cpu()).square().sum().item() for p, q in zip(parameters, snapshot))
    den = sum(q.float().cpu().square().sum().item() for q in snapshot)
    return math.sqrt(num / max(den, 1e-30))


def select_failed_parent(record, harness):
    from redag.harness.environment import state_from_record
    from redag.graph.dag import compute_active_frontier
    state, dag = state_from_record(record, harness)
    frontier = compute_active_frontier(dag, state.scores, harness.verifier.gate_tau)
    target = record.get('target_node_id')
    return state, dag, target if target in frontier else frontier[0] if frontier else None


def sample_parent_action(controller, harness, state, dag, target, seed, temperature):
    counts = getattr(harness.grounder, 'counts', {})
    return controller.sample(state, dag, target, seed=seed, temperature=temperature, detected_count=counts.get(dag.nodes[target].entity_phrase, 1))


@torch.no_grad()
def ablation_reward(mode, parent, delivered, dag, local_reward, harness, controller, editor, rollout_steps):
    from copy import deepcopy
    from redag.learn.credit import compute_subgraph_reward
    if mode == 'subgraph_credit_ours':
        return local_reward
    if mode == 'step_full_table':
        scores = harness.verifier.verify_subgraph_after_edit(delivered.image, dag, parent.scores, set(dag.nodes))
        return compute_subgraph_reward(dag, parent.scores, scores, parent.protected_set, set(dag.nodes), harness.step_cost, harness.lam)
    history, candidate, last_run = list(harness.history), harness.last_candidate, dict(harness.last_run)
    grounding, unknown = harness.grounding_state(), set(harness.unknown_nodes)
    limits = harness.verifier.call_limit, harness.render_limit, harness.verifier.two_stage, harness.mode
    try:
        graph = deepcopy(dag)
        if harness.grounder:
            graph = harness.grounder.ground(delivered.image, graph)
        terminal, graph, steps = harness.run_session(delivered, graph, controller, editor, max_steps=rollout_steps - 1, seed=42, group_size=1)
        scores = harness.verifier.verify_subgraph_after_edit(terminal.image, graph, terminal.scores, set(graph.nodes))
        return sum(graph.nodes[n].weight * q for n, q in scores.items()) / graph.total_weight_Z / (parent.step_idx + len(steps) + 1)
    finally:
        harness.history, harness.last_candidate, harness.last_run = history, candidate, last_run
        harness.unknown_nodes = unknown
        harness.restore_grounding(grounding)
        harness.verifier.call_limit, harness.render_limit, harness.verifier.two_stage, harness.mode = limits


@torch.no_grad()
def evaluate_rl_development(records, harness, controller, editor, reference):
    current, accepted, attempts, kl_sum, tokens = controller.snapshot(), 0, 0, 0.0, 0
    for record in records:
        state, dag, target = select_failed_parent(record, harness)
        if target is None:
            continue
        sample = sample_parent_action(controller, harness, state, dag, target, 42, 0)
        _, _, accept, _ = harness.evaluate_candidate_action(state, dag, target, sample['action'], editor.render(state.image, sample['action'], seed=42))
        accepted, attempts = accepted + int(accept), attempts + 1
        new = controller.forward_sample(sample, action_only=True)
        try:
            controller.load_snapshot(reference)
            ref = controller.forward_sample(sample, action_only=True)
        finally:
            controller.load_snapshot(current)
        length = int(sample['action_mask'].sum())
        ids = sample['token_ids'][:, -(length + 1):].to(new.device)
        _, metrics = compute_controller_rl_loss(new, new, ref, ids, torch.ones((1, length), device=new.device), torch.zeros(1, device=new.device))
        tokens, kl_sum = tokens + length, kl_sum + metrics['ctrl_kl_sft'] * length
    if not attempts:
        raise ValueError('Development data must contain at least one failed parent state')
    return dict(dev_acceptance_p=accepted / attempts, dev_attempts=attempts, ctrl_kl_sft=kl_sum / tokens)


def optimize_rollout(stage, groups, controller, editor, optimizer, ctrl_old, ctrl_reference, edit_reference, cfg):
    parameters = controller.lora_parameters() if stage == 'ctrl_rl' else editor.lora_parameters()
    total_tokens = sum(int(sample['action_mask'].sum()) for group, _, _ in groups for sample in group) if stage == 'ctrl_rl' else 0
    updates, last_loss, max_kl, stopped = 0, 0.0, 0.0, False
    active = any(bool(adv.abs().any()) for _, adv, _ in groups)
    for _ in range(cfg.ppo_epochs if active else 0):
        optimizer.zero_grad(set_to_none=True)
        current, epoch_loss = controller.snapshot(), 0.0
        for items, adv, conditioning in groups:
            for index, item in enumerate(items):
                if stage == 'ctrl_rl':
                    with torch.no_grad():
                        try:
                            controller.load_snapshot(ctrl_old)
                            old = controller.forward_sample(item, action_only=True)
                            controller.load_snapshot(ctrl_reference)
                            ref = controller.forward_sample(item, action_only=True)
                        finally:
                            controller.load_snapshot(current)
                    new = controller.forward_sample(item, action_only=True)
                    length = int(item['action_mask'].sum())
                    ids = item['token_ids'][:, -(length + 1):].to(new.device)
                    loss, metrics = compute_controller_rl_loss(new, old, ref, ids, torch.ones((1, length), device=new.device), adv[index:index + 1], cfg.clip_eps_ctrl, cfg.beta_kl_ctrl)
                    loss = loss * length / total_tokens
                else:
                    editor.train()
                    loss, metrics = compute_editor_rl_loss(editor.dit_model, [item], **conditioning, advantages=adv[index:index + 1],
                                    lora_params_current=parameters, lora_params_sft_snapshot=edit_reference, dt=1 / cfg.num_denoise_steps,
                                    eta=cfg.sde_eta, cfg_scale=cfg.cfg_scale, clip_eps=cfg.clip_eps_edit, gamma_anchor=cfg.gamma_anchor)
                    loss = loss / (len(groups) * len(items))
                policy_kl = metrics['ctrl_policy_kl' if stage == 'ctrl_rl' else 'edit_policy_kl']
                max_kl = max(max_kl, policy_kl)
                if metrics['ratio_guard_hit'] or policy_kl > (cfg.target_kl_ctrl if stage == 'ctrl_rl' else cfg.target_kl_edit):
                    stopped = True
                    break
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite policy loss')
                loss.backward()
                epoch_loss += loss.detach().item()
            if stopped:
                break
        if stopped:
            optimizer.zero_grad(set_to_none=True)
            break
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        updates, last_loss = updates + 1, epoch_loss
    return dict(optimizer_updates=updates, updated=updates > 0, loss=last_loss, max_old_policy_kl=max_kl, kl_early_stop=stopped, actual_action_tokens=total_tokens)


def save_training_audit(output, identifier, parent, delivered, dag, affected, method, stage, step, protection_lambda):
    import json
    from redag.verify.grounder import save_image
    views = {}
    nodes = {nid: dict(question=n.question, threshold=n.threshold, weight=n.weight,
                      scope=n.scope, depends_on=n.depends_on, bbox=n.bbox) for nid, n in dag.nodes.items()}
    for name, state in (('before', parent), ('after', delivered)):
        view_id = identifier + '_' + name
        path = output / 'training_audit_images' / (view_id + '.png')
        save_image(state.image, path)
        views[name] = dict(id=view_id, image=str(path.resolve()), nodes=nodes)
    with (output / 'training_audits.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(**views, affected=sorted(affected), protected=sorted(parent.protected_set),
            method=method, stage=stage, step=step, source='training', protection_lambda=protection_lambda), ensure_ascii=False) + '\n')


def train_alternating(harness, controller, editor, records, dev_records, output, config=None, rounds=2, epochs=1, batch_size=4,
                      reward_mode='subgraph_credit_ours', rollout_steps=5, seed=42, sources=None):
    import json
    import time
    from pathlib import Path
    from dataclasses import asdict
    from redag.learn.credit import compute_pairwise_advantages
    cfg = config or harness.config
    if rounds != 2 or epochs < 1 or batch_size != 4 or not records or not dev_records or rollout_steps < 1:
        raise ValueError('Training requires two rounds, batches of four, positive epochs and nonempty train/development data')
    if reward_mode not in ('terminal_broadcast', 'step_full_table', 'subgraph_credit_ours'):
        raise ValueError('Unknown reward method')
    for key in ('id', 'image'):
        train_ids = {r[key] for r in records if isinstance(r.get(key), str)}
        if train_ids & {r[key] for r in dev_records if isinstance(r.get(key), str)}:
            raise ValueError('Training and development IDs and image paths must be disjoint')
    output = Path(output)
    if (output / 'metrics.jsonl').exists():
        raise ValueError('Choose a fresh training output directory')
    output.mkdir(parents=True, exist_ok=True)
    ctrl_params, edit_params = controller.lora_parameters(), editor.lora_parameters()
    if not ctrl_params or not edit_params:
        raise ValueError('Both components require SFT adapters')
    ctrl_ref, edit_ref = controller.snapshot(), [p.detach().cpu().clone() for p in edit_params]
    devices = [i for i in range(torch.cuda.device_count()) if torch.cuda.memory_allocated(i)]
    torch.manual_seed(seed)
    rng, history, step = random.Random(seed), [], 0
    audit_rng = random.Random(seed)
    checkpoints = {'zero_shot': {'controller': None, 'editor': None}}
    def save_stage(name):
        controller.save_adapter(output / name / 'controller')
        editor.save_adapter(output / name / 'editor')
        checkpoints[name] = {kind: f'{name}/{kind}' for kind in ('controller', 'editor')}
        (output / 'stage_checkpoints.json').write_text(json.dumps(checkpoints, indent=2))
    save_stage('sft')
    environment = dict(config=asdict(harness.config), mode=harness.mode, gate_tau=harness.verifier.gate_tau,
                       feather_px=harness.feather_px, step_cost=harness.step_cost, protection_lambda=harness.lam)
    (output / 'training_config.json').write_text(json.dumps(dict(config=asdict(cfg), environment=environment, sources=sources,
        method=reward_mode, seed=seed, epochs=epochs, batch_size=batch_size, rollout_steps=rollout_steps), indent=2))
    (output / 'training_audits.jsonl').touch()
    editor.num_steps, editor.eta, editor.cfg_scale = cfg.num_denoise_steps, cfg.sde_eta, cfg.cfg_scale
    for round_index in range(rounds):
        for stage in ('ctrl_rl', 'edit_rl'):
            for p in ctrl_params:
                p.requires_grad_(stage == 'ctrl_rl')
            for p in edit_params:
                p.requires_grad_(stage == 'edit_rl')
            editor.eval()
            optimizer = torch.optim.AdamW(ctrl_params if stage == 'ctrl_rl' else edit_params,
                                         lr=cfg.lr_rl_controller if stage == 'ctrl_rl' else cfg.lr_rl_editor, weight_decay=0)
            baseline = evaluate_rl_development(dev_records, harness, controller, editor, ctrl_ref)
            best_p, best_ctrl = baseline['dev_acceptance_p'], controller.snapshot()
            for _ in range(epochs):
                order = list(records)
                rng.shuffle(order)
                for start in range(0, len(order), batch_size):
                    for device in devices:
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    groups, rewards_all, advantages_all, old = [], [], [], controller.snapshot()
                    for parent_index, record in enumerate(order[start:start + batch_size]):
                        state, dag, target = select_failed_parent(record, harness)
                        if target is None:
                            continue
                        items, rewards, conditioning = [], [], None
                        editor.eval()
                        if stage == 'edit_rl':
                            action = sample_parent_action(controller, harness, state, dag, target, seed, 0)['action']
                            conditioning = editor.prepare(state.image, action)
                        for index in range(cfg.group_size_ctrl if stage == 'ctrl_rl' else cfg.group_size_edit):
                            if stage == 'ctrl_rl':
                                item = sample_parent_action(controller, harness, state, dag, target, seed + step * 100 + index, 0.8)
                                action, crop = item['action'], editor.render(state.image, item['action'], seed=42)
                                item['inputs'] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in item['inputs'].items()}
                            else:
                                item = editor.sample(state.image, action, seed=index, conditioning=conditioning)
                                crop = item.pop('edited_crop')
                            delivered, reward, _, affected = harness.evaluate_candidate_action(state, dag, target, action, crop.to(state.image.device))
                            if audit_rng.random() < 0.05:
                                save_training_audit(output, f'{reward_mode}_{step}_{parent_index}_{index}', state, delivered, dag, affected, reward_mode, stage, step, harness.lam)
                            rewards.append(ablation_reward(reward_mode, state, delivered, dag, reward, harness, controller, editor, rollout_steps))
                            items.append(item)
                        values = torch.tensor(rewards)
                        adv = compute_pairwise_advantages(rewards, cfg.delta_noise, cfg.reward_scale_s0, cfg.adv_clip_max) if reward_mode == 'subgraph_credit_ours' else values - (values.sum() - values) / (len(values) - 1)
                        rewards_all.extend(rewards)
                        advantages_all.extend(adv.tolist())
                        groups.append((items, adv, conditioning))
                    metrics = optimize_rollout(stage, groups, controller, editor, optimizer, old, ctrl_ref, edit_ref, cfg)
                    editor.eval()
                    metrics.update(evaluate_rl_development(dev_records, harness, controller, editor, ctrl_ref))
                    for device in devices:
                        torch.cuda.synchronize(device)
                    metrics.update(step_wall_seconds=time.perf_counter() - started,
                                   peak_cuda_allocated_bytes={f'cuda:{d}': torch.cuda.max_memory_allocated(d) for d in devices},
                                   dev_initial_acceptance_p=baseline['dev_acceptance_p'],
                                   dev_acceptance_gain=metrics['dev_acceptance_p'] - baseline['dev_acceptance_p'])
                    metrics.update(stage=stage, round=round_index + 1, step=step, method=reward_mode,
                                   sampled_parents=len(groups), sampled_candidates=len(rewards_all),
                                   ctrl_lora_relative_displacement=relative_lora_displacement(ctrl_params, list(ctrl_ref.values())),
                                   edit_lora_relative_displacement=relative_lora_displacement(edit_params, edit_ref),
                                   reward_variance=float(torch.tensor(rewards_all).var(unbiased=False)) if rewards_all else 0.0,
                                   advantage_variance=float(torch.tensor(advantages_all).var(unbiased=False)) if advantages_all else 0.0,
                                   mean_reward=sum(rewards_all) / max(len(rewards_all), 1))
                    with (output / 'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(metrics, allow_nan=False) + '\n')
                    history.append(metrics)
                    step += 1
                    if stage == 'ctrl_rl' and metrics['dev_acceptance_p'] >= best_p:
                        best_p, best_ctrl = metrics['dev_acceptance_p'], controller.snapshot()
            if stage == 'ctrl_rl':
                controller.load_snapshot(best_ctrl)
            save_stage(stage if round_index == 0 else f'round_2/{stage}')
    save_stage('alternating_rl')
    checkpoints = {k: v for k, v in checkpoints.items() if not k.startswith('round_2/')}
    (output / 'stage_checkpoints.json').write_text(json.dumps(checkpoints, indent=2))
    return history
