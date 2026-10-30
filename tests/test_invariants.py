import torch
from redag.graph.schema import ConditionDAG, ConditionNode, ControllerAction
from redag.graph.dag import compute_affected_subgraph
from redag.edit.compositor import paste_window_strictly_local
from redag.verify.verifier import TwoStageVerifier
from redag.learn.credit import compute_pairwise_advantages, compute_subgraph_reward
from redag.learn.rl_trainer import compute_controller_rl_loss, compute_flow_sde_drift_mean, compute_masked_gaussian_logprob

def build_mock_dag() -> ConditionDAG:
    nodes = {f'v{i}': ConditionNode(node_id=f'v{i}', node_type='entity', question=f'Is entity {i} present?', entity_phrase=f'entity_{i}', bbox=(0.05 * i, 0.05 * i, 0.05 * i + 0.08, 0.05 * i + 0.08), weight=1.0) for i in range(1, 11)}
    return ConditionDAG(nodes=nodes, comp_edges=[('v1', 'v2')], supp_edges=[], int_edges=[('v2', 'v3')])

def test_eq3_strict_pixel_invariance_outside_window():
    canvas = torch.rand(3, 256, 256)
    edited_crop = torch.rand(3, 128, 128)
    box = (0.25, 0.25, 0.75, 0.75)
    out = paste_window_strictly_local(canvas, edited_crop, box, feather_px=8)
    mask = torch.ones((256, 256), dtype=torch.bool)
    mask[64:192, 64:192] = False
    max_diff_outside = torch.max(torch.abs(out[:, mask] - canvas[:, mask])).item()
    assert max_diff_outside == 0.0

def test_eq4_eq5_subgraph_isolation_and_zero_delta():
    dag = build_mock_dag()
    action = ControllerAction(op='modify_attr', box=(0.04, 0.04, 0.14, 0.14), instruction='fix v1')
    affected = compute_affected_subgraph(dag, action, target_node_id='v1')
    assert 'v1' in affected and 'v2' in affected and ('v3' in affected)
    assert 'v9' not in affected and 'v10' not in affected
    verifier = TwoStageVerifier(vlm_yes_no_scorer=lambda img, q: 0.85)
    prev_scores = {nid: 0.75 for nid in dag.nodes}
    new_scores = verifier.verify_subgraph_after_edit(torch.rand(3, 128, 128), dag, prev_scores, affected)
    for nid in dag.nodes:
        if nid not in affected:
            assert new_scores[nid] - prev_scores[nid] == 0.0

def test_eq27_global_denominator_invariance():
    dag = build_mock_dag()
    prev = {nid: 0.5 for nid in dag.nodes}
    delivered = dict(prev)
    delivered['v1'] = 0.9
    r_small = compute_subgraph_reward(dag, prev, delivered, set(), {'v1', 'v2'}, step_cost=0.01)
    r_large = compute_subgraph_reward(dag, prev, delivered, set(), {'v1', 'v2', 'v3', 'v4', 'v5'}, step_cost=0.01)
    assert abs(r_small - r_large) < 1e-08
    assert abs(r_small - (0.4 / 10.0 - 0.01)) < 1e-08

def test_eq34_eq35_pairwise_advantage_zero_sum_and_all_fail_silence():
    adv = compute_pairwise_advantages([-0.05, 0.12, 0.01, -0.02, 0.18, -0.05])
    assert torch.abs(adv.sum()).item() < 1e-06
    adv_ties = compute_pairwise_advantages([-0.01, -0.012, -0.009, -0.011, -0.01, -0.013], delta_noise=0.015)
    assert torch.all(adv_ties == 0.0).item()

def test_eq39_controller_rl_loss_independent_of_padding_length():
    torch.manual_seed(0)
    G, V = (4, 32)
    logits_base = torch.randn(G, 60, V, requires_grad=True)
    logits_old = logits_base.detach().clone()
    logits_sft = torch.randn(G, 60, V)
    tokens = torch.randint(0, V, (G, 60))
    mask = torch.zeros(G, 59)
    mask[:, 10:30] = 1.0
    adv = torch.tensor([0.5, -0.5, 0.2, -0.2])
    loss1, _ = compute_controller_rl_loss(logits_base, logits_old, logits_sft, tokens, mask, adv)
    pad_logits = torch.cat([logits_base.detach(), torch.randn(G, 40, V)], dim=1).requires_grad_(True)
    pad_old = pad_logits.detach().clone()
    pad_sft = torch.cat([logits_sft, torch.randn(G, 40, V)], dim=1)
    pad_tokens = torch.cat([tokens, torch.zeros(G, 40, dtype=torch.long)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(G, 40)], dim=1)
    loss2, _ = compute_controller_rl_loss(pad_logits, pad_old, pad_sft, pad_tokens, pad_mask, adv)
    assert abs(loss1.item() - loss2.item()) < 1e-06

def test_eq50_eq51_flow_sde_masked_logprob_gradient():
    z_t = torch.randn(1, 4, 16, 16)
    v_cfg = torch.randn(1, 4, 16, 16, requires_grad=True)
    t = torch.tensor([0.5])
    mu, sigma_t = compute_flow_sde_drift_mean(z_t, v_cfg, t, dt=0.02, eta=0.35)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1.0
    z_next_1 = torch.randn(1, 4, 16, 16)
    z_next_2 = z_next_1.clone()
    z_next_2[:, :, :4, :4] += 100.0
    lp1 = compute_masked_gaussian_logprob(z_next_1, mu, sigma_t, mask, dt=0.02)
    lp2 = compute_masked_gaussian_logprob(z_next_2, mu, sigma_t, mask, dt=0.02)
    assert abs(lp1.item() - lp2.item()) < 1e-07

def test_topological_gating_and_outside_nodes_are_never_scored():
    calls = []
    def score(image, question):
        calls.append(question)
        return {'root': 0.8, 'middle': 0.1, 'leaf': 0.9}[question]
    nodes = {n: ConditionNode(n, kind, n, n) for n, kind in [('leaf', 'entity'), ('middle', 'attribute'), ('root', 'entity'), ('outside', 'entity')]}
    dag = ConditionDAG(nodes, comp_edges=[('root', 'middle'), ('middle', 'leaf')])
    scores = dict.fromkeys(nodes, 0.9)
    result = TwoStageVerifier(score).verify_subgraph_after_edit(torch.rand(3, 12, 20), dag, scores, {'root', 'middle', 'leaf'})
    assert calls == ['root', 'middle']
    assert result['leaf'] == 0.0 and result['outside'] == scores['outside']
    assert scores == dict.fromkeys(nodes, 0.9)

def test_two_stage_uses_unpadded_original_crop_only_after_alarm():
    seen = []
    def score(image, question):
        seen.append(image.clone())
        return 0.2 if len(seen) == 1 else 0.95
    image = torch.arange(3 * 40 * 80).reshape(3, 40, 80).float() / 9600
    node = ConditionNode('n', 'attribute', 'question', 'entity', bbox=(0.25, 0.25, 0.75, 0.75))
    verifier = TwoStageVerifier(score)
    assert verifier.verify_single_node_two_stage(image, node) == 0.95
    assert len(seen) == 2 and torch.equal(seen[0], image)
    assert torch.equal(seen[1], image[:, 8:32, 16:64])
    verifier.verify_single_node_two_stage(image, node)
    assert len(seen) == 3

def test_harness_rollback_restores_parent_and_cost_only_reward():
    from redag.graph.schema import CanvasState
    from redag.harness.environment import ReDAGHarness
    nodes = {n: ConditionNode(n, 'entity', n, n, bbox=b) for n, b in [('target', (0.1, 0.1, 0.6, 0.6)), ('protected', (0.4, 0.4, 0.7, 0.7))]}
    dag = ConditionDAG(nodes)
    parent = CanvasState(torch.rand(3, 40, 80), {'target': 0.1, 'protected': 0.9}, {'protected'})
    original = parent.image.clone()
    scores = {'target': 0.95, 'protected': 0.5}
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: scores[question]), None)
    action = ControllerAction('add_entity', (0.1, 0.1, 0.65, 0.65), 'Add the target.')
    delivered, reward, accepted, affected = harness.evaluate_candidate_action(parent, dag, 'target', action, torch.ones(3, 16, 16))
    assert not accepted and reward == -0.01
    assert delivered.scores == parent.scores and delivered.protected_set == parent.protected_set
    assert torch.equal(delivered.image, original) and torch.equal(parent.image, original)
    assert parent.step_idx == 0 and delivered.step_idx == 1
    assert affected == {'target', 'protected'}

def test_controller_gradients_are_finite_action_only_and_reference_frozen():
    torch.manual_seed(12)
    logits = torch.randn(2, 9, 17, requires_grad=True)
    old = logits.detach().clone().requires_grad_(True)
    reference = torch.randn_like(logits, requires_grad=True)
    tokens = torch.randint(0, 17, (2, 9))
    mask = torch.zeros(2, 8)
    mask[0, 2:5] = 1
    mask[1, 3:7] = 1
    advantage = torch.tensor([0.4, -0.4], requires_grad=True)
    loss, metrics = compute_controller_rl_loss(logits, old, reference, tokens, mask, advantage)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[:, :-1][mask.bool()].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[:, :-1][~mask.bool()]) == 0
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    assert old.grad is None and reference.grad is None and advantage.grad is None
    assert metrics['ctrl_active_tokens'] == 7 and metrics['ctrl_kl_sft'] >= 0

def test_flow_logprob_has_finite_local_gradients_in_half_precision():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        z = torch.randn(1, 3, 6, 8, dtype=dtype)
        velocity = torch.randn_like(z, requires_grad=True)
        mask = torch.zeros(1, 1, 6, 8)
        mask[:, :, 2:4, 2:6] = 1
        mean, sigma = compute_flow_sde_drift_mean(z, velocity, torch.tensor([0.5]), 0.02)
        loss = compute_masked_gaussian_logprob(torch.zeros_like(z), mean, sigma, mask, 0.02).sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(velocity.grad).all()
        assert velocity.grad[:, :, 2:4, 2:6].abs().sum() > 0
        assert torch.count_nonzero(velocity.grad * (1 - mask)) == 0

class SmallVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, z_t, z_src, text_embeds, timesteps):
        return self.weight * z_t - 0.2 * z_src + text_embeds.mean() * 0.01

def test_flow_trajectory_seed_anchoring_and_replay_likelihood():
    from redag.models.editor import sample_flow_sde_trajectory
    from redag.learn.rl_trainer import predict_cfg_velocity
    model = SmallVelocity()
    source = torch.rand(1, 3, 8, 12)
    text = torch.ones(1, 2, 4)
    null = torch.zeros_like(text)
    mask = torch.zeros(1, 1, 8, 12)
    mask[:, :, 2:6, 3:9] = 1
    first = sample_flow_sde_trajectory(model, lambda z: z, source, text, null, mask, seed=37)
    second = sample_flow_sde_trajectory(model, lambda z: z, source, text, null, mask, seed=37)
    assert first['strata_indices'] == second['strata_indices']
    assert len(first['strata_indices']) == 4 and len(set(first['strata_indices'])) == 4
    assert torch.equal(first['edited_crop'], second['edited_crop'])
    assert torch.equal(first['z_next_list'][-1] * (1 - mask[0]), source[0] * (1 - mask[0]))
    assert all(not z.requires_grad for z in first['z_t_list'])
    for index in first['strata_indices']:
        t = first['t_steps'][index].unsqueeze(0)
        z = first['z_t_list'][index].unsqueeze(0)
        velocity = predict_cfg_velocity(model, z, source, text, null, t)
        mean, sigma = compute_flow_sde_drift_mean(z, velocity, t, 0.02)
        logp = compute_masked_gaussian_logprob(first['z_next_list'][index].unsqueeze(0), mean, sigma, mask, 0.02)
        assert sigma.item() > 0 and torch.isfinite(logp).all()
        assert torch.allclose(logp.squeeze(), first['logp_old_list'][index], atol=1e-5)

def test_sft_normalization_and_mask_weighting():
    from redag.learn.sft_trainer import compute_autoregressive_sft_loss, compute_editor_masked_flow_sft_loss
    logits = torch.zeros(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, -100, -100, 2, -100]])
    loss = compute_autoregressive_sft_loss(logits, labels)
    assert torch.allclose(loss, torch.tensor(7.0).log())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    source = torch.zeros(2, 3, 4, 6)
    target = torch.randn_like(source)
    mask = torch.zeros(2, 1, 4, 6)
    mask[:, :, 1:3, 2:4] = 1
    model = SmallVelocity()
    torch.manual_seed(19)
    t = torch.rand(2)
    epsilon = torch.randn_like(target)
    noisy = (1 - t[:, None, None, None]) * target + t[:, None, None, None] * epsilon
    error = model(noisy, source, torch.zeros(1), t) - (epsilon - target)
    weights = 1 + 3 * mask
    expected = ((weights * error.square()).sum((1, 2, 3)) / (3 * weights.sum((1, 2, 3)))).mean()
    torch.manual_seed(19)
    actual = compute_editor_masked_flow_sft_loss(model, source, target, torch.zeros(1), mask)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(model.weight.grad) and model.weight.grad.abs() > 0

def test_global_style_and_counts_do_not_expand_spatial_closure():
    from redag.verify.grounder import Grounder
    nodes = {'near': ConditionNode('near', 'entity', 'near', 'cat', (0.1, 0.1, 0.2, 0.2)), 'far': ConditionNode('far', 'entity', 'far', 'dog', (0.8, 0.8, 0.9, 0.9))}
    nodes['style'] = ConditionNode('style', 'background', 'style', '')
    nodes['cats'] = ConditionNode('cats', 'count', 'cats', 'cat', expected_count=1)
    nodes['dogs'] = ConditionNode('dogs', 'count', 'dogs', 'dog', expected_count=1)
    dag = Grounder(lambda image, phrase: []).ground(torch.zeros(3, 100, 100), ConditionDAG(nodes), {'cats', 'dogs'})
    action = ControllerAction('modify_attr', (0.09, 0.09, 0.21, 0.21), 'Change the near cat.')
    assert dag.nodes['cats'].bbox is None and dag.nodes['dogs'].bbox is None
    assert compute_affected_subgraph(dag, action, 'near') == {'near', 'cats', 'style'}

def test_flow_sft_large_half_precision_reduction_is_finite():
    from redag.learn.sft_trainer import compute_editor_masked_flow_sft_loss
    source = torch.zeros(1, 8, 128, 128, dtype=torch.float16)
    model = SmallVelocity()
    loss = compute_editor_masked_flow_sft_loss(model, source, torch.ones_like(source), torch.zeros(1), torch.ones(1, 1, 128, 128, dtype=torch.float16))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.weight.grad)


def test_multiturn_new_dependency_invalidates_previously_satisfied_child():
    from types import SimpleNamespace
    from redag.harness.environment import ReDAGHarness
    nodes = {n: ConditionNode(n, 'entity', n, n) for n in ('root', 'child')}
    graphs = iter([ConditionDAG(nodes), ConditionDAG(nodes, comp_edges=[('root', 'child')])])
    planner = SimpleNamespace(plan=lambda prompt, previous_dag: next(graphs))
    verifier = TwoStageVerifier(lambda image, question: 0.1 if question == 'root' else 0.9)
    state, dag, trace = ReDAGHarness(verifier, None).run_conversation(torch.zeros(3, 8, 12), ['first', 'add dependency'], planner, None, None, max_steps=0)
    assert state.scores == {'root': 0.1, 'child': 0.0}
    assert not state.protected_set and not trace and dag.comp_edges == [('root', 'child')]
