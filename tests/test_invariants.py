import random
from types import SimpleNamespace
import pytest
import torch
from redag.graph.schema import CanvasState, ConditionDAG, ConditionNode, ControllerAction
from redag.graph.dag import compute_affected_subgraph
from redag.edit.compositor import paste_window_strictly_local
from redag.verify.verifier import TwoStageVerifier
from redag.harness.environment import ReDAGHarness
from redag.learn.credit import compute_pairwise_advantages, compute_subgraph_reward
from redag.learn.rl_trainer import compute_controller_rl_loss, compute_flow_sde_drift_mean, compute_masked_gaussian_logprob

def build_mock_dag():
    nodes = {f'v{i}': ConditionNode(f'v{i}', 'entity', f'entity {i}', f'entity_{i}', (0.05*i, 0.05*i, 0.05*i+0.08, 0.05*i+0.08), scope='local') for i in range(1, 11)}
    return ConditionDAG(nodes, comp_edges=[('v1', 'v2')], int_edges=[('v2', 'v3')])

def test_eq3_strict_pixel_invariance_outside_window():
    canvas = torch.rand(3, 256, 256)
    out = paste_window_strictly_local(canvas, torch.rand(3, 128, 128), (0.25, 0.25, 0.75, 0.75), feather_px=8)
    mask = torch.ones((256, 256), dtype=torch.bool)
    mask[64:192, 64:192] = False
    assert torch.equal(out[:, mask], canvas[:, mask])

def test_protected_pixels_inside_the_window_are_copied_from_the_canvas():
    canvas, crop = torch.rand(3, 64, 64), torch.ones(3, 32, 32)
    editable = torch.ones(64, 64)
    editable[20:28, 20:28] = 0
    out = paste_window_strictly_local(canvas, crop, (0.25, 0.25, 0.75, 0.75), feather_px=0, editable=editable)
    assert torch.equal(out[:, 20:28, 20:28], canvas[:, 20:28, 20:28]) and torch.equal(out[:, :16], canvas[:, :16])
    assert torch.equal(out[:, 16:20, 16:48], torch.ones(3, 4, 32))
    with pytest.raises(ValueError):
        paste_window_strictly_local(canvas, crop, (0.25, 0.25, 0.75, 0.75), editable=torch.full((64, 64), 0.5))
    assert float(paste_window_strictly_local(torch.zeros(3, 20, 20), torch.ones(3, 2, 2), (0.0, 0.0, 1.0, 1.0)).max()) == 1.0


def test_eq4_eq5_subgraph_isolation_and_zero_delta():
    dag = build_mock_dag()
    affected = compute_affected_subgraph(dag, ControllerAction('modify_attr', (0.04, 0.04, 0.14, 0.14), 'fix v1'), 'v1')
    assert {'v1', 'v2', 'v3'} <= affected and not {'v9', 'v10'} & affected
    previous = dict.fromkeys(dag.nodes, 0.75)
    scores = TwoStageVerifier(lambda image, question: 0.85).verify_subgraph_after_edit(torch.rand(3, 128, 128), dag, previous, affected)
    assert all(scores[n] - previous[n] == 0 for n in dag.nodes if n not in affected)

def test_eq27_global_denominator_invariance():
    dag = build_mock_dag()
    previous = dict.fromkeys(dag.nodes, 0.5)
    delivered = dict(previous, v1=0.9)
    small = compute_subgraph_reward(dag, previous, delivered, set(), {'v1', 'v2'})
    large = compute_subgraph_reward(dag, previous, delivered, set(), {'v1', 'v2', 'v3', 'v4', 'v5'})
    assert abs(small - large) < 1e-8 and abs(small - (0.4 / 10 - 0.01)) < 1e-8

def test_eq34_eq35_pairwise_advantage_zero_sum_and_all_fail_silence():
    assert compute_pairwise_advantages([-0.05, 0.12, 0.01, -0.02, 0.18, -0.05]).sum().abs() < 1e-6
    assert torch.all(compute_pairwise_advantages([-0.010, -0.012, -0.009, -0.011, -0.010, -0.013]) == 0)

def test_eq39_controller_rl_loss_independent_of_padding_length():
    torch.manual_seed(0)
    new, reference, tokens = torch.randn(4, 60, 32), torch.randn(4, 60, 32), torch.randint(0, 32, (4, 60))
    mask, advantage = torch.zeros(4, 59), torch.tensor([0.5, -0.5, 0.2, -0.2])
    mask[:, 10:30] = 1
    loss, _ = compute_controller_rl_loss(new, new, reference, tokens, mask, advantage)
    extended = torch.cat([new, torch.randn(4, 40, 32)], 1)
    padded, _ = compute_controller_rl_loss(extended, extended, torch.cat([reference, torch.randn(4, 40, 32)], 1), torch.cat([tokens, torch.zeros(4, 40, dtype=torch.long)], 1), torch.cat([mask, torch.zeros(4, 40)], 1), advantage)
    assert abs(loss.item() - padded.item()) < 1e-6

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_eq50_eq51_flow_sde_masked_logprob_gradient(dtype):
    z = torch.randn(1, 3, 6, 8, dtype=dtype)
    velocity = torch.randn_like(z, requires_grad=True)
    mean, sigma = compute_flow_sde_drift_mean(z, velocity, torch.tensor([0.5]), 0.02)
    mask = torch.zeros(1, 1, 6, 8)
    mask[:, :, 2:4, 2:6] = 1
    target = torch.randn_like(z)
    changed = target + 100 * (1 - mask)
    first = compute_masked_gaussian_logprob(target, mean, sigma, mask, 0.02)
    assert torch.equal(first, compute_masked_gaussian_logprob(changed, mean, sigma, mask, 0.02))
    first.sum().backward()
    assert torch.isfinite(first).all() and torch.isfinite(velocity.grad).all()
    assert velocity.grad.abs().sum() > 0 and torch.count_nonzero(velocity.grad * (1 - mask)) == 0

def test_topological_gating_and_outside_nodes_are_never_scored():
    calls = []
    def score(image, question):
        calls.append(question)
        return {'root': 0.8, 'middle': 0.1, 'leaf': 0.9}[question]
    nodes = {n: ConditionNode(n, kind, n, n) for n, kind in [('leaf', 'entity'), ('middle', 'attribute'), ('root', 'entity'), ('outside', 'entity')]}
    dag = ConditionDAG(nodes, comp_edges=[('root', 'middle'), ('middle', 'leaf')])
    previous = dict.fromkeys(nodes, 0.9)
    scores = TwoStageVerifier(score).verify_subgraph_after_edit(torch.rand(3, 12, 20), dag, previous, {'root', 'middle', 'leaf'})
    assert calls == ['root', 'middle'] and scores['leaf'] == 0 and scores['outside'] == previous['outside']
    assert previous == dict.fromkeys(nodes, 0.9)

def test_two_stage_uses_unpadded_original_crop_only_after_alarm():
    seen = []
    def score(image, question):
        seen.append(image.clone())
        return 0.2 if len(seen) == 1 else 0.95
    image = torch.arange(9600).reshape(3, 40, 80).float() / 9600
    node = ConditionNode('n', 'attribute', 'question', 'entity', (0.25, 0.25, 0.75, 0.75), scope='local')
    verifier = TwoStageVerifier(score)
    assert verifier.verify_single_node_two_stage(image, node) == 0.95
    assert len(seen) == 2 and torch.equal(seen[1], image[:, 8:32, 16:64])
    verifier.verify_single_node_two_stage(image, node)
    assert len(seen) == 3

@pytest.mark.parametrize('kind,phrase', [('prohibition', 'red object'), ('count', 'ball')])
def test_remote_unique_red_condition_rejects_and_rolls_back(kind, phrase):
    left, right = (0.1, 0.1, 0.3, 0.3), (0.7, 0.7, 0.9, 0.9)
    nodes = {n: ConditionNode(n, 'entity', n, n, b, scope='local') for n, b in [('cup', left), ('ball', right)]}
    nodes['unique'] = ConditionNode('unique', kind, 'only red object', phrase, scope='global', depends_on=['cup'])
    dag, canvas, seen = ConditionDAG(nodes), torch.zeros(3, 20, 20), []
    canvas[0, 2:6, 2:6] = 1
    parent = CanvasState(canvas, {'cup': 1.0, 'ball': 0.1, 'unique': 1.0}, {'cup', 'unique'})
    def score(image, question):
        seen.append(image.shape)
        return 1.0 if question == 'ball' else float(((image[0] > 0.5) & (image[1] < 0.5)).sum() == 16)
    harness = ReDAGHarness(TwoStageVerifier(score), None, feather_px=0)
    delivered, reward, accepted, affected = harness.evaluate_candidate_action(parent, dag, 'ball', ControllerAction('add_entity', right, 'Add a red ball.'), torch.tensor([1., 0., 0.])[:, None, None].expand(3, 4, 4))
    assert affected == {'ball', 'unique'} and harness.last_candidate['raw_scores']['unique'] == 0
    assert not accepted and reward == -0.01 and delivered.scores == parent.scores and delivered.step_idx == 1
    assert torch.equal(delivered.image, canvas) and seen == [canvas.shape, canvas.shape]
    assert torch.equal(harness.last_candidate['candidate_state'].image[:, :14], canvas[:, :14])

def test_controller_gradients_and_clip_are_action_only_with_frozen_references():
    new, old, reference = [torch.randn(2, 9, 17, requires_grad=True) for _ in range(3)]
    tokens, advantage = torch.randint(0, 17, (2, 9)), torch.tensor([0.4, -0.4], requires_grad=True)
    mask = torch.zeros(2, 8)
    mask[:, 2:5] = 1
    loss, metrics = compute_controller_rl_loss(new, old, reference, tokens, mask, advantage)
    loss.backward()
    assert torch.isfinite(new.grad).all() and new.grad[:, :-1][mask.bool()].abs().sum() > 0
    assert torch.count_nonzero(new.grad[:, :-1][~mask.bool()]) == 0 and torch.count_nonzero(new.grad[:, -1]) == 0
    assert old.grad is None and reference.grad is None and advantage.grad is None
    assert metrics['ctrl_active_tokens'] == 6 and metrics['ctrl_kl_sft'] >= 0
    confident = torch.tensor([[[0.0, 4.0], [0.0, 0.0]]], requires_grad=True)
    clipped, _ = compute_controller_rl_loss(confident, torch.zeros_like(confident), confident.detach(), torch.tensor([[0, 1]]), torch.ones(1, 1), torch.ones(1), beta_kl=0)
    clipped.backward()
    assert abs(clipped.item() + 1.2) < 1e-6 and torch.count_nonzero(confident.grad) == 0

class SmallVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, z_t, z_src, text_embeds, timesteps):
        return self.weight * z_t - 0.2 * z_src + text_embeds.mean() * 0.01

def test_flow_trajectory_seed_anchoring_and_replay_likelihood():
    from redag.models.editor import sample_flow_sde_trajectory
    from redag.learn.rl_trainer import predict_cfg_velocity
    model, source, text = SmallVelocity(), torch.rand(1, 3, 8, 12), torch.ones(1, 2, 4)
    mask = torch.zeros(1, 1, 8, 12)
    mask[:, :, 2:6, 3:9] = 1
    first = sample_flow_sde_trajectory(model, lambda z: z, source, text, torch.zeros_like(text), mask, seed=37)
    second = sample_flow_sde_trajectory(model, lambda z: z, source, text, torch.zeros_like(text), mask, seed=37)
    assert first['strata_indices'] == second['strata_indices'] and len(set(first['strata_indices'])) == 4
    assert torch.equal(first['edited_crop'], second['edited_crop'])
    assert torch.equal(first['z_next_list'][-1] * (1 - mask[0]), source[0] * (1 - mask[0]))
    assert first['logp_old_list'][0] is None and all(not z.requires_grad for z in first['z_t_list'])
    for k in first['strata_indices']:
        t, z = first['t_steps'][k][None], first['z_t_list'][k][None]
        mean, sigma = compute_flow_sde_drift_mean(z, predict_cfg_velocity(model, z, source, text, torch.zeros_like(text), t), t, 0.02)
        logp = compute_masked_gaussian_logprob(first['z_next_list'][k][None], mean, sigma, mask, 0.02)
        assert sigma.item() > 0 and torch.allclose(logp.squeeze(), first['logp_old_list'][k], atol=1e-5)
        assert torch.equal(mean.detach()[0], first['mu_old_list'][k])

def test_sft_normalization_mask_weighting_and_half_precision_reduction():
    from redag.learn.sft_trainer import compute_autoregressive_sft_loss, compute_editor_masked_flow_sft_loss
    logits = torch.zeros(2, 5, 7, requires_grad=True)
    loss = compute_autoregressive_sft_loss(logits, torch.tensor([[-100, 1, 2, 3, 4], [-100, -100, -100, 2, -100]]))
    loss.backward()
    assert torch.allclose(loss, torch.tensor(7.0).log()) and torch.isfinite(logits.grad).all()
    source, target, mask = torch.zeros(2, 3, 4, 6), torch.randn(2, 3, 4, 6), torch.zeros(2, 1, 4, 6)
    mask[:, :, 1:3, 2:4] = 1
    model = SmallVelocity()
    torch.manual_seed(19)
    t, noise = torch.rand(2, 1, 1, 1), torch.randn_like(target)
    error = model((1-t)*target+t*noise, source, torch.zeros(1), t.flatten()) - (noise-target)
    expected = (((1+3*mask)*error.square()).sum((1, 2, 3)) / (3*(1+3*mask).sum((1, 2, 3)))).mean()
    torch.manual_seed(19)
    actual = compute_editor_masked_flow_sft_loss(model, source, target, torch.zeros(1), mask)
    assert torch.allclose(actual, expected)
    large = torch.zeros(1, 8, 128, 128, dtype=torch.float16)
    loss = compute_editor_masked_flow_sft_loss(model, large, torch.ones_like(large), torch.zeros(1), torch.ones(1, 1, 128, 128))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.weight.grad)

def test_affected_subgraph_follows_footprints_dependencies_and_protection():
    from redag.graph.dag import CANVAS, compute_active_frontier, compute_spatial_subgraph, edit_footprint
    left, right, table = (0.1, 0.1, 0.2, 0.2), (0.8, 0.8, 0.9, 0.9), (0.0, 0.15, 1.0, 0.75)
    nodes = {n: ConditionNode(n, 'entity', n, n, b, scope=s) for n, b, s in [('near', left, 'local'), ('far', right, 'global'), ('table', table, 'local')]}
    nodes['relation'] = ConditionNode('relation', 'relation', 'relation', '', (0.1, 0.1, 0.9, 0.9), scope='local', depends_on=['near', 'far'])
    nodes['span'] = ConditionNode('span', 'relation', 'span', '', (0.0, 0.1, 1.0, 0.9), scope='local', depends_on=['far', 'table'])
    nodes['far_color'] = ConditionNode('far_color', 'attribute', 'far color', 'far', None, scope='global', depends_on=['far'])
    nodes['style'] = ConditionNode('style', 'background', 'style', 'scene')
    nodes['lighting'] = ConditionNode('lighting', 'attribute', 'lighting', 'scene', scope='unknown')
    nodes['ban'] = ConditionNode('ban', 'prohibition', 'ban', 'bird', scope='global')
    nodes['near_count'] = ConditionNode('near_count', 'count', 'near count', 'near', expected_count=1, scope='global')
    nodes['cat_count'] = ConditionNode('cat_count', 'count', 'cat count', 'cat', expected_count=2, scope='global')
    nodes['ghost'] = ConditionNode('ghost', 'entity', 'ghost', 'ghost', scope='local')
    nodes['ghost_size'] = ConditionNode('ghost_size', 'attribute', 'ghost size', 'ghost', scope='local', depends_on=['ghost'])
    dag = ConditionDAG(nodes)
    assert edit_footprint(dag, 'far') == right and edit_footprint(dag, 'far_color') == right
    assert edit_footprint(dag, 'style') == CANVAS and edit_footprint(dag, 'lighting') == CANVAS
    assert edit_footprint(dag, 'ghost') is None and edit_footprint(dag, 'ghost_size') is None and edit_footprint(dag, 'span') is None
    small = ControllerAction('modify_attr', left, 'Change near.')
    protected = set(nodes) - {'near'}
    assert compute_spatial_subgraph(dag, small, 'near') == {'near'}
    assert compute_affected_subgraph(dag, small, 'near', protected) == {'near', 'relation', 'near_count', 'ban'}
    assert compute_affected_subgraph(dag, small, 'near') == {'near', 'relation', 'near_count'}
    assert compute_affected_subgraph(dag, ControllerAction('add_entity', (0.4, 0.4, 0.5, 0.5), 'Add a ghost in the gap.'), 'ghost', protected) == {'ghost', 'ghost_size', 'ban'}
    large = ControllerAction('modify_attr', (0.0, 0.0, 0.5, 0.5), 'Change the left half.')
    assert compute_spatial_subgraph(dag, large, 'near') == {'near', 'table', 'style', 'lighting', 'ban'}
    dag.int_edges = [('near', 'far')]
    assert compute_affected_subgraph(dag, small, 'near') == {'near', 'far', 'far_color', 'relation', 'span', 'near_count'}
    scores = {n: 0.1 if n in ('near', 'relation') else 0.9 for n in nodes}
    assert 'relation' not in compute_active_frontier(dag, scores) and 'relation' in compute_active_frontier(dag, scores, gated=False)
    dag.comp_edges = [('near', 'relation')]
    for mode, ready in [('redag_full', False), ('spatial_local', False), ('spatial_no_gates', True), ('flat_full', True)]:
        assert ('relation' in ReDAGHarness(TwoStageVerifier(None), None, mode=mode).active_frontier(dag, scores)) == ready
    with pytest.raises(ValueError):
        ConditionDAG({'bad': ConditionNode('bad', 'attribute', 'bad', '', right, scope='local', depends_on=['missing'])})

def test_four_strata_exact_expectation_and_safe_endpoint_selection():
    from redag.learn.rl_trainer import get_stochastic_strata, strata_weights, sample_4_strata_indices
    strata, weights = get_stochastic_strata(50), strata_weights(50)
    assert list(map(len, strata)) == [12, 12, 12, 13]
    coefficients = {k: weight/len(layer) for layer, weight in zip(strata, weights) for k in layer}
    assert set(coefficients) == set(range(1, 50)) and all(abs(x-1/49) < 1e-14 for x in coefficients.values())
    assert all(k in layer for k, layer in zip(sample_4_strata_indices(50, random.Random(7)), strata))

def test_joint_gaussian_ratio_matches_density_and_masked_kl():
    from redag.learn.rl_trainer import compute_transition_log_ratio
    target, old, new, sigma, mask = torch.tensor([[[[0.3, -0.2]]]]), torch.zeros(1, 1, 1, 2), torch.full((1, 1, 1, 2), 0.1), torch.tensor([0.4]), torch.ones(1, 1, 1, 2)
    log_ratio, kl = compute_transition_log_ratio(target, new, old, sigma, mask, 0.02)
    def distribution(mean):
        return torch.distributions.Normal(mean.double(), sigma.double().view(1, 1, 1, 1) * 0.02**0.5)
    expected = (distribution(new).log_prob(target.double())-distribution(old).log_prob(target.double())).mean()
    assert torch.allclose(log_ratio.squeeze(), expected) and torch.allclose(kl.squeeze(), torch.tensor(1.5625, dtype=torch.float64))
    with pytest.raises(ValueError):
        compute_masked_gaussian_logprob(target, old, torch.zeros(1), mask, 0.02)

def test_multiturn_new_dependency_invalidates_previously_satisfied_child():
    nodes = {n: ConditionNode(n, 'entity', n, n) for n in ('root', 'child')}
    graphs = iter([ConditionDAG(nodes), ConditionDAG(nodes, comp_edges=[('root', 'child')])])
    planner = SimpleNamespace(plan=lambda prompt, previous_dag, **kwargs: next(graphs))
    verifier = TwoStageVerifier(lambda image, question: 0.1 if question == 'root' else 0.9)
    state, dag, trace = ReDAGHarness(verifier, None).run_conversation(torch.zeros(3, 8, 12), ['first', 'add dependency'], planner, None, None, max_steps=0)
    assert state.scores == {'root': 0.1, 'child': 0.0} and not state.protected_set and not trace
    from redag.models.planner import preserve_previous_conditions
    revised = preserve_previous_conditions({'nodes': [{'node_id': 'root', 'node_type': 'entity', 'question': 'Is the cup blue?', 'entity_phrase': 'cup'}], 'superseded_ids': ['root']}, ConditionDAG(nodes))
    assert revised.nodes['root'].question == 'Is the cup blue?' and revised.nodes['child'].question == 'child' and nodes['root'].question == 'root'

@pytest.mark.parametrize('mode,box,two_stage', [('flat_full', (0.0, 0.0, 1.0, 1.0), False), ('tree_rollback_only', (0.0, 0.0, 1.0, 1.0), False),
                                                  ('redag_full', (0.1, 0.1, 0.2, 0.2), True)])
def test_baselines_edit_the_whole_image_and_verify_in_one_stage(mode, box, two_stage):
    nodes = {'a': ConditionNode('a', 'entity', 'a', 'a', (0.1, 0.1, 0.2, 0.2), scope='local')}
    parent = CanvasState(torch.zeros(3, 20, 20), {'a': 0.1}, set())
    seen = []
    controller = SimpleNamespace(sample=lambda state, graph, target, **kwargs: {'action': ControllerAction('modify_attr', (0.1, 0.1, 0.2, 0.2), 'Fix it.')})
    editor = SimpleNamespace(render=lambda image, action, seed: seen.append(action.box) or torch.ones(3, 4, 4))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.9), None, mode=mode)
    candidates = harness.sample_group(parent, ConditionDAG(nodes), controller, editor, 'a', group_size=1)
    assert seen == [box] and candidates[0]['action'].box == box and harness.verifier.two_stage is two_stage

def test_session_moves_on_after_two_failed_attempts_on_one_target():
    nodes = {'a': ConditionNode('a', 'entity', 'a', 'a', (0.1, 0.1, 0.2, 0.2), weight=2.0, scope='local'),
             'b': ConditionNode('b', 'entity', 'b', 'b', (0.7, 0.7, 0.8, 0.8), scope='local')}
    dag = ConditionDAG(nodes)
    parent = CanvasState(torch.zeros(3, 20, 20), {'a': 0.1, 'b': 0.1}, set())
    controller = SimpleNamespace(sample=lambda state, graph, target, **kwargs: {'action': ControllerAction('modify_attr', graph.nodes[target].bbox, 'Fix it.')})
    editor = SimpleNamespace(render=lambda *args, **kwargs: torch.ones(3, 4, 4))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.9 if question == 'b' else 0.1), None)
    state, _, trace = harness.run_session(parent, dag, controller, editor, max_steps=4, group_size=1)
    assert [step['target_node_id'] for step in trace] == ['a', 'a', 'b', 'a'] and state.scores['b'] == 0.9
    harness.retry_limit = 10
    _, _, trace = harness.run_session(parent, dag, controller, editor, max_steps=4, group_size=1)
    assert [step['target_node_id'] for step in trace] == ['a', 'a', 'a', 'a']

def test_cost_budget_counts_generations_and_verifier_calls_together():
    nodes = {f'n{i}': ConditionNode(f'n{i}', 'entity', f'n{i}', f'n{i}', (0.8, 0.8, 0.9, 0.9) if i else (0.1, 0.1, 0.2, 0.2), scope='local') for i in range(20)}
    dag = ConditionDAG(nodes)
    parent = CanvasState(torch.zeros(3, 20, 20), {n: 0.1 if n == 'n0' else 0.9 for n in nodes}, set(nodes) - {'n0'})
    controller = SimpleNamespace(sample=lambda state, graph, target, **kwargs: {'action': ControllerAction('add_entity', (0.1, 0.1, 0.2, 0.2), 'Add it.')})
    editor = SimpleNamespace(render=lambda *args, **kwargs: torch.ones(3, 2, 2))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.2), None, mode='flat_full')
    harness.cost_budget, harness.generation_cost = 25.0, 10.0
    _, _, trace = harness.run_session(parent, dag, controller, editor, max_steps=3, group_size=3)
    assert len(trace) == 1 and harness.last_run == dict(generation_calls=1, verifier_calls=20, stop_reason='execution_budget', unknown_node_ids=[])
    assert harness.cost_origin is None
    _, _, trace = harness.run_session(parent, dag, controller, editor, max_steps=3, group_size=3)
    assert len(trace) == 1 and harness.last_run['generation_calls'] == 1
    harness.cost_budget = None
    _, _, trace = harness.run_session(parent, dag, controller, editor, max_steps=3, group_size=3)
    assert len(trace) == 3 and harness.last_run['generation_calls'] == 9


def test_per_turn_budget_resets_every_conversation_turn():
    from redag.harness.environment import run_record
    graph = {'nodes': [{'node_id': 'a', 'node_type': 'entity', 'question': 'a', 'entity_phrase': 'a', 'bbox': [0.1, 0.1, 0.2, 0.2], 'scope': 'local'}]}
    second = {'nodes': graph['nodes'] + [{'node_id': 'b', 'node_type': 'entity', 'question': 'b', 'entity_phrase': 'b', 'bbox': [0.7, 0.7, 0.8, 0.8], 'scope': 'local'}]}
    record = {'image': torch.zeros(3, 20, 20), 'turns': [{'user_prompt': 'a', 'graph': graph}, {'user_prompt': 'b', 'graph': second}]}
    controller = SimpleNamespace(sample=lambda state, dag, target, **kwargs: {'action': ControllerAction('add_entity', dag.nodes[target].bbox, 'Add it.')})
    editor = SimpleNamespace(render=lambda *args, **kwargs: torch.ones(3, 2, 2))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.2), None)
    harness.generation_budget = 2
    _, _, trace = run_record(record, harness, controller, editor, max_steps=3)
    assert harness.last_run['generation_calls'] == 2 and [s['turn'] for s in trace] == [0, 0]
    harness.budget_per_turn = True
    _, _, trace = run_record(record, harness, controller, editor, max_steps=3)
    assert harness.last_run['generation_calls'] == 4 and [s['turn'] for s in trace] == [0, 0, 1, 1]


@pytest.mark.parametrize('budget', [1, 2])
def test_local_budget_spent_per_call_and_sibling_grounding_is_restored(budget):
    nodes = {f'n{i}': ConditionNode(f'n{i}', 'entity', f'n{i}', f'n{i}', (0.8, 0.8, 0.9, 0.9) if i else (0.1, 0.1, 0.2, 0.2), scope='local') for i in range(20)}
    dag = ConditionDAG(nodes)
    parent = CanvasState(torch.zeros(3, 20, 20), {n: 0.1 if n == 'n0' else 0.9 for n in nodes}, set(nodes) - {'n0'})
    grounder = SimpleNamespace(counts={'n0': 3}, masks={}, ground=lambda image, graph, selected: graph)
    seen = []
    def sample(state, graph, target, **kwargs):
        seen.append(kwargs['detected_count'])
        grounder.counts['n0'] = 99
        return {'action': ControllerAction('add_entity', (0.1, 0.1, 0.2, 0.2), 'Add the target.')}
    results = iter([0.9] if budget == 1 else [0.2, 0.9])
    verifier = TwoStageVerifier(lambda image, question: next(results))
    harness = ReDAGHarness(verifier, grounder)
    harness.verification_budget = budget
    controller, editor = SimpleNamespace(sample=sample), SimpleNamespace(render=lambda *args, **kwargs: torch.ones(3, 4, 4))
    state, _, trace = harness.run_session(parent, dag, controller, editor, max_steps=1, group_size=3)
    assert state.scores['n0'] == 0.9 and len(trace) == 1
    assert harness.last_run['verifier_calls'] == budget and harness.last_run['generation_calls'] == 1
    assert verifier.call_limit is None and grounder.counts == {'n0': 3}
    verifier.scorer = lambda image, question: 0.9
    verifier.call_limit = None
    candidates = harness.sample_group(parent, dag, controller, editor, 'n0', group_size=2)
    assert len(candidates) == 2 and seen == [3, 3, 3] and grounder.counts == {'n0': 3}


def test_local_edits_keep_verified_protected_entities_out_of_the_editable_mask():
    nodes = {'cat': ConditionNode('cat', 'entity', 'cat', 'cat', (0.0, 0.0, 0.5, 0.5), scope='local'),
             'red': ConditionNode('red', 'attribute', 'red', 'cat', None, scope='local'),
             'cup': ConditionNode('cup', 'entity', 'cup', 'cup', (0.5, 0.0, 1.0, 0.5), scope='local'),
             'dog': ConditionNode('dog', 'entity', 'dog', 'dog', (0.5, 0.5, 1.0, 1.0), scope='local'),
             'ball': ConditionNode('ball', 'entity', 'ball', 'ball', (0.3, 0.3, 0.6, 0.6), scope='local'),
             'mat': ConditionNode('mat', 'entity', 'mat', 'mat', (0.15, 0.15, 0.45, 0.45), scope='local'),
             'sky': ConditionNode('sky', 'background', 'sky', 'sky', None, scope='global')}
    dag = ConditionDAG(nodes, comp_edges=[('cat', 'red')], int_edges=[('cat', 'cup')])
    masks = {name: torch.zeros(40, 40, dtype=torch.bool) for name in ('cat', 'cup', 'dog', 'ball', 'mat')}
    masks['cat'][2:10, 2:10], masks['cup'][2:10, 30:38], masks['dog'][30:38, 30:38] = True, True, True
    masks['ball'][12:24, 12:24], masks['mat'][6:18, 6:18] = True, True
    grounder = SimpleNamespace(counts={}, masks=masks, ground=lambda image, graph, selected=None: graph)
    scores = {'cat': 0.9, 'red': 0.1, 'cup': 0.9, 'dog': 0.9, 'ball': 0.9, 'mat': 0.9, 'sky': 0.1}
    state = CanvasState(torch.zeros(3, 40, 40), scores, {'cat', 'cup', 'dog', 'ball', 'mat'})
    action = ControllerAction('modify_attr', (0.2, 0.2, 0.4, 0.4), 'Fix it.')
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.9), grounder)
    harness.mask_margin, harness.min_editable_pixels = 2, 10
    assert harness.editable_mask(state, dag, 'red', action) is None
    harness.protect_masks = True
    red = harness.editable_mask(state, dag, 'red', action)
    assert red[14, 14] == 0 and red[10, 10] == 0 and red[9, 9] == 1 and red[8, 8] == 1 and red[33, 33] == 1 and red[7, 7] == 1
    sky = harness.editable_mask(state, dag, 'sky', action)
    assert sky[5, 5] == 0 and sky[14, 14] == 0 and sky[33, 33] == 1 and sky[20, 5] == 1 and sky[27, 27] == 1
    harness.min_editable_pixels = 30
    assert harness.editable_mask(state, dag, 'red', action) is None
    harness.min_editable_pixels = 10
    whole = ControllerAction('modify_attr', (0.0, 0.0, 1.0, 1.0), 'Restyle.')
    full = harness.editable_mask(state, dag, 'sky', whole)
    assert full[12, 12] == 0 and full[33, 33] == 0 and full[5, 33] == 0 and full[25, 5] == 1
    seen = []
    def render(image, action, seed, mask=None):
        seen.append(mask)
        return torch.ones(3, 8, 8)
    controller = SimpleNamespace(sample=lambda state, graph, target, **kwargs: {'action': action})
    harness.feather_px = 0
    candidates = harness.sample_group(state, dag, controller, SimpleNamespace(render=render), 'red', group_size=1)
    pasted = candidates[0]['candidate_state'].image
    assert len(candidates) == 1 and torch.equal(seen[0], red) and float(pasted[:, 14, 14].sum()) == 0 and float(pasted[:, 9, 9].sum()) == 3
    harness.mode = 'spatial_no_gates'
    assert harness.editable_mask(state, dag, 'red', action) is None


def test_local_windows_render_at_a_working_resolution_with_context(monkeypatch):
    import math
    import redag.models.editor as editor_module
    editor = object.__new__(type('CpuEditor', (editor_module.FluxEditor,), {'device': torch.device('cpu')}))
    editor.work_side, editor.context, editor.max_upscale = None, 0.25, 4.0
    editor.num_steps, editor.eta, editor.cfg_scale = 5, 0.35, 3.5
    editor.vae = SimpleNamespace(config=SimpleNamespace(block_out_channels=[1, 2, 3, 4]))
    editor.encode_image = lambda crop: torch.zeros(1, 4, math.ceil(crop.shape[-2] / 16), math.ceil(crop.shape[-1] / 16))
    editor.encode_text = lambda text: torch.zeros(1, 2, 8)
    image, action = torch.rand(3, 96, 96), ControllerAction('modify_attr', (0.25, 0.25, 0.5, 0.5), 'Change it.')
    assert editor._geometry(image, action) == ((24, 24, 48, 48), (24, 24, 48, 48), (24, 24))
    assert editor.prepare(image, action)['mask_latent'].all()
    editor.work_side = 512
    assert editor._geometry(image, action) == ((24, 24, 48, 48), (18, 18, 54, 54), (144, 144))
    mask = editor.prepare(image, action)['mask_latent']
    assert mask.shape == (1, 1, 9, 9) and mask[0, 0, 4, 4] == 1 and mask[0, 0, 0, 0] == 0 and mask[0, 0, 8, 8] == 0
    seen = {}
    def fake_trajectory(model, decoder, **kwargs):
        seen.update(kwargs)
        return {'edited_crop': torch.full((3, 144, 144), 0.5)}
    monkeypatch.setattr(editor_module, 'sample_flow_sde_trajectory', fake_trajectory)
    crop = editor.sample(image, action, seed=1)['edited_crop']
    assert crop.shape == (3, 24, 24) and torch.allclose(crop, torch.full_like(crop, 0.5)) and seen['mask_latent'].shape == (1, 1, 9, 9)
    whole = ControllerAction('modify_attr', (0.0, 0.0, 1.0, 1.0), 'Change it.')
    assert editor._geometry(image, whole) == ((0, 0, 96, 96), (0, 0, 96, 96), (384, 384))


def test_requirement_statements_turn_questions_into_targets():
    from redag.harness.environment import requirement_statement
    def node(question, phrase):
        return ConditionNode('n', 'attribute', question, phrase)
    assert requirement_statement(node('Is there a cup in the image?', 'cup')) == 'there is a cup in the image'
    assert requirement_statement(node('Is the cup red?', 'cup')) == 'the cup is red'
    assert requirement_statement(node('Are the apples green?', 'apples')) == 'the apples are green'
    assert requirement_statement(node('Does the cup have a handle?', 'cup')) == 'the cup has a handle'
    assert requirement_statement(node('Is it sunny?', 'sky')) == 'the answer to "Is it sunny?" is yes'


def test_closure_instructions_fold_pending_children_into_the_entity_edit():
    nodes = {'cup': ConditionNode('cup', 'entity', 'Is there a cup in the image?', 'cup', (0.1, 0.1, 0.4, 0.4), scope='local'),
             'red': ConditionNode('red', 'attribute', 'Is the cup red?', 'cup', scope='local'),
             'handle': ConditionNode('handle', 'attribute', 'Does the cup have a handle?', 'cup', scope='local'),
             'on': ConditionNode('on', 'relation', 'Is the cup on the table?', 'cup', (0.0, 0.1, 1.0, 1.0), scope='local', depends_on=['cup', 'table']),
             'table': ConditionNode('table', 'entity', 'Is there a table?', 'table', (0.0, 0.3, 1.0, 1.0), scope='local'),
             'dog': ConditionNode('dog', 'entity', 'Is there a dog?', 'dog', (0.6, 0.6, 0.9, 0.9), scope='local'),
             'brown': ConditionNode('brown', 'attribute', 'Is the dog brown?', 'dog', scope='local'),
             'sky': ConditionNode('sky', 'background', 'Is the sky blue?', 'sky', scope='global')}
    dag = ConditionDAG(nodes, comp_edges=[('cup', 'red'), ('cup', 'handle'), ('dog', 'brown')])
    scores = {'cup': 0.1, 'red': 0.1, 'handle': 0.2, 'on': 0.1, 'table': 0.9, 'dog': 0.9, 'brown': 0.1, 'sky': 0.1}
    state = CanvasState(torch.zeros(3, 20, 20), scores, {'table', 'dog'})
    seen = []
    controller = SimpleNamespace(sample=lambda state, graph, target, **kwargs: {'action': ControllerAction('add_entity', (0.1, 0.1, 0.4, 0.4), 'Add a cup.')})
    editor = SimpleNamespace(render=lambda image, action, seed: seen.append(action.instruction) or torch.ones(3, 6, 6))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.2), None)
    harness.sample_group(state, dag, controller, editor, 'cup', group_size=1)
    harness.closure_instructions, harness.closure_limit = True, 2
    candidates = harness.sample_group(state, dag, controller, editor, 'cup', group_size=1)
    assert seen[0] == 'Add a cup.' and candidates[0]['sample']['closure_nodes'] == ['red', 'on']
    assert seen[1] == 'Add a cup. In the same edit, also make sure that the cup is red; the cup is on the table.'
    harness.closure_limit = 3
    assert harness.closure_requirements(state, dag, 'cup', candidates[0]['action']) == ['red', 'on', 'handle']
    assert harness.closure_requirements(state, dag, 'brown', ControllerAction('modify_attr', (0.6, 0.6, 0.9, 0.9), 'x')) == []
    assert harness.closure_requirements(state, dag, 'cup', ControllerAction('add_entity', (0.7, 0.0, 1.0, 0.2), 'x')) == ['on']
    harness.mode = 'spatial_no_gates'
    harness.sample_group(state, dag, controller, editor, 'cup', group_size=1)
    assert seen[2] == 'Add a cup.'


@pytest.mark.parametrize('box,confirm,accepted,refuted', [((0.7, 0.7, 0.9, 0.9), False, False, []), ((0.7, 0.7, 0.9, 0.9), True, True, ['cup']),
                                                          ((0.5, 0.1, 0.9, 0.5), True, False, [])])
def test_confirm_drops_restores_untouched_remote_conditions_only(box, confirm, accepted, refuted):
    nodes = {'ball': ConditionNode('ball', 'entity', 'ball', 'ball', box, scope='local'),
             'cup': ConditionNode('cup', 'entity', 'cup', 'cup', (0.1, 0.1, 0.5, 0.5), scope='local'),
             'red': ConditionNode('red', 'attribute', 'red', 'cup', scope='local')}
    dag = ConditionDAG(nodes, comp_edges=[('cup', 'red')], int_edges=[('ball', 'cup')])
    parent = CanvasState(torch.zeros(3, 20, 20), {'ball': 0.1, 'cup': 0.95, 'red': 0.9}, {'cup', 'red'})
    calls = []
    def score(image, question):
        calls.append(question)
        if question == 'cup':
            return 0.3 if box[0] > 0.6 else 0.2 if bool((image[0] > 0.5).any()) else 0.95
        return 0.9
    harness = ReDAGHarness(TwoStageVerifier(score), None, feather_px=0)
    harness.confirm_drops = confirm
    action = ControllerAction('add_entity', box, 'Add a red ball.')
    delivered, _, ok, affected = harness.evaluate_candidate_action(parent, dag, 'ball', action, torch.tensor([1., 0., 0.])[:, None, None].expand(3, 8, 8))
    assert affected == {'ball', 'cup', 'red'} and ok is accepted and harness.last_candidate['refuted_drops'] == refuted
    if accepted:
        assert delivered.scores == {'ball': 0.9, 'cup': 0.95, 'red': 0.9} and calls.count('red') == 1
    if box[0] < 0.6:
        assert harness.last_candidate['raw_scores']['cup'] == 0.2 and harness.last_candidate['raw_scores']['red'] == 0.0


def test_soft_gating_only_waits_for_entity_presence():
    nodes = {'cup': ConditionNode('cup', 'entity', 'cup', 'cup'), 'red': ConditionNode('red', 'attribute', 'red', 'cup'),
             'shiny': ConditionNode('shiny', 'attribute', 'shiny', 'cup', depends_on=['red']),
             'bowl': ConditionNode('bowl', 'entity', 'bowl', 'bowl'), 'blue': ConditionNode('blue', 'attribute', 'blue', 'bowl', depends_on=['bowl'])}
    dag = ConditionDAG(nodes, comp_edges=[('cup', 'red')])
    scores = {'cup': 0.9, 'red': 0.1, 'shiny': 0.1, 'bowl': 0.1, 'blue': 0.1}
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.5), None)
    assert sorted(harness.active_frontier(dag, scores)) == ['bowl', 'red']
    harness.soft_gating = True
    frontier = harness.active_frontier(dag, scores)
    assert sorted(frontier[:2]) == ['bowl', 'red'] and frontier[2:] == ['shiny']
    harness.mode = 'spatial_no_gates'
    assert sorted(harness.active_frontier(dag, scores)) == ['blue', 'bowl', 'red', 'shiny']
