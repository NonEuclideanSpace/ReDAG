from types import SimpleNamespace
import torch
from redag.graph.schema import CanvasState, ConditionDAG, ConditionNode, ControllerAction
from redag.harness.environment import ReDAGHarness, run_record
from redag.learn.strategy import ACCEPT_FEATURES, ROUTE_FEATURES, StrategyPolicy, default_weights, group_advantages, policy_gradient
from redag.verify.verifier import TwoStageVerifier


def node(nid, kind, question, phrase, bbox=None, scope='local', depends_on=()):
    return ConditionNode(nid, kind, question, phrase, bbox, scope=scope, depends_on=list(depends_on))


def graph_dict(*nodes):
    return {'nodes': [dict(node_id=n.node_id, node_type=n.node_type, question=n.question, entity_phrase=n.entity_phrase, bbox=n.bbox,
                           weight=n.weight, threshold=n.threshold, scope=n.scope, depends_on=n.depends_on) for n in nodes]}


def test_policy_gradient_moves_toward_the_better_branch_only():
    weights = default_weights()
    features = [1.0] + [0.0] * (len(ACCEPT_FEATURES) - 1)
    rollouts = group_advantages([dict(state_id='s', reward=1.0, decisions=[dict(kind='accept', features=features, action=1, p=0.3)]),
                                 dict(state_id='s', reward=0.0, decisions=[dict(kind='accept', features=features, action=0, p=0.3)]),
                                 dict(state_id='t', reward=2.0, decisions=[dict(kind='accept', features=features, action=0, p=0.3)])])
    assert rollouts[2]['advantage'] == 0.0
    updated = policy_gradient(weights, rollouts, lr=1.0, l2=0.0)
    assert updated['accept'][0] > weights['accept'][0] and updated['route'] == weights['route']
    assert policy_gradient(weights, [dict(rollouts[0], advantage=0.0)], lr=1.0, l2=0.0) == weights


def test_strategy_routes_and_decides_only_ambiguous_trade_offs():
    dag = ConditionDAG({'a': node('a', 'entity', 'Is there an apple?', 'apple', (0.1, 0.1, 0.3, 0.3)),
                        'b': node('b', 'background', 'Is the sky at sunset?', 'sky', scope='global')})
    state = CanvasState(torch.zeros(3, 16, 16), {'a': 0.9, 'b': 0.1}, {'a'})
    bright = lambda image: float(image.mean()) > 0.5
    scores = lambda image, question: (0.9 if bright(image) else 0.1) if 'sky' in question else (0.2 if bright(image) else 0.9)
    harness = ReDAGHarness(TwoStageVerifier(scores), None)
    harness.turn_request = 'make it a sunset'
    policy = StrategyPolicy(sample=False)
    harness.strategy = policy
    editor = SimpleNamespace(render=lambda image, act, seed=42: torch.ones(3, 16, 16))
    first = harness.sample_group(state, dag, None, editor, 'b', group_size=1, seed=1)
    assert [d['kind'] for d in policy.log] == ['route', 'accept'] and first[0]['route'] == 'global' and not first[0]['accepted']
    assert len(policy.log[0]['features']) == len(ROUTE_FEATURES) and len(policy.log[1]['features']) == len(ACCEPT_FEATURES)
    assert policy.log[1]['features'][2] > 0 and policy.log[1]['features'][1] == 0
    harness.global_failures, policy.log = {}, []
    policy.weights['accept'][0] = 10.0
    second = harness.sample_group(state, dag, None, editor, 'b', group_size=1, seed=2)
    assert second[0]['accepted'] and [d['kind'] for d in policy.log] == ['route', 'accept']
    policy.log = []
    harness.strategy, harness.scope_routing = None, True
    third = harness.sample_group(state, dag, None, editor, 'b', group_size=1, seed=3)
    assert not third[0]['accepted'] and policy.log == []


def test_turn_start_hook_sees_every_turn_with_its_new_conditions():
    a = node('a', 'entity', 'Is there an apple?', 'apple', (0.1, 0.1, 0.3, 0.3))
    b = node('b', 'entity', 'Is there a banana?', 'banana', (0.5, 0.5, 0.9, 0.9))
    record = {'kind': 'i2i', 'image': torch.zeros(3, 16, 16),
              'turns': [{'user_prompt': 'add an apple', 'graph': graph_dict(a)}, {'user_prompt': 'add a banana', 'graph': graph_dict(b)}]}
    editor = SimpleNamespace(render=lambda image, act, seed=42: torch.ones(3, 16, 16))
    controller = SimpleNamespace(sample=lambda *args, **kwargs: {'action': ControllerAction('add_entity', (0.5, 0.5, 0.9, 0.9), 'Add it.')})
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.9), None)
    seen = []
    harness.on_turn_start = lambda h, state, dag: seen.append((h.turn_index, h.turn_count, sorted(h.turn_nodes), sorted(dag.nodes), h.turn_request))
    run_record(record, harness, controller, editor, max_steps=1)
    assert seen == [(0, 2, ['a'], ['a'], 'add an apple'), (1, 2, ['b'], ['a', 'b'], 'add a banana')]
    assert harness.budget_left() == 1.0


def test_rl_sessions_are_selected_per_stratum_independent_of_file_order():
    import importlib.util
    import random
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('run_strategy_rl', Path(__file__).resolve().parents[1] / 'scripts' / 'run_strategy_rl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [dict(id=f'{kind}{k}-{i}', kind=kind, k=k, turns=[{}] * (1 if kind == 't2i' and i == 0 else 3))
               for kind in ('t2i', 'i2i') for k in (10, 20) for i in range(5)]
    picked = module.select_sessions(records, 2, 0)
    shuffled = records[:]
    random.Random(1).shuffle(shuffled)
    assert [r['id'] for r in module.select_sessions(shuffled, 2, 0)] == [r['id'] for r in picked]
    assert sorted((r['kind'], r['k']) for r in picked) == sorted([(kind, k) for kind in ('t2i', 'i2i') for k in (10, 20)] * 2)
    assert not any(r['kind'] == 't2i' and len(r['turns']) < 2 for r in picked)
