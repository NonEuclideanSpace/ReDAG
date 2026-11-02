from types import SimpleNamespace
import torch
from redag.graph.dag import CANVAS
from redag.graph.schema import CanvasState, ConditionDAG, ConditionNode, ControllerAction
from redag.graph.dag import compute_affected_subgraph
from redag.harness.environment import (ReDAGHarness, configure_routing, consistent_count, direct_consistent_count, joint_acceptance,
                                       net_progress_acceptance, request_snippet, route_class, run_record, valid_nodes, with_user_request)
from redag.verify.verifier import TwoStageVerifier


def node(nid, kind, question, phrase, bbox=None, scope='local', depends_on=()):
    return ConditionNode(nid, kind, question, phrase, bbox, scope=scope, depends_on=list(depends_on))


def graph_dict(*nodes):
    return {'nodes': [dict(node_id=n.node_id, node_type=n.node_type, question=n.question, entity_phrase=n.entity_phrase, bbox=n.bbox,
                           weight=n.weight, threshold=n.threshold, scope=n.scope, depends_on=n.depends_on) for n in nodes]}


def test_route_classes_cover_global_and_wide_relations():
    nodes = {
        'cup': node('cup', 'entity', 'Is there a cup?', 'cup', (0.1, 0.1, 0.2, 0.2)),
        'dog': node('dog', 'entity', 'Is there a dog?', 'dog', (0.7, 0.7, 0.9, 0.9)),
        'near': node('near', 'relation', 'Is the cup near the plate?', 'cup', (0.1, 0.1, 0.3, 0.2), depends_on=['cup', 'plate']),
        'plate': node('plate', 'entity', 'Is there a plate?', 'plate', (0.15, 0.1, 0.3, 0.2)),
        'far': node('far', 'relation', 'Is the cup left of the dog?', 'cup', (0.1, 0.1, 0.9, 0.9), depends_on=['cup', 'dog']),
        'loose': node('loose', 'relation', 'Is the cup under the lamp?', 'cup', depends_on=['cup']),
        'sky': node('sky', 'background', 'Is the sky at sunset?', 'sky', scope='global'),
        'scene': node('scene', 'entity', 'Is there a greenhouse?', 'greenhouse', scope='global'),
        'grade': node('grade', 'attribute', 'Is a warm color grade applied?', 'image', scope='global'),
        'none': node('none', 'prohibition', 'Is the image free of carrots?', 'carrot', scope='global'),
    }
    dag = ConditionDAG(nodes)
    assert route_class(dag, 'cup') == 'local' and route_class(dag, 'near') == 'relation_local'
    assert route_class(dag, 'far') == 'relation_wide' and route_class(dag, 'loose') == 'relation_wide' and route_class(dag, 'sky') == 'background'
    assert route_class(dag, 'scene') == 'entity_global' and route_class(dag, 'grade') == 'global' and route_class(dag, 'none') == 'prohibition'
    harness = configure_routing(ReDAGHarness(TwoStageVerifier(lambda image, question: 0.5), None),
                                SimpleNamespace(scope_routing=True, global_types='background,global,relation_wide'))
    assert harness.routes_globally(dag, 'far') and harness.routes_globally(dag, 'grade') and not harness.routes_globally(dag, 'scene')
    harness.mode = 'spatial_no_gates'
    assert not harness.routes_globally(dag, 'sky')


def test_request_snippet_keeps_only_sentences_about_the_target():
    nodes = {'jar': node('jar', 'entity', 'Is there a specimen jar?', 'specimen jar'),
             'root': node('root', 'attribute', 'Does the jar hold a strange root?', 'strange root', depends_on=['jar'])}
    dag = ConditionDAG(nodes)
    request = 'Also make sure that: A workbench is present. A specimen jar is present. The specimen jar contains a strange root. A rack is present.'
    assert request_snippet(request, dag, 'root') == 'A specimen jar is present. The specimen jar contains a strange root.'
    assert request_snippet('change the vegetables into broccoli', dag, 'jar') == 'change the vegetables into broccoli'
    assert request_snippet(' '.join(['word'] * 60), dag, 'jar') == ''
    assert with_user_request('Add a jar.', 'A specimen jar is present.') == 'Add a jar. The user asked: "A specimen jar is present."'
    assert with_user_request('Add a jar.', '') == 'Add a jar.'


def test_joint_gate_requires_gain_target_and_protected_floor():
    dag = ConditionDAG({'a': node('a', 'entity', 'a', 'a'), 'b': node('b', 'entity', 'b', 'b'), 'c': node('c', 'entity', 'c', 'c')})
    prev = {'a': 0.9, 'b': 0.1, 'c': 0.75}
    ok = {'a': 0.85, 'b': 0.9, 'c': 0.72}
    hurt = {'a': 0.5, 'b': 0.9, 'c': 0.9}
    assert joint_acceptance(dag, ['b'], prev, ok, {'a', 'c'}, {'a', 'b', 'c'})
    assert not joint_acceptance(dag, ['b'], prev, hurt, {'a', 'c'}, {'a', 'b', 'c'})
    assert joint_acceptance(dag, ['b'], prev, hurt, {'a', 'c'}, {'a', 'b', 'c'}, gate='mean')
    assert not joint_acceptance(dag, ['b'], prev, dict(prev, b=0.12), {'a', 'c'}, {'a', 'b', 'c'}, gate='mean')
    assert not joint_acceptance(dag, ['b'], prev, {'a': 0.2, 'b': 0.9, 'c': 0.2}, {'a', 'c'}, {'a', 'b', 'c'}, gate='mean')


def test_net_gate_counts_dependency_consistent_conditions():
    dag = ConditionDAG({'a': node('a', 'entity', 'a', 'a'), 'b': node('b', 'attribute', 'b', 'a', depends_on=['a']), 'c': node('c', 'entity', 'c', 'c')})
    prev = {'a': 0.9, 'b': 0.9, 'c': 0.1}
    nodes = {'a', 'b', 'c'}
    trade = {'a': 0.9, 'b': 0.5, 'c': 0.9}
    assert not joint_acceptance(dag, ['c'], prev, trade, {'a', 'b'}, nodes)
    assert joint_acceptance(dag, ['c'], prev, trade, {'a', 'b'}, nodes, gate='mean')
    assert not joint_acceptance(dag, ['c'], prev, trade, {'a', 'b'}, nodes, gate='net')
    assert joint_acceptance(dag, ['c'], prev, {'a': 0.9, 'b': 0.9, 'c': 0.9}, {'a', 'b'}, nodes, gate='net')
    assert not joint_acceptance(dag, ['c'], prev, {'a': 0.6, 'b': 0.95, 'c': 0.95}, {'a', 'b'}, nodes, gate='net')
    assert consistent_count(dag, {'a': 0.6, 'b': 0.95, 'c': 0.95}, nodes) == 1
    prev2 = {'a': 0.9, 'b': 0.9, 'c': 0.1, 'd': 0.1}
    dag2 = ConditionDAG({**dag.nodes, 'd': node('d', 'entity', 'd', 'd')})
    assert joint_acceptance(dag2, ['c', 'd'], prev2, {'a': 0.9, 'b': 0.5, 'c': 0.9, 'd': 0.9}, {'a', 'b'}, {'a', 'b', 'c', 'd'}, gate='net')


def test_global_retries_fall_back_to_the_local_route_within_a_turn():
    dag = ConditionDAG({'sky': node('sky', 'background', 'Is the sky at sunset?', 'sky', scope='global')})
    harness = configure_routing(ReDAGHarness(TwoStageVerifier(lambda image, question: 0.1), None),
                                SimpleNamespace(scope_routing=True, global_types='background', global_retries=1))
    state = CanvasState(torch.zeros(3, 16, 16), {'sky': 0.1}, set())
    editor = SimpleNamespace(render=lambda image, act, seed=42: torch.ones(3, 16, 16))
    controller = SimpleNamespace(sample=lambda *args, **kwargs: {'action': ControllerAction('modify_attr', (0.0, 0.0, 1.0, 0.5), 'Make the sky orange.')})
    harness.turn_request = 'make it a sunset'
    first = harness.sample_group(state, dag, controller, editor, 'sky', group_size=1, seed=1)
    second = harness.sample_group(state, dag, controller, editor, 'sky', group_size=1, seed=2)
    assert first[0]['route'] == 'global' and not first[0]['accepted'] and second[0]['route'] == 'local'
    harness.turn_index = 1
    assert harness.routes_globally(dag, 'sky')
    harness.global_retries = 0
    harness.turn_index = 0
    assert harness.routes_globally(dag, 'sky')


def test_global_route_skips_the_controller_and_edits_the_whole_canvas():
    dag = ConditionDAG({'cup': node('cup', 'entity', 'Is there a cup?', 'cup', (0.1, 0.1, 0.4, 0.4)),
                        'sky': node('sky', 'background', 'Is the sky at sunset?', 'sky', scope='global')})
    state = CanvasState(torch.zeros(3, 16, 16), {'cup': 0.9, 'sky': 0.1}, {'cup'})
    seen = []
    editor = SimpleNamespace(render=lambda image, act, seed=42: seen.append(act) or torch.ones(3, 16, 16))
    scores = lambda image, question: 0.9 if 'cup' in question or float(image.mean()) > 0.5 else 0.1
    harness = ReDAGHarness(TwoStageVerifier(scores), None)
    harness.scope_routing, harness.turn_request = True, 'make it a sunset scene'
    candidates = harness.sample_group(state, dag, None, editor, 'sky', group_size=1, seed=7)
    assert len(candidates) == 1 and candidates[0]['route'] == 'global' and candidates[0]['accepted']
    assert seen[0].box == CANVAS and seen[0].instruction.startswith('make it a sunset scene. Make sure that')
    assert candidates[0]['affected'] == {'cup', 'sky'} and harness.generation_counter == 1


def test_turn_pass_runs_the_user_request_once_per_turn_and_counts_toward_the_budget():
    a = node('a', 'entity', 'Is there an apple?', 'apple', (0.1, 0.1, 0.3, 0.3))
    b = node('b', 'entity', 'Is there a banana?', 'banana', scope='global')
    record = {'kind': 'i2i', 'image': torch.zeros(3, 16, 16),
              'turns': [{'user_prompt': 'keep the apple', 'graph': graph_dict(a)}, {'user_prompt': 'add a banana', 'graph': graph_dict(b)}]}
    renders = []
    editor = SimpleNamespace(render=lambda image, act, seed=42: renders.append(act.instruction) or torch.ones(3, 16, 16))
    controller = SimpleNamespace(sample=lambda *args, **kwargs: {'action': ControllerAction('add_entity', (0.5, 0.5, 0.9, 0.9), 'Add a banana.')})

    def run(apple_survives, gate):
        renders.clear()
        scores = lambda image, question: (0.9 if apple_survives or float(image.mean()) < 0.5 else 0.2) if 'apple' in question else (0.9 if float(image.mean()) > 0.5 else 0.1)
        harness = ReDAGHarness(TwoStageVerifier(scores), None)
        harness.turn_pass, harness.turn_gate, harness.generation_budget = True, gate, 5
        state, dag, steps = run_record(record, harness, controller, editor, max_steps=1)
        return harness, steps

    harness, steps = run(True, 'joint')
    assert [s['route'] for s in steps] == ['turn'] and steps[0]['accepted'] and steps[0]['targets'] == ['b']
    assert renders == ['add a banana. Keep everything else in the image unchanged.'] and harness.generation_counter == 1
    harness, steps = run(False, 'joint')
    assert steps[0]['route'] == 'turn' and not steps[0]['accepted']
    harness, steps = run(False, 'mean')
    assert steps[0]['accepted']
    c = node('c', 'entity', 'Is there a cherry?', 'cherry', scope='global')
    record['turns'][0] = {'user_prompt': 'draw an apple and a cherry', 'graph': graph_dict(a, c)}
    harness, steps = run(True, 'joint')
    assert [s['route'] for s in steps] == ['turn'] and steps[0]['targets'] == ['c']
    record['kind'] = 't2i'
    harness, steps = run(True, 'joint')
    assert [s['route'] for s in steps] == ['local', 'turn']


def test_rejected_turn_pass_is_retried_with_the_failed_requirements_spelled_out():
    a = node('a', 'entity', 'Is there an apple?', 'apple', (0.1, 0.1, 0.3, 0.3))
    b = node('b', 'entity', 'Is there a banana?', 'banana', scope='global')
    record = {'kind': 'i2i', 'image': torch.zeros(3, 16, 16),
              'turns': [{'user_prompt': 'keep the apple', 'graph': graph_dict(a)}, {'user_prompt': 'add a banana', 'graph': graph_dict(b)}]}
    renders = []

    def render(image, act, seed=42):
        renders.append(act.instruction)
        return torch.full((3, 16, 16), 0.8 if 'Also keep these true' in act.instruction else 0.6)

    def scores(image, question):
        level = float(image.mean())
        if 'apple' in question:
            return 0.9 if level < 0.5 or level > 0.7 else 0.2
        return 0.9 if level > 0.5 else 0.1

    editor = SimpleNamespace(render=render)
    controller = SimpleNamespace(sample=lambda *args, **kwargs: {'action': ControllerAction('add_entity', (0.5, 0.5, 0.9, 0.9), 'Add a banana.')})
    for retries, routes in ((1, ['turn', 'turn']), (0, ['turn', 'local'])):
        renders.clear()
        harness = configure_routing(ReDAGHarness(TwoStageVerifier(scores), None), SimpleNamespace(turn_pass=True, turn_retries=retries))
        harness.generation_budget = 5
        state, dag, steps = run_record(record, harness, controller, editor, max_steps=2)
        assert [s['route'] for s in steps] == routes and not steps[0]['accepted']
        if retries:
            assert steps[1]['accepted'] and state.scores['b'] >= 0.7 and state.scores['a'] >= 0.7
            assert renders[1] == 'add a banana. Keep everything else in the image unchanged. Also keep these true: there is an apple.'
            assert harness.turn_retry_source(steps) is None


def test_turn_priority_targets_the_new_request_before_older_open_conditions():
    a = node('a', 'entity', 'Is there an apple?', 'apple', (0.1, 0.1, 0.3, 0.3))
    c = node('c', 'entity', 'Is there a cherry?', 'cherry', (0.6, 0.6, 0.9, 0.9))
    record = {'kind': 'i2i', 'image': torch.zeros(3, 16, 16),
              'turns': [{'user_prompt': 'add an apple', 'graph': graph_dict(a)}, {'user_prompt': 'add a cherry', 'graph': graph_dict(a, c)}]}
    targets = []

    def sample(state, graph, target, **kwargs):
        targets.append(target)
        return {'action': ControllerAction('add_entity', graph.nodes[target].bbox, 'Add it.')}

    editor = SimpleNamespace(render=lambda image, act, seed=42: torch.zeros(3, 16, 16))
    for priority, expected in ((False, ['a', 'a']), (True, ['a', 'c'])):
        targets.clear()
        harness = configure_routing(ReDAGHarness(TwoStageVerifier(lambda image, question: 0.1), None), SimpleNamespace(turn_priority=priority))
        run_record(record, harness, SimpleNamespace(sample=sample), editor, max_steps=1)
        assert targets == expected


def chain_dag():
    nodes = {'a': node('a', 'entity', 'a', 'a'), 'b': node('b', 'attribute', 'b', 'a', depends_on=['a']),
             'c': node('c', 'attribute', 'c', 'a', depends_on=['b']), 'd': node('d', 'attribute', 'd', 'a', depends_on=['c'])}
    nodes.update({k: node(k, 'entity', k, k) for k in 'efg'})
    return ConditionDAG(nodes)


def test_consistency_is_recursive_over_the_whole_graph():
    dag = chain_dag()
    before = dict.fromkeys('abcd', 0.71) | dict.fromkeys('efg', 0.69)
    after = dict(a=0.69, b=0.71, c=0.71, d=0.71) | dict.fromkeys('efg', 0.99)
    assert consistent_count(dag, before) == 4 and consistent_count(dag, after) == 3 and valid_nodes(dag, after) == set('efg')
    assert direct_consistent_count(dag, after, dag.nodes) == 5
    assert not net_progress_acceptance(dag, ['e'], before, after)
    assert consistent_count(dag, after, {'c', 'd', 'e'}) == 1
    diamond = ConditionDAG({'p': node('p', 'entity', 'p', 'p'), 'q': node('q', 'entity', 'q', 'q'),
                            'r': node('r', 'relation', 'r', 'p', depends_on=['p', 'q']), 's': node('s', 'attribute', 's', 'p', depends_on=['r'])})
    scores = dict(p=0.9, q=0.5, r=0.9, s=0.9)
    assert valid_nodes(diamond, scores) == {'p'} and consistent_count(diamond, dict(scores, q=0.9)) == 4


def test_net2_gate_counts_and_never_trades_down():
    dag = ConditionDAG({'x': node('x', 'entity', 'x', 'x'), 'y': node('y', 'entity', 'y', 'y')})
    before, after = dict(x=0.69, y=0.99), dict(x=0.71, y=0.70)
    assert consistent_count(dag, before) == 1 and consistent_count(dag, after) == 2
    assert net_progress_acceptance(dag, ['x'], before, after)
    assert not joint_acceptance(dag, ['x'], before, after, {'y'}, {'x', 'y'}, gate='net')
    assert joint_acceptance(dag, ['x'], before, after, {'y'}, {'x', 'y'}, gate='net2')
    assert net_progress_acceptance(dag, ['x'], before, dict(before, x=0.6)) is False
    assert net_progress_acceptance(dag, ['x'], dict(x=0.3, y=0.99), dict(x=0.5, y=0.98))
    assert not net_progress_acceptance(dag, ['x'], dict(x=0.3, y=0.99), dict(x=0.5, y=0.6))
    swap = ConditionDAG({**dag.nodes, 'z': node('z', 'entity', 'z', 'z')})
    assert not net_progress_acceptance(swap, ['x'], dict(x=0.3, y=0.99, z=0.9), dict(x=0.9, y=0.5, z=0.9))


def test_all_routes_share_the_net2_decision():
    dag = ConditionDAG({'cup': node('cup', 'entity', 'Is there a cup?', 'cup', (0.1, 0.1, 0.4, 0.4)),
                        'mug': node('mug', 'entity', 'Is there a mug?', 'mug', (0.5, 0.5, 0.9, 0.9)),
                        'tea': node('tea', 'entity', 'Is there tea?', 'tea', (0.5, 0.1, 0.9, 0.4))})
    state = CanvasState(torch.zeros(3, 16, 16), {'cup': 0.72, 'mug': 0.95, 'tea': 0.1}, {'cup', 'mug'})
    bright = lambda image: float(image.mean()) > 0.5
    scores = lambda image, question: (0.9 if bright(image) else 0.1) if 'tea' in question else (0.69 if bright(image) else 0.72) if 'cup' in question else 0.95
    decisions = []
    for route in ('local', 'global', 'turn'):
        harness = configure_routing(ReDAGHarness(TwoStageVerifier(scores), None, feather_px=0),
                                    SimpleNamespace(turn_gate='net2', global_gate='net2', local_gate='net2'))
        action = ControllerAction('add_entity', (0.0, 0.0, 1.0, 1.0), 'Add tea.')
        harness.evaluate_candidate_action(state, dag, 'tea', action, torch.ones(3, 16, 16), None, route=route, targets=['tea'])
        decisions.append(harness.last_candidate['accepted'])
    assert decisions == [False, False, False]
    harness = configure_routing(ReDAGHarness(TwoStageVerifier(scores), None, feather_px=0), SimpleNamespace())
    harness.evaluate_candidate_action(state, dag, 'tea', ControllerAction('add_entity', (0.0, 0.0, 1.0, 1.0), 'Add tea.'), torch.ones(3, 16, 16), None, route='local')
    assert harness.last_candidate['accepted']


def test_affected_set_closes_over_children_of_count_and_prohibition_nodes():
    dag = ConditionDAG({'dog': node('dog', 'entity', 'Is there a dog?', 'dog', (0.1, 0.1, 0.4, 0.4)),
                        'two': node('two', 'count', 'Are there two dogs?', 'dog', scope='global'),
                        'kid': node('kid', 'relation', 'Is a kid next to the two dogs?', 'kid', scope='global', depends_on=['two'])})
    affected = compute_affected_subgraph(dag, ControllerAction('modify_attr', (0.1, 0.1, 0.4, 0.4), 'Edit the dog.'), 'dog')
    assert {'dog', 'two', 'kid'} <= affected


def test_localized_turn_pass_edits_where_the_replaced_object_is():
    from redag.verify.grounder import Grounder
    dag = ConditionDAG({'no': node('no', 'prohibition', 'Is the image free of carrots?', 'carrot', scope='global'),
                        'new': node('new', 'entity', 'Is there broccoli?', 'broccoli', (0.6, 0.6, 0.9, 0.9)),
                        'bowl': node('bowl', 'entity', 'Is there a bowl?', 'bowl', (0.1, 0.4, 0.3, 0.6)),
                        'rice': node('rice', 'entity', 'Is there rice in the bowl?', 'rice', (0.7, 0.0, 0.9, 0.2), depends_on=['bowl']),
                        'next': node('next', 'relation', 'Is the broccoli next to the bowl?', 'broccoli', depends_on=['new', 'bowl']),
                        'sky': node('sky', 'background', 'Is the sky blue?', 'sky', scope='global')})
    scores = {'no': 0.1, 'new': 0.1, 'bowl': 0.9, 'rice': 0.1, 'next': 0.1, 'sky': 0.9}
    image, found = torch.zeros(3, 16, 16), {'carrot': [[3.2, 8.0, 6.4, 11.2], [3.2, 8.0, 6.4, 11.3], [10, 10, 9, 12]]}
    grounder = Grounder(lambda img, phrase: found.get(phrase, []))
    assert grounder.locate(image, 'carrot') == [(0.2, 0.5, 0.4, 0.7)]
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.5), grounder)
    assert harness.turn_box(dag, ['no', 'new'], image, scores) == CANVAS
    harness = configure_routing(harness, SimpleNamespace(turn_localize=True))
    assert [round(x, 4) for x in harness.turn_box(dag, ['no', 'new'], image, scores)] == [0.175, 0.475, 0.425, 0.725]
    assert [round(x, 4) for x in harness.turn_box(dag, ['no', 'new', 'next'], image, scores)] == [0.0625, 0.3625, 0.4375, 0.7375]
    assert [round(x, 4) for x in harness.turn_box(dag, ['rice'], image, scores)] == [0.075, 0.375, 0.325, 0.625]
    assert harness.turn_box(dag, ['sky', 'new'], image, scores) == CANVAS
    found['carrot'] = []
    assert harness.turn_box(dag, ['no', 'new'], image, scores) == CANVAS
    assert harness.turn_box(dag, ['new'], image, scores) == CANVAS


def test_protected_turn_pass_keeps_unrelated_satisfied_objects_out_of_the_edit():
    dag = ConditionDAG({'cat': node('cat', 'entity', 'Is there a cat?', 'cat', (0.0, 0.0, 0.5, 0.5)),
                        'cup': node('cup', 'entity', 'Is there a cup?', 'cup', (0.5, 0.0, 1.0, 0.5)),
                        'table': node('table', 'entity', 'Is there a table?', 'table', (0.0, 0.35, 1.0, 1.0)),
                        'red': node('red', 'attribute', 'Is the cat red?', 'cat', depends_on=['cat']),
                        'lamp': node('lamp', 'entity', 'Is there a lamp?', 'lamp', scope='global'),
                        'sky': node('sky', 'background', 'Is the sky dark?', 'sky', scope='global')})
    masks = {name: torch.zeros(40, 40, dtype=torch.bool) for name in ('cat', 'cup', 'table')}
    masks['cat'][2:10, 2:10], masks['cup'][2:10, 30:38], masks['table'][15:40, :] = True, True, True
    grounder = SimpleNamespace(counts={}, masks=masks, ground=lambda image, graph, selected=None: graph)
    state = CanvasState(torch.zeros(3, 40, 40), {'cat': 0.9, 'cup': 0.9, 'table': 0.9, 'red': 0.1, 'lamp': 0.1, 'sky': 0.1}, {'cat', 'cup', 'table'})
    harness = configure_routing(ReDAGHarness(TwoStageVerifier(lambda image, question: 0.9), grounder, feather_px=0), SimpleNamespace(turn_pass=True))
    harness.mask_margin, harness.min_editable_pixels = 2, 10
    action = ControllerAction('modify_attr', CANVAS, 'Make the cat red and add a lamp.')
    assert harness.turn_mask(state, dag, ['red', 'lamp'], action) is None
    harness.turn_protect = True
    mask = harness.turn_mask(state, dag, ['red', 'lamp'], action)
    assert mask[5, 33] == 0 and mask[5, 5] == 1 and mask[30, 20] == 1
    assert harness.turn_mask(state, dag, ['sky', 'lamp'], action) is None
    seen = []

    def render(image, act, seed=42, mask=None):
        seen.append((act.box, mask))
        return torch.ones(3, 40, 40)
    harness.turn_request = 'make the cat red and add a lamp'
    candidates = harness.turn_candidates(state, dag, SimpleNamespace(render=render), ['red', 'lamp'], seed=1)
    pasted = candidates[0]['candidate_state'].image
    assert seen[0][0] == CANVAS and torch.equal(seen[0][1], mask)
    assert float(pasted[:, 5, 33].sum()) == 0 and float(pasted[:, 5, 5].sum()) == 3
