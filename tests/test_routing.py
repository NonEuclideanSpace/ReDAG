from types import SimpleNamespace
import torch
from redag.graph.dag import CANVAS
from redag.graph.schema import CanvasState, ConditionDAG, ConditionNode, ControllerAction
from redag.harness.environment import (ReDAGHarness, configure_routing, consistent_count, joint_acceptance, request_snippet, route_class,
                                       run_record, with_user_request)
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
