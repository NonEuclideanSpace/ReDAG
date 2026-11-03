from types import SimpleNamespace
import torch
from redag.graph.schema import CanvasState, ConditionDAG, ConditionNode, ControllerAction
from redag.verify.verifier import TwoStageVerifier
from redag.harness.environment import (
    ReDAGHarness, configure_routing, focused_global_instruction, focused_turn_instruction,
    run_record, unlocked_satisfied_descendants,
)
from redag.models.controller import clean_controller_instruction


def node(nid, ntype='entity', q='Is there x?', phrase='x', bbox=None, scope='local', depends_on=None):
    return ConditionNode(nid, ntype, q, phrase, bbox, scope=scope, depends_on=depends_on or [])


def graph_dict(*nodes, comp=None):
    return {'nodes': [
        {'node_id': n.node_id, 'node_type': n.node_type, 'question': n.question,
         'entity_phrase': n.entity_phrase, 'bbox': n.bbox, 'scope': n.scope, 'depends_on': n.depends_on}
        for n in nodes
    ], 'comp_edges': comp or []}


def test_cross_turn_and_source_invariant_prompt_isolation():
    inv = node('i1', 'entity', 'Is there a wooden door?', 'wooden door', (0.1, 0.1, 0.3, 0.5))
    t1 = node('t1_1', 'entity', 'Is there a pink car?', 'pink car', (0.4, 0.4, 0.8, 0.8))
    t2 = node('t2_1', 'entity', 'Is there a balloon?', 'balloon', scope='global')
    rec = {
        'kind': 'i2i',
        'image': torch.zeros(3, 16, 16),
        'turns': [
            {'user_prompt': 'make the car pink', 'graph': graph_dict(inv, t1)},
            {'user_prompt': 'put a balloon in the sky', 'graph': graph_dict(t2)},
        ],
    }
    seen = []

    def render(image, act, seed=42, mask=None):
        seen.append(act.instruction)
        return torch.zeros(3, 16, 16)

    def scorer(image, question):
        return 0.9 if 'balloon' in question else 0.2

    ctrl = SimpleNamespace(sample=lambda state, graph, tid, **kw: {
        'action': ControllerAction('modify_attr', (0.1, 0.1, 0.4, 0.5), f'Repair {tid}.')
    })
    harness = configure_routing(
        ReDAGHarness(TwoStageVerifier(scorer), None),
        SimpleNamespace(user_text=True, scope_routing=True, turn_priority=False),
    )
    run_record(rec, harness, ctrl, SimpleNamespace(render=render), max_steps=2)
    assert harness.node_turn_request == {}
    turn2_insts = seen[2:]
    assert all('balloon' not in inst for inst in turn2_insts)
    assert any('Repair i1.' == inst for inst in turn2_insts)
    harness.turn_request = 'direct session prompt'
    assert harness.node_request('new_node') == 'direct session prompt'


def test_focused_turn_strips_satisfied_and_converts_prohibitions():
    n1 = node('q1', 'entity', 'Is there a lantern?', 'lantern', scope='global')
    n2 = node('q2', 'entity', 'Is there a telescope?', 'telescope', scope='global')
    n3 = node('q3', 'prohibition', 'Is there no teacup?', 'teacup', scope='global')
    n4 = node('q4', 'prohibition', 'Is the obelisk free of cracks?', 'obelisk', scope='global')
    n5 = node('q5', 'relation', "Is the text 'ALC. 14.5% BY VOL.' positioned at the bottom?", 'label', scope='global')
    dag = ConditionDAG({'q1': n1, 'q2': n2, 'q3': n3, 'q4': n4, 'q5': n5})
    req = (
        "Also make sure that: A lantern is present. A telescope is present. "
        "No teacup is present. The obelisk has no cracks. "
        "The label contains the text 'ALC. 14.5% BY VOL.'. 'ALC. 14.5% BY VOL.' is at the bottom."
    )
    inst = focused_turn_instruction(req, dag, ['q1', 'q3', 'q4', 'q5'])
    assert 'A lantern is present.' in inst
    assert 'Remove any teacup from the image.' in inst
    assert 'Remove any cracks from the image.' in inst
    assert 'Remove any obelisk' not in inst
    assert "'ALC. 14.5% BY VOL.'" in inst
    assert 'telescope' not in inst
    g_inst = focused_global_instruction(req, dag, 'q4')
    assert g_inst.startswith('Remove any cracks from the image.')
    assert 'Remove any obelisk' not in g_inst
    q_turn = focused_turn_instruction('Can it be a boy not a girl?', dag, ['q1'])
    assert '?.' not in q_turn and 'Can it be a boy not a girl?' in q_turn


def test_single_stage_verify_and_two_stage_prescreen():
    n = node('a', 'entity', 'Is there a cat?', 'cat', (0.2, 0.2, 0.5, 0.5), scope='local')
    p1 = node('p1', 'entity', 'Is there p1?', 'p1', (0.2, 0.2, 0.5, 0.5), scope='local')
    p2 = node('p2', 'entity', 'Is there p2?', 'p2', (0.2, 0.2, 0.5, 0.5), scope='local')
    dag = ConditionDAG({'a': n, 'p1': p1, 'p2': p2})

    def scorer(image, q):
        if 'p' in q:
            return 0.9
        if image.shape[-1] == 20:
            return 0.15
        return 0.95 if float(image.mean()) > 0.5 else 0.2

    h1 = configure_routing(
        ReDAGHarness(TwoStageVerifier(scorer), None, feather_px=0),
        SimpleNamespace(candidate_k=2),
    )
    st1 = CanvasState(torch.zeros(3, 20, 20), {'a': 0.2, 'p1': 0.9, 'p2': 0.9}, {'p1', 'p2'})
    ctrl = SimpleNamespace(sample=lambda s, g, tid, **kw: {
        'action': ControllerAction('add_entity', (0.2, 0.2, 0.5, 0.5), 'Add cat.')
    })
    ed = SimpleNamespace(
        render=lambda img, act, seed=42, mask=None: torch.full((3, 20, 20), 0.9 if seed > 1000 else 0.0)
    )
    out1, _, steps1 = h1.run_session(st1, dag, ctrl, ed, max_steps=1, seed=10)
    assert steps1[0]['accepted'] and out1.scores['a'] >= 0.7

    h2 = configure_routing(ReDAGHarness(TwoStageVerifier(scorer), None), SimpleNamespace(single_stage_verify=True))
    s2, _ = h2.initialize(torch.zeros(3, 20, 20), ConditionDAG({'a': n}))
    assert s2.scores['a'] == 0.15 and 'a' not in s2.protected_set


def test_cascade_priority_and_early_exit_best_of_k():
    p = node('p', 'entity', 'Is there a table?', 'table', (0.1, 0.5, 0.9, 0.9), scope='local')
    c1 = node('c1', 'attribute', 'Is the table wooden?', 'table', scope='local', depends_on=['p'])
    c2 = node('c2', 'attribute', 'Is the table round?', 'table', scope='local', depends_on=['p'])
    other = node('o', 'entity', 'Is there a lamp?', 'lamp', (0.1, 0.1, 0.3, 0.3), scope='local')
    dag = ConditionDAG({'o': other, 'p': p, 'c1': c1, 'c2': c2}, comp_edges=[('p', 'c1'), ('p', 'c2')])
    scores = {'o': 0.1, 'p': 0.1, 'c1': 0.9, 'c2': 0.85}
    assert unlocked_satisfied_descendants(dag, scores, 'p') == 2
    assert unlocked_satisfied_descendants(dag, scores, 'o') == 0

    seeds_seen = []

    def render(image, act, seed=42, mask=None):
        seeds_seen.append(seed)
        val = 0.9 if seed > 1000 else 0.1
        return torch.full((3, image.shape[-2], image.shape[-1]), val)

    def scorer(image, q):
        if 'wooden' in q or 'round' in q:
            return 0.9
        if 'table' in q:
            return 0.9 if float(image.mean()) > 0.2 else 0.1
        return 0.1

    ctrl = SimpleNamespace(sample=lambda s, g, tid, **kw: {
        'action': ControllerAction('add_entity', g.nodes[tid].bbox, f'Add {tid}.')
    })
    harness = configure_routing(
        ReDAGHarness(TwoStageVerifier(scorer), None, feather_px=0),
        SimpleNamespace(cascade_priority=True, candidate_k=3, local_gate='net2', single_stage_verify=True),
    )
    state = CanvasState(torch.zeros(3, 20, 20), scores, {'c1', 'c2'})
    out_state, _, steps = harness.run_session(state, dag, ctrl, SimpleNamespace(render=render), max_steps=1, seed=10)
    assert steps[0]['target_node_id'] == 'p' and steps[0]['accepted']
    assert seeds_seen == [10, 10017]
    assert out_state.scores['p'] >= 0.7


def test_repair_gate_reserve_and_retry_limit_no_starvation():
    t = node('t', 'entity', 'Is there t?', 't', (0.0, 0.0, 1.0, 1.0), scope='local')
    d = node('d', 'entity', 'Is there d?', 'd', (0.1, 0.1, 0.4, 0.4), scope='local')
    o = node('o', 'entity', 'Is there o?', 'o', (0.5, 0.5, 0.8, 0.8), scope='local')

    def render_res(img, act, seed=42, mask=None):
        if 'Edit d' in act.instruction:
            return torch.full((3, 16, 16), 0.9)
        return torch.full((3, 16, 16), 0.5 if seed == 42 else 0.2)

    def score_res(img, q):
        m = float(img.mean())
        if 't?' in q:
            return 0.95 if m > 0.4 else 0.1
        return 0.95 if m > 0.48 and m > 0.8 else (0.95 if m > 0.52 else 0.60)

    ctrl = SimpleNamespace(sample=lambda s, g, tid, **kw: {
        'action': ControllerAction('add_entity', g.nodes[tid].bbox, f'Edit {tid}.')
    })
    h = configure_routing(
        ReDAGHarness(TwoStageVerifier(score_res), None, feather_px=0),
        SimpleNamespace(
            candidate_k=2, local_gate='net2', repair_gate=True, monotonic_turn=True, single_stage_verify=True
        ),
    )
    st_init = CanvasState(torch.zeros(3, 16, 16), {'t': 0.1, 'd': 0.9}, {'d'})
    h.render_limit = h.generation_counter + 2
    ed_res = SimpleNamespace(render=render_res)
    final_s, _, steps = h.run_session(st_init, ConditionDAG({'t': t, 'd': d}), ctrl, ed_res, max_steps=2)
    assert len(steps) == 2 and steps[0]['repair_drop'] == 'd' and steps[1]['target_node_id'] == 'd'
    assert final_s.scores['t'] >= 0.7 and final_s.scores['d'] >= 0.7

    chosen_targets = []

    def render_fail(img, act, seed=42, mask=None):
        return torch.full((3, 16, 16), 0.5)

    def score_fail(img, q):
        m = float(img.mean())
        return (0.95 if m > 0.4 else 0.1) if 't?' in q else 0.5

    h2 = configure_routing(
        ReDAGHarness(TwoStageVerifier(score_fail), None, feather_px=0),
        SimpleNamespace(local_gate='net2', repair_gate=True, single_stage_verify=True),
    )
    h2.retry_limit = 1

    def sample_track(s, g, tid, **kw):
        chosen_targets.append(tid)
        return {'action': ControllerAction('add_entity', g.nodes[tid].bbox, f'Edit {tid}.')}

    st2 = CanvasState(torch.zeros(3, 16, 16), {'t': 0.1, 'd': 0.9, 'o': 0.1}, {'d'})
    ctrl2, ed2 = SimpleNamespace(sample=sample_track), SimpleNamespace(render=render_fail)
    h2.run_session(st2, ConditionDAG({'t': t, 'd': d, 'o': o}), ctrl2, ed2, max_steps=4)
    assert chosen_targets[:3] == ['t', 'd', 'o']


def test_turn_box_word_boundary_no_ground_in_background():
    g_node = node('i8', 'entity', 'Is there ground?', 'ground', (0.0, 0.7, 1.0, 1.0), scope='local')
    b_node = node('t1', 'attribute', 'Is the sign red?', 'sign', (0.2, 0.05, 0.5, 0.3), scope='local')
    dag = ConditionDAG({'i8': g_node, 't1': b_node})
    h = configure_routing(
        ReDAGHarness(TwoStageVerifier(lambda img, q: 0.5), None),
        SimpleNamespace(turn_localize=True, focused_turn=True),
    )
    h.turn_request = 'put a hot air balloon in the background'
    box = h.turn_box(dag, ['t1'], torch.zeros(3, 20, 20), {'i8': 0.9, 't1': 0.1})
    assert box[3] < 0.5


def test_clean_controller_instruction():
    tok = SimpleNamespace(
        encode=lambda s, add_special_tokens=False: s.split(),
        decode=lambda ids, skip_special_tokens=True: ' '.join(ids),
    )
    out, count, changed = clean_controller_instruction('Make the mug red', tok)
    assert 'unchanged' in out and count < 30 and changed


def test_target_first_fast_reject_and_no_double_verify():
    nodes = {'t': node('t', 'count', 'Are there 3 cups?', 'cup', scope='global')}
    for idx in range(1, 9):
        nodes[f'p{idx}'] = node(f'p{idx}', 'entity', f'Is p{idx} ok?', f'p{idx}', (0.1, 0.1, 0.3, 0.3))
    dag = ConditionDAG(nodes)
    init_scores = {'t': 0.1, **{f'p{idx}': 0.95 for idx in range(1, 9)}}
    state = CanvasState(torch.zeros(3, 16, 16), init_scores, {f'p{idx}' for idx in range(1, 9)})

    def render_fn(img, act, seed=42, mask=None):
        val = 0.8 if seed == 42 + 2 * 10007 else 0.2
        return torch.full((3, 16, 16), val)

    def score_fn(img, q):
        m = float(img.mean())
        if 'cups' in q:
            return 0.92 if m > 0.5 else 0.05
        return 0.95

    ver = TwoStageVerifier(score_fn)
    h = configure_routing(
        ReDAGHarness(ver, None, feather_px=0),
        SimpleNamespace(
            scope_routing=True, global_gate='net2', local_gate='net2',
            single_stage_verify=True, candidate_k=3, global_retries=1,
        ),
    )
    ctrl = SimpleNamespace(sample=lambda s, g, tid, **kw: {
        'action': ControllerAction('modify_attr', (0.0, 0.0, 1.0, 1.0), 'Edit.')
    })
    out_s, _, steps = h.run_session(state, dag, ctrl, SimpleNamespace(render=render_fn), max_steps=1, seed=42)
    assert steps[0]['accepted'] and out_s.scores['t'] >= 0.7
    assert ver.call_counter == 1 + 1 + 9

    ver2 = TwoStageVerifier(lambda img, q: 0.05 if 'cups' in q else 0.95)
    h2 = configure_routing(
        ReDAGHarness(ver2, None, feather_px=0),
        SimpleNamespace(
            scope_routing=True, global_gate='net2', local_gate='net2',
            single_stage_verify=True, candidate_k=3, global_retries=1,
        ),
    )
    h2.run_session(state, dag, ctrl, SimpleNamespace(render=lambda img, act, seed=42, mask=None: img), max_steps=1, seed=42)
    assert ver2.call_counter == 3
    assert h2.global_failures.get((0, 't')) == 1


def test_i2i_source_invariant_isolation_and_prohibited_repair():
    i1 = node('i1', 'entity', 'Is table there?', 'table', (0.1, 0.5, 0.9, 0.9))
    i12 = node('i12', 'count', 'Are there two hot dogs?', 'hot dog', scope='global')
    i14 = node('i14', 'attribute', 'Is hot dog topped with relish?', 'hot dog', (0.2, 0.2, 0.5, 0.5))
    t1_1 = node('t1_1', 'entity', 'Is chocolate cake there?', 'cake', (0.2, 0.2, 0.6, 0.6))
    t1_2 = node('t1_2', 'prohibition', 'Is image free of hot dogs?', 'hot dog', scope='global')
    dag = ConditionDAG({'i1': i1, 'i12': i12, 'i14': i14, 't1_1': t1_1, 't1_2': t1_2})

    def score_i2i(img, q):
        m = float(img.mean())
        if 'two hot dogs' in q:
            return 0.1
        if 'chocolate cake' in q or 'free of hot dogs' in q:
            return 0.95 if m > 0.4 else 0.05
        if 'relish' in q:
            return 0.4 if m > 0.4 else 0.9
        return 0.95

    h = configure_routing(
        ReDAGHarness(TwoStageVerifier(score_i2i), None, feather_px=0),
        SimpleNamespace(
            turn_pass=True, focused_turn=True, turn_priority=True,
            turn_gate='net2', global_gate='net2', local_gate='net2',
            repair_gate=True, single_stage_verify=True,
        ),
    )
    rec = {
        'kind': 'i2i',
        'image': torch.zeros(3, 16, 16),
        'turns': [{'user_prompt': 'Put chocolate cake instead of hot dog.', 'graph': {
            'nodes': [
                {'node_id': k, 'node_type': v.node_type, 'question': v.question,
                 'entity_phrase': v.entity_phrase, 'bbox': v.bbox, 'scope': v.scope,
                 'threshold': v.threshold, 'weight': v.weight, 'depends_on': v.depends_on}
                for k, v in dag.nodes.items()
            ],
            'comp_edges': [], 'supp_edges': [], 'int_edges': [],
        }}],
    }
    ctrl = SimpleNamespace(sample=lambda s, g, tid, **kw: {
        'action': ControllerAction('add_entity', (0.0, 0.0, 1.0, 1.0), f'Edit {tid}.')
    })
    ed = SimpleNamespace(render=lambda img, act, seed=42, mask=None: torch.full((3, 16, 16), 0.8))
    final_s, _, steps = run_record(rec, h, ctrl, ed, max_steps=4, seed=7)
    assert final_s.scores['t1_1'] >= 0.7 and final_s.scores['t1_2'] >= 0.7
    assert all(st['target_node_id'] != 'i12' for st in steps)
    assert len(steps) == 1


def test_source_invariant_unblocking_and_misnamed_edits():
    i10 = node('i10', 'entity', 'Is table there?', 'table', (0.1, 0.5, 0.9, 0.9))
    i5 = node('i5', 'entity', 'Is lamp there?', 'lamp', (0.7, 0.1, 0.9, 0.3))
    t1_1 = node('t1_1', 'entity', 'Is croissant there?', 'croissant', (0.2, 0.5, 0.6, 0.8))
    i13 = node('i13', 'attribute', 'Does croissant have cheese?', 'croissant', (0.2, 0.5, 0.6, 0.8), depends_on=['t1_1'])
    t2_2 = node('t2_2', 'relation', 'Is cake on table?', 'cake', (0.2, 0.5, 0.8, 0.9), depends_on=['i10'])
    dag = ConditionDAG(
        {'i10': i10, 'i5': i5, 't1_1': t1_1, 'i13': i13, 't2_2': t2_2},
        comp_edges=[('t1_1', 'i13'), ('i10', 't2_2')],
    )
    h = configure_routing(
        ReDAGHarness(TwoStageVerifier(lambda img, q: 0.9), None, feather_px=0),
        SimpleNamespace(focused_turn=True, turn_priority=True, local_gate='net2', single_stage_verify=True),
    )
    inv = h.active_source_invariants(dag)
    assert inv == {'i10', 'i5'} and 'i13' not in inv

    visited = []
    ctrl = SimpleNamespace(sample=lambda s, g, tid, **kw: (
        visited.append(tid) or {'action': ControllerAction('add_entity', (0.0, 0.0, 1.0, 1.0), f'Edit {tid}.')}
    ))
    ed = SimpleNamespace(render=lambda img, act, seed=42, mask=None: torch.ones(3, 16, 16))
    st = CanvasState(torch.zeros(3, 16, 16), {'i10': 0.3, 'i5': 0.3, 't1_1': 0.9, 'i13': 0.2, 't2_2': 0.2}, {'t1_1'})
    out_s, _, _ = h.run_session(st, dag, ctrl, ed, max_steps=3, seed=1)
    assert 'i10' in visited or out_s.scores['t2_2'] >= 0.7
    assert 'i5' not in visited

    rec_mt = {
        'kind': 'i2i',
        'image': torch.zeros(3, 16, 16),
        'turns': [
            {'user_prompt': 'Make pizza a croissant.', 'graph': {
                'nodes': [
                    {'node_id': 'i5', 'node_type': 'entity', 'question': 'Lamp?', 'entity_phrase': 'lamp',
                     'bbox': (0.7, 0.1, 0.9, 0.3), 'scope': 'local', 'threshold': 0.7, 'weight': 1.0, 'depends_on': []},
                    {'node_id': 't1_1', 'node_type': 'entity', 'question': 'Croissant?', 'entity_phrase': 'croissant',
                     'bbox': (0.2, 0.5, 0.6, 0.8), 'scope': 'local', 'threshold': 0.7, 'weight': 1.0, 'depends_on': []},
                ],
                'comp_edges': [], 'supp_edges': [], 'int_edges': [],
            }},
            {'user_prompt': 'Cut kiwi in half.', 'graph': {
                'nodes': [
                    {'node_id': 'i15', 'node_type': 'attribute', 'question': 'Kiwi cut?', 'entity_phrase': 'kiwi',
                     'bbox': (0.3, 0.3, 0.5, 0.5), 'scope': 'local', 'threshold': 0.7, 'weight': 1.0, 'depends_on': []},
                ],
                'comp_edges': [], 'supp_edges': [], 'int_edges': [],
            }},
        ],
    }
    seen_t2 = []
    h2 = configure_routing(
        ReDAGHarness(TwoStageVerifier(lambda img, q: 0.2 if 'Kiwi' in q else 0.9), None, feather_px=0),
        SimpleNamespace(
            turn_pass=False, focused_turn=True, turn_priority=True,
            user_text=True, local_gate='net2', single_stage_verify=True,
        ),
    )
    ctrl2 = SimpleNamespace(sample=lambda s, g, tid, **kw: (
        seen_t2.append((tid, h2.node_request(tid)))
        or {'action': ControllerAction('modify_attr', (0.3, 0.3, 0.5, 0.5), 'Cut kiwi.')}
    ))
    run_record(rec_mt, h2, ctrl2, ed, max_steps=1, seed=2)
    assert ('i15', 'Cut kiwi in half.') in seen_t2
