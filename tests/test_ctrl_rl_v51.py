import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from build_rl_v51_splits import enrich_metadata
from redag.graph.schema import CanvasState, ConditionDAG, ConditionNode, ControllerAction
from redag.harness.environment import ReDAGHarness, configure_routing
from redag.verify.verifier import TwoStageVerifier
from run_ctrl_rl_v51 import (
    PreparedParent, assert_v51_alignment, compute_v51_step_reward, configure_parent_context,
    prepare_parents, rollout_parent_group, select_v51_local_target, train_ctrl_rl_v51,
)


def make_node(nid, ntype='entity', q='Is x?', phrase='x', bbox=(0.1, 0.1, 0.4, 0.4), scope='local', deps=None):
    return ConditionNode(nid, ntype, q, phrase, bbox, scope=scope, depends_on=deps or [])


class ToyTrainableController(nn.Module):
    def __init__(self, vocab_size=8, init_good_logit=0.0):
        super().__init__()
        self.logits_param = nn.Parameter(torch.zeros(vocab_size))
        with torch.no_grad():
            self.logits_param[3] = init_good_logit
        self.clean_instruction = True

    def lora_parameters(self):
        return [self.logits_param]

    def snapshot(self):
        return {'logits_param': self.logits_param.detach().cpu().clone()}

    def load_snapshot(self, snap):
        with torch.no_grad():
            self.logits_param.copy_(snap['logits_param'].to(self.logits_param.device))

    def save_adapter(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.snapshot(), p / 'toy_adapter.pt')

    def sample(self, state, dag, target_node_id, seed=42, temperature=0.8, detected_count=1):
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        probs = torch.softmax(self.logits_param.detach() / temperature, dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=gen).item())
        if tok == 3:
            act = ControllerAction('modify_attr', (0.15, 0.15, 0.45, 0.45), 'Make the car bright red cleanly.')
            repaired = False
        else:
            act = ControllerAction('modify_attr', (0.0, 0.0, 1.0, 1.0), 'Bad whole canvas edit.')
            repaired = True
        token_ids = torch.tensor([[1, 2, tok]], dtype=torch.long)
        action_mask = torch.tensor([[0.0, 1.0]])
        return {
            'action': act,
            'inputs': {'input_ids': token_ids},
            'token_ids': token_ids,
            'action_mask': action_mask,
            'temperature': temperature,
            'instruction_repaired': repaired,
        }

    def forward_sample(self, sample, action_only=False):
        v = self.logits_param.shape[0]
        seq = 2 if action_only else 3
        out = self.logits_param.view(1, 1, v).expand(1, seq, v)
        return out / sample.get('temperature', 1.0)


class FrozenToyEditor:
    def __init__(self):
        self.w = nn.Parameter(torch.ones(2), requires_grad=True)
        self.context_pad = 0.25
        self.num_steps = 28

    def lora_parameters(self):
        return [self.w]

    def eval(self):
        return self

    def render(self, image, action, seed=42, mask=None):
        h = round((action.box[3] - action.box[1]) * image.shape[-2])
        w = round((action.box[2] - action.box[0]) * image.shape[-1])
        val = 0.9 if 'cleanly' in action.instruction else 0.85
        return torch.full((3, max(1, h), max(1, w)), val)


def build_v51_harness(scorer):
    ns = SimpleNamespace(
        scope_routing=True, global_types='background,global,relation_wide,count,prohibition',
        global_gate='net2', local_gate='net2', global_retries=1, user_text=True,
        turn_pass=True, turn_gate='net2', turn_retries=0, turn_priority=True,
        turn_localize=True, turn_protect=True, clean_instruction=True,
        single_stage_verify=True, focused_turn=True, smart_turn_pass=True,
        cascade_priority=True, repair_gate=True, monotonic_turn=True, candidate_k=3,
        strategy_policy=None,
    )
    h = configure_routing(ReDAGHarness(TwoStageVerifier(scorer), None), ns)
    h.protect_masks = False
    h.closure_instructions = False
    h.confirm_drops = False
    h.soft_gating = False
    h.preserve_instructions = False
    h.dag_box = False
    h.edge_policy = 'full'
    h.feather_px = 0
    return h


def test_select_v51_local_target_invariants_and_global_fallback():
    inv = make_node('i0', 'entity', 'Is tree present?', 'tree', (0.6, 0.1, 0.9, 0.5))
    e1 = make_node('e1', 'entity', 'Is car present?', 'car', (0.1, 0.1, 0.4, 0.4))
    a1 = make_node('a1', 'attribute', 'Is car red?', 'car', (0.1, 0.1, 0.4, 0.4), deps=['e1'])
    r_wide = make_node('r1', 'relation', 'Is car near tree?', '', None, scope='unknown', deps=['e1', 'i0'])
    bg = make_node('b1', 'background', 'Is sky sunset?', 'sky', None, scope='global')
    dag = ConditionDAG(
        nodes={'i0': inv, 'e1': e1, 'a1': a1, 'r1': r_wide, 'b1': bg},
        comp_edges=[('e1', 'a1'), ('e1', 'r1'), ('i0', 'r1')],
        supp_edges=[],
        int_edges=[],
    )
    h = build_v51_harness(lambda img, q: 0.5)

    scores = {'i0': 0.2, 'e1': 0.2, 'a1': 0.8, 'r1': 0.2, 'b1': 0.2}
    tgt, fb = select_v51_local_target(
        h, dag, scores, turn_nodes={'e1', 'a1', 'b1'}, source_invariants={'i0'}, draft_first_turn=False,
    )
    assert tgt == 'e1' and fb is False

    scores_only_inv = {'i0': 0.2, 'e1': 0.9, 'a1': 0.9, 'r1': 0.9, 'b1': 0.2}
    tgt2, fb2 = select_v51_local_target(
        h, dag, scores_only_inv, turn_nodes={'e1', 'a1', 'b1'}, source_invariants={'i0'}, draft_first_turn=False,
    )
    assert tgt2 is None and fb2 is False

    scores_rel_wide = {'i0': 0.9, 'e1': 0.9, 'a1': 0.9, 'r1': 0.2, 'b1': 0.9}
    tgt3, fb3 = select_v51_local_target(
        h, dag, scores_rel_wide, turn_nodes={'r1'}, source_invariants={'i0'}, draft_first_turn=False,
    )
    assert tgt3 == 'r1' and fb3 is True


def test_v51_reward_ordering():
    e1 = make_node('e1', 'entity', 'Is car present?', 'car', (0.1, 0.1, 0.4, 0.4))
    a1 = make_node('a1', 'attribute', 'Is car shiny?', 'car', (0.1, 0.1, 0.4, 0.4), deps=['e1'])
    p1 = make_node('p1', 'entity', 'Is dog present?', 'dog', (0.6, 0.6, 0.9, 0.9))
    dag = ConditionDAG(nodes={'e1': e1, 'a1': a1, 'p1': p1}, comp_edges=[('e1', 'a1')], supp_edges=[], int_edges=[])
    h = build_v51_harness(lambda img, q: 0.5)
    parent = CanvasState(torch.zeros(3, 16, 16), {'e1': 0.2, 'a1': 0.85, 'p1': 0.9}, {'a1', 'p1'}, 0)

    def mk_cand(raw_scores, accepted, box=(0.1, 0.1, 0.4, 0.4), repaired=False):
        deliv_scores = raw_scores if accepted else dict(parent.scores)
        deliv = CanvasState(torch.zeros(3, 16, 16), deliv_scores, {k for k, v in deliv_scores.items() if v >= 0.7}, 1)
        return {
            'raw_scores': raw_scores,
            'delivered_state': deliv,
            'accepted': accepted,
            'affected': {'e1', 'a1', 'p1'},
            'action': ControllerAction('add_entity', box, 'Add car.'),
            'sample': {'controller_box': box, 'instruction_repaired': repaired},
        }

    r_cascade, g_cascade = compute_v51_step_reward(h, parent, dag, mk_cand({'e1': 0.85, 'a1': 0.85, 'p1': 0.9}, True))
    r_partial, g_partial = compute_v51_step_reward(h, parent, dag, mk_cand({'e1': 0.50, 'a1': 0.85, 'p1': 0.9}, False))
    r_noop, _ = compute_v51_step_reward(h, parent, dag, mk_cand({'e1': 0.20, 'a1': 0.85, 'p1': 0.9}, False))
    r_damage, _ = compute_v51_step_reward(h, parent, dag, mk_cand({'e1': 0.80, 'a1': 0.85, 'p1': 0.1}, False))
    r_pen, _ = compute_v51_step_reward(
        h, parent, dag, mk_cand({'e1': 0.85, 'a1': 0.85, 'p1': 0.9}, True, box=(0.0, 0.0, 1.0, 1.0), repaired=True),
    )
    assert g_cascade == 2 and g_partial == 0
    assert r_cascade > r_pen > r_partial > r_noop > r_damage


def test_rollout_parent_group_penalizes_wide_box_collateral_damage_under_single_stage():
    def scorer(image, question):
        if 'car' in question:
            return 0.22
        dog_patch = float(image[:, 16:30, 16:30].mean().item())
        return 0.9 if dog_patch < 0.2 else 0.05

    h = build_v51_harness(scorer)
    h.dag_box = False
    e1 = make_node('e1', 'entity', 'Is car present?', 'car', (0.05, 0.05, 0.35, 0.35))
    p1 = make_node('p1', 'entity', 'Is dog present?', 'dog', (0.5, 0.5, 0.95, 0.95))
    p2 = make_node('p2', 'entity', 'Is lamp present?', 'lamp', (0.5, 0.5, 0.95, 0.95))
    dag = ConditionDAG(nodes={'e1': e1, 'p1': p1, 'p2': p2}, comp_edges=[], supp_edges=[], int_edges=[])

    class TwoChoiceController:
        clean_instruction = True

        def sample(self, state, graph, target_node_id, seed=42, temperature=0.8, detected_count=1):
            idx = seed % 2
            box = (0.05, 0.05, 0.35, 0.35) if idx == 0 else (0.05, 0.05, 0.95, 0.95)
            toks = torch.tensor([[1, 2, idx]], dtype=torch.long)
            return {
                'action': ControllerAction('add_entity', box, 'Attempt car.'),
                'inputs': {'input_ids': toks},
                'token_ids': toks,
                'action_mask': torch.tensor([[0.0, 1.0]]),
                'temperature': temperature,
                'instruction_repaired': False,
            }

    class BrightEditor:
        def render(self, image, action, seed=42, mask=None):
            hh = round((action.box[3] - action.box[1]) * image.shape[-2])
            ww = round((action.box[2] - action.box[0]) * image.shape[-1])
            return torch.full((3, max(1, hh), max(1, ww)), 0.9)

    parent = PreparedParent(
        record_id='r0', session_id='s0', origin='magicbrush', turn=1, image_path='',
        dag=dag, scores={'e1': 0.2, 'p1': 0.9, 'p2': 0.9}, grounding=None,
        user_prompt='add car', node_turn_request={'e1': 'add car', 'p1': '', 'p2': ''},
        turn_nodes={'e1'}, source_invariants={'p1', 'p2'}, draft_first_turn=False,
        target_node_id='e1', fallback_global=False,
    )
    rolled = rollout_parent_group(
        h, TwoChoiceController(), BrightEditor(), parent, group_size=2, noise_seed=10,
        inline_image=torch.zeros(3, 32, 32),
    )
    assert h.counterfactual_credit is False
    assert rolled['rewards'][0] > rolled['rewards'][1] + 0.5


def test_enrich_metadata_and_prepare_parents_cache_safety(tmp_path):
    raw_graph = {
        'nodes': [
            {'node_id': 'i0', 'node_type': 'entity', 'question': 'Is table there?', 'entity_phrase': 'table',
             'bbox': None, 'weight': 1.0, 'threshold': 0.7, 'expected_count': None, 'scope': 'local', 'depends_on': []},
            {'node_id': 't1_1', 'node_type': 'entity', 'question': 'Is cup there?', 'entity_phrase': 'cup',
             'bbox': None, 'weight': 1.0, 'threshold': 0.7, 'expected_count': None, 'scope': 'local', 'depends_on': []},
            {'node_id': 'i1', 'node_type': 'relation', 'question': 'Is cup on table?', 'entity_phrase': '',
             'bbox': None, 'weight': 1.0, 'threshold': 0.7, 'expected_count': None, 'scope': 'local', 'depends_on': ['i0', 't1_1']},
            {'node_id': 't2_1', 'node_type': 'attribute', 'question': 'Is cup blue?', 'entity_phrase': 'cup',
             'bbox': None, 'weight': 1.0, 'threshold': 0.7, 'expected_count': None, 'scope': 'local', 'depends_on': ['t1_1']},
        ],
        'comp_edges': [['i0', 'i1'], ['t1_1', 'i1'], ['t1_1', 't2_1']],
        'supp_edges': [],
        'int_edges': [],
    }
    session_row = {
        'id': 'mb:train:1', 'source': 'magicbrush', 'kind': 'i2i', 'mode': 'instruction',
        'turns': [{'instruction': 'put a cup on the table'}, {'instruction': 'paint the cup blue'}],
        'node_turn': {'i0': 0, 'i1': 0, 't1_1': 1, 't2_1': 2},
        'graph': raw_graph,
    }
    rec = {'id': 'state:mb:train:1:t2', 'session_id': 'mb:train:1', 'origin': 'magicbrush', 'turn': 2,
           'image': 'x.png', 'graph': raw_graph, 'scores': {'i0': 0.1, 'i1': 0.1, 't1_1': 0.1, 't2_1': 0.1}}
    enriched = enrich_metadata(rec, session_row)
    assert 'scores' not in enriched
    assert enriched['source_invariants'] == ['i0']
    assert 'i1' not in enriched['source_invariants']
    assert enriched['node_turn_request']['i0'] == ''
    assert enriched['node_turn_request']['t1_1'] == 'put a cup on the table'
    assert enriched['node_turn_request']['t2_1'] == 'paint the cup blue'
    assert enriched['draft_first_turn'] is False

    calls = {'n': 0}

    def scorer(image, question):
        calls['n'] += 1
        return 0.2 if 'blue' in question else 0.9

    h = build_v51_harness(scorer)
    cache_file = tmp_path / 'cache.jsonl'
    stale_row = dict(enriched, id='r_stale', image=torch.zeros(3, 16, 16), scores={'i0': 0.2, 'i1': 0.2, 't1_1': 0.2, 't2_1': 0.2})
    parents = prepare_parents([stale_row], h, cache_path=cache_file)
    assert calls['n'] > 0
    assert len(parents) == 1 and parents[0].target_node_id == 't2_1'
    configure_parent_context(h, parents[0])
    assert h.node_request('t1_1') == 'put a cup on the table'
    assert h.node_request('i0') == ''
    saved_lines = [json.loads(l) for l in cache_file.read_text().splitlines() if l.strip()]
    assert len(saved_lines) == 1 and saved_lines[0]['verifier_mode'] == 'single'


def test_end_to_end_train_ctrl_rl_v51_and_hard_gates(tmp_path):
    def scorer(image, question):
        if 'car' in question:
            car_patch = float(image[:, 4:15, 4:15].mean().item())
            return 0.9 if car_patch > 0.5 else 0.1
        bg_patch = float(image[:, 19:30, 19:30].mean().item())
        return 0.9 if bg_patch < 0.3 else 0.05

    h = build_v51_harness(scorer)
    e1 = make_node('e1', 'entity', 'Is car red?', 'car', (0.15, 0.15, 0.45, 0.45))
    p1 = make_node('p1', 'entity', 'Is background calm?', 'room', (0.6, 0.6, 0.95, 0.95))
    dag = ConditionDAG(nodes={'e1': e1, 'p1': p1}, comp_edges=[], supp_edges=[], int_edges=[])

    def make_parent(prefix, idx):
        rid = f'{prefix}:{idx}'
        prompt = 'Create an image where: The car is bright red.'
        return PreparedParent(
            record_id=rid, session_id=f'sess:{prefix}:{idx}', origin='lgbench', turn=1,
            image_path='', dag=dag, scores={'e1': 0.1, 'p1': 0.9}, grounding=None,
            user_prompt=prompt, node_turn_request={'e1': prompt, 'p1': prompt},
            turn_nodes={'e1', 'p1'}, source_invariants=set(), draft_first_turn=True,
            target_node_id='e1', fallback_global=False,
        )

    train_parents = [make_parent('tr', i) for i in range(6)]
    dev_parents = [make_parent('dv', i) for i in range(6)]
    inline_images = {p.record_id: torch.zeros(3, 32, 32) for p in train_parents + dev_parents}

    ctrl = ToyTrainableController(init_good_logit=-0.3)
    ed = FrozenToyEditor()
    ed_before = ed.w.detach().clone()
    assert_v51_alignment(h, ctrl, ed)

    h.dag_box = True
    with pytest.raises(ValueError, match='no_dag_box'):
        assert_v51_alignment(h, ctrl, ed)
    h.dag_box = False

    h.protect_masks = True
    with pytest.raises(ValueError, match='no_protect_masks'):
        assert_v51_alignment(h, ctrl, ed)
    h.protect_masks = False

    ed.num_steps = 50
    with pytest.raises(ValueError, match='num_denoise_steps_28'):
        assert_v51_alignment(h, ctrl, ed)
    ed.num_steps = 28

    with pytest.raises(ValueError, match='max_steps'):
        train_ctrl_rl_v51(h, ctrl, ed, train_parents, dev_parents, tmp_path / 'zero_steps', max_steps=0, min_dev_states=4)

    with pytest.raises(ValueError, match='Hard dev gate'):
        train_ctrl_rl_v51(
            h, ctrl, ed, train_parents, dev_parents, tmp_path / 'too_small',
            max_steps=2, min_dev_states=64, inline_images=inline_images,
        )

    overlap_dev = [make_parent('tr', 0)] + dev_parents[1:]
    with pytest.raises(ValueError, match='strictly disjoint'):
        train_ctrl_rl_v51(
            h, ctrl, ed, train_parents, overlap_dev, tmp_path / 'overlap',
            max_steps=2, min_dev_states=4, inline_images=inline_images,
        )

    summary, history = train_ctrl_rl_v51(
        h, ctrl, ed, train_parents, dev_parents, tmp_path / 'ok_run',
        max_steps=6, batch_size=3, group_size=4, eval_every=3, final_k=2,
        lr=0.4, ppo_epochs=2, target_kl=2.0, min_dev_states=6, seed=7, inline_images=inline_images,
    )
    assert torch.equal(ed.w, ed_before) and not ed.w.requires_grad
    assert len(history) == 6
    assert float(ctrl.logits_param[3].item()) > 0.2
    assert summary['gate_passed'] is True
    assert (tmp_path / 'ok_run' / 'ctrl_best' / 'toy_adapter.pt').exists()
