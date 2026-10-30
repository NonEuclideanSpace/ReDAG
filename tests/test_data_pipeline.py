import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'data'))
from annotate_graphs import heuristic, to_graph  # noqa: E402
from build_sets import failure_state, merged, planner_pairs, select_exact, session_turns, table  # noqa: E402
from redag.models.planner import preserve_previous_conditions  # noqa: E402

CHECKLIST = {'id': 'cb:C-MR-001', 'kind': 't2i', 'prompt': 'A chef in a kitchen.', 'checklist': [
    {'qid': 'q01', 'question': 'If the chef and the hat are present, is the chef wearing the hat?', 'tags': ['relation'], 'fixed': True},
    {'qid': 'q02', 'question': 'Is there a knife in the image?', 'tags': ['instance_pos'], 'fixed': True},
    {'qid': 'q03', 'question': 'Is the knife silver?', 'tags': ['attribute_pos'], 'fixed': True}]}
SESSION = {'id': 'mb:dev:1', 'source': 'magicbrush', 'kind': 'i2i', 'image': '/tmp/a.png',
           'turns': [{'turn': 1, 'instruction': 'make the car red', 'target': '/tmp/t1.png'},
                     {'turn': 2, 'instruction': 'actually make it blue', 'target': '/tmp/t2.png'}]}


def node(nid, kind, question, phrase='', deps=(), scope='local'):
    return {'node_id': nid, 'node_type': kind, 'question': question, 'entity_phrase': phrase, 'scope': scope,
            'depends_on': list(deps), 'statement': question.rstrip('?') + '.'}


def instruction_graph():
    value = {'nodes': [node('i1', 'entity', 'Is there a car?', 'car'), node('i2', 'entity', 'Is there a tree?', 'tree'),
                       node('i3', 'attribute', 'Is the tree green?', 'tree', ['i2']), node('i4', 'entity', 'Is there a road?', 'road'),
                       node('t1_1', 'attribute', 'Is the car red?', 'car', ['i1']),
                       node('t2_1', 'attribute', 'Is the car blue?', 'car', ['i1'])],
             'comp_edges': [['i1', 't1_1'], ['i1', 't2_1'], ['t2_1', 'i1']], 'supp_edges': [['i4', 'i1']],
             'int_edges': [['i1', 'i9']], 'superseded': {'2': ['t1_1']}}
    graph, statements, node_turn, superseded, warnings = to_graph(value, SESSION, 'instruction')
    return dict(SESSION, mode='instruction', graph=graph, statements=statements, node_turn=node_turn, superseded=superseded)


def test_checklist_answer_keeps_questions_and_adds_entities():
    value = {'nodes': [node('q01', 'relation', 'changed text?', 'chef', ['e1', 'e2']), node('q02', 'entity', 'x?', 'knife'),
                       node('q03', 'attribute', 'y?', 'knife', ['q02']), node('e1', 'entity', 'Is there a chef?', 'chef'),
                       node('e2', 'entity', 'Is there a hat?', 'hat')], 'comp_edges': [['e1', 'q01'], ['q02', 'q03']]}
    graph, _, _, _, _ = to_graph(value, CHECKLIST, 'checklist')
    nodes = {n['node_id']: n for n in graph['nodes']}
    assert nodes['q01']['question'] == CHECKLIST['checklist'][0]['question'] and nodes['q01']['scope'] == 'local'
    with pytest.raises(ValueError):
        to_graph({'nodes': value['nodes'][1:]}, CHECKLIST, 'checklist')  # q01 missing


def test_checklist_answer_with_the_models_own_nodes_is_salvaged():
    # the model adds a relation node r1 and an object node o1 (wrong id) next to the checklist items
    value = {'nodes': [node('q01', 'relation', 'x?', 'chef', ['r1']), node('r1', 'relation', 'Is the chef near the hat?', 'chef', ['o1', 'e2']),
                       node('o1', 'entity', 'Is there a chef?', 'chef'), node('e2', 'entity', 'Is there a hat?', 'hat'),
                       node('q02', 'entity', 'x?', 'knife'), node('q03', 'attribute', 'y?', 'knife', ['q02'])],
             'comp_edges': [['o1', 'q01'], ['e2', 'r1'], ['q02', 'q03']]}
    graph, _, _, _, warnings = to_graph(value, CHECKLIST, 'checklist')
    nodes = {n['node_id']: n for n in graph['nodes']}
    assert set(nodes) == {'q01', 'q02', 'q03', 'e2', 'e3'}                 # r1 dropped, o1 renamed to the next free e<k>
    assert sorted(nodes['q01']['depends_on']) == ['e2', 'e3']             # inherited r1's dependencies
    assert ['e3', 'q01'] in graph['comp_edges'] and not any('r1' in e for e in graph['comp_edges'])
    assert any('dropped' in w for w in warnings) and any('renamed' in w for w in warnings)


def test_instruction_answer_is_sanitized_and_superseded_is_kept():
    row = instruction_graph()
    assert row['superseded'] == {'2': ['t1_1']} and ['t2_1', 'i1'] not in row['graph']['comp_edges']
    assert row['node_turn'] == {'i1': 0, 'i2': 0, 'i3': 0, 'i4': 0, 't1_1': 1, 't2_1': 2}
    nodes, parents = table(row)
    chosen = select_exact(nodes, parents, 5 + 1, ['t1_1', 't2_1'], random.Random(0))
    turns = session_turns(row, chosen, None)
    final = merged(turns)
    assert len(final.nodes) == 5 and 't1_1' not in final.nodes and turns[1]['superseded_ids'] == ['t1_1']
    assert turns[0]['user_prompt'] == 'make the car red' and all(n['bbox'] is None for t in turns for n in t['graph']['nodes'])


def test_failure_states_and_planner_pairs_are_valid_graphs():
    row = instruction_graph()
    for seed in range(5):
        state = failure_state(row, random.Random(seed), (6, 16))
        assert state is None or preserve_previous_conditions(state['graph'], None)
    previous = None
    for pair in planner_pairs(row, random.Random(0)):
        previous = preserve_previous_conditions(pair['response'], pair['previous_dag'], pair['response'].get('superseded_ids', []))
    assert previous is not None and 't2_1' in previous.nodes and 't1_1' not in previous.nodes
    graph, *_ = heuristic(CHECKLIST, 'checklist')
    cb = dict(CHECKLIST, source='corebench', mode='checklist', graph=graph, statements={}, node_turn={}, superseded={})
    for pair in planner_pairs(cb, random.Random(1)):
        preserve_previous_conditions(pair['response'], pair['previous_dag'])


def test_heuristics_cover_every_mode():
    for session, mode in ((CHECKLIST, 'checklist'), (SESSION, 'instruction'),
                          ({'prompt': 'A red car on a road. A tall green tree beside it. Two birds in the sky.'}, 'decompose')):
        graph, statements, node_turn, _, _ = heuristic(session, mode)
        preserve_previous_conditions(graph, None)
        assert json.dumps(graph) and set(node_turn) == {n['node_id'] for n in graph['nodes']}


def test_magicbrush_converter_on_synthetic_parquet(tmp_path):
    """Two-turn session in MagicBrush's parquet layout: alpha-hole mask, then white mask; 4:3 source, square target."""
    import io
    import os
    import subprocess
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw
    root = Path(__file__).resolve().parents[1]

    def png(image):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return {'bytes': buffer.getvalue(), 'path': None}

    source = Image.new('RGB', (80, 60), (40, 120, 200))
    ImageDraw.Draw(source).rectangle((5, 5, 30, 25), fill=(250, 250, 0))
    target1 = source.resize((128, 128))
    ImageDraw.Draw(target1).rectangle((70, 70, 110, 110), fill=(255, 0, 0))
    mask1 = target1.convert('RGBA')
    ImageDraw.Draw(mask1).rectangle((66, 66, 114, 114), fill=(0, 0, 0, 0))  # DALL-E style transparent hole
    target2 = target1.copy()
    ImageDraw.Draw(target2).ellipse((10, 60, 40, 90), fill=(0, 255, 0))
    mask2 = Image.new('L', (128, 128), 0)
    ImageDraw.Draw(mask2).rectangle((6, 56, 44, 94), fill=255)  # white = editable
    image_type = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
    table = pa.table({'img_id': ['777', '777'], 'turn_index': pa.array([1, 2], pa.int32()),
                      'source_img': pa.array([png(source), png(target1)], image_type),
                      'mask_img': pa.array([png(mask1), png(mask2)], image_type),
                      'instruction': ['make a red square', 'add a green ball'],
                      'target_img': pa.array([png(target1), png(target2)], image_type)})
    raw = tmp_path / 'raw' / 'magicbrush' / 'data'
    raw.mkdir(parents=True)
    pq.write_table(table, raw / 'dev-00000-of-00001.parquet')
    env = dict(os.environ, REDAG_DATA=str(tmp_path))
    subprocess.run([sys.executable, str(root / 'scripts' / 'data' / 'prep_magicbrush.py'), '--splits', 'dev'], check=True, env=env,
                   capture_output=True)
    turns = [json.loads(l) for l in (tmp_path / 'processed' / 'magicbrush' / 'turns_dev.jsonl').read_text().splitlines()]
    assert [t['mask_rule'] for t in turns] == ['alpha', 'white'] and turns[0]['align'] == 'resize'
    assert turns[1]['align'] == 'identity' and all(t['outside_err'] < 0.02 for t in turns)
    box = turns[0]['edit_box']
    assert abs(box[0] - 66 / 128) < 0.02 and abs(box[2] - 115 / 128) < 0.02
    sessions = [json.loads(l) for l in (tmp_path / 'processed' / 'sessions' / 'magicbrush_dev.jsonl').read_text().splitlines()]
    assert len(sessions) == 1 and [t['turn'] for t in sessions[0]['turns']] == [1, 2]
    final = tmp_path / 'final' / 'editor_sft_dev.jsonl'
    rows = [json.loads(l) for l in final.read_text().splitlines()]
    sizes = {Image.open(final.parent / rows[0][k]).size for k in ('source', 'target', 'mask')}
    assert len(rows) == 2 and len(sizes) == 1
