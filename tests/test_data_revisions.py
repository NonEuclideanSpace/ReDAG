import importlib.util
from pathlib import Path
import pytest
from redag.harness.environment import validate_dataset

spec = importlib.util.spec_from_file_location('revise_eval_main', Path(__file__).resolve().parents[1] / 'scripts' / 'data' / 'revise_eval_main.py')
revise_eval_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revise_eval_main)


def graph(count, prefix='n'):
    return {'nodes': [{'node_id': f'{prefix}{i}', 'node_type': 'entity', 'question': f'Is there object {i}?', 'entity_phrase': f'object {i}'}
                      for i in range(count)], 'comp_edges': [], 'supp_edges': [], 'int_edges': []}


def test_revision_retires_conflicting_conditions_and_their_edges():
    record = {'id': 's', 'k': 10, 'turns': [{'user_prompt': 'replace object 1', 'graph': graph(10)}]}
    record['turns'][0]['graph']['int_edges'] = [['n1', 'n2'], ['n3', 'n4']]
    revised = revise_eval_main.revise(record, {'s': dict(turn=0, retire=['n1', 'n2'], reason='replaced')})
    nodes = [n['node_id'] for n in revised['turns'][0]['graph']['nodes']]
    assert 'n1' not in nodes and 'n2' not in nodes and len(nodes) == 8
    assert revised['turns'][0]['graph']['int_edges'] == [['n3', 'n4']] and revised['revisions'][0]['retired'] == ['n1', 'n2']
    assert len(record['turns'][0]['graph']['nodes']) == 10
    record['turns'][0]['graph']['nodes'][3]['depends_on'] = ['n1']
    with pytest.raises(ValueError):
        revise_eval_main.revise(record, {'s': dict(turn=0, retire=['n1'], reason='replaced')})
    with pytest.raises(ValueError):
        revise_eval_main.revise(record, {'s': dict(turn=0, retire=['missing'], reason='replaced')})


def test_revised_sessions_keep_their_declared_tier():
    records = [{'id': f'{k}-{i}', 'k': k, 'turns': [{'user_prompt': 'x', 'graph': graph(k)}]} for k in (10, 20, 30) for i in range(100)]
    records[150] = revise_eval_main.revise(records[150], {records[150]['id']: dict(turn=0, retire=['n0', 'n1'], reason='replaced')})
    validate_dataset(records, 1, False, require_graphs=True)
    del records[150]['revisions']
    with pytest.raises(ValueError):
        validate_dataset(records, 1, False, require_graphs=True)
