"""Graph annotation keeps generate() batches full and homogeneous, retries inside later batches, splits on OOM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'data'))


def test_batches_are_full_homogeneous_and_retries_are_merged(tmp_path, monkeypatch):
    import annotate_graphs as ag
    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    checklist = [{'id': f'c{i}', 'source': 'corebench', 'kind': 't2i', 'prompt': 'x' * (10 + i),
                  'checklist': [{'question': 'q'}] * (i + 1)} for i in range(5)]
    turns = [{'id': f'm{i}', 'source': 'magicbrush', 'kind': 'i2i', 'prompt': 'y', 'turns': [{'instruction': 'z'}]} for i in range(3)]
    (sessions / 'corebench.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in checklist))
    (sessions / 'magicbrush_dev.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in turns))
    batches, calls = [], {'n': 0}

    def fake_generate(planner, conversations, images, max_new_tokens):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('CUDA out of memory. Tried to allocate 2 GiB')  # first call: split the batch
        batches.append([c['id'] for c in conversations])
        return [('bad' if c['id'] == 'c4' or (c['id'] == 'c2' and c['attempt'] == 1) else 'good') for c in conversations]

    def fake_to_graph(obj, session, mode):
        if obj != 'good':
            raise ValueError('invalid graph')
        return ({'nodes': []}, [], {}, [], [])

    monkeypatch.setattr(ag, 'load_model', lambda args: object())
    monkeypatch.setattr(ag, 'generate', fake_generate)
    monkeypatch.setattr(ag, 'messages', lambda s, mode, inv, fb=None, ans=None: (
        {'id': s['id'], 'mode': mode, 'attempt': 1 if fb is None else 2}, None))
    monkeypatch.setattr(ag, 'to_graph', fake_to_graph)
    monkeypatch.setattr(ag, 'heuristic', lambda s, mode: ({'nodes': []}, [], {}, [], []))
    import redag.models.planner as planner_module
    monkeypatch.setattr(planner_module, 'parse_json_object', lambda text: text)
    out = tmp_path / 'graphs'
    monkeypatch.setattr(sys, 'argv', ['annotate_graphs.py', '--sessions', str(sessions / 'corebench.jsonl'), str(sessions / 'magicbrush_dev.jsonl'),
                                      '--output_dir', str(out), '--batch_size', '4', '--attempts', '3'])
    ag.main()
    rows = [json.loads(line) for path in out.glob('*.jsonl') for line in path.read_text().splitlines()]
    status = {r['id']: r['annotation']['status'] for r in rows}
    assert len(rows) == 8 and set(status) == {f'c{i}' for i in range(5)} | {f'm{i}' for i in range(3)}
    assert status['c2'] == 'repaired' and status['c4'] == 'fallback'
    assert all(v == 'ok' for k, v in status.items() if k not in ('c2', 'c4'))
    for batch in batches:  # never mixes text-only checklist sessions with image (MagicBrush) sessions
        assert len({b[0] for b in batch}) == 1
    assert max(len(b) for b in batches if b[0].startswith('c')) == 2  # 4 halved after the out-of-memory error
    assert max(len(b) for b in batches if b[0].startswith('m')) <= 3
    assert batches[0][0] == 'c4'  # largest checklist first


def test_cut_off_answers_get_a_compact_json_retry_instruction():
    import annotate_graphs as ag
    assert ag.cut_off('{"nodes": [{"node_id": "q1"}, {"node_id": "q2"') and ag.cut_off('')
    assert not ag.cut_off('{"nodes": []}\n') and not ag.cut_off('```json\n{"nodes": []}\n```')
    assert 'single line' in ag.FIELD_RULES and 'single-line' in ag.CUT_OFF
    assert '"superseded": {}}' in ag.FIELD_RULES.replace('"int_edges": [...]}', '"int_edges": [...], "superseded": {}}')
