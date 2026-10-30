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


def test_generate_reports_steps_and_answer_lengths(monkeypatch):
    import torch
    import annotate_graphs as ag
    logged = []
    monkeypatch.setattr(ag, 'log', logged.append)

    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            return conversation[-1]['content']

        def __call__(self, text, padding, return_tensors, images=None):
            return {'input_ids': torch.ones((len(text), 5), dtype=torch.long), 'attention_mask': torch.ones((len(text), 5))}

        def batch_decode(self, answers, skip_special_tokens):
            return [' '.join(str(int(t)) for t in row if int(t) != 0) for row in answers]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = type('C', (), {'pad_token_id': 0})()

        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample):
            answers = torch.tensor([[7, 7, 7, 0], [8, 8, 8, 8]])  # the second answer hits the limit of 4
            return torch.cat([input_ids, answers], dim=1)

    planner = type('P', (), {'processor': Processor(), 'model': Model()})()
    texts = ag.generate(planner, [[{'role': 'user', 'content': 'a'}], [{'role': 'user', 'content': 'b'}]], [None, None], 4)
    assert texts == ['7 7 7', '8 8 8 8']
    assert 'median 4, max 4, 1 at the limit' in logged[-1] and '4 steps' in logged[-1]
    rules = ag.INSTRUCTION_RULES
    assert 'statement' not in rules and '"int_edges": [...]}' in rules


def test_grouped_decode_attention_matches_sdpa():
    import torch
    import annotate_graphs as ag
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    try:
        attention = ag.gqa_decode_attention()
        assert ALL_ATTENTION_FUNCTIONS['sdpa'] is attention
        module = type('M', (), {'num_key_value_groups': 6, 'is_causal': True})()
        torch.manual_seed(0)
        batch, kv, length, dim = 3, 4, 11, 32
        key, value = torch.randn(batch, kv, length, dim), torch.randn(batch, kv, length, dim)
        mask = torch.ones(batch, 1, 1, length, dtype=torch.bool)
        mask[0, ..., :4] = False  # left padding of different lengths
        mask[2, ..., :1] = False
        query = torch.randn(batch, 1, kv * 6, dim).transpose(1, 2)  # (b, heads, 1, d), not contiguous like the model's
        for m in (mask, None, torch.where(mask, 0.0, float('-inf'))):
            ours, _ = attention(module, query, key, value, m, scaling=dim ** -0.5)
            reference, _ = sdpa_attention_forward(module, query, key, value, m, scaling=dim ** -0.5)
            assert ours.shape == reference.shape == (batch, 1, kv * 6, dim)
            assert torch.allclose(ours, reference, atol=1e-5)
        prefill = torch.randn(batch, kv * 6, 5, dim)  # several query tokens: the original path
        causal = torch.ones(batch, 1, 5, length, dtype=torch.bool).tril(length - 5)
        assert torch.equal(attention(module, prefill, key, value, causal, scaling=0.1)[0],
                           sdpa_attention_forward(module, prefill, key, value, causal, scaling=0.1)[0])
    finally:
        type(ALL_ATTENTION_FUNCTIONS).register('sdpa', original)
