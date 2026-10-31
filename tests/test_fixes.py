import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from redag.graph.schema import CanvasState, ConditionDAG, ConditionNode, ControllerAction
from redag.harness.environment import ReDAGHarness, validate_dataset
from redag.models.editor import FluxEditor
from redag.models.planner import dag_to_dict, lora_target_modules
from redag.verify.verifier import TwoStageVerifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class PixelVAE(torch.nn.Module):
    """Lossless stand-in for the FLUX.2 VAE: 8x pixel-unshuffle (fails on sizes that are not multiples of 8)."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_out_channels=[1, 1, 1, 1], batch_norm_eps=0.0)
        self.bn = SimpleNamespace(running_mean=torch.zeros(3 * 64 * 4), running_var=torch.ones(3 * 64 * 4))

    @property
    def dtype(self):
        return torch.float32

    def encode(self, x):
        latent = F.pixel_unshuffle(x, 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, z, return_dict=False):
        return (F.pixel_shuffle(z, 8),)


def pixel_editor():
    editor = FluxEditor.__new__(FluxEditor)  # bypass the pipeline checks; only the latent layout is under test
    editor.vae, editor.transformer = PixelVAE(), torch.nn.Linear(1, 1)
    return editor


@pytest.mark.parametrize('size', [(37, 53), (48, 64), (16, 31), (9, 12)])
def test_vae_roundtrip_crops_without_inventing_border_pixels(size):
    editor = pixel_editor()
    image = torch.rand(3, *size)
    latent = editor.encode_image(image)
    assert latent.shape[-2:] == tuple(-(-max(s, 16) // 16) for s in size) or min(size) < 16
    decoded = editor.decode_latents(latent, output_size=size)
    assert decoded.shape == (1, 3, *size)
    if min(size) >= 16:  # no rescaling: exact reconstruction, i.e. no replicated rows or columns
        assert torch.allclose(decoded[0], image, atol=1e-6)
    mask = torch.zeros(1, 1, *size)
    mask[..., size[0] // 4:, size[1] // 4:] = 1
    assert editor.encode_mask(mask, latent).shape == (1, 1, *latent.shape[-2:])


def test_group_shares_editor_noise_and_groups_differ():
    nodes = {'n0': ConditionNode('n0', 'entity', 'Is there a cup?', 'cup', (0.1, 0.1, 0.4, 0.4), scope='local')}
    dag, seen = ConditionDAG(nodes), []
    state = CanvasState(torch.zeros(3, 16, 16), {'n0': 0.1}, set())
    action = ControllerAction('add_entity', (0.1, 0.1, 0.4, 0.4), 'Add a cup.')
    controller = SimpleNamespace(sample=lambda *args, **kwargs: {'action': action})
    editor = SimpleNamespace(render=lambda image, act, seed: seen.append(seed) or torch.ones(3, 4, 4))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.9), None)
    harness.sample_group(state, dag, controller, editor, 'n0', group_size=3, seed=100)
    harness.sample_group(state, dag, controller, editor, 'n0', group_size=2, seed=200)
    assert seen == [100, 100, 100, 200, 200]


def test_validate_dataset_requires_graphs_after_planning():
    nodes = [{'node_id': f'n{i}', 'node_type': 'entity', 'question': f'Is there object {i}?', 'entity_phrase': f'object {i}'}
             for i in range(10)]
    planned = [{'turns': [{'user_prompt': 'x', 'graph': {'nodes': nodes}}]}]
    unplanned = planned + [{'turns': ['draw something']}]
    validate_dataset(unplanned, 1, False)  # planner still to run: tiers are checked later
    with pytest.raises(ValueError):
        validate_dataset(unplanned, 1, False, require_graphs=True)
    with pytest.raises(ValueError):
        validate_dataset(planned, 1, False, require_graphs=True)  # one session is not 100 per tier
    validate_dataset(planned, 1, True, require_graphs=True)  # diagnostic runs skip the tier rule


def test_lora_targets_cover_every_decoder_projection_but_not_vision():
    class Layer(torch.nn.Module):
        def __init__(self, full):
            super().__init__()
            if full:
                self.self_attn = torch.nn.ModuleDict({k: torch.nn.Linear(2, 2) for k in ('q_proj', 'k_proj', 'v_proj', 'o_proj')})
            else:
                self.linear_attn = torch.nn.ModuleDict({k: torch.nn.Linear(2, 2) for k in ('in_proj_qkvz', 'in_proj_ba', 'out_proj')})
                self.linear_attn['conv1d'] = torch.nn.Conv1d(2, 2, 1)
            self.mlp = torch.nn.ModuleDict({k: torch.nn.Linear(2, 2) for k in ('gate_proj', 'up_proj', 'down_proj')})

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([Layer(False), Layer(True)])
    model.model.visual = torch.nn.ModuleDict({'qkv': torch.nn.Linear(2, 2), 'o_proj': torch.nn.Linear(2, 2)})
    model.lm_head = torch.nn.Linear(2, 2)
    import re
    pattern = re.compile(lora_target_modules(model))
    hits = {name for name, module in model.named_modules() if pattern.fullmatch(name)}
    assert len(hits) == 3 + 3 + 4 + 3 and not any('visual' in h or 'conv1d' in h or 'lm_head' in h for h in hits)
    assert lora_target_modules(torch.nn.Linear(2, 2)) == ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def test_planner_graph_metrics():
    from run_eval_planner import score
    response = {'nodes': [{'node_id': 'cup', 'node_type': 'entity', 'question': 'Is there a cup?', 'entity_phrase': 'cup', 'scope': 'local'},
                          {'node_id': 'red', 'node_type': 'attribute', 'question': 'Is the cup red?', 'entity_phrase': 'cup',
                           'scope': 'local', 'depends_on': ['cup']}], 'comp_edges': [['cup', 'red']]}
    record = {'user_prompt': 'A red cup.', 'previous_dag': None, 'response': response}
    perfect = score(record, json.dumps(response))
    assert perfect['valid'] and perfect['node_f1'] == 1 and perfect['comp_f1'] == 1 and perfect['dep_f1'] == 1
    renamed = json.loads(json.dumps(response).replace('"cup"', '"c1"').replace('"red"', '"c2"'))
    renamed['nodes'][1]['scope'] = 'unknown'
    partial = score(record, 'Here you go: ' + json.dumps(renamed))
    assert partial['node_f1'] == 1 and partial['scope_acc'] == 0.5 and partial['comp_f1'] == 1
    assert not score(record, 'no json here')['valid']
    follow = {'user_prompt': 'Make it blue instead.', 'previous_dag': dag_to_dict(ConditionDAG(
        {n['node_id']: ConditionNode(**n) for n in response['nodes']}, comp_edges=[('cup', 'red')])),
        'response': {'nodes': [{'node_id': 'blue', 'node_type': 'attribute', 'question': 'Is the cup blue?', 'entity_phrase': 'cup',
                                'scope': 'local', 'depends_on': ['cup']}], 'superseded_ids': ['red']}}
    result = score(follow, json.dumps(follow['response']))
    assert result['node_f1'] == 1 and result['superseded_f1'] == 1 and result['reference_nodes'] == 1


def test_rejection_sampling_resumes_without_duplicates(tmp_path):
    from redag.learn.sft_trainer import collect_controller_rs
    graph = {'nodes': [{'node_id': 'n0', 'node_type': 'entity', 'question': 'Is there a cup?', 'entity_phrase': 'cup',
                        'bbox': [0.1, 0.1, 0.5, 0.5], 'scope': 'local'}]}
    records = [{'id': f's{i}', 'image': torch.rand(3, 16, 16), 'graph': graph, 'scores': {'n0': 0.1}} for i in range(3)]
    action = ControllerAction('add_entity', (0.1, 0.1, 0.5, 0.5), 'Add a cup.')
    controller = SimpleNamespace(sample=lambda *args, **kwargs: {'action': action})
    editor = SimpleNamespace(render=lambda image, act, seed: torch.ones(3, 4, 4))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.95), None)
    output = tmp_path / 'rs.jsonl'
    first = collect_controller_rs(records, harness, controller, editor, output)
    assert first['accepted'] == 3
    with output.open('a') as stream:
        stream.write('{"id": "s9", "trunc')  # crash in the middle of a write
    second = collect_controller_rs(records, harness, controller, editor, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert second == {'parents': 3, 'accepted': 0, 'previously_accepted': 3}
    assert [r['id'] for r in rows] == ['s0', 's1', 's2'] and all((tmp_path / r['image']).exists() for r in rows)


def test_rejection_sampling_draws_more_candidates_for_hard_targets_and_logs_groups(tmp_path):
    from redag.learn.sft_trainer import collect_controller_rs
    easy = {'nodes': [{'node_id': 'n0', 'node_type': 'entity', 'question': 'Is there a cup?', 'entity_phrase': 'cup',
                       'bbox': [0.1, 0.1, 0.5, 0.5], 'scope': 'local'}]}
    hard = {'nodes': [{'node_id': 'n0', 'node_type': 'prohibition', 'question': 'Is the image free of cups?', 'entity_phrase': 'cup',
                       'scope': 'global'}]}
    records = [{'id': 'easy', 'image': torch.rand(3, 16, 16), 'graph': easy, 'scores': {'n0': 0.1}},
               {'id': 'hard', 'image': torch.rand(3, 16, 16), 'graph': hard, 'scores': {'n0': 0.1}}]
    calls = []
    def sample(state, graph, target, **kwargs):
        calls.append(target)
        return {'action': ControllerAction('remove_extra', (0.1, 0.1, 0.5, 0.5), 'Remove the cup.')}
    controller = SimpleNamespace(sample=sample)
    editor = SimpleNamespace(render=lambda image, act, seed: torch.ones(3, 4, 4))
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: 0.95), None)
    output = tmp_path / 'rs.jsonl'
    collect_controller_rs(records, harness, controller, editor, output, group_size=2, hard_multiplier=3)
    assert len(calls) == 2 + 6
    groups = [json.loads(line) for line in (tmp_path / 'rs.groups.jsonl').read_text().splitlines()]
    assert [(g['id'], g['target_type'], len(g['candidates'])) for g in groups] == [('easy', 'entity', 2), ('hard', 'prohibition', 6)]
    assert all(set(c) == {'action', 'reward', 'accepted'} for g in groups for c in g['candidates'])


def test_saved_images_appear_only_when_complete(tmp_path, monkeypatch):
    import os
    from redag.verify.grounder import load_image, save_image
    seen, real = [], os.replace
    monkeypatch.setattr(os, 'replace', lambda a, b: seen.append((Path(a).name, Path(b).exists())) or real(a, b))
    image = torch.rand(3, 8, 8)
    save_image(image, tmp_path / 'a.png')
    assert seen == [('a.png.tmp', False)] and not (tmp_path / 'a.png.tmp').exists()
    assert torch.allclose(load_image(tmp_path / 'a.png'), (image * 255).round() / 255, atol=1e-6)
