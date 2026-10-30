import json
import subprocess
import sys
from pathlib import Path

import torch

from redag.config import ReDAGConfig
from redag.learn.rl_trainer import get_stochastic_strata, predict_cfg_velocity

ROOT = Path(__file__).resolve().parents[1]


class TwoInput(torch.nn.Module):
    def __init__(self, batched):
        super().__init__()
        self.batched_cfg = batched
        self.weight = torch.nn.Parameter(torch.tensor(0.3))
        self.calls = 0

    def forward(self, z_t, z_src, text_embeds, timesteps):
        self.calls += 1
        return self.weight * z_t - 0.2 * z_src + text_embeds.mean(dim=(1, 2)).view(-1, 1, 1, 1) + timesteps.view(-1, 1, 1, 1)


def test_batched_cfg_matches_two_forwards_including_gradients():
    torch.manual_seed(0)
    z, src, text, null, t = torch.randn(1, 4, 6, 8), torch.randn(1, 4, 6, 8), torch.randn(1, 5, 3), torch.zeros(1, 5, 3), torch.tensor([0.4])
    plain, fused = TwoInput(False), TwoInput(True)
    a = predict_cfg_velocity(plain, z, src, text, null, t, 3.5)
    b = predict_cfg_velocity(fused, z, src, text, null, t, 3.5)
    assert plain.calls == 2 and fused.calls == 1 and torch.allclose(a, b, atol=1e-6)
    a.sum().backward()
    b.sum().backward()
    assert torch.allclose(plain.weight.grad, fused.weight.grad)
    mismatched = predict_cfg_velocity(TwoInput(True), z, src, text, torch.zeros(1, 2, 3), t)
    assert mismatched.shape == z.shape  # falls back to two forwards when text lengths differ


def test_reduced_step_schedules_keep_four_strata():
    for steps in (16, 28):
        ReDAGConfig(num_denoise_steps=steps)
        strata = get_stochastic_strata(steps)
        assert len(strata) == 4 and all(strata) and sum(map(len, strata)) == steps - 1


def write_part(path, exp_id, cases, extra=None):
    part = {'exp_id': exp_id, 'diagnostic_only': True, 'evaluation_cases': cases, 'human_ratings': None,
            'human_evaluation_status': 'pending', 'budgets': {'verification': 10, 'generation': 4}}
    part.update(extra or {})
    path.write_text(json.dumps(part))


def test_merge_parts_and_blind_pairs(tmp_path):
    from PIL import Image
    stages = ('zero_shot', 'sft', 'ctrl_rl', 'edit_rl', 'alternating_rl')
    parts, views = [], {}
    for shard in range(2):
        cases, final = [], {}
        for stage in stages:
            for index in range(shard, 4, 2):
                for seed in range(3):
                    image = tmp_path / f'{stage}_{index}_{seed}.png'
                    Image.new('RGB', (4, 4)).save(image)
                    final.setdefault(f'{index}:{seed}', {})[stage] = str(image)
                    cases.append({'id': f'case_s{shard}_{stage}_{len(cases)}', 'group': f'{stage}/redag_full/10'})
        views.update(final)
        path = tmp_path / f'part{shard}.json'
        write_part(path, 3, cases, {'part': {'seed': 7}, 'final_views': final,
                                    'descriptions': {k: {'requirements': ['q'], 'source': None} for k in final}})
        parts.append(str(path))
    output = tmp_path / 'merged.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'merge_eval_results.py'), '--inputs', *parts, '--output', str(output),
                    '--allow-small-dataset'], check=True, capture_output=True)
    merged = json.loads(output.read_text())
    assert len(merged['evaluation_cases']) == 5 * 4 * 3 and len(merged['blind_pairs']) == 12
    assert all(p['a'] != p['b'] for p in merged['blind_pairs'])
    key = json.loads(output.with_suffix('.blind_key.json').read_text())
    assert all(row['a'] in stages and row['b'] in stages and row['a'] != row['b'] for row in key)
