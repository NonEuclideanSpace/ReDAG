"""Interrupted runs must end exactly like uninterrupted ones (SFT, alternating RL, evaluation sessions)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from redag.graph.schema import ConditionDAG, ConditionNode, ControllerAction
from redag.harness.environment import ReDAGHarness
from redag.learn import checkpoint
from redag.learn import rl_trainer
from redag.learn.sft_trainer import train_sft
from redag.verify.verifier import TwoStageVerifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


# ----------------------------------------------------------------------------------------------- SFT
class TinyLM:
    """Planner/Controller SFT interface around one LoRA-like matrix."""

    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        self.model = torch.nn.Module()
        self.model.lora_A = torch.nn.Parameter(torch.randn(8, 5, generator=generator) * 0.1)

    def lora_parameters(self):
        return [self.model.lora_A]

    def snapshot(self):
        return {'lora_A': self.model.lora_A.detach().clone()}

    def load_snapshot(self, snapshot):
        with torch.no_grad():
            self.model.lora_A.copy_(snapshot['lora_A'])

    def sft_batch(self, records):
        x = torch.tensor(records[0]['x'])
        return (x @ self.model.lora_A).unsqueeze(0), torch.tensor(records[0]['y']).unsqueeze(0)

    def save_adapter(self, path):
        torch.save(self.snapshot(), Path(path) / 'adapter.pt')


def sft_records():
    generator = torch.Generator().manual_seed(1)
    return [{'x': torch.randn(4, 8, generator=generator).tolist(), 'y': torch.randint(0, 5, (4,), generator=generator).tolist()}
            for _ in range(7)]


def test_sft_resume_matches_uninterrupted_run(tmp_path):
    records = sft_records()
    clean = TinyLM()
    expected = train_sft(clean, records, tmp_path / 'clean', 'controller', epochs=2, batch_size=2, learning_rate=0.05, checkpoint_minutes=0)
    with pytest.raises(checkpoint.Interrupted):
        train_sft(TinyLM(), records, tmp_path / 'cut', 'controller', epochs=2, batch_size=2, learning_rate=0.05,
                  checkpoint_minutes=1e9, interrupt_after=3)
    assert (tmp_path / 'cut' / 'resume' / 'COMPLETE').exists()
    resumed = TinyLM()
    result = train_sft(resumed, records, tmp_path / 'cut', 'controller', epochs=2, batch_size=2, learning_rate=0.05, checkpoint_minutes=1e9)
    assert result == expected
    assert torch.equal(resumed.model.lora_A, clean.model.lora_A)
    rows = lambda name: (tmp_path / name / 'metrics.jsonl').read_text().splitlines()
    assert rows('cut') == rows('clean') and len(rows('clean')) == 8
    assert not (tmp_path / 'cut' / 'resume').exists()
    again = TinyLM()  # a finished run returns at once
    assert train_sft(again, records, tmp_path / 'cut', 'controller', epochs=2, batch_size=2, learning_rate=0.05) == expected
    with pytest.raises(ValueError):  # a checkpoint from other settings is refused
        with pytest.raises(checkpoint.Interrupted):
            train_sft(TinyLM(), records, tmp_path / 'other', 'controller', epochs=2, batch_size=2, learning_rate=0.05, interrupt_after=1)
        train_sft(TinyLM(), records, tmp_path / 'other', 'controller', epochs=3, batch_size=2, learning_rate=0.05)


def test_checkpoint_survives_crash_between_renames(tmp_path):
    checkpoint.save(tmp_path, {'step': 1}, tensors={'m': {'w': torch.ones(2)}})
    checkpoint.save(tmp_path, {'step': 2}, tensors={'m': {'w': torch.full((2,), 2.0)}})
    (tmp_path / 'resume').rename(tmp_path / 'resume.tmp')  # as if the process died before the final rename
    state, payload = checkpoint.load(tmp_path)
    assert state['step'] == 2 and torch.equal(payload['tensors']['m']['w'], torch.full((2,), 2.0))
    assert checkpoint.keep_jsonl_rows(tmp_path / 'missing.jsonl', lambda row: True) == []
    log = tmp_path / 'm.jsonl'
    log.write_text('{"step": 0}\n{"step": 1}\n{"step": 2}\n{"ste')
    assert checkpoint.keep_jsonl_rows(log, lambda row: row['step'] < 2) == [{'step': 0}, {'step': 1}]
    assert log.read_text() == '{"step": 0}\n{"step": 1}\n'


# ----------------------------------------------------------------------------------------------- RL
class FakeAdapter:
    def __init__(self, value):
        self.weight = torch.nn.Parameter(torch.full((3,), value))

    def lora_parameters(self):
        return [self.weight]

    def snapshot(self):
        return {'lora_w': self.weight.detach().clone()}

    def load_snapshot(self, snapshot):
        with torch.no_grad():
            self.weight.copy_(snapshot['lora_w'])

    def save_adapter(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.snapshot(), Path(path) / 'adapter.pt')


class FakeController(FakeAdapter):
    def sample(self, state, dag, target, seed=42, temperature=0.8, detected_count=1):
        return {'action': ControllerAction('add_entity', (0.1, 0.1, 0.6, 0.6), f'Add a cup {seed % 7}.'), 'inputs': {'ids': torch.ones(2)}}


class FakeEditor(FakeAdapter):
    num_steps, eta, cfg_scale = 6, 0.35, 3.5

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def render(self, image, action, seed=42):
        generator = torch.Generator().manual_seed(int(seed) % 100000)
        return (torch.rand(3, 6, 6, generator=generator) * 0.5 + self.weight.detach().mean()).clamp(0, 1)

    def prepare(self, image, action):
        return {}

    def sample(self, image, action, seed=42, conditioning=None):
        return {'edited_crop': self.render(image, action, seed)}


def fake_rollout(stage, groups, controller, editor, optimizer, old, ctrl_ref, edit_ref, cfg):
    target = controller.weight if stage == 'ctrl_rl' else editor.weight
    advantage = sum(float(adv.sum()) + float(adv.abs().mean()) for _, adv, _ in groups)
    with torch.no_grad():
        target.add_(0.01 * advantage + 0.001)
    return dict(optimizer_updates=1, updated=True, loss=advantage, max_old_policy_kl=0.0, kl_early_stop=False, actual_action_tokens=0)


def fake_development(records, harness, controller, editor, reference):
    return dict(dev_acceptance_p=float(torch.sigmoid(controller.weight.sum() - editor.weight.sum())), dev_attempts=1, ctrl_kl_sft=0.0)


def rl_run(output, interrupt_after=None):
    node = ConditionNode('cup', 'entity', 'Is there a cup?', 'cup', (0.1, 0.1, 0.6, 0.6), scope='local')
    graph = {'nodes': [dict(node_id='cup', node_type='entity', question='Is there a cup?', entity_phrase='cup', bbox=[0.1, 0.1, 0.6, 0.6], scope='local')]}
    generator = torch.Generator().manual_seed(3)
    records = [{'id': f'p{i}', 'image': torch.rand(3, 16, 16, generator=generator), 'graph': graph, 'scores': {'cup': 0.1}} for i in range(8)]
    dev = [{'id': 'd0', 'image': torch.rand(3, 16, 16, generator=generator), 'graph': graph, 'scores': {'cup': 0.1}}]
    harness = ReDAGHarness(TwoStageVerifier(lambda image, question: float(image.mean())), None)
    controller, editor = FakeController(0.0), FakeEditor(0.2)
    history = rl_trainer.train_alternating(harness, controller, editor, records, dev, output, rounds=2, batch_size=4,
                                           rollout_steps=1, interrupt_after=interrupt_after)
    del node
    return history, controller, editor


def comparable(rows):
    return [{k: v for k, v in r.items() if k not in ('step_wall_seconds', 'peak_cuda_allocated_bytes')} for r in rows]


def test_alternating_rl_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_trainer, 'optimize_rollout', fake_rollout)
    monkeypatch.setattr(rl_trainer, 'evaluate_rl_development', fake_development)
    clean_history, clean_ctrl, clean_edit = rl_run(tmp_path / 'clean')
    assert len(clean_history) == 8 and [r['step'] for r in clean_history] == list(range(8))
    for cut in (1, 2, 5):  # inside a stage, at a stage boundary, in round two
        out = tmp_path / f'cut{cut}'
        with pytest.raises(checkpoint.Interrupted):
            rl_run(out, interrupt_after=cut)
        history, ctrl, edit = rl_run(out)
        assert comparable(history) == comparable(clean_history), cut
        assert comparable(json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()) == comparable(clean_history)
        assert torch.equal(ctrl.weight, clean_ctrl.weight) and torch.equal(edit.weight, clean_edit.weight), cut
        for stage in ('sft', 'ctrl_rl', 'edit_rl', 'alternating_rl'):
            assert torch.equal(torch.load(out / stage / 'controller' / 'adapter.pt')['lora_w'],
                               torch.load(tmp_path / 'clean' / stage / 'controller' / 'adapter.pt')['lora_w']), (cut, stage)
        assert (out / 'stage_checkpoints.json').read_text() == (tmp_path / 'clean' / 'stage_checkpoints.json').read_text()
        assert (out / 'rl_complete.json').exists() and not (out / 'resume').exists()
    finished, _, _ = rl_run(tmp_path / 'clean')  # complete runs return their history without training
    assert comparable(finished) == comparable(clean_history)


# ----------------------------------------------------------------------------------------------- evaluation
def eval_main(monkeypatch, tmp_path, argv, fail_after=None):
    import run_eval_benchmarks as reb
    calls = {'n': 0}
    graph = ConditionDAG({'cup': ConditionNode('cup', 'entity', 'Is there a cup?', 'cup', (0.1, 0.1, 0.5, 0.5), scope='local')})

    def run_record(record, harness, controller, editor, max_steps, seed, planner=None):
        calls['n'] += 1
        if fail_after is not None and calls['n'] > fail_after:
            raise RuntimeError('simulated crash')
        return SimpleNamespace(image=None), graph, []

    def export_case(state, dag, trace, folder, identifier, group, budget, config):
        return {'id': identifier, 'group': group, 'views': {identifier + '_final': {'image': f'/x/{identifier}.png'}},
                'final': identifier + '_final', 'steps': [], 'execution': {}, 'mode_seen': group}

    harness = SimpleNamespace(last_run={}, config=None, mode=None)
    monkeypatch.setattr(reb, 'load_runtime', lambda args: (harness, None, None))
    monkeypatch.setattr(reb, 'run_record', run_record)
    monkeypatch.setattr(reb, 'export_case', export_case)
    monkeypatch.setattr(sys, 'argv', ['run_eval_benchmarks.py'] + [str(a) for a in argv])
    reb.main()
    return calls['n']


def test_evaluation_resumes_per_session(tmp_path, monkeypatch):
    data = tmp_path / 'eval.jsonl'
    turn = {'user_prompt': 'A cup.', 'graph': {'nodes': [{'node_id': 'cup', 'node_type': 'entity', 'question': 'Is there a cup?',
                                                         'entity_phrase': 'cup', 'scope': 'local'}]}}
    data.write_text(''.join(json.dumps({'id': f's{i}', 'image': f'img{i}.png', 'k': 10, 'turns': [turn]}) + '\n' for i in range(3)))
    base = ['--exp_id', 1, '--data', data, '--allow-small-dataset', '--exp1_modes', 'flat_full', 'redag_full',
            '--max_steps', 2, '--generation_budget', 3, '--verification_budget', 50]
    assert eval_main(monkeypatch, tmp_path, base + ['--output', tmp_path / 'clean.json']) == 6
    with pytest.raises(RuntimeError):
        eval_main(monkeypatch, tmp_path, base + ['--output', tmp_path / 'cut.json'], fail_after=4)
    assert eval_main(monkeypatch, tmp_path, base + ['--output', tmp_path / 'cut.json']) == 2  # only the missing sessions
    clean, cut = (json.loads((tmp_path / f'{n}.json').read_text()) for n in ('clean', 'cut'))
    assert [c['id'] for c in cut['evaluation_cases']] == [c['id'] for c in clean['evaluation_cases']]
    assert [c['group'] for c in cut['evaluation_cases']] == [c['group'] for c in clean['evaluation_cases']]
    with pytest.raises(ValueError):  # the cache of one setting is never mixed into another
        eval_main(monkeypatch, tmp_path, base[:-1] + [99, '--output', tmp_path / 'cut.json'])
    # experiment 3 parts only need the stages they run (pre-RL stage file)
    stages = tmp_path / 'stages.json'
    stages.write_text(json.dumps({'zero_shot': {'controller': None, 'editor': None}}))
    exp3 = ['--exp_id', 3, '--data', data, '--allow-small-dataset', '--stage_checkpoints', stages, '--stages', 'zero_shot',
            '--shard', 0, '--num_shards', 2, '--max_steps', 2, '--generation_budget', 3, '--verification_budget', 50,
            '--output', tmp_path / 'exp3_zero_shot.s0.json']
    assert eval_main(monkeypatch, tmp_path, exp3) == 2 * 3  # records 0 and 2, three seeds
    part = json.loads((tmp_path / 'exp3_zero_shot.s0.json').read_text())
    assert set(part['final_views']) == {'0:0', '0:1', '0:2', '2:0', '2:1', '2:2'}


def test_experiment3_sft_stage_reuses_experiment1_sessions(tmp_path, monkeypatch):
    data = tmp_path / 'eval.jsonl'
    turn = {'user_prompt': 'A cup.', 'graph': {'nodes': [{'node_id': 'cup', 'node_type': 'entity', 'question': 'Is there a cup?',
                                                         'entity_phrase': 'cup', 'scope': 'local'}]}}
    data.write_text(''.join(json.dumps({'id': f's{i}', 'image': f'img{i}.png', 'k': 10, 'turns': [turn]}) + '\n' for i in range(3)))
    for kind in ('controller', 'editor'):
        (tmp_path / 'sft' / kind).mkdir(parents=True)
    common = ['--data', data, '--allow-small-dataset', '--max_steps', 2, '--generation_budget', 3, '--verification_budget', 50]
    exp1 = ['--exp_id', 1, '--exp1_modes', 'flat_full', 'redag_full', '--controller_adapter', tmp_path / 'sft' / 'controller',
            '--editor_adapter', tmp_path / 'sft' / 'editor', '--output', tmp_path / 'eval' / 'exp1.s0.json']
    assert eval_main(monkeypatch, tmp_path, common + exp1) == 6
    stages = tmp_path / 'eval' / 'stages.json'
    stages.write_text(json.dumps({'zero_shot': {'controller': None, 'editor': None},
                                  'sft': {'controller': '../sft/controller', 'editor': '../sft/editor'}}))
    exp3 = ['--exp_id', 3, '--stage_checkpoints', stages, '--stages', 'sft', '--output', tmp_path / 'eval' / 'exp3_sft.s0.json',
            '--reuse_cases', tmp_path / 'eval' / 'exp1.s0.cases.jsonl']
    assert eval_main(monkeypatch, tmp_path, common + exp3) == 3 * 2  # seeds 1 and 2 only; seed 0 comes from experiment 1
    part = json.loads((tmp_path / 'eval' / 'exp3_sft.s0.json').read_text())
    assert len(part['evaluation_cases']) == 9 and all(c['group'].startswith('sft/redag_full/') for c in part['evaluation_cases'])
    assert all(c['id'].startswith('case_s0_sft_') for c in part['evaluation_cases'])
    exp1_cases = {c['id'] for c in json.loads((tmp_path / 'eval' / 'exp1.s0.json').read_text())['evaluation_cases']}
    assert set(part['final_views']) == {f'{i}:{s}' for i in range(3) for s in range(3)} and exp1_cases
    # other budgets: nothing is reused
    other = [a if a != 50 else 60 for a in common]
    assert eval_main(monkeypatch, tmp_path, other + exp3[:-4] + ['--output', tmp_path / 'eval' / 'exp3_other.s0.json',
                                                                 '--reuse_cases', tmp_path / 'eval' / 'exp1.s0.cases.jsonl']) == 9
