"""Health report and decision gates (standard library only)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'launch'))
import health  # noqa: E402


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_health_flags_problems_and_notifies_once(tmp_path):
    root, runs = tmp_path, tmp_path / 'runs'
    write(root / 'data/processed/graphs/corebench.shard0of8.jsonl',
          [{'annotation': {'status': 'ok'}}] * 20 + [{'annotation': {'status': 'fallback'}}] * 20)
    write(runs / 'rl/main/metrics.jsonl', [dict(step=i, stage='ctrl_rl', optimizer_updates=0, reward_variance=0.0, kl_early_stop=False,
                                                dev_acceptance_p=0.2, dev_initial_acceptance_p=0.4, dev_acceptance_gain=-0.2, mean_reward=0.0)
                                           for i in range(10)])
    header = {'fingerprint': {}}
    def reason(i, mode):  # flat checklist satisfied 2/3 of the time, ReDAG 1/3: the effect is missing
        return 'satisfied' if (i % 3 == 0) == (mode == 'redag_full') else 'step_budget'
    cases = [{'key': f'k{i}', 'case': {'group': f'fixed_weights/{mode}/10', 'execution': {'stop_reason': reason(i, mode)},
                                       'steps': [{'accepted': True}]}} for i, mode in enumerate(['redag_full', 'flat_full'] * 40)]
    write(runs / 'eval/exp1.s0.cases.jsonl', [header] + cases)
    (runs / 'state/jobs').mkdir(parents=True)
    (runs / 'state/jobs/rs_part0.progress').write_text(json.dumps({'done': 200, 'total': 250}))
    write(root / 'data/final/controller_sft.part0.jsonl', [{'id': i} for i in range(6)])
    sent = []
    notifier = type('N', (), {'send': lambda self, title, text='': sent.append((title, text))})()
    report, worse = health.refresh(root, runs, notifier)
    status = {c['key']: c['status'] for c in report['checks']}
    assert status['annotation/corebench'] == 'warn'            # 50 % fallback
    assert status['rl/main'] == 'bad'                          # no updates and falling dev acceptance
    assert status['rs/acceptance'] == 'bad'                    # 6 / 200 = 3 %
    assert status['eval/exp1_effect'] == 'warn'                # flat checklist ahead
    assert status['sets'] == 'wait' and (runs / 'HEALTH.md').exists()
    assert len(sent) == 1 and 'rl/main' in sent[0][1]
    health.refresh(root, runs, notifier)                       # unchanged problems are not sent again
    assert len(sent) == 1


def test_gates_hold_jobs_until_released(tmp_path, monkeypatch):
    import pipeline
    monkeypatch.setenv('RUNS', str(tmp_path / 'runs'))
    monkeypatch.setenv('HOLD', 'exp3 after_rl')
    monkeypatch.setattr(pipeline, 'query_gpus', lambda: [dict(index=0, name='GPU', used=0, total=97887)])
    s = pipeline.settings()
    scheduler, jobs, _, _ = pipeline.make_scheduler(argparse.Namespace(), s, quiet=True)
    by_name = {j.name: j for j in jobs}
    assert not by_name['exp3_zero_shot_s0'].ready() and not by_name['rl_exp2_step_full_table'].ready()
    assert by_name['exp1_s0'].ready is None and by_name['rl_main'].ready is None
    assert 'go.sh release exp3' in by_name['exp3_zero_shot_s0'].waiting_for
    assert pipeline.cmd_release(argparse.Namespace(gate='exp3'), s) == 0
    assert by_name['exp3_zero_shot_s0'].ready()
    assert not by_name['exp3_ctrl_rl_s0'].ready()               # also needs after_rl and its checkpoint
    assert '--reuse_cases' in by_name['exp3_sft_s3'].cmd and 'exp1_s3' in by_name['exp3_sft_s3'].deps
