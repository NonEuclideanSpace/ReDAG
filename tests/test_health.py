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


def test_rl_health_flags_a_policy_that_edits_the_whole_image(tmp_path):
    runs = tmp_path / 'runs'
    write(runs / 'rl/main/metrics.jsonl', [dict(step=i, stage='ctrl_rl', optimizer_updates=1, reward_variance=0.1, kl_early_stop=False,
                                                dev_acceptance_p=0.4, dev_initial_acceptance_p=0.4, dev_acceptance_gain=0.0, mean_reward=0.2,
                                                action_full_box_share=share, action_box_area_mean=0.5, action_ops={'modify_attr': 4})
                                           for i, share in enumerate([0.2] * 4 + [0.9] * 8)])
    check = next(c for c in health.rl(runs) if c['key'] == 'rl/main')
    assert check['status'] == 'warn' and 'whole image' in check['detail'] and 'whole-image boxes 90%' in check['value']
    write(runs / 'rl/main/metrics.jsonl', [dict(step=i, stage='ctrl_rl', optimizer_updates=1, reward_variance=0.1, kl_early_stop=False,
                                                dev_acceptance_p=0.4, dev_initial_acceptance_p=0.4, dev_acceptance_gain=0.0, mean_reward=0.2,
                                                action_full_box_share=0.25) for i in range(12)])
    assert next(c for c in health.rl(runs) if c['key'] == 'rl/main')['status'] == 'ok'


def test_health_flags_affected_subgraphs_that_cover_the_whole_graph(tmp_path):
    runs = tmp_path / 'runs'
    write(runs / 'rl/main/metrics.jsonl', [dict(step=i, stage='ctrl_rl', optimizer_updates=1, reward_variance=0.1, kl_early_stop=False,
                                                dev_acceptance_p=0.4, dev_initial_acceptance_p=0.4, dev_acceptance_gain=0.0, mean_reward=0.2,
                                                action_full_box_share=0.1, affected_share_mean=share) for i, share in enumerate([0.3] * 4 + [0.95] * 8)])
    check = next(c for c in health.rl(runs) if c['key'] == 'rl/main')
    assert check['status'] == 'warn' and 'full table' in check['detail'] and 'affected subgraph 95% of the graph' in check['value']
    views = {'b': {'nodes': {n: {} for n in 'abcd'}}}
    cases = [{'case': {'group': f'fixed_weights/{mode}/10', 'views': views, 'execution': {'stop_reason': 'step_budget'},
                       'steps': [{'before': 'b', 'affected': list('abcd' if mode == 'redag_full' else 'a'), 'accepted': True}]}}
             for mode in ('redag_full', 'flat_full')]
    write(runs / 'eval/exp1.s0.cases.jsonl', [{'fingerprint': 'x'}] + cases)
    checks = {c['key']: c for c in health.evaluations(runs)}
    assert checks['eval/exp1:fixed_weights/redag_full']['status'] == 'warn' and '100% of the graph' in checks['eval/exp1:fixed_weights/redag_full']['value']
    assert checks['eval/exp1:fixed_weights/flat_full']['status'] == 'ok'


def test_gates_hold_jobs_until_released(tmp_path, monkeypatch):
    import pipeline
    for name in ('CLAIMS_HOST', 'LOCAL_JOBS', 'EXTERNAL_JOBS', 'SHARED_JOBS', 'NOT_BEFORE', 'SYNC_PEER'):
        monkeypatch.delenv(name, raising=False)
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


def test_graph_structure_flags_degenerate_sources(tmp_path):
    def node(i, kind, deps=(), scope='local'):
        return {'node_id': i, 'node_type': kind, 'depends_on': list(deps), 'scope': scope}
    good = {'nodes': [node('e1', 'entity'), node('q1', 'attribute', ['e1']), node('q2', 'relation', ['e1', 'e2']), node('e2', 'entity'),
                      node('q3', 'attribute', ['q2'])]}
    unlinked = {'nodes': [node('e1', 'entity', scope='global'), node('q1', 'attribute', scope='global'), node('q2', 'relation', scope='global')]}
    presence = {'nodes': [node('q1', 'entity'), node('q2', 'entity'), node('q3', 'prohibition', scope='global')]}  # flat by nature
    write(tmp_path / 'data/processed/graphs/a.shard0of8.jsonl',
          [{'source': 'a', 'annotation': {'status': 'ok'}, 'graph': good}] * 20 + [{'source': 'a', 'annotation': {'status': 'ok'}, 'graph': presence}] * 20)
    write(tmp_path / 'data/processed/graphs/b.shard0of8.jsonl', [{'source': 'b', 'annotation': {'status': 'ok'}, 'graph': unlinked}] * 30)
    checks = {c['key']: c for c in health.graph_structure(tmp_path)}
    assert checks['graphs/a']['status'] == 'ok' and 'depth 1.0' in checks['graphs/a']['value']
    assert checks['graphs/b']['status'] == 'warn' and 'no dependency' in checks['graphs/b']['detail']
    assert health.structure(good)['depth'] == 2 and not health.structure(presence)['flat_structured']
