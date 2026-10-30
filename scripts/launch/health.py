"""Pipeline health report: early warnings from what the jobs have written so far (standard library, a few seconds).

  bash scripts/launch/go.sh health
The scheduler refreshes runs/HEALTH.md and runs/state/health.json every 10 minutes (HEALTH_EVERY_MIN) and sends a
notification when a check turns yellow (warn) or red (bad).  The thresholds are rules of thumb for spotting trouble
early (a stalled RL, a starved rejection sampler, a missing effect); they are not results and change nothing.
"""
import json
import statistics
import time
from pathlib import Path

OK, WARN, BAD, WAIT = 'ok', 'warn', 'bad', 'wait'
ICON = {OK: 'OK  ', WARN: 'WARN', BAD: 'BAD ', WAIT: '... '}
RANK = {WAIT: 0, OK: 1, WARN: 2, BAD: 3}


def rows(path):
    out = []
    try:
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def item(key, stage, status, value, detail=''):
    return dict(key=key, stage=stage, status=status, value=value, detail=detail)


def annotation(root):
    stats = {}
    for path in sorted((root / 'data' / 'processed' / 'graphs').glob('*.jsonl')):
        counts = stats.setdefault(path.name.split('.')[0], {'ok': 0, 'repaired': 0, 'fallback': 0})
        for row in rows(path):
            status = row.get('annotation', {}).get('status')
            if status in counts:
                counts[status] += 1
    out = []
    for source, counts in sorted(stats.items()):
        n = sum(counts.values())
        fallback = counts['fallback'] / n if n else 0.0
        status = WAIT if n < 30 else BAD if fallback > 0.5 else WARN if fallback > 0.25 else OK
        out.append(item(f'annotation/{source}', '1 data', status, f'{n} graphs, {fallback:.0%} fallback',
                        'fallback graphs are not used; many of them can leave a K tier short' if status in (WARN, BAD) else ''))
    return out


def structure(graph):
    """Shape of one condition DAG: depth (longest depends_on chain), share of local conditions, share of attribute /
    relation conditions that depend on something, and whether it degenerates (flat although it has structured nodes,
    or a single chain)."""
    nodes = {n.get('node_id'): n for n in graph.get('nodes', []) if isinstance(n, dict)}
    deps = {i: [d for d in (n.get('depends_on') or []) if d in nodes and d != i] for i, n in nodes.items()}
    memo = {}

    def depth(i, seen=frozenset()):
        if i not in memo:
            memo[i] = 0 if i in seen else 1 + max((depth(d, seen | {i}) for d in deps[i]), default=-1)
        return memo[i]
    n = len(nodes)
    structured = [i for i, x in nodes.items() if x.get('node_type') in ('attribute', 'relation')]
    longest = max((depth(i) for i in nodes), default=0)
    return dict(n=n, depth=longest, local=sum(x.get('scope') == 'local' for x in nodes.values()) / max(n, 1),
                linked=sum(1 for i in structured if deps[i]) / len(structured) if structured else None,
                flat_structured=longest == 0 and bool(structured), chain=n >= 6 and longest >= 0.8 * n)


def graph_structure(root):
    """Degenerate graphs would erase what the DAG adds over a flat checklist (local verification, dependency-aware
    repair): flag sources whose graphs are flat despite attributes/relations, mostly global, unlinked or chain-like.
    Flat graphs of pure presence checklists (e.g. CoReBench multi-instance) are normal and not flagged."""
    by_source = {}
    for path in sorted((root / 'data' / 'processed' / 'graphs').glob('*.jsonl')):
        for row in rows(path):
            if row.get('annotation', {}).get('status') in ('ok', 'repaired') and isinstance(row.get('graph'), dict):
                by_source.setdefault(row.get('source') or path.name.split('.')[0], []).append(structure(row['graph']))
    out = []
    for source, shapes in sorted(by_source.items()):
        if len(shapes) < 30:
            out.append(item(f'graphs/{source}', '1 data', WAIT, f'{len(shapes)} graphs'))
            continue
        mean = lambda key: statistics.fmean(s[key] for s in shapes if s[key] is not None) if any(s[key] is not None for s in shapes) else 1.0
        share = lambda key: sum(1 for s in shapes if s[key]) / len(shapes)
        depths = [s['depth'] for s in shapes]
        problems = []
        if share('flat_structured') > 0.15:
            problems.append(f"{share('flat_structured'):.0%} have attributes/relations but no dependency at all")
        if mean('local') < 0.3:
            problems.append(f"only {mean('local'):.0%} of conditions are local")
        if mean('linked') < 0.6:
            problems.append(f"only {mean('linked'):.0%} of attributes/relations depend on an entity")
        if share('chain') > 0.05:
            problems.append(f"{share('chain'):.0%} are single chains")
        out.append(item(f'graphs/{source}', '1 data', WARN if problems else OK,
                        f"{len(shapes)} graphs, {statistics.fmean(s['n'] for s in shapes):.0f} conditions, depth {statistics.fmean(depths):.1f} "
                        f"(max {max(depths)}, {sum(d == 0 for d in depths) / len(depths):.0%} flat), local {mean('local'):.0%}, "
                        f"linked {mean('linked'):.0%}", '; '.join(problems)))
    return out


def data_sets(root):
    manifest = load(root / 'data' / 'final' / 'manifest.json')
    if not manifest:
        return [item('sets', '1 data', WAIT, 'not built yet')]
    short = manifest.get('short_tiers')
    files = {name: info.get('rows') for name, info in manifest.get('files', {}).items()}
    return [item('sets/eval_tiers', '1 data', BAD if short else OK, f"eval K counts {manifest.get('eval_k')}",
                 f'short tiers: {short}' if short else ''),
            item('sets/files', '1 data', OK, ', '.join(f'{k.replace(".jsonl", "")} {v}' for k, v in files.items()))]


def loss_trend(path, name):
    losses = [r['loss'] for r in rows(path) if isinstance(r.get('loss'), (int, float))]
    if len(losses) < 20:
        return item(name, '2 SFT', WAIT, f'{len(losses)} steps')
    k = max(5, len(losses) // 10)
    first, last = statistics.fmean(losses[:k]), statistics.fmean(losses[-k:])
    rising = len(losses) >= 200 and last > first * 1.05
    return item(name, '2 SFT', WARN if rising else OK, f'{len(losses)} steps, loss {first:.4f} -> {last:.4f}',
                'loss is rising' if rising else '')


def sft(runs):
    out = [loss_trend(runs / 'sft' / kind / 'metrics.jsonl', f'sft/{kind}') for kind in ('editor', 'planner', 'controller')]
    zero, tuned = load(runs / 'eval' / 'planner_zero_shot.json'), load(runs / 'eval' / 'planner_sft.json')
    if zero and tuned:
        a, b = zero['summary'].get('node_f1'), tuned['summary'].get('node_f1')
        if a is not None and b is not None:
            out.append(item('sft/planner_graph_f1', '2 SFT', OK if b > a else WARN, f'node F1 zero-shot {a:.3f} -> SFT {b:.3f}',
                            '' if b > a else 'planner SFT does not beat the zero-shot planner'))
    return out


def rejection_sampling(root, runs):
    parts = sorted((root / 'data' / 'final').glob('controller_sft.part*.jsonl'))
    accepted = sum(len(rows(p)) for p in parts) or len(rows(root / 'data' / 'final' / 'controller_sft.jsonl'))
    seen = sum((load(p) or {}).get('done', 0) for p in (runs / 'state' / 'jobs').glob('rs_*.progress'))
    if seen < 50:
        return [item('rs/acceptance', '2 SFT', WAIT, f'{accepted} accepted of {seen} states')]
    rate = accepted / seen
    status = BAD if rate < 0.05 else WARN if rate < 0.15 else OK
    return [item('rs/acceptance', '2 SFT', status, f'{accepted} accepted of {seen} states ({rate:.0%})',
                 'few accepted edits: controller SFT gets little data and RL may see no reward differences' if status != OK else '')]


def rl(runs):
    out = []
    for metrics in sorted(runs.glob('rl/**/metrics.jsonl')):
        if 'smoke' in metrics.parts or 'wiring' in metrics.parts:
            continue
        name = str(metrics.parent.relative_to(runs / 'rl'))
        steps = rows(metrics)
        if len(steps) < 3:
            out.append(item(f'rl/{name}', '3 RL', WAIT, f'{len(steps)} steps'))
            continue
        last, recent = steps[-1], steps[-8:]
        no_update = sum(1 for s in steps if not s.get('optimizer_updates')) / len(steps)
        same_rewards = sum(1 for s in steps if (s.get('reward_variance') or 0) < 1e-8) / len(steps)
        kl_stops = sum(1 for s in steps if s.get('kl_early_stop')) / len(steps)
        guards = sum(1 for s in steps if s.get('ratio_guard_hit'))
        gain = statistics.fmean(s.get('dev_acceptance_gain', 0.0) for s in recent)
        problems = []
        if no_update > 0.5:
            problems.append(f'{no_update:.0%} of steps made no update (no learning signal)')
        if same_rewards > 0.5:
            problems.append(f'{same_rewards:.0%} of steps had identical rewards in every group')
        if kl_stops > 0.5:
            problems.append(f'{kl_stops:.0%} of steps stopped at the KL limit (steps too large)')
        if guards:
            problems.append(f'{guards} ratio-guard hits')
        if gain < -0.1:
            problems.append(f'dev acceptance {gain:+.2f} below the stage start over the last {len(recent)} steps')
        status = BAD if no_update > 0.8 or gain < -0.2 else WARN if problems else OK
        out.append(item(f'rl/{name}', '3 RL', status,
                        f"step {last.get('step')} ({last.get('stage')}): dev acceptance {last.get('dev_acceptance_p', 0):.2f} "
                        f"(stage start {last.get('dev_initial_acceptance_p', 0):.2f}), mean reward {last.get('mean_reward', 0):.3f}",
                        '; '.join(problems)))
    return out


def evaluations(runs):
    groups = {}
    for cache in sorted((runs / 'eval').glob('*.cases.jsonl')):
        experiment = cache.name.split('.')[0].split('_')[0]
        for row in rows(cache)[1:]:
            case = row.get('case')
            if not isinstance(case, dict) or '/' not in str(case.get('group', '')):
                continue
            stage, mode, _ = case['group'].rsplit('/', 2)
            g = groups.setdefault(f'{experiment}:{stage}/{mode}', {'n': 0, 'satisfied': 0, 'steps': 0, 'accepted': 0})
            g['n'] += 1
            g['satisfied'] += (case.get('execution') or {}).get('stop_reason') == 'satisfied'
            g['steps'] += len(case.get('steps', []))
            g['accepted'] += sum(1 for s in case.get('steps', []) if s.get('accepted'))
    out = [item(f'eval/{key}', '4 eval', OK, f"{g['n']} sessions: all conditions met in {g['satisfied'] / g['n']:.0%}, "
                f"{g['accepted'] / max(g['steps'], 1):.0%} of edit steps accepted") for key, g in groups.items()]
    exp1 = {key.split('/')[-1]: g for key, g in groups.items() if key.startswith('exp1:')}
    if {'redag_full', 'flat_full'} <= set(exp1) and min(exp1['redag_full']['n'], exp1['flat_full']['n']) >= 30:
        ours, flat = (exp1[m]['satisfied'] / exp1[m]['n'] for m in ('redag_full', 'flat_full'))
        out.append(item('eval/exp1_effect', '4 eval', OK if ours > flat else WARN,
                        f'fully satisfied sessions: ReDAG {ours:.0%} vs flat checklist {flat:.0%} (training verifier as a proxy)',
                        '' if ours > flat else 'the environment effect is not visible yet at this budget'))
    return out


def report(root, runs):
    root, runs = Path(root), Path(runs)
    checks = (annotation(root) + graph_structure(root) + data_sets(root) + sft(runs) + rejection_sampling(root, runs) + rl(runs)
              + evaluations(runs))
    return dict(updated=time.time(), checks=checks)


def markdown(health):
    lines = [f"# Pipeline health ({time.strftime('%F %T', time.localtime(health['updated']))})", '',
             '| | check | value | note |', '|---|---|---|---|']
    for c in health['checks']:
        lines.append(f"| {ICON[c['status']].strip()} | {c['key']} | {c['value']} | {c['detail']} |")
    return '\n'.join(lines) + '\n'


def refresh(root, runs, notifier=None):
    """Recompute, write runs/HEALTH.md and runs/state/health.json, notify about checks that got worse."""
    runs = Path(runs)
    previous = {c['key']: c['status'] for c in (load(runs / 'state' / 'health.json') or {}).get('checks', [])}
    health = report(root, runs)
    (runs / 'state').mkdir(parents=True, exist_ok=True)
    tmp = runs / 'state' / 'health.json.tmp'
    tmp.write_text(json.dumps(health, indent=1, ensure_ascii=False), encoding='utf-8')
    tmp.replace(runs / 'state' / 'health.json')
    (runs / 'HEALTH.md').write_text(markdown(health), encoding='utf-8')
    worse = [c for c in health['checks'] if c['status'] in (WARN, BAD) and RANK[c['status']] > RANK.get(previous.get(c['key'], OK), 1)]
    if worse and notifier is not None:
        notifier.send(f"health: {len(worse)} check(s) need attention",
                      '\n'.join(f"[{c['status']}] {c['key']}: {c['value']} {c['detail']}" for c in worse))
    return health, worse
