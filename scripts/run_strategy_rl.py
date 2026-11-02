import argparse
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.graph.schema import CanvasState
from redag.harness.environment import add_runtime_arguments, consistent_count, load_runtime, run_record
from redag.learn.sft_trainer import read_records
from redag.learn.strategy import StrategyPolicy, default_weights, group_advantages, policy_gradient


def save_image(tensor, path):
    array = (tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def split_of(session_id, dev_share):
    return 'dev' if int(hashlib.sha1(session_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < dev_share else 'train'


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def read_jsonl_required(path):
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f'Required input is missing or empty: {path}')
    return rows


def select_sessions(records, per_stratum, seed):
    strata = {}
    for record in records:
        if record['kind'] == 't2i' and len(record.get('turns') or []) < 2:
            continue
        strata.setdefault((record['kind'], record['k']), []).append(record)
    order = lambda r: hashlib.sha1(f"{seed}:{r['id']}".encode()).hexdigest()
    picked = [sorted(group, key=order)[:per_stratum] for _, group in sorted(strata.items())]
    for (kind, k), group in zip(sorted(strata), picked):
        print(f'stratum {kind} K{k}: {len(group)} of {per_stratum}', flush=True)
    return [group[i] for i in range(max(map(len, picked), default=0)) for group in picked if i < len(group)]


def runtime(args):
    harness, controller, editor = load_runtime(args)
    harness.verification_budget, harness.generation_budget, harness.budget_per_turn = args.verification_budget, args.generation_budget, True
    return harness, controller, editor


def shard(items, args):
    return [item for index, item in enumerate(items) if index % args.num_shards == args.shard]


def bank(args):
    from redag.models.planner import dag_to_dict
    harness, controller, editor = runtime(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = out.parent / (out.stem + '_images')
    images.mkdir(exist_ok=True)
    done = {row['session_id'] for row in read_jsonl(out)}
    records = read_records(args.data)
    records = select_sessions(records, args.per_stratum, args.seed) if args.per_stratum else records[:args.limit or None]
    records = shard(records, args)
    with out.open('a') as stream:
        for record in records:
            if record['id'] in done:
                continue
            rows = []

            def capture(h, state, dag):
                if h.turn_index == 0 and h.draft_first_turn:
                    return
                state_id = f"{record['id']}:t{h.turn_index}"
                path = images / (hashlib.sha1(state_id.encode()).hexdigest()[:16] + '.png')
                save_image(state.image, path)
                rows.append(dict(state_id=state_id, session_id=record['id'], kind=record['kind'], k=record['k'], turn_index=h.turn_index,
                                 turn_count=h.turn_count, turn_request=h.turn_request, turn_nodes=sorted(h.turn_nodes), image=str(path),
                                 dag=dag_to_dict(dag), scores={n: float(v) for n, v in state.scores.items()}, protected=sorted(state.protected_set),
                                 split=split_of(record['id'], args.dev_share)))

            harness.on_turn_start = capture
            try:
                run_record(record, harness, controller, editor, args.max_steps, args.seed)
            finally:
                harness.on_turn_start = None
            for row in rows or [dict(session_id=record['id'], empty=True)]:
                stream.write(json.dumps(row) + '\n')
            stream.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def rollout(args):
    from redag.models.planner import dag_from_dict
    from redag.verify.grounder import load_image
    harness, controller, editor = runtime(args)
    policy = StrategyPolicy.load(args.policy, sample=not args.greedy)
    harness.strategy = policy
    states = [s for s in read_jsonl_required(args.states) if not s.get('empty') and s.get('split', 'train') == args.split]
    random.Random(f'{args.seed}:{args.epoch}').shuffle(states)
    states = shard(states[:args.states_per_epoch or None], args)
    branches = 1 if args.greedy else args.branches
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r['state_id'], r['branch']) for r in read_jsonl(out)}
    with out.open('a') as stream:
        for s in states:
            if all((s['state_id'], b) in done for b in range(branches)):
                continue
            image = load_image(s['image'])
            dag = dag_from_dict(s['dag'])
            if harness.grounder is not None:
                dag = harness.grounder.ground(image, dag)
            state = CanvasState(image, {n: float(s['scores'].get(n, 0.0)) for n in dag.nodes}, set(s['protected']) & set(dag.nodes), 0)
            grounding, start = harness.grounding_state(), consistent_count(dag, state.scores, dag.nodes)
            seed = args.seed + int(hashlib.sha1(s['state_id'].encode()).hexdigest()[:6], 16) % 100000
            for b in range(branches):
                if (s['state_id'], b) in done:
                    continue
                harness.restore_grounding(grounding)
                harness.turn_request, harness.turn_nodes = s['turn_request'], set(s['turn_nodes']) & set(dag.nodes)
                harness.turn_index, harness.turn_count, harness.draft_first_turn = s['turn_index'], s['turn_count'], s['kind'] == 't2i'
                harness.global_failures, harness.local_failures, harness.unknown_nodes, harness.history = {}, {}, set(), []
                policy.log, policy.rng = [], random.Random(f"{args.seed}:{args.epoch}:{s['state_id']}:{b}")
                renders = harness.generation_counter
                end_state, end_dag, steps = harness.run_session(state, deepcopy(dag), controller, editor, args.max_steps, seed)
                used = harness.generation_counter - renders
                gain = consistent_count(end_dag, end_state.scores, end_dag.nodes) - start
                stream.write(json.dumps(dict(state_id=s['state_id'], split=args.split, epoch=args.epoch, branch=b, gain=gain, generations=used,
                                             reward=gain - args.generation_cost * used, accepted=sum(bool(x['accepted']) for x in steps),
                                             routes=[x.get('route') for x in steps], decisions=policy.log)) + '\n')
                stream.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def summarize(rollouts):
    if not rollouts:
        return {}
    rewards = [r['reward'] for r in rollouts]
    mean = sum(rewards) / len(rewards)
    sd = math.sqrt(sum((x - mean) ** 2 for x in rewards) / max(1, len(rewards) - 1))
    decisions = [d for r in rollouts for d in r['decisions']]
    out = dict(rollouts=len(rollouts), states=len({r['state_id'] for r in rollouts}), reward=round(mean, 4),
               reward_ci95=round(1.96 * sd / math.sqrt(len(rewards)), 4), gain=round(sum(r['gain'] for r in rollouts) / len(rollouts), 4),
               generations=round(sum(r['generations'] for r in rollouts) / len(rollouts), 3))
    for kind in ('route', 'accept', 'retry', 'turn'):
        items = [d for d in decisions if d['kind'] == kind]
        if items:
            out[kind + '_decisions'] = len(items)
            out[kind + '_rate'] = round(sum(d['action'] for d in items) / len(items), 4)
            scored = [(d['action'], r['advantage']) for r in rollouts if 'advantage' in r for d in r['decisions'] if d['kind'] == kind]
            for action in (0, 1):
                values = [a for x, a in scored if x == action]
                if values:
                    out[f'{kind}_{action}_advantage'] = round(sum(values) / len(values), 4)
    return out


def learn(args):
    rollouts = group_advantages([r for path in args.rollouts for r in read_jsonl_required(path)])
    if not any(r['advantage'] for r in rollouts):
        raise SystemExit('no_learning_signal: every rollout group has identical rewards')
    policy = StrategyPolicy.load(args.policy)
    prior = json.loads(Path(args.prior).read_text())['weights'] if args.prior else default_weights()
    weights = policy_gradient(policy.weights, rollouts, lr=args.lr, prior=prior, l2=args.l2)
    stats = summarize(rollouts)
    StrategyPolicy(weights).save(args.out, stats=stats, parent=str(args.policy))
    print(json.dumps(stats), flush=True)


def report(args):
    for path in args.rollouts:
        print(Path(path).name, json.dumps(summarize(group_advantages(read_jsonl(path)))), flush=True)


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.add_argument('command', choices=('init', 'bank', 'rollout', 'learn', 'report'))
    parser.add_argument('--data', default='data/final/train_sessions.jsonl')
    parser.add_argument('--states', default='runs/strategy/turn_states.jsonl')
    parser.add_argument('--out')
    parser.add_argument('--policy')
    parser.add_argument('--prior')
    parser.add_argument('--rollouts', nargs='+', default=[])
    parser.add_argument('--split', choices=('train', 'dev'), default='train')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--branches', type=int, default=6)
    parser.add_argument('--states_per_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--generation_cost', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0.05)
    parser.add_argument('--dev_share', type=float, default=0.2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--per_stratum', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=3)
    parser.add_argument('--generation_budget', type=int, default=3)
    parser.add_argument('--verification_budget', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'init':
        StrategyPolicy().save(args.out)
    elif args.command == 'bank':
        bank(args)
    elif args.command == 'rollout':
        rollout(args)
    elif args.command == 'learn':
        learn(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
