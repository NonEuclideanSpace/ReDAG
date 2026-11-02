import json
import math
import random
from pathlib import Path

NODE_TYPES = ('entity', 'attribute', 'relation', 'background', 'count', 'prohibition')
ROUTE_CLASSES = ('local', 'relation_local', 'relation_wide', 'background', 'global', 'entity_global', 'count', 'prohibition')
GLOBAL_DEFAULT = ('background', 'global', 'relation_wide', 'count', 'prohibition')
ROUTE_FEATURES = ('bias',) + tuple('type=' + t for t in NODE_TYPES) + tuple('class=' + c for c in ROUTE_CLASSES) + (
    'global_failures', 'local_failures', 'target_score', 'last_turn', 'budget_left')
ACCEPT_FEATURES = ('bias', 'drops', 'drops_this_turn', 'drops_earlier_turns', 'drop_descendants', 'repairable_drops', 'single_repairable_drop',
                   'targets_hit', 'mean_gain', 'route_turn', 'last_turn', 'budget_left')
RETRY_FEATURES = ('bias', 'undrawn_share', 'damaged', 'budget_left', 'last_turn', 'attempts')
SCHEMA = dict(route=ROUTE_FEATURES, accept=ACCEPT_FEATURES, retry=RETRY_FEATURES)


def default_weights():
    route = {name: 0.0 for name in ROUTE_FEATURES}
    route['bias'] = -2.0
    for cls in GLOBAL_DEFAULT:
        route['class=' + cls] = 4.0
    route['global_failures'] = -12.0
    accept = {name: 0.0 for name in ACCEPT_FEATURES}
    accept.update(bias=-2.0, drops=-1.5, drop_descendants=-1.0, single_repairable_drop=2.0)
    retry = {name: 0.0 for name in RETRY_FEATURES}
    retry.update(bias=1.0)
    return {'route': [route[n] for n in ROUTE_FEATURES], 'accept': [accept[n] for n in ACCEPT_FEATURES], 'retry': [retry[n] for n in RETRY_FEATURES]}


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class StrategyPolicy:
    def __init__(self, weights=None, sample=True, seed=0):
        weights = weights or default_weights()
        if set(weights) != set(SCHEMA) or any(len(weights[kind]) != len(names) for kind, names in SCHEMA.items()):
            raise ValueError('Strategy weights do not match the feature schema')
        self.weights = {kind: [float(x) for x in weights[kind]] for kind in SCHEMA}
        if not all(math.isfinite(x) for w in self.weights.values() for x in w):
            raise ValueError('Strategy weights must be finite')
        self.sample, self.rng, self.log = sample, random.Random(seed), []

    @classmethod
    def load(cls, path, sample=False, seed=0):
        data = json.loads(Path(path).read_text())
        if {kind: list(data.get(kind + '_features', [])) for kind in SCHEMA} != {kind: list(names) for kind, names in SCHEMA.items()}:
            raise ValueError(f'{path} was saved with another feature schema')
        return cls(data['weights'], sample=sample, seed=seed)

    def save(self, path, **extra):
        Path(path).write_text(json.dumps(dict(weights=self.weights, **{kind + '_features': names for kind, names in SCHEMA.items()}, **extra), indent=1))

    def probability(self, kind, features):
        return sigmoid(sum(w * f for w, f in zip(self.weights[kind], features)))

    def decide(self, kind, features):
        p = self.probability(kind, features)
        action = self.rng.random() < p if self.sample else p >= 0.5
        self.log.append(dict(kind=kind, features=[round(float(f), 4) for f in features], action=int(action), p=round(p, 6)))
        return bool(action)


def policy_gradient(weights, rollouts, lr=0.5, prior=None, l2=0.05, clip=16.0):
    grads = {kind: [0.0] * len(weights[kind]) for kind in weights}
    count = 0
    for rollout in rollouts:
        advantage = rollout['advantage']
        if advantage == 0:
            continue
        count += 1
        for decision in rollout['decisions']:
            g = grads[decision['kind']]
            for i, f in enumerate(decision['features']):
                g[i] += advantage * (decision['action'] - decision['p']) * f
    prior = prior or default_weights()
    updated = {}
    for kind, w in weights.items():
        scale = 1.0 / max(1, count)
        updated[kind] = [max(-clip, min(clip, wi + lr * (gi * scale - l2 * (wi - pi)))) for wi, gi, pi in zip(w, grads[kind], prior[kind])]
    return updated


def group_advantages(rollouts):
    groups = {}
    for rollout in rollouts:
        groups.setdefault(rollout['state_id'], []).append(rollout)
    for items in groups.values():
        rewards = [r['reward'] for r in items]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
        for r in items:
            r['advantage'] = 0.0 if std < 1e-6 else (r['reward'] - mean) / (std + 1e-6)
    return rollouts
