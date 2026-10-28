import hashlib
import json
import random
from collections import defaultdict


def stream_seed(*parts):
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class PhaseBalancedSampler:
    def __init__(self, records, quotas=(5, 7, 5, 3), seed=0, max_per_state=2):
        self.records = list(records)
        self.quotas = tuple(quotas)
        self.rng = random.Random(seed)
        self.max_per_state = max_per_state
        self.pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, record in enumerate(self.records):
            phase = record["phase"]
            self.pools[phase][record["family_id"]][record["state_id"]].append(index)
        for phase, quota in enumerate(self.quotas):
            if quota and not self.pools[phase]:
                raise ValueError(f"SFT phase {phase} has no verified actions")

    def next_batch(self):
        counts = defaultdict(int)
        selected = []
        remaining = list(self.quotas)
        while sum(remaining):
            phase_options = []
            for phase, quota in enumerate(remaining):
                if not quota:
                    continue
                available = {}
                for family, states in self.pools[phase].items():
                    valid = {
                        state: indices
                        for state, indices in states.items()
                        if counts[(family, state)] < self.max_per_state
                    }
                    if valid:
                        available[family] = valid
                if not available:
                    raise ValueError(
                        "Not enough distinct state bundles to fill the SFT phase quotas"
                    )
                phase_options.append(
                    (sum(len(states) for states in available.values()), phase, available)
                )
            _, phase, available = min(
                phase_options, key=lambda item: (item[0], -remaining[item[1]])
            )
            family = self.rng.choice(sorted(available))
            state = self.rng.choice(sorted(available[family]))
            selected.append(self.rng.choice(available[family][state]))
            counts[(family, state)] += 1
            remaining[phase] -= 1
        self.rng.shuffle(selected)
        return [self.records[index] for index in selected]

    def state_dict(self):
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
