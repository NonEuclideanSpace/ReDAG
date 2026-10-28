def effective_components(committed):
    result = []
    for name in committed.get("optimizer_steps", ()):
        change = committed.get("parameter_changes", {}).get(name, {})
        gradient = committed.get("gradients", {}).get(name, {})
        if (
            gradient.get("gradient_norm", 0) > 0
            and change.get("changed_tensors", 0) > 0
            and change.get("l2", 0) > 0
            and change.get("max_abs", 0) > 0
        ):
            result.append(name)
    return result


def rl_round_limits(rl, mode, rounds=None):
    nominal = rounds if rounds is not None else rl['agent_rounds' if mode == 'agent' else 'joint_rounds']
    target = rl.get(f'{mode}_effective_updates')
    enabled = target is not None
    maximum = rounds if rounds is not None else rl.get(f'{mode}_max_rounds', nominal)
    if not enabled and f'{mode}_max_rounds' in rl:
        raise ValueError('An RL maximum round override requires an explicit effective update target')
    target = target if enabled else nominal
    for name, value in (('nominal rounds', nominal), ('effective updates', target), ('maximum rounds', maximum)):
        if type(value) is not int or value < 1:
            raise ValueError(f'RL {name} must be a positive integer')
    if maximum < target:
        raise ValueError('The RL maximum round budget cannot be smaller than its effective update target')
    return {'nominal_rounds': nominal, 'effective_updates': target, 'max_rounds': maximum, 'enabled': enabled}


def required_rl_components(mode):
    return {'agent': {'actor'}, 'renderer': {'renderer'}, 'joint': {'actor', 'renderer'}}[mode]


def accumulate_effective_updates(counts, mode, committed):
    active = set(effective_components(committed))
    for name in ('actor', 'renderer'):
        counts[name] += int(name in active)
    counts['required_components'] += int(required_rl_components(mode) <= active)
    return sorted(active)


def committed_round_metrics(committed, checkpoint, counts, fallback=None):
    from pathlib import Path

    base = committed.get('metadata', {}).get('rl_round_metrics') or fallback
    if base is None:
        raise ValueError('A committed RL round lacks recoverable metric evidence')
    return {**base, 'gradients': committed['gradients'], 'probes': committed['probes'],
        'parameter_changes': committed['parameter_changes'], 'optimizer_steps': committed['optimizer_steps'],
        'effective_components': sorted(effective_components(committed)),
        'effective_update_counts': dict(counts), 'checkpoint': str(Path(checkpoint).resolve())}


def restore_rl_history(folder, mode, start, behavior):
    import json
    import math
    from pathlib import Path

    folder = Path(folder)
    if start and behavior != f'{mode}:{start}':
        raise ValueError('Restored RL behavior and round cursor disagree')
    commits = {}
    for marker in (folder / 'checkpoints').glob('*/COMMITTED.json'):
        committed = json.loads(marker.read_text())
        version = committed.get('new_version', '').split(':')
        if len(version) != 2 or version[0] != mode or not version[1].isdigit():
            raise ValueError('An RL commit has a different mode or malformed version')
        round_number = int(version[1])
        if (committed.get('status') != 'COMMITTED' or committed.get('metadata', {}).get('mode') != mode
                or committed.get('transaction_id') != marker.parent.name
                or committed.get('next_data_cursor', {}).get('round') != round_number
                or round_number in commits):
            raise ValueError('RL commit history is inconsistent or contains duplicate rounds')
        commits[round_number] = (marker.parent, committed)
    if sorted(commits) != list(range(1, start + 1)):
        raise ValueError('RL commits must cover every round through the restored cursor exactly once')
    path = folder / 'metrics.jsonl'
    lines = path.read_text().splitlines() if path.is_file() else []
    existing = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            existing.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise ValueError('RL metrics contain corruption before their final append')
    if [row.get('round') for row in existing] != list(range(1, len(existing) + 1)) or len(existing) > start:
        raise ValueError('RL metrics must be a contiguous prefix of the committed round history')
    counts = {'actor': 0, 'renderer': 0, 'required_components': 0}
    metrics = []
    zero_rounds = 0
    proof_keys = ('gradients', 'probes', 'parameter_changes', 'optimizer_steps', 'effective_components', 'effective_update_counts')
    for round_number in range(1, start + 1):
        checkpoint, committed = commits[round_number]
        accumulate_effective_updates(counts, mode, committed)
        prior = existing[round_number - 1] if round_number <= len(existing) else None
        row = committed_round_metrics(committed, checkpoint, counts, prior)
        if row.get('round') != round_number:
            raise ValueError('Stored RL metric evidence belongs to another round')
        if prior is not None:
            if Path(prior.get('checkpoint', '')).resolve() != checkpoint.resolve():
                raise ValueError('RL metric checkpoint differs from its actual commit')
            for key in proof_keys:
                if key in prior and prior[key] != row[key]:
                    raise ValueError(f'RL metric {key} differs from its actual commit')
            for key, value in committed.get('metadata', {}).get('rl_round_metrics', {}).items():
                if prior.get(key) != value:
                    raise ValueError('RL metrics differ from the evidence saved before committing')
        fraction = row.get('zero_advantage_fraction')
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('RL history lacks a valid zero-advantage measurement')
        zero_rounds = zero_rounds + 1 if fraction > 0.8 else 0
        metrics.append(row)
    if metrics != existing or len(lines) != len(existing):
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in metrics))
        temporary.replace(path)
    return counts, zero_rounds
