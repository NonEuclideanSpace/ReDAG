"""Atomic resume checkpoints for the long training loops (SFT, alternating RL).

save() writes <folder>/resume through a temporary directory and renames, so a crash or a stopped instance at any
moment leaves the previous or the new checkpoint intact, never a half-written one.  load() returns None when there
is nothing to resume.  Only LoRA tensors, optimizer state, RNG states and small JSON progress are stored; the frozen
base weights are reloaded from the model folders as usual.
"""
import json
import random
import shutil
import time
from pathlib import Path

import torch

NAME = 'resume'
_ORDER = (NAME, NAME + '.tmp', NAME + '.old')


def rng_state():
    state = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda'])


def save(folder, state, tensors=None, optimizer=None, objects=None):
    """state: JSON-serialisable progress; tensors: {name: {param_name: tensor}}; objects: other picklable values."""
    folder = Path(folder)
    final, tmp, old = (folder / name for name in _ORDER)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    payload = {'tensors': {k: {n: t.detach().cpu() for n, t in v.items()} for k, v in (tensors or {}).items()},
               'optimizer': optimizer.state_dict() if optimizer is not None else None,
               'objects': objects or {}, 'rng': rng_state()}
    torch.save(payload, tmp / 'payload.pt')
    (tmp / 'state.json').write_text(json.dumps(dict(state, saved_at=time.time()), indent=2), encoding='utf-8')
    (tmp / 'COMPLETE').touch()
    if final.exists():
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def load(folder):
    """(state, payload) of the newest complete checkpoint in folder, or None."""
    folder = Path(folder)
    for name in _ORDER:
        path = folder / name
        if (path / 'COMPLETE').exists():
            state = json.loads((path / 'state.json').read_text(encoding='utf-8'))
            payload = torch.load(path / 'payload.pt', map_location='cpu', weights_only=False)
            return state, payload
    return None


def clear(folder):
    for name in _ORDER:
        shutil.rmtree(Path(folder) / name, ignore_errors=True)


def keep_jsonl_rows(path, keep):
    """Rewrite an append-only JSONL log keeping the parsable rows for which keep(row) is true (atomic)."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line cut off by a crash
        if isinstance(row, dict) and keep(row):
            rows.append(row)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    tmp.replace(path)
    return rows


class Interrupted(RuntimeError):
    """Raised by the interrupt_after test hooks right after a checkpoint, to simulate a crash."""
