import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any, Callable
from uuid import uuid4

import torch

from .optim import clip_policy_gradients, finite_gradients


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def parameter_change(before, after):
    maximum = 0.0
    changed = 0
    squared = 0.0
    pending = [(before, after)]
    while pending:
        old, new = pending.pop()
        if isinstance(old, dict):
            if old.keys() != new.keys():
                raise ValueError("A committed update changed the model state structure")
            pending.extend((old[key], new[key]) for key in old)
        elif isinstance(old, (list, tuple)):
            if len(old) != len(new):
                raise ValueError("A committed update changed the model state length")
            pending.extend(zip(old, new))
        elif isinstance(old, torch.Tensor):
            if old.shape != new.shape:
                raise ValueError("A committed update changed a parameter shape")
            delta = new.float() - old.float()
            if not torch.isfinite(delta).all():
                raise FloatingPointError("A committed model contains a nonfinite parameter change")
            value = float(delta.abs().max()) if delta.numel() else 0.0
            maximum = max(maximum, value)
            changed += int(value > 0)
            squared += float(delta.square().sum())
    return {"max_abs": maximum, "l2": squared**0.5, "changed_tensors": changed}


def rng_state():
    import numpy as np

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype("int64")),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
    }


def restore_rng(state):
    import numpy as np

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["keys"].numpy().astype("uint32"),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class TrainingComponent:
    name: str
    parameters: list[torch.nn.Parameter]
    optimizer: torch.optim.Optimizer | None
    capture_model: Callable[[], dict[str, Any]]
    restore_model: Callable[[dict[str, Any]], None]
    scheduler: Any = None
    active: bool = True

    def capture(self):
        return {
            "model": cpu_copy(self.capture_model()),
            "optimizer": cpu_copy(self.optimizer.state_dict())
            if self.optimizer is not None
            else None,
            "scheduler": cpu_copy(self.scheduler.state_dict())
            if self.scheduler is not None
            else None,
            "active": self.active,
        }

    def restore(self, state):
        self.restore_model(state["model"])
        if self.optimizer is not None and state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.active = state["active"]


class JointTransaction:
    def __init__(self, directory, components):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.components = {component.name: component for component in components}
        if len(self.components) != len(components):
            raise ValueError("Training component names must be unique")
        self.transaction_path = None
        self.before = None
        self.manifest = None

    def begin(self, behavior_version, data_cursor, batch_id=None, metadata=None):
        if self.before is not None:
            raise RuntimeError("A joint transaction is already active")
        transaction_id = uuid4().hex
        self.transaction_path = self.directory / transaction_id
        self.transaction_path.mkdir()
        self.manifest = {
            "transaction_id": transaction_id,
            "behavior_version": behavior_version,
            "batch_id": batch_id,
            "status": "COLLECTING",
            "created_ns": time_ns(),
            "metadata": metadata or {},
        }
        self.before = {
            "components": {
                name: component.capture() for name, component in self.components.items()
            },
            "rng": rng_state(),
            "data_cursor": cpu_copy(data_cursor),
            "behavior_version": behavior_version,
        }
        atomic_torch_save(self.transaction_path / "before.pt", self.before)
        atomic_json(self.transaction_path / "transaction.json", self.manifest)
        return transaction_id

    def set_batch(self, batch_id, data_cursor=None):
        self._require_active()
        self.manifest["batch_id"] = batch_id
        if data_cursor is not None:
            self.manifest["next_data_cursor"] = data_cursor
        self.manifest["status"] = "BATCH_COMPLETE"
        atomic_json(self.transaction_path / "transaction.json", self.manifest)

    def _require_active(self):
        if self.before is None:
            raise RuntimeError("No joint transaction is active")

    def restore_before(self):
        self._require_active()
        for name, component in self.components.items():
            component.restore(self.before["components"][name])
        restore_rng(self.before["rng"])
        return self.before["data_cursor"]

    def abort(self, reason):
        self.restore_before()
        self.manifest.update(status="ABORTED", reason=str(reason))
        atomic_json(self.transaction_path / "transaction.json", self.manifest)
        self.before = None

    def step_and_commit(
        self,
        new_version,
        probe=None,
        limits=None,
        max_retries=2,
        max_gradient_norm=1.0,
        data_cursor=None,
    ):
        self._require_active()
        limits = limits or {"actor_kl": 0.1, "renderer_kl": 0.01}
        active_flags = {name: component.active for name, component in self.components.items()}
        active_names = [
            name
            for name, component in self.components.items()
            if component.active and component.optimizer is not None
        ]
        if any(not finite_gradients(self.components[name].parameters) for name in active_names):
            self.abort("nonfinite_gradient")
            raise FloatingPointError(
                "A non-finite component gradient prevented the complete joint update"
            )
        metrics = {
            name: clip_policy_gradients(self.components[name].parameters, max_gradient_norm)
            for name in active_names
        }
        self.manifest["status"] = "GRADIENTS_READY"
        atomic_json(self.transaction_path / "transaction.json", self.manifest)
        last_probe = {}
        try:
            for attempt in range(max_retries + 1):
                if attempt:
                    self.restore_before()
                    for name, component in self.components.items():
                        component.active = active_flags[name]
                    for name in active_names:
                        component = self.components[name]
                        factor = 0.5**attempt
                        for group in component.optimizer.param_groups:
                            group["lr"] *= factor
                        if component.scheduler is not None:
                            component.scheduler.base_lrs = [
                                value * factor for value in component.scheduler.base_lrs
                            ]
                for name in active_names:
                    self.components[name].optimizer.step()
                last_probe = probe() if probe is not None else {}
                if not all(
                    torch.isfinite(torch.tensor(float(value))) for value in last_probe.values()
                ):
                    acceptable = False
                else:
                    acceptable = all(
                        last_probe.get(name, 0.0) <= bound for name, bound in limits.items()
                    )
                if acceptable:
                    break
                if attempt == max_retries:
                    raise RuntimeError(f"Update probe remained outside its bounds: {last_probe}")
            for name in active_names:
                scheduler = self.components[name].scheduler
                if scheduler is not None:
                    scheduler.step()
            files = {}
            changes = {}
            self.manifest["status"] = "PREPARED"
            atomic_json(self.transaction_path / "transaction.json", self.manifest)
            for name, component in self.components.items():
                filename = f"{name}.pt"
                captured = component.capture()
                changes[name] = parameter_change(
                    self.before["components"][name]["model"], captured["model"]
                )
                atomic_torch_save(self.transaction_path / filename, captured)
                files[name] = filename
            state = {
                "rng": rng_state(),
                "data_cursor": cpu_copy(
                    data_cursor
                    if data_cursor is not None
                    else self.manifest.get("next_data_cursor", self.before["data_cursor"])
                ),
                "behavior_version": new_version,
            }
            atomic_torch_save(self.transaction_path / "training_state.pt", state)
            committed = dict(
                self.manifest,
                status="COMMITTED",
                new_version=new_version,
                components=files,
                optimizer_steps=active_names,
                attempt=attempt,
                probes=last_probe,
                gradients=metrics,
                parameter_changes=changes,
                committed_ns=time_ns(),
            )
            atomic_json(self.transaction_path / "COMMITTED.json", committed)
            self.before = None
            try:
                atomic_json(
                    self.directory / "LATEST.json",
                    {"transaction_id": self.manifest["transaction_id"], "version": new_version},
                )
            except OSError as error:
                committed["latest_hint_error"] = str(error)
            return committed
        except Exception as error:
            if self.before is not None:
                self.abort(error)
            raise

    def load_committed(self, transaction_id=None):
        if self.before is not None:
            raise RuntimeError("Abort the active transaction before restoring a checkpoint")
        if transaction_id is None:
            commits = []
            for marker in self.directory.glob("*/COMMITTED.json"):
                committed = json.loads(marker.read_text())
                commits.append(
                    (committed.get("committed_ns", marker.stat().st_mtime_ns), marker.parent.name)
                )
            if commits:
                transaction_id = max(commits)[1]
            else:
                interrupted = []
                for marker in self.directory.glob("*/transaction.json"):
                    manifest = json.loads(marker.read_text())
                    if (marker.parent / "before.pt").is_file():
                        interrupted.append(
                            (
                                manifest.get("created_ns", marker.stat().st_mtime_ns),
                                marker.parent,
                                manifest,
                            )
                        )
                if not interrupted:
                    raise FileNotFoundError(
                        "No committed or recoverable initial training state exists"
                    )
                _, path, manifest = max(interrupted, key=lambda item: (item[0], item[1].name))
                state = torch.load(path / "before.pt", map_location="cpu", weights_only=True)
                if set(state["components"]) != set(self.components):
                    raise ValueError("Interrupted state must contain every training component")
                for name, component in self.components.items():
                    component.restore(state["components"][name])
                restore_rng(state["rng"])
                return {key: value for key, value in state.items() if key != "components"} | {
                    "recovered_before_transaction": manifest["transaction_id"],
                    "recovery_status": manifest["status"],
                }
        path = self.directory / transaction_id
        manifest = json.loads((path / "COMMITTED.json").read_text())
        if manifest["status"] != "COMMITTED" or set(manifest["components"]) != set(self.components):
            raise ValueError(
                "A checkpoint must contain every training component and a committed manifest"
            )
        for name, filename in manifest["components"].items():
            self.components[name].restore(
                torch.load(path / filename, map_location="cpu", weights_only=True)
            )
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        restore_rng(state["rng"])
        return state
