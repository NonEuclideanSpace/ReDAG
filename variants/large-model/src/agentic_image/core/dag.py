from __future__ import annotations

import copy
import time
import uuid
from collections import deque
from typing import Any, Literal

from pydantic import Field

from .artifacts import canonical_json, content_id
from .types import ContractError, FrozenModel, PatchOperation


MUTABLE_KINDS = frozenset({"plan", "requirement", "entity", "region", "binding"})
FINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})


def new_id(kind: str) -> str:
    return f"{kind}:{uuid.uuid4().hex}"


class InputSlot(FrozenModel):
    slot_name: str
    index: int = 0
    artifact_value_id: str
    producer_version_id: str | None = None


class NodeSpec(FrozenModel):
    node_version_id: str
    logical_node_id: str
    episode_id: str
    branch_id: str
    kind: str
    schema_version: str = "1"
    operation: str
    typed_parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: tuple[InputSlot, ...] = ()
    data_parents: tuple[str, ...] = ()
    observation_parents: tuple[str, ...] = ()
    request_span_ids: tuple[str, ...] = ()
    plan_requirement_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    frame_ids: tuple[str, ...] = ()
    version_vector: dict[str, str] = Field(default_factory=dict)
    rng_key: int | None = None


class NodeEvent(FrozenModel):
    event_seq: int
    node_version_id: str
    event_type: Literal[
        "CREATED", "READY", "RUNNING", "SUCCEEDED", "FAILED", "CACHE_HIT", "CANCELLED", "RETRY"
    ]
    timestamp: float
    attempt: int = 0
    output_artifact_ids: tuple[str, ...] = ()
    error_code: str | None = None
    evidence_ids: tuple[str, ...] = ()


class BranchManifest(FrozenModel):
    branch_id: str
    revision_id: str
    parent_revision_id: str | None = None
    bindings: dict[str, str] = Field(default_factory=dict)
    selected_image_id: str | None = None
    selected_plan_revision: str | None = None
    active_control_set_id: str | None = None

    @property
    def signature(self) -> str:
        return content_id("manifest", canonical_json(self))


class TypedDag:
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self._nodes: dict[str, NodeSpec] = {}
        self._events: list[NodeEvent] = []
        self._states: dict[str, str] = {}
        self._attempts: dict[str, int] = {}
        self._revisions: dict[str, BranchManifest] = {}

    @property
    def nodes(self) -> dict[str, NodeSpec]:
        return {key: value.model_copy(deep=True) for key, value in self._nodes.items()}

    @property
    def events(self) -> tuple[NodeEvent, ...]:
        return tuple(self._events)

    def node(self, version_id: str) -> NodeSpec:
        if version_id not in self._nodes:
            raise ContractError("UNKNOWN_NODE", f"Unknown node version {version_id}")
        return self._nodes[version_id].model_copy(deep=True)

    def state(self, version_id: str) -> str:
        return self._states[version_id]

    def revision(self, revision_id: str) -> BranchManifest:
        if revision_id not in self._revisions:
            raise ContractError("UNKNOWN_REVISION", f"Unknown revision {revision_id}")
        return self._revisions[revision_id].model_copy(deep=True)

    def save_revision(self, manifest: BranchManifest) -> BranchManifest:
        if (
            manifest.revision_id in self._revisions
            and self._revisions[manifest.revision_id] != manifest
        ):
            raise ValueError("A revision cannot be overwritten")
        for logical_id, version_id in manifest.bindings.items():
            node = self.node(version_id)
            if node.logical_node_id != logical_id:
                raise ValueError("Manifest binding does not match the node identity")
        self._revisions[manifest.revision_id] = manifest.model_copy(deep=True)
        return manifest.model_copy(deep=True)

    def empty_manifest(self, branch_id: str) -> BranchManifest:
        return self.save_revision(
            BranchManifest(branch_id=branch_id, revision_id=new_id("revision"))
        )

    def append(self, node: NodeSpec) -> NodeSpec:
        if node.node_version_id in self._nodes:
            raise ValueError("Node versions are immutable")
        if node.episode_id != self.episode_id:
            raise ValueError("Node belongs to a different episode")
        parents = set(node.data_parents) | set(node.observation_parents)
        if node.node_version_id in parents:
            raise ContractError("DEPENDENCY_CYCLE", "A node cannot depend on itself")
        if not parents <= self._nodes.keys():
            raise ContractError(
                "UNKNOWN_PARENT", "All dependencies must refer to existing versions"
            )
        if any(self._states[parent] != "SUCCEEDED" for parent in node.observation_parents):
            raise ContractError(
                "UNFINISHED_OBSERVATION", "A decision can only read completed evidence"
            )
        canonical_json(node)
        self._nodes[node.node_version_id] = node.model_copy(deep=True)
        self._attempts[node.node_version_id] = 0
        self._record(node.node_version_id, "CREATED")
        return node.model_copy(deep=True)

    def _record(self, node_id: str, event_type: str, **kwargs):
        event = NodeEvent(
            event_seq=len(self._events),
            node_version_id=node_id,
            event_type=event_type,
            timestamp=time.time(),
            attempt=self._attempts[node_id],
            **kwargs,
        )
        self._events.append(event)
        if event_type != "RETRY":
            self._states[node_id] = event_type

    def transition(self, node_id: str, event_type: str, **kwargs):
        state = self._states[node_id]
        allowed = {
            "CREATED": {"READY", "CANCELLED"},
            "READY": {"RUNNING", "CACHE_HIT", "CANCELLED"},
            "RUNNING": {"SUCCEEDED", "FAILED", "CANCELLED"},
            "CACHE_HIT": {"SUCCEEDED"},
        }
        if event_type not in allowed.get(state, set()):
            raise ValueError(f"Invalid node transition {state} -> {event_type}")
        if event_type == "READY":
            node = self._nodes[node_id]
            if any(self._states[parent] != "SUCCEEDED" for parent in node.data_parents):
                raise ValueError("Data dependencies are not ready")
        self._record(node_id, event_type, **kwargs)

    def complete(self, node: NodeSpec, output_artifact_ids: tuple[str, ...] = ()) -> NodeSpec:
        self.append(node)
        self.transition(node.node_version_id, "READY")
        self.transition(node.node_version_id, "RUNNING")
        self.transition(node.node_version_id, "SUCCEEDED", output_artifact_ids=output_artifact_ids)
        return node

    def retry(self, node_id: str):
        if self._states[node_id] not in {"FAILED", "CANCELLED"}:
            raise ValueError("Only failed or cancelled jobs can retry")
        self._attempts[node_id] += 1
        self._record(node_id, "RETRY")
        self._record(node_id, "READY")

    def ancestors(self, node_ids: set[str], include_self: bool = False) -> set[str]:
        result = set(node_ids) if include_self else set()
        queue = list(node_ids)
        while queue:
            node = self.node(queue.pop())
            for parent in (*node.data_parents, *node.observation_parents):
                if parent not in result:
                    result.add(parent)
                    queue.append(parent)
        return result

    def descendants(self, node_ids: set[str], include_self: bool = False) -> set[str]:
        children: dict[str, set[str]] = {node_id: set() for node_id in self._nodes}
        for node in self._nodes.values():
            for parent in (*node.data_parents, *node.observation_parents):
                children[parent].add(node.node_version_id)
        result = set(node_ids) if include_self else set()
        queue = list(node_ids)
        while queue:
            for child in children[queue.pop()]:
                if child not in result:
                    result.add(child)
                    queue.append(child)
        return result

    def recompute_closure(self, changed: set[str], outputs: set[str]) -> set[str]:
        return self.descendants(changed, include_self=True) & self.ancestors(
            outputs, include_self=True
        )

    def suspect_nodes(
        self,
        evidence_version_id: str,
        entity_ids: set[str] | None = None,
        span_ids: set[str] | None = None,
        frame_ids: set[str] | None = None,
    ) -> tuple[str, ...]:
        candidates = [
            self._nodes[node_id]
            for node_id in self.ancestors({evidence_version_id})
            if self._nodes[node_id].kind in MUTABLE_KINDS
        ]
        filters = [
            (entity_ids, "entity_ids"),
            (span_ids, "request_span_ids"),
            (frame_ids, "frame_ids"),
        ]
        for values, field in filters:
            if values:
                narrowed = [node for node in candidates if values & set(getattr(node, field))]
                if narrowed:
                    candidates = narrowed
        return tuple(node.node_version_id for node in candidates)

    def topological(self, node_ids: set[str] | None = None) -> tuple[str, ...]:
        selected = set(self._nodes) if node_ids is None else node_ids
        incoming = {node_id: 0 for node_id in selected}
        children = {node_id: [] for node_id in selected}
        for node_id in selected:
            node = self.node(node_id)
            for parent in set((*node.data_parents, *node.observation_parents)) & selected:
                incoming[node_id] += 1
                children[parent].append(node_id)
        queue = deque(sorted(node_id for node_id, count in incoming.items() if count == 0))
        result = []
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            for child in children[node_id]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    queue.append(child)
        if len(result) != len(selected):
            raise ContractError("DEPENDENCY_CYCLE", "Execution graph contains a cycle")
        return tuple(result)

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "nodes": [node.model_dump(mode="json") for node in self._nodes.values()],
            "events": [event.model_dump(mode="json") for event in self._events],
            "revisions": [
                revision.model_dump(mode="json") for revision in self._revisions.values()
            ],
        }

    @classmethod
    def restore(cls, value: dict[str, Any]) -> TypedDag:
        graph = cls(value["episode_id"])
        graph._nodes = {
            node["node_version_id"]: NodeSpec.model_validate(node) for node in value["nodes"]
        }
        graph._events = [NodeEvent.model_validate(event) for event in value["events"]]
        for event in graph._events:
            if event.event_type != "RETRY":
                graph._states[event.node_version_id] = event.event_type
            graph._attempts[event.node_version_id] = event.attempt
        graph._revisions = {
            item["revision_id"]: BranchManifest.model_validate(item) for item in value["revisions"]
        }
        graph.topological()
        return graph


def apply_field_patch(value: dict[str, Any], operation: PatchOperation) -> dict[str, Any]:
    output = copy.deepcopy(value)
    path = operation.field_path
    if not path:
        if operation.op == "retire":
            raise ContractError(
                "INVALID_PATCH", "Whole-node retirement must be handled by the branch manager"
            )
        if not isinstance(operation.typed_value_or_artifact_id, dict):
            raise ContractError("INVALID_PATCH", "Whole-node replacement requires an object")
        return copy.deepcopy(operation.typed_value_or_artifact_id)
    current: Any = output
    for key in path[:-1]:
        if (
            isinstance(current, dict)
            and key in current
            or isinstance(current, list)
            and isinstance(key, int)
            and 0 <= key < len(current)
        ):
            current = current[key]
        else:
            raise ContractError("INVALID_PATCH_PATH", f"Unknown patch path {path}")
    key = path[-1]
    if isinstance(current, dict):
        exists = key in current
        if operation.op == "add" and exists:
            raise ContractError("INVALID_PATCH", "An added field must not exist")
        if operation.op in {"replace", "retire"} and not exists:
            raise ContractError("INVALID_PATCH", "A replaced or retired field must exist")
        if operation.op == "retire":
            del current[key]
        else:
            current[key] = copy.deepcopy(operation.typed_value_or_artifact_id)
    elif isinstance(current, list) and isinstance(key, int):
        if operation.op == "add" and key == len(current):
            current.append(copy.deepcopy(operation.typed_value_or_artifact_id))
        elif 0 <= key < len(current):
            if operation.op == "retire":
                current.pop(key)
            elif operation.op == "replace":
                current[key] = copy.deepcopy(operation.typed_value_or_artifact_id)
            else:
                raise ContractError("INVALID_PATCH", "List additions must append")
        else:
            raise ContractError("INVALID_PATCH_PATH", "List index is out of range")
    else:
        raise ContractError("INVALID_PATCH_PATH", "Patch path must address an object or list")
    return output
