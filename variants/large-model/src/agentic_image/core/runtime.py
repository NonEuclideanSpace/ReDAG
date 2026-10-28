from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from pydantic import ValidationError

from .artifacts import ArtifactStore, canonical_json, content_id
from .dag import (
    BranchManifest,
    InputSlot,
    MUTABLE_KINDS,
    NodeSpec,
    TypedDag,
    apply_field_patch,
    new_id,
)
from .geometry import box_mask, expand_mask, mask_iou, validate_mask
from .protocol import (
    ApplyAction,
    DraftRepairAction,
    FinishAction,
    InspectAction,
    PlanAction,
    ReadNodesAction,
    RenderAction,
    SelectAction,
    parse_action,
)
from .types import (
    ContractError,
    Entity,
    ErrorRecord,
    FrozenModel,
    Observation,
    PatchOperation,
    PlanRequirement,
    PlanSpec,
    PublicTask,
    ReferenceBinding,
    RegionControl,
    RenderResult,
    ServiceFailure,
    StepResult,
)
from .view import AgentView, AgentViewBuilder


class Budget(FrozenModel):
    actions: int = 0
    renders: int = 0
    inspects: int = 0
    reads: int = 0
    invalid: int = 0


@dataclass(frozen=True)
class ImageState:
    image_id: str
    manifest: BranchManifest
    plan: PlanSpec
    bundle: Any = None
    render: RenderResult | None = None
    render_node_id: str | None = None


@dataclass(frozen=True)
class RepairCandidate:
    candidate_id: str
    source_image_id: str
    parent_revision: str
    parent_manifest_signature: str
    plan: PlanSpec
    scope: str
    patch_operations: tuple[PatchOperation, ...]
    changed_nodes: tuple[str, ...]
    recompute_nodes: tuple[str, ...]
    bundle: Any
    bundle_id: str
    source_for_render: str | None
    validation_result: str = "valid"

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_image_id": self.source_image_id,
            "parent_revision": self.parent_revision,
            "scope": self.scope,
            "changed_nodes": list(self.changed_nodes),
            "recompute_nodes": list(self.recompute_nodes),
            "bundle_id": self.bundle_id,
            "required_render_calls": 1,
            "validation_result": self.validation_result,
        }


@dataclass
class EpisodeState:
    dag: TypedDag
    manifest: BranchManifest
    budget: Budget = field(default_factory=Budget)
    plan: PlanSpec | None = None
    disclosed_ids: set[str] = field(default_factory=set)
    observations: dict[str, Observation] = field(default_factory=dict)
    observation_nodes: dict[str, str] = field(default_factory=dict)
    image_nodes: dict[str, str] = field(default_factory=dict)
    history: dict[str, ImageState] = field(default_factory=dict)
    candidates: dict[str, RepairCandidate] = field(default_factory=dict)
    errors: list[ErrorRecord] = field(default_factory=list)
    executed_renders: list[RenderResult] = field(default_factory=list)
    action_events: list[dict[str, Any]] = field(default_factory=list)
    cursor_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    public_aliases: dict[str, str] = field(default_factory=dict)
    last_view_node_ids: tuple[str, ...] = ()
    active_crop_id: str | None = None
    last_tool_result: dict[str, Any] | None = None
    last_tool_node_ids: tuple[str, ...] = ()
    checks_revealed: bool = False
    done: bool = False
    delivered_image_id: str | None = None
    protocol_failure: bool = False


class EpisodeEnvironment:
    def __init__(
        self,
        task: PublicTask,
        artifacts: ArtifactStore,
        compiler: Any,
        renderer: Any,
        observer: Any,
        episode_id: str | None = None,
        branch_id: str = "root",
        versions: dict[str, str] | None = None,
        token_counter: Callable[[str], int] | None = None,
        max_input_tokens: int = 6144,
    ):
        self.task = task
        self.artifacts = artifacts
        self.compiler = compiler
        self.renderer = renderer
        self.observer = observer
        self.episode_id = episode_id or new_id("episode")
        self.versions = versions or {}
        self.token_counter = token_counter
        self.view_builder = AgentViewBuilder(max_input_tokens, token_counter)
        dag = TypedDag(self.episode_id)
        request = NodeSpec(
            node_version_id=new_id("node"),
            logical_node_id="user_request",
            episode_id=self.episode_id,
            branch_id=branch_id,
            kind="request",
            operation="user_request",
            typed_parameters=task.user_request.model_dump(mode="json"),
            version_vector=self.versions,
        )
        dag.complete(request)
        manifest = BranchManifest(
            branch_id=branch_id,
            revision_id=new_id("revision"),
            bindings={"user_request": request.node_version_id},
        )
        dag.save_revision(manifest)
        self.state = EpisodeState(
            dag=dag, manifest=manifest, disclosed_ids={"user_request", request.node_version_id}
        )
        for image_id in (
            task.user_request.source_image_id,
            *task.user_request.appearance_image_ids,
        ):
            if image_id is not None and image_id not in self.state.image_nodes:
                ref = artifacts.get_ref(image_id)
                if ref.kind != "rgb":
                    raise ValueError("Public image inputs must be RGB artifacts")
                image_node = self._node(
                    image_id,
                    "image",
                    "public_image",
                    {"image_id": image_id},
                    data_parents=(request.node_version_id,),
                    inputs=(
                        InputSlot(
                            slot_name="image",
                            artifact_value_id=ref.value_id,
                            producer_version_id=ref.producer_id,
                        ),
                    ),
                )
                dag.complete(image_node, (image_id,))
                self.state.image_nodes[image_id] = image_node.node_version_id
                self.state.disclosed_ids.add(image_id)
        self.state.last_view_node_ids = (request.node_version_id, *self.state.image_nodes.values())

        def disclose_controls(value):
            if isinstance(value, dict):
                for item in value.values():
                    disclose_controls(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    disclose_controls(item)
            elif isinstance(value, str) and value.startswith("artifact:"):
                artifacts.get_ref(value)
                self.state.disclosed_ids.add(value)

        disclose_controls(task.user_request.controls)

    def _node(
        self,
        logical_id: str,
        kind: str,
        operation: str,
        parameters: dict[str, Any],
        data_parents: tuple[str, ...] = (),
        observation_parents: tuple[str, ...] = (),
        **kwargs,
    ) -> NodeSpec:
        return NodeSpec(
            node_version_id=new_id("node"),
            logical_node_id=logical_id,
            episode_id=self.episode_id,
            branch_id=self.state.manifest.branch_id,
            kind=kind,
            operation=operation,
            typed_parameters=parameters,
            data_parents=tuple(dict.fromkeys(data_parents)),
            observation_parents=tuple(dict.fromkeys(observation_parents)),
            version_vector=self.versions,
            **kwargs,
        )

    def view(self) -> AgentView:
        return self.view_builder.build(self)

    def public_id(self, identifier: str) -> str:
        if identifier in self.state.public_aliases:
            return self.state.public_aliases[identifier]
        if identifier in self.state.public_aliases.values():
            return identifier
        prefix = identifier.split(":", 1)[0]
        if identifier in self.state.image_nodes:
            kind = "image"
        elif identifier in self.state.observations:
            kind = "evidence"
        elif prefix == "artifact":
            kind = "image" if self.artifacts.get_ref(identifier).kind == "rgb" else "artifact"
        elif prefix in {
            "node",
            "revision",
            "candidate",
            "cursor",
            "frame",
            "bundle",
            "condition",
            "render",
        }:
            kind = {"cursor": "page", "bundle": "condition"}.get(prefix, prefix)
        else:
            return identifier
        count = sum(alias.startswith(kind + ":") for alias in self.state.public_aliases.values())
        alias = f"{kind}:{count + 1:04d}"
        self.state.public_aliases[identifier] = alias
        return alias

    def _translate(self, value, public: bool, key: str | None = None):
        text_fields = {
            "text",
            "global_text",
            "predicate",
            "instruction",
            "reason",
            "message",
            "summary",
            "name",
            "category",
            "raw_span",
            "entity_id",
            "requirement_id",
            "region_id",
            "binding_id",
            "entity_ids",
            "requirement_ids",
            "reference_ids",
            "depends_on",
            "dependencies",
            "target_logical_node_id",
        }
        if key in text_fields:
            return copy.deepcopy(value)
        if isinstance(value, dict):
            output = {}
            for name, item in value.items():
                if (
                    name == "typed_value_or_artifact_id"
                    and value.get("field_path")
                    and value["field_path"][-1] in text_fields
                ):
                    output[name] = copy.deepcopy(item)
                else:
                    output[name] = self._translate(item, public, name)
            return output
        if isinstance(value, (list, tuple)):
            return [self._translate(item, public, key) for item in value]
        if isinstance(value, str):
            if public:
                return self.public_id(value)
            reverse = {alias: internal for internal, alias in self.state.public_aliases.items()}
            return reverse.get(value, value)
        return value

    def public_action(self, action: str | dict) -> dict:
        value = json.loads(action) if isinstance(action, str) else action
        return self._translate(value, True)

    def resolve_action(self, action: str | dict) -> dict:
        parsed = parse_action(action)
        return self._translate(
            parsed.model_dump(mode="json", by_alias=True, exclude_unset=True), False
        )

    def fork(self, branch_id: str) -> EpisodeEnvironment:
        if branch_id == self.state.manifest.branch_id:
            raise ValueError("A fork needs a new branch ID")
        child = copy.copy(self)
        child.state = copy.copy(self.state)
        child.state.dag = TypedDag.restore(self.state.dag.snapshot())
        for name in (
            "disclosed_ids",
            "observations",
            "observation_nodes",
            "image_nodes",
            "history",
            "candidates",
            "errors",
            "executed_renders",
            "action_events",
            "cursor_records",
            "public_aliases",
        ):
            setattr(child.state, name, copy.copy(getattr(self.state, name)))
        child.state.cursor_records = {}
        child.state.candidates = {}
        manifest = self.state.manifest.model_copy(
            update={
                "branch_id": branch_id,
                "revision_id": new_id("revision"),
                "parent_revision_id": self.state.manifest.revision_id,
            },
            deep=True,
        )
        child.state.manifest = self.state.dag.save_revision(manifest)
        return child

    def _bump(self, name: str):
        self.state.budget = self.state.budget.model_copy(
            update={name: getattr(self.state.budget, name) + 1}
        )

    def _ensure_known(self, ids):
        unknown = {identifier for identifier in ids if identifier not in self.state.disclosed_ids}
        if unknown:
            raise ContractError(
                "UNKNOWN_ID",
                "Action references undisclosed IDs: "
                + ", ".join(sorted(str(identifier) for identifier in unknown)),
            )

    def _ensure_revision(self, expected: str):
        if expected != self.state.manifest.revision_id:
            raise ContractError("STALE_REVISION", "Action does not match the current revision")

    def _budget_available(self, name: str, limit: int):
        if getattr(self.state.budget, name) >= limit:
            raise ContractError(
                "BUDGET_EXHAUSTED", f"The {name} budget is exhausted", policy_invalid=False
            )

    def _service(self, call: Callable, *args, **kwargs):
        for attempt in range(3):
            try:
                return call(*args, **kwargs)
            except (TimeoutError, ConnectionError) as error:
                if attempt == 2:
                    raise ServiceFailure(f"Service failed after three attempts: {error}") from error

    def step(self, raw: str | dict, seed: int = 0) -> StepResult:
        previous = copy.copy(self.state)
        for name in (
            "disclosed_ids",
            "observations",
            "observation_nodes",
            "image_nodes",
            "history",
            "candidates",
            "errors",
            "executed_renders",
            "action_events",
            "cursor_records",
            "public_aliases",
        ):
            setattr(previous, name, copy.copy(getattr(self.state, name)))
        graph_snapshot = self.state.dag.snapshot()
        try:
            return self._step_once(raw, seed)
        except Exception:
            previous.dag = TypedDag.restore(graph_snapshot)
            self.state = previous
            raise

    def _step_once(self, raw: str | dict, seed: int = 0) -> StepResult:
        if self.state.done:
            raise RuntimeError("Cannot act after episode termination")
        if self.state.budget.actions >= self.task.limits.max_actions:
            raise RuntimeError("Action budget has already been exhausted")
        self._bump("actions")
        decision = self._node(
            new_id("decision"),
            "decision",
            "controller_action",
            {"raw": raw},
            observation_parents=self.state.last_view_node_ids,
            rng_key=seed,
        )
        self.state.dag.complete(decision)
        action_state = copy.copy(self.state)
        for name in (
            "disclosed_ids",
            "observations",
            "observation_nodes",
            "image_nodes",
            "history",
            "candidates",
            "errors",
            "executed_renders",
            "action_events",
            "cursor_records",
            "public_aliases",
        ):
            setattr(action_state, name, copy.copy(getattr(self.state, name)))
        action_graph = self.state.dag.snapshot()
        error_record = None
        render = None
        output: dict[str, Any] = {}
        action_kind = None
        try:
            action = parse_action(self.resolve_action(raw))
            action_kind = action.kind
            if self.token_counter is not None and self.token_counter(action.reason) > 40:
                raise ContractError(
                    "REASON_TOO_LONG", "An action reason may contain at most 40 tokens"
                )
            if self.state.plan is None and not isinstance(action, PlanAction):
                raise ContractError(
                    "PLAN_REQUIRED", "A valid PLAN is required before any other action"
                )
            if isinstance(action, PlanAction):
                output = self._plan(action, decision.node_version_id)
            elif isinstance(action, RenderAction):
                self._budget_available("renders", self.task.limits.max_renders)
                self._ensure_revision(action.expected_revision)
                self._ensure_known((action.control_set_id,))
                if action.control_set_id != self.state.manifest.active_control_set_id:
                    raise ContractError(
                        "INVALID_CONTROL_SET", "RENDER must use the current control set"
                    )
                source_id = self.state.manifest.selected_image_id
                bundle = self._compile(self.state.plan, source_id)
                render = self._render(
                    bundle, self.state.plan, source_id, seed, decision.node_version_id
                )
            elif isinstance(action, DraftRepairAction):
                output = self._draft(action)
            elif isinstance(action, ApplyAction):
                self._budget_available("renders", self.task.limits.max_renders)
                self._ensure_revision(action.expected_revision)
                self._ensure_known((action.candidate_id,))
                candidate = self.state.candidates.get(action.candidate_id)
                if candidate is None:
                    raise ContractError("UNKNOWN_CANDIDATE", "The repair candidate is unavailable")
                if (
                    candidate.parent_revision != self.state.manifest.revision_id
                    or candidate.parent_manifest_signature != self.state.manifest.signature
                    or candidate.source_image_id != self.state.manifest.selected_image_id
                ):
                    raise ContractError(
                        "STALE_CANDIDATE", "The image or manifest changed after drafting"
                    )
                self._materialize_plan(candidate.plan, decision.node_version_id)
                render = self._render(
                    candidate.bundle,
                    candidate.plan,
                    candidate.source_for_render,
                    seed,
                    decision.node_version_id,
                )
            elif isinstance(action, InspectAction):
                output = self._inspect(action, decision.node_version_id)
            elif isinstance(action, ReadNodesAction):
                output = self._read(action)
            elif isinstance(action, SelectAction):
                self._select(action.image_id)
                output = {"selected_image_id": action.image_id}
            elif isinstance(action, FinishAction):
                image_id = (
                    action.image_id
                    if action.image_was_provided
                    else self.state.manifest.selected_image_id
                )
                if image_id is not None:
                    self._ensure_known((image_id,))
                    if image_id not in self.state.history:
                        raise ContractError(
                            "INVALID_DELIVERY", "FINISH can only deliver a legal historical image"
                        )
                self.state.done = True
                self.state.delivered_image_id = image_id
        except ContractError as error:
            action_state.dag = TypedDag.restore(action_graph)
            self.state = action_state
            error_record = ErrorRecord(
                code=error.code,
                message=str(error),
                policy_invalid=error.policy_invalid,
                node_ids=error.node_ids,
            )
            self.state.errors.append(error_record)
            if error.policy_invalid:
                self._bump("invalid")
                if self.state.budget.invalid >= 2:
                    self.state.done = True
                    self.state.protocol_failure = True
                    self.state.delivered_image_id = None
        except ValidationError as error:
            action_state.dag = TypedDag.restore(action_graph)
            self.state = action_state
            error_record = ErrorRecord(
                code="INVALID_CONDITION", message=str(error), policy_invalid=True
            )
            self.state.errors.append(error_record)
            self._bump("invalid")
            if self.state.budget.invalid >= 2:
                self.state.done = True
                self.state.protocol_failure = True
                self.state.delivered_image_id = None
        if not self.state.done and self.state.budget.actions >= self.task.limits.max_actions:
            self.state.done = True
            self.state.delivered_image_id = self.state.manifest.selected_image_id
            output["termination"] = "action_budget"
        self.state.last_tool_result = None
        read_nodes = (
            self.state.last_tool_node_ids
            if action_kind == "READ_NODES" and error_record is None
            else ()
        )
        self.state.last_tool_node_ids = ()
        if output:
            self.state.last_tool_node_ids = tuple(
                dict.fromkeys((decision.node_version_id, *read_nodes))
            )
            public_output = {"action": action_kind, "result": copy.deepcopy(output)}
            counter = self.token_counter or (lambda value: (len(value.encode("utf-8")) + 2) // 3)
            if (
                counter(
                    json.dumps(
                        self._translate(public_output, True),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                > 1200
            ):
                stored = self.artifacts.put_json(
                    public_output,
                    producer_id=decision.node_version_id,
                    metadata={"visibility": "public", "role": "tool_result"},
                )
                self.state.disclosed_ids.add(stored.artifact_id)
                public_output = {
                    "action": action_kind,
                    "result_artifact_id": stored.artifact_id,
                    "fields": list(output),
                    "message": "Tool result is available through READ_NODES with selected fields",
                }
            self.state.last_tool_result = public_output
        result = StepResult(
            action_kind=action_kind,
            done=self.state.done,
            delivered_image_id=self.state.delivered_image_id,
            protocol_failure=self.state.protocol_failure,
            revision_id=self.state.manifest.revision_id,
            error=error_record,
            output=output,
            render=render,
        )
        self.state.action_events.append(
            {
                "turn": self.state.budget.actions - 1,
                "decision_node_id": decision.node_version_id,
                "raw": copy.deepcopy(raw),
                "result": result.model_dump(mode="json"),
            }
        )
        return result

    def _validate_plan(self, plan: PlanSpec, internal_masks: set[str] | None = None):
        limits = self.task.limits
        if (
            len(plan.entities) > limits.max_entities
            or len(plan.regions) > limits.max_active_regions
        ):
            raise ContractError(
                "CAPACITY_EXCEEDED", "Plan exceeds the entity or active-region capacity"
            )
        reference_images = {binding.image_id for binding in plan.references}
        source = self.state.manifest.selected_image_id or self.task.user_request.source_image_id
        reference_count = len(reference_images) + int(source is not None)
        if reference_count > limits.max_references:
            raise ContractError(
                "CAPACITY_EXCEEDED", "Source plus appearance references exceeds capacity"
            )
        spans = {span.span_id for span in self.task.user_request.spans}
        for item in (*plan.entities, *plan.requirements):
            if not set(item.request_span_ids) <= spans:
                raise ContractError(
                    "UNKNOWN_SPAN", "Plan refers to an unknown original-request span"
                )
        allowed = self.state.disclosed_ids | (internal_masks or set())
        for binding in plan.references:
            if (
                binding.image_id not in allowed
                or binding.crop_mask_id is not None
                and binding.crop_mask_id not in allowed
            ):
                raise ContractError(
                    "UNKNOWN_REFERENCE", "Reference binding uses an undisclosed artifact"
                )
            if binding.scope == "regional" and not plan.regions:
                raise ContractError(
                    "DISCONNECTED_REFERENCE",
                    "References must use global scope when no regions are active",
                )
        for region in plan.regions:
            for mask_id in (region.source_mask_id, region.target_mask_id):
                if mask_id is not None:
                    if mask_id not in allowed:
                        raise ContractError("UNKNOWN_MASK", "Region uses an undisclosed mask")
                    ref = self.artifacts.get_ref(mask_id)
                    if ref.kind != "array":
                        raise ContractError("INVALID_MASK", "Region masks must be arrays")
                    try:
                        mask = validate_mask(self.artifacts.get_array(mask_id))
                    except (ValueError, TypeError) as error:
                        raise ContractError("INVALID_MASK", str(error)) from error
                    if mask.shape != (self.task.canvas.height, self.task.canvas.width):
                        raise ContractError(
                            "INVALID_FRAME", "Region mask must be registered to the task canvas"
                        )
            if source is None and (
                region.source_mask_id is not None or region.source_box is not None
            ):
                raise ContractError("SOURCE_REQUIRED", "A source region requires a source image")
        if plan.source_image_id is not None:
            self._ensure_known((plan.source_image_id,))
            if plan.source_image_id != source:
                raise ContractError(
                    "INVALID_SOURCE",
                    "Ordinary edits must use the selected image; SELECT a historical image first",
                )
        if hasattr(self.compiler, "validate_plan"):
            self.compiler.validate_plan(plan, source)

    def _plan(self, action: PlanAction, decision_id: str) -> dict[str, Any]:
        if self.state.plan is not None:
            raise ContractError(
                "PLAN_ALREADY_EXISTS", "Use typed patches to revise an existing plan"
            )
        self._validate_plan(action.plan)
        self._materialize_plan(action.plan, decision_id)
        self.state.checks_revealed = True
        check_node = self._node(
            "check_spec",
            "check_spec",
            "reveal_checks",
            self.task.check_spec.model_dump(mode="json"),
            data_parents=(self.state.manifest.bindings["user_request"],),
        )
        self.state.dag.complete(check_node)
        bindings = dict(self.state.manifest.bindings)
        bindings["check_spec"] = check_node.node_version_id
        manifest = self.state.manifest.model_copy(
            update={
                "bindings": bindings,
                "revision_id": new_id("revision"),
                "parent_revision_id": self.state.manifest.revision_id,
            }
        )
        self.state.manifest = self.state.dag.save_revision(manifest)
        source = self.task.user_request.source_image_id
        if source:
            manifest = self.state.manifest.model_copy(
                update={
                    "selected_image_id": source,
                    "selected_plan_revision": self.state.manifest.revision_id,
                    "revision_id": new_id("revision"),
                    "parent_revision_id": self.state.manifest.revision_id,
                }
            )
            self.state.manifest = self.state.dag.save_revision(manifest)
            self.state.history[source] = ImageState(source, self.state.manifest, action.plan)
            observations = self._service(self.observer.check, self.task, source)
            self._record_observations(observations, source)
        self._disclose_manifest()
        return {
            "active_control_set_id": self.state.manifest.active_control_set_id,
            "revision_id": self.state.manifest.revision_id,
            "check_spec_revealed": True,
        }

    def _plan_items(self, plan: PlanSpec) -> dict[str, tuple[str, dict[str, Any]]]:
        result = {
            "plan": (
                "plan",
                {
                    "global_text": plan.global_text,
                    "dependencies": [list(edge) for edge in plan.dependencies],
                    "source_image_id": plan.source_image_id,
                },
            )
        }
        for kind, values, key in (
            ("entity", plan.entities, "entity_id"),
            ("requirement", plan.requirements, "requirement_id"),
            ("binding", plan.references, "binding_id"),
            ("region", plan.regions, "region_id"),
        ):
            for item in values:
                result[getattr(item, key)] = (kind, item.model_dump(mode="json"))
        return result

    def _materialize_plan(self, plan: PlanSpec, decision_id: str):
        items = self._plan_items(plan)
        parents = {key: {"plan"} if key != "plan" else set() for key in items}
        for parent, child in plan.dependencies:
            if parent == "controls" or child == "controls":
                raise ContractError(
                    "INVALID_DEPENDENCY", "Control-set execution dependencies are compiler-owned"
                )
            parents[child].add(parent)
        for item in plan.requirements:
            parents[item.requirement_id].update((*item.depends_on, *item.entity_ids))
        for item in plan.references:
            if item.entity_id:
                parents[item.binding_id].add(item.entity_id)
        for item in plan.regions:
            parents[item.region_id].update((*item.entity_ids, *item.reference_ids))
        remaining = set(items)
        order = []
        while remaining:
            ready = sorted(node for node in remaining if not parents[node] & remaining)
            if not ready:
                raise ContractError("DEPENDENCY_CYCLE", "Typed plan dependencies contain a cycle")
            order.extend(ready)
            remaining.difference_update(ready)
        old_manifest = self.state.manifest
        bindings = {
            logical_id: version_id
            for logical_id, version_id in old_manifest.bindings.items()
            if self.state.dag.node(version_id).kind not in MUTABLE_KINDS | {"control_set"}
        }
        for logical_id in order:
            kind, parameters = items[logical_id]
            data_parents = (
                bindings["user_request"],
                *(bindings[parent] for parent in sorted(parents[logical_id])),
            )
            old_version = old_manifest.bindings.get(logical_id)
            old = self.state.dag.node(old_version) if old_version else None
            if (
                old is not None
                and old.typed_parameters == parameters
                and old.data_parents == data_parents
            ):
                bindings[logical_id] = old_version
                continue
            node = self._node(
                logical_id,
                kind,
                "plan_revision",
                parameters,
                data_parents=data_parents,
                observation_parents=(decision_id,),
                request_span_ids=tuple(parameters.get("request_span_ids", ())),
                entity_ids=tuple(
                    parameters.get(
                        "entity_ids",
                        (parameters["entity_id"],) if parameters.get("entity_id") else (),
                    )
                ),
                plan_requirement_ids=(logical_id,) if kind == "requirement" else (),
                frame_ids=tuple(
                    parameters[key]
                    for key in ("source_frame_id", "target_frame_id")
                    if parameters.get(key)
                ),
            )
            self.state.dag.complete(node)
            bindings[logical_id] = node.node_version_id
        controls = self._node(
            "controls",
            "control_set",
            "compile_controls",
            {"plan": plan.model_dump(mode="json")},
            data_parents=tuple(bindings[key] for key in order),
        )
        self.state.dag.complete(controls)
        bindings["controls"] = controls.node_version_id
        revision_id = new_id("revision")
        manifest = BranchManifest(
            branch_id=old_manifest.branch_id,
            revision_id=revision_id,
            parent_revision_id=old_manifest.revision_id,
            bindings=bindings,
            selected_image_id=old_manifest.selected_image_id,
            selected_plan_revision=revision_id,
            active_control_set_id=controls.node_version_id,
        )
        self.state.manifest = self.state.dag.save_revision(manifest)
        self.state.plan = plan
        self.state.candidates = {}
        self._disclose_manifest()

    def _disclose_manifest(self):
        self.state.disclosed_ids.update(self.state.manifest.bindings)
        self.state.disclosed_ids.update(self.state.manifest.bindings.values())
        if self.state.manifest.active_control_set_id:
            self.state.disclosed_ids.add(self.state.manifest.active_control_set_id)
        self.state.disclosed_ids.add(self.state.manifest.revision_id)

    def _compile(self, plan: PlanSpec, source_id: str | None):
        try:
            return self._service(self.compiler.compile, plan, source_id)
        except ContractError:
            raise
        except (ValueError, TypeError) as error:
            raise ContractError("INVALID_CONDITION", str(error)) from error

    @staticmethod
    def _bundle_id(bundle) -> str:
        value = (
            bundle.get("bundle_id")
            if isinstance(bundle, dict)
            else getattr(bundle, "bundle_id", None)
        )
        if not value:
            raise ServiceFailure("Compiler returned a bundle without an immutable identifier")
        return str(value)

    def _render(
        self, bundle, plan: PlanSpec, source_id: str | None, seed: int, decision_id: str
    ) -> RenderResult:
        self._budget_available("renders", self.task.limits.max_renders)
        bundle_id = self._bundle_id(bundle)
        job_id = content_id(
            "render",
            canonical_json(
                {
                    "episode": self.episode_id,
                    "branch": self.state.manifest.branch_id,
                    "turn": self.state.budget.actions,
                    "bundle": bundle_id,
                    "seed": seed,
                    "versions": self.versions,
                }
            ),
        )
        input_images = [("source", source_id)] if source_id else []
        input_images.extend(("appearance", binding.image_id) for binding in plan.references)
        parents = [self.state.manifest.active_control_set_id]
        slots = []
        for index, (role, image_id) in enumerate(input_images):
            if image_id not in self.state.image_nodes:
                raise ContractError(
                    "UNKNOWN_IMAGE", "Renderer input has no public image provenance"
                )
            parents.append(self.state.image_nodes[image_id])
            slots.append(
                InputSlot(
                    slot_name=role,
                    index=index,
                    artifact_value_id=self.artifacts.get_ref(image_id).value_id,
                    producer_version_id=self.state.image_nodes[image_id],
                )
            )
        operation = self._node(
            job_id,
            "render",
            "render",
            {"bundle_id": bundle_id, "source_image_id": source_id, "job_id": job_id},
            data_parents=tuple(parents),
            observation_parents=(decision_id,),
            inputs=tuple(slots),
            rng_key=seed,
        )
        self.state.dag.append(operation)
        self.state.dag.transition(operation.node_version_id, "READY")
        self.state.dag.transition(operation.node_version_id, "RUNNING")
        self._bump("renders")
        try:
            rendered = self._service(self.renderer.render, bundle, seed=seed, job_id=job_id)
            try:
                result = (
                    rendered
                    if isinstance(rendered, RenderResult)
                    else RenderResult.model_validate(rendered)
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise ServiceFailure("Renderer returned an invalid response") from error
            if result.bundle_id != bundle_id:
                raise ServiceFailure("Renderer returned a different compiled condition")
            ref = self.artifacts.get_ref(result.image_id)
            if ref.kind != "rgb" or ref.shape[:2] != (
                self.task.canvas.height,
                self.task.canvas.width,
            ):
                raise ServiceFailure("Renderer returned an invalid image shape or type")
        except Exception:
            self.state.dag.transition(
                operation.node_version_id, "FAILED", error_code="SERVICE_FAILURE"
            )
            raise
        self.state.dag.transition(
            operation.node_version_id, "SUCCEEDED", output_artifact_ids=(result.image_id,)
        )
        image = self._node(
            result.image_id,
            "image",
            "render_output",
            {"image_id": result.image_id, "trajectory_id": result.trajectory_id},
            data_parents=(operation.node_version_id,),
        )
        self.state.dag.complete(image, (result.image_id,))
        self.state.image_nodes[result.image_id] = image.node_version_id
        self.state.disclosed_ids.add(result.image_id)
        self.state.executed_renders.append(result)
        old_manifest = self.state.manifest
        manifest = old_manifest.model_copy(
            update={
                "revision_id": new_id("revision"),
                "parent_revision_id": old_manifest.revision_id,
                "selected_image_id": result.image_id,
            }
        )
        self.state.manifest = self.state.dag.save_revision(manifest)
        self.state.history[result.image_id] = ImageState(
            result.image_id, self.state.manifest, plan, bundle, result, operation.node_version_id
        )
        self.state.candidates = {}
        self.state.active_crop_id = None
        observations = self._service(self.observer.check, self.task, result.image_id)
        self._record_observations(observations, result.image_id)
        self._disclose_manifest()
        return result

    def _record_observations(self, observations, image_id: str, decision_id: str | None = None):
        if isinstance(observations, (dict, Observation)):
            observations = [observations]
        check_ids = {atom.check_id for atom in self.task.check_spec.atoms}
        for raw in observations:
            try:
                item = raw if isinstance(raw, Observation) else Observation.model_validate(raw)
            except (ValidationError, TypeError, ValueError) as error:
                raise ServiceFailure("Observer returned an invalid response") from error
            if item.image_id != image_id:
                raise ServiceFailure("Observer attached evidence to a different image")
            if item.origin == "check_spec" and item.check_id not in check_ids:
                raise ServiceFailure(
                    "Observer returned a check outside the fixed public specification"
                )
            if (
                item.observation_id in self.state.observations
                and self.state.observations[item.observation_id] != item
            ):
                raise ServiceFailure("Observation identities cannot be overwritten")
            parent_ids = [self.state.image_nodes[image_id]]
            if "check_spec" in self.state.manifest.bindings and item.origin == "check_spec":
                parent_ids.append(self.state.manifest.bindings["check_spec"])
            if item.origin == "plan":
                parent_ids.append(self.state.manifest.active_control_set_id)
            observation_node = self._node(
                item.observation_id,
                "observation",
                item.kind,
                item.model_dump(mode="json"),
                data_parents=tuple(parent_ids),
                observation_parents=(decision_id,) if decision_id else (),
                entity_ids=item.entity_ids,
                frame_ids=(item.frame_id,) if item.frame_id else (),
            )
            self.state.dag.complete(observation_node, item.artifact_ids)
            self.state.observations[item.observation_id] = item
            self.state.observation_nodes[item.observation_id] = observation_node.node_version_id
            self.state.disclosed_ids.add(item.observation_id)
            self.state.disclosed_ids.add(observation_node.node_version_id)
            self.state.disclosed_ids.update(item.artifact_ids)
            for artifact_id in item.artifact_ids:
                ref = self.artifacts.get_ref(artifact_id)
                if ref.kind == "rgb" and artifact_id not in self.state.image_nodes:
                    image_node = self._node(
                        artifact_id,
                        "image",
                        "observed_crop",
                        {"image_id": artifact_id},
                        data_parents=(observation_node.node_version_id,),
                    )
                    self.state.dag.complete(image_node, (artifact_id,))
                    self.state.image_nodes[artifact_id] = image_node.node_version_id

    def _patch_plan(
        self, operations: tuple[PatchOperation, ...]
    ) -> tuple[PlanSpec, tuple[str, ...]]:
        current = self._plan_items(self.state.plan)
        changed = []
        seen = set()
        for operation in operations:
            logical_id = operation.target_logical_node_id
            if logical_id in {"user_request", "check_spec", "controls"}:
                raise ContractError("IMMUTABLE_NODE", f"Node {logical_id} cannot be patched")
            current_version = self.state.manifest.bindings.get(logical_id)
            if operation.op == "add":
                if (
                    current_version is not None
                    or logical_id in current
                    or operation.expected_old_version_id is not None
                ):
                    raise ContractError(
                        "STALE_PATCH", "Adding a node requires a new ID and a null old version"
                    )
                if operation.field_path:
                    raise ContractError(
                        "INVALID_PATCH", "Adding a node requires its whole typed value"
                    )
                value = operation.typed_value_or_artifact_id
                if (
                    not isinstance(value, dict)
                    or value.get("kind") not in MUTABLE_KINDS - {"plan"}
                    or not isinstance(value.get("parameters"), dict)
                ):
                    raise ContractError(
                        "INVALID_PATCH", "Added nodes need a mutable kind and typed parameters"
                    )
                current[logical_id] = (value["kind"], copy.deepcopy(value["parameters"]))
            else:
                self._ensure_known((logical_id, operation.expected_old_version_id))
                if (
                    current_version != operation.expected_old_version_id
                    or logical_id not in current
                ):
                    raise ContractError(
                        "STALE_PATCH",
                        "Patch old version does not match the current binding",
                        node_ids=(logical_id,),
                    )
                if logical_id in seen and logical_id not in current:
                    raise ContractError("INVALID_PATCH", "A retired node cannot be patched again")
                if operation.op == "retire" and not operation.field_path:
                    if logical_id == "plan":
                        raise ContractError("IMMUTABLE_NODE", "The active plan cannot be retired")
                    del current[logical_id]
                else:
                    kind, parameters = current[logical_id]
                    current[logical_id] = (kind, apply_field_patch(parameters, operation))
            changed.append(logical_id)
            seen.add(logical_id)
        plan_values = current["plan"][1]
        grouped = {kind: [] for kind in ("entity", "requirement", "region", "binding")}
        key_names = {
            "entity": "entity_id",
            "requirement": "requirement_id",
            "region": "region_id",
            "binding": "binding_id",
        }
        for logical_id, (kind, values) in current.items():
            if kind in grouped:
                if values.get(key_names[kind]) != logical_id:
                    raise ContractError(
                        "IDENTITY_CHANGE",
                        "A patch cannot change a stable logical entity or control ID",
                    )
                grouped[kind].append(values)
        plan = PlanSpec(
            global_text=plan_values["global_text"],
            source_image_id=plan_values.get("source_image_id"),
            dependencies=plan_values.get("dependencies", ()),
            entities=tuple(Entity.model_validate(value) for value in grouped["entity"]),
            requirements=tuple(
                PlanRequirement.model_validate(value) for value in grouped["requirement"]
            ),
            regions=tuple(RegionControl.model_validate(value) for value in grouped["region"]),
            references=tuple(
                ReferenceBinding.model_validate(value) for value in grouped["binding"]
            ),
        )
        if self.state.manifest.selected_image_id:
            plan = plan.model_copy(
                update={"source_image_id": self.state.manifest.selected_image_id}
            )
        self._validate_plan(plan)
        return plan, tuple(dict.fromkeys(changed))

    def _region_mask(self, region: RegionControl, side: str) -> np.ndarray | None:
        mask_id = getattr(region, side + "_mask_id")
        box = getattr(region, side + "_box")
        if mask_id:
            return self.artifacts.get_array(mask_id)
        if box:
            return box_mask(box, (self.task.canvas.height, self.task.canvas.width))
        return None

    def _expanded_plan(self, plan: PlanSpec, changed: tuple[str, ...]) -> tuple[PlanSpec, set[str]]:
        original = {item.region_id: item for item in self.state.plan.regions}
        changed_entities = {item.entity_id for item in plan.entities if item.entity_id in changed}
        focus = []
        for region in plan.regions:
            if (
                region.region_id in changed
                or set(region.entity_ids) & changed_entities
                or original.get(region.region_id) != region
            ):
                focus.append(region)
        if not focus:
            raise ContractError(
                "NO_LOCAL_REGION",
                "No reliable region is associated with this repair",
                policy_invalid=False,
            )
        active = {region.region_id for region in focus}
        for region in plan.regions:
            if region.region_id in active:
                continue
            region_mask = self._region_mask(region, "target")
            for target in focus:
                target_mask = self._region_mask(target, "target")
                if (
                    set(region.entity_ids) & set(target.entity_ids)
                    or region_mask is not None
                    and target_mask is not None
                    and mask_iou(region_mask, target_mask) > 0.2
                ):
                    active.add(region.region_id)
                    break
        if len(active) > self.task.limits.max_active_regions:
            raise ContractError(
                "CAPACITY_EXCEEDED",
                "Expanded repair exceeds the active-region limit",
                policy_invalid=False,
            )
        updated = []
        created_masks = set()
        for region in plan.regions:
            changes = {}
            if region.region_id in active:
                for side in ("source", "target"):
                    mask = self._region_mask(region, side)
                    if mask is not None:
                        ref = self.artifacts.put_array(
                            expand_mask(mask, pixels=32),
                            producer_id=self.state.manifest.active_control_set_id,
                            metadata={
                                "kind": "region_mask",
                                "side": side,
                                "scope": "expanded",
                                "image_id": self.state.manifest.selected_image_id,
                            },
                        )
                        changes[side + "_mask_id"] = ref.artifact_id
                        changes[side + "_box"] = None
                        created_masks.add(ref.artifact_id)
            updated.append(region.model_copy(update=changes))
        return plan.model_copy(update={"regions": tuple(updated)}), created_masks

    def _global_plan(self, plan: PlanSpec) -> PlanSpec:
        region_texts = [region.text for region in plan.regions if region.text.strip()]
        full_text = plan.global_text
        if region_texts:
            full_text += "\nRegional requirements: " + "; ".join(region_texts)
        references = tuple(
            binding.model_copy(update={"scope": "global"}) for binding in plan.references
        )
        return plan.model_copy(
            update={"global_text": full_text, "regions": (), "references": references}
        )

    def _draft(self, action: DraftRepairAction) -> dict[str, Any]:
        self._ensure_known((action.source_image_id, *action.evidence_ids, *action.suspect_node_ids))
        if action.source_image_id != self.state.manifest.selected_image_id:
            raise ContractError(
                "INVALID_SOURCE", "Select a historical image before drafting an edit to it"
            )
        for evidence_id in action.evidence_ids:
            if evidence_id not in self.state.observations:
                raise ContractError(
                    "INVALID_EVIDENCE", "Repair evidence must identify a disclosed observation"
                )
            if self.state.observations[evidence_id].image_id != action.source_image_id:
                raise ContractError(
                    "STALE_EVIDENCE", "Repair evidence belongs to a different image version"
                )
        for suspect_id in action.suspect_node_ids:
            version = self.state.manifest.bindings.get(suspect_id, suspect_id)
            if self.state.dag.node(version).kind not in MUTABLE_KINDS | {"render", "control_set"}:
                raise ContractError(
                    "INVALID_SUSPECT",
                    "Repair suspect must be a plan, control, binding, or render node",
                )
        if action.scope == "retry":
            previous = self.state.history[action.source_image_id]
            if previous.bundle is None:
                raise ContractError(
                    "NO_RENDER_TO_RETRY",
                    "The original source image has no renderer operation to retry",
                )
            bundle = previous.bundle
            source = self.state.dag.node(previous.render_node_id).typed_parameters[
                "source_image_id"
            ]
            candidate = RepairCandidate(
                new_id("candidate"),
                action.source_image_id,
                self.state.manifest.revision_id,
                self.state.manifest.signature,
                previous.plan,
                "retry",
                (),
                (),
                (previous.render_node_id,),
                bundle,
                self._bundle_id(bundle),
                source,
            )
            self.state.candidates = {candidate.candidate_id: candidate}
            self.state.disclosed_ids.add(candidate.candidate_id)
            return {"candidates": [candidate.summary()], "rejected": []}
        plan, changed = self._patch_plan(action.patch_ops)
        scopes = ("local", "expanded", "global") if action.scope == "automatic" else (action.scope,)
        candidates = {}
        seen_bundles = set()
        rejected = []
        changed_versions = {
            self.state.manifest.bindings[logical_id]
            for logical_id in changed
            if logical_id in self.state.manifest.bindings
        }
        closure = (
            self.state.dag.recompute_closure(
                changed_versions, {self.state.manifest.active_control_set_id}
            )
            if changed_versions
            else set()
        )
        for scope in scopes:
            try:
                created_masks = set()
                candidate_plan = plan
                if scope == "expanded":
                    candidate_plan, created_masks = self._expanded_plan(plan, changed)
                elif scope == "global":
                    candidate_plan = self._global_plan(plan)
                elif scope == "local" and not plan.regions:
                    raise ContractError(
                        "NO_LOCAL_REGION",
                        "A local repair requires a reliable active region",
                        policy_invalid=False,
                    )
                self._validate_plan(candidate_plan, created_masks)
                bundle = self._compile(candidate_plan, action.source_image_id)
                bundle_id = self._bundle_id(bundle)
                if bundle_id in seen_bundles:
                    continue
                seen_bundles.add(bundle_id)
                candidate = RepairCandidate(
                    new_id("candidate"),
                    action.source_image_id,
                    self.state.manifest.revision_id,
                    self.state.manifest.signature,
                    candidate_plan,
                    scope,
                    action.patch_ops,
                    changed,
                    tuple(sorted(closure)),
                    bundle,
                    bundle_id,
                    action.source_image_id,
                )
                candidates[candidate.candidate_id] = candidate
            except ContractError as error:
                if action.scope != "automatic":
                    raise
                rejected.append({"scope": scope, "code": error.code, "message": str(error)})
        self.state.candidates = candidates
        self.state.disclosed_ids.update(candidates)
        return {
            "candidates": [candidate.summary() for candidate in candidates.values()],
            "rejected": rejected,
        }

    def _select(self, image_id: str):
        self._ensure_known((image_id,))
        if image_id not in self.state.history:
            raise ContractError("INVALID_SELECTION", "Only legal historical images can be selected")
        previous = self.state.history[image_id]
        manifest = previous.manifest.model_copy(
            update={
                "branch_id": self.state.manifest.branch_id,
                "revision_id": new_id("revision"),
                "parent_revision_id": self.state.manifest.revision_id,
                "selected_image_id": image_id,
            },
            deep=True,
        )
        self.state.manifest = self.state.dag.save_revision(manifest)
        self.state.plan = previous.plan
        self.state.candidates = {}
        self.state.active_crop_id = None
        self._disclose_manifest()

    def _inspect(self, action: InspectAction, decision_id: str) -> dict[str, Any]:
        self._budget_available("inspects", self.task.limits.max_inspects)
        self._ensure_known((action.image_id,))
        if action.image_id not in self.state.image_nodes:
            raise ContractError("INVALID_IMAGE", "INSPECT requires a publicly available image")
        if action.entity_id is not None:
            self._ensure_known((action.entity_id,))
            if action.entity_id not in {entity.entity_id for entity in self.state.plan.entities}:
                raise ContractError("UNKNOWN_ENTITY", "INSPECT entity is not in the current plan")
        self._bump("inspects")
        if action.inspect_kind == "recheck":
            observations = self._service(self.observer.check, self.task, action.image_id)
        else:
            observations = self._service(
                self.observer.inspect,
                self.task,
                action.image_id,
                action.inspect_kind,
                action.entity_id,
                action.region,
                plan=self.state.plan,
            )
        if isinstance(observations, (dict, Observation)):
            observations = [observations]
        try:
            parsed = [
                item if isinstance(item, Observation) else Observation.model_validate(item)
                for item in observations
            ]
        except (ValidationError, TypeError, ValueError) as error:
            raise ServiceFailure("Observer returned an invalid inspection response") from error
        expected_origin = "plan" if action.inspect_kind == "plan_consistency" else None
        if expected_origin and any(item.origin != expected_origin for item in parsed):
            raise ServiceFailure("Plan-consistency evidence must be labeled with origin=plan")
        self._record_observations(parsed, action.image_id, decision_id)
        if action.inspect_kind == "crop":
            for item in parsed:
                for artifact_id in item.artifact_ids:
                    if self.artifacts.get_ref(artifact_id).kind == "rgb":
                        self.state.active_crop_id = artifact_id
                        break
        return {"observations": [item.model_dump(mode="json") for item in parsed]}

    def _public_node(self, node):
        return {
            "node_version_id": node.node_version_id,
            "logical_node_id": node.logical_node_id,
            "kind": node.kind,
            "operation": node.operation,
            "typed_parameters": node.typed_parameters,
            "data_parents": list(node.data_parents),
            "observation_parents": list(node.observation_parents),
            "request_span_ids": list(node.request_span_ids),
            "entity_ids": list(node.entity_ids),
            "frame_ids": list(node.frame_ids),
            "input_slots": [
                {
                    "slot_name": item.slot_name,
                    "index": item.index,
                    "producer_version_id": item.producer_version_id,
                }
                for item in node.inputs
            ],
        }

    def public_records(self) -> list[dict[str, Any]]:
        if not self.state.checks_revealed:
            return []
        records = []
        for logical_id, version_id in self.state.manifest.bindings.items():
            node = self.state.dag.node(version_id)
            if node.kind not in {"request", "check_spec", "control_set"}:
                records.append(
                    {
                        "id": logical_id,
                        "version_id": version_id,
                        "kind": node.kind,
                        "value": self._public_node(node),
                    }
                )
        for atom in self.task.check_spec.atoms:
            records.append(
                {"id": atom.check_id, "kind": "check", "value": atom.model_dump(mode="json")}
            )
        for item in self.state.observations.values():
            records.append(
                {
                    "id": item.observation_id,
                    "kind": "observation",
                    "value": item.model_dump(mode="json"),
                }
            )
        return records

    def make_cursor(self, offset: int) -> str:
        records = self.public_records()
        projection_id = content_id("projection", canonical_json(records))
        for cursor, value in self.state.cursor_records.items():
            if (
                value["revision_id"] == self.state.manifest.revision_id
                and value["offset"] == offset
                and value["projection_id"] == projection_id
            ):
                return cursor
        cursor = new_id("cursor")
        self.state.cursor_records[cursor] = {
            "episode_id": self.episode_id,
            "branch_id": self.state.manifest.branch_id,
            "revision_id": self.state.manifest.revision_id,
            "collection_kind": "public_nodes_and_checks",
            "offset": offset,
            "projection_id": projection_id,
        }
        return cursor

    def _read(self, action: ReadNodesAction) -> dict[str, Any]:
        self._budget_available("reads", self.task.limits.max_reads)
        self._bump("reads")
        records = self.public_records()
        offset = 0
        if action.page_cursor:
            cursor = self.state.cursor_records.get(action.page_cursor)
            if cursor is None:
                raise ContractError("UNKNOWN_CURSOR", "Cursor was not disclosed to this branch")
            projection_id = content_id("projection", canonical_json(records))
            if (
                cursor["revision_id"] != self.state.manifest.revision_id
                or cursor["branch_id"] != self.state.manifest.branch_id
                or cursor["projection_id"] != projection_id
            ):
                current_cursor = self.make_cursor(0)
                return {"error": "STALE_CURSOR", "page_cursor": current_cursor, "records": []}
            offset = cursor["offset"]
            selected = records[offset : offset + 16]
        else:
            self._ensure_known(action.node_ids)
            lookup = {item["id"]: item for item in records}
            lookup.update({item["version_id"]: item for item in records if "version_id" in item})
            for node_id in action.node_ids:
                if node_id not in lookup and node_id.startswith("artifact:"):
                    reference = self.artifacts.get_ref(node_id)
                    if (
                        reference.kind == "json"
                        and reference.metadata.get("visibility") == "public"
                    ):
                        value = self.artifacts.get_json(node_id)
                        lookup[node_id] = {
                            "id": node_id,
                            "kind": "public_artifact",
                            "value": value.get("result", value),
                        }
                if node_id not in lookup and node_id in self.state.dag.nodes:
                    node = self.state.dag.node(node_id)
                    if node.kind in MUTABLE_KINDS | {
                        "image",
                        "render",
                        "control_set",
                        "observation",
                    }:
                        lookup[node_id] = {
                            "id": node_id,
                            "version_id": node_id,
                            "kind": node.kind,
                            "value": self._public_node(node),
                        }
            if any(node_id not in lookup for node_id in action.node_ids):
                raise ContractError(
                    "UNKNOWN_NODE", "Requested node is unavailable in this public revision"
                )
            selected = [lookup[node_id] for node_id in action.node_ids]
        page = []
        read_nodes = []
        token_count = self.token_counter or (lambda value: (len(value.encode("utf-8")) + 2) // 3)
        consumed = 0
        for item in selected:
            entry = copy.deepcopy(item)
            if action.fields:
                entry["value"] = {
                    key: value for key, value in entry["value"].items() if key in action.fields
                }
            candidate_page = [*page, entry]
            if (
                token_count(json.dumps(candidate_page, ensure_ascii=False, separators=(",", ":")))
                > 1024
            ):
                if page:
                    break
                artifact = self.artifacts.put_json(
                    entry["value"],
                    producer_id=entry.get("version_id"),
                    metadata={"visibility": "public", "record_id": entry["id"]},
                )
                entry = {
                    "id": entry["id"],
                    "kind": entry["kind"],
                    "summary": "Record exceeds page capacity; request selected fields",
                    "fields": list(item["value"]),
                    "artifact_id": artifact.artifact_id,
                }
                self.state.disclosed_ids.add(artifact.artifact_id)
            page.append(entry)
            consumed += 1
            self.state.disclosed_ids.add(item["id"])
            if "version_id" in item:
                self.state.disclosed_ids.add(item["version_id"])
                read_nodes.append(item["version_id"])
            elif item["id"] in self.state.observation_nodes:
                read_nodes.append(self.state.observation_nodes[item["id"]])
        self.state.last_tool_node_ids = tuple(read_nodes)
        next_offset = offset + consumed
        next_cursor = (
            self.make_cursor(next_offset)
            if action.page_cursor and next_offset < len(records)
            else None
        )
        return {
            "records": page,
            "page_cursor": next_cursor,
            "remaining": len(records) - next_offset
            if action.page_cursor
            else len(selected) - consumed,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task.model_dump(mode="json"),
            "versions": self.versions,
            "dag": self.state.dag.snapshot(),
            "manifest": self.state.manifest.model_dump(mode="json"),
            "budget": self.state.budget.model_dump(),
            "plan": self.state.plan.model_dump(mode="json") if self.state.plan else None,
            "disclosed_ids": sorted(self.state.disclosed_ids),
            "observations": [
                item.model_dump(mode="json") for item in self.state.observations.values()
            ],
            "observation_nodes": self.state.observation_nodes,
            "image_nodes": self.state.image_nodes,
            "history": [
                {
                    "image_id": item.image_id,
                    "manifest": item.manifest.model_dump(mode="json"),
                    "plan": item.plan.model_dump(mode="json"),
                    "bundle_id": self._bundle_id(item.bundle) if item.bundle is not None else None,
                    "render": item.render.model_dump(mode="json") if item.render else None,
                    "render_node_id": item.render_node_id,
                }
                for item in self.state.history.values()
            ],
            "candidates": [
                {
                    **item.summary(),
                    "parent_manifest_signature": item.parent_manifest_signature,
                    "plan": item.plan.model_dump(mode="json"),
                    "patch_operations": [
                        operation.model_dump(mode="json") for operation in item.patch_operations
                    ],
                    "source_for_render": item.source_for_render,
                }
                for item in self.state.candidates.values()
            ],
            "errors": [error.model_dump(mode="json") for error in self.state.errors],
            "executed_renders": [
                item.model_dump(mode="json") for item in self.state.executed_renders
            ],
            "action_events": self.state.action_events,
            "cursor_records": self.state.cursor_records,
            "public_aliases": self.state.public_aliases,
            "last_view_node_ids": self.state.last_view_node_ids,
            "active_crop_id": self.state.active_crop_id,
            "last_tool_result": self.state.last_tool_result,
            "last_tool_node_ids": self.state.last_tool_node_ids,
            "checks_revealed": self.state.checks_revealed,
            "done": self.state.done,
            "delivered_image_id": self.state.delivered_image_id,
            "protocol_failure": self.state.protocol_failure,
        }

    @classmethod
    def restore(
        cls,
        snapshot: dict[str, Any],
        artifacts: ArtifactStore,
        compiler,
        renderer,
        observer,
        bundle_loader: Callable[[str], Any],
        token_counter: Callable[[str], int] | None = None,
        max_input_tokens: int = 6144,
    ) -> EpisodeEnvironment:
        instance = cls.__new__(cls)
        instance.task = PublicTask.model_validate(snapshot["task"])
        instance.artifacts = artifacts
        instance.compiler = compiler
        instance.renderer = renderer
        instance.observer = observer
        instance.episode_id = snapshot["episode_id"]
        instance.versions = snapshot["versions"]
        instance.token_counter = token_counter
        instance.view_builder = AgentViewBuilder(max_input_tokens, token_counter)
        state = EpisodeState(
            dag=TypedDag.restore(snapshot["dag"]),
            manifest=BranchManifest.model_validate(snapshot["manifest"]),
            budget=Budget.model_validate(snapshot["budget"]),
            plan=PlanSpec.model_validate(snapshot["plan"]) if snapshot["plan"] else None,
            disclosed_ids=set(snapshot["disclosed_ids"]),
            observations={
                item["observation_id"]: Observation.model_validate(item)
                for item in snapshot["observations"]
            },
            observation_nodes=dict(snapshot["observation_nodes"]),
            image_nodes=dict(snapshot["image_nodes"]),
            errors=[ErrorRecord.model_validate(item) for item in snapshot["errors"]],
            executed_renders=[
                RenderResult.model_validate(item) for item in snapshot["executed_renders"]
            ],
            action_events=copy.deepcopy(snapshot["action_events"]),
            cursor_records=copy.deepcopy(snapshot["cursor_records"]),
            public_aliases=dict(snapshot.get("public_aliases", {})),
            last_view_node_ids=tuple(snapshot["last_view_node_ids"]),
            active_crop_id=snapshot["active_crop_id"],
            last_tool_result=copy.deepcopy(snapshot.get("last_tool_result")),
            last_tool_node_ids=tuple(snapshot.get("last_tool_node_ids", ())),
            checks_revealed=snapshot["checks_revealed"],
            done=snapshot["done"],
            delivered_image_id=snapshot["delivered_image_id"],
            protocol_failure=snapshot["protocol_failure"],
        )
        for item in snapshot["history"]:
            state.history[item["image_id"]] = ImageState(
                item["image_id"],
                BranchManifest.model_validate(item["manifest"]),
                PlanSpec.model_validate(item["plan"]),
                bundle_loader(item["bundle_id"]) if item["bundle_id"] else None,
                RenderResult.model_validate(item["render"]) if item["render"] else None,
                item["render_node_id"],
            )
        for item in snapshot["candidates"]:
            state.candidates[item["candidate_id"]] = RepairCandidate(
                candidate_id=item["candidate_id"],
                source_image_id=item["source_image_id"],
                parent_revision=item["parent_revision"],
                parent_manifest_signature=item["parent_manifest_signature"],
                plan=PlanSpec.model_validate(item["plan"]),
                scope=item["scope"],
                patch_operations=tuple(
                    PatchOperation.model_validate(operation)
                    for operation in item["patch_operations"]
                ),
                changed_nodes=tuple(item["changed_nodes"]),
                recompute_nodes=tuple(item["recompute_nodes"]),
                bundle=bundle_loader(item["bundle_id"]),
                bundle_id=item["bundle_id"],
                source_for_render=item["source_for_render"],
                validation_result=item["validation_result"],
            )
        instance.state = state
        return instance
