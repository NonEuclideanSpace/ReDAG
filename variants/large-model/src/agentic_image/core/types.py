from __future__ import annotations

import math
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ContractError(ValueError):
    def __init__(
        self, code: str, message: str, policy_invalid: bool = True, node_ids: tuple[str, ...] = ()
    ):
        self.code = code
        self.policy_invalid = policy_invalid
        self.node_ids = node_ids
        super().__init__(message)


class ServiceFailure(RuntimeError):
    pass


class ViewOverflow(ServiceFailure):
    pass


class RequestSpan(FrozenModel):
    span_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end <= self.start:
            raise ValueError("Request span must be a nonempty half-open interval")
        return self


class UserRequest(FrozenModel):
    text: str = Field(min_length=1)
    source_image_id: str | None = None
    appearance_image_ids: tuple[str, ...] = ()
    spans: tuple[RequestSpan, ...] = ()
    controls: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value):
        return unicodedata.normalize("NFC", value)

    @model_validator(mode="after")
    def validate_spans(self):
        ids = [span.span_id for span in self.spans]
        if len(set(ids)) != len(ids):
            raise ValueError("Request span IDs must be unique")
        for span in self.spans:
            if self.text[span.start : span.end] != span.text or span.end > len(self.text):
                raise ValueError(
                    f"Request span {span.span_id} does not match the normalized request"
                )
        return self


class Canvas(FrozenModel):
    height: int = Field(default=512, ge=16, multiple_of=16)
    width: int = Field(default=512, ge=16, multiple_of=16)


class TaskLimits(FrozenModel):
    max_entities: int = 16
    max_active_regions: int = 2
    max_references: int = 2
    check_atoms_per_batch: int = 12
    max_actions: int = 10
    max_renders: int = 3
    max_inspects: int = 2
    max_reads: int = 2


class PublicCheckAtom(FrozenModel):
    check_id: str
    origin: Literal["request", "source_inference"] = "request"
    raw_span_ids: tuple[str, ...] = ()
    predicate: str
    entity_refs: tuple[str, ...] = ()
    expected_value: Any = None
    public_reference_ids: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0, le=1)


class CheckSpec(FrozenModel):
    version: str = "1"
    extractor_version: str
    atoms: tuple[PublicCheckAtom, ...]
    fallback: bool = False

    @model_validator(mode="after")
    def validate_ids(self):
        ids = [atom.check_id for atom in self.atoms]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Public checks must be nonempty with unique IDs")
        return self


class PublicTask(FrozenModel):
    task_id: str
    split_id: str = "train"
    family_id: str
    source_dataset: str = "user"
    license_record: dict[str, Any] = Field(default_factory=dict)
    user_request: UserRequest
    canvas: Canvas = Field(default_factory=Canvas)
    task_mode: Literal["generate", "edit"] = "generate"
    check_spec: CheckSpec
    limits: TaskLimits = Field(default_factory=TaskLimits)

    @model_validator(mode="after")
    def validate_source(self):
        if self.task_mode == "edit" and not self.user_request.source_image_id:
            raise ValueError("Editing requires a public source image")
        span_ids = {span.span_id for span in self.user_request.spans}
        public_images = set(self.user_request.appearance_image_ids)
        if self.user_request.source_image_id:
            public_images.add(self.user_request.source_image_id)
        for atom in self.check_spec.atoms:
            if not set(atom.raw_span_ids) <= span_ids:
                raise ValueError("Public check contains an unknown request span")
            if not set(atom.public_reference_ids) <= public_images:
                raise ValueError("Public check contains a nonpublic reference")
            if isinstance(atom.expected_value, dict) and atom.expected_value.get(
                "reference_image_id"
            ) not in public_images | {None}:
                raise ValueError("Public check contains a hidden reference image")
        return self


class PrivateEvalAtom(FrozenModel):
    eval_id: str
    raw_span: str
    predicate: str
    expected_value: Any = None
    checker_route: str = "semantic"
    threshold: float = Field(default=0.8, ge=0, le=1)
    mandatory: bool = True


class PrivateEvaluation(FrozenModel):
    private_task_key: str
    eval_atoms: tuple[PrivateEvalAtom, ...]
    original_protected_mask_id: str | None = None
    source_baseline_observations: tuple[dict[str, Any], ...] = ()
    annotation_provenance: dict[str, Any] = Field(default_factory=dict)
    evaluator_version: str


class BoundingBox(FrozenModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_bounds(self):
        values = (self.x0, self.y0, self.x1, self.y1)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding boxes must be finite")
        if not 0 <= self.x0 < self.x1 <= 1 or not 0 <= self.y0 < self.y1 <= 1:
            raise ValueError("Bounding boxes use nonempty normalized half-open xyxy coordinates")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


class Entity(FrozenModel):
    entity_id: str
    name: str
    category: str = "object"
    attributes: dict[str, Any] = Field(default_factory=dict)
    request_span_ids: tuple[str, ...] = ()


class PlanRequirement(FrozenModel):
    requirement_id: str
    text: str
    request_span_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()


class ReferenceBinding(FrozenModel):
    binding_id: str
    image_id: str
    entity_id: str | None = None
    crop_mask_id: str | None = None
    role: Literal["appearance", "style"] = "appearance"
    scope: Literal["regional", "global"] = "regional"
    source_frame_id: str | None = None


class RegionControl(FrozenModel):
    region_id: str
    text: str
    entity_ids: tuple[str, ...]
    target_mask_id: str | None = None
    source_mask_id: str | None = None
    target_box: BoundingBox | None = None
    source_box: BoundingBox | None = None
    target_frame_id: str | None = None
    source_frame_id: str | None = None
    reference_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_masks(self):
        if self.target_mask_id is not None and self.target_box is not None:
            raise ValueError("A target region must use either a mask or a box")
        if self.source_mask_id is not None and self.source_box is not None:
            raise ValueError("A source region must use either a mask or a box")
        if self.target_mask_id is None and self.target_box is None:
            raise ValueError("An active region needs a target edit area")
        return self


class PlanSpec(FrozenModel):
    global_text: str = Field(min_length=1)
    entities: tuple[Entity, ...] = ()
    requirements: tuple[PlanRequirement, ...] = ()
    regions: tuple[RegionControl, ...] = ()
    references: tuple[ReferenceBinding, ...] = ()
    source_image_id: str | None = None
    dependencies: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def validate_graph(self):
        entities = {entity.entity_id for entity in self.entities}
        requirements = {item.requirement_id for item in self.requirements}
        region_ids = {region.region_id for region in self.regions}
        bindings = {binding.binding_id for binding in self.references}
        all_ids = (
            [entity.entity_id for entity in self.entities]
            + [item.requirement_id for item in self.requirements]
            + [region.region_id for region in self.regions]
            + [binding.binding_id for binding in self.references]
        )
        if set(all_ids) & {"plan", "controls", "user_request", "check_spec"}:
            raise ValueError("Plan node IDs cannot use reserved system names")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("All plan node IDs must be unique")
        for item in self.requirements:
            if not set(item.entity_ids) <= entities or not set(item.depends_on) <= requirements:
                raise ValueError("A plan requirement refers to an unknown node")
        for region in self.regions:
            if not set(region.entity_ids) <= entities or not set(region.reference_ids) <= bindings:
                raise ValueError("A region refers to an unknown entity or binding")
        for binding in self.references:
            if binding.entity_id is not None and binding.entity_id not in entities:
                raise ValueError("A reference binding refers to an unknown entity")
        nodes = entities | requirements | region_ids | bindings | {"plan", "controls"}
        edges = list(self.dependencies)
        edges.extend(
            (parent, item.requirement_id)
            for item in self.requirements
            for parent in item.depends_on
        )
        incoming = {node: 0 for node in nodes}
        children = {node: [] for node in nodes}
        for parent, child in edges:
            if parent not in nodes or child not in nodes:
                raise ValueError("A dependency refers to an unknown plan node")
            children[parent].append(child)
            incoming[child] += 1
        ready = [node for node, count in incoming.items() if count == 0]
        count = 0
        while ready:
            node = ready.pop()
            count += 1
            for child in children[node]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
        if count != len(nodes):
            raise ValueError("Plan dependencies contain a cycle")
        return self


class Observation(FrozenModel):
    observation_id: str
    image_id: str
    kind: str = "check"
    origin: Literal["check_spec", "plan", "tool"] = "check_spec"
    check_id: str | None = None
    known: bool = False
    score: float = Field(default=0, ge=0, le=1)
    passed: bool = False
    entity_ids: tuple[str, ...] = ()
    node_ids: tuple[str, ...] = ()
    frame_id: str | None = None
    artifact_ids: tuple[str, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)


class RenderResult(FrozenModel):
    image_id: str
    bundle_id: str
    trajectory_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatchOperation(FrozenModel):
    target_logical_node_id: str
    expected_old_version_id: str | None
    op: Literal["add", "replace", "retire"]
    field_path: tuple[str | int, ...] = ()
    typed_value_or_artifact_id: Any = None


class ErrorRecord(FrozenModel):
    code: str
    message: str
    policy_invalid: bool
    node_ids: tuple[str, ...] = ()


class StepResult(FrozenModel):
    action_kind: str | None
    done: bool
    delivered_image_id: str | None
    protocol_failure: bool
    revision_id: str
    error: ErrorRecord | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    render: RenderResult | None = None
