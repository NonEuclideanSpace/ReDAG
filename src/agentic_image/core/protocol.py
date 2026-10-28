from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from .types import BoundingBox, ContractError, FrozenModel, PatchOperation, PlanSpec


class ActionBase(FrozenModel):
    reason: str = ""


class PlanAction(ActionBase):
    kind: Literal["PLAN"]
    plan: PlanSpec


class RenderAction(ActionBase):
    kind: Literal["RENDER"]
    control_set_id: str
    expected_revision: str


class DraftRepairAction(ActionBase):
    kind: Literal["DRAFT_REPAIR"]
    source_image_id: str
    evidence_ids: tuple[str, ...]
    suspect_node_ids: tuple[str, ...]
    patch_ops: tuple[PatchOperation, ...] = ()
    instruction: str
    scope: Literal["automatic", "local", "expanded", "global", "custom", "retry"] = "automatic"

    @model_validator(mode="after")
    def validate_patch(self):
        if self.scope != "retry" and not self.patch_ops:
            raise ValueError("A repair must provide at least one typed patch")
        if self.scope == "retry" and self.patch_ops:
            raise ValueError("A retry cannot change its historical condition")
        return self


class ApplyAction(ActionBase):
    kind: Literal["APPLY"]
    candidate_id: str
    expected_revision: str


class InspectAction(ActionBase):
    kind: Literal["INSPECT"]
    image_id: str
    inspect_kind: Literal["crop", "detect", "segment", "recheck", "plan_consistency"] = Field(
        alias="inspection"
    )
    entity_id: str | None = None
    region: BoundingBox | None = None

    @model_validator(mode="after")
    def validate_arguments(self):
        if self.inspect_kind in {"crop", "segment"} and self.region is None:
            raise ValueError("Crop and segment inspections require an explicit normalized region")
        if self.inspect_kind == "detect" and self.entity_id is None:
            raise ValueError("Detection requires a named public entity")
        return self


class ReadNodesAction(ActionBase):
    kind: Literal["READ_NODES"]
    node_ids: tuple[str, ...] = ()
    page_cursor: str | None = None
    fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_source(self):
        if bool(self.node_ids) == bool(self.page_cursor):
            raise ValueError("READ_NODES must specify node_ids or page_cursor")
        if len(self.node_ids) > 16:
            raise ValueError("A page can contain at most 16 records")
        return self


class SelectAction(ActionBase):
    kind: Literal["SELECT"]
    image_id: str


class FinishAction(ActionBase):
    kind: Literal["FINISH"]
    image_id: str | None = None

    @property
    def image_was_provided(self) -> bool:
        return "image_id" in self.model_fields_set


Action = Annotated[
    PlanAction
    | RenderAction
    | DraftRepairAction
    | ApplyAction
    | InspectAction
    | ReadNodesAction
    | SelectAction
    | FinishAction,
    Field(discriminator="kind"),
]
ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(raw: str | dict) -> Action:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    try:
        data = (
            json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
            if isinstance(raw, str)
            else raw
        )
        if not isinstance(data, dict):
            raise ValueError("An action must be one JSON object")
        return ACTION_ADAPTER.validate_python(data)
    except (ValidationError, ValueError, TypeError) as error:
        raise ContractError("INVALID_ACTION", str(error)) from error


def action_schema() -> dict:
    return ACTION_ADAPTER.json_schema()
