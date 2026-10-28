from __future__ import annotations

import json
from typing import Any, Callable


from .types import FrozenModel, ViewOverflow


class ViewImage(FrozenModel):
    image_id: str
    roles: tuple[str, ...]
    public_ids: tuple[str, ...] = ()


class AgentView(FrozenModel):
    episode_id: str
    branch_id: str
    revision_id: str
    task_id: str
    payload: dict[str, Any]
    images: tuple[ViewImage, ...]
    visible_node_ids: tuple[str, ...] = ()
    protocol_version: str = "1"

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(image.image_id for image in self.images)

    def text(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def to_messages(
        self, image_resolver: Callable[[str], Any] | None = None
    ) -> list[dict[str, Any]]:
        content = [{"type": "text", "text": self.text()}]
        for image in self.images:
            content.append(
                {
                    "type": "text",
                    "text": "Image roles: "
                    + ", ".join(image.roles)
                    + "; image_ids="
                    + ",".join(image.public_ids or (image.image_id,)),
                }
            )
            content.append(
                {
                    "type": "image",
                    "image": image_resolver(image.image_id) if image_resolver else image.image_id,
                }
            )
        return [{"role": "user", "content": content}]


class AgentViewBuilder:
    def __init__(
        self, max_input_tokens: int = 6144, token_counter: Callable[[str], int] | None = None
    ):
        self.max_input_tokens = max_input_tokens
        self.token_counter = token_counter

    def build(self, environment) -> AgentView:
        state = environment.state
        task = environment.task
        root = state.plan is None
        image_roles = []
        if task.user_request.source_image_id:
            image_roles.append((task.user_request.source_image_id, "original_source"))
        if not root and state.manifest.selected_image_id:
            image_roles.append((state.manifest.selected_image_id, "current"))
        if not root and state.active_crop_id:
            image_roles.append((state.active_crop_id, "active_crop"))
        image_roles.extend(
            (image_id, "appearance_reference")
            for image_id in task.user_request.appearance_image_ids
        )
        images: list[ViewImage] = []
        value_indices = {}
        for image_id, role in image_roles:
            value_id = environment.artifacts.get_ref(image_id).value_id
            if value_id in value_indices:
                index = value_indices[value_id]
                images[index] = images[index].model_copy(
                    update={
                        "roles": (*images[index].roles, role),
                        "public_ids": tuple(
                            dict.fromkeys(
                                (*images[index].public_ids, environment.public_id(image_id))
                            )
                        ),
                    }
                )
            elif len(images) < 4:
                value_indices[value_id] = len(images)
                images.append(
                    ViewImage(
                        image_id=image_id,
                        roles=(role,),
                        public_ids=(environment.public_id(image_id),),
                    )
                )
        request = task.user_request.model_dump(mode="json")
        payload: dict[str, Any] = {
            "request": request,
            "task_mode": task.task_mode,
            "canvas": task.canvas.model_dump(),
            "revision_id": state.manifest.revision_id,
            "budget": state.budget.model_dump(),
            "limits": task.limits.model_dump(),
            "action_schema_version": "1",
        }
        visible = {state.manifest.bindings["user_request"]}
        visible.update(state.last_tool_node_ids)
        if root:
            payload["legal_actions"] = ["PLAN"]
            payload["recent_errors"] = [
                error.model_dump(mode="json") for error in state.errors[-3:]
            ]
        else:
            current_plan = state.plan.model_dump(mode="json")
            payload.update(
                {
                    "tool_result": state.last_tool_result,
                    "plan": current_plan,
                    "active_control_set_id": state.manifest.active_control_set_id,
                    "selected_image_id": state.manifest.selected_image_id,
                    "legal_actions": [
                        "RENDER",
                        "DRAFT_REPAIR",
                        "APPLY",
                        "INSPECT",
                        "READ_NODES",
                        "SELECT",
                        "FINISH",
                    ],
                    "recent_errors": [error.model_dump(mode="json") for error in state.errors[-3:]],
                    "candidates": [candidate.summary() for candidate in state.candidates.values()],
                    "history": [
                        {
                            "image_id": item.image_id,
                            "plan_revision": item.manifest.selected_plan_revision,
                        }
                        for item in state.history.values()
                    ],
                }
            )
            records = environment.public_records()
            payload["node_directory"] = [
                {"id": item["id"], "kind": item["kind"]} for item in records
            ]
            payload["page_cursor"] = environment.make_cursor(0)
            payload["check_spec"] = {
                "version": task.check_spec.version,
                "fallback": task.check_spec.fallback,
                "atoms": [
                    atom.model_dump(mode="json")
                    for atom in task.check_spec.atoms[: task.limits.check_atoms_per_batch]
                ],
                "total_atoms": len(task.check_spec.atoms),
            }
            current_observations = [
                item
                for item in state.observations.values()
                if item.image_id == state.manifest.selected_image_id
            ]
            payload["evidence"] = [
                item.model_dump(mode="json") for item in current_observations[:12]
            ]
            payload["nodes"] = [
                {
                    "logical_node_id": logical_id,
                    "node_version_id": version_id,
                    "kind": state.dag.node(version_id).kind,
                }
                for logical_id, version_id in state.manifest.bindings.items()
                if state.dag.node(version_id).kind
                in {"plan", "entity", "requirement", "binding", "region"}
            ]
            visible.update(state.manifest.bindings.values())
            visible.update(
                state.observation_nodes[item.observation_id] for item in current_observations[:12]
            )
            for item in records:
                state.disclosed_ids.add(item["id"])
            state.disclosed_ids.update(
                atom.check_id for atom in task.check_spec.atoms[: task.limits.check_atoms_per_batch]
            )
        for image in images:
            state.disclosed_ids.add(image.image_id)
            if image.image_id in state.image_nodes:
                visible.add(state.image_nodes[image.image_id])
        payload = environment._translate(payload, True)
        if self.token_counter:

            def count():
                return self.token_counter(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                ) + 256 * len(images)

            while count() > self.max_input_tokens and payload.get("recent_errors"):
                payload["recent_errors"].pop(0)
            while count() > self.max_input_tokens and payload.get("evidence"):
                removed = payload["evidence"].pop(0)
                observation_id = next(
                    (
                        internal
                        for internal, alias in state.public_aliases.items()
                        if alias == removed["observation_id"]
                    ),
                    removed["observation_id"],
                )
                node = state.observation_nodes.get(observation_id)
                if node:
                    visible.discard(node)
            while count() > self.max_input_tokens and payload.get("history"):
                payload["history"].pop(0)
            while count() > self.max_input_tokens and len(payload.get("nodes", [])) > 1:
                payload["nodes"].pop()
            if count() > self.max_input_tokens:
                images = [
                    image
                    for image in images
                    if "active_crop" not in image.roles
                    or "current" in image.roles
                    or "original_source" in image.roles
                ]
            if count() > self.max_input_tokens:
                raise ViewOverflow(
                    "Required public context exceeds the configured controller input limit"
                )
        state.disclosed_ids.update(visible)
        state.last_view_node_ids = tuple(sorted(visible))
        return AgentView(
            episode_id=environment.episode_id,
            branch_id=state.manifest.branch_id,
            revision_id=state.manifest.revision_id,
            task_id=task.task_id,
            payload=payload,
            images=tuple(images),
            visible_node_ids=tuple(sorted(visible)),
        )
