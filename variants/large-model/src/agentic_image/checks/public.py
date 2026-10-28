import json
import unicodedata
from uuid import uuid4

from agentic_image.core.types import (
    CheckSpec,
    Observation,
    PublicCheckAtom,
    PublicTask,
    RequestSpan,
    UserRequest,
)

from .evidence import StructuredChecks


DECOMPOSE_PROMPT = (
    "Split the original user request into visually checkable requirements using only the "
    "request and its attachments. Return one JSON object with an atoms array. Each atom has "
    "start, end, quote, predicate, expected_value, and confidence. start and end are Unicode "
    "code-point offsets in the NFC request, with an exclusive end. quote must equal that "
    "substring exactly. Separate object counts, attributes, spatial relations, written text, "
    "reference appearance, and preservation. predicate must be a complete self-contained "
    "property. Do not invent objects, backgrounds, styles, or requirements. Retain ambiguity "
    "as unknown instead of resolving it. Do not use information outside the request.\n"
    "Original request: {request}"
)


class PublicTaskBuilder:
    def __init__(self, judge, artifacts):
        self.judge = judge
        self.artifacts = artifacts

    def build(
        self,
        *,
        task_id,
        text,
        source_image_id=None,
        appearance_image_ids=(),
        controls=None,
        family_id=None,
        split_id="train",
        source_dataset="user",
        license_record=None,
    ):
        text = unicodedata.normalize("NFC", text)
        image_ids = ([source_image_id] if source_image_id else []) + list(appearance_image_ids)
        images = [self.artifacts.get_image(identifier) for identifier in image_ids]
        prompt = DECOMPOSE_PROMPT.format(request=text)
        fallback = False
        spans = []
        atoms = []
        for attempt in range(2):
            try:
                result = self.judge.generate_json(prompt, images)
                if not isinstance(result.get("atoms"), list) or not result["atoms"]:
                    raise ValueError("Expected a nonempty atoms array")
                spans = []
                atoms = []
                for index, item in enumerate(result["atoms"]):
                    span = RequestSpan(
                        span_id=f"span:{index:04d}",
                        start=item["start"],
                        end=item["end"],
                        text=item["quote"],
                    )
                    if span.end > len(text) or text[span.start : span.end] != span.text:
                        raise ValueError("Quoted span does not match the original request")
                    spans.append(span)
                    atoms.append(
                        PublicCheckAtom(
                            check_id=f"check:{index:04d}",
                            raw_span_ids=(span.span_id,),
                            predicate=item["predicate"],
                            expected_value=item.get("expected_value"),
                            public_reference_ids=tuple(appearance_image_ids),
                            confidence=float(item.get("confidence", 1.0)),
                        )
                    )
                break
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                if attempt == 0:
                    prompt = DECOMPOSE_PROMPT.format(request=text) + (
                        "\nReturn valid JSON only. Recheck every quoted span and Unicode offset."
                    )
                else:
                    fallback = True
                    spans = [RequestSpan(span_id="span:0000", start=0, end=len(text), text=text)]
                    atoms = [
                        PublicCheckAtom(
                            check_id="check:0000",
                            raw_span_ids=("span:0000",),
                            predicate="The image satisfies the complete original request.",
                            expected_value=text,
                        )
                    ]
        request = UserRequest(
            text=text,
            source_image_id=source_image_id,
            appearance_image_ids=tuple(appearance_image_ids),
            spans=tuple(spans),
            controls=controls or {},
        )
        return PublicTask(
            task_id=task_id,
            family_id=family_id or task_id,
            split_id=split_id,
            source_dataset=source_dataset,
            license_record=license_record or {},
            user_request=request,
            task_mode="edit" if source_image_id else "generate",
            check_spec=CheckSpec(
                extractor_version=self.judge.config.revision, atoms=tuple(atoms), fallback=fallback
            ),
        )


class PublicObserver:
    def __init__(self, judge, artifacts, tools=None):
        self.judge = judge
        self.artifacts = artifacts
        self.tools = tools
        self.cache = {}
        self.structured = StructuredChecks(tools)

    def check(self, task, image_id):
        observations = []
        for atom in task.check_spec.atoms:
            key = (
                image_id,
                task.task_id,
                task.check_spec.version,
                atom.check_id,
                json.dumps(self.judge.thresholds, sort_keys=True),
                self.judge.require_calibration,
            )
            if key in self.cache:
                observations.append(self.cache[key])
                continue
            references = list(atom.public_reference_ids)
            if not references and task.user_request.source_image_id:
                references = [task.user_request.source_image_id]
            reference = references[:1]
            ids = [image_id, *reference]
            images = [self.artifacts.get_image(identifier) for identifier in ids]
            kind = (
                atom.expected_value.get("kind", "semantic")
                if isinstance(atom.expected_value, dict)
                else "semantic"
            )
            structured = self.structured.evaluate(
                atom.check_id,
                image_id,
                kind,
                atom.expected_value,
                reference_id=reference[0] if reference else None,
            )
            score = self.judge.check(
                atom_id=atom.check_id,
                request=task.user_request.text,
                property_text=f"{atom.predicate}\nExpected: {json.dumps(atom.expected_value)}",
                images=images,
                roles="candidate" + (", public reference or original source" if reference else ""),
                kind=kind,
                structured=structured,
            )
            observation = Observation(
                observation_id=f"observation:{uuid4().hex}",
                image_id=image_id,
                check_id=atom.check_id,
                known=score.known,
                score=score.value,
                passed=score.passed,
                entity_ids=atom.entity_refs,
                artifact_ids=score.evidence,
                frame_id=self.artifacts.get_ref(image_id).frame.frame_id
                if self.artifacts.get_ref(image_id).frame
                else None,
                details={
                    "predicate": atom.predicate,
                    "threshold": score.threshold,
                    "judge_version": self.judge.config.revision,
                    "route": kind,
                },
            )
            self.cache[key] = observation
            observations.append(observation)
        return observations

    def inspect(self, task, image_id, kind, entity_id=None, region=None, plan=None):
        from agentic_image.core.types import BoundingBox

        if kind in {"recheck", "public_check", "checks"}:
            return self.check(task, image_id)
        image = self.artifacts.get_image(image_id)
        details = {}
        artifact_ids = ()
        known = True
        if kind == "crop":
            box = region if isinstance(region, BoundingBox) else BoundingBox.model_validate(region)
            crop = image.crop(
                (
                    int(box.x0 * image.width),
                    int(box.y0 * image.height),
                    max(1, round(box.x1 * image.width)),
                    max(1, round(box.y1 * image.height)),
                )
            )
            reference = self.artifacts.put_image(
                crop, producer_id=image_id, metadata={"box": box.model_dump()}
            )
            artifact_ids = (reference.artifact_id,)
            details = {"box": box.model_dump()}
        elif kind == "plan_consistency":
            if plan is None:
                raise ValueError("Plan consistency requires the current public plan")
            property_text = "The image agrees with this public plan: " + plan.model_dump_json()
            score = self.judge.check(
                atom_id="plan_consistency",
                request=task.user_request.text,
                property_text=property_text,
                images=[image],
                roles="candidate",
                kind="semantic",
            )
            return Observation(
                observation_id=f"observation:{uuid4().hex}",
                image_id=image_id,
                kind=kind,
                origin="plan",
                known=score.known,
                passed=score.passed,
                score=score.value,
                details={"plan": plan.model_dump(mode="json")},
            )
        elif kind in {"detect", "segment", "ocr", "features"}:
            if self.tools is None:
                raise RuntimeError("The configured visual tool service is unavailable")
            arguments = {
                "image_id": image_id,
                "entity_id": entity_id,
                "region": region.model_dump() if hasattr(region, "model_dump") else region,
            }
            if kind == "detect":
                entities = {entity.entity_id: entity for entity in plan.entities} if plan else {}
                if entity_id not in entities:
                    raise ValueError("Detection must name a public plan entity")
                arguments["text"] = entities[entity_id].name
            details = self.tools.call(kind, **arguments)
            known = details.get("known", True)
            artifact_ids = tuple(details.get("artifact_ids", ()))
        else:
            raise ValueError(f"Unknown inspection kind: {kind}")
        return Observation(
            observation_id=f"observation:{uuid4().hex}",
            image_id=image_id,
            kind=kind,
            origin="tool",
            known=known,
            artifact_ids=artifact_ids,
            entity_ids=(entity_id,) if entity_id else (),
            details=details,
        )
