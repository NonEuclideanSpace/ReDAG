import tempfile
import unittest

import numpy as np
from PIL import Image

from agentic_image.core import (
    ArtifactStore,
    BoundingBox,
    CheckSpec,
    Entity,
    EpisodeEnvironment,
    Observation,
    PlanSpec,
    PublicCheckAtom,
    PublicTask,
    RegionControl,
    RenderResult,
    UserRequest,
    box_mask,
    make_frame,
    letterbox,
    pool_membership,
    transform_points,
)
from agentic_image.core.artifacts import canonical_json, content_id


class Compiler:
    def __init__(self):
        self.bundles = {}

    def compile(self, plan, source_image_id):
        bundle_id = content_id("bundle", canonical_json({"plan": plan, "source": source_image_id}))
        bundle = {"bundle_id": bundle_id, "source_image_id": source_image_id, "plan": plan}
        self.bundles[bundle_id] = bundle
        return bundle


class Renderer:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.jobs = {}
        self.calls = []

    def render(self, bundle, seed, job_id):
        self.calls.append(job_id)
        if job_id not in self.jobs:
            image = Image.new("RGB", (512, 512), (seed % 255, 80, 120))
            ref = self.artifacts.put_image(image, producer_id=job_id)
            self.jobs[job_id] = RenderResult(
                image_id=ref.artifact_id, bundle_id=bundle["bundle_id"], trajectory_id=job_id
            )
        return self.jobs[job_id]


class Observer:
    def __init__(self):
        self.fail = False

    def check(self, task, image_id):
        if self.fail:
            raise ConnectionError("Temporarily unavailable")
        return [
            Observation(
                observation_id="observation:" + image_id.split(":")[1],
                image_id=image_id,
                check_id=task.check_spec.atoms[0].check_id,
                known=True,
                score=0.9,
                passed=True,
            )
        ]

    def inspect(self, task, image_id, kind, entity_id, region, plan=None):
        return Observation(
            observation_id="inspect:" + kind,
            image_id=image_id,
            origin="plan" if kind == "plan_consistency" else "tool",
            kind=kind,
            known=True,
        )


class CoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.artifacts = ArtifactStore(self.directory.name)
        self.source = self.artifacts.put_image(
            Image.new("RGB", (512, 512), (10, 20, 30))
        ).artifact_id
        self.task = PublicTask(
            task_id="task:1",
            family_id="family:1",
            user_request=UserRequest(text="Make the cup blue", source_image_id=self.source),
            task_mode="edit",
            check_spec=CheckSpec(
                extractor_version="test",
                atoms=(PublicCheckAtom(check_id="check:cup", predicate="The cup is blue"),),
            ),
        )
        self.compiler = Compiler()
        self.renderer = Renderer(self.artifacts)
        self.observer = Observer()
        self.environment = EpisodeEnvironment(
            self.task,
            self.artifacts,
            self.compiler,
            self.renderer,
            self.observer,
            episode_id="episode:1",
        )
        self.plan = PlanSpec(
            global_text="A blue cup",
            entities=(Entity(entity_id="entity:cup", name="cup"),),
            regions=(
                RegionControl(
                    region_id="region:cup",
                    text="A blue cup",
                    entity_ids=("entity:cup",),
                    source_box=BoundingBox(x0=0.1, y0=0.2, x1=0.3, y1=0.5),
                    target_box=BoundingBox(x0=0.6, y0=0.2, x1=0.8, y1=0.5),
                ),
            ),
        )

    def tearDown(self):
        self.directory.cleanup()

    def submit_plan(self):
        return self.environment.step({"kind": "PLAN", "plan": self.plan.model_dump(mode="json")})

    def render(self, seed=1):
        state = self.environment.state
        return self.environment.step(
            {
                "kind": "RENDER",
                "control_set_id": state.manifest.active_control_set_id,
                "expected_revision": state.manifest.revision_id,
            },
            seed,
        )

    def test_public_root_fork_and_finish_semantics(self):
        root = self.environment.view()
        self.assertNotIn("check_spec", root.payload)
        invalid = self.environment.step({"kind": "FINISH"})
        self.assertEqual(invalid.error.code, "PLAN_REQUIRED")
        self.submit_plan()
        self.assertIn("check_spec", self.environment.view().payload)
        first = self.environment.fork("branch:1")
        second = self.environment.fork("branch:2")
        result = first.step({"kind": "FINISH"})
        self.assertEqual(result.delivered_image_id, self.source)
        result = second.step({"kind": "FINISH", "image_id": None})
        self.assertIsNone(result.delivered_image_id)
        self.assertEqual(first.state.budget.renders, 0)
        self.assertFalse(self.environment.state.done)

    def test_repair_preserves_source_target_masks_and_selection(self):
        self.submit_plan()
        first = self.render()
        evidence_id = next(
            key
            for key, item in self.environment.state.observations.items()
            if item.image_id == first.render.image_id
        )
        state = self.environment.state
        old_version = state.manifest.bindings["region:cup"]
        draft = self.environment.step(
            {
                "kind": "DRAFT_REPAIR",
                "source_image_id": first.render.image_id,
                "evidence_ids": [evidence_id],
                "suspect_node_ids": ["region:cup"],
                "patch_ops": [
                    {
                        "target_logical_node_id": "region:cup",
                        "expected_old_version_id": old_version,
                        "op": "replace",
                        "field_path": ["text"],
                        "typed_value_or_artifact_id": "A bright blue cup",
                    }
                ],
                "instruction": "Make the cup brighter",
            }
        )
        self.assertIsNone(draft.error)
        self.assertEqual(len(draft.output["candidates"]), 3)
        local = next(
            candidate
            for candidate in self.environment.state.candidates.values()
            if candidate.scope == "local"
        )
        self.assertNotEqual(local.plan.regions[0].source_box, local.plan.regions[0].target_box)
        result = self.environment.step(
            {
                "kind": "APPLY",
                "candidate_id": local.candidate_id,
                "expected_revision": state.manifest.revision_id,
            },
            seed=2,
        )
        self.assertIsNone(result.error)
        self.assertEqual(len(self.environment.state.executed_renders), 2)
        self.environment.step({"kind": "SELECT", "image_id": first.render.image_id})
        self.assertEqual(self.environment.state.plan.regions[0].text, "A blue cup")
        third = self.render(seed=3)
        bundle = self.compiler.bundles[third.render.bundle_id]
        self.assertEqual(bundle["source_image_id"], first.render.image_id)
        self.assertEqual(len(self.environment.state.executed_renders), 3)

    def test_service_retry_restores_state_and_render_identity(self):
        self.submit_plan()
        state = self.environment.state
        action = {
            "kind": "RENDER",
            "control_set_id": state.manifest.active_control_set_id,
            "expected_revision": state.manifest.revision_id,
        }
        before = state.budget
        self.observer.fail = True
        with self.assertRaises(Exception):
            self.environment.step(action, seed=7)
        self.assertEqual(self.environment.state.budget, before)
        self.observer.fail = False
        result = self.environment.step(action, seed=7)
        self.assertIsNone(result.error)
        self.assertEqual(self.renderer.calls[0], self.renderer.calls[1])
        self.assertEqual(len(self.renderer.jobs), 1)
        self.assertEqual(len(self.environment.state.executed_renders), 1)
        restored = EpisodeEnvironment.restore(
            self.environment.snapshot(),
            self.artifacts,
            self.compiler,
            self.renderer,
            self.observer,
            self.compiler.bundles.__getitem__,
        )
        self.assertEqual(
            restored.step({"kind": "FINISH"}).delivered_image_id, result.render.image_id
        )

    def test_stale_patch_and_sibling_references_are_rejected(self):
        self.submit_plan()
        first = self.environment.fork("branch:first")
        second = self.environment.fork("branch:second")
        state = first.state
        result = first.step(
            {
                "kind": "RENDER",
                "control_set_id": state.manifest.active_control_set_id,
                "expected_revision": state.manifest.revision_id,
            },
            seed=8,
        )
        rejected = second.step({"kind": "SELECT", "image_id": result.render.image_id})
        self.assertEqual(rejected.error.code, "UNKNOWN_ID")
        self.assertNotIn(result.render.image_id, second.state.history)

    def test_registered_geometry_and_tiny_region(self):
        frame = make_frame((201, 403), (512, 512), "frame:1")
        image = letterbox(Image.new("RGB", (403, 201), "white"), frame)
        self.assertEqual(image.size, (512, 512))
        points = np.array([[0.5, 0.5], [402.5, 200.5]])
        np.testing.assert_allclose(
            transform_points(transform_points(points, frame), frame, inverse=True),
            points,
            atol=1e-12,
        )
        mask = box_mask(BoundingBox(x0=0.101, y0=0.201, x1=0.102, y1=0.202), (512, 512))
        membership, fallback = pool_membership(mask)
        self.assertTrue(fallback)
        self.assertEqual(int(membership.sum()), 1)


if __name__ == "__main__":
    unittest.main()
