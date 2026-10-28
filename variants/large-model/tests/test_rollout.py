import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from agentic_image.actor.records import ActionSample, ActorSnapshot
from agentic_image.checks.reward import AtomScore, terminal_reward
from agentic_image.core import (
    ArtifactStore,
    CheckSpec,
    Observation,
    PublicCheckAtom,
    PublicTask,
    RenderResult,
    UserRequest,
)
from agentic_image.core.artifacts import canonical_json, content_id
from agentic_image.learning.objectives import ScoreWeights, hierarchical_rloo
from agentic_image.runtime.rollout import RolloutCollector


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return list(range(max(1, len(text) // 5)))


class TinyActor:
    def __init__(self):
        self.processor = SimpleNamespace(tokenizer=TinyTokenizer())
        self.logit = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = 0

    def prepare(self, messages, image_artifact_ids, metadata):
        ids = torch.ones(1, 1, dtype=torch.long)
        return ActorSnapshot(
            {"input_ids": ids, "attention_mask": ids},
            torch.zeros_like(ids, dtype=torch.bool),
            None,
            torch.zeros(4, 1, 1, dtype=torch.long),
            torch.zeros(1, 1),
            image_artifact_ids,
            metadata,
        )

    def sample(self, snapshot, seed, behavior_version):
        self.calls += 1
        value = int(random.Random(seed).random() < float(self.logit.detach().sigmoid()))
        view = snapshot.metadata["view"]["payload"]
        if view["budget"]["actions"] == 0:
            action = {
                "kind": "PLAN",
                "plan": {"global_text": "Bright red" if value else "Dark red"},
            }
        elif view["budget"]["actions"] == 1:
            action = {
                "kind": "RENDER",
                "control_set_id": view["active_control_set_id"],
                "expected_revision": view["revision_id"],
            }
        else:
            action = {"kind": "FINISH"}
        probabilities = torch.stack((-self.logit, self.logit)).log_softmax(0)
        return ActionSample(
            snapshot,
            torch.tensor([value]),
            probabilities[value : value + 1].detach(),
            torch.ones(1),
            json.dumps(action),
            "eos",
            seed,
            behavior_version,
        )


class TinyCompiler:
    def __init__(self):
        self.bundles = {}

    def compile(self, plan, source_image_id):
        identifier = content_id("bundle", canonical_json({"plan": plan, "source": source_image_id}))
        bundle = {"bundle_id": identifier, "text": plan.global_text}
        self.bundles[identifier] = bundle
        return bundle

    def load(self, identifier):
        return self.bundles[identifier]


class TinyRenderer:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.behavior_version = "test"
        self.jobs = {}
        self.calls = []

    def render(self, bundle, seed, job_id):
        self.calls.append(job_id)
        if job_id not in self.jobs:
            level = (180 if bundle["text"].startswith("Bright") else 30) + seed % 50
            image = self.artifacts.put_image(
                Image.new("RGB", (512, 512), (level, 0, 0)), producer_id=job_id
            )
            self.jobs[job_id] = RenderResult(
                image_id=image.artifact_id, bundle_id=bundle["bundle_id"], trajectory_id=job_id
            )
        return self.jobs[job_id]


class TinyObserver:
    def __init__(self):
        self.failures_remaining = 3

    def check(self, task, image_id):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ConnectionError("Interrupted checker")
        return [
            Observation(
                observation_id="evidence:" + image_id.split(":")[1],
                image_id=image_id,
                check_id="check:red",
                known=True,
            )
        ]


class TinyEvaluator:
    def __init__(self, artifacts):
        self.artifacts = artifacts

    def evaluate(self, task, image_id, **arguments):
        value = self.artifacts.get_image(image_id).getpixel((0, 0))[0] / 255
        return terminal_reward([AtomScore("private:red", value, True, 0.5)], **arguments)


class RolloutIntegrationTests(unittest.TestCase):
    def test_complete_hierarchical_group_resumes_and_supplies_policy_gradient(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ArtifactStore(Path(temporary) / "artifacts")
            components = SimpleNamespace(
                config={
                    "actor": {"max_input_tokens": 6144},
                    "rl": {"roots": 2, "suffixes": 2},
                    "seed": 17,
                },
                artifacts=artifacts,
                actor=TinyActor(),
                compiler=TinyCompiler(),
                render_bridge=TinyRenderer(artifacts),
                observer=TinyObserver(),
                evaluator=TinyEvaluator(artifacts),
            )
            task = PublicTask(
                task_id="task:red",
                family_id="family:red",
                user_request=UserRequest(text="A bright red image"),
                check_spec=CheckSpec(
                    extractor_version="test",
                    atoms=(
                        PublicCheckAtom(check_id="check:red", predicate="The image is bright red"),
                    ),
                ),
            )
            collector = RolloutCollector(components)
            directory = Path(temporary) / "round"
            with self.assertRaises(Exception):
                collector.collect([task], 0, "test", directory)
            batch = RolloutCollector(components).collect([task], 0, "test", directory)
            self.assertEqual(len(batch["roots"]), 2)
            self.assertEqual(len(batch["episodes"]), 4)
            self.assertEqual(len(components.render_bridge.jobs), 4)
            self.assertEqual(components.render_bridge.calls[0], components.render_bridge.calls[1])
            self.assertEqual(components.actor.calls, 10)
            self.assertTrue(all(len(row["action_paths"]) == 2 for row in batch["episodes"]))
            self.assertTrue(all(len(row["render_records"]) == 1 for row in batch["episodes"]))
            advantages = hierarchical_rloo(batch["rewards"])
            weights = ScoreWeights(tasks=1, roots=2, suffixes=2)
            loss = torch.tensor(0.0)
            for row in batch["roots"]:
                sample = ActionSample.from_state_dict(torch.load(row["path"], weights_only=True))
                log_probs = torch.stack(
                    (-components.actor.logit, components.actor.logit)
                ).log_softmax(0)
                loss = (
                    loss
                    - weights.actor_root
                    * advantages.root[0, row["root_index"]]
                    * log_probs[sample.token_ids].sum()
                )
            for row in batch["episodes"]:
                advantage = advantages.suffix[0, row["root_index"], row["suffix_index"]]
                for path in row["action_paths"]:
                    sample = ActionSample.from_state_dict(torch.load(path, weights_only=True))
                    self.assertEqual(sample.behavior_version, "test")
                    log_probs = torch.stack(
                        (-components.actor.logit, components.actor.logit)
                    ).log_softmax(0)
                    loss = (
                        loss - weights.actor_suffix * advantage * log_probs[sample.token_ids].sum()
                    )
            loss.backward()
            self.assertTrue(torch.isfinite(components.actor.logit.grad))
            self.assertGreater(abs(float(components.actor.logit.grad)), 1e-9)
            before = components.actor.logit.detach().clone()
            torch.optim.AdamW([components.actor.logit], lr=0.001).step()
            self.assertFalse(torch.equal(before, components.actor.logit))


if __name__ == "__main__":
    unittest.main()
