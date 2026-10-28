import json
from dataclasses import dataclass
from pathlib import Path

import torch

from agentic_image.actor.records import ActionSample
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import PublicTask, ViewOverflow
from agentic_image.learning.sampling import stream_seed
from agentic_image.learning.transaction import atomic_json, atomic_torch_save

from .prompt import SYSTEM_PROMPT, actor_messages


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_tasks(path, split=None):
    tasks = [PublicTask.model_validate(row) for row in read_jsonl(path)]
    return [task for task in tasks if split is None or task.split_id == split]


@dataclass
class EpisodeRecord:
    task_id: str
    root_index: int
    suffix_index: int
    action_paths: list[str]
    render_records: list[dict]
    reward: float
    terminal: dict
    environment_path: str


class RolloutCollector:
    def __init__(self, components):
        self.components = components
        self.config = components.config
        self.actor = components.actor
        self.tokenizer = self.actor.processor.tokenizer
        self.system_tokens = (
            len(self.tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)) + 128
        )

    def environment(self, task, episode_id, branch_id="root"):
        return EpisodeEnvironment(
            task,
            self.components.artifacts,
            self.components.compiler,
            self.components.render_bridge,
            self.components.observer,
            episode_id=episode_id,
            branch_id=branch_id,
            versions={"behavior": self.components.render_bridge.behavior_version},
            token_counter=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False)),
            max_input_tokens=self.config["actor"]["max_input_tokens"] - self.system_tokens,
        )

    def restore(self, path):
        return EpisodeEnvironment.restore(
            json.loads(Path(path).read_text()),
            self.components.artifacts,
            self.components.compiler,
            self.components.render_bridge,
            self.components.observer,
            self.components.compiler.load,
            token_counter=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False)),
            max_input_tokens=self.config["actor"]["max_input_tokens"] - self.system_tokens,
        )

    def admit(self, task):
        text = json.dumps(task.user_request.model_dump(mode="json"), ensure_ascii=False)
        required = (
            len(self.tokenizer.encode(text, add_special_tokens=False))
            + self.system_tokens
            + 1024
            + 1024
            + 512
        )
        if required > self.config["actor"]["max_input_tokens"]:
            raise ViewOverflow(
                f"Task {task.task_id} requires a larger context profile before sampling"
            )

    def restore_progress(self, folder, behavior_version):
        paths = [Path(folder) / name for name in ("state.json", "pending.json")]
        candidates = [(path, json.loads(path.read_text())) for path in paths if path.is_file()]
        if not candidates:
            return None
        path, snapshot = max(
            candidates,
            key=lambda item: (item[1]["budget"]["actions"], item[0].name == "pending.json"),
        )
        if snapshot["versions"].get("behavior") != behavior_version:
            raise RuntimeError("An interrupted episode belongs to a different behavior point")
        return self.restore(path)

    def snapshot(self, environment):
        view = environment.view()
        return self.actor.prepare(
            actor_messages(view, self.components.artifacts),
            image_artifact_ids=view.image_ids,
            metadata={
                "task_id": view.task_id,
                "branch_id": view.branch_id,
                "revision_id": view.revision_id,
                "view": view.model_dump(mode="json"),
            },
        )

    def advance(self, environment, folder, seed, behavior_version):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        index = environment.state.budget.actions
        action_path = folder / f"action_{index:02d}.pt"
        pending_path = folder / "pending.json"
        if action_path.is_file():
            sample = ActionSample.from_state_dict(
                torch.load(action_path, map_location="cpu", weights_only=True)
            )
            if sample.behavior_version != behavior_version or sample.seed != seed:
                raise RuntimeError(
                    "An interrupted action belongs to a different behavior or RNG stream"
                )
            if (
                sample.snapshot.metadata.get("revision_id")
                != environment.state.manifest.revision_id
            ):
                raise RuntimeError("An interrupted action belongs to a different public state")
        else:
            snapshot = self.snapshot(environment)
            atomic_json(pending_path, environment.snapshot())
            sample = self.actor.sample(snapshot, seed, behavior_version)
            atomic_torch_save(action_path, sample.state_dict())
        atomic_json(pending_path, environment.snapshot())
        result = environment.step(sample.text, seed=stream_seed(seed, "render"))
        atomic_json(folder / "state.json", environment.snapshot())
        atomic_json(folder / f"result_{index:02d}.json", result.model_dump(mode="json"))
        pending_path.unlink(missing_ok=True)
        return str(action_path.resolve()), result

    def finish(self, environment, folder, root_index, suffix_index, action_paths):
        budget = environment.state.budget
        terminal_reason = "protocol_failure" if environment.state.protocol_failure else "finished"
        reward = self.components.evaluator.evaluate(
            environment.task,
            environment.state.delivered_image_id,
            renders=budget.renders,
            inspects=budget.inspects,
            actions=budget.actions,
            invalids=budget.invalid,
            terminal_reason=terminal_reason,
        )
        terminal = reward.to_dict()
        terminal["delivered_image_id"] = environment.state.delivered_image_id
        terminal["budget"] = budget.model_dump()
        atomic_json(Path(folder) / "terminal_private.json", terminal)
        return EpisodeRecord(
            environment.task.task_id,
            root_index,
            suffix_index,
            action_paths,
            [item.model_dump(mode="json") for item in environment.state.executed_renders],
            reward.reward,
            terminal,
            str((Path(folder) / "state.json").resolve()),
        )

    def collect(self, tasks, round_index, behavior_version, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.components.render_bridge.behavior_version = behavior_version
        root_records = []
        episodes = []
        roots = self.config["rl"]["roots"]
        suffixes = self.config["rl"]["suffixes"]
        rewards = torch.empty(len(tasks), roots, suffixes)
        for task_index, task in enumerate(tasks):
            self.admit(task)
            for root_index in range(roots):
                root_folder = directory / f"task_{task_index:03d}" / f"root_{root_index}"
                root = self.restore_progress(root_folder, behavior_version)
                if root is None:
                    root = self.environment(
                        task, f"episode:{round_index}:{task_index}:{root_index}"
                    )
                if root.task != task:
                    raise RuntimeError("An interrupted root belongs to a different task")
                if root.state.budget.actions == 0:
                    seed = stream_seed(
                        self.config["seed"], round_index, task_index, root_index, "root"
                    )
                    root_path, _ = self.advance(root, root_folder, seed, behavior_version)
                else:
                    root_path = str((root_folder / "action_00.pt").resolve())
                root_records.append(
                    {"task_index": task_index, "root_index": root_index, "path": root_path}
                )
                for suffix_index in range(suffixes):
                    folder = root_folder / f"suffix_{suffix_index}"
                    environment = self.restore_progress(folder, behavior_version)
                    if environment is None:
                        environment = root.fork(f"suffix:{suffix_index}")
                    if environment.task != task or environment.episode_id != root.episode_id:
                        raise RuntimeError(
                            "An interrupted suffix belongs to a different task or root"
                        )
                    paths = [str(path.resolve()) for path in sorted(folder.glob("action_*.pt"))]
                    while not environment.state.done:
                        turn = environment.state.budget.actions
                        seed = stream_seed(
                            self.config["seed"],
                            round_index,
                            task_index,
                            root_index,
                            suffix_index,
                            turn,
                            "controller",
                        )
                        path, _ = self.advance(environment, folder, seed, behavior_version)
                        if path not in paths:
                            paths.append(path)
                    episode = self.finish(environment, folder, root_index, suffix_index, paths)
                    episodes.append({"task_index": task_index, **episode.__dict__})
                    rewards[task_index, root_index, suffix_index] = episode.reward
        batch = {
            "behavior_version": behavior_version,
            "round": round_index,
            "roots": root_records,
            "episodes": episodes,
            "rewards": rewards.tolist(),
            "task_ids": [task.task_id for task in tasks],
        }
        atomic_json(directory / "COMPLETE.json", batch)
        return batch

    def infer(self, task, seed, output_directory):
        self.admit(task)
        folder = Path(output_directory)
        behavior = self.components.render_bridge.behavior_version
        environment = self.restore_progress(folder, behavior)
        if environment is None:
            environment = self.environment(
                task, f"inference:{task.task_id}:{seed}:{folder.resolve()}"
            )
        if environment.task != task:
            raise RuntimeError("An interrupted inference belongs to a different task")
        while not environment.state.done:
            self.advance(
                environment, folder, stream_seed(seed, environment.state.budget.actions), behavior
            )
        return environment
