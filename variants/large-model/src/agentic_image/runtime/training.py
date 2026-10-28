import json
import random
from pathlib import Path

import torch

from agentic_image.actor.records import ActionSample, ActorSnapshot
from agentic_image.core.types import PlanSpec
from agentic_image.learning.objectives import ScoreWeights, hierarchical_rloo, phase_action_weights
from agentic_image.learning.optim import (
    make_optimizer,
    make_rl_scheduler,
    make_sft_scheduler,
    sum_policy_gradients,
)
from agentic_image.learning.sampling import PhaseBalancedSampler, stream_seed
from agentic_image.learning.transaction import JointTransaction, TrainingComponent, atomic_json
from agentic_image.render.sde import FlowSDE, gaussian_mean_kl
from agentic_image.render.training import FlowMatchingExample, backward_flow_matching
from agentic_image.render.types import ConditionBundle, RenderTrace

from .rollout import RolloutCollector, read_jsonl, read_tasks


def load_action(path):
    return ActionSample.from_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def append_metrics(path, metrics):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(metrics, allow_nan=False) + "\n")
    print(json.dumps(metrics, allow_nan=False), flush=True)


def prune_checkpoints(directory, keep=2, interval=25):
    directory = Path(directory)
    committed = sorted(directory.glob("*/COMMITTED.json"), key=lambda path: path.stat().st_mtime)
    for index, manifest in enumerate(committed):
        if index < len(committed) - keep and (index + 1) % interval:
            for path in manifest.parent.glob("*.pt"):
                path.unlink()
        elif (manifest.parent / "before.pt").exists():
            (manifest.parent / "before.pt").unlink()


def resolve_bundle(record, components):
    path = record.get("bundle_path")
    if path and Path(path).is_file():
        return ConditionBundle.load(path)
    if "recipe" in record or "condition_recipe" in record:
        recipe = record.get("recipe", record.get("condition_recipe"))
        bundle = components.compiler.compile(
            PlanSpec.model_validate(recipe["plan"]), recipe.get("source_image_id")
        )
        if record.get("bundle_id") and bundle.bundle_id != record["bundle_id"]:
            raise RuntimeError("Frozen condition reconstruction changed the training example")
        return bundle
    return components.compiler.load(record["bundle_id"])


def train_actor_sft(components, stage="actor", updates=None, resume=False):
    from agentic_image.data.audit import require_pair_audit

    config = components.config
    if stage != "protocol":
        require_pair_audit(config["data"]["paired_audit"])
    actor = components.load_actor()
    records = read_jsonl(config["data"]["actor_records"])
    records = [
        row for row in records if row.get("verified") and row.get("split_id", "train") == "train"
    ]
    if not records:
        raise ValueError("No verified training actions are available")
    protocol = stage == "protocol"
    if protocol:
        records = [row for row in records if row.get("phase", 0) in {0, 3}]
        if not records:
            raise ValueError("Protocol SFT requires verified planning or protocol recovery records")
    total = updates or config["sft"][f"{stage}_updates"]
    parameters = actor.policy_parameters()
    optimizer = make_optimizer(parameters, config["sft"]["learning_rate"], "sft")
    scheduler = make_sft_scheduler(optimizer, total)
    component = TrainingComponent(
        "actor", parameters, optimizer, actor.adapter_state, actor.restore_adapter_state, scheduler
    )
    folder = components.directory / f"sft_{stage}"
    transaction = JointTransaction(folder / "checkpoints", [component])
    quotas = (
        tuple(config["sft"].get("protocol_phase_counts", (16, 0, 0, 4)))
        if protocol
        else tuple(config["sft"]["phase_counts"])
    )
    if sum(quotas) != config["sft"]["actor_batch"]:
        raise ValueError("Actor phase quotas must fill the declared global batch")
    sampler = PhaseBalancedSampler(records, quotas=quotas, seed=config["seed"])
    start = 0
    if resume:
        state = transaction.load_committed()
        start = state["data_cursor"]["step"]
        if sampler is not None:
            sampler.load_state_dict(state["data_cursor"]["sampler"])
    for step in range(start, total):
        cursor = {"step": step, "sampler": sampler.state_dict() if sampler else None}
        transaction.begin(f"{stage}:{step}", cursor)
        try:
            batch = sampler.next_batch()
            if protocol:
                weights = [1 / len(batch)] * len(batch)
            else:
                weights = phase_action_weights(
                    [row["phase"] for row in batch],
                    config["sft"]["phase_counts"],
                    config["sft"]["phase_weights"],
                )
            optimizer.zero_grad(set_to_none=True)
            metrics = []
            for record, weight in zip(batch, weights):
                snapshot = ActorSnapshot.load(record["snapshot_path"])
                metrics.append(
                    actor.sft_backward(
                        snapshot, record["target_ids"], record["target_mask"], weight
                    )
                )
            sum_policy_gradients(parameters)
            cursor = {"step": step + 1, "sampler": sampler.state_dict() if sampler else None}
            committed = transaction.step_and_commit(f"{stage}:{step + 1}", data_cursor=cursor)
            append_metrics(
                folder / "metrics.jsonl",
                {
                    "step": step + 1,
                    "stage": stage,
                    "loss": sum(item.loss for item in metrics),
                    "gradients": committed["gradients"],
                },
            )
            if (step + 1) % 25 == 0 or step + 1 == total:
                actor.save_policy(folder / "policy")
            prune_checkpoints(folder / "checkpoints")
        except Exception:
            if transaction.before is not None:
                transaction.abort("actor_sft_failed")
            raise
    return str((folder / "policy").resolve())


def train_renderer_sft(components, updates=None, resume=False):
    from agentic_image.data.audit import require_pair_audit

    config = components.config
    require_pair_audit(config["data"]["paired_audit"])
    renderer = components.load_renderer()
    records = read_jsonl(config["data"]["renderer_records"])
    records = [
        row
        for row in records
        if (row.get("verified") or row.get("verification_provenance"))
        and row.get("split_id", "train") == "train"
    ]
    if not records:
        raise ValueError("No aligned renderer training examples are available")
    total = updates or config["sft"]["renderer_updates"]
    parameters = renderer.policy_parameters()
    optimizer = make_optimizer(parameters, config["sft"]["learning_rate"], "sft")
    scheduler = make_sft_scheduler(optimizer, total)
    component = TrainingComponent(
        "renderer",
        parameters,
        optimizer,
        renderer.policy_state,
        renderer.load_policy_state,
        scheduler,
    )
    folder = components.directory / "sft_renderer"
    transaction = JointTransaction(folder / "checkpoints", [component])
    start = transaction.load_committed()["data_cursor"]["step"] if resume else 0
    batch_size = config["sft"]["renderer_batch"]
    region_counts = tuple(config["sft"].get("renderer_region_counts", (6, 3, 3)))
    if sum(region_counts) != batch_size:
        raise ValueError("Renderer region quotas must fill the declared global batch")
    pools = {count: {} for count in range(3)}
    for record in records:
        count = record.get("region_count")
        if count not in pools:
            raise ValueError("Renderer records must declare their number of active regions")
        pools[count].setdefault(record["family_id"], []).append(record)
    if any(quota and not pools[count] for count, quota in enumerate(region_counts)):
        raise ValueError("A required renderer region bucket has no verified training examples")
    for step in range(start, total):
        transaction.begin(f"renderer-sft:{step}", {"step": step})
        try:
            generator = random.Random(stream_seed(config["seed"], "renderer_sft", step))
            batch = []
            for count, quota in enumerate(region_counts):
                families = sorted(pools[count])
                batch.extend(
                    generator.choice(pools[count][generator.choice(families)]) for _ in range(quota)
                )
            generator.shuffle(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for record in batch:
                bundle = resolve_bundle(record, components)
                target = components.encoders.encode_image(
                    components.compiler.image(record["target_image_id"])
                )
                valid = torch.from_numpy(components.artifacts.get_array(record["valid_mask_id"]))
                edit = (
                    torch.from_numpy(components.artifacts.get_array(record["edit_mask_id"]))
                    if record.get("edit_mask_id")
                    else None
                )
                example = FlowMatchingExample(record["sample_id"], bundle, target, valid, edit)
                metrics = backward_flow_matching(
                    renderer,
                    example,
                    step * batch_size // len(records),
                    step,
                    batch_size,
                    config["sft"]["text_dropout"],
                )
                loss += metrics["loss"]
            sum_policy_gradients(parameters)
            committed = transaction.step_and_commit(
                f"renderer-sft:{step + 1}", data_cursor={"step": step + 1}
            )
            append_metrics(
                folder / "metrics.jsonl",
                {"step": step + 1, "loss": loss, "gradients": committed["gradients"]},
            )
            if (step + 1) % 25 == 0 or step + 1 == total:
                renderer.save_adapters(folder / "adapters")
            prune_checkpoints(folder / "checkpoints")
        except Exception:
            if transaction.before is not None:
                transaction.abort("renderer_sft_failed")
            raise
    return str((folder / "adapters").resolve())


class UpdateProbe:
    def __init__(self, components, batch, actor_active=True, renderer_active=True):
        self.components = components
        self.actions = []
        self.transitions = []
        if actor_active:
            paths = [row["path"] for row in batch["roots"]]
            paths.extend(path for row in batch["episodes"] for path in row["action_paths"])
            for path in paths[:16]:
                sample = load_action(path)
                self.actions.append((sample, components.actor.distribution_probe(sample)))
        if renderer_active:
            for episode in batch["episodes"]:
                for record in episode["render_records"]:
                    trace = RenderTrace.load(record["trajectory_id"])
                    bundle = components.compiler.load(record["bundle_id"])
                    for chosen in trace.replay_steps:
                        step = chosen.step_index
                        state = trace.states[step]
                        transition = FlowSDE(trace.sigmas, trace.eta).transition(step)
                        with torch.no_grad():
                            velocity = components.renderer.predict_velocity(
                                bundle, state, trace.sigmas[step]
                            )
                            mean = transition.mean(state.to(velocity.device), velocity).cpu()
                        self.transitions.append(
                            (bundle, state, trace.sigmas[step], transition, mean)
                        )
                        if len(self.transitions) == 16:
                            break
                    if len(self.transitions) == 16:
                        break
                if len(self.transitions) == 16:
                    break

    def __call__(self):
        actor_values = []
        renderer_values = []
        for sample, old in self.actions:
            current = self.components.actor.distribution_probe(sample, positions=tuple(old))
            for position in old:
                actor_values.append(
                    float((old[position].exp() * (old[position] - current[position])).sum())
                )
        for bundle, state, sigma, transition, old in self.transitions:
            with torch.no_grad():
                velocity = self.components.renderer.predict_velocity(bundle, state, sigma)
                mean = transition.mean(state.to(velocity.device), velocity)
                renderer_values.append(
                    float(gaussian_mean_kl(mean, old.to(mean.device), transition.std))
                )
        return {
            "actor_kl": sum(actor_values) / max(1, len(actor_values)),
            "renderer_kl": sum(renderer_values) / max(1, len(renderer_values)),
            "actor_kl_max": max(actor_values, default=0.0),
            "renderer_kl_max": max(renderer_values, default=0.0),
        }


def train_rl(
    components, mode="joint", rounds=None, resume=False, initialize_from=None, pilot=False
):
    config = components.config
    components.load_all()
    actor = components.actor
    renderer = components.renderer
    if not actor.has_reference or not renderer._reference_exists:
        raise RuntimeError("RL requires independent frozen SFT references for both models")
    if not components.judge.thresholds:
        raise RuntimeError("Calibrate the actual NF4 judge before collecting RL rewards")
    components.judge.require_calibration = True
    sampler_path = components.directory / "sampler_selection.json"
    selection_path = sampler_path
    if not selection_path.is_file() and initialize_from:
        selection_path = Path(initialize_from).parent.parent / "sampler_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("Run the sampler quality comparison before RL")
    selection = json.loads(selection_path.read_text())
    if not selection.get("accepted", False):
        raise RuntimeError("The selected stochastic sampler has not passed its quality comparison")
    for key in ("steps", "eta", "cfg_scale"):
        if selection.get(key) != config["renderer"][key]:
            raise ValueError(f"Active renderer {key} differs from the validated sampler")
    if selection.get("renderer_policy") != config["models"].get("renderer_policy"):
        raise ValueError("Sampler selection belongs to a different SFT renderer checkpoint")
    if selection_path != sampler_path:
        atomic_json(sampler_path, selection)
    tasks = read_tasks(config["data"]["public_tasks"], split="train")
    if len(tasks) < config["rl"]["tasks"]:
        raise ValueError("The RL pool cannot provide a complete task batch")
    for task in tasks:
        for key, expected in config["episode"].items():
            if getattr(task.limits, key) != expected:
                raise ValueError(
                    f"Task {task.task_id} has a different {key} budget from the fixed policy-gradient denominator"
                )
    total = rounds or config["rl"]["agent_rounds" if mode == "agent" else "joint_rounds"]
    actor_parameters = actor.policy_parameters()
    renderer_parameters = renderer.policy_parameters()
    actor_optimizer = make_optimizer(
        actor_parameters, config["rl"]["actor_learning_rate"], epsilon=config["rl"]["epsilon"]
    )
    renderer_optimizer = make_optimizer(
        renderer_parameters, config["rl"]["renderer_learning_rate"], epsilon=config["rl"]["epsilon"]
    )
    warmup = 0 if pilot else config["rl"]["warmup_rounds"]
    actor_scheduler = make_rl_scheduler(actor_optimizer, warmup)
    renderer_scheduler = make_rl_scheduler(renderer_optimizer, warmup)
    actor_component = TrainingComponent(
        "actor",
        actor_parameters,
        actor_optimizer,
        actor.adapter_state,
        actor.restore_adapter_state,
        actor_scheduler,
        mode != "renderer",
    )
    renderer_component = TrainingComponent(
        "renderer",
        renderer_parameters,
        renderer_optimizer,
        renderer.policy_state,
        renderer.load_policy_state,
        renderer_scheduler,
        mode != "agent",
    )
    folder = components.directory / f"rl_{mode}"
    transaction = JointTransaction(folder / "checkpoints", [actor_component, renderer_component])
    start = 0
    behavior = "sft-pair"
    if initialize_from:
        previous = JointTransaction(
            initialize_from, [actor_component, renderer_component]
        ).load_committed()
        behavior = previous["behavior_version"]
        actor_component.active = mode != "renderer"
        renderer_component.active = mode != "agent"
        for optimizer, key in (
            (actor_optimizer, "actor_learning_rate"),
            (renderer_optimizer, "renderer_learning_rate"),
        ):
            for group in optimizer.param_groups:
                group["lr"] = config["rl"][key]
                group["initial_lr"] = config["rl"][key]
        actor_scheduler = make_rl_scheduler(actor_optimizer, warmup)
        renderer_scheduler = make_rl_scheduler(renderer_optimizer, warmup)
        actor_component.scheduler = actor_scheduler
        renderer_component.scheduler = renderer_scheduler
        renderer_optimizer.state.clear()
    if resume:
        state = transaction.load_committed()
        start = state["data_cursor"]["round"]
        behavior = state["behavior_version"]
    collector = RolloutCollector(components)
    weights = ScoreWeights(
        config["rl"]["tasks"],
        config["rl"]["roots"],
        config["rl"]["suffixes"],
        config["episode"]["max_actions"],
        config["actor"]["max_new_tokens"],
        config["episode"]["max_renders"],
    )
    zero_rounds = 0
    for round_index in range(start, total):
        generator = random.Random(stream_seed(config["seed"], round_index, "tasks"))
        batch_tasks = generator.sample(tasks, weights.tasks)
        transaction.begin(
            behavior, {"round": round_index}, metadata={"mode": mode, "config": config}
        )
        try:
            batch_folder = folder / "batches" / f"{round_index:05d}"
            complete_path = batch_folder / "COMPLETE.json"
            if complete_path.is_file():
                batch = json.loads(complete_path.read_text())
                if batch["behavior_version"] != behavior or batch.get("task_ids") != [
                    task.task_id for task in batch_tasks
                ]:
                    raise RuntimeError(
                        "Saved rollout batch belongs to a different behavior point or task group"
                    )
            else:
                batch = collector.collect(batch_tasks, round_index, behavior, batch_folder)
            transaction.set_batch(str(batch_folder.resolve()), {"round": round_index + 1})
            advantages = hierarchical_rloo(batch["rewards"])
            actor_component.active = mode != "renderer"
            renderer_component.active = mode != "agent" and any(
                row["render_records"] for row in batch["episodes"]
            )
            actor_optimizer.zero_grad(set_to_none=True)
            renderer_optimizer.zero_grad(set_to_none=True)
            probe = UpdateProbe(
                components, batch, actor_component.active, renderer_component.active
            )
            actor_loss = 0.0
            renderer_loss = 0.0
            if actor_component.active:
                for record in batch["roots"]:
                    sample = load_action(record["path"])
                    if sample.behavior_version != behavior:
                        raise RuntimeError("Root action was not sampled at this behavior point")
                    metrics = actor.rl_backward(
                        sample,
                        float(advantages.root[record["task_index"], record["root_index"]]),
                        weights.actor_root,
                        config["rl"]["beta_actor"],
                    )
                    actor_loss += metrics.loss
            for episode in batch["episodes"]:
                advantage = float(
                    advantages.suffix[
                        episode["task_index"], episode["root_index"], episode["suffix_index"]
                    ]
                )
                if actor_component.active:
                    for path in episode["action_paths"]:
                        sample = load_action(path)
                        if sample.behavior_version != behavior:
                            raise RuntimeError(
                                "Controller action was not sampled at this behavior point"
                            )
                        metrics = actor.rl_backward(
                            sample, advantage, weights.actor_suffix, config["rl"]["beta_actor"]
                        )
                        actor_loss += metrics.loss
                if renderer_component.active:
                    for record in episode["render_records"]:
                        trace = RenderTrace.load(record["trajectory_id"])
                        bundle = components.compiler.load(record["bundle_id"])
                        metrics = renderer.backward_replay(
                            bundle,
                            trace,
                            advantage,
                            weights.renderer,
                            config["rl"]["beta_renderer"],
                            expected_behavior_version=behavior,
                        )
                        renderer_loss += metrics["loss"]
            if actor_component.active:
                sum_policy_gradients(actor_parameters)
            if renderer_component.active:
                sum_policy_gradients(renderer_parameters)
            behavior = f"{mode}:{round_index + 1}"
            committed = transaction.step_and_commit(
                behavior,
                probe=probe,
                limits={
                    "actor_kl": config["rl"]["probe_actor_kl"],
                    "renderer_kl": config["rl"]["probe_renderer_kl"],
                },
                max_gradient_norm=config["rl"]["max_grad_norm"],
                data_cursor={"round": round_index + 1},
            )
            zero_fraction = float((advantages.suffix == 0).float().mean())
            zero_rounds = zero_rounds + 1 if zero_fraction > 0.8 else 0
            metrics = {
                "round": round_index + 1,
                "actor_loss": actor_loss,
                "renderer_loss": renderer_loss,
                "reward": float(torch.tensor(batch["rewards"]).mean()),
                "zero_advantage_fraction": zero_fraction,
                "renders": sum(len(row["render_records"]) for row in batch["episodes"]),
                "no_renderer_action": not any(row["render_records"] for row in batch["episodes"]),
                "gradients": committed["gradients"],
                "probes": committed["probes"],
            }
            append_metrics(folder / "metrics.jsonl", metrics)
            if (round_index + 1) % config["rl"]["evaluate_every"] == 0 or round_index + 1 == total:
                actor.save_policy(folder / "actor_policy")
                renderer.save_adapters(folder / "renderer_adapters")
                if Path(config["data"]["development_tasks"]).is_file():
                    components.render_bridge.behavior_version = behavior
                    evaluate_development(components, round_index + 1, folder / "development")
            prune_checkpoints(folder / "checkpoints")
            if zero_rounds >= 25:
                raise RuntimeError(
                    "Reward differences remained absent for 25 rounds; inspect judge and task coverage"
                )
        except Exception:
            if transaction.before is not None:
                transaction.abort("rl_round_failed")
            raise
    return str((folder / "checkpoints").resolve())


def evaluate_development(components, round_index, directory):
    from agentic_image.checks.private import PrivateEvaluator

    tasks = read_tasks(components.config["data"]["development_tasks"])
    if any(task.split_id == "train" for task in tasks):
        raise ValueError("Development evaluation cannot use training tasks")
    tasks = tasks[: components.config["rl"]["development_tasks"]]
    evaluator = PrivateEvaluator.from_file(
        components.judge,
        components.artifacts,
        components.config["data"]["development_private"],
        components.observer.tools,
    )
    collector = RolloutCollector(components)
    results = []
    for index, task in enumerate(tasks):
        environment = collector.infer(task, 101, Path(directory) / str(round_index) / str(index))
        budget = environment.state.budget
        result = evaluator.evaluate(
            task,
            environment.state.delivered_image_id,
            renders=budget.renders,
            inspects=budget.inspects,
            actions=budget.actions,
            invalids=budget.invalid,
            terminal_reason="protocol_failure"
            if environment.state.protocol_failure
            else "finished",
        )
        results.append({"task_id": task.task_id, **result.to_dict()})
    path = Path(directory) / str(round_index) / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, results)
    return results
