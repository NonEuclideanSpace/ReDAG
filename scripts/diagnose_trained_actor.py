import argparse
import ast
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.config import load_config
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.runtime.components import Components
from agentic_image.runtime.prompt import SYSTEM_PROMPT
from agentic_image.runtime.rollout import RolloutCollector, read_tasks
from diagnose_actor_replay import errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    config["devices"]["actor"] = arguments.device
    config["models"]["actor_policy"] = arguments.policy
    config["run_dir"] = str(Path(arguments.output).with_suffix(""))
    components = Components(config)
    actor = components.load_actor()
    task = read_tasks(arguments.tasks)[0]
    tokenizer = actor.processor.tokenizer
    reserve = len(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)) + 128
    environment = EpisodeEnvironment(
        task, components.artifacts, None, None, None,
        episode_id="preflight",
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        max_input_tokens=config["actor"]["max_input_tokens"] - reserve,
    )
    snapshot = RolloutCollector(components).snapshot(environment)
    started = time.monotonic()
    report = {"policy": arguments.policy, "task_id": task.task_id, "optimizer_updates": 0}

    def emit(stage, values):
        report[stage] = values
        report["seconds"] = time.monotonic() - started
        Path(arguments.output).write_text(json.dumps(report, indent=2))
        print(json.dumps({"stage": stage, **values}), flush=True)

    def diagnose(stage):
        backend_before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            result = actor.diagnose(snapshot)
        except RuntimeError as error:
            prefix = "Actor sampling/replay preflight failed: "
            if not str(error).startswith(prefix):
                raise
            result = ast.literal_eval(str(error)[len(prefix):])
        result["backend_before"] = backend_before
        result["backend_after"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        result["backend_restored"] = result["backend_after"] == backend_before
        if not result["backend_restored"]:
            raise RuntimeError("Actor numerical context changed the caller's backend setting")
        emit(stage, result)
        return result

    baseline = diagnose("trained_policy_baseline")
    if baseline["passed"]:
        return
    with torch.no_grad():
        embeddings, _, _ = actor._assemble(snapshot)
        inputs = actor.base.model.language_model.layers[0].input_layernorm(embeddings)
        selected = inputs[:, -32:]
        projections = actor.base.model.language_model.layers[0].linear_attn

        def compare():
            return {
                name: errors(
                    getattr(projections, name)(inputs)[:, -32:],
                    torch.cat([getattr(projections, name)(selected[:, index:index + 1]) for index in range(32)], dim=1),
                )
                for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z")
            }

        before = compare()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        after = compare()
        emit("full_precision_accumulation_projection", {"before": before, "after": after})
    if sum(item["mean"] for item in after.values()) < sum(item["mean"] for item in before.values()) * 0.5:
        diagnose("trained_policy_full_precision_accumulation")
    emit("complete", {"production_modified": False})


if __name__ == "__main__":
    main()
