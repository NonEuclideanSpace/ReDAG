import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.actor.records import ActionSample
from agentic_image.config import load_config
from agentic_image.runtime.components import Components
from diagnose_actor_replay import errors, fixed_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    config["devices"]["actor"] = arguments.device
    config["run_dir"] = str(Path(arguments.output).with_suffix(""))
    actor = Components(config).load_actor()
    sample = ActionSample.from_state_dict(torch.load(arguments.sample, map_location="cpu", weights_only=True))
    tokens = sample.token_ids.to(actor.device)
    start = sample.snapshot.prefix_length - 1
    started = time.monotonic()
    modules = dict(actor.base.model.language_model.named_modules())
    names = [f"layers.{index}" for index in range(32)] + [
        "layers.0.input_layernorm", "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_a", "layers.0.linear_attn.in_proj_b",
        "layers.0.linear_attn.in_proj_z", "layers.0.linear_attn.out_proj",
        "layers.0.linear_attn", "layers.0.post_attention_layernorm", "layers.0.mlp",
        "layers.3.input_layernorm", "layers.3.self_attn.q_proj", "layers.3.self_attn.k_proj",
        "layers.3.self_attn.v_proj", "layers.3.self_attn", "layers.3.post_attention_layernorm",
    ]
    captured = {name: [] for name in names}
    mode = "decode"

    def observe(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if mode == "decode":
                captured[name].append(value[:, -1].detach().clone())
            else:
                captured[name].append(value[:, start:start + tokens.numel()].detach().clone())
        return hook

    handles = [modules[name].register_forward_hook(observe(name)) for name in names]
    with torch.no_grad():
        fixed_decode(actor, sample.snapshot, sample.token_ids)
        mode = "replay"
        actor._hidden(sample.snapshot, tokens)
    for handle in handles:
        handle.remove()
    report = {"task_id": sample.snapshot.metadata["task_id"], "tokens": tokens.numel(), "optimizer_updates": 0, "prefix_length": sample.snapshot.prefix_length, "layers": {}}
    for name, values in captured.items():
        decode = torch.cat(values[:-1], dim=0)
        replay = values[-1][0]
        report["layers"][name] = {
            "all": errors(decode, replay),
            "prefix_last": errors(decode[0], replay[0]),
            "first_decode": errors(decode[1], replay[1]),
            "last_decode": errors(decode[-1], replay[-1]),
        }
    report["seconds"] = time.monotonic() - started
    Path(arguments.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
