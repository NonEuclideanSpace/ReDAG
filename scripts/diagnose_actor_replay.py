import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.actor.cache import make_fp32_cache, validate_cache_precision
from agentic_image.actor.head import head_log_probs
from agentic_image.config import load_config
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.runtime.components import Components
from agentic_image.runtime.prompt import SYSTEM_PROMPT
from agentic_image.runtime.rollout import RolloutCollector, read_tasks


def errors(left, right):
    difference = (left.float().cpu() - right.float().cpu()).abs()
    return {
        "mean": float(difference.mean()),
        "p99": float(torch.quantile(difference.flatten(), 0.99)),
        "max": float(difference.max()),
    }


@torch.no_grad()
def fixed_decode(actor, snapshot, tokens):
    actor._set_adapter("policy", training=False)
    embeddings, attention, positions = actor._assemble(snapshot)
    cache = make_fp32_cache(actor.base.config)
    hidden = []
    for index in range(tokens.numel()):
        output = actor.base.model.language_model(
            inputs_embeds=embeddings,
            position_ids=positions,
            attention_mask=attention,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        hidden.append(output.last_hidden_state[:, -1].clone())
        if index + 1 < tokens.numel():
            text = torch.tensor([[[snapshot.prefix_length + index]]], device=actor.device)
            visual = (text + snapshot.rope_delta.to(actor.device).reshape(1, 1, 1)).expand(3, 1, 1)
            positions = torch.cat((text, visual), dim=0)
            attention = torch.cat((attention, attention.new_ones((1, 1))), dim=-1)
            embeddings = actor.base.get_input_embeddings()(tokens[index].reshape(1, 1).to(actor.device))
    validate_cache_precision(cache)
    return torch.cat(hidden, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    config["devices"]["actor"] = arguments.device
    config["run_dir"] = str(Path(arguments.output).with_suffix(""))
    components = Components(config)
    started = time.monotonic()
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
    report = {"device": arguments.device, "task_id": task.task_id, "optimizer_updates": 0}

    def emit(stage, values):
        report[stage] = values
        report["seconds"] = time.monotonic() - started
        Path(arguments.output).write_text(json.dumps(report, indent=2))
        print(json.dumps({"stage": stage, **values}), flush=True)

    captured = []
    handle = actor.base.lm_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    sample = actor.sample(snapshot, 20261027, "replay-diagnostic", probe_positions=(0, 15, 31))
    handle.remove()
    sampled_hidden = torch.cat(captured, dim=0)
    tokens = sample.token_ids.to(actor.device)
    sample_path = Path(arguments.output).with_suffix(".sample.pt")
    torch.save(sample.state_dict(), sample_path)
    with torch.no_grad():
        replay_hidden = actor._hidden(snapshot, tokens)
        replay = head_log_probs(replay_hidden, actor.base.lm_head, tokens, chunk_size=32)
        single = head_log_probs(replay_hidden, actor.base.lm_head, tokens, chunk_size=1)
        captured_replay = head_log_probs(sampled_hidden, actor.base.lm_head, tokens, chunk_size=1)
    emit("baseline", {
        "tokens": tokens.numel(),
        "sample_file": str(sample_path),
        "sample_replay": errors(sample.behavior_log_probs, replay),
        "same_hidden_head_batching": errors(single, replay),
        "captured_sample_replay": errors(sample.behavior_log_probs, captured_replay),
        "hidden_difference": errors(sampled_hidden, replay_hidden),
    })
    with torch.no_grad():
        weight = actor.base.lm_head.weight.float()

        def fp32_log_probs(hidden):
            values = []
            for offset in range(0, tokens.numel(), 32):
                logits = F.linear(hidden[offset:offset + 32].float(), weight)
                logp = F.log_softmax(logits / actor.config.temperature, dim=-1)
                values.append(logp.gather(-1, tokens[offset:offset + 32, None]).squeeze(-1))
            return torch.cat(values)

        emit("fp32_head_only", {"sample_replay": errors(fp32_log_probs(sampled_hidden), fp32_log_probs(replay_hidden))})
        del weight

    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit

    original_matmul = bnb.matmul_4bit

    def consistent_matmul(inputs, weights, quant_state, out=None, bias=None):
        return MatMul4Bit.apply(inputs, weights.view(-1, 1), out, bias, quant_state)

    bnb.matmul_4bit = consistent_matmul
    with torch.no_grad():
        consistent_sample = fixed_decode(actor, snapshot, sample.token_ids)
        consistent_replay = actor._hidden(snapshot, tokens)
        sampled_logp = head_log_probs(consistent_sample, actor.base.lm_head, tokens, chunk_size=1)
        replay_logp = head_log_probs(consistent_replay, actor.base.lm_head, tokens, chunk_size=32)
        weight = actor.base.lm_head.weight.float()
        emit("consistent_nf4", {
            "sample_replay": errors(sampled_logp, replay_logp),
            "fp32_head_sample_replay": errors(fp32_log_probs(consistent_sample), fp32_log_probs(consistent_replay)),
            "hidden_difference": errors(consistent_sample, consistent_replay),
        })
        del weight
    bnb.matmul_4bit = original_matmul
    emit("complete", {"peak_reserved_gib": torch.cuda.max_memory_reserved(actor.device) / 2**30})


if __name__ == "__main__":
    main()
