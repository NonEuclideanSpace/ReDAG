import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.actor.head import head_log_probs
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
    started = time.monotonic()
    modules = [module for module in actor.base.model.language_model.modules() if hasattr(module, "recurrent_gated_delta_rule")]
    originals = [module.recurrent_gated_delta_rule for module in modules]
    report = {"task_id": sample.snapshot.metadata["task_id"], "tokens": tokens.numel(), "optimizer_updates": 0, "linear_attention_layers": len(modules)}

    from fla.modules.l2norm import l2norm_fwd

    def normalized(function):
        def forward(query, key, value, **kwargs):
            if kwargs.get("use_qk_l2norm_in_kernel"):
                query = l2norm_fwd(query)[0]
                key = l2norm_fwd(key)[0]
                kwargs["use_qk_l2norm_in_kernel"] = False
            return function(query, key, value, **kwargs)
        return forward

    def emit(stage, values):
        report[stage] = values
        report["seconds"] = time.monotonic() - started
        Path(arguments.output).write_text(json.dumps(report, indent=2))
        print(json.dumps({"stage": stage, **values}), flush=True)

    with torch.no_grad():
        reference_hidden = actor._hidden(sample.snapshot, tokens)
        reference_logp = head_log_probs(reference_hidden, actor.base.lm_head, tokens)
        for module, original in zip(modules, originals):
            module.recurrent_gated_delta_rule = normalized(original)
        hidden = fixed_decode(actor, sample.snapshot, sample.token_ids)
        logp = head_log_probs(hidden, actor.base.lm_head, tokens, chunk_size=1)
        result = errors(logp, reference_logp)
        emit("aligned_qk_normalization", {"sample_replay": result, "hidden_difference": errors(hidden, reference_hidden)})
        if result["mean"] <= 0.01 and result["p99"] <= 0.05:
            emit("complete", {"candidate": "aligned_qk_normalization", "fixed_sequence_passed": True})
            return
        for module in modules:
            module.recurrent_gated_delta_rule = module.chunk_gated_delta_rule
        hidden = fixed_decode(actor, sample.snapshot, sample.token_ids)
        logp = head_log_probs(hidden, actor.base.lm_head, tokens, chunk_size=1)
        result = errors(logp, reference_logp)
        emit("chunk_decode", {"sample_replay": result, "hidden_difference": errors(hidden, reference_hidden)})
        emit("complete", {"candidate": "chunk_decode", "fixed_sequence_passed": result["mean"] <= 0.01 and result["p99"] <= 0.05})


if __name__ == "__main__":
    main()
