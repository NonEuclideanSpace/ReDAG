import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redag.harness.environment import add_runtime_arguments, load_runtime
from redag.learn.rl_trainer import train_alternating
from redag.config import ReDAGConfig
from redag.learn.sft_trainer import read_records


def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dev_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, choices=[2], default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, choices=[4], default=4)
    parser.add_argument("--reward_mode", choices=["terminal_broadcast", "step_full_table", "subgraph_credit_ours"],
                        default="subgraph_credit_ours")
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (Path(args.output) / 'rl_complete.json').exists():  # finished earlier: nothing to load or train
        print('already complete:', Path(args.output) / 'rl_complete.json')
        return
    if not args.controller_adapter or not args.editor_adapter:
        parser.error("Controller and editor SFT adapters are required")
    records, dev_records = read_records(args.data), read_records(args.dev_data)
    sources = {name: getattr(args, name) for name in ('data', 'dev_data', 'controller_model', 'editor_model',
               'verifier_model', 'detector_model', 'sam_model', 'controller_adapter', 'editor_adapter')}
    for name in ('data', 'dev_data', 'controller_adapter', 'editor_adapter'):
        sources[name] = str(Path(sources[name]).resolve())
    started = time.perf_counter()
    harness, controller, editor = load_runtime(args, trainable_controller=True, trainable_editor=True)
    steps = {'num_denoise_steps': args.num_denoise_steps} if args.num_denoise_steps else {}
    history = train_alternating(harness, controller, editor, records, dev_records, args.output,
                      rounds=args.rounds, epochs=args.epochs, batch_size=args.batch_size,
                      reward_mode=args.reward_mode, rollout_steps=args.rollout_steps, seed=args.seed,
                      config=ReDAGConfig(ppo_epochs=args.ppo_epochs, **steps), sources=sources)
    runtime = dict(wall_seconds_including_model_load=time.perf_counter() - started,
                   optimizer_updates=sum(row['optimizer_updates'] for row in history), training_steps=len(history),
                   step_wall_seconds_total=sum(row['step_wall_seconds'] for row in history))
    (Path(args.output) / 'runtime.json').write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
