import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redag.harness.environment import add_runtime_arguments, load_runtime
from redag.learn.rl_trainer import train_alternating
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
    args = parser.parse_args()
    if not args.controller_adapter or not args.editor_adapter:
        parser.error("Controller and editor SFT adapters are required")
    records, dev_records = read_records(args.data), read_records(args.dev_data)
    harness, controller, editor = load_runtime(args, trainable_controller=True, trainable_editor=True)
    train_alternating(harness, controller, editor, records, dev_records, args.output,
                      rounds=args.rounds, epochs=args.epochs, batch_size=args.batch_size,
                      reward_mode=args.reward_mode, rollout_steps=args.rollout_steps)


if __name__ == "__main__":
    main()
