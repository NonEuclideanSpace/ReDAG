import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.harness.environment import add_runtime_arguments, load_runtime
from redag.learn.sft_trainer import read_records, collect_controller_rs


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    records = read_records(args.data)
    harness, controller, editor = load_runtime(args)
    print(json.dumps(collect_controller_rs(records, harness, controller, editor, args.output, args.seed)))


if __name__ == '__main__':
    main()
