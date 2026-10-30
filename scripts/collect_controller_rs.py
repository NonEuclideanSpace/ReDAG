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
    parser.add_argument('--piece', type=int, default=0)
    parser.add_argument('--pieces', type=int, default=1)
    args = parser.parse_args()
    records = read_records(args.data)
    start, stop = len(records) * args.piece // args.pieces, len(records) * (args.piece + 1) // args.pieces
    harness, controller, editor = load_runtime(args)
    print(json.dumps(collect_controller_rs(records[start:stop], harness, controller, editor, args.output, args.seed, index_offset=start)))


if __name__ == '__main__':
    main()
