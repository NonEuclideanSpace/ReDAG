import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.harness.environment import add_runtime_arguments, load_runtime
from redag.learn.sft_trainer import read_records, collect_controller_rs, piece_bounds

STATE_COST = {'magicbrush': 1.0}


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--piece', type=int, default=0)
    parser.add_argument('--pieces', type=int, default=1)
    parser.add_argument('--checklist_state_cost', type=float, default=2.4)
    parser.add_argument('--group_size', type=int, default=3)
    parser.add_argument('--hard_multiplier', type=int, default=1)
    args = parser.parse_args()
    records = read_records(args.data)
    costs = [STATE_COST.get(record.get('origin'), args.checklist_state_cost) for record in records]
    start, stop = piece_bounds(costs, args.piece, args.pieces)
    harness, controller, editor = load_runtime(args)
    print(json.dumps(collect_controller_rs(records[start:stop], harness, controller, editor, args.output, args.seed, index_offset=start,
                                             group_size=args.group_size, hard_multiplier=args.hard_multiplier)))


if __name__ == '__main__':
    main()
