import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from redag.models.controller import Controller
from redag.learn.sft_trainer import read_records, train_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--adapter')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    records = read_records(args.data)
    model = Controller.from_pretrained(args.model, args.device, torch.float32 if args.device == 'cpu' else torch.bfloat16, args.adapter, True)
    result = train_sft(model, records, args.output, 'controller', args.epochs, args.batch_size, args.learning_rate, args.seed)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
