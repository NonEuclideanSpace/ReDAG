import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.verify.grounder import load_image


def compute_outside_l1_consistency(src_img, fin_img, steps):
    if src_img.shape != fin_img.shape:
        from torch.nn import functional as F
        fin_img = F.interpolate(fin_img.unsqueeze(0), size=src_img.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    h, w = src_img.shape[-2:]
    mask = torch.ones((1, h, w), dtype=torch.float32, device=src_img.device)
    for st in steps:
        if not st.get('accepted', True):
            continue
        box = st.get('box') or st.get('controller_box')
        if box and len(box) == 4 and (box[2] - box[0]) * (box[3] - box[1]) < 0.9:
            x1, y1 = int(round(box[0] * w)), int(round(box[1] * h))
            x2, y2 = int(round(box[2] * w)), int(round(box[3] * h))
            mask[:, max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0.0
    denom = float(mask.sum().item()) * 3.0
    if denom < 100.0:
        diff = float((src_img - fin_img).abs().mean().item())
        return 1.0 - diff
    diff = float(((src_img - fin_img).abs() * mask).sum().item()) / denom
    return max(0.0, min(1.0, 1.0 - diff))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/root/autodl-tmp/ReDAG/data/final/eval_edival100.jsonl')
    parser.add_argument('--official_metrics', default='/root/autodl-tmp/ReDAG/runs/eval/edival100_official_metrics.json')
    parser.add_argument('--judge_jsonl', required=False, default=None)
    parser.add_argument('--redag_cases', nargs='*', default=[])
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
    official = json.loads(Path(args.official_metrics).read_text(encoding='utf-8'))

    print('=== EdiVal-100 Official Metrics Summary (n=100 3-turn sessions) ===')
    print(f'{"Model":24s} | {"T1 IF":>7s} | {"T2 CumIF":>8s} | {"T3 CumIF":>8s} | {"T3 L1":>7s} | {"T3 DINO":>7s} | {"Overall":>7s}')
    print('-' * 82)
    for m, st in official['models'].items():
        print(f'{m:24s} | {st["T1_IF"]:6.1f}% | {st["T2_CumIF"]:7.1f}% | {st["T3_CumIF"]:7.1f}% | {st["T3_L1"]:6.1f}% | {st["T3_DINO"]:6.1f}% | {st["T3_Overall"]:6.1f}%')

    if args.redag_cases:
        l1_list = []
        for cpath in args.redag_cases:
            p = Path(cpath)
            if not p.exists():
                continue
            for line in p.read_text(encoding='utf-8').splitlines()[1:]:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                idx = int(row['key'].split('|')[2])
                case = row['case']
                fin_view = case['views'][case['final']]
                src_img = load_image(records[idx]['image'])
                fin_img = load_image(fin_view['image'])
                l1_list.append(compute_outside_l1_consistency(src_img, fin_img, case.get('steps', [])))
        if l1_list:
            print(f'ReDAG Unchanged-Region L1 Consistency (n={len(l1_list)}): {sum(l1_list)/len(l1_list)*100:.2f}%')


if __name__ == '__main__':
    main()
