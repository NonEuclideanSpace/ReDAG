import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.verify.verifier import score_view


def load_scorer(model_id, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    yes, no = [processor.tokenizer.encode(word, add_special_tokens=False) for word in ('Yes', 'No')]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError('The judge needs single-token Yes and No answers')

    @torch.no_grad()
    def score(image, question):
        rgb = Image.fromarray((image.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': rgb}, {'type': 'text', 'text': question + ' Answer only Yes or No.'}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device)
        logits = model(**inputs).logits[0, -1, [yes[0], no[0]]].float()
        return float(logits.softmax(-1)[0])
    return score


def final_digest(case):
    image = Path(case['views'][case['final']]['image'])
    return hashlib.sha1(image.read_bytes()).hexdigest() if image.exists() else None


def pending_cases(paths, done):
    for path in map(Path, paths):
        for line in path.read_text(encoding='utf-8').splitlines()[1:]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            digest = final_digest(row['case'])
            if (path.name, row['key'], digest) not in done and (path.name, row['key'], None) not in done:
                yield path.name, row, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out_path = Path(args.out)
    done = {(row.get('file'), row.get('key'), row.get('image_sha1')) for row in map(json.loads, out_path.open())} if out_path.exists() else set()
    pending = list(pending_cases(args.cases, done))
    print('pending finals', len(pending), flush=True)
    if not pending:
        return
    scorer = load_scorer(args.model, args.device)
    with out_path.open('a') as out:
        for name, row, digest in pending:
            case = row['case']
            view = case['views'][case['final']]
            scores, nodes = score_view(view, scorer), view['nodes']
            weight = sum(node['weight'] for node in nodes.values())
            satisfied = sum(scores[nid] >= node['threshold'] for nid, node in nodes.items())
            out.write(json.dumps(dict(id=case['id'], file=name, key=row['key'], group=case['group'], scores=scores, k=len(nodes),
                                      s_final=sum(node['weight'] * scores[nid] for nid, node in nodes.items()) / weight,
                                      satisfied=satisfied, all_satisfied=satisfied == len(nodes), image_sha1=digest)) + '\n')
            out.flush()


if __name__ == '__main__':
    main()
