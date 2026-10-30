"""Graph-level evaluation of the planner (eq. 15-17) on held-out (request, previous graph) pairs.

run_eval_benchmarks.py evaluates sessions with given condition graphs, so the planner's own quality is
measured here: the zero-shot base model and the SFT adapter parse the same held-out requests and the
predicted graphs are compared with the references.
  validity            JSON parses and preserve_previous_conditions accepts the graph
  node P/R/F1         new nodes matched by normalized question (exact, then token Jaccard >= 0.5)
  type / scope acc.   on matched nodes
  edge / dependency F1  comp, supp, int edges and depends_on pairs mapped through the node matching
  superseded F1       explicit replacements
  python scripts/run_eval_planner.py --data data/final/planner_sft_dev.jsonl --model <planner> [--adapter runs/sft/planner]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from redag.learn.sft_trainer import read_records
from redag.models.planner import Planner, parse_json_object, preserve_previous_conditions

WORD = re.compile(r'[a-z0-9]+')


def tokens(text):
    return set(WORD.findall(text.lower()))


def match_nodes(predicted, reference):
    """Greedy one-to-one matching: exact normalized question first, then the best token-Jaccard >= 0.5."""
    pairs, used = {}, set()
    norm = {nid: ' '.join(sorted(tokens(n.question))) for nid, n in predicted.items()}
    for rid, ref in reference.items():
        key = ' '.join(sorted(tokens(ref.question)))
        hit = next((pid for pid, value in norm.items() if value == key and pid not in used), None)
        if hit:
            pairs[rid] = hit
            used.add(hit)
    scored = []
    for rid, ref in reference.items():
        if rid in pairs:
            continue
        for pid, pred in predicted.items():
            if pid in used:
                continue
            a, b = tokens(ref.question), tokens(pred.question)
            if a and b:
                scored.append((len(a & b) / len(a | b), rid, pid))
    for score, rid, pid in sorted(scored, reverse=True):
        if score < 0.5:
            break
        if rid not in pairs and pid not in used:
            pairs[rid] = pid
            used.add(pid)
    return pairs


def f1(hit, predicted, reference):
    p = hit / predicted if predicted else (1.0 if not reference else 0.0)
    r = hit / reference if reference else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def edge_sets(dag, keep):
    sets = {'comp': {tuple(e) for e in dag.comp_edges}, 'supp': {tuple(e) for e in dag.supp_edges},
            'int': {frozenset(e) for e in dag.int_edges},
            'dep': {(d, n) for n, node in dag.nodes.items() for d in node.depends_on}}
    return {k: {e for e in v if any(x in keep for x in e)} for k, v in sets.items()}


def score(record, text):
    previous = record.get('previous_dag')
    response = parse_json_object(record['response'])
    reference = preserve_previous_conditions(response, previous, response.get('superseded_ids', []))
    old = set(preserve_previous_conditions(previous, None).nodes) if previous else set()
    ref_new = {k: v for k, v in reference.nodes.items() if k not in old}
    try:
        value = parse_json_object(text)
        predicted = preserve_previous_conditions(value, previous, value.get('superseded_ids', []))
    except (ValueError, TypeError, KeyError) as error:
        return {'valid': False, 'error': str(error)[:200], 'reference_nodes': len(ref_new)}
    pred_new = {k: v for k, v in predicted.nodes.items() if k not in old}
    pairs = match_nodes(pred_new, ref_new)
    out = {'valid': True, 'reference_nodes': len(ref_new), 'predicted_nodes': len(pred_new), 'matched': len(pairs)}
    out['node_p'], out['node_r'], out['node_f1'] = f1(len(pairs), len(pred_new), len(ref_new))
    if pairs:
        out['type_acc'] = sum(ref_new[r].node_type == pred_new[p].node_type for r, p in pairs.items()) / len(pairs)
        out['scope_acc'] = sum(ref_new[r].scope == pred_new[p].scope for r, p in pairs.items()) / len(pairs)
    mapping = dict(pairs)
    mapping.update({nid: nid for nid in old})  # retained nodes keep their IDs on both sides
    ref_edges = edge_sets(reference, set(ref_new))
    pred_edges = edge_sets(predicted, set(pred_new))
    inverse = {p: r for r, p in mapping.items()}
    for kind in ('comp', 'supp', 'int', 'dep'):
        translated = set()
        for edge in pred_edges[kind]:
            ends = [inverse.get(x) for x in edge]
            if all(ends):
                translated.add(frozenset(ends) if kind == 'int' else tuple(ends))
        hit = len(translated & ref_edges[kind])
        out[f'{kind}_f1'] = f1(hit, len(pred_edges[kind]), len(ref_edges[kind]))[2]
    ref_sup, pred_sup = set(response.get('superseded_ids', [])), set(value.get('superseded_ids', []) or [])
    out['superseded_f1'] = f1(len(ref_sup & pred_sup), len(pred_sup), len(ref_sup))[2]
    return out


def generate(planner, batch, max_new_tokens):
    texts = [planner.processor.apply_chat_template(planner.messages(r['user_prompt'], r.get('previous_dag')), tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False) for r in batch]
    inputs = planner.processor(text=texts, padding=True, return_tensors='pt')
    device = next(planner.model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        out = planner.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return planner.processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--adapter')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    records = read_records(args.data)[:args.limit] if args.limit else read_records(args.data)
    planner = Planner.from_pretrained(args.model, args.device, torch.float32 if args.device == 'cpu' else torch.bfloat16,
                                      args.adapter)
    tokenizer = getattr(planner.processor, 'tokenizer', planner.processor)
    tokenizer.padding_side = 'left'
    rows = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        for record, text in zip(batch, generate(planner, batch, args.max_new_tokens)):
            rows.append(dict(score(record, text), id=record.get('id'), output=text[-4000:]))
        print(json.dumps({'done': len(rows), 'of': len(records)}), flush=True)
    valid = [r for r in rows if r['valid']]
    keys = ('node_p', 'node_r', 'node_f1', 'type_acc', 'scope_acc', 'comp_f1', 'supp_f1', 'int_f1', 'dep_f1', 'superseded_f1')
    summary = {'records': len(rows), 'validity': len(valid) / max(len(rows), 1), 'model': args.model, 'adapter': args.adapter}
    for key in keys:
        values = [r[key] for r in valid if key in r]
        summary[key] = sum(values) / len(values) if values else None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({'summary': summary, 'records': rows}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
