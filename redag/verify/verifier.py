import torch
from typing import Dict, Set
from redag.graph.schema import BBox, ConditionDAG, ConditionNode
from redag.edit.compositor import defnorm_box_to_pixel_slice
from redag.graph.dag import semantic_support, topological_nodes
import math
import json
from pathlib import Path
from collections import defaultdict
from redag.verify.grounder import load_image, save_image

def expand_bbox_with_margin(box: BBox, scale: float=1.2) -> BBox:
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    hw = 0.5 * (box[2] - box[0]) * scale
    hh = 0.5 * (box[3] - box[1]) * scale
    return (max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh))

class TwoStageVerifier:

    def __init__(self, vlm_yes_no_scorer, gate_tau: float=0.5):
        self.scorer = vlm_yes_no_scorer
        self.gate_tau = gate_tau
        self.call_counter = 0
        self.records = []
        self.two_stage = True
        self.call_limit = None

    def score(self, image, question):
        if self.call_limit is not None and self.call_counter >= self.call_limit:
            raise RuntimeError('Verifier call budget exhausted')
        self.call_counter += 1
        result = float(self.scorer(image, question))
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError('Verifier returned an invalid probability')
        return result

    def verify_single_node_two_stage(self, image: torch.Tensor, node: ConditionNode, dag=None) -> float:
        q_1 = self.score(image, node.question)
        record = {"node_id": node.node_id, "stage1": q_1, "stage2": None}
        self.records.append(record)
        support = semantic_support(dag, node.node_id) if dag is not None else node.bbox if node.scope == 'local' and node.node_type in ('entity', 'attribute') and not node.depends_on else None
        if q_1 >= node.threshold or support is None or not self.two_stage:
            return q_1
        _, H, W = image.shape
        crop_box = expand_bbox_with_margin(support, scale=1.2)
        X1, Y1, X2, Y2 = defnorm_box_to_pixel_slice(crop_box, H, W)
        crop_img = image[:, Y1:Y2, X1:X2]
        q_2 = self.score(crop_img, node.question)
        record["stage2"] = q_2
        return q_2

    def verify_subgraph_after_edit(self, image_new: torch.Tensor, dag: ConditionDAG, prev_scores: Dict[str, float], affected_nodes: Set[str]) -> Dict[str, float]:
        new_scores: Dict[str, float] = dict(prev_scores)
        comp_parents = {nid: [] for nid in dag.nodes}
        for p, c in dag.comp_edges:
            if c in comp_parents:
                comp_parents[c].append(p)
        if not affected_nodes <= dag.nodes.keys():
            raise ValueError("Unknown affected node")
        ordered_affected = [nid for nid in topological_nodes(dag) if nid in affected_nodes]
        for v in ordered_affected:
            if any((new_scores.get(p, 0.0) < self.gate_tau for p in comp_parents[v])):
                new_scores[v] = 0.0
            else:
                new_scores[v] = self.verify_single_node_two_stage(image_new, dag.nodes[v], dag)
        return new_scores


def load_vlm_scorer(model_id='Qwen/Qwen3-VL-8B-Instruct', device='cuda', dtype=torch.bfloat16):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import numpy as np
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    yes, no = [processor.tokenizer.encode(word, add_special_tokens=False) for word in ('Yes', 'No')]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError('Verifier tokenizer must have single-token Yes and No answers')

    @torch.no_grad()
    def score(image, question):
        rgb = Image.fromarray((image.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': rgb}, {'type': 'text', 'text': question + ' Answer only Yes or No.'}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device)
        logits = model(**inputs, logits_to_keep=1).logits[0, -1, [yes[0], no[0]]].float()
        return float(logits.softmax(-1)[0])
    return score


def ratio(a, b):
    return a / b if b else None


def export_view(image, dag, folder, identifier):
    path = folder / (identifier + '.png')
    save_image(image, path)
    return {'id': identifier, 'image': str(path.resolve()), 'nodes': {nid: {'question': n.question, 'threshold': n.threshold, 'weight': n.weight,
            'scope': n.scope, 'depends_on': n.depends_on, 'bbox': n.bbox} for nid, n in dag.nodes.items()}}


def score_view(view, scorer=None, annotations=None):
    if annotations is not None:
        scores = annotations.get(view['id'])
        if scores is None:
            return None
    else:
        image = load_image(view['image'])
        scores = {nid: float(scorer(image, n['question'])) for nid, n in view['nodes'].items()}
    if set(scores) != set(view['nodes']) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError('Annotations must provide a finite [0,1] score for every node in a view')
    return scores


def summarize(case, scorer=None, annotations=None):
    scores = {key: score_view(view, scorer, annotations) for key, view in case['views'].items()}
    if any(v is None for v in scores.values()):
        return None
    totals = dict(protected=0, regressions=0, good=0, rejected_good=0, stage1_alerts=0, stage2_alerts=0, true1=0, true2=0)
    for step in case['steps']:
        before, candidate = scores[step['before']], scores[step['candidate']]
        nodes = case['views'][step['candidate']]['nodes']
        delivered = scores[step['delivered']] if 'delivered' in step else candidate if step['accepted'] else before
        target, protected = step['target'], step['protected']
        totals['protected'] += len(protected)
        totals['regressions'] += sum(delivered[v] < before[v] - case.get('max_drop', 0.08) for v in protected)
        good = (candidate[target] - before[target] >= case.get('min_gain', 0.05) or candidate[target] >= nodes[target]['threshold']) and all(candidate[v] >= before[v] - case.get('max_drop', 0.08) for v in protected)
        totals['good'] += good
        totals['rejected_good'] += good and not step['accepted']
        for record in step['stage_scores']:
            nid = record['node_id']
            for number, value in ((1, record['stage1']), (2, record['stage2'] if record['stage2'] is not None else record['stage1'])):
                alert = value < nodes[nid]['threshold']
                totals[f'stage{number}_alerts'] += alert
                totals[f'true{number}'] += alert and candidate[nid] < nodes[nid]['threshold']
    final = case['views'][case['final']]['nodes']
    return {'all_satisfied': all(scores[case['final']][n] >= final[n]['threshold'] for n in final), **totals}


def aggregate(rows):
    counts = {key: sum(row[key] for row in rows) for key in rows[0]}
    return {'sessions': len(rows), 'all_satisfied_rate': counts['all_satisfied'] / len(rows), 'protected_regression_rate': ratio(counts['regressions'], counts['protected']), 'stage1_vs_stage2_precision': {'stage1': ratio(counts['true1'], counts['stage1_alerts']), 'stage2': ratio(counts['true2'], counts['stage2_alerts'])}, 'good_edit_false_reject_rate': ratio(counts['rejected_good'], counts['good']), 'counts': counts}


def apply_annotations(result, annotations=None, scorer=None, source='human'):
    groups, expected = defaultdict(list), defaultdict(int)
    for case in result.get('evaluation_cases', []):
        expected[case['group']] += 1
        row = summarize(case, scorer, annotations)
        if row is not None:
            groups[case['group']].append(row)
    result[source + '_results'] = {key: {'annotated_sessions': len(groups[key]), 'expected_sessions': count, 'metrics': aggregate(groups[key]) if groups[key] else None} for key, count in expected.items()}
    for entry in result.get('full_audits', []):
        before, after = score_view(entry['before'], scorer, annotations), score_view(entry['after'], scorer, annotations)
        nodes = entry['before']['nodes']
        entry[source + '_E_leak'] = None if before is None or after is None else sum(n['weight'] * (1 + entry.get('protection_lambda', 1.5) if nid in entry['protected'] else 1) * abs(after[nid] - before[nid]) for nid, n in nodes.items() if nid not in entry['affected']) / sum(n['weight'] for n in nodes.values())


def apply_ratings(result, path, key_path=None):
    from redag.learn.sft_trainer import read_records
    pairs = {str(p['id']): p for p in result.get('blind_pairs', [])}
    ratings = list(read_records(path))
    votes, seen = defaultdict(lambda: {'a': 0, 'b': 0, 'tie': 0}), set()
    for row in ratings:
        identity = str(row['id']), str(row['annotator'])
        if identity in seen or identity[0] not in pairs or row['winner'] not in ('a', 'b', 'tie'):
            raise ValueError('Ratings require unique pair/annotator IDs and a, b or tie votes')
        seen.add(identity)
        votes[identity[0]][row['winner']] += 1
    comparison = defaultdict(lambda: {'wins': 0, 'ties': 0, 'losses': 0})
    if key_path:
        key = {str(row['id']): row for row in json.loads(Path(key_path).read_text())}
        if set(key) != set(pairs):
            raise ValueError('Blind key must match every rated pair')
        for identifier, counts in votes.items():
            for side, other in (('a', 'b'), ('b', 'a')):
                row = comparison[key[identifier][side]]
                row['wins'] += counts[side]
                row['losses'] += counts[other]
                row['ties'] += counts['tie']
        for row in comparison.values():
            row['preference_rate'] = ratio(row['wins'] + row['ties'] / 2, sum(row.values()))
    result['human_ratings'] = {'rated_pairs': len(votes), 'expected_pairs': len(pairs), 'votes': dict(votes), 'stage_comparison': dict(comparison) if key_path else None, 'complete': bool(pairs) and len(votes) == len(pairs)}
