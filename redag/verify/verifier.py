import torch
from typing import Dict, Set
from redag.graph.schema import BBox, ConditionDAG, ConditionNode
from redag.edit.compositor import defnorm_box_to_pixel_slice
from redag.graph.dag import topological_nodes
import math

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

    def verify_single_node_two_stage(self, image: torch.Tensor, node: ConditionNode) -> float:
        q_1 = self.score(image, node.question)
        record = {"node_id": node.node_id, "stage1": q_1, "stage2": None}
        self.records.append(record)
        if q_1 >= node.threshold or node.bbox is None or not self.two_stage:
            return q_1
        _, H, W = image.shape
        crop_box = expand_bbox_with_margin(node.bbox, scale=1.2)
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
                new_scores[v] = self.verify_single_node_two_stage(image_new, dag.nodes[v])
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
