import json
import math
from dataclasses import asdict

import torch

from redag.edit.compositor import defnorm_box_to_pixel_slice
from redag.graph.dag import classify_failure_mode, compute_affected_subgraph
from redag.graph.schema import CanvasState, ConditionDAG, ControllerAction
from redag.models.planner import (Planner, append_tokens, dag_from_dict, image_to_pil,
                                  parse_json_object, teacher_forced_batch)
from redag.verify.verifier import expand_bbox_with_margin


OPS = {"missing": "add_entity", "attribute_error": "modify_attr", "relation_error": "fix_relation",
       "over_count": "remove_extra", "entangled": "separate_instances"}


def parse_controller_action(text, node, failure_mode="attribute_error"):
    fallback = expand_bbox_with_margin(node.bbox, 1.25) if node.bbox else (0.0, 0.0, 1.0, 1.0)
    try:
        value = parse_json_object(text)
    except (ValueError, TypeError):
        value = {}
    op = value.get("op", OPS[failure_mode])
    if op not in OPS.values():
        op = OPS[failure_mode]
    try:
        box = tuple(float(x) for x in value.get("box", fallback))
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError("Invalid box")
        box = tuple(max(0.0, min(1.0, x)) for x in box)
        if box[2] <= box[0] or box[3] <= box[1]:
            box = fallback
    except (TypeError, ValueError, OverflowError):
        box = fallback
    instruction = value.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = (f"Edit the selected region so the answer to this requirement is Yes: {node.question} "
                       "Preserve all unrelated objects, their identity, layout, colors, lighting and texture. "
                       "Change only what is necessary to satisfy the requirement.")
    return ControllerAction(op=op, box=box, instruction=instruction.strip())



def bounded_instruction(text, tokenizer):
    original = text.strip()
    text = original
    suffix = " Preserve the identity, position, shape, colors, lighting and texture of every unrelated object. Modify only the requested region and keep all satisfied requirements intact."
    for _ in range(8):
        if len(tokenizer.encode(text, add_special_tokens=False)) >= 80:
            break
        text += suffix
    encoded = tokenizer.encode(text, add_special_tokens=False)
    limit = min(len(encoded), 120)
    text = tokenizer.decode(encoded[:limit], skip_special_tokens=True).strip()
    while len(tokenizer.encode(text, add_special_tokens=False)) > 120 and limit > 80:
        limit -= 1
        text = tokenizer.decode(encoded[:limit], skip_special_tokens=True).strip()
    count = len(tokenizer.encode(text, add_special_tokens=False))
    if not 80 <= count <= 120:
        raise ValueError("Tokenizer cannot produce an instruction within 80–120 tokens")
    return text, count, text != original


class Controller(Planner):
    default_model_id = "Qwen/Qwen3.5-9B"

    def __init__(self, model, processor, max_new_tokens=384):
        super().__init__(model, processor, max_new_tokens)

    def messages(self, state, dag, target_node_id, detected_count=1):
        node = dag.nodes[target_node_id]
        box = node.bbox or (0.0, 0.0, 1.0, 1.0)
        failure = classify_failure_mode(dag, target_node_id, state.scores, detected_count)
        affected = compute_affected_subgraph(dag, ControllerAction(OPS[failure], box, node.question), target_node_id)
        summary = [{"question": dag.nodes[nid].question, "score": state.scores[nid],
                    "protected": nid in state.protected_set, "bbox": dag.nodes[nid].bbox}
                   for nid in dag.nodes if nid in affected]
        prompt = (
            f"[Target NodeQuestion]: {node.question}\n"
            f"[Current Node Score]: {state.scores[target_node_id]:.2f} (Threshold: {node.threshold:.2f})\n"
            f"[Failure Mode]: {failure}\n"
            f"[Target Current BBox]: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]\n"
            f"[Nearby Protected Nodes]: {json.dumps(summary, ensure_ascii=False)}\n"
            'Output ONLY a JSON object with keys "op", "box", and "instruction":'
        )
        x1, y1, x2, y2 = defnorm_box_to_pixel_slice(box, *state.image.shape[-2:])
        system = (
            "You control local image editing. The first image is the complete original canvas; "
            "the second is the target crop. All box coordinates refer to the original canvas in [0,1]. "
            "Choose op from add_entity, modify_attr, fix_relation, remove_extra, separate_instances. "
            "Write an actionable natural-language instruction of 80-120 tokens describing the desired result "
            "and preserving nearby satisfied requirements. Output exactly op, box and instruction. "
            "Do not output node IDs, canvas IDs, parent pointers, versions or reasoning."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": [
            {"type": "image", "image": image_to_pil(state.image)},
            {"type": "image", "image": image_to_pil(state.image[:, y1:y2, x1:x2])},
            {"type": "text", "text": prompt}]}]

    @torch.no_grad()
    def sample(self, state, dag, target_node_id, seed=42, temperature=0.8, detected_count=1):
        if temperature < 0:
            raise ValueError("Temperature must be nonnegative")
        self.model.eval()
        inputs = self.encode(self.messages(state, dag, target_node_id, detected_count))
        device = inputs["input_ids"].device
        devices = [device.index or 0] if device.type == "cuda" else []
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=temperature > 0,
                      top_k=0, top_p=1.0, repetition_penalty=1.0)
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            generated = self.model.generate(**inputs, **kwargs)
        length = inputs["input_ids"].shape[1]
        action_tokens = generated[:, length:]
        if action_tokens.numel() == 0:
            raise ValueError("The controller generated no action tokens")
        text = self.processor.decode(action_tokens[0], skip_special_tokens=True)
        failure = classify_failure_mode(dag, target_node_id, state.scores, detected_count)
        action = parse_controller_action(text, dag.nodes[target_node_id], failure)
        action.instruction, instruction_tokens, repaired = bounded_instruction(action.instruction, self.processor.tokenizer)
        full = append_tokens(inputs, action_tokens)
        action_mask = torch.zeros((1, generated.shape[1] - 1), device=device)
        action_mask[:, length - 1:] = 1.0
        return {"action": action, "inputs": full, "token_ids": generated, "action_mask": action_mask,
                "text": text, "temperature": temperature if temperature > 0 else 1.0,
                "instruction_tokens": instruction_tokens, "instruction_repaired": repaired}

    def forward_sample(self, sample, action_only=False):
        self.model.train(torch.is_grad_enabled())
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        keep = int(sample["action_mask"].sum()) + 1 if action_only else 0
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample["inputs"].items()}
        return self.model(**inputs, use_cache=False, logits_to_keep=keep).logits / sample.get("temperature", 1.0)

    def sft_batch(self, records):
        examples = []
        for record in records:
            dag = record.get("dag", record.get("graph"))
            if not isinstance(dag, ConditionDAG):
                dag = dag_from_dict(dag)
            state = record.get("state")
            if not isinstance(state, CanvasState):
                import numpy as np
                image = record["image"]
                if not torch.is_tensor(image):
                    image = torch.from_numpy(np.array(image_to_pil(image), copy=True)).permute(2, 0, 1).float() / 255
                scores = record["scores"]
                protected = set(record.get("protected_set", [nid for nid, score in scores.items()
                                                             if score >= dag.nodes[nid].threshold]))
                state = CanvasState(image, scores, protected)
            target = record["target_node_id"]
            action = record["action"]
            if not isinstance(action, ControllerAction):
                action = parse_controller_action(action, dag.nodes[target],
                         classify_failure_mode(dag, target, state.scores, record.get("detected_count", 1)))
            action = ControllerAction(action.op, action.box, bounded_instruction(action.instruction, self.processor.tokenizer)[0])
            examples.append((self.messages(state, dag, target, record.get("detected_count", 1)),
                             json.dumps(asdict(action), ensure_ascii=False)))
        return teacher_forced_batch(self, examples)
