import ast
import copy
import json
import re
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from redag.graph.schema import ConditionDAG, ConditionNode

def parse_json_object(text):
    if isinstance(text, dict):
        return text
    text = str(text).strip()
    candidates = [text, re.sub(r",\s*([}\]])", r"\1", text)]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError):
                continue
        try:
            value = ast.literal_eval(candidate[candidate.index("{"):candidate.rindex("}") + 1])
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError, RecursionError):
            continue
    raise ValueError("Expected a JSON object")

def dag_to_dict(dag):
    return {"nodes": [asdict(n) for n in dag.nodes.values()],
            "comp_edges": dag.comp_edges, "supp_edges": dag.supp_edges, "int_edges": dag.int_edges}

def dag_from_dict(value):
    value = parse_json_object(value)
    raw_nodes = value.get("nodes", [])
    if isinstance(raw_nodes, dict):
        raw_nodes = [dict(n, node_id=nid) for nid, n in raw_nodes.items()]
    nodes = {}
    for raw in raw_nodes:
        node = ConditionNode(**{k: tuple(v) if k == "bbox" and v is not None else v for k, v in raw.items()})
        if node.node_id in nodes:
            raise ValueError(f"Duplicate node ID: {node.node_id}")
        nodes[node.node_id] = node
    edges = {key: [tuple(edge) for edge in value.get(key, [])]
             for key in ("comp_edges", "supp_edges", "int_edges")}
    return ConditionDAG(nodes=nodes, **edges)

def preserve_previous_conditions(value, previous_dag=None, superseded_ids=None):
    value = dag_to_dict(value) if isinstance(value, ConditionDAG) else parse_json_object(value)
    superseded = value.get("superseded_ids", []) if superseded_ids is None else superseded_ids
    if not isinstance(superseded, (tuple, list)) or any(not isinstance(nid, str) for nid in superseded):
        raise ValueError("Superseded IDs must be a list of previous condition IDs")
    if previous_dag is None:
        if superseded:
            raise ValueError("Cannot supersede conditions without a previous graph")
        return dag_from_dict(value)
    previous = previous_dag if isinstance(previous_dag, ConditionDAG) else dag_from_dict(previous_dag)
    superseded = set(superseded)
    if not superseded <= previous.nodes.keys():
        raise ValueError("Superseded IDs must identify previous conditions")
    raw = value.get("nodes", [])
    raw = [dict(node, node_id=nid) for nid, node in raw.items()] if isinstance(raw, dict) else raw
    if len({node["node_id"] for node in raw}) != len(raw):
        raise ValueError("Duplicate planned node ID")
    nodes = {node["node_id"]: node for node in raw}
    nodes.update({nid: asdict(node) for nid, node in previous.nodes.items() if nid not in superseded})
    for nid, node in nodes.items():
        missing = set(node.get("depends_on", [])) - nodes.keys()
        if nid in previous.nodes and nid not in superseded and missing and missing <= superseded:
            node["depends_on"], node["scope"] = [dep for dep in node["depends_on"] if dep not in missing], "unknown"
    merged = {"nodes": list(nodes.values())}
    for name in ("comp_edges", "supp_edges", "int_edges"):
        retained = [edge for edge in getattr(previous, name) if not set(edge) & superseded]
        edges = [tuple(edge) for edge in value.get(name, [])] + retained
        merged[name] = list(dict.fromkeys(edges))
    return dag_from_dict(merged)

def image_to_pil(image):
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = image.detach().float().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)

def append_tokens(inputs, tokens):
    result = dict(inputs)
    result["input_ids"] = torch.cat((inputs["input_ids"], tokens), dim=1)
    result["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(tokens)), dim=1)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in inputs:
            result[key] = torch.cat((inputs[key], torch.zeros_like(tokens, dtype=inputs[key].dtype)), dim=1)
    result.pop("position_ids", None)
    return result

def teacher_forced_batch(owner, examples):
    logits, labels = [], []
    owner.model.train()
    for module in owner.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    tokenizer = getattr(owner.processor, "tokenizer", owner.processor)
    for messages, response in examples:
        inputs = owner.encode(messages)
        encoded = tokenizer.encode(response, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            encoded.append(tokenizer.eos_token_id)
        if not encoded:
            raise ValueError("Empty supervised response")
        tokens = torch.tensor([encoded], device=inputs["input_ids"].device)
        full = append_tokens(inputs, tokens)
        target = full["input_ids"][:, -(len(encoded) + 1):].clone()
        target[:, 0] = -100
        logits.append(owner.model(**full, use_cache=False, logits_to_keep=len(encoded) + 1).logits)
        labels.append(target)
    if not logits:
        raise ValueError("Empty supervised batch")
    width = max(x.shape[1] for x in logits)
    return (torch.cat([F.pad(x, (0, 0, 0, width - x.shape[1])) for x in logits]),
            torch.cat([F.pad(x, (0, width - x.shape[1]), value=-100) for x in labels]))

LORA_PATTERN = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                r"|linear_attn\.(in_proj_\w+|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def lora_target_modules(model):
    """LoRA on every projection of every language-model layer.

    Qwen3.5-family models interleave three Gated-DeltaNet (linear attention) layers with one full-attention
    layer, so the usual q/k/v/o targets would adapt only a quarter of the decoder.  The regex covers both
    layer kinds and the MLPs but never the vision tower or the LM head; unknown layouts fall back to q/k/v/o.
    """
    pattern = re.compile(LORA_PATTERN)
    if any(isinstance(module, torch.nn.Linear) and pattern.fullmatch(name) for name, module in model.named_modules()):
        return LORA_PATTERN
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


class Planner:
    default_model_id = "Qwen/Qwen3.8-27B"

    def __init__(self, model, processor, max_new_tokens=2048):
        self.model, self.processor = model, processor
        self.max_new_tokens = max_new_tokens
        self.last_superseded_ids = ()
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_id=None, device="cuda", dtype=torch.bfloat16,
                        adapter_path=None, trainable=False):
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model_id = model_id or cls.default_model_id
        processor = AutoProcessor.from_pretrained(model_id)
        quantized = cls is Planner and trainable and str(device).startswith("cuda")
        if quantized:
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype,
                    quantization_config=quantization, device_map={"": device})
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        else:
            model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype).to(device)
        model.requires_grad_(False)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
        elif trainable:
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                  target_modules=lora_target_modules(model),
                                  bias="none", task_type="CAUSAL_LM"))
        if trainable:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(model, processor)

    def encode(self, messages):
        encoded = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                  enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}

    def lora_parameters(self):
        return [p for name, p in self.model.named_parameters() if "lora_" in name]

    def snapshot(self):
        return {name: p.detach().cpu().clone() for name, p in self.model.named_parameters() if "lora_" in name}

    def load_snapshot(self, snapshot):
        parameters = dict(self.model.named_parameters())
        expected = {name for name in parameters if "lora_" in name}
        if set(snapshot) != expected:
            raise ValueError("Adapter snapshot parameters do not match the model")
        with torch.no_grad():
            for name, tensor in snapshot.items():
                parameters[name].copy_(tensor.to(parameters[name]))

    def save_adapter(self, path):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        if not self.lora_parameters():
            raise ValueError("No LoRA adapter to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(self.model)
        save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, path / "adapter_model.safetensors")
        config = copy.deepcopy(self.model.peft_config["default"])
        config.inference_mode = True
        config.save_pretrained(path)
        self.processor.save_pretrained(path)

    def messages(self, user_prompt, previous_dag=None):
        system = (
            "Parse requirements into atomic yes/no conditions. Return JSON: nodes, comp_edges, supp_edges, int_edges. "
            "Each node contains node_id, node_type, question, entity_phrase, weight, scope, depends_on; optional bbox, threshold, expected_count. "
            "node_type: background, entity, attribute, relation, count, prohibition. Weights are positive. "
            "Use original-image normalized bbox [x1,y1,x2,y2] if known, otherwise null. "
            "comp_edges: entity to dependent; supp_edges: bearer to borne; int_edges: undirected interference. Dependencies must be acyclic. "
            "scope is local, global or unknown. depends_on lists every condition or object ID needed to judge this node. "
            "Use global for uniqueness, unrestricted counts, absence, whole-image style and comparisons to all objects, "
            "including 'the only red object'. New objects can invalidate these even outside the target box. "
            "Use local only when all dependencies are known and bounded; relations must list every participating object. "
            "Use unknown whenever coverage is uncertain. A box alone never proves a condition is local. "
            "Keep previous conditions and IDs unless the new user request explicitly supersedes them. "
            "Only explicit replacements/removals get optional top-level superseded_ids listing previous IDs. "
            "Additions never supersede prior conditions. Return the complete graph."
        )
        text = user_prompt
        if previous_dag is not None:
            previous = dag_to_dict(previous_dag) if isinstance(previous_dag, ConditionDAG) else previous_dag
            text = "Previous graph: " + json.dumps(previous, ensure_ascii=False) + "\nNew request: " + user_prompt
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    @torch.no_grad()
    def plan(self, user_prompt, previous_dag=None, superseded_ids=None):
        self.model.eval()
        if superseded_ids:
            user_prompt += "\nExplicitly superseded previous condition IDs: " + json.dumps(list(superseded_ids))
        inputs = self.encode(self.messages(user_prompt, previous_dag))
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        value = parse_json_object(text)
        superseded = value.get("superseded_ids", []) if superseded_ids is None else superseded_ids
        dag = preserve_previous_conditions(value, previous_dag, superseded)
        self.last_superseded_ids = tuple(superseded)
        return dag

    def sft_batch(self, records):
        examples = []
        for record in records:
            response = record.get("response", record.get("dag", record.get("graph")))
            response = dag_to_dict(response) if isinstance(response, ConditionDAG) else parse_json_object(response)
            dag = preserve_previous_conditions(response, record.get("previous_dag"))
            target = dag_to_dict(dag)
            if "superseded_ids" in response:
                target["superseded_ids"] = response["superseded_ids"]
            examples.append((self.messages(record["user_prompt"], record.get("previous_dag")),
                             json.dumps(target, ensure_ascii=False)))
        return teacher_forced_batch(self, examples)
