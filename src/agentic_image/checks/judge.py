import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from .reward import AtomScore


TEMPLATES = (
    "Inspect the attached image(s). The first image is the candidate. Any additional image "
    "has the role stated below.\nRequest context: {request}\nImage roles: {roles}\n"
    "Check exactly this property: {property}\nDoes the candidate visibly satisfy the entire "
    "property? Answer only Yes or No. Judge the image, not an assistant's claim.",
    "Compare the candidate with the stated requirement and any named reference.\nRequest "
    "context: {request}\nImage roles: {roles}\nRequired property: {property}\nIs this property "
    "clearly supported by what is visible in the candidate? Answer only Yes or No. Do not infer "
    "a requested object is present merely because it is mentioned in the requirement.",
)
JUDGE_PROFILES = {
    "Qwen/Qwen3-VL-32B-Instruct": {
        "revision": "1e6810acf5d8ebd189b4e75590877b64358e454c",
        "quantized_modules": 448,
        "quantized_parameters": 31205621760,
        "base_parameters": 33357390064,
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "quantized_modules": 252,
        "quantized_parameters": 6945767424,
        "base_parameters": 8767123696,
    },
}


@dataclass
class JudgeConfig:
    model: str = "Qwen/Qwen3-VL-32B-Instruct"
    revision: str = "1e6810acf5d8ebd189b4e75590877b64358e454c"
    source_path: str | None = None
    nf4_path: str = "models/judge-nf4"
    device: str = "cuda:3"
    max_input_tokens: int = 8192
    max_pixels: int = 262144
    threshold: float = 0.8
    calibration_path: str | None = None
    precision: str = "nf4-double-bf16"

    def __post_init__(self):
        profile = JUDGE_PROFILES.get(self.model)
        if profile is None or self.revision != profile["revision"]:
            raise ValueError("Judge model and revision must match a registered architecture profile")
        if self.precision != "nf4-double-bf16":
            raise ValueError("The registered judge precision is NF4 double quantization with BF16 compute")

    @property
    def calibration_identity(self):
        return {"model_id": self.model, "revision": self.revision, "precision": self.precision}


def validate_judge_base(model, config=None):
    from bitsandbytes.nn import Linear4bit

    config = config or JudgeConfig()
    profile = JUDGE_PROFILES[config.model]
    if config.model == "Qwen/Qwen3-VL-8B-Instruct":
        text = model.config.get_text_config()
        expected = {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
        mismatches = {
            key: (getattr(text, key, None), value)
            for key, value in expected.items()
            if getattr(text, key, None) != value
        }
        if mismatches or model.config.tie_word_embeddings:
            raise ValueError(f"Unexpected Qwen3-VL-8B architecture: {mismatches}")
    targets = r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    modules = dict(model.named_modules())
    quantized = {name: module for name, module in modules.items() if isinstance(module, Linear4bit)}
    expected = {name for name in modules if re.fullmatch(targets, name)}
    if set(quantized) != expected or len(quantized) != profile["quantized_modules"]:
        raise ValueError(
            f"The judge NF4 base must contain exactly {profile['quantized_modules']} registered language projections"
        )
    if (
        sum(module.in_features * module.out_features for module in quantized.values())
        != profile["quantized_parameters"]
    ):
        raise ValueError("The quantized judge dimensions differ from its registered checkpoint")
    resident = 0
    for name, parameter in model.named_parameters():
        if parameter.__class__.__name__ == "Params4bit":
            state = parameter.quant_state
            if parameter.dtype != torch.uint8 or parameter.requires_grad:
                raise TypeError("Judge quantized payloads must remain frozen uint8 tensors")
            if (
                state is None
                or state.quant_type != "nf4"
                or not state.nested
                or state.state2 is None
            ):
                raise ValueError("Judge weights require serialized double-quantized NF4 state")
            if (
                state.blocksize != 64
                or state.state2.blocksize != 256
                or state.dtype != torch.bfloat16
            ):
                raise ValueError(
                    "Judge NF4 block sizes or original compute precision are incompatible"
                )
            if state.absmax.dtype != torch.uint8 or state.state2.absmax.dtype != torch.float32:
                raise TypeError("Judge double-quantization scale precision is incompatible")
            resident += math.prod(state.shape)
        else:
            resident += parameter.numel()
            if (
                parameter.requires_grad
                or parameter.is_floating_point()
                and parameter.dtype != torch.bfloat16
            ):
                raise TypeError(f"Retained judge parameters must be frozen BF16: {name}")
    if resident != profile["base_parameters"]:
        raise ValueError(f"Judge resident parameter count differs from the fixed model: {resident}")
    if any(module.compute_dtype != torch.bfloat16 for module in quantized.values()):
        raise TypeError("Judge NF4 projections must compute in BF16")


class FrozenJudge:
    supports_probability_audit = True

    def __init__(self, model, processor, config):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.config = config
        self.lock = threading.RLock()
        self.thresholds = {}
        self.require_calibration = False
        self.queries = 0
        self.forwards = 0
        if config.calibration_path and Path(config.calibration_path).is_file():
            identity_path = Path(str(config.calibration_path) + ".model.json")
            if config.model == "Qwen/Qwen3-VL-8B-Instruct" or identity_path.is_file():
                if not identity_path.is_file():
                    raise ValueError("The 8B judge requires a model-specific calibration identity")
                if json.loads(identity_path.read_text()) != config.calibration_identity:
                    raise ValueError("Judge calibration belongs to a different checkpoint or precision")
            self.thresholds = json.loads(Path(config.calibration_path).read_text())

    @classmethod
    def load(cls, config, quantize=False):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if not config.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The frozen NF4 judge requires a configured CUDA device")
        torch.cuda.set_device(torch.device(config.device))
        local = Path(config.nf4_path)
        manifest_path = local / "base_manifest.json"
        if local.exists():
            if not manifest_path.is_file():
                raise FileExistsError(
                    "The judge NF4 directory contains an unfinished or unregistered export"
                )
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("model_id") != config.model
                or manifest.get("revision") != config.revision
            ):
                raise ValueError(
                    "The serialized judge base belongs to a different model or revision"
                )
            quantize = False
        elif not quantize:
            raise FileNotFoundError(
                "Create the serialized judge NF4 base with prepare-models first"
            )
        original_source = config.source_path or config.model
        source = original_source if quantize else str(local)
        source_options = (
            {"local_files_only": True}
            if config.source_path or not quantize
            else {"revision": config.revision}
        )
        kwargs = dict(source_options)
        if quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.uint8,
                llm_int8_skip_modules=["model.visual", "lm_head"],
            )
        model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
            source,
            device_map={"": config.device},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
            **kwargs,
        )
        issues = {
            key: loading.get(key, [])
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading.get(key)
        }
        if issues:
            raise RuntimeError(f"Judge checkpoint did not load strictly: {issues}")
        model.eval().requires_grad_(False)
        validate_judge_base(model, config)
        processor = AutoProcessor.from_pretrained(
            source,
            max_pixels=config.max_pixels,
            **source_options,
        )
        if processor.tokenizer.convert_tokens_to_ids("<|im_end|>") != 151645:
            raise ValueError("Judge tokenizer does not match the registered completion token")
        if quantize:
            staging = local.with_name(local.name + ".preparing")
            if staging.exists():
                raise FileExistsError(f"An unfinished judge quantization export exists: {staging}")
            staging.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
            processor.save_pretrained(staging)
            (staging / "base_manifest.json").write_text(
                json.dumps(
                    {
                        "model_id": config.model,
                        "revision": config.revision,
                        "quantization": "nf4-double",
                        "compute_dtype": "bfloat16",
                        "source_path": config.source_path,
                    },
                    indent=2,
                )
            )
            staging.rename(local)
        return cls(model, processor, config)

    def _prepare(self, text, images, max_images=2):
        if len(images) > max_images:
            raise ValueError(f"Judge input exceeds its {max_images}-image request profile")
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": text},
                ],
            }
        ]
        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs = {"images": images} if images else {}
        inputs = self.processor(text=[formatted], return_tensors="pt", **kwargs)
        if inputs["input_ids"].shape[1] > self.config.max_input_tokens:
            raise ValueError("JUDGE_INPUT_OVERFLOW: the complete request cannot be truncated")
        return {key: value.to(self.config.device) for key, value in inputs.items()}

    @torch.no_grad()
    def answer_probability(self, text, images, *, audit=None):
        with self.lock:
            inputs = self._prepare(text, images)
            prefix = inputs["input_ids"].shape[1]
            scores = []
            eos = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if eos != 151645:
                raise ValueError("Unexpected judge EOS token")
            for answer in ("Yes", "No"):
                ids = self.processor.tokenizer.encode(answer, add_special_tokens=False) + [eos]
                labels = torch.tensor([ids], device=self.config.device, dtype=torch.long)
                arguments = dict(inputs)
                arguments["input_ids"] = torch.cat((inputs["input_ids"], labels), dim=1)
                arguments["attention_mask"] = torch.cat(
                    (inputs["attention_mask"], torch.ones_like(labels)),
                    dim=1,
                )
                if "mm_token_type_ids" in inputs:
                    arguments["mm_token_type_ids"] = torch.cat(
                        (inputs["mm_token_type_ids"], torch.zeros_like(labels)),
                        dim=1,
                    )
                self.model.model.rope_deltas = None
                positions = torch.arange(
                    prefix - 1, prefix + len(ids) - 1, device=self.config.device
                )
                logits = self.model(
                    **arguments,
                    use_cache=False,
                    logits_to_keep=positions,
                    return_dict=True,
                ).logits.float()
                scores.append(logits.log_softmax(-1).gather(-1, labels[..., None]).sum())
                self.forwards += 1
            self.queries += 1
            value = torch.sigmoid(scores[0] - scores[1]).item()
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError("Nonfinite judge likelihood")
            if audit is not None:
                grid = inputs.get("image_grid_thw")
                audit.update(
                    executed=True,
                    probability=value,
                    image_grid_thw=grid.detach().cpu().tolist() if grid is not None else None,
                    input_tokens=prefix,
                )
            return value

    def check(
        self,
        *,
        atom_id,
        request,
        property_text,
        images,
        roles,
        kind="semantic",
        conflict=False,
        semantic_sufficient=True,
        exact_pass=None,
        structured=None,
        audit=None,
    ):
        arguments = {"request": request, "roles": roles, "property": property_text}
        calibration = self.thresholds.get(kind)
        threshold = calibration["threshold"] if calibration else self.config.threshold
        if self.require_calibration and (
            calibration is None or not calibration.get("confirmed", False)
        ):
            raise ValueError(f"The {kind} judge route has no confirmed calibration")
        first_audit = {} if audit is not None else None
        second_audit = {} if audit is not None else None
        first = self.answer_probability(TEMPLATES[0].format(**arguments), images,
            **({"audit": first_audit} if audit is not None else {}))
        if structured is not None:
            conflict = conflict or structured.known and (first >= threshold) != structured.passed
        second_required = 0.2 <= first <= 0.8 or conflict or kind in {"count", "text", "identity"}
        second = (
            self.answer_probability(TEMPLATES[1].format(**arguments), images,
                **({"audit": second_audit} if audit is not None else {}))
            if second_required
            else first
        )
        known = abs(first - second) <= 0.25 and not conflict and semantic_sufficient
        if calibration is not None and not calibration.get("confirmed", False):
            known = False
        value = min(first, second)
        evidence = ()
        if structured is not None:
            known = known and structured.known and (second >= threshold) == structured.passed
            value = min(value, structured.value) if kind == "identity" else structured.value
            exact_pass = structured.exact_pass
            evidence = structured.evidence
        if audit is not None:
            reasons = []
            if abs(first - second) > 0.25:
                reasons.append("template_disagreement")
            if conflict:
                reasons.append("conflict")
            if not semantic_sufficient:
                reasons.append("semantic_insufficient")
            if calibration is not None and not calibration.get("confirmed", False):
                reasons.append("unconfirmed_calibration")
            if structured is not None:
                if not structured.known:
                    reasons.append("structured_unknown")
                if (second >= threshold) != structured.passed:
                    reasons.append("structured_semantic_disagreement")
            audit.update(
                schema="frozen_judge_probability_audit_v1",
                kind=kind,
                roles=roles,
                first=first_audit,
                second=second_audit if second_required else {
                    "executed": False, "probability": None, "image_grid_thw": None,
                    "input_tokens": None,
                },
                second_required=bool(second_required),
                second_executed=bool(second_required),
                decision_second_value=second,
                known_rejection_reasons=reasons,
                judge_threshold=threshold,
            )
        return AtomScore(atom_id, value, known, threshold, exact_pass, evidence)

    @torch.no_grad()
    def generate_json(self, text, images=(), max_new_tokens=1024):
        with self.lock:
            inputs = self._prepare(text, list(images), max_images=4)
            self.model.model.rope_deltas = None
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            tokens = output[0, inputs["input_ids"].shape[1] :]
            decoded = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if decoded.startswith("```json") and decoded.endswith("```"):
                decoded = decoded[7:-3].strip()
            return json.loads(decoded)
