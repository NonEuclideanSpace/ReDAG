import json
import math
import re
import threading
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .cache import make_fp32_cache, validate_cache_precision
from .config import (
    ACTOR_TARGETS,
    ActorConfig,
    adapter_directory,
    validate_actor_architecture,
    validate_actor_tokenizer,
)
from .head import head_backward, head_log_probs
from .precision import (
    VISUAL_PRECISION_PROFILE,
    actor_matmul_precision,
    configure_visual_layer_norms,
)
from .records import ActionSample, ActorSnapshot
from .visual import image_features


def validate_model_loading(info):
    missing = info.get("missing_keys", [])
    unexpected = [key for key in info.get("unexpected_keys", []) if not key.startswith("mtp.")]
    mismatched = info.get("mismatched_keys", [])
    errors = info.get("error_msgs", [])
    if missing or unexpected or mismatched or errors:
        raise RuntimeError(
            f"Actor checkpoint did not load strictly: missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}, errors={errors}"
        )


def prepare_actor_device(config, require_fast_kernels=True):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if not config.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The NF4 actor requires a configured CUDA device")
    torch.cuda.set_device(torch.device(config.device))
    if require_fast_kernels and not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError(
            "The actor requires the locked flash-linear-attention and causal-conv1d kernels"
        )


def validate_quantized_base(model, config):
    from bitsandbytes.nn import Linear4bit

    validate_actor_architecture(model, config)
    modules = dict(model.named_modules())
    quantized = {name: module for name, module in modules.items() if isinstance(module, Linear4bit)}
    expected = {
        name
        for name in modules
        if re.fullmatch(ACTOR_TARGETS, name) and not name.endswith(("in_proj_a", "in_proj_b"))
    }
    if set(quantized) != expected or len(quantized) != config.expected_quantized_modules:
        raise ValueError(
            f"Actor NF4 conversion must cover exactly {config.expected_quantized_modules} registered language projections"
        )
    quantized_parameters = sum(
        module.in_features * module.out_features for module in quantized.values()
    )
    if quantized_parameters != config.expected_quantized_parameters:
        raise ValueError("Actor NF4 projection dimensions do not match its registered checkpoint")
    resident_parameters = 0
    for parameter in model.parameters():
        if parameter.__class__.__name__ == "Params4bit":
            state = parameter.quant_state
            if state is None or state.quant_type != "nf4" or not state.nested:
                raise ValueError("Actor weights require serialized NF4 double-quantization state")
            resident_parameters += math.prod(state.shape)
        else:
            resident_parameters += parameter.numel()
    if resident_parameters != config.expected_base_parameters:
        raise ValueError(
            f"Expected {config.expected_base_parameters} resident actor parameters, found {resident_parameters}"
        )
    for name, module in quantized.items():
        if module.compute_dtype != torch.bfloat16:
            raise TypeError(f"Actor NF4 projection must compute in BF16: {name}")


def validate_actor_precision(model):
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            if parameter.dtype != torch.float32:
                raise TypeError(f"Actor adapters must use FP32: {name}")
        elif parameter.__class__.__name__ == "Params4bit":
            if parameter.requires_grad or parameter.dtype != torch.uint8:
                raise TypeError(f"Quantized actor weights must remain frozen uint8: {name}")
        elif parameter.is_floating_point():
            expected = (
                torch.float32
                if "norm" in name.lower() or name.endswith(("A_log", "dt_bias"))
                else torch.bfloat16
            )
            if parameter.dtype != expected or parameter.requires_grad:
                raise TypeError(
                    f"Frozen actor parameter has incompatible precision or gradients: {name}"
                )


def export_nf4_base(config: ActorConfig):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    destination = Path(config.quantized_model)
    if destination.exists():
        manifest_path = destination / "base_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("model_id") == config.model_id
                and manifest.get("revision") == config.revision
            ):
                return destination
        raise FileExistsError(
            f"The existing quantized model directory belongs to another or incomplete export: {destination}"
        )
    prepare_actor_device(config, require_fast_kernels=False)
    source = config.source_path or config.model_id
    source_options = (
        {"local_files_only": True} if config.source_path else {"revision": config.revision}
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
        llm_int8_skip_modules=["lm_head", "model.visual", "in_proj_a", "in_proj_b"],
    )
    model, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
        source,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": config.device},
        attn_implementation="sdpa",
        output_loading_info=True,
        **source_options,
    )
    validate_model_loading(loading_info)
    validate_quantized_base(model, config)
    staging = destination.with_name(destination.name + ".preparing")
    if staging.exists():
        raise FileExistsError(f"An unfinished quantization export exists: {staging}")
    model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
    processor = AutoProcessor.from_pretrained(
        source, max_pixels=config.max_image_pixels, **source_options
    )
    validate_actor_tokenizer(processor.tokenizer, config)
    processor.save_pretrained(staging)
    (staging / "base_manifest.json").write_text(
        json.dumps(
            {
                "model_id": config.model_id,
                "revision": config.revision,
                "quantization": "nf4-double",
                "compute_dtype": "bfloat16",
                "source_path": config.source_path,
            },
            indent=2,
        )
    )
    staging.rename(destination)
    return destination


class ActorWorker:
    def __init__(self, model, processor, config: ActorConfig):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = torch.device(config.device)
        self.lock = threading.RLock()
        self.has_reference = "reference" in model.peft_config
        self._policy_names = tuple(
            name for name, _ in model.named_parameters() if ".policy." in name and "lora_" in name
        )
        self._set_adapter("policy", training=False)

    @classmethod
    def load(cls, config: ActorConfig):
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        prepare_actor_device(config)
        path = Path(config.quantized_model)
        if not (path / "base_manifest.json").is_file():
            raise FileNotFoundError(f"Export the fixed NF4 base before loading the actor: {path}")
        manifest = json.loads((path / "base_manifest.json").read_text())
        if manifest["model_id"] != config.model_id or manifest["revision"] != config.revision:
            raise ValueError("The serialized NF4 base does not match the actor configuration")
        model, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map={"": config.device},
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        validate_model_loading(loading_info)
        validate_quantized_base(model, config)
        target_names = [
            name for name, _ in model.named_modules() if re.fullmatch(ACTOR_TARGETS, name)
        ]
        if len(target_names) != config.expected_targets:
            raise ValueError(
                f"Expected {config.expected_targets} language LoRA targets, found {len(target_names)}"
            )
        if not getattr(model, "is_loaded_in_4bit", False):
            raise ValueError("The actor base must be the serialized NF4 model")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        for name, parameter in model.named_parameters():
            if parameter.__class__.__name__ == "Params4bit" or not parameter.is_floating_point():
                continue
            preserve_fp32 = "norm" in name.lower() or name.endswith(("A_log", "dt_bias"))
            parameter.data = parameter.data.to(torch.float32 if preserve_fp32 else torch.bfloat16)
        configure_visual_layer_norms(model.model.visual)
        if config.policy_adapter:
            model = PeftModel.from_pretrained(
                model,
                adapter_directory(config.policy_adapter),
                adapter_name="policy",
                is_trainable=True,
            )
        else:
            lora = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=ACTOR_TARGETS
            )
            model = get_peft_model(model, lora, adapter_name="policy")
        if config.reference_adapter:
            model.load_adapter(
                adapter_directory(config.reference_adapter, "reference"),
                adapter_name="reference",
                is_trainable=False,
            )
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        validate_actor_precision(model)
        actual = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name and ".policy." in name
        )
        if actual != config.expected_adapter_parameters:
            raise ValueError(
                f"Expected {config.expected_adapter_parameters} actor adapter parameters, found {actual}"
            )
        base = model.get_base_model()
        if config.gradient_checkpointing:
            base.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        base.model.visual.gradient_checkpointing_disable()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        processor = AutoProcessor.from_pretrained(path, max_pixels=config.max_image_pixels)
        validate_actor_tokenizer(processor.tokenizer, config)
        return cls(model, processor, config)

    @property
    def base(self):
        return self.model.get_base_model()

    def policy_parameters(self):
        names = set(self._policy_names)
        return [parameter for name, parameter in self.model.named_parameters() if name in names]

    def _set_adapter(self, name, training):
        self.model.set_adapter(name, inference_mode=not training)
        self.model.eval()
        if training:
            self.base.model.language_model.train()
        self.base.model.visual.eval()
        names = set(self._policy_names)
        for parameter_name, parameter in self.model.named_parameters():
            parameter.requires_grad_(training and name == "policy" and parameter_name in names)
        self.base.config.use_cache = not training
        self.base.model.rope_deltas = None

    def _position_ids(self, ids, token_types, attention, grid):
        position, delta = self.base.model.get_rope_index(
            input_ids=ids,
            mm_token_type_ids=token_types,
            image_grid_thw=grid,
            attention_mask=attention,
        )
        text_positions = attention.long().cumsum(-1) - 1
        text_positions.masked_fill_(attention == 0, 0)
        return torch.cat((text_positions.unsqueeze(0), position), dim=0), delta

    @torch.no_grad()
    def prepare(self, messages, image_artifact_ids=(), metadata=None):
        with self.lock:
            processed = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                preserve_thinking=False,
                tools=None,
                return_dict=True,
                return_tensors="pt",
            )
            tensors = {
                name: value if name == "pixel_values" else value.to(self.device)
                for name, value in processed.items()
                if isinstance(value, torch.Tensor)
            }
            ids = tensors["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError("Actor requests must have microbatch size one")
            if ids.shape[-1] > self.config.max_input_tokens:
                raise ValueError(
                    f"VIEW_OVERFLOW: {ids.shape[-1]} tokens exceed {self.config.max_input_tokens}"
                )
            attention = tensors.setdefault("attention_mask", torch.ones_like(ids))
            if not bool(attention.all()):
                raise ValueError("A single action snapshot must not contain prefix padding")
            grid = tensors.get("image_grid_thw")
            if grid is not None and grid.shape[0] > self.config.max_images:
                raise ValueError("Actor image budget exceeded")
            if grid is not None and "mm_token_type_ids" not in tensors:
                raise ValueError("The processor did not return multimodal token types")
            token_types = tensors.setdefault("mm_token_type_ids", torch.zeros_like(ids))
            positions = (
                ((ids[0] == self.base.config.image_token_id) & (token_types[0] == 1))
                .nonzero()
                .flatten()
            )
            features = None
            visual_files = ()
            if grid is not None:
                features, visual_files = image_features(
                    self.base.model, tensors, image_artifact_ids, self.config, self.device
                )
                if features.shape[0] != positions.numel():
                    raise ValueError("Prefix image slots do not match frozen visual features")
                image_tokens = grid.prod(-1) // self.base.config.vision_config.spatial_merge_size**2
                if bool((image_tokens > 256).any()):
                    raise ValueError("An image exceeds the 256-token visual budget")
            self.base.model.rope_deltas = None
            position_ids, delta = self._position_ids(ids, token_types, attention, grid)
            saved = {name: value.detach().cpu().clone() for name, value in tensors.items()}
            details = dict(metadata or {})
            details.update(
                {
                    "model_id": self.config.model_id,
                    "model_revision": self.config.revision,
                    "vision_precision_profile": VISUAL_PRECISION_PROFILE,
                    "enable_thinking": False,
                    "preserve_thinking": False,
                    "max_pixels": self.config.max_image_pixels,
                    "image_patch_offsets": (
                        torch.cat(
                            (
                                torch.zeros(1, dtype=torch.long, device=grid.device),
                                grid.prod(-1).cumsum(0),
                            )
                        )
                        .cpu()
                        .tolist()
                        if grid is not None
                        else [0]
                    ),
                    "image_token_offsets": (
                        torch.cat(
                            (
                                torch.zeros(1, dtype=torch.long, device=grid.device),
                                image_tokens.cumsum(0),
                            )
                        )
                        .cpu()
                        .tolist()
                        if grid is not None
                        else [0]
                    ),
                }
            )
            return ActorSnapshot(
                saved,
                positions.cpu(),
                features.cpu() if features is not None else None,
                position_ids.cpu(),
                delta.cpu(),
                tuple(image_artifact_ids),
                details,
                visual_files,
            )

    @torch.no_grad()
    def _assemble(self, snapshot, completion=None):
        if (
            snapshot.image_features is not None
            and snapshot.metadata.get("vision_precision_profile") != VISUAL_PRECISION_PROFILE
        ):
            raise ValueError("Actor visual snapshot uses a different precision profile")
        identity = {"model_id": self.config.model_id, "model_revision": self.config.revision}
        if any(
            snapshot.metadata.get(key, value if self.config.model_id == "Qwen/Qwen3.8-27B" else None)
            != value
            for key, value in identity.items()
        ):
            raise ValueError("Actor snapshot belongs to a different or unregistered checkpoint")
        prefix = snapshot.tensors["input_ids"].to(self.device)
        completion = (
            torch.empty(0, dtype=torch.long, device=self.device)
            if completion is None
            else completion.to(self.device)
        )
        ids = torch.cat((prefix, completion.reshape(1, -1)), dim=-1)
        embeddings = self.base.get_input_embeddings()(ids)
        if snapshot.image_features is not None:
            embeddings[0, snapshot.prefix_image_positions.to(self.device)] = (
                snapshot.image_features.to(self.device, embeddings.dtype)
            )
        attention = torch.cat(
            (
                snapshot.tensors["attention_mask"].to(self.device),
                torch.ones((1, completion.numel()), dtype=torch.long, device=self.device),
            ),
            dim=-1,
        )
        token_types = torch.cat(
            (
                snapshot.tensors["mm_token_type_ids"].to(self.device),
                torch.zeros((1, completion.numel()), dtype=torch.long, device=self.device),
            ),
            dim=-1,
        )
        grid = snapshot.tensors.get("image_grid_thw")
        grid = grid.to(self.device) if grid is not None else None
        positions, _ = self._position_ids(ids, token_types, attention, grid)
        if not torch.equal(positions[..., : snapshot.prefix_length].cpu(), snapshot.position_ids):
            raise ValueError("Replay position IDs differ from the saved action prefix")
        return embeddings, attention, positions

    @torch.no_grad()
    def sample(self, snapshot, seed, behavior_version, probe_positions=()):
        with self.lock, actor_matmul_precision():
            self._set_adapter("policy", training=False)
            embeddings, attention, positions = self._assemble(snapshot)
            cache = make_fp32_cache(self.base.config)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            output = self.base.model.language_model(
                inputs_embeds=embeddings,
                position_ids=positions,
                attention_mask=attention,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            hidden = output.last_hidden_state[:, -1]
            tokens, log_probs, probes = [], [], {}
            eos_ids = set(self.config.eos_token_ids)
            finish_reason = "length"
            for index in range(self.config.max_action_tokens):
                logits = self.base.lm_head(hidden).float()[0] / self.config.temperature
                distribution = F.log_softmax(logits, dim=-1)
                token = torch.multinomial(distribution.exp(), 1, generator=generator)
                tokens.append(token.item())
                log_probs.append(distribution[token].cpu())
                if index in probe_positions:
                    probes[index] = logits.cpu()
                if token.item() in eos_ids:
                    finish_reason = "eos"
                    break
                sequence_position = snapshot.prefix_length + index
                text_position = torch.tensor([[[sequence_position]]], device=self.device)
                vision_position = (
                    text_position + snapshot.rope_delta.to(self.device).reshape(1, 1, 1)
                ).expand(3, 1, 1)
                position = torch.cat((text_position, vision_position), dim=0)
                attention = torch.cat(
                    (attention, torch.ones((1, 1), dtype=attention.dtype, device=self.device)),
                    dim=-1,
                )
                embedding = self.base.get_input_embeddings()(token.reshape(1, 1))
                output = self.base.model.language_model(
                    inputs_embeds=embedding,
                    position_ids=position,
                    attention_mask=attention,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                hidden = output.last_hidden_state[:, -1]
            validate_cache_precision(cache)
            token_ids = torch.tensor(tokens, dtype=torch.long)
            text_tokens = tokens[:-1] if finish_reason == "eos" else tokens
            text = self.processor.tokenizer.decode(
                text_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            return ActionSample(
                snapshot,
                token_ids,
                torch.cat(log_probs),
                torch.ones(len(tokens), dtype=torch.bool),
                text,
                finish_reason,
                seed,
                behavior_version,
                probes,
                self.config.numerics_profile,
            )

    def _hidden(self, snapshot, tokens):
        self.base.model.rope_deltas = None
        embeddings, attention, positions = self._assemble(snapshot, tokens)
        output = self.base.model.language_model(
            inputs_embeds=embeddings,
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        start = snapshot.prefix_length - 1
        return output.last_hidden_state[0, start : start + tokens.numel()]

    def sft_backward(self, snapshot, target_ids, target_mask=None, weight=1.0):
        with self.lock, actor_matmul_precision():
            tokens = torch.as_tensor(target_ids, dtype=torch.long, device=self.device)
            mask = (
                torch.ones_like(tokens, dtype=torch.bool)
                if target_mask is None
                else torch.as_tensor(target_mask, dtype=torch.bool, device=self.device)
            )
            if tokens.numel() > self.config.max_action_tokens or not mask.any():
                raise ValueError(
                    "SFT actions must fit the action budget and have supervised tokens"
                )
            self._set_adapter("policy", training=True)
            hidden = self._hidden(snapshot, tokens)
            return head_backward(
                hidden,
                self.base.lm_head,
                tokens,
                mask,
                weight / mask.sum().item(),
                chunk_size=self.config.head_chunk_size,
            )

    def rl_backward(self, sample, advantage, coefficient, beta=0.02):
        with self.lock, actor_matmul_precision():
            self._validate_sample_numerics(sample)
            if not self.has_reference:
                raise RuntimeError("Freeze the SFT reference before RL")
            tokens = sample.token_ids.to(self.device)
            mask = sample.response_mask.to(self.device)
            self._set_adapter("reference", training=False)
            with torch.no_grad():
                reference_hidden = self._hidden(sample.snapshot, tokens).detach()
            self._set_adapter("policy", training=True)
            hidden = self._hidden(sample.snapshot, tokens)
            return head_backward(
                hidden,
                self.base.lm_head,
                tokens,
                mask,
                coefficient,
                advantage,
                reference_hidden,
                beta,
                self.config.temperature,
                self.config.head_chunk_size,
            )

    @torch.no_grad()
    def replay_log_probs(self, sample, adapter="policy"):
        with self.lock, actor_matmul_precision():
            self._validate_sample_numerics(sample)
            self._set_adapter(adapter, training=False)
            tokens = sample.token_ids.to(self.device)
            hidden = self._hidden(sample.snapshot, tokens)
            return head_log_probs(
                hidden,
                self.base.lm_head,
                tokens,
                self.config.temperature,
                self.config.head_chunk_size,
            ).cpu()

    @torch.no_grad()
    def distribution_probe(self, sample, positions=(0, 15, 31)):
        with self.lock, actor_matmul_precision():
            self._validate_sample_numerics(sample)
            self._set_adapter("policy", training=False)
            tokens = sample.token_ids.to(self.device)
            hidden = self._hidden(sample.snapshot, tokens)
            positions = [position for position in positions if position < tokens.numel()]
            return {
                position: F.log_softmax(
                    self.base.lm_head(hidden[position : position + 1]).float()[0]
                    / self.config.temperature,
                    dim=-1,
                ).cpu()
                for position in positions
            }

    def _validate_sample_numerics(self, sample):
        if sample.numerics_profile != self.config.numerics_profile:
            raise ValueError("Sampled behavior belongs to a different actor numerical profile")

    def diagnose(
        self,
        snapshot,
        seed=20261027,
        behavior_version="preflight",
        mean_limit=0.01,
        p99_limit=0.05,
        kl_limit=1e-3,
    ):
        with self.lock, actor_matmul_precision():
            sample = self.sample(snapshot, seed, behavior_version, probe_positions=(0, 15, 31))
            replay = self.replay_log_probs(sample)
            differences = (replay - sample.behavior_log_probs).abs()
            probes = self.distribution_probe(sample, positions=tuple(sample.probe_logits))
            probe_kls = []
            for position, behavior_logits in sample.probe_logits.items():
                behavior_log_probs = F.log_softmax(behavior_logits.float(), dim=-1)
                probe_kls.append(
                    float(
                        (behavior_log_probs.exp() * (behavior_log_probs - probes[position])).sum()
                    )
                )
            for parameter in self.policy_parameters():
                parameter.grad = None
            self._set_adapter("policy", training=True)
            tokens = sample.token_ids.to(self.device)
            hidden = self._hidden(sample.snapshot, tokens)
            with torch.no_grad():
                training_log_probs = head_log_probs(
                    hidden.detach(),
                    self.base.lm_head,
                    tokens,
                    self.config.temperature,
                    self.config.head_chunk_size,
                ).cpu()
            head_backward(
                hidden,
                self.base.lm_head,
                tokens,
                sample.response_mask.to(self.device),
                1.0 / tokens.numel(),
                temperature=self.config.temperature,
                chunk_size=self.config.head_chunk_size,
            )
            gradient_squared = sum(
                float(parameter.grad.float().square().sum())
                for parameter in self.policy_parameters()
                if parameter.grad is not None
            )
            whitelist = set(self._policy_names)
            forbidden_gradients = [
                name
                for name, parameter in self.model.named_parameters()
                if name not in whitelist and parameter.grad is not None
            ]
            training_difference = (training_log_probs - replay).abs()
            special_ids = [
                getattr(self.base.config, name, None)
                for name in (
                    "image_token_id",
                    "video_token_id",
                    "vision_start_token_id",
                    "vision_end_token_id",
                )
            ]
            special_ids = torch.tensor(
                [value for value in special_ids if value is not None]
                + [self.config.action_eos_token_id],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                special_hidden = self._hidden(snapshot, special_ids)
                special_log_probs = head_log_probs(
                    special_hidden,
                    self.base.lm_head,
                    special_ids,
                    self.config.temperature,
                    self.config.head_chunk_size,
                )
            result = {
                "numerics_profile": self.config.numerics_profile,
                "sampled_tokens": tokens.numel(),
                "mean_logp_error": float(differences.mean()),
                "p99_logp_error": float(torch.quantile(differences, 0.99)),
                "max_logp_error": float(differences.max()),
                "max_probe_kl": max(probe_kls, default=0.0),
                "training_mean_logp_error": float(training_difference.mean()),
                "training_p99_logp_error": float(torch.quantile(training_difference, 0.99)),
                "policy_gradient_norm": gradient_squared**0.5,
                "forbidden_gradients": forbidden_gradients,
                "special_completion_replays": bool(torch.isfinite(special_log_probs).all()),
            }
            for parameter in self.policy_parameters():
                parameter.grad = None
            self._set_adapter("policy", training=False)
            result["passed"] = bool(
                result["mean_logp_error"] <= mean_limit
                and result["p99_logp_error"] <= p99_limit
                and result["max_probe_kl"] <= kl_limit
                and result["training_mean_logp_error"] <= mean_limit
                and result["training_p99_logp_error"] <= p99_limit
                and gradient_squared > 0
                and not forbidden_gradients
                and result["special_completion_replays"]
            )
            if not result["passed"]:
                raise RuntimeError(f"Actor sampling/replay preflight failed: {result}")
            return result

    def save_policy(self, path):
        with self.lock:
            self.model.save_pretrained(
                path,
                selected_adapters=["policy"],
                safe_serialization=True,
                save_embedding_layers=False,
            )
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "actor_config.json").write_text(json.dumps(asdict(self.config), indent=2))
            return adapter_directory(path)

    def freeze_reference(self, path):
        with self.lock:
            if self.has_reference:
                raise RuntimeError("The fixed SFT reference already exists")
            policy_path = self.save_policy(path)
            self.model.load_adapter(policy_path, adapter_name="reference", is_trainable=False)
            named = dict(self.model.named_parameters())
            for name in self._policy_names:
                reference_name = name.replace(".policy.", ".reference.")
                reference = named[reference_name]
                reference.data = reference.data.float()
                if reference.data_ptr() == named[name].data_ptr() or not torch.equal(
                    reference, named[name]
                ):
                    raise RuntimeError(
                        "The SFT reference must be an independent exact adapter copy"
                    )
            self.has_reference = True
            self._set_adapter("policy", training=False)
            return policy_path

    def adapter_state(self):
        from peft import get_peft_model_state_dict

        with self.lock:
            names = ["policy", "reference"] if self.has_reference else ["policy"]
            return {
                name: {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in get_peft_model_state_dict(
                        self.model, adapter_name=name, save_embedding_layers=False
                    ).items()
                }
                for name in names
            }

    def restore_adapter_state(self, state):
        from peft import set_peft_model_state_dict

        with self.lock:
            for name, tensors in state.items():
                if name not in self.model.peft_config:
                    raise ValueError(f"Missing adapter required by checkpoint: {name}")
                set_peft_model_state_dict(self.model, tensors, adapter_name=name)
            self._set_adapter("policy", training=False)


prepare_nf4 = export_nf4_base
