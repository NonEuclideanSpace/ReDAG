from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import re
from threading import RLock
from typing import Callable
from uuid import uuid4

import torch
from torch.utils.checkpoint import checkpoint

from .attention import configure_attention
from .sde import (
    FlowSDE,
    gaussian_mean_logp,
    gaussian_noise,
    sigma_schedule,
    stratified_steps,
    stream_seed,
    velocity_vjp,
)
from .types import ConditionBundle, RenderTrace, RendererConfig, ReplayStep


LORA_TARGETS = r"^(?:transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)|single_transformer_blocks\.\d+\.attn\.(?:to_qkv_mlp_proj|to_out))$"


class KleinRenderer:
    def __init__(self, transformer, config: RendererConfig):
        self.transformer = transformer
        self.config = config
        self.device = torch.device(config.device)
        self.lock = RLock()
        self.behavior_version = "sft-initial"
        self._active_adapter = None
        self._reference_exists = False
        self.control_adapter = None
        self._base_inference = False
        self._configure()

    @classmethod
    def load(cls, config: RendererConfig) -> KleinRenderer:
        from diffusers import Flux2Transformer2DModel

        transformer, loading = Flux2Transformer2DModel.from_pretrained(
            config.model_id,
            subfolder="transformer",
            revision=config.revision,
            torch_dtype=torch.bfloat16,
            local_files_only=config.local_files_only,
            output_loading_info=True,
        )
        problems = {
            key: loading.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading.get(key)
        }
        if problems:
            raise RuntimeError(f"Incomplete Klein transformer checkpoint: {problems}")
        return cls(transformer.to(config.device), config)

    def _configure(self) -> None:
        from peft import LoraConfig

        if getattr(self.transformer.config, "guidance_embeds", None) is not False:
            raise ValueError("Klein base 9B must have guidance_embeds=false")
        targets = [
            name
            for name, module in self.transformer.named_modules()
            if re.fullmatch(LORA_TARGETS, name) and isinstance(module, torch.nn.Linear)
        ]
        if self.config.strict_architecture:
            expected = {
                "in_channels": 128,
                "num_layers": 8,
                "num_single_layers": 24,
                "joint_attention_dim": 12288,
                "num_attention_heads": 32,
                "attention_head_dim": 128,
            }
            mismatches = {
                key: (getattr(self.transformer.config, key, None), value)
                for key, value in expected.items()
                if getattr(self.transformer.config, key, None) != value
            }
            if mismatches or len(targets) != 112:
                raise ValueError(
                    f"Unexpected Klein 9B architecture: {mismatches}, LoRA targets={len(targets)}"
                )
            if (
                sum(parameter.numel() for parameter in self.transformer.parameters())
                != 9_078_581_248
            ):
                raise ValueError("Unexpected Klein 9B base parameter count")
        if not targets:
            raise ValueError("No Klein attention projections match the LoRA configuration")
        self.transformer.requires_grad_(False)
        self.lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=LORA_TARGETS,
            init_lora_weights=True,
        )
        self.transformer.add_adapter(self.lora_config, adapter_name="policy")
        self._set_adapter("policy")
        if (
            self.config.strict_architecture
            and sum(parameter.numel() for parameter in self.policy_parameters()) != 31_981_568
        ):
            raise ValueError("Unexpected Klein 9B policy adapter parameter count")
        if self.config.enable_controls:
            from .controls import ControlAdapters

            self.control_adapter = ControlAdapters(self.transformer.inner_dim).to(
                self.device, torch.float32
            )
            self.control_adapter.attach(self.transformer)
        configure_attention(
            self.transformer, self.config.attention_backend, self.config.query_chunk_size
        )
        if self.config.gradient_checkpointing:

            def checkpoint_function(module, *args):
                return checkpoint(module, *args, use_reentrant=False, preserve_rng_state=False)

            self.transformer.enable_gradient_checkpointing(
                gradient_checkpointing_func=checkpoint_function
            )
        if self.config.policy_path:
            self.load_policy_state(
                torch.load(self.config.policy_path, map_location="cpu", weights_only=True)
            )
        if self.config.reference_path:
            self.freeze_reference(
                torch.load(self.config.reference_path, map_location="cpu", weights_only=True)
            )

    def _set_adapter(self, name: str) -> None:
        if self._base_inference:
            if name != "policy":
                raise RuntimeError("The base inference context cannot evaluate a learned reference")
            self.transformer.disable_adapters()
            self.transformer.requires_grad_(False)
            return
        if name == "reference" and not self._reference_exists:
            raise RuntimeError("The fixed SFT reference has not been initialized")
        if self._active_adapter != name:
            self.transformer.set_adapter(name)
            self._active_adapter = name
        for parameter_name, parameter in self.transformer.named_parameters():
            adapter_parameter = ".lora_A." in parameter_name or ".lora_B." in parameter_name
            if adapter_parameter:
                parameter.data = parameter.data.float()
            parameter.requires_grad_(
                adapter_parameter and ".policy." in parameter_name and name == "policy"
            )
        if self.control_adapter is not None:
            self.control_adapter.set_adapter(name)

    @contextmanager
    def base_policy(self):
        with self.lock:
            previous = self._active_adapter or "policy"
            was_training = self.transformer.training
            if self._base_inference:
                raise RuntimeError("Base policy contexts cannot be nested")
            self._base_inference = True
            self.transformer.disable_adapters()
            self.transformer.requires_grad_(False)
            try:
                with torch.no_grad():
                    yield self
            finally:
                self._base_inference = False
                self.transformer.enable_adapters()
                self._active_adapter = None
                self._set_adapter(previous)
                self.transformer.train(was_training)

    def policy_parameters(self) -> list[torch.nn.Parameter]:
        parameters = [
            parameter
            for name, parameter in self.transformer.named_parameters()
            if (".lora_A.policy." in name or ".lora_B.policy." in name)
        ]
        if self.control_adapter is not None:
            parameters.extend(self.control_adapter.policy.parameters())
        return parameters

    def policy_state(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in get_peft_model_state_dict(
                self.transformer, adapter_name="policy"
            ).items()
        }
        if self.control_adapter is not None:
            state.update(
                {
                    f"control_adapter.{key}": tensor.detach().cpu().clone()
                    for key, tensor in self.control_adapter.policy.state_dict().items()
                }
            )
        return state

    def reference_state(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        if not self._reference_exists:
            raise RuntimeError("The fixed SFT reference has not been initialized")
        state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in get_peft_model_state_dict(
                self.transformer, adapter_name="reference"
            ).items()
        }
        if self.control_adapter is not None:
            state.update(
                {
                    f"control_adapter.{key}": tensor.detach().cpu().clone()
                    for key, tensor in self.control_adapter.reference.state_dict().items()
                }
            )
        return state

    def load_policy_state(self, state: dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict

        expected = self.policy_state()
        if state.keys() != expected.keys() or any(
            state[key].shape != expected[key].shape for key in expected
        ):
            raise ValueError("Policy adapter state does not match the current architecture")
        set_peft_model_state_dict(
            self.transformer,
            {
                key: tensor
                for key, tensor in state.items()
                if not key.startswith("control_adapter.")
            },
            adapter_name="policy",
        )
        if self.control_adapter is not None:
            self.control_adapter.policy.load_state_dict(
                {
                    key.removeprefix("control_adapter."): tensor
                    for key, tensor in state.items()
                    if key.startswith("control_adapter.")
                },
                strict=True,
            )
        self._set_adapter("policy")

    def freeze_reference(self, state: dict[str, torch.Tensor] | None = None) -> None:
        from peft import set_peft_model_state_dict

        with self.lock:
            state = self.policy_state() if state is None else state
            expected = self.policy_state()
            if state.keys() != expected.keys() or any(
                state[key].shape != expected[key].shape for key in expected
            ):
                raise ValueError("Reference adapter state does not match the policy architecture")
            if not self._reference_exists:
                self.transformer.add_adapter(self.lora_config, adapter_name="reference")
                self._reference_exists = True
            set_peft_model_state_dict(
                self.transformer,
                {
                    key: tensor.detach().clone()
                    for key, tensor in state.items()
                    if not key.startswith("control_adapter.")
                },
                adapter_name="reference",
            )
            if self.control_adapter is not None:
                self.control_adapter.freeze_reference(
                    {
                        key.removeprefix("control_adapter."): tensor
                        for key, tensor in state.items()
                        if key.startswith("control_adapter.")
                    }
                )
            self._active_adapter = None
            self._set_adapter("policy")

    def save_adapters(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_state(), destination / "policy.pt")
        if self._reference_exists:
            torch.save(self.reference_state(), destination / "reference.pt")

    def _validate_bundle(self, bundle: ConditionBundle, state: torch.Tensor) -> None:
        if state.ndim != 3 or state.shape != (1, bundle.num_target_tokens, 128):
            raise ValueError("Renderer microbatch must contain one target latent with 128 channels")
        total_images = state.shape[1] + bundle.reference_tokens.shape[1]
        text_length = bundle.positive_embeds.shape[1]
        if bundle.image_ids.shape != (total_images, 4) or bundle.text_ids.shape != (text_length, 4):
            raise ValueError("Condition position IDs do not match the actual token layout")
        shape = (1, 1, total_images + text_length, total_images + text_length)
        if bundle.positive_mask.shape != shape or bundle.negative_mask.shape != shape:
            raise ValueError("Attention masks do not match the full text and image layout")
        if bundle.positive_mask.dtype != torch.bool or bundle.negative_mask.dtype != torch.bool:
            raise ValueError("Klein masks must be boolean with True meaning visible")
        if bundle.negative_embeds.shape != bundle.positive_embeds.shape:
            raise ValueError("CFG requires matching positive and negative text layouts")
        if bundle.attention_backend != self.config.attention_backend:
            raise ValueError("The condition and renderer attention backends do not match")

    def predict_branch(
        self,
        bundle: ConditionBundle,
        state: torch.Tensor,
        sigma: torch.Tensor | float,
        adapter: str = "policy",
        negative: bool = False,
    ) -> torch.Tensor:
        self._validate_bundle(bundle, state)
        self._set_adapter(adapter)
        embeddings = bundle.negative_embeds if negative else bundle.positive_embeds
        mask = bundle.negative_mask if negative else bundle.positive_mask
        model_input = torch.cat(
            (
                state.to(self.device, torch.bfloat16),
                bundle.reference_tokens.to(self.device, torch.bfloat16),
            ),
            dim=1,
        )
        timestep = torch.as_tensor(sigma, dtype=torch.float32, device=self.device).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(state.shape[0])
        if timestep.shape != (state.shape[0],):
            raise ValueError("One sigma is required per image")
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        if bundle.controls and self._base_inference:
            raise ValueError("The raw base model has no trained depth or pose control adapters")
        if bundle.controls and self.control_adapter is None:
            raise ValueError(
                "Depth or pose controls require the explicitly enabled control-adapter profile"
            )
        control_context = (
            self.control_adapter.condition(bundle.controls, bundle.num_target_tokens)
            if self.control_adapter is not None
            else nullcontext()
        )
        with context, control_context:
            output = self.transformer(
                hidden_states=model_input,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=embeddings.to(self.device, torch.bfloat16),
                txt_ids=bundle.text_ids.to(self.device),
                img_ids=bundle.image_ids.to(self.device),
                joint_attention_kwargs={"attention_mask": mask.to(self.device)},
                return_dict=False,
            )[0]
        return output[:, : bundle.num_target_tokens].float()

    def predict_velocity(
        self,
        bundle: ConditionBundle,
        state: torch.Tensor,
        sigma: torch.Tensor | float,
        adapter: str = "policy",
    ) -> torch.Tensor:
        positive = self.predict_branch(bundle, state, sigma, adapter, negative=False)
        if bundle.cfg_scale == 1:
            return positive
        negative = self.predict_branch(bundle, state, sigma, adapter, negative=True)
        return bundle.cfg_scale * positive + (1 - bundle.cfg_scale) * negative

    @torch.no_grad()
    def rollout(
        self,
        bundle: ConditionBundle,
        seed: int,
        behavior_version: str | None = None,
        render_id: str | None = None,
        steps: int | None = None,
        eta: float | None = None,
        replay_seed: int | None = None,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> RenderTrace:
        with self.lock:
            self.transformer.eval()
            bundle = bundle.to(self.device)
            steps = self.config.steps if steps is None else steps
            eta = self.config.eta if eta is None else eta
            schedule = sigma_schedule(steps, bundle.num_target_tokens)
            kernel = FlowSDE(schedule, eta)
            seed_init = stream_seed(seed, "init")
            state = gaussian_noise((1, bundle.num_target_tokens, 128), seed_init, self.device)
            states = [state.cpu()]
            log_probabilities = []
            step_seeds = []
            for index in range(steps):
                transition = kernel.transition(index)
                velocity = self.predict_velocity(bundle, state, transition.sigma, "policy")
                mean = transition.mean(state, velocity)
                step_seed = stream_seed(seed, index, "sde")
                next_state = mean + transition.std.to(self.device) * gaussian_noise(
                    tuple(state.shape), step_seed, self.device
                )
                if not torch.isfinite(next_state).all():
                    raise FloatingPointError(f"Nonfinite renderer state at transition {index}")
                log_probabilities.append(gaussian_mean_logp(next_state, mean, transition.std).cpu())
                step_seeds.append(step_seed)
                states.append(next_state.cpu())
                state = next_state
                if on_step is not None:
                    on_step(index, state)
            selected = stratified_steps(
                steps, 4, stream_seed(seed, "replay") if replay_seed is None else replay_seed
            )
            return RenderTrace(
                render_id or f"render:{uuid4().hex}",
                bundle.bundle_id,
                behavior_version or self.behavior_version,
                torch.stack(states),
                schedule,
                kernel.std,
                torch.stack(log_probabilities),
                seed_init,
                tuple(step_seeds),
                tuple(range(steps)),
                eta,
                bundle.cfg_scale,
                selected,
                {
                    "attention_backend": self.config.attention_backend,
                    "model_revision": self.config.revision,
                    "compiler_version": bundle.compiler_version,
                },
            )

    @torch.no_grad()
    def rollout_ode(self, bundle: ConditionBundle, seed: int, steps: int = 50) -> torch.Tensor:
        with self.lock:
            self.transformer.eval()
            bundle = bundle.to(self.device)
            sigmas = sigma_schedule(steps, bundle.num_target_tokens).to(self.device)
            state = gaussian_noise(
                (1, bundle.num_target_tokens, 128), stream_seed(seed, "init"), self.device
            )
            for index in range(steps):
                state = state + (sigmas[index + 1] - sigmas[index]) * self.predict_velocity(
                    bundle, state, sigmas[index], "policy"
                )
                if not torch.isfinite(state).all():
                    raise FloatingPointError(f"Nonfinite ODE state at transition {index}")
            return state.cpu()

    def backward_replay(
        self,
        bundle: ConditionBundle,
        trace: RenderTrace,
        advantage: float,
        coefficient: float,
        beta: float = 0.01,
        replay_steps: tuple[ReplayStep, ...] | None = None,
        expected_behavior_version: str | None = None,
        logp_tolerance: float = 2e-3,
    ) -> dict[str, float]:
        if trace.bundle_id != bundle.bundle_id or trace.cfg_scale != bundle.cfg_scale:
            raise ValueError("REPLAY_CONDITION_MISMATCH")
        if (
            expected_behavior_version is not None
            and trace.behavior_version != expected_behavior_version
        ):
            raise ValueError("REPLAY_VERSION_MISMATCH")
        if trace.stochastic_indices != tuple(range(trace.steps)):
            raise ValueError("Replay requires an entirely stochastic Flow-SDE trajectory")
        if trace.states.dtype != torch.float32:
            raise ValueError("Replay requires saved FP32 states")
        selected = trace.replay_steps if replay_steps is None else replay_steps
        if not selected or abs(sum(step.weight for step in selected) - 1) > 1e-6:
            raise ValueError("Replay step weights must represent the full time schedule")
        kernel = FlowSDE(trace.sigmas, trace.eta)
        if not torch.equal(kernel.std, trace.transition_std):
            raise ValueError("REPLAY_VARIANCE_MISMATCH")
        totals = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "max_logp_error": 0.0, "transitions": 0.0}
        with self.lock:
            bundle = bundle.to(self.device)
            self.transformer.train()
            for selected_step in selected:
                index = selected_step.step_index
                transition = kernel.transition(index)
                state = trace.states[index].detach().to(self.device)
                next_state = trace.states[index + 1].detach().to(self.device)
                with torch.no_grad():
                    reference_velocity = self.predict_velocity(
                        bundle, state, transition.sigma, "reference"
                    )
                    current_velocity = self.predict_velocity(
                        bundle, state, transition.sigma, "policy"
                    )
                    gradient, metrics = velocity_vjp(
                        state,
                        next_state,
                        current_velocity,
                        reference_velocity,
                        transition,
                        advantage,
                        coefficient * selected_step.weight,
                        beta,
                    )
                    error = abs(metrics["mean_logp"] - float(trace.stored_mean_logp[index]))
                    if error > logp_tolerance:
                        raise RuntimeError(
                            f"REPLAY_LOGP_MISMATCH: {error:.6g} exceeds {logp_tolerance:.6g}"
                        )
                self._set_adapter("policy")
                for negative, weight in ((False, bundle.cfg_scale), (True, 1 - bundle.cfg_scale)):
                    if weight == 0:
                        continue
                    context = (
                        torch.autograd.graph.save_on_cpu(pin_memory=True)
                        if self.config.saved_tensor_cpu_offload
                        else nullcontext()
                    )
                    with context:
                        velocity = self.predict_branch(
                            bundle, state, transition.sigma, "policy", negative
                        )
                        (weight * gradient * velocity).sum().backward()
                    del velocity
                for key in ("loss", "pg", "kl"):
                    totals[key] += metrics[key]
                totals["max_logp_error"] = max(totals["max_logp_error"], error)
                totals["transitions"] += 1
            return totals
