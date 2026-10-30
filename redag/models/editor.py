import math
import random

import torch
import torch.nn.functional as F

from redag.edit.compositor import defnorm_box_to_pixel_slice
from redag.learn.rl_trainer import (
    compute_flow_sde_drift_mean,
    compute_masked_gaussian_logprob,
    predict_cfg_velocity,
    sample_4_strata_indices,
)


@torch.no_grad()
def sample_flow_sde_trajectory(
    dit_model, vae_decoder, z_src, text_embeds, null_text_embeds, mask_latent,
    num_steps=50, eta=0.35, cfg_scale=3.5, seed=42,
):
    if z_src.ndim != 4 or z_src.shape[0] != 1:
        raise ValueError("A trajectory requires one BCHW source latent")
    if num_steps < 5 or eta <= 0 or not math.isfinite(eta):
        raise ValueError("Flow-SDE replay requires at least five steps and positive eta")
    if mask_latent.shape != (1, 1, *z_src.shape[-2:]):
        raise ValueError("The latent mask must have shape [1, 1, H, W]")
    if not torch.all((mask_latent == 0) | (mask_latent == 1)) or not mask_latent.any():
        raise ValueError("The editable mask must be binary and nonempty")
    source = z_src.float()
    mask = mask_latent.to(device=source.device, dtype=torch.float32)
    rng_device = source.device if source.device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=rng_device).manual_seed(seed)

    def noise():
        return torch.randn(source.shape, generator=generator, device=rng_device).to(source.device)

    dt = 1.0 / num_steps
    z_t, eps_src = noise(), noise()
    z_t = mask * z_t + (1 - mask) * eps_src
    t_steps, z_t_list, z_next_list, logp_old_list, mu_old_list = [], [], [], [], []
    for k in range(num_steps):
        t_val, t_next = (num_steps - k) / num_steps, (num_steps - k - 1) / num_steps
        t = torch.tensor([t_val], device=source.device, dtype=torch.float32)
        velocity = predict_cfg_velocity(
            dit_model, z_t, source, text_embeds, null_text_embeds, t, cfg_scale=cfg_scale,
        ).float()
        mu, sigma = compute_flow_sde_drift_mean(z_t, velocity, t, dt=dt, eta=eta)
        z_next = mu + sigma.reshape(1, 1, 1, 1) * math.sqrt(dt) * noise()
        old_logprob = compute_masked_gaussian_logprob(z_next, mu, sigma, mask, dt=dt) if bool((sigma > 0).all()) else None
        z_next = (1 - mask) * ((1 - t_next) * source + t_next * eps_src) + mask * z_next
        if not torch.isfinite(z_next).all():
            raise FloatingPointError("Flow-SDE generated a non-finite latent")
        t_steps.append(t[0].detach().cpu())
        z_t_list.append(z_t[0].detach().cpu())
        z_next_list.append(z_next[0].detach().cpu())
        mu_old_list.append(mu[0].detach().cpu())
        logp_old_list.append(old_logprob[0].detach().cpu() if old_logprob is not None else None)
        z_t = z_next
    crop = vae_decoder(z_t).float().clamp(0, 1)
    if crop.ndim != 4 or crop.shape[:2] != (1, 3) or not torch.isfinite(crop).all():
        raise ValueError("The VAE decoder must return a finite [1, 3, H, W] image")
    return {
        "edited_crop": crop[0], "t_steps": t_steps, "z_t_list": z_t_list,
        "z_next_list": z_next_list, "logp_old_list": logp_old_list, "mu_old_list": mu_old_list,
        "strata_indices": sample_4_strata_indices(num_steps, rng=random.Random(seed)),
    }


class FluxEditor:
    def __init__(self, pipeline, num_steps=50, eta=0.35, cfg_scale=3.5):
        if getattr(pipeline, 'config', {}).get('is_distilled', False) or getattr(pipeline.transformer.config, 'guidance_embeds', False):
            raise ValueError('Flow-SDE requires an undistilled Klein Base model with external classifier-free guidance')
        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        self.dit_model = self
        self.batched_cfg = True
        self.num_steps, self.eta, self.cfg_scale = num_steps, eta, cfg_scale
        self.vae.requires_grad_(False).eval()
        self.pipeline.text_encoder.requires_grad_(False).eval()

    @classmethod
    def from_pretrained(
        cls, model_id="black-forest-labs/FLUX.2-klein-base-9B", device="cuda",
        dtype=torch.bfloat16, adapter_path=None, trainable=False,
    ):
        from diffusers import Flux2KleinPipeline
        from peft import LoraConfig

        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if Flux2KleinPipeline.load_config(model_id).get('is_distilled', False):
            raise ValueError('Use FLUX.2-klein-base-9B for the 50-step SFT/Flow-SDE pipeline')
        pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        pipeline.transformer.requires_grad_(False).eval()
        if adapter_path:
            pipeline.transformer.load_lora_adapter(
                adapter_path, prefix=None, adapter_name="default", use_safetensors=True,
                weight_name="pytorch_lora_weights.safetensors",
            )
        elif trainable:
            pipeline.transformer.add_adapter(LoraConfig(
                r=16, lora_alpha=16, lora_dropout=0.0,
                target_modules=["to_q", "to_k", "to_v", "to_qkv_mlp_proj", "to_out.0"],
            ))
        for name, parameter in pipeline.transformer.named_parameters():
            parameter.requires_grad_(trainable and "lora_" in name)
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if trainable:
            pipeline.transformer.enable_gradient_checkpointing()
        return cls(pipeline)

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def lora_parameters(self):
        return [p for name, p in self.transformer.named_parameters() if "lora_" in name]

    def snapshot(self):
        return {name: p.detach().cpu().clone() for name, p in self.transformer.named_parameters() if "lora_" in name}

    def load_snapshot(self, snapshot):
        parameters = dict(self.transformer.named_parameters())
        if set(snapshot) != {name for name in parameters if "lora_" in name}:
            raise ValueError("Adapter snapshot parameters do not match the editor")
        with torch.no_grad():
            for name, tensor in snapshot.items():
                parameters[name].copy_(tensor.to(parameters[name]))

    def save_adapter(self, path):
        if not self.lora_parameters():
            raise ValueError("No editor LoRA adapter is loaded")
        self.transformer.save_lora_adapter(path, adapter_name="default")

    def train(self, mode=True):
        self.transformer.train(mode)
        self.vae.eval()
        self.pipeline.text_encoder.eval()
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, z_t, z_src, text_embeds, timesteps):
        if z_t.shape != z_src.shape or z_t.ndim != 4:
            raise ValueError("Source and noisy latents must share a BCHW shape")
        batch, channels, height, width = z_t.shape
        tokens = z_t.flatten(2).transpose(1, 2)
        source_tokens = z_src.flatten(2).transpose(1, 2)
        rows, columns = torch.meshgrid(
            torch.arange(height, device=z_t.device),
            torch.arange(width, device=z_t.device), indexing="ij",
        )
        ids = torch.stack((torch.zeros_like(rows), rows, columns, torch.zeros_like(rows)), -1)
        ids = ids.reshape(1, -1, 4).expand(batch, -1, -1)
        source_ids = ids.clone()
        source_ids[:, :, 0] = 10
        text_ids = torch.zeros(batch, text_embeds.shape[1], 4, device=z_t.device, dtype=torch.long)
        text_ids[:, :, 3] = torch.arange(text_embeds.shape[1], device=z_t.device)
        prediction = self.transformer(
            hidden_states=torch.cat((tokens, source_tokens), 1).to(dtype=self.dtype),
            encoder_hidden_states=text_embeds.to(device=z_t.device, dtype=self.dtype),
            timestep=timesteps.to(device=z_t.device, dtype=self.dtype), guidance=None,
            img_ids=torch.cat((ids, source_ids), 1), txt_ids=text_ids, return_dict=False,
        )[0][:, :height * width]
        return prediction.transpose(1, 2).reshape(batch, channels, height, width).float()

    @torch.no_grad()
    def encode_image(self, image):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 1:
            raise ValueError("An RGB image is required")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Image values must be finite and within [0, 1]")
        stride = 2 ** len(self.vae.config.block_out_channels)
        scale = max(1, math.ceil(stride / min(image.shape[-2:])))
        if scale > 1:
            image = F.interpolate(image.float(), scale_factor=scale, mode="bilinear", align_corners=False)
        pad_h, pad_w = (-image.shape[-2]) % stride, (-image.shape[-1]) % stride
        if pad_h or pad_w:
            # whole latent tokens only: decode_latents then crops instead of replicating missing border rows
            mode = "reflect" if pad_h < image.shape[-2] and pad_w < image.shape[-1] else "replicate"
            image = F.pad(image.float(), (0, pad_w, 0, pad_h), mode=mode)
        image = image.to(device=self.device, dtype=self.vae.dtype) * 2 - 1
        latent = self.vae.encode(image).latent_dist.mode()
        if latent.shape[-2] % 2 or latent.shape[-1] % 2:
            latent = F.pad(latent, (0, latent.shape[-1] % 2, 0, latent.shape[-2] % 2), mode="replicate")
        batch, channels, height, width = latent.shape
        latent = latent.reshape(batch, channels, height // 2, 2, width // 2, 2)
        latent = latent.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)
        mean = self.vae.bn.running_mean.float().reshape(1, -1, 1, 1)
        std = (self.vae.bn.running_var.float() + self.vae.config.batch_norm_eps).sqrt().reshape(1, -1, 1, 1)
        return (latent.float() - mean) / std

    def decode_latents(self, latent, output_size=None):
        mean = self.vae.bn.running_mean.float().reshape(1, -1, 1, 1)
        std = (self.vae.bn.running_var.float() + self.vae.config.batch_norm_eps).sqrt().reshape(1, -1, 1, 1)
        latent = latent.float() * std + mean
        batch, channels, height, width = latent.shape
        latent = latent.reshape(batch, channels // 4, 2, 2, height, width)
        latent = latent.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, 2 * height, 2 * width)
        image = (self.vae.decode(latent.to(dtype=self.vae.dtype), return_dict=False)[0].float() + 1) / 2
        if output_size is not None:
            scale = max(1, math.ceil((2 ** len(self.vae.config.block_out_channels)) / min(output_size)))
            height, width = (int(size) * scale for size in output_size)
            if image.shape[-2] < height or image.shape[-1] < width:
                image = F.pad(image, (0, max(0, width - image.shape[-1]), 0, max(0, height - image.shape[-2])), mode="replicate")
            image = image[:, :, :height, :width]
            if scale > 1:
                image = F.interpolate(image, size=output_size, mode="bilinear", align_corners=False)
        return image.clamp(0, 1)

    def encode_mask(self, mask, latent):
        if not torch.all((mask == 0) | (mask == 1)) or not mask.any():
            raise ValueError("The editable mask must be binary and nonempty")
        stride = 2 ** len(self.vae.config.block_out_channels)
        scale = max(1, math.ceil(stride / min(mask.shape[-2:])))
        mask = mask.to(device=self.device, dtype=torch.float32)
        if scale > 1:
            mask = F.interpolate(mask, scale_factor=scale, mode="nearest")
        height, width = (size * stride for size in latent.shape[-2:])
        mask = F.pad(mask, (0, max(0, width - mask.shape[-1]), 0, max(0, height - mask.shape[-2])), mode="replicate")
        # a latent token is editable when any pixel it covers is editable (nearest sampling kept only the
        # top-left pixel of every 16 x 16 cell and could drop thin or off-grid masks entirely)
        mask = F.max_pool2d(mask[:, :, :height, :width], kernel_size=stride, stride=stride)
        if not mask.any():
            raise ValueError("The edit mask disappears at latent resolution")
        return mask

    @torch.no_grad()
    def encode_text(self, instruction):
        return self.pipeline.encode_prompt(instruction, device=self.device)[0].detach()

    def _crop(self, image, action):
        x1, y1, x2, y2 = defnorm_box_to_pixel_slice(action.box, *image.shape[-2:])
        return image[:, y1:y2, x1:x2]

    @torch.no_grad()
    def prepare(self, image, action, mask=None):
        crop = self._crop(image, action)
        z_src = self.encode_image(crop)
        if mask is None:
            latent_mask = torch.ones((1, 1, *z_src.shape[-2:]), device=self.device)
        else:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3 or mask.shape[0] != 1 or mask.shape[-2:] != image.shape[-2:]:
                raise ValueError("The edit mask must match the full canvas")
            latent_mask = self.encode_mask(self._crop(mask, action).unsqueeze(0), z_src)
        return {
            "z_src": z_src, "text_embeds": self.encode_text(action.instruction),
            "null_text_embeds": self.encode_text(""), "mask_latent": latent_mask,
        }

    @torch.no_grad()
    def prepare_sft(self, source, target, instruction, mask):
        if source.shape != target.shape:
            raise ValueError("SFT source and target must have identical image dimensions")
        z_src, z_0 = self.encode_image(source), self.encode_image(target)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape != (z_src.shape[0], 1, *source.shape[-2:]):
            raise ValueError("The SFT mask must match the source image")
        latent_mask = self.encode_mask(mask, z_src)
        return {"z_src": z_src, "z_0": z_0, "text_embeds": self.encode_text(instruction), "mask_latent": latent_mask}

    @torch.no_grad()
    def sample(self, image, action, seed=42, conditioning=None):
        conditioning = self.prepare(image, action) if conditioning is None else conditioning
        crop_size = self._crop(image, action).shape[-2:]
        return sample_flow_sde_trajectory(
            self, lambda z: self.decode_latents(z, output_size=crop_size), **conditioning,
            num_steps=self.num_steps, eta=self.eta, cfg_scale=self.cfg_scale, seed=seed,
        )

    def render(self, image, action, seed=42):
        return self.sample(image, action, seed=seed)["edited_crop"].to(image.device)
