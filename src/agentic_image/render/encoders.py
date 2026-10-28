from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
import torch

from .types import EncoderConfig, klein_profile


def patchify(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4 or latents.shape[2] % 2 or latents.shape[3] % 2:
        raise ValueError("VAE latents must have even spatial dimensions")
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, channels * 4, height // 2, width // 2)
    )


def unpatchify(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4 or latents.shape[1] % 4:
        raise ValueError("Patched latents need channels divisible by four")
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(batch, channels // 4, 2, 2, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, channels // 4, height * 2, width * 2)
    )


def pack(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError("Latents must have shape BCHW")
    return latents.flatten(2).transpose(1, 2).contiguous()


def unpack(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != grid_hw[0] * grid_hw[1]:
        raise ValueError("Packed token count must match the requested spatial grid")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *grid_hw).contiguous()


class TensorDiskCache:
    def __init__(self, directory: str | None, max_bytes: int):
        self.directory = Path(directory) if directory else None
        self.max_bytes = max_bytes
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> torch.Tensor | None:
        if self.directory is None:
            return None
        path = self.directory / f"{key}.pt"
        if not path.exists():
            return None
        value = torch.load(path, weights_only=True, map_location="cpu")
        path.touch()
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.directory is None or self.max_bytes <= 0:
            return
        path = self.directory / f"{key}.pt"
        temporary = self.directory / f".{uuid4().hex}.pt"
        torch.save(value.detach().cpu(), temporary)
        temporary.replace(path)
        files = sorted(self.directory.glob("*.pt"), key=lambda item: item.stat().st_mtime)
        size = sum(item.stat().st_size for item in files)
        for item in files:
            if size <= self.max_bytes:
                break
            size -= item.stat().st_size
            item.unlink(missing_ok=True)


class FrozenEncoders:
    def __init__(self, text_encoder, tokenizer, vae, config: EncoderConfig):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.config = config
        self.device = torch.device(config.device)
        self.lock = RLock()
        self.cache = TensorDiskCache(config.cache_directory, config.cache_bytes)
        self.text_encoder.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)
        if config.strict_architecture:
            profile = klein_profile(config.profile)
            expected = {
                "model_type": "qwen3",
                "hidden_size": profile.encoder_hidden_size,
                "num_hidden_layers": profile.encoder_layers,
            }
            mismatches = {
                key: (getattr(self.text_encoder.config, key, None), value)
                for key, value in expected.items()
                if getattr(self.text_encoder.config, key, None) != value
            }
            if mismatches:
                raise ValueError(f"Unexpected paired {config.profile} text encoder: {mismatches}")
            if self.vae.config.latent_channels != 32 or tuple(self.vae.config.patch_size) != (2, 2):
                raise ValueError("The paired Klein VAE requires 32 channels and 2x2 patches")
        if (
            not hasattr(self.vae, "bn")
            or self.vae.bn.running_mean.numel() != 128
            or self.vae.bn.running_var.numel() != 128
        ):
            raise ValueError("The Klein VAE must contain 128-channel BN statistics")
        if (
            not torch.isfinite(self.vae.bn.running_mean).all()
            or not torch.isfinite(self.vae.bn.running_var).all()
        ):
            raise ValueError("Nonfinite VAE normalization statistics")
        if torch.any(self.vae.bn.running_var < 0):
            raise ValueError("Negative VAE normalization variance")

    @classmethod
    def load(cls, config: EncoderConfig) -> FrozenEncoders:
        from diffusers import AutoencoderKLFlux2
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        if config.strict_architecture:
            klein_profile(config.profile).validate_source(config.model_id, config.revision)
        options = {"revision": config.revision, "local_files_only": config.local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, subfolder="tokenizer", **options)
        encoder, encoder_loading = Qwen3ForCausalLM.from_pretrained(
            config.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
            **options,
        )
        vae, vae_loading = AutoencoderKLFlux2.from_pretrained(
            config.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            output_loading_info=True,
            **options,
        )
        for name, loading in (("text_encoder", encoder_loading), ("vae", vae_loading)):
            problems = {
                key: loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading.get(key)
            }
            if problems:
                raise RuntimeError(f"Incomplete Klein {name} checkpoint: {problems}")
        return cls(encoder.to(config.device), tokenizer, vae.to(config.device), config)

    @property
    def version_manifest(self) -> dict[str, str]:
        return {
            "encoder_model": self.config.model_id,
            "encoder_revision": self.config.revision,
            "encoder_profile": self.config.profile,
            "encoder_hidden_layers": ",".join(map(str, self.config.hidden_layers)),
            "encoder_text_width": str(self.config.text_hidden_size),
            "encoder_template": sha256(
                json.dumps(self.tokenizer.chat_template, sort_keys=True).encode()
            ).hexdigest(),
            "vae_revision": self.config.revision,
        }

    def _key(self, kind: str, payload: bytes) -> str:
        header = json.dumps(
            {
                "kind": kind,
                "versions": self.version_manifest,
                "tokens": self.config.text_block_tokens,
                "layers": self.config.hidden_layers,
                "codec": "klein-v3-profiled",
            },
            sort_keys=True,
        ).encode()
        return sha256(header + payload).hexdigest()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("At least one text block is required")
        with self.lock:
            outputs = []
            for text in texts:
                key = self._key("text", text.encode("utf-8"))
                cached = self.cache.get(key)
                if cached is not None:
                    self._validate_text(cached)
                    outputs.append(cached)
                    continue
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                unpadded = self.tokenizer(formatted, truncation=False)["input_ids"]
                if len(unpadded) > self.config.text_block_tokens:
                    raise ValueError(
                        f"TEXT_TOO_LONG: {len(unpadded)} template tokens exceed {self.config.text_block_tokens}"
                    )
                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=False,
                    max_length=self.config.text_block_tokens,
                ).to(self.device)
                result = self.text_encoder.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                embeddings = (
                    torch.cat(
                        [result.hidden_states[layer] for layer in self.config.hidden_layers], dim=-1
                    )
                    .to(torch.bfloat16)
                    .cpu()
                )
                self._validate_text(embeddings)
                self.cache.put(key, embeddings)
                outputs.append(embeddings)
            return torch.cat(outputs, dim=0)

    def _validate_text(self, embeddings: torch.Tensor) -> None:
        if (
            embeddings.shape != (1, self.config.text_block_tokens, self.config.text_hidden_size)
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("Text embeddings do not match the registered Klein encoder profile")

    @torch.no_grad()
    def encode_image(self, image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = ImageOps.exif_transpose(image)
            if "A" in image.getbands() or image.info.get("transparency") is not None:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
                image = Image.alpha_composite(background, rgba)
            image = np.asarray(image.convert("RGB")).copy()
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Array images must be uint8 HWC RGB")
            pixels = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 255
        else:
            pixels = image.detach().cpu()
            if pixels.ndim == 3:
                pixels = pixels[None]
            if pixels.ndim != 4 or pixels.shape[1] != 3 or not pixels.is_floating_point():
                raise ValueError("Tensor images must be float BCHW RGB in [0, 1]")
            pixels = pixels.float()
        if pixels.shape[0] != 1 or any(size % 16 for size in pixels.shape[-2:]):
            raise ValueError(
                "Image encoding expects one registered canvas with dimensions divisible by 16"
            )
        if not torch.isfinite(pixels).all() or torch.any((pixels < 0) | (pixels > 1)):
            raise ValueError("Invalid image values")
        key = self._key(
            "image", str(tuple(pixels.shape)).encode() + pixels.contiguous().numpy().tobytes()
        )
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            context = (
                torch.autocast(self.device.type, enabled=False)
                if self.device.type in {"cuda", "cpu"}
                else nullcontext()
            )
            with context:
                latent = self.vae.encode(
                    pixels.to(self.device, torch.float32) * 2 - 1
                ).latent_dist.mode()
                latent = patchify(latent.float())
                mean = self.vae.bn.running_mean.float().reshape(1, -1, 1, 1)
                std = (
                    (self.vae.bn.running_var.float() + self.vae.config.batch_norm_eps)
                    .sqrt()
                    .reshape(1, -1, 1, 1)
                )
                tokens = pack((latent - mean) / std).float().cpu()
            if tokens.shape[-1] != 128 or not torch.isfinite(tokens).all():
                raise ValueError("Invalid Klein VAE output")
            self.cache.put(key, tokens)
            return tokens

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, canvas_hw: tuple[int, int]) -> torch.Tensor:
        if tokens.shape[-1] != 128 or not torch.isfinite(tokens).all():
            raise ValueError("Invalid packed Klein latents")
        with self.lock:
            context = (
                torch.autocast(self.device.type, enabled=False)
                if self.device.type in {"cuda", "cpu"}
                else nullcontext()
            )
            with context:
                latent = unpack(
                    tokens.to(self.device, torch.float32), (canvas_hw[0] // 16, canvas_hw[1] // 16)
                )
                mean = self.vae.bn.running_mean.float().reshape(1, -1, 1, 1)
                std = (
                    (self.vae.bn.running_var.float() + self.vae.config.batch_norm_eps)
                    .sqrt()
                    .reshape(1, -1, 1, 1)
                )
                decoded = self.vae.decode(unpatchify(latent * std + mean), return_dict=False)[0]
                return (decoded.float() / 2 + 0.5).clamp(0, 1).cpu()

    def decode_image(self, tokens: torch.Tensor, canvas_hw: tuple[int, int]) -> Image.Image:
        pixels = self.decode(tokens, canvas_hw)
        if pixels.shape[0] != 1:
            raise ValueError("decode_image expects exactly one image")
        array = pixels[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
        return Image.fromarray(array)
