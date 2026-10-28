from __future__ import annotations

from hashlib import sha256
import json
from typing import Protocol

import torch
import torch.nn.functional as F

from .types import ConditionBundle, ConditionRequest, TokenSpan


class TextEncoder(Protocol):
    def encode_text(self, texts: list[str]) -> torch.Tensor: ...


def patch_coverage(
    mask: torch.Tensor, canvas_hw: tuple[int, int], patch_size: int = 16
) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.float32, device="cpu")
    if value.shape != canvas_hw:
        raise ValueError(f"MASK_SHAPE: expected {canvas_hw}, received {tuple(value.shape)}")
    if not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)):
        raise ValueError("INVALID_MASK: mask values must be finite and within [0, 1]")
    if any(size % patch_size for size in canvas_hw):
        raise ValueError(
            "INVALID_CANVAS: canvas dimensions must be divisible by the latent patch size"
        )
    return F.avg_pool2d(value[None, None], patch_size, patch_size).flatten()


def rasterize(mask: torch.Tensor | None, canvas_hw: tuple[int, int]) -> tuple[torch.Tensor, bool]:
    count = canvas_hw[0] * canvas_hw[1] // 256
    if mask is None:
        return torch.zeros(count, dtype=torch.bool), False
    coverage = patch_coverage(mask, canvas_hw)
    if coverage.sum().item() <= 0:
        raise ValueError("EMPTY_REGION: a supplied region mask must have positive area")
    membership = coverage >= 0.25
    fallback = not membership.any().item()
    if fallback:
        membership[coverage.argmax()] = True
    return membership, fallback


def image_position_ids(canvas_hw: tuple[int, int], reference_count: int) -> torch.Tensor:
    height, width = (size // 16 for size in canvas_hw)
    grid = torch.cartesian_prod(torch.arange(height), torch.arange(width))
    blocks = []
    for index in range(reference_count + 1):
        ids = torch.zeros(height * width, 4, dtype=torch.int64)
        ids[:, 0] = 10 * index
        ids[:, 1:3] = grid
        blocks.append(ids)
    return torch.cat(blocks)


def text_position_ids(blocks: int, tokens_per_block: int) -> torch.Tensor:
    ids = torch.zeros(blocks, tokens_per_block, 4, dtype=torch.int64)
    ids[:, :, 3] = torch.arange(tokens_per_block)
    return ids.reshape(-1, 4)


def _connect(mask: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor) -> None:
    if rows.numel() and columns.numel():
        mask[rows[:, None], columns[None, :]] = True


class ConditionCompiler:
    def __init__(
        self,
        encoders: TextEncoder,
        max_regions: int = 2,
        max_references: int = 2,
        text_tokens: int = 512,
    ):
        self.encoders = encoders
        self.max_regions = max_regions
        self.max_references = max_references
        self.text_tokens = text_tokens

    def compile(self, request: ConditionRequest) -> ConditionBundle:
        if len(request.regions) > self.max_regions or len(request.references) > self.max_references:
            raise ValueError("CAPACITY_EXCEEDED: too many active regions or image references")
        if not request.global_text.strip():
            raise ValueError("EMPTY_TEXT: a global description is required")
        if request.cfg_scale < 1 or not torch.isfinite(torch.tensor(request.cfg_scale)):
            raise ValueError("INVALID_CFG: guidance must be finite and at least one")
        if any(size <= 0 or size % 16 for size in request.canvas_hw):
            raise ValueError("INVALID_CANVAS: dimensions must be positive multiples of 16")
        if len({region.region_id for region in request.regions}) != len(request.regions):
            raise ValueError("DUPLICATE_REGION: region IDs must be unique")
        source_indices = [
            index for index, ref in enumerate(request.references) if ref.role == "source"
        ]
        if len(source_indices) > 1 or source_indices and source_indices[0] != 0:
            raise ValueError("INVALID_REFERENCE_ORDER: the source must be the first reference")
        text_blocks = [request.global_text, *(region.text for region in request.regions)]
        if any(not text.strip() for text in text_blocks):
            raise ValueError("EMPTY_REGION_TEXT: active regions require a text condition")
        positive = self.encoders.encode_text(text_blocks).detach().cpu()
        empty = self.encoders.encode_text([""]).detach().cpu()
        if positive.shape != (len(text_blocks), self.text_tokens, 12288) or empty.shape != (
            1,
            self.text_tokens,
            12288,
        ):
            raise ValueError(
                "TEXT_ENCODER_SHAPE: Klein 9B requires 512-token blocks with 12288 channels"
            )
        if not torch.isfinite(positive).all() or not torch.isfinite(empty).all():
            raise ValueError("NONFINITE_EMBEDDING")
        positive = positive.reshape(1, -1, 12288).to(torch.bfloat16)
        negative = empty.expand(len(text_blocks), -1, -1).reshape(1, -1, 12288).to(torch.bfloat16)
        length = request.canvas_hw[0] * request.canvas_hw[1] // 256
        fallbacks = []
        target_membership, source_membership = [], []
        for region in request.regions:
            if region.target_mask is None and region.source_mask is None:
                raise ValueError("EMPTY_REGION: a region needs source or target membership")
            if region.source_mask is not None and not source_indices:
                raise ValueError(
                    "MISSING_SOURCE: a source mask requires an actual source reference"
                )
            target, target_fallback = rasterize(region.target_mask, request.canvas_hw)
            source, source_fallback = rasterize(region.source_mask, request.canvas_hw)
            target_membership.append(target)
            source_membership.append(source)
            if target_fallback:
                fallbacks.append(f"{region.region_id}:target")
            if source_fallback:
                fallbacks.append(f"{region.region_id}:source")
        targets = (
            torch.stack(target_membership)
            if target_membership
            else torch.zeros((0, length), dtype=torch.bool)
        )
        sources = (
            torch.stack(source_membership)
            if source_membership
            else torch.zeros((0, length), dtype=torch.bool)
        )
        reference_tokens = []
        for ref in request.references:
            latent = ref.latent.detach().cpu()
            if latent.shape != (1, length, 128) or not torch.isfinite(latent).all():
                raise ValueError(
                    "REFERENCE_LATENT_SHAPE: each reference must match the canvas latent grid"
                )
            reference_tokens.append(latent.to(torch.bfloat16))
        references = (
            torch.cat(reference_tokens, dim=1)
            if reference_tokens
            else torch.empty(1, 0, 128, dtype=torch.bfloat16)
        )
        text_length = self.text_tokens * len(text_blocks)
        total = text_length + length * (1 + len(request.references))
        mask = torch.zeros(total, total, dtype=torch.bool)
        global_ids = torch.arange(self.text_tokens)
        target_ids = torch.arange(text_length, text_length + length)
        source_ids = (
            torch.arange(text_length + length, text_length + 2 * length)
            if source_indices
            else torch.empty(0, dtype=torch.long)
        )
        base_visual = torch.cat((target_ids, source_ids))
        reference_ids = [
            torch.arange(text_length + length * (index + 1), text_length + length * (index + 2))
            for index in range(len(request.references))
        ]
        global_references = [
            reference_ids[index]
            for index, ref in enumerate(request.references)
            if ref.role != "source" and ref.scope == "global"
        ]
        base_keys = torch.cat((global_ids, base_visual, *global_references))
        _connect(mask, torch.cat((global_ids, base_visual)), base_keys)
        all_visual = torch.arange(text_length, total)
        appearance_membership = {}
        for index, ref in enumerate(request.references):
            if ref.role == "source":
                continue
            ids = reference_ids[index]
            _connect(mask, ids, torch.cat((ids, global_ids)))
            if ref.scope == "global":
                _connect(mask, ids, all_visual)
                for region_index, region in enumerate(request.regions):
                    if ref.image_id in region.appearance_bindings:
                        region_ids = torch.arange(
                            (region_index + 1) * self.text_tokens,
                            (region_index + 2) * self.text_tokens,
                        )
                        _connect(mask, region_ids, ids)
                        _connect(mask, ids, region_ids)
                        appearance_membership[f"{ref.image_id}:{region.region_id}"] = torch.ones(
                            length, dtype=torch.bool
                        )
                continue
            connected = False
            for region_index, region in enumerate(request.regions):
                if ref.image_id not in region.appearance_bindings:
                    continue
                supplied = [
                    ref.memberships[entity]
                    for entity in region.entity_ids
                    if entity in ref.memberships
                ]
                if region.region_id in ref.memberships:
                    supplied.append(ref.memberships[region.region_id])
                if not supplied:
                    raise ValueError(
                        f"MISSING_APPEARANCE_MEMBERSHIP: {ref.image_id} / {region.region_id}"
                    )
                subset = torch.zeros(length, dtype=torch.bool)
                for entity_mask in supplied:
                    tensor = torch.as_tensor(entity_mask)
                    if tensor.shape == (length,) and tensor.dtype == torch.bool:
                        membership, fallback = tensor.cpu(), False
                        if not membership.any():
                            raise ValueError("EMPTY_APPEARANCE_MEMBERSHIP")
                    else:
                        membership, fallback = rasterize(tensor, request.canvas_hw)
                    subset |= membership
                    if fallback:
                        fallbacks.append(f"{ref.image_id}:{region.region_id}")
                appearance_membership[f"{ref.image_id}:{region.region_id}"] = subset
                region_ids = torch.arange(
                    (region_index + 1) * self.text_tokens, (region_index + 2) * self.text_tokens
                )
                local_visual = torch.cat(
                    (
                        target_ids[targets[region_index]],
                        source_ids[sources[region_index]] if source_indices else source_ids,
                    )
                )
                _connect(mask, ids[subset], torch.cat((region_ids, local_visual)))
                _connect(mask, torch.cat((region_ids, local_visual)), ids[subset])
                connected = True
            if not connected:
                raise ValueError(
                    f"UNBOUND_REFERENCE: {ref.image_id} needs an active binding or explicit global scope"
                )
        reference_names = {ref.image_id for ref in request.references if ref.role != "source"}
        for index, region in enumerate(request.regions):
            if not set(region.appearance_bindings) <= reference_names:
                raise ValueError("UNKNOWN_APPEARANCE_BINDING")
            region_ids = torch.arange(
                (index + 1) * self.text_tokens, (index + 2) * self.text_tokens
            )
            local_visual = torch.cat(
                (
                    target_ids[targets[index]],
                    source_ids[sources[index]] if source_indices else source_ids,
                )
            )
            _connect(mask, region_ids, torch.cat((region_ids, local_visual)))
            _connect(mask, local_visual, region_ids)
        if not mask.any(dim=-1).all():
            raise ValueError("EMPTY_ATTENTION_ROW")
        spans = [TokenSpan("text", "global", 0, self.text_tokens)]
        spans.extend(
            TokenSpan(
                "text",
                region.region_id,
                (index + 1) * self.text_tokens,
                (index + 2) * self.text_tokens,
            )
            for index, region in enumerate(request.regions)
        )
        spans.append(TokenSpan("target", "target", text_length, text_length + length))
        spans.extend(
            TokenSpan(
                ref.role,
                ref.image_id,
                text_length + length * (index + 1),
                text_length + length * (index + 2),
            )
            for index, ref in enumerate(request.references)
        )
        layout = {
            "canvas": request.canvas_hw,
            "text_blocks": len(text_blocks),
            "text_tokens": self.text_tokens,
            "roles": [ref.role for ref in request.references],
        }
        signature = sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:24]
        bindings = dict(request.bindings)
        bindings["text_blocks"] = text_blocks
        bindings["reference_slots"] = [
            {
                "role": ref.role,
                "image_id": ref.image_id,
                "scope": ref.scope,
                "frame_id": ref.frame_id,
            }
            for ref in request.references
        ]
        condition_digest = sha256(
            json.dumps(
                {
                    "layout": layout,
                    "bindings": bindings,
                    "models": request.model_versions,
                    "compiler": "dag-agent-v3",
                    "cfg": request.cfg_scale,
                    "sigma_table": request.sigma_table_id,
                    "attention": request.attention_backend,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        for tensor in (positive, negative, references, mask, targets, sources):
            condition_digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            condition_digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        for name, tensor in sorted(appearance_membership.items()):
            condition_digest.update(name.encode())
            condition_digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return ConditionBundle(
            bundle_id=f"bundle:{condition_digest.hexdigest()}",
            canvas_hw=request.canvas_hw,
            positive_embeds=positive,
            negative_embeds=negative,
            text_ids=text_position_ids(len(text_blocks), self.text_tokens),
            image_ids=image_position_ids(request.canvas_hw, len(request.references)),
            reference_tokens=references,
            positive_mask=mask[None, None],
            negative_mask=mask[None, None],
            token_span_table=tuple(spans),
            target_membership=targets,
            source_membership=sources,
            appearance_membership=appearance_membership,
            cfg_scale=request.cfg_scale,
            sigma_table_id=request.sigma_table_id,
            attention_backend=request.attention_backend,
            layout_signature=signature,
            artifact_bindings=bindings,
            model_versions=dict(request.model_versions),
            raster_fallbacks=tuple(fallbacks),
        )
