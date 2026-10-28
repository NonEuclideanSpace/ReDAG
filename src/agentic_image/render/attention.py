from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint


def query_chunk_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_size: int = 128,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("The attention query chunk size must be positive")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Attention tensors must have shape BSHD")
    outputs = []
    for start in range(0, query.shape[1], chunk_size):
        stop = min(start + chunk_size, query.shape[1])
        chunk_mask = None if mask is None else mask[..., start:stop, :]

        def attend(q, k, v, allowed):
            with torch.autocast(q.device.type, enabled=False):
                scores = torch.matmul(
                    q.float().transpose(1, 2), k.float().transpose(1, 2).transpose(-1, -2)
                ) / math.sqrt(q.shape[-1])
                if allowed is not None:
                    scores = (
                        scores.masked_fill(~allowed, -torch.inf)
                        if allowed.dtype == torch.bool
                        else scores + allowed.float()
                    )
                probabilities = scores.softmax(dim=-1)
                output = torch.matmul(probabilities, v.float().transpose(1, 2)).transpose(1, 2)
            return output.to(q.dtype)

        arguments = (query[:, start:stop], key, value, chunk_mask)
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in arguments[:3]):
            output = checkpoint(attend, *arguments, use_reentrant=False, preserve_rng_state=False)
        else:
            output = attend(*arguments)
        outputs.append(output)
    return torch.cat(outputs, dim=1)


class QueryChunkFlux2Processor:
    def __init__(self, chunk_size: int = 128):
        self.chunk_size = chunk_size

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        from diffusers.models.transformers.transformer_flux2 import (
            _get_qkv_projections,
            apply_rotary_emb,
        )

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if attn.added_kv_proj_dim is not None:
            encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
            encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            query = torch.cat((encoder_query, query), dim=1)
            key = torch.cat((encoder_key, key), dim=1)
            value = torch.cat((encoder_value, value), dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        output = query_chunk_attention(query, key, value, attention_mask, self.chunk_size).flatten(
            2, 3
        )
        if encoder_hidden_states is not None:
            encoder_output, output = output.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
            )
            encoder_output = attn.to_add_out(encoder_output)
        output = attn.to_out[1](attn.to_out[0](output))
        return (output, encoder_output) if encoder_hidden_states is not None else output


class QueryChunkFlux2ParallelProcessor:
    def __init__(self, chunk_size: int = 128):
        self.chunk_size = chunk_size

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.transformers.transformer_flux2 import apply_rotary_emb

        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp = torch.split(
            projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        output = query_chunk_attention(query, key, value, attention_mask, self.chunk_size).flatten(
            2, 3
        )
        return attn.to_out(torch.cat((output, attn.mlp_act_fn(mlp)), dim=-1))


def configure_attention(transformer, backend: str, chunk_size: int = 128) -> None:
    import inspect

    if backend == "query_chunk":
        for block in transformer.transformer_blocks:
            block.attn.set_processor(QueryChunkFlux2Processor(chunk_size))
        for block in transformer.single_transformer_blocks:
            block.attn.set_processor(QueryChunkFlux2ParallelProcessor(chunk_size))
    elif backend == "_native_efficient":
        transformer.set_attention_backend(backend)
    else:
        raise ValueError(f"Unsupported attention backend: {backend}")
    for name, processor in transformer.attn_processors.items():
        if "attention_mask" not in inspect.signature(processor.__call__).parameters:
            raise ValueError(f"Attention processor {name} cannot consume region masks")
