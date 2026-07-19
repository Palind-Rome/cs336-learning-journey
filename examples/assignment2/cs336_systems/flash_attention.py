"""Tiled PyTorch and Triton implementations of FlashAttention-2.

The PyTorch implementation deliberately mirrors the online-softmax algorithm.
It is useful as a readable oracle for debugging the fused Triton kernel.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:  # Triton is optional so the PyTorch reference still runs on CPU-only hosts.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the local CUDA toolchain
    triton = None
    tl = None


def _causal_mask(n_queries: int, n_keys: int, device: torch.device) -> torch.Tensor:
    query_index = torch.arange(n_queries, device=device)[:, None]
    key_index = torch.arange(n_keys, device=device)[None, :]
    return query_index >= key_index


def _flash_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equations 13--19 from the assignment, including recomputation."""

    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-1, -2) * scale
    if is_causal:
        scores = scores.masked_fill(
            ~_causal_mask(q.shape[-2], k.shape[-2], q.device),
            -1e6,
        )

    probabilities = torch.exp(scores - logsumexp.unsqueeze(-1))
    row_dot = (output * grad_output).sum(dim=-1, keepdim=True)
    grad_v = probabilities.transpose(-1, -2) @ grad_output
    grad_probabilities = grad_output @ v.transpose(-1, -2)
    grad_scores = probabilities * (grad_probabilities - row_dot)
    grad_q = grad_scores @ k * scale
    grad_k = grad_scores.transpose(-1, -2) @ q * scale
    return grad_q, grad_k, grad_v


class FlashAttentionPyTorch(torch.autograd.Function):
    """A tiled, pure-PyTorch implementation of online softmax attention."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("expected Q, K and V with shape (batch, sequence, d)")
        if q.shape[0] != k.shape[0] or k.shape != v.shape:
            raise ValueError("batch and K/V shapes must agree")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("Q and K embedding dimensions must agree")

        batch, n_queries, d = q.shape
        n_keys = k.shape[-2]
        query_tile_size = min(32, n_queries)
        key_tile_size = min(32, n_keys)
        scale = d**-0.5
        output = torch.empty_like(q)
        logsumexp = torch.empty(
            (batch, n_queries), device=q.device, dtype=torch.float32
        )

        # Each query tile is independent; the inner loop streams over K/V.
        for query_start in range(0, n_queries, query_tile_size):
            query_end = min(query_start + query_tile_size, n_queries)
            query = q[:, query_start:query_end]
            tile_rows = query_end - query_start
            running_max = torch.full(
                (batch, tile_rows),
                -torch.inf,
                device=q.device,
                dtype=torch.float32,
            )
            running_denominator = torch.zeros_like(running_max)
            running_output = torch.zeros(
                (batch, tile_rows, v.shape[-1]),
                device=q.device,
                dtype=torch.float32,
            )

            for key_start in range(0, n_keys, key_tile_size):
                key_end = min(key_start + key_tile_size, n_keys)
                key = k[:, key_start:key_end]
                value = v[:, key_start:key_end]
                scores = (query @ key.transpose(-1, -2)).float() * scale

                if is_causal:
                    query_indices = torch.arange(
                        query_start, query_end, device=q.device
                    )[:, None]
                    key_indices = torch.arange(
                        key_start, key_end, device=q.device
                    )[None, :]
                    scores = scores.masked_fill(
                        ~(query_indices >= key_indices), -1e6
                    )

                tile_max = scores.amax(dim=-1)
                next_max = torch.maximum(running_max, tile_max)
                correction = torch.exp(running_max - next_max)
                probabilities = torch.exp(scores - next_max.unsqueeze(-1))
                running_denominator = (
                    correction * running_denominator
                    + probabilities.sum(dim=-1)
                )
                running_output = (
                    correction.unsqueeze(-1) * running_output
                    + probabilities @ value.float()
                )
                running_max = next_max

            normalized = running_output / running_denominator.unsqueeze(-1)
            output[:, query_start:query_end] = normalized.to(output.dtype)
            logsumexp[:, query_start:query_end] = (
                running_max + torch.log(running_denominator)
            )

        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_backward(
            q, k, v, output, grad_output, logsumexp, ctx.is_causal
        )
        return grad_q, grad_k, grad_v, None


if triton is not None:  # pragma: no branch - definition depends on optional import

    @triton.jit
    def _flash_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        logsumexp_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        n_queries,
        n_keys,
        scale,
        d: tl.constexpr,
        query_tile_size: tl.constexpr,
        key_tile_size: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_start = query_tile_index * query_tile_size

        q_block_ptr = tl.make_block_ptr(
            q_ptr + batch_index * stride_qb,
            shape=(n_queries, d),
            strides=(stride_qq, stride_qd),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d),
            order=(1, 0),
        )
        k_block_ptr = tl.make_block_ptr(
            k_ptr + batch_index * stride_kb,
            shape=(n_keys, d),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(key_tile_size, d),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            v_ptr + batch_index * stride_vb,
            shape=(n_keys, d),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(key_tile_size, d),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr + batch_index * stride_ob,
            shape=(n_queries, d),
            strides=(stride_oq, stride_od),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d),
            order=(1, 0),
        )
        logsumexp_block_ptr = tl.make_block_ptr(
            logsumexp_ptr + batch_index * stride_lb,
            shape=(n_queries,),
            strides=(stride_lq,),
            offsets=(query_start,),
            block_shape=(query_tile_size,),
            order=(0,),
        )

        query = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        running_max = tl.full((query_tile_size,), -float("inf"), tl.float32)
        running_denominator = tl.zeros((query_tile_size,), tl.float32)
        accumulator = tl.zeros((query_tile_size, d), tl.float32)
        query_indices = query_start + tl.arange(0, query_tile_size)

        for key_start in range(0, n_keys, key_tile_size):
            key = tl.load(
                k_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            value = tl.load(
                v_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            scores = tl.dot(query, tl.trans(key)) * scale
            key_indices = key_start + tl.arange(0, key_tile_size)
            valid = (query_indices[:, None] < n_queries) & (
                key_indices[None, :] < n_keys
            )
            if is_causal:
                valid = valid & (query_indices[:, None] >= key_indices[None, :])
            scores = tl.where(valid, scores, -1.0e6)

            next_max = tl.maximum(running_max, tl.max(scores, axis=1))
            correction = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            running_denominator = (
                running_denominator * correction
                + tl.sum(probabilities, axis=1)
            )
            accumulator *= correction[:, None]
            accumulator = tl.dot(
                probabilities.to(value.dtype), value, acc=accumulator
            )
            running_max = next_max
            k_block_ptr = k_block_ptr.advance((key_tile_size, 0))
            v_block_ptr = v_block_ptr.advance((key_tile_size, 0))

        accumulator /= running_denominator[:, None]
        lse = running_max + tl.log(running_denominator)
        tl.store(
            output_block_ptr,
            accumulator.to(output_block_ptr.type.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(logsumexp_block_ptr, lse, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 with a fused Triton forward and recomputed backward."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is not installed; use FlashAttentionPyTorch")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("the Triton implementation requires CUDA tensors")
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
            raise ValueError("expected Q, K, V shaped (batch, sequence, d)")

        batch, n_queries, d = q.shape
        n_keys = k.shape[-2]
        if d > 128 or not math.log2(d).is_integer():
            raise ValueError("the tutorial kernel supports power-of-two d <= 128")

        output = torch.empty_like(q)
        logsumexp = torch.empty(
            (batch, n_queries), device=q.device, dtype=torch.float32
        )
        query_tile_size = 32
        key_tile_size = 32 if d >= 64 else 64
        grid = (triton.cdiv(n_queries, query_tile_size), batch)
        _flash_forward_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            *logsumexp.stride(),
            n_queries,
            n_keys,
            d**-0.5,
            d=d,
            query_tile_size=query_tile_size,
            key_tile_size=key_tile_size,
            is_causal=is_causal,
        )
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_backward(
            q,
            k,
            v,
            output,
            grad_output,
            logsumexp,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None


flash_attention_pytorch = FlashAttentionPyTorch.apply
flash_attention_triton = FlashAttentionTriton.apply
