"""Modern decoder-only Transformer components, implemented without torch.nn layers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """A bias-free linear map with weights stored as ``[d_out, d_in]``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class Embedding(nn.Module):
    """Map integer IDs to rows of a learnable embedding table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize by root-mean-square without subtracting the mean."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float.square(), dim=-1, keepdim=True) + self.eps)
        return ((x_float / rms) * self.weight.float()).to(input_dtype)


def silu(x: Tensor) -> Tensor:
    """SiLU(x) = x * sigmoid(x)."""

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """The gated feed-forward network ``W2(SiLU(W1 x) * W3 x)``."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 64 * math.ceil((8 * d_model / 3) / 64)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along one named dimension."""

    shifted = x - x.amax(dim=dim, keepdim=True)
    numerator = torch.exp(shifted)
    return numerator / numerator.sum(dim=dim, keepdim=True)


class RotaryPositionalEmbedding(nn.Module):
    """Apply pairwise RoPE rotations to the final dimension of Q or K."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires an even d_k, got {d_k}")
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute attention for arbitrary leading batch-like dimensions."""

    d_k = query.shape[-1]
    scores = torch.einsum("... q d, ... k d -> ... q k", query, key) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = softmax(scores, dim=-1)
    return torch.einsum("... q k, ... k d -> ... q d", probabilities, value)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE on Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        *leading, sequence_length, _ = x.shape
        return x.reshape(*leading, sequence_length, self.num_heads, self.d_head).transpose(-3, -2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        x = x.transpose(-3, -2).contiguous()
        *leading, sequence_length, _, _ = x.shape
        return x.reshape(*leading, sequence_length, self.d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        positions = torch.arange(sequence_length, device=x.device)
        causal_mask = positions[:, None] >= positions[None, :]
        attended = scaled_dot_product_attention(query, key, value, causal_mask)
        return self.output_proj(self._merge_heads(attended))


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with two residual branches."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    """A complete decoder-only language model that returns unnormalized logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError(
                f"sequence length {token_ids.shape[-1]} exceeds context length {self.context_length}"
            )
        x = self.token_embeddings(token_ids)
        positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        for layer in self.layers:
            x = layer(x, token_positions=positions)
        return self.lm_head(self.ln_final(x))
