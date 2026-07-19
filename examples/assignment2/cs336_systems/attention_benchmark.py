"""Benchmark vanilla, compiled and FlashAttention variants."""

from __future__ import annotations

import itertools
from timeit import default_timer

import torch

from .flash_attention import flash_attention_triton


def vanilla_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] >= torch.arange(
            k.shape[-2], device=q.device
        )[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    return torch.softmax(scores, dim=-1) @ v


def timed(operation, repetitions: int = 100, warmup: int = 5) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = default_timer()
    for _ in range(repetitions):
        operation()
    torch.cuda.synchronize()
    return (default_timer() - start) * 1000 / repetitions


def sweep() -> list[dict[str, object]]:
    device = torch.device("cuda")
    results: list[dict[str, object]] = []
    compiled = torch.compile(vanilla_attention)
    for d, sequence_length in itertools.product(
        [16, 32, 64, 128], [256, 1024, 4096, 8192, 16384]
    ):
        try:
            q = torch.randn(8, sequence_length, d, device=device, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)
            upstream = torch.randn_like(q)
            row = {"d": d, "sequence_length": sequence_length}
            for name, implementation in [
                ("pytorch", vanilla_attention),
                ("compiled", compiled),
                ("flash", flash_attention_triton),
            ]:
                row[f"{name}_forward_ms"] = timed(
                    lambda implementation=implementation: implementation(q, k, v, True)
                )

                def forward_backward(implementation=implementation):
                    q.grad = k.grad = v.grad = None
                    implementation(q, k, v, True).backward(upstream)

                row[f"{name}_forward_backward_ms"] = timed(forward_backward)
            results.append(row)
        except torch.OutOfMemoryError:
            results.append({"d": d, "sequence_length": sequence_length, "oom": True})
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    import pandas as pd

    print(pd.DataFrame(sweep()).to_string(index=False))
