"""Activation-checkpointing helpers used in the memory experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def checkpoint_sequential_blocks(
    blocks: Sequence[nn.Module],
    x: torch.Tensor,
    blocks_per_checkpoint: int,
) -> torch.Tensor:
    """Checkpoint consecutive groups without nesting (one recomputation pass)."""

    if blocks_per_checkpoint < 1:
        raise ValueError("blocks_per_checkpoint must be positive")
    for start in range(0, len(blocks), blocks_per_checkpoint):
        group = blocks[start : start + blocks_per_checkpoint]

        def run_group(value: torch.Tensor, group=group) -> torch.Tensor:
            for block in group:
                value = block(value)
            return value

        x = checkpoint(run_group, x, use_reentrant=False)
    return x


def recursively_checkpoint_blocks(
    blocks: Sequence[nn.Module], x: torch.Tensor
) -> torch.Tensor:
    """Recursively bisect a stack, reaching O(log N) peak activations.

    Ignoring constant-size checkpoint bookkeeping, the nested strategy keeps
    one branch at each recursion depth and recomputes work at every level. Peak
    activation memory is O(log N), while total compute is O(N log N).
    """

    if not blocks:
        return x
    if len(blocks) == 1:
        return blocks[0](x)
    split = math.ceil(len(blocks) / 2)

    def left(value: torch.Tensor) -> torch.Tensor:
        return recursively_checkpoint_blocks(blocks[:split], value)

    def right(value: torch.Tensor) -> torch.Tensor:
        return recursively_checkpoint_blocks(blocks[split:], value)

    x = checkpoint(left, x, use_reentrant=False)
    return checkpoint(right, x, use_reentrant=False)
