"""Memory-map-friendly random next-token batches."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample B windows of length T+1 and split them into aligned inputs/targets."""

    if len(dataset) <= context_length:
        raise ValueError("dataset needs at least context_length + 1 tokens")
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    inputs = np.stack([np.asarray(dataset[start : start + context_length]) for start in starts.tolist()])
    targets = np.stack([np.asarray(dataset[start + 1 : start + context_length + 1]) for start in starts.tolist()])
    x = torch.from_numpy(inputs.astype(np.int64, copy=False)).to(device)
    y = torch.from_numpy(targets.astype(np.int64, copy=False)).to(device)
    return x, y
