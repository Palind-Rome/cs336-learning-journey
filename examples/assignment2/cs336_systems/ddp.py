"""A small DDP wrapper that overlaps all-reduce with backpropagation."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class DistributedDataParallel(nn.Module):
    """Replicate a module and average gradients as soon as they are ready.

    Every trainable parameter receives a post-accumulate hook. Autograd visits
    parameters in reverse layer order, so each asynchronous all-reduce starts as
    soon as that layer's gradient exists while earlier layers are still running.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before wrapping the model")
        self.module = module
        self.world_size = dist.get_world_size()
        self._pending: list[dist.Work] = []
        self._hook_handles: list[Any] = []

        # Rank 0 is the source of truth, including frozen parameters and buffers.
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        for parameter in self.module.parameters():
            if parameter.requires_grad:
                handle = parameter.register_post_accumulate_grad_hook(
                    self._make_gradient_hook()
                )
                self._hook_handles.append(handle)

    def _make_gradient_hook(self):
        def synchronize(parameter: nn.Parameter) -> None:
            if parameter.grad is None:
                return
            # Average rather than sum so the local mean loss matches a global
            # mean over the union of all per-rank minibatches.
            parameter.grad.div_(self.world_size)
            work = dist.all_reduce(parameter.grad, async_op=True)
            self._pending.append(work)

        return synchronize

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Make every asynchronously reduced gradient safe for optimizer.step."""

        for work in self._pending:
            work.wait()
        self._pending.clear()


def ddp_on_after_backward(
    model: DistributedDataParallel, optimizer: torch.optim.Optimizer | None = None
) -> None:
    del optimizer
    model.finish_gradient_synchronization()
