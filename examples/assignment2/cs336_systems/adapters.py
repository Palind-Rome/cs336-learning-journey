"""Drop-in adapter bodies for the official Assignment 2 test suite."""

from __future__ import annotations

import torch

from .ddp import DistributedDataParallel
from .flash_attention import FlashAttentionPyTorch, FlashAttentionTriton
from .fsdp import FullyShardedDataParallel
from .sharded_optimizer import ShardedOptimizer


def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> type:
    return FlashAttentionTriton


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return DistributedDataParallel(module)


def ddp_on_after_backward(
    ddp_model: DistributedDataParallel, optimizer: torch.optim.Optimizer
) -> None:
    del optimizer
    ddp_model.finish_gradient_synchronization()


def get_fsdp(
    module: torch.nn.Module, compute_dtype: torch.dtype | None = None
) -> torch.nn.Module:
    return FullyShardedDataParallel(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(
    fsdp_model: FullyShardedDataParallel, optimizer: torch.optim.Optimizer
) -> None:
    del optimizer
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(
    fsdp_model: FullyShardedDataParallel,
) -> dict[str, torch.Tensor]:
    return fsdp_model.gather_full_parameters()


def get_sharded_optimizer(
    params,
    optimizer_cls: type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
