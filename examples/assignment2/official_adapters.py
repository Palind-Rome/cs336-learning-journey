"""Use these bodies as ``tests/adapters.py`` in the official repository."""

from __future__ import annotations

import torch

from cs336_systems.ddp import DistributedDataParallel
from cs336_systems.flash_attention import FlashAttentionPyTorch, FlashAttentionTriton
from cs336_systems.fsdp import FullyShardedDataParallel
from cs336_systems.sharded_optimizer import ShardedOptimizer


def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> type:
    return FlashAttentionTriton


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return DistributedDataParallel(module)


def ddp_on_after_backward(ddp_model, optimizer) -> None:
    del optimizer
    ddp_model.finish_gradient_synchronization()


def get_fsdp(module, compute_dtype: torch.dtype | None = None):
    return FullyShardedDataParallel(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model, optimizer) -> None:
    del optimizer
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model) -> dict[str, torch.Tensor]:
    return fsdp_model.gather_full_parameters()


def get_sharded_optimizer(params, optimizer_cls, **kwargs):
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
