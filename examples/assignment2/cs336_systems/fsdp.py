"""A teaching-sized fully-sharded data parallel implementation.

Linear and Embedding weights are flattened and sharded. Their custom autograd
functions all-gather just-in-time for forward, discard the full weight, gather
again during backward, and asynchronously average the full gradient before the
owner keeps only its shard. Norm parameters remain replicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclass
class _ShardMetadata:
    name: str
    parameter: nn.Parameter
    full_shape: tuple[int, ...]
    full_numel: int
    shard_size: int


class _ShardedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        shard: torch.Tensor,
        owner: "FullyShardedDataParallel",
        metadata: _ShardMetadata,
    ) -> torch.Tensor:
        full_weight = owner._gather_for_compute(shard, metadata)
        ctx.save_for_backward(inputs, shard)
        ctx.owner = owner
        ctx.shard_metadata = metadata
        # The staff Linear uses this exact contraction.
        return torch.einsum("...i,oi->...o", inputs, full_weight)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        inputs, shard = ctx.saved_tensors
        owner: FullyShardedDataParallel = ctx.owner
        metadata: _ShardMetadata = ctx.shard_metadata
        full_weight = owner._gather_for_compute(shard, metadata)

        grad_inputs = torch.einsum("...o,oi->...i", grad_output, full_weight)
        grad_weight = torch.einsum("...o,...i->oi", grad_output, inputs)
        placeholder = owner._queue_sharded_gradient(
            metadata, grad_weight.to(shard.dtype)
        )
        return grad_inputs, placeholder, None, None


class _ShardedEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        token_ids: torch.Tensor,
        shard: torch.Tensor,
        owner: "FullyShardedDataParallel",
        metadata: _ShardMetadata,
    ) -> torch.Tensor:
        full_weight = owner._gather_for_compute(shard, metadata)
        ctx.save_for_backward(token_ids, shard)
        ctx.owner = owner
        ctx.shard_metadata = metadata
        return full_weight[token_ids, :]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        token_ids, shard = ctx.saved_tensors
        owner: FullyShardedDataParallel = ctx.owner
        metadata: _ShardMetadata = ctx.shard_metadata
        grad_weight = torch.zeros(
            metadata.full_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_weight.index_add_(
            0,
            token_ids.reshape(-1),
            grad_output.reshape(-1, metadata.full_shape[1]),
        )
        placeholder = owner._queue_sharded_gradient(
            metadata, grad_weight.to(shard.dtype)
        )
        return None, placeholder, None, None


class FullyShardedDataParallel(nn.Module):
    """Shard custom CS336 Linear/Embedding weights across the process group."""

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before wrapping the model")
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._metadata_by_parameter: dict[int, _ShardMetadata] = {}
        self._shards_by_name: dict[str, _ShardMetadata] = {}
        self._pending_sharded: list[
            tuple[dist.Work, torch.Tensor, _ShardMetadata]
        ] = []
        self._pending_replicated: list[dist.Work] = []
        self._hook_handles: list[Any] = []

        # Ensure every rank shards the same rank-0 parameters.
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        from cs336_basics.model import Embedding, Linear

        target_parameter_ids: set[int] = set()
        for module_name, child in self.module.named_modules():
            if not isinstance(child, (Linear, Embedding)):
                continue

            parameter = child.weight
            parameter_id = id(parameter)
            if parameter_id not in self._metadata_by_parameter:
                full_shape = tuple(parameter.shape)
                full_numel = parameter.numel()
                shard_size = (full_numel + self.world_size - 1) // self.world_size
                padded_numel = shard_size * self.world_size
                padded = torch.zeros(
                    padded_numel, device=parameter.device, dtype=parameter.dtype
                )
                padded[:full_numel].copy_(parameter.detach().reshape(-1))
                start = self.rank * shard_size
                parameter.data = padded[start : start + shard_size].clone()
                name = f"{module_name}.weight" if module_name else "weight"
                metadata = _ShardMetadata(
                    name=name,
                    parameter=parameter,
                    full_shape=full_shape,
                    full_numel=full_numel,
                    shard_size=shard_size,
                )
                self._metadata_by_parameter[parameter_id] = metadata
                self._shards_by_name[name] = metadata
            else:
                metadata = self._metadata_by_parameter[parameter_id]

            target_parameter_ids.add(parameter_id)
            if isinstance(child, Linear):

                def linear_forward(
                    child_module: nn.Module,
                    inputs: torch.Tensor,
                    *,
                    _owner=self,
                    _metadata=metadata,
                ) -> torch.Tensor:
                    return _ShardedLinearFunction.apply(
                        inputs, child_module.weight, _owner, _metadata
                    )

                child.forward = MethodType(linear_forward, child)
            else:

                def embedding_forward(
                    child_module: nn.Module,
                    token_ids: torch.Tensor,
                    *,
                    _owner=self,
                    _metadata=metadata,
                ) -> torch.Tensor:
                    return _ShardedEmbeddingFunction.apply(
                        token_ids, child_module.weight, _owner, _metadata
                    )

                child.forward = MethodType(embedding_forward, child)

        # Non-target parameters (for example RMSNorm scales) stay replicated.
        for parameter in self.module.parameters():
            if id(parameter) in target_parameter_ids or not parameter.requires_grad:
                continue
            handle = parameter.register_post_accumulate_grad_hook(
                self._make_replicated_gradient_hook()
            )
            self._hook_handles.append(handle)

    def _make_replicated_gradient_hook(self):
        def synchronize(parameter: nn.Parameter) -> None:
            if parameter.grad is None:
                return
            parameter.grad.div_(self.world_size)
            work = dist.all_reduce(parameter.grad, async_op=True)
            self._pending_replicated.append(work)

        return synchronize

    def _all_gather_flat(
        self, shard: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        communication_shard = shard.detach().to(dtype).contiguous()
        pieces = [torch.empty_like(communication_shard) for _ in range(self.world_size)]
        dist.all_gather(pieces, communication_shard)
        return torch.cat(pieces)

    def _gather_for_compute(
        self, shard: torch.Tensor, metadata: _ShardMetadata
    ) -> torch.Tensor:
        dtype = self.compute_dtype if self.compute_dtype is not None else shard.dtype
        flat = self._all_gather_flat(shard, dtype)
        return flat[: metadata.full_numel].view(metadata.full_shape)

    def _queue_sharded_gradient(
        self, metadata: _ShardMetadata, full_gradient: torch.Tensor
    ) -> torch.Tensor:
        padded_numel = metadata.shard_size * self.world_size
        padded_gradient = torch.zeros(
            padded_numel,
            device=full_gradient.device,
            dtype=metadata.parameter.dtype,
        )
        padded_gradient[: metadata.full_numel].copy_(full_gradient.reshape(-1))
        padded_gradient.div_(self.world_size)
        work = dist.all_reduce(padded_gradient, async_op=True)
        self._pending_sharded.append((work, padded_gradient, metadata))

        # Autograd needs a correctly-shaped return now; finish() replaces this
        # zero placeholder with the reduced local shard before optimizer.step().
        return torch.zeros_like(metadata.parameter)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for work in self._pending_replicated:
            work.wait()
        self._pending_replicated.clear()

        for work, padded_gradient, metadata in self._pending_sharded:
            work.wait()
            start = self.rank * metadata.shard_size
            local_gradient = padded_gradient[
                start : start + metadata.shard_size
            ]
            if metadata.parameter.grad is None:
                metadata.parameter.grad = local_gradient.clone()
            else:
                metadata.parameter.grad.add_(local_gradient)
        self._pending_sharded.clear()

    @torch.no_grad()
    def gather_full_parameters(self) -> dict[str, torch.Tensor]:
        """Reconstruct a conventional state dictionary on every rank."""

        result: dict[str, torch.Tensor] = {}
        for name, parameter in self.module.named_parameters():
            metadata = self._metadata_by_parameter.get(id(parameter))
            if metadata is None:
                result[name] = parameter.detach().clone()
            else:
                flat = self._all_gather_flat(parameter, parameter.dtype)
                result[name] = (
                    flat[: metadata.full_numel]
                    .view(metadata.full_shape)
                    .detach()
                    .clone()
                )
        return result


def fsdp_on_after_backward(
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    del optimizer
    model.finish_gradient_synchronization()


def fsdp_gather_full_params(
    model: FullyShardedDataParallel,
) -> dict[str, torch.Tensor]:
    return model.gather_full_parameters()
