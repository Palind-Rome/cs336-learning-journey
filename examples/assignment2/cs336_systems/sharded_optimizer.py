"""Optimizer-state sharding with deterministic parameter ownership."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Store optimizer state for only one rank's subset of parameters.

    Parameters stay replicated. Each rank updates only the parameters it owns,
    then broadcasts those updated values. The wrapped optimizer therefore holds
    roughly ``1 / world_size`` of Adam's first- and second-moment tensors.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ):
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before the optimizer")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)
        self._owners: list[tuple[torch.Tensor, int]] = []
        self._rank_loads = [0 for _ in range(self.world_size)]
        self._local_groups: list[dict[str, Any]] = []
        self.local_optimizer: Optimizer | None = None

        # Optimizer.__init__ normalizes generators and invokes our add_param_group.
        super().__init__(params, defaults=kwargs)
        non_empty_groups = [g for g in self._local_groups if g["params"]]
        if not non_empty_groups:
            raise ValueError("this rank was assigned no parameters")
        self.local_optimizer = optimizer_cls(non_empty_groups, **kwargs)

    def _choose_owner(self, parameter: torch.Tensor) -> int:
        # Greedy load balancing is deterministic because every rank traverses the
        # same ordered parameter list.
        owner = min(range(self.world_size), key=self._rank_loads.__getitem__)
        self._rank_loads[owner] += parameter.numel()
        return owner

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not isinstance(param_group, dict):
            raise TypeError("a parameter group must be a dict")
        group = dict(param_group)
        group["params"] = list(group["params"])
        super().add_param_group(group)

        local_group = {key: value for key, value in group.items() if key != "params"}
        local_group["params"] = []
        for parameter in group["params"]:
            owner = self._choose_owner(parameter)
            self._owners.append((parameter, owner))
            if owner == self.rank:
                local_group["params"].append(parameter)
        self._local_groups.append(local_group)

        # add_param_group may also be called after construction.
        if self.local_optimizer is not None and local_group["params"]:
            self.local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        if self.local_optimizer is None:
            raise RuntimeError("local optimizer has not been initialized")
        loss = self.local_optimizer.step(closure=closure, **kwargs)
        for parameter, owner in self._owners:
            dist.broadcast(parameter.data, src=owner)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Clear all replicated gradients, not only locally-owned ones.
        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        if self.local_optimizer is None:
            raise RuntimeError("local optimizer has not been initialized")
        return {
            "local_optimizer": self.local_optimizer.state_dict(),
            "rank_loads": self._rank_loads,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.local_optimizer is None:
            raise RuntimeError("local optimizer has not been initialized")
        self.local_optimizer.load_state_dict(state_dict["local_optimizer"])
