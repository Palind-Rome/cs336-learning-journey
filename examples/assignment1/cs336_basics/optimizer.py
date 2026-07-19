"""Loss, optimizer, schedule, and gradient utilities for Assignment 1."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Average negative log likelihood without materializing log-softmax."""

    shifted = logits - logits.amax(dim=-1, keepdim=True)
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1))
    return (log_normalizer - target_logits).mean()


class AdamW(torch.optim.Optimizer):
    """Adam with bias correction and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"invalid beta values: {betas}")
        if eps < 0:
            raise ValueError(f"invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                parameter.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                adjusted_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                denominator = exp_avg_sq.sqrt().add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-adjusted_lr)
        return loss


def learning_rate_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup, cosine decay, then a constant learning-rate floor."""

    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    if iteration <= cosine_cycle_iters:
        progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
    return min_learning_rate


@torch.no_grad()
def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """Scale all existing gradients by one common factor when the global norm is too large."""

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient.detach(), ord=2) for gradient in gradients]),
        ord=2,
    )
    scale = torch.clamp(max_l2_norm / (total_norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))
