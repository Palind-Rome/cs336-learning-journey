"""GRPO primitives and a complete single-update training step."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F


Baseline = Literal["mean", "none"]
AdvantageNormalizer = Literal["std", "none", "mean"]
ImportanceMethod = Literal["none", "noclip", "grpo", "gspo"]
LossNormalization = Literal["sequence", "constant"]


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize separately, concatenate without separators, then shift once."""

    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have equal length")
    if not prompt_strs:
        raise ValueError("the batch must contain at least one prompt")

    examples: list[tuple[list[int], int]] = []
    for prompt, output in zip(prompt_strs, output_strs, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        combined = prompt_ids + output_ids
        if len(combined) < 2:
            raise ValueError("each prompt/output pair must contain at least two tokens")
        examples.append((combined, len(prompt_ids)))

    maximum_length = max(len(token_ids) for token_ids, _ in examples)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id must be configured")
    padded = torch.full((len(examples), maximum_length), pad_id, dtype=torch.long)
    response_mask = torch.zeros(
        (len(examples), maximum_length - 1), dtype=torch.bool
    )
    for row, (token_ids, prompt_length) in enumerate(examples):
        padded[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        # labels[k] is combined[k + 1]. Response begins at combined[prompt_length].
        start = max(prompt_length - 1, 0)
        stop = len(token_ids) - 1
        response_mask[row, start:stop] = True

    return {
        "input_ids": padded[:, :-1],
        "labels": padded[:, 1:],
        "response_mask": response_mask,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_distributions = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(
        log_distributions, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    result = {"log_probs": log_probs}
    if return_token_entropy:
        probabilities = log_distributions.exp()
        result["token_entropy"] = -(probabilities * log_distributions).sum(dim=-1)
    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollouts and ground truths must have equal length")
    score_dicts = [
        reward_fn(response, truth)
        for response, truth in zip(
            rollout_responses, repeated_ground_truths, strict=True
        )
    ]
    raw_rewards = torch.tensor(
        [scores["reward"] for scores in score_dicts], dtype=torch.float32
    )
    metadata = {
        "reward": float(raw_rewards.mean()) if len(raw_rewards) else 0.0,
        "format_reward": (
            sum(scores.get("format_reward", 0.0) for scores in score_dicts)
            / len(score_dicts)
            if score_dicts
            else 0.0
        ),
        "answer_reward": (
            sum(scores.get("answer_reward", 0.0) for scores in score_dicts)
            / len(score_dicts)
            if score_dicts
            else 0.0
        ),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Baseline = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: AdvantageNormalizer = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    if group_size <= 0 or raw_rewards.numel() % group_size:
        raise ValueError("group_size must divide the rollout batch size")
    groups = raw_rewards.reshape(-1, group_size)
    group_means = groups.mean(dim=1, keepdim=True)

    if baseline == "mean":
        advantages = groups - group_means
    elif baseline == "none":
        advantages = groups.clone()
    else:
        raise NotImplementedError(f"unsupported baseline: {baseline}")

    if advantage_normalizer == "std":
        normalizer = groups.std(dim=1, keepdim=True) + advantage_eps
        advantages = advantages / normalizer
    elif advantage_normalizer == "mean":
        advantages = advantages / (group_means + advantage_eps)
    elif advantage_normalizer != "none":
        raise NotImplementedError(
            f"unsupported advantage normalizer: {advantage_normalizer}"
        )

    metadata = {
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()) if raw_rewards.numel() > 1 else 0.0,
        "reward_min": float(raw_rewards.min()),
        "reward_max": float(raw_rewards.max()),
        "advantage_mean": float(advantages.mean()),
    }
    return advantages.reshape_as(raw_rewards), metadata


def _as_column(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = values.to(device=target.device, dtype=target.dtype)
    if values.ndim == 1:
        values = values.unsqueeze(1)
    if values.shape != (target.shape[0], 1):
        raise ValueError("advantages must have shape (batch,) or (batch, 1)")
    return values


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: ImportanceMethod = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = _as_column(raw_rewards_or_advantages, policy_log_probs)
    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        return -advantages * policy_log_probs, metadata
    if old_log_probs is None:
        raise ValueError("old_log_probs are required for off-policy losses")
    old_log_probs = old_log_probs.to(policy_log_probs.device)
    log_ratio = policy_log_probs - old_log_probs

    if importance_reweighting_method == "noclip":
        ratio = log_ratio.exp()
        return -(advantages * ratio), metadata

    if cliprange is None or cliprange < 0:
        raise ValueError("a non-negative cliprange is required")

    if importance_reweighting_method == "grpo":
        ratio = log_ratio.exp()
        unclipped = advantages * ratio
        clipped = advantages * ratio.clamp(1 - cliprange, 1 + cliprange)
        surrogate = torch.minimum(unclipped, clipped)
        metadata["clip_fraction"] = (unclipped != surrogate).float().mean()
        return -surrogate, metadata

    if importance_reweighting_method == "gspo":
        if response_mask is None:
            raise ValueError("response_mask is required for GSPO")
        mask = response_mask.to(device=log_ratio.device, dtype=log_ratio.dtype)
        counts = mask.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("every GSPO sequence must contain a response token")
        sequence_ratio = ((log_ratio * mask).sum(dim=1, keepdim=True) / counts).exp()
        unclipped = advantages * sequence_ratio
        clipped = advantages * sequence_ratio.clamp(
            1 - cliprange, 1 + cliprange
        )
        surrogate = torch.minimum(unclipped, clipped)
        metadata["clip_fraction"] = (unclipped != surrogate).float().mean()
        return -surrogate.expand_as(policy_log_probs), metadata

    raise NotImplementedError(
        f"unsupported importance method: {importance_reweighting_method}"
    )


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: LossNormalization = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    mask_float = mask.to(
        device=per_token_policy_gradient_loss.device,
        dtype=per_token_policy_gradient_loss.dtype,
    )
    masked_loss = per_token_policy_gradient_loss * mask_float
    if loss_normalization == "sequence":
        counts = mask_float.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("every sequence must contain a response token")
        return (masked_loss.sum(dim=1) / counts).mean()
    if loss_normalization == "constant":
        if normalization_constant is None or normalization_constant <= 0:
            raise ValueError("normalization_constant must be positive")
        return masked_loss.sum() / normalization_constant
    raise NotImplementedError(f"unsupported loss normalization: {loss_normalization}")


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Baseline = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: AdvantageNormalizer = "std",
    importance_reweighting_method: ImportanceMethod = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: LossNormalization = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    batch_size = len(rollout_responses)
    if not (
        len(repeated_prompts)
        == len(repeated_ground_truths)
        == batch_size
    ):
        raise ValueError("prompts, responses, and truths must have equal length")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )
    tokenized = tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer
    )

    # Removing exact-zero rows is mathematically exact and especially valuable for RFT.
    keep = advantages != 0
    kept_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
    device = _model_device(model)
    optimizer.zero_grad(set_to_none=True)
    if kept_indices.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, {**reward_metadata, **advantage_metadata, "gradient_norm": 0.0}

    input_ids = tokenized["input_ids"][kept_indices]
    labels = tokenized["labels"][kept_indices]
    response_mask = tokenized["response_mask"][kept_indices]
    kept_advantages = advantages[kept_indices]
    kept_old_log_probs = (
        old_log_probs[
            kept_indices, : tokenized["input_ids"].shape[1]
        ]
        if old_log_probs is not None
        else None
    )

    # Preserve the original microbatch size after zero-advantage pruning.
    original_microbatch_size = max(
        1, (batch_size + gradient_accumulation_steps - 1)
        // gradient_accumulation_steps,
    )
    batch_loss = torch.tensor(0.0, device=device)
    entropy_total = torch.tensor(0.0, device=device)
    entropy_tokens = 0
    clip_fractions: list[torch.Tensor] = []
    model.train()

    for start in range(0, len(kept_indices), original_microbatch_size):
        stop = min(start + original_microbatch_size, len(kept_indices))
        micro_input = input_ids[start:stop].to(device)
        micro_labels = labels[start:stop].to(device)
        micro_mask = response_mask[start:stop].to(device)
        scores = get_response_log_probs(
            model, micro_input, micro_labels, return_token_entropy=True
        )
        micro_old = (
            kept_old_log_probs[start:stop].to(device)
            if kept_old_log_probs is not None
            else None
        )
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            kept_advantages[start:stop],
            scores["log_probs"],
            importance_reweighting_method,
            micro_old,
            cliprange,
            micro_mask,
        )
        micro_loss = aggregate_loss_across_microbatch(
            per_token_loss,
            micro_mask,
            loss_normalization,
            normalization_constant,
        )
        if loss_normalization == "sequence":
            micro_weight = (stop - start) / batch_size
            backward_loss = micro_loss * micro_weight
        else:
            backward_loss = micro_loss
        backward_loss.backward()
        batch_loss = batch_loss + backward_loss.detach()

        entropy = scores["token_entropy"]
        entropy_total = entropy_total + (entropy * micro_mask).sum().detach()
        entropy_tokens += int(micro_mask.sum())
        if "clip_fraction" in loss_metadata:
            clip_fractions.append(loss_metadata["clip_fraction"].detach())

    if max_grad_norm is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )
    else:
        squared = sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        gradient_norm = squared.sqrt()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata: dict[str, torch.Tensor | float] = {
        **reward_metadata,
        **advantage_metadata,
        "gradient_norm": gradient_norm.detach(),
        "token_entropy": float(entropy_total / max(entropy_tokens, 1)),
        "active_sequences": int(kept_indices.numel()),
    }
    if clip_fractions:
        metadata["clip_fraction"] = torch.stack(clip_fractions).mean()
    return batch_loss, metadata
