"""End-to-end SFT, DPO, and rollout/update loops without Trainer."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import HHPreference
from .dpo import compute_per_instance_dpo_loss, sequence_log_probability
from .evaluation import extract_gsm8k_answer
from .grpo import get_response_log_probs, grpo_train_step, tokenize_prompt_and_output


def causal_lm_loss(model, batch: dict[str, torch.Tensor], device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids).logits
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


@torch.no_grad()
def validation_lm_loss(model, batches: Iterable[dict], device) -> float:
    was_training = model.training
    model.eval()
    losses = [float(causal_lm_loss(model, batch, device)) for batch in batches]
    model.train(was_training)
    return sum(losses) / len(losses)


def train_sft_epoch(
    model,
    optimizer,
    batches: Iterable[dict[str, torch.Tensor]],
    device,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None = 1.0,
    scheduler=None,
    log_fn: Callable[[dict], None] | None = None,
) -> None:
    """One packed-data epoch with a correct final partial accumulation group."""

    iterator = iter(batches)
    optimizer.zero_grad(set_to_none=True)
    update = 0
    while True:
        group = list(itertools.islice(iterator, gradient_accumulation_steps))
        if not group:
            break
        total_loss = 0.0
        for batch in group:
            loss = causal_lm_loss(model, batch, device)
            (loss / len(group)).backward()
            total_loss += float(loss.detach())
        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if max_grad_norm is not None
            else torch.tensor(float("nan"))
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if log_fn is not None:
            log_fn(
                {
                    "update": update,
                    "train_loss": total_loss / len(group),
                    "gradient_norm": float(gradient_norm),
                }
            )
        update += 1


@torch.no_grad()
def dpo_validation_accuracy(model, tokenizer, examples: list[HHPreference]) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    for example in examples:
        chosen = sequence_log_probability(
            model, tokenizer, example.instruction, example.chosen
        )
        rejected = sequence_log_probability(
            model, tokenizer, example.instruction, example.rejected
        )
        correct += bool(chosen > rejected)
    model.train(was_training)
    return correct / len(examples)


def train_dpo_epoch(
    model,
    reference_model,
    tokenizer,
    optimizer,
    training_examples: list[HHPreference],
    validation_examples: list[HHPreference],
    beta: float = 0.1,
    gradient_accumulation_steps: int = 64,
    validation_interval: int = 100,
    output_directory: str | Path | None = None,
    log_fn: Callable[[dict], None] | None = None,
) -> float:
    """Train one epoch and save the checkpoint with best validation accuracy."""

    optimizer.zero_grad(set_to_none=True)
    best_accuracy = -1.0
    for start in range(0, len(training_examples), gradient_accumulation_steps):
        group = training_examples[start : start + gradient_accumulation_steps]
        total_loss = 0.0
        for example in group:
            loss = compute_per_instance_dpo_loss(
                model,
                reference_model,
                tokenizer,
                beta,
                example.instruction,
                example.chosen,
                example.rejected,
            )
            (loss / len(group)).backward()
            total_loss += float(loss.detach())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update = start // gradient_accumulation_steps + 1
        metrics = {"update": update, "dpo_loss": total_loss / len(group)}
        if update % validation_interval == 0 or start + len(group) == len(training_examples):
            accuracy = dpo_validation_accuracy(
                model, tokenizer, validation_examples
            )
            metrics["validation_accuracy"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if output_directory is not None:
                    model.save_pretrained(output_directory)
                    tokenizer.save_pretrained(output_directory)
        if log_fn is not None:
            log_fn(metrics)
    return best_accuracy


@torch.no_grad()
def score_old_log_probs(model, tokenizer, prompts, responses) -> torch.Tensor:
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    device = next(model.parameters()).device
    scores = get_response_log_probs(
        model,
        tokenized["input_ids"].to(device),
        tokenized["labels"].to(device),
    )
    return scores["log_probs"].cpu()


def _groups(items: list, group_size: int) -> Iterator[list]:
    if len(items) % group_size:
        raise ValueError("rollout batch must contain complete groups")
    for start in range(0, len(items), group_size):
        yield items[start : start + group_size]


def run_grpo_updates(
    model,
    tokenizer,
    optimizer,
    vllm_server,
    prompt_examples: list[dict],
    prompt_template: str,
    reward_fn,
    group_size: int = 8,
    rollout_batch_size: int = 256,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 32,
    steps: int = 200,
    sampling_params: dict | None = None,
    train_step_kwargs: dict | None = None,
    log_fn: Callable[[dict], None] | None = None,
) -> None:
    """Generic on/off-policy loop; train_batch_size must contain full groups."""

    if rollout_batch_size % group_size or train_batch_size % group_size:
        raise ValueError("rollout/train batch sizes must be divisible by group_size")
    sampling_params = sampling_params or {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    train_step_kwargs = train_step_kwargs or {}
    prompt_count = rollout_batch_size // group_size
    example_cycle = itertools.cycle(prompt_examples)

    for step in range(steps):
        examples = [next(example_cycle) for _ in range(prompt_count)]
        prompts = [prompt_template.format(question=item["question"]) for item in examples]
        truths = [extract_gsm8k_answer(item["answer"]) for item in examples]
        repeated_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        repeated_truths = [truth for truth in truths for _ in range(group_size)]

        vllm_server.sync_policy_weights(model)
        completions = vllm_server.generate_completions(
            repeated_prompts, sampling_params
        )
        responses = [completion.text for completion in completions]
        old_log_probs = (
            score_old_log_probs(model, tokenizer, repeated_prompts, responses)
            if train_step_kwargs.get("importance_reweighting_method", "none")
            != "none"
            else None
        )

        metrics_for_step = []
        groups_per_train_batch = train_batch_size // group_size
        prompt_groups = list(_groups(repeated_prompts, group_size))
        response_groups = list(_groups(responses, group_size))
        truth_groups = list(_groups(repeated_truths, group_size))
        old_groups = (
            list(_groups(list(old_log_probs), group_size))
            if old_log_probs is not None
            else None
        )
        for group_start in range(0, len(prompt_groups), groups_per_train_batch):
            group_stop = group_start + groups_per_train_batch
            batch_prompts = sum(prompt_groups[group_start:group_stop], [])
            batch_responses = sum(response_groups[group_start:group_stop], [])
            batch_truths = sum(truth_groups[group_start:group_stop], [])
            batch_old = (
                torch.stack(sum(old_groups[group_start:group_stop], []))
                if old_groups is not None
                else None
            )
            loss, metadata = grpo_train_step(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=1.0,
                reward_fn=reward_fn,
                repeated_prompts=batch_prompts,
                rollout_responses=batch_responses,
                repeated_ground_truths=batch_truths,
                group_size=group_size,
                old_log_probs=batch_old,
                **train_step_kwargs,
            )
            metrics_for_step.append({"loss": float(loss), **metadata})
        if log_fn is not None:
            log_fn({"rollout_step": step, "updates": metrics_for_step})
