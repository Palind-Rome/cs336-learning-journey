"""A compact but complete training loop used by the Assignment 1 chapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoint import save_checkpoint
from .data import get_batch
from .model import TransformerLM
from .optimizer import AdamW, cross_entropy, gradient_clipping, learning_rate_schedule


@dataclass(frozen=True)
class TrainConfig:
    train_tokens: Path
    validation_tokens: Path
    checkpoint_dir: Path
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1_344
    rope_theta: float = 10_000.0
    batch_size: int = 32
    steps: int = 5_000
    warmup_steps: int = 500
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_every: int = 100
    eval_batches: int = 20
    save_every: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    data: np.ndarray,
    config: TrainConfig,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(config.eval_batches):
        inputs, targets = get_batch(data, config.batch_size, config.context_length, config.device)
        logits = model(inputs)
        losses.append(float(cross_entropy(logits, targets).item()))
    model.train()
    return sum(losses) / len(losses)


def train(config: TrainConfig) -> TransformerLM:
    """Train, evaluate, log, and checkpoint a TinyStories-style language model."""

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_data = np.load(config.train_tokens, mmap_mode="r")
    validation_data = np.load(config.validation_tokens, mmap_mode="r")
    model = TransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.rope_theta,
        device=config.device,
    )
    optimizer = AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay, betas=(0.9, 0.95))

    model.train()
    for step in range(config.steps):
        learning_rate = learning_rate_schedule(
            step,
            config.max_lr,
            config.min_lr,
            config.warmup_steps,
            config.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(train_data, config.batch_size, config.context_length, config.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        gradient_clipping(model.parameters(), config.max_grad_norm)
        optimizer.step()

        if step % config.eval_every == 0:
            validation_loss = evaluate(model, validation_data, config)
            print(
                f"step={step:05d} train_loss={loss.item():.4f} "
                f"validation_loss={validation_loss:.4f} lr={learning_rate:.3e}"
            )
        if step > 0 and step % config.save_every == 0:
            save_checkpoint(model, optimizer, step, config.checkpoint_dir / f"step-{step:05d}.pt")

    save_checkpoint(model, optimizer, config.steps, config.checkpoint_dir / "final.pt")
    return model
