"""Temperature and nucleus sampling for an autoregressive Transformer."""

from __future__ import annotations

import torch
from torch import Tensor

from .model import TransformerLM


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    eos_id: int | None,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    """Append sampled IDs until EOS or the requested generation budget is exhausted."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    device = next(model.parameters()).device
    generated = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = generated[-model.context_length :]
        token_ids = torch.tensor([context], device=device, dtype=torch.long)
        logits = model(token_ids)[0, -1] / temperature
        probabilities = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        next_id = int(sorted_ids[sampled_index].item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return generated
