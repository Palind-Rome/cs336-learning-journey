"""Direct Preference Optimization primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import ALPACA_TEMPLATE


def sequence_log_probability(model, tokenizer, prompt: str, response: str):
    text = ALPACA_TEMPLATE.format(instruction=prompt, response=response)
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    token_ids.append(tokenizer.eos_token_id)
    device = next(model.parameters()).device
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    input_ids = tokens[:-1].unsqueeze(0)
    labels = tokens[1:].unsqueeze(0)
    logits = model(input_ids).logits
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    )
    return token_log_probs.sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    policy_chosen = sequence_log_probability(
        lm, tokenizer, prompt, response_chosen
    )
    policy_rejected = sequence_log_probability(
        lm, tokenizer, prompt, response_rejected
    )
    with torch.no_grad():
        reference_chosen = sequence_log_probability(
            lm_ref, tokenizer, prompt, response_chosen
        )
        reference_rejected = sequence_log_probability(
            lm_ref, tokenizer, prompt, response_rejected
        )
    reference_log_ratio = (reference_chosen - reference_rejected).to(
        policy_chosen.device
    )
    policy_log_ratio = policy_chosen - policy_rejected
    preference_logit = beta * (policy_log_ratio - reference_log_ratio)
    return -F.logsigmoid(preference_logit)
