"""Map a continuous scaling-law target N to a valid Transformer configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    head_dim: int
    num_attention_heads: int
    estimated_non_embedding_parameters: int

    def api_dict(self) -> dict[str, object]:
        return {
            "attention_bias": False,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_attention_heads,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "vocab_size": 32_000,
        }


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def nearest_architecture(target_parameters: float) -> Architecture:
    """Search valid head_dim=64 shapes using N ~= 12 * layers * hidden^2."""

    candidates: list[tuple[float, Architecture]] = []
    for hidden_size in range(256, 8193, 64):
        for layers in range(4, 129):
            estimate = 12 * layers * hidden_size**2
            error = abs(math.log(estimate / target_parameters))
            intermediate = round_to_multiple(8 * hidden_size / 3, 128)
            architecture = Architecture(
                hidden_size=hidden_size,
                intermediate_size=intermediate,
                num_hidden_layers=layers,
                head_dim=64,
                num_attention_heads=hidden_size // 64,
                estimated_non_embedding_parameters=estimate,
            )
            candidates.append((error, architecture))
    return min(candidates, key=lambda item: item[0])[1]


def divisible_token_count(tokens: int, train_batch_size: int, sequence_length: int = 512) -> int:
    quantum = train_batch_size * sequence_length
    return max(quantum, tokens // quantum * quantum)
