"""Packed SFT data and Anthropic HH preference loading."""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


class PackedSFTDataset(Dataset):
    """Concatenate Alpaca-formatted documents and expose shifted fixed chunks."""

    def __init__(self, tokenizer, dataset_path, seq_length: int, shuffle: bool):
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        with _open_text(dataset_path) as file:
            documents = [json.loads(line) for line in file if line.strip()]
        if shuffle:
            random.shuffle(documents)

        token_ids: list[int] = []
        for example in documents:
            text = ALPACA_TEMPLATE.format(
                instruction=example["prompt"], response=example["response"]
            )
            # Each document gets the tokenizer's normal prefix plus an explicit EOS.
            token_ids.extend(tokenizer.encode(text, add_special_tokens=True))
            token_ids.append(tokenizer.eos_token_id)

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_length = seq_length
        self.num_examples = max(0, (len(token_ids) - 1) // seq_length)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.seq_length
        return {
            "input_ids": self.token_ids[start : start + self.seq_length],
            "labels": self.token_ids[start + 1 : start + self.seq_length + 1],
        }


def iterate_batches(
    dataset: Dataset, batch_size: int, shuffle: bool
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@dataclass(frozen=True)
class HHPreference:
    instruction: str
    chosen: str
    rejected: str
    source: str


def _split_single_turn(conversation: str) -> tuple[str, str] | None:
    marker = "\n\nHuman:"
    if conversation.count(marker) != 1 or "\n\nAssistant:" not in conversation:
        return None
    human_and_assistant = conversation.split(marker, 1)[1]
    instruction, response = human_and_assistant.split("\n\nAssistant:", 1)
    return instruction.strip(), response.strip()


def load_hh_preferences(paths: list[str | Path]) -> list[HHPreference]:
    """Load four HH splits and discard divergent multi-turn conversations."""

    preferences: list[HHPreference] = []
    for path_like in paths:
        path = Path(path_like)
        with _open_text(path) as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                chosen = _split_single_turn(record["chosen"])
                rejected = _split_single_turn(record["rejected"])
                if chosen is None or rejected is None:
                    continue
                chosen_prompt, chosen_response = chosen
                rejected_prompt, rejected_response = rejected
                if chosen_prompt != rejected_prompt:
                    continue
                preferences.append(
                    HHPreference(
                        instruction=chosen_prompt,
                        chosen=chosen_response,
                        rejected=rejected_response,
                        source=path.stem.replace(".jsonl", ""),
                    )
                )
    return preferences
