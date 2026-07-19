"""Strict, auditable parsers for MMLU and GSM8K generations."""

from __future__ import annotations

import re


MMLU_PATTERN = re.compile(
    r"\bthe\s+correct\s+answer\s+is\s*([A-D])\b", re.IGNORECASE
)
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)


def parse_mmlu_response(mmlu_example: dict, model_output: str) -> str | None:
    del mmlu_example
    match = MMLU_PATTERN.search(model_output)
    return match.group(1).upper() if match else None


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = NUMBER_PATTERN.findall(model_output)
    return matches[-1].replace(",", "") if matches else None
