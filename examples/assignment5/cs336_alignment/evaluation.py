"""Prompt rendering, generation evaluation, and serialization helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GenerationRecord:
    prompt: str
    response: str
    ground_truth: str | None
    reward: float | None
    format_reward: float | None
    answer_reward: float | None


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def extract_gsm8k_answer(answer_with_rationale: str) -> str:
    return answer_with_rationale.rsplit("####", 1)[-1].strip()


def render_prompts(
    examples: Iterable[dict], template: str, field: str = "question"
) -> list[str]:
    prompts = []
    for example in examples:
        values = dict(example)
        values["question"] = example[field]
        prompts.append(template.format(**values))
    return prompts


def evaluate_generations(
    prompts: list[str],
    ground_truths: list[str],
    generate: Callable[[list[str], dict], list],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: dict,
) -> tuple[list[GenerationRecord], dict[str, float]]:
    started = time.perf_counter()
    completions = generate(prompts, sampling_params)
    elapsed = time.perf_counter() - started
    records: list[GenerationRecord] = []
    for prompt, truth, completion in zip(
        prompts, ground_truths, completions, strict=True
    ):
        response = completion.text if hasattr(completion, "text") else str(completion)
        scores = reward_fn(response, truth)
        records.append(
            GenerationRecord(
                prompt=prompt,
                response=response,
                ground_truth=truth,
                reward=float(scores["reward"]),
                format_reward=float(scores.get("format_reward", 0.0)),
                answer_reward=float(scores.get("answer_reward", scores["reward"])),
            )
        )
    metrics = {
        "examples": float(len(records)),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(records) / elapsed if elapsed else 0.0,
        "reward": sum(record.reward or 0.0 for record in records) / len(records),
        "format_reward": sum(record.format_reward or 0.0 for record in records)
        / len(records),
        "average_response_characters": sum(len(record.response) for record in records)
        / len(records),
    }
    return records, metrics


def write_generation_records(
    records: list[GenerationRecord], output_path: str | Path
) -> None:
    with Path(output_path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def pass_at_k(binary_rewards: list[float], group_size: int, k: int) -> float:
    if not 1 <= k <= group_size or len(binary_rewards) % group_size:
        raise ValueError("invalid group_size or k")
    groups = [
        binary_rewards[index : index + group_size]
        for index in range(0, len(binary_rewards), group_size)
    ]
    return sum(any(value > 0 for value in group[:k]) for group in groups) / len(groups)
