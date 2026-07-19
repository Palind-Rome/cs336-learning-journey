"""End-to-end compute and memory benchmark for the CS336 Transformer."""

from __future__ import annotations

import argparse
import json
import statistics
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass
class Timings:
    forward: list[float]
    backward: list[float]
    optimizer: list[float]

    def summary(self) -> dict[str, dict[str, float]]:
        answer: dict[str, dict[str, float]] = {}
        for name, values in vars(self).items():
            if values:
                answer[name] = {
                    "mean_ms": statistics.fmean(values) * 1000,
                    "std_ms": statistics.stdev(values) * 1000 if len(values) > 1 else 0.0,
                }
        return answer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    config = MODEL_SIZES[args.size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    labels = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    def precision_context():
        return (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if args.mixed_precision
            else nullcontext()
        )

    def step(measure: bool) -> tuple[float, float, float]:
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        start = default_timer()
        with precision_context():
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, args.vocab_size), labels.reshape(-1)
            )
        synchronize(device)
        after_forward = default_timer()

        if args.mode == "forward":
            return after_forward - start, 0.0, 0.0

        loss.backward()
        synchronize(device)
        after_backward = default_timer()
        if args.mode == "forward-backward":
            return after_forward - start, after_backward - after_forward, 0.0

        optimizer.step()
        synchronize(device)
        after_optimizer = default_timer()
        return (
            after_forward - start,
            after_backward - after_forward,
            after_optimizer - after_backward,
        )

    for _ in range(args.warmup):
        step(measure=False)

    if args.memory_snapshot:
        if device.type != "cuda":
            raise ValueError("memory snapshots require a CUDA device")
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    timings = Timings([], [], [])
    for _ in range(args.steps):
        forward, backward, optimizer_time = step(measure=True)
        timings.forward.append(forward)
        if args.mode != "forward":
            timings.backward.append(backward)
        if args.mode == "train":
            timings.optimizer.append(optimizer_time)

    if args.memory_snapshot:
        torch.cuda.memory._dump_snapshot(args.memory_snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)

    return {
        "model": args.size,
        "mode": args.mode,
        "mixed_precision": args.mixed_precision,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "timings": timings.summary(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODEL_SIZES, default="small")
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train"], default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--memory-snapshot", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_benchmark(parse_args()), indent=2))
