"""Single-node all-reduce benchmark with cross-rank aggregation."""

from __future__ import annotations

import os
import statistics
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, size_mb: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elements = size_mb * 1024**2 // torch.tensor([], dtype=torch.float32).element_size()
    payload = torch.ones(elements, dtype=torch.float32, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(5):
        dist.all_reduce(payload)
        synchronize()

    measurements: list[float] = []
    for _ in range(10):
        dist.barrier()
        synchronize()
        start = default_timer()
        dist.all_reduce(payload)
        synchronize()
        measurements.append(default_timer() - start)

    gathered: list[list[float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, measurements)
    if rank == 0:
        flattened = [value for rank_values in gathered for value in rank_values or []]
        print(
            {
                "world_size": world_size,
                "size_mb": size_mb,
                "mean_ms": statistics.fmean(flattened) * 1000,
                "std_ms": statistics.stdev(flattened) * 1000,
            }
        )
    dist.destroy_process_group()


def run(world_size: int, size_mb: int) -> None:
    mp.spawn(_worker, args=(world_size, size_mb), nprocs=world_size, join=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--size-mb", type=int, choices=[1, 10, 100, 1024], default=100)
    arguments = parser.parse_args()
    run(arguments.world_size, arguments.size_mb)
