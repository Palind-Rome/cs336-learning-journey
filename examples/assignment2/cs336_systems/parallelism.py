"""Closed-form cost models for the Assignment 2 parallelism questions."""

from __future__ import annotations


def ring_all_gather_seconds(size_bytes: float, devices: int, bandwidth: float) -> float:
    return (devices - 1) / devices * size_bytes / bandwidth


def ring_reduce_scatter_seconds(size_bytes: float, devices: int, bandwidth: float) -> float:
    return ring_all_gather_seconds(size_bytes, devices, bandwidth)


def ring_all_reduce_seconds(size_bytes: float, devices: int, bandwidth: float) -> float:
    return 2 * ring_all_gather_seconds(size_bytes, devices, bandwidth)


def alternate_ring_all_reduce_seconds(
    size_bytes: float, devices: int, bandwidth: float
) -> float:
    # The alternate algorithm sends a full S-byte tensor on every one of N-1 rounds.
    return (devices - 1) * size_bytes / bandwidth


def ffn_matmul_flops(batch_tokens: int, d_model: int, d_ff: int) -> dict[str, int]:
    """Forward/backward FLOPs for the three-matmul SwiGLU FFN."""

    forward = 6 * batch_tokens * d_model * d_ff
    backward = 12 * batch_tokens * d_model * d_ff
    return {"forward": forward, "backward": backward}


def dp_backward_costs(
    batch_tokens: int,
    d_model: int,
    d_ff: int,
    devices: int,
    accelerator_flops: float,
    bandwidth: float,
) -> dict[str, float]:
    compute = (
        12 * batch_tokens * d_model * d_ff / devices / accelerator_flops
    )
    # Three FP16 matrices contain 3 * D * D_ff weights.
    gradient_bytes = 6 * d_model * d_ff
    communication = ring_all_reduce_seconds(
        gradient_bytes, devices, bandwidth
    )
    return {"compute_seconds": compute, "communication_seconds": communication}


def fsdp_costs(
    batch_tokens: int,
    d_model: int,
    d_ff: int,
    devices: int,
    accelerator_flops: float,
    bandwidth: float,
) -> dict[str, float]:
    forward_compute = 6 * batch_tokens * d_model * d_ff / devices / accelerator_flops
    backward_compute = 12 * batch_tokens * d_model * d_ff / devices / accelerator_flops
    weight_bytes = 6 * d_model * d_ff
    one_collective = ring_all_gather_seconds(weight_bytes, devices, bandwidth)
    return {
        "forward_compute_seconds": forward_compute,
        "backward_compute_seconds": backward_compute,
        "forward_communication_seconds": one_collective,
        "backward_communication_seconds": 2 * one_collective,
    }


def tp_costs(
    batch_tokens: int,
    d_model: int,
    d_ff: int,
    devices: int,
    accelerator_flops: float,
    bandwidth: float,
) -> dict[str, float]:
    forward_compute = 6 * batch_tokens * d_model * d_ff / devices / accelerator_flops
    backward_compute = 12 * batch_tokens * d_model * d_ff / devices / accelerator_flops
    activation_bytes = 2 * batch_tokens * d_model
    one_all_reduce = ring_all_reduce_seconds(activation_bytes, devices, bandwidth)
    return {
        "forward_compute_seconds": forward_compute,
        "backward_compute_seconds": backward_compute,
        "forward_communication_seconds": one_all_reduce,
        "backward_communication_seconds": one_all_reduce,
    }
