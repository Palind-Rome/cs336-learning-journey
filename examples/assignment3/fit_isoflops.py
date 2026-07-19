"""Reproduce the Chinchilla IsoFLOPs fit on the supplied synthetic runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PowerLaw:
    coefficient: float
    exponent: float

    def __call__(self, compute: np.ndarray | float) -> np.ndarray:
        return self.coefficient * np.asarray(compute, dtype=np.float64) ** self.exponent


def load_runs(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def select_isoflops_minima(runs: list[dict[str, float]]) -> list[dict[str, float]]:
    """Take the lowest-loss observed run at every exact compute budget."""

    grouped: dict[float, list[dict[str, float]]] = {}
    for run in runs:
        grouped.setdefault(float(run["compute_budget"]), []).append(run)
    return [
        min(grouped[compute], key=lambda run: run["final_loss"])
        for compute in sorted(grouped)
    ]


def fit_power_law(x: np.ndarray, y: np.ndarray) -> PowerLaw:
    """Fit log(y) = log(A) + exponent * log(x) by least squares."""

    design = np.column_stack([np.ones_like(x), np.log(x)])
    log_coefficient, exponent = np.linalg.lstsq(
        design, np.log(y), rcond=None
    )[0]
    return PowerLaw(float(np.exp(log_coefficient)), float(exponent))


def fit_isoflops(minima: list[dict[str, float]]) -> tuple[PowerLaw, PowerLaw]:
    compute = np.asarray([run["compute_budget"] for run in minima])
    parameters = np.asarray([run["parameters"] for run in minima])
    tokens = compute / (6 * parameters)
    return (
        fit_power_law(compute, parameters),
        fit_power_law(compute, tokens),
    )


def plot_fit(
    minima: list[dict[str, float]],
    model_law: PowerLaw,
    data_law: PowerLaw,
    output: Path,
) -> None:
    compute = np.asarray([run["compute_budget"] for run in minima])
    parameters = np.asarray([run["parameters"] for run in minima])
    tokens = compute / (6 * parameters)
    curve_compute = np.geomspace(compute.min(), 1e24, 300)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, observed, law, label in [
        (axes[0], parameters, model_law, "parameters"),
        (axes[1], tokens, data_law, "training tokens"),
    ]:
        axis.loglog(compute, observed, "o", label="observed IsoFLOPs minima")
        axis.loglog(curve_compute, law(curve_compute), label=f"fit: exponent={law.exponent:.4f}")
        axis.axvline(1e23, color="grey", linestyle="--", linewidth=1)
        axis.axvline(1e24, color="grey", linestyle=":", linewidth=1)
        axis.set_xlabel("training compute C (FLOPs)")
        axis.set_ylabel(label)
        axis.grid(which="both", alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--plot", type=Path, default=Path("isoflops-fit.png"))
    args = parser.parse_args()
    minima = select_isoflops_minima(load_runs(args.data))
    model_law, data_law = fit_isoflops(minima)
    plot_fit(minima, model_law, data_law, args.plot)
    result = {
        "model_law": asdict(model_law),
        "data_law": asdict(data_law),
        "predictions": {
            f"{compute:.0e}": {
                "parameters": float(model_law(compute)),
                "tokens": float(data_law(compute)),
            }
            for compute in (1e23, 1e24)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
