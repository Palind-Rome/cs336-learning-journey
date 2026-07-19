"""Fit a Chinchilla-style loss surface and optimize N/D under C = 6ND."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


@dataclass(frozen=True)
class ChinchillaLaw:
    irreducible_loss: float
    model_coefficient: float
    model_exponent: float
    data_coefficient: float
    data_exponent: float

    def predict(self, parameters, tokens):
        n = np.asarray(parameters, dtype=np.float64)
        d = np.asarray(tokens, dtype=np.float64)
        return (
            self.irreducible_loss
            + self.model_coefficient / n**self.model_exponent
            + self.data_coefficient / d**self.data_exponent
        )


def _decode(theta: np.ndarray) -> ChinchillaLaw:
    # Positive parameterization prevents invalid negative coefficients/exponents.
    e, a, alpha, b, beta = np.exp(theta)
    return ChinchillaLaw(e, a, alpha, b, beta)


def fit_loss_surface(
    parameters: np.ndarray,
    tokens: np.ndarray,
    losses: np.ndarray,
) -> ChinchillaLaw:
    parameters = np.asarray(parameters, dtype=np.float64)
    tokens = np.asarray(tokens, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if not (len(parameters) == len(tokens) == len(losses)):
        raise ValueError("parameters, tokens and losses must have equal lengths")
    if len(losses) < 8:
        raise ValueError("collect at least eight completed runs before fitting")

    initial = np.log([max(0.1, losses.min() * 0.8), 100.0, 0.3, 100.0, 0.3])

    def residual(theta: np.ndarray) -> np.ndarray:
        law = _decode(theta)
        # Relative residuals prevent the noisiest high-loss pilots from dominating.
        return (law.predict(parameters, tokens) - losses) / losses

    result = least_squares(residual, initial, loss="soft_l1", f_scale=0.02)
    if not result.success:
        raise RuntimeError(result.message)
    return _decode(result.x)


def compute_optimal_nd(
    law: ChinchillaLaw,
    compute_budget: float,
    parameter_bounds: tuple[float, float],
) -> tuple[float, float, float]:
    """Return compute-optimal (N, D, predicted loss)."""

    low, high = map(np.log, parameter_bounds)

    def objective(log_parameters: float) -> float:
        parameters = np.exp(log_parameters)
        tokens = compute_budget / (6 * parameters)
        return float(law.predict(parameters, tokens))

    result = minimize_scalar(objective, bounds=(low, high), method="bounded")
    parameters = float(np.exp(result.x))
    tokens = float(compute_budget / (6 * parameters))
    return parameters, tokens, float(result.fun)


def bootstrap_optima(
    parameters: np.ndarray,
    tokens: np.ndarray,
    losses: np.ndarray,
    compute_budget: float,
    parameter_bounds: tuple[float, float],
    samples: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Quantify extrapolation uncertainty by refitting resampled runs."""

    rng = np.random.default_rng(seed)
    optima = []
    for _ in range(samples):
        index = rng.integers(0, len(losses), len(losses))
        try:
            law = fit_loss_surface(parameters[index], tokens[index], losses[index])
            optima.append(compute_optimal_nd(law, compute_budget, parameter_bounds))
        except (RuntimeError, ValueError, FloatingPointError):
            continue
    return np.asarray(optima)
