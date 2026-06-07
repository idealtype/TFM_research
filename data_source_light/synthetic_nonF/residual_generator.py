#!/usr/bin/env python3
"""Residual noise generators with common SNR-based scaling."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gamma, pareto, t as student_t, weibull_min


RESIDUAL_DISTRIBUTIONS = ["normal", "student_t", "exponential", "gamma", "weibull", "pareto"]
SNR_CHOICES = [2, 4, 8, 16]
STUDENT_T_DF_CHOICES = [2, 5, 10]
EPS_STD = 1e-8


def sample_residual_params(distribution: str, rng: np.random.Generator) -> dict[str, Any]:
    if distribution == "normal":
        return {}
    if distribution == "student_t":
        return {"df": int(rng.choice(STUDENT_T_DF_CHOICES))}
    if distribution == "exponential":
        return {}
    if distribution in {"gamma", "weibull"}:
        return {"k": float(rng.uniform(0.5, 3.0))}
    if distribution == "pareto":
        return {"alpha": float(rng.uniform(2.5, 5.0))}
    raise ValueError(f"Unknown residual distribution: {distribution}")


def sample_snr(rng: np.random.Generator) -> int:
    return int(rng.choice(SNR_CHOICES))


def generate_residual(
    distribution: str,
    signal: np.ndarray,
    snr: float,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Generate zero-mean residual noise scaled to ``std(signal) / snr``."""
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=np.float64)
    n = int(signal.shape[0])
    signal_std = float(np.std(signal))
    target_std = signal_std / float(snr)

    if distribution == "normal":
        raw = rng.normal(0.0, 1.0, n)
    elif distribution == "student_t":
        df = int(kwargs.get("df", 5))
        raw = student_t.rvs(df=df, size=n, random_state=rng)
    elif distribution == "exponential":
        raw = rng.exponential(1.0, n)
    elif distribution == "gamma":
        k = float(kwargs.get("k", 1.0))
        raw = rng.gamma(shape=k, scale=1.0, size=n)
    elif distribution == "weibull":
        k = float(kwargs.get("k", 1.0))
        raw = weibull_min.rvs(c=k, size=n, random_state=rng)
    elif distribution == "pareto":
        alpha = float(kwargs.get("alpha", 3.0))
        raw = pareto.rvs(b=alpha, size=n, random_state=rng)
    else:
        raise ValueError(f"Unknown residual distribution: {distribution}")

    raw = np.asarray(raw, dtype=np.float64)
    raw = raw - float(np.mean(raw))
    raw_std = float(np.std(raw))
    if raw_std < EPS_STD or target_std < EPS_STD:
        return np.zeros(n, dtype=np.float32)
    return (raw / (raw_std + EPS_STD) * target_std).astype(np.float32)


def generate_residual_batch(
    distribution: str,
    signals: np.ndarray,
    base_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    signals = np.asarray(signals, dtype=np.float32)
    residuals = np.empty_like(signals, dtype=np.float32)
    params: list[dict[str, Any]] = []
    for idx, signal in enumerate(signals):
        rng = np.random.default_rng(int(base_seed) + idx * 1009)
        snr = sample_snr(rng)
        dist_params = sample_residual_params(distribution, rng)
        residuals[idx] = generate_residual(
            distribution,
            signal,
            snr,
            seed=int(base_seed) + idx * 1009 + 17,
            **dist_params,
        )
        params.append(
            {
                "distribution": distribution,
                "snr": int(snr),
                "target_noise_std": float(np.std(signal) / float(snr)),
                **dist_params,
            }
        )
    return residuals, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=RESIDUAL_DISTRIBUTIONS, required=True)
    parser.add_argument("--n", type=int, default=608)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal = np.sin(2.0 * np.pi * np.arange(args.n, dtype=np.float64) / 24.0)
    rng = np.random.default_rng(args.seed)
    snr = sample_snr(rng)
    params = sample_residual_params(args.distribution, rng)
    residual = generate_residual(args.distribution, signal, snr, seed=args.seed + 1, **params)
    summary = {
        "distribution": args.distribution,
        "snr": snr,
        "params": params,
        "shape": list(residual.shape),
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
