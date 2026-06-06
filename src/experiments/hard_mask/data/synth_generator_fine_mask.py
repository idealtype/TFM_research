#!/usr/bin/env python3
"""Fine-mask synthetic dataset generator. Extends synth_generator.py with monthly period support."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Re-export shared utilities from the original generator
sys.path.insert(0, str(Path("/home/sia2/project/data/synthetic")))
from synth_generator import (  # noqa: E402
    EPS_SIGMA,
    LEVELS,
    VALID_HORIZONS,
    MAX_TREND_REJECTION_ATTEMPTS,
    TREND_REJECTION_LEVELS,
    generate_trend_dataset,
    save_trend_dataset,
    piecewise_linear,
    place_breaks,
    _place_one_break,
    _sample_breaks_and_slopes,
    load_config,
)


# Extended seasonality periods including monthly
SEASONALITY_PERIODS = {
    "daily": 1.0,
    "weekly": 7.0,
    "monthly": 30.4375,
    "yearly": 365.25,
}

# Frequency in days per step (granularity of the generated signal)
FREQ_DAYS = {
    "hourly": 1.0 / 24.0,
    "daily": 1.0,
    "weekly": 7.0,
}

# SM1-SM8: daily, SM9-SM10: hourly, SM11-SM12: hourly (new), SM13-SM14: weekly (new)
SEASONAL_LEVELS_FINE = [f"SM{i}" for i in range(1, 15)]
SEASONAL_GRANULARITIES_FINE = {
    "SM1":  ["daily"],
    "SM2":  ["daily"],
    "SM3":  ["daily"],
    "SM4":  ["daily"],
    "SM5":  ["daily"],
    "SM6":  ["daily"],
    "SM7":  ["daily"],
    "SM8":  ["daily"],
    "SM9":  ["hourly"],
    "SM10": ["hourly"],
    "SM11": ["hourly"],
    "SM12": ["hourly"],
    "SM13": ["weekly"],
    "SM14": ["weekly"],
}


def fourier_seasonal_fine(
    t: np.ndarray,
    active_types: list[str],
    n_terms: dict[str, int],
    granularity: str,
    coeffs: dict[str, list[dict[str, float]]],
) -> np.ndarray:
    """Evaluate a seasonal signal with monthly support.

    Supports active_types from {daily, weekly, monthly, yearly}.
    """
    if granularity not in FREQ_DAYS:
        raise ValueError(f"Unknown granularity: {granularity}")

    t = np.asarray(t, dtype=np.float64)
    y = np.zeros_like(t, dtype=np.float64)
    for seasonality_type in active_types:
        if seasonality_type not in SEASONALITY_PERIODS:
            raise ValueError(f"Unknown seasonality_type: {seasonality_type}")
        period = SEASONALITY_PERIODS[seasonality_type] / FREQ_DAYS[granularity]
        for k in range(1, int(n_terms[seasonality_type]) + 1):
            coef = coeffs[seasonality_type][k - 1]
            angle = 2.0 * np.pi * k * t / period
            y += float(coef["a"]) * np.sin(angle) + float(coef["b"]) * np.cos(angle)
    return y


def _sample_fourier_coeffs_fine(
    active_types: list[str],
    n_terms: dict[str, int],
    amplitude: float,
    decay_alpha: float,
    rng: np.random.Generator,
) -> dict[str, list[dict[str, float]]]:
    """Sample Fourier coefficients for all active types including monthly."""
    coeffs: dict[str, list[dict[str, float]]] = {}
    for seasonality_type in active_types:
        coeffs[seasonality_type] = []
        for k in range(1, int(n_terms[seasonality_type]) + 1):
            amplitude_k = amplitude / (k ** decay_alpha)
            coeffs[seasonality_type].append(
                {
                    "k": k,
                    "a": float(rng.uniform(-amplitude_k, amplitude_k)),
                    "b": float(rng.uniform(-amplitude_k, amplitude_k)),
                    "amplitude_k": float(amplitude_k),
                }
            )
    return coeffs


def generate_fine_seasonal_dataset(
    level: str,
    granularity: str,
    horizon: int,
    context_len: int,
    n_samples: int,
    seed: int,
    cfg: dict,
) -> dict:
    """Generate a seasonal-only dataset for SM levels (with monthly support).

    Returns:
      signal        [N, context_len+horizon]
      future_n      [N, horizon]   normalized future
      mu            [N]
      sigma         [N]
      gt_seasonal_n [N, horizon]   equal to future_n for seasonal-only data
      active_types  list[str]
      granularity   str
      meta          dict
    """
    if level not in SEASONAL_LEVELS_FINE:
        raise ValueError(f"level must be one of {SEASONAL_LEVELS_FINE}, got {level}")
    if granularity not in SEASONAL_GRANULARITIES_FINE.get(level, []):
        raise ValueError(
            f"granularity={granularity} is invalid for {level}; "
            f"expected {SEASONAL_GRANULARITIES_FINE[level]}"
        )
    if horizon not in VALID_HORIZONS:
        raise ValueError(f"horizon must be one of {sorted(VALID_HORIZONS)}, got {horizon}")
    if context_len <= 0 or horizon <= 0 or n_samples <= 0:
        raise ValueError("context_len, horizon, and n_samples must be positive")

    seasonal_cfg = cfg["seasonal_levels"]
    level_cfg = seasonal_cfg[level]
    active_types = list(level_cfg["active"])
    n_terms = {key: int(value) for key, value in level_cfg["n_terms"].items()}
    amplitude = float(seasonal_cfg["amplitude"])
    decay_alpha = float(seasonal_cfg["decay_alpha"])

    # Validate n_terms <= N_FOURIER_TERMS_MAX
    max_terms = seasonal_cfg.get("N_FOURIER_TERMS_MAX", {})
    for family, n in n_terms.items():
        if family in max_terms and n > int(max_terms[family]):
            raise ValueError(
                f"{level}: n_terms[{family}]={n} exceeds N_FOURIER_TERMS_MAX={max_terms[family]}"
            )

    L = context_len + horizon
    t = np.arange(L, dtype=np.float64)
    rng = np.random.default_rng(seed)

    signals = np.empty((n_samples, L), dtype=np.float32)
    future_n = np.empty((n_samples, horizon), dtype=np.float32)
    mu = np.empty(n_samples, dtype=np.float32)
    sigma = np.empty(n_samples, dtype=np.float32)
    sample_meta: list[dict[str, Any]] = []

    for i in range(n_samples):
        coeffs = _sample_fourier_coeffs_fine(active_types, n_terms, amplitude, decay_alpha, rng)
        signal = fourier_seasonal_fine(t, active_types, n_terms, granularity, coeffs)
        context = signal[:context_len]
        sample_mu = float(np.mean(context))
        sample_sigma = float(np.std(context))
        denom = sample_sigma if sample_sigma >= EPS_SIGMA else 1.0
        signal_n = (signal - sample_mu) / denom

        signals[i] = signal.astype(np.float32)
        future_n[i] = signal_n[context_len:].astype(np.float32)
        mu[i] = sample_mu
        sigma[i] = sample_sigma
        sample_meta.append({"coeffs": coeffs})

    meta = {
        "level": level,
        "granularity": granularity,
        "horizon": horizon,
        "context_len": context_len,
        "n_samples": n_samples,
        "seed": seed,
        "active_types": active_types,
        "n_terms": n_terms,
        "amplitude": amplitude,
        "decay_alpha": decay_alpha,
        "level_cfg": level_cfg,
        "samples": sample_meta,
    }
    return {
        "signal": signals,
        "future_n": future_n,
        "mu": mu,
        "sigma": sigma,
        "gt_seasonal_n": future_n.copy(),
        "active_types": active_types,
        "granularity": granularity,
        "meta": meta,
    }


def save_fine_seasonal_dataset(dataset: dict, output_dir: Path) -> Path:
    meta = dataset["meta"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{meta['level']}_{meta['granularity']}_seed{meta['seed']}_"
        f"c{meta['context_len']}_h{meta['horizon']}.npz"
    )
    np.savez_compressed(
        path,
        signal=dataset["signal"],
        future_n=dataset["future_n"],
        mu=dataset["mu"],
        sigma=dataset["sigma"],
        gt_seasonal_n=dataset["gt_seasonal_n"],
        active_types=np.asarray(dataset["active_types"], dtype=str),
        granularity=np.asarray(dataset["granularity"]),
        meta=json.dumps(meta),
    )
    return path
