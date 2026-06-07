#!/usr/bin/env python3
"""Synthetic dataset generator for April evaluation."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic"
DEFAULT_CONFIG = ROOT / "synth_config.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "trend"
DEFAULT_SEASONAL_OUTPUT_DIR = ROOT / "seasonal"
DEFAULT_DEBUG_DIR = ROOT / "debug"
LEVELS = [f"T{i}" for i in range(1, 7)]
SEASONAL_LEVELS = [f"S{i}" for i in range(1, 11)]
VALID_HORIZONS = {96, 192, 336, 720}
EPS_SIGMA = 1e-6
TREND_REJECTION_LEVELS = {"T4", "T5", "T6"}
MAX_TREND_REJECTION_ATTEMPTS = 100
SEASONALITY_PERIODS = {"daily": 1.0, "weekly": 7.0, "yearly": 365.25}
FREQ_DAYS = {"hourly": 1.0 / 24.0, "daily": 1.0, "weekly": 7.0}
SEASONAL_GRANULARITIES = {
    "S1": ["hourly"],
    "S2": ["hourly"],
    "S3": ["hourly", "daily"],
    "S4": ["hourly", "daily"],
    "S5": ["weekly"],
    "S6": ["weekly"],
    "S7": ["weekly"],
    "S8": ["hourly"],
    "S9": ["hourly"],
    "S10": ["weekly"],
}


# y(t) = intercept + slopes[0]*t
#        + sum_j (slopes[j+1] - slopes[j]) * ReLU(t - breakpoints[j])
# breakpoints: integer indices in the full interval [0, L), sorted ascending
# slopes: len(breakpoints)+1 values
def piecewise_linear(
    t: np.ndarray,
    breakpoints: np.ndarray,
    slopes: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Evaluate the continuous hinge form of a piecewise linear curve."""
    t = np.asarray(t, dtype=np.float64)
    breakpoints = np.asarray(breakpoints, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)

    y = intercept + slopes[0] * t
    for j, breakpoint in enumerate(breakpoints):
        y += (slopes[j + 1] - slopes[j]) * np.maximum(t - breakpoint, 0.0)
    return y


def _place_even_jittered(
    n_breaks: int,
    low: int,
    high: int,
    rng: np.random.Generator,
) -> list[int]:
    if n_breaks <= 0:
        return []
    if high < low:
        raise ValueError(f"Invalid breakpoint range: low={low}, high={high}")

    anchors = np.linspace(low, high, n_breaks + 2, dtype=np.float64)[1:-1]
    spacing = (high - low) / float(n_breaks + 1)
    jitter = rng.uniform(-0.2 * spacing, 0.2 * spacing, size=n_breaks)
    breaks = np.rint(anchors + jitter).astype(int)
    breaks = np.clip(breaks, low, high)

    used: set[int] = set()
    repaired: list[int] = []
    for value in sorted(int(x) for x in breaks):
        candidate = value
        if candidate in used:
            right = candidate
            while right <= high and right in used:
                right += 1
            left = candidate
            while left >= low and left in used:
                left -= 1
            if right <= high:
                candidate = right
            elif left >= low:
                candidate = left
            else:
                raise ValueError(
                    f"Cannot place {n_breaks} unique breakpoints in [{low}, {high}]"
                )
        used.add(candidate)
        repaired.append(candidate)
    return sorted(repaired)


def place_breaks(
    n_context_breaks: int,
    n_horizon_breaks: int,
    L: int,
    context_len: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place context and horizon breakpoints with even spacing plus jitter."""
    context_margin = int(L * 0.05)
    context_low = context_margin
    context_high = context_len - context_margin

    horizon_low = context_len + int(horizon * 0.15)
    horizon_high = context_len + int(horizon * 0.85)

    context_breaks = _place_even_jittered(
        n_context_breaks, context_low, context_high, rng
    )
    horizon_breaks = _place_even_jittered(
        n_horizon_breaks, horizon_low, horizon_high, rng
    )
    return np.array(sorted(context_breaks + horizon_breaks), dtype=np.int64)


def _choice(pool: list[float], rng: np.random.Generator) -> float:
    if not pool:
        raise ValueError("Cannot sample from an empty slope pool")
    return float(rng.choice(np.asarray(pool, dtype=np.float64)))


def _place_one_break(
    level_cfg: dict[str, Any],
    L: int,
    context_len: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    horizon_prob = float(level_cfg.get("horizon_break_prob", 0.0))
    if rng.random() < horizon_prob:
        return place_breaks(0, 1, L, context_len, horizon, rng)
    return place_breaks(1, 0, L, context_len, horizon, rng)


def _net_direction(signal: np.ndarray, start: int, end: int) -> int:
    delta = float(signal[end] - signal[start])
    if abs(delta) < EPS_SIGMA:
        return 0
    return 1 if delta > 0 else -1


def _trend_direction_is_consistent(
    signal: np.ndarray,
    context_len: int,
    expected_direction: str | None = None,
) -> bool:
    full_sign = _net_direction(signal, 0, len(signal) - 1)
    context_sign = _net_direction(signal, 0, context_len - 1)
    horizon_sign = _net_direction(signal, context_len, len(signal) - 1)
    if full_sign == 0 or context_sign == 0 or horizon_sign == 0:
        return False
    if expected_direction == "positive" and full_sign < 0:
        return False
    if expected_direction == "negative" and full_sign > 0:
        return False
    return context_sign == full_sign and horizon_sign == full_sign


def _sample_breaks_and_slopes(
    level: str,
    horizon: int,
    context_len: int,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    L = context_len + horizon
    level_cfg = cfg["trend_levels"][level]

    if level == "T1":
        breakpoints = np.array([], dtype=np.int64)
        slopes = np.array([_choice(level_cfg["pool_pos"], rng)], dtype=np.float64)
    elif level == "T2":
        breakpoints = np.array([], dtype=np.int64)
        slopes = np.array([_choice(level_cfg["pool_neg"], rng)], dtype=np.float64)
    elif level == "T3":
        breakpoints = np.array([], dtype=np.int64)
        slopes = np.array([0.0], dtype=np.float64)
    elif level == "T4":
        breakpoints = _place_one_break(level_cfg, L, context_len, horizon, rng)
        slopes = np.array(
            [_choice(level_cfg["pool_free"], rng) for _ in range(2)],
            dtype=np.float64,
        )
    elif level == "T5":
        breakpoints = _place_one_break(level_cfg, L, context_len, horizon, rng)
        slopes = np.array(
            [
                _choice(level_cfg["pool_pos"], rng),
                _choice(level_cfg["pool_pos_soft"], rng),
            ],
            dtype=np.float64,
        )
    elif level == "T6":
        breakpoints = _place_one_break(level_cfg, L, context_len, horizon, rng)
        slopes = np.array(
            [
                _choice(level_cfg["pool_neg"], rng),
                _choice(level_cfg["pool_neg_soft"], rng),
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown trend level: {level}")

    return breakpoints, slopes


def generate_trend_dataset(
    level: str,
    horizon: int,
    context_len: int,
    n_samples: int,
    seed: int,
    cfg: dict,
) -> dict:
    """
    Returns:
      signal    [N, context_len+horizon]  raw signal before RevIN
      future_n  [N, horizon]              normalized future, equal to gt_trend_n
      mu        [N]
      sigma     [N]
      meta      dict                      generation parameters and per-sample params
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level}")
    if horizon not in VALID_HORIZONS:
        raise ValueError(f"horizon must be one of {sorted(VALID_HORIZONS)}, got {horizon}")
    if context_len <= 0 or horizon <= 0 or n_samples <= 0:
        raise ValueError("context_len, horizon, and n_samples must be positive")

    L = context_len + horizon
    t = np.arange(L, dtype=np.float64)
    rng = np.random.default_rng(seed)

    signals = np.empty((n_samples, L), dtype=np.float32)
    future_n = np.empty((n_samples, horizon), dtype=np.float32)
    mu = np.empty(n_samples, dtype=np.float32)
    sigma = np.empty(n_samples, dtype=np.float32)
    sample_meta: list[dict[str, Any]] = []

    for i in range(n_samples):
        accepted = False
        for attempt in range(1, MAX_TREND_REJECTION_ATTEMPTS + 1):
            breakpoints, slopes = _sample_breaks_and_slopes(
                level, horizon, context_len, cfg, rng
            )
            signal = piecewise_linear(t, breakpoints, slopes, intercept=0.0)
            context = signal[:context_len]
            sample_mu = float(np.mean(context))
            sample_sigma = float(np.std(context))

            if level in TREND_REJECTION_LEVELS and sample_sigma < EPS_SIGMA:
                continue
            expected_direction = None
            if level == "T5":
                expected_direction = "positive"
            elif level == "T6":
                expected_direction = "negative"
            if level in TREND_REJECTION_LEVELS and not _trend_direction_is_consistent(
                signal, context_len, expected_direction
            ):
                continue
            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                f"Failed to generate a non-degenerate {level} sample after "
                f"{MAX_TREND_REJECTION_ATTEMPTS} attempts; "
                f"context_len={context_len}, horizon={horizon}"
            )

        denom = sample_sigma if sample_sigma >= EPS_SIGMA else 1.0
        signal_n = (signal - sample_mu) / denom

        signals[i] = signal.astype(np.float32)
        future_n[i] = signal_n[context_len:].astype(np.float32)
        mu[i] = sample_mu
        sigma[i] = sample_sigma
        sample_meta.append(
            {
                "breakpoints": breakpoints.astype(int).tolist(),
                "slopes": slopes.astype(float).tolist(),
                "intercept": 0.0,
                "net_delta": float(signal[-1] - signal[0]),
                "context_delta": float(signal[context_len - 1] - signal[0]),
                "horizon_delta": float(signal[-1] - signal[context_len]),
                "resample_attempts": attempt,
            }
        )

    meta = {
        "level": level,
        "horizon": horizon,
        "context_len": context_len,
        "n_samples": n_samples,
        "seed": seed,
        "level_cfg": cfg["trend_levels"][level],
        "level_intent": cfg["trend_levels"][level].get("intent", ""),
        "samples": sample_meta,
    }
    return {
        "signal": signals,
        "future_n": future_n,
        "mu": mu,
        "sigma": sigma,
        "meta": meta,
    }


def save_trend_dataset(dataset: dict, output_dir: Path) -> Path:
    meta = dataset["meta"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{meta['level']}_seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}.npz"
    )
    np.savez_compressed(
        path,
        signal=dataset["signal"],
        future_n=dataset["future_n"],
        mu=dataset["mu"],
        sigma=dataset["sigma"],
        meta=json.dumps(meta),
    )
    return path


# y(t) = Σ_{type ∈ active_types}
#          Σ_{k=1}^{n_terms[type]}
#            a_k * sin(2π*k*t / P_type) + b_k * cos(2π*k*t / P_type)
#
# P_type (steps) = SEASONALITY_PERIODS[type] / FREQ_DAYS[granularity]
# SEASONALITY_PERIODS = {"daily": 1.0, "weekly": 7.0, "yearly": 365.25} in days
# FREQ_DAYS = {"hourly": 1/24, "daily": 1.0, "weekly": 7.0}
def fourier_seasonal(
    t: np.ndarray,
    active_types: list[str],
    n_terms: dict[str, int],
    granularity: str,
    coeffs: dict[str, list[dict[str, float]]],
) -> np.ndarray:
    """Evaluate a seasonal signal from sampled Fourier coefficients."""
    if granularity not in FREQ_DAYS:
        raise ValueError(f"Unknown granularity: {granularity}")

    t = np.asarray(t, dtype=np.float64)
    y = np.zeros_like(t, dtype=np.float64)
    for seasonality_type in active_types:
        period = SEASONALITY_PERIODS[seasonality_type] / FREQ_DAYS[granularity]
        for k in range(1, int(n_terms[seasonality_type]) + 1):
            coef = coeffs[seasonality_type][k - 1]
            angle = 2.0 * np.pi * k * t / period
            y += float(coef["a"]) * np.sin(angle) + float(coef["b"]) * np.cos(angle)
    return y


def _sample_fourier_coeffs(
    active_types: list[str],
    n_terms: dict[str, int],
    amplitude: float,
    decay_alpha: float,
    rng: np.random.Generator,
) -> dict[str, list[dict[str, float]]]:
    coeffs: dict[str, list[dict[str, float]]] = {}
    for seasonality_type in active_types:
        coeffs[seasonality_type] = []
        for k in range(1, int(n_terms[seasonality_type]) + 1):
            amplitude_k = amplitude / (k**decay_alpha)
            coeffs[seasonality_type].append(
                {
                    "k": k,
                    "a": float(rng.uniform(-amplitude_k, amplitude_k)),
                    "b": float(rng.uniform(-amplitude_k, amplitude_k)),
                    "amplitude_k": float(amplitude_k),
                }
            )
    return coeffs


def generate_seasonal_dataset(
    level: str,
    granularity: str,
    horizon: int,
    context_len: int,
    n_samples: int,
    seed: int,
    cfg: dict,
) -> dict:
    """
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
    if level not in SEASONAL_LEVELS:
        raise ValueError(f"level must be one of {SEASONAL_LEVELS}, got {level}")
    if granularity not in SEASONAL_GRANULARITIES[level]:
        raise ValueError(
            f"granularity={granularity} is invalid for {level}; "
            f"expected {SEASONAL_GRANULARITIES[level]}"
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

    L = context_len + horizon
    t = np.arange(L, dtype=np.float64)
    rng = np.random.default_rng(seed)

    signals = np.empty((n_samples, L), dtype=np.float32)
    future_n = np.empty((n_samples, horizon), dtype=np.float32)
    mu = np.empty(n_samples, dtype=np.float32)
    sigma = np.empty(n_samples, dtype=np.float32)
    sample_meta: list[dict[str, Any]] = []

    for i in range(n_samples):
        coeffs = _sample_fourier_coeffs(
            active_types, n_terms, amplitude, decay_alpha, rng
        )
        signal = fourier_seasonal(t, active_types, n_terms, granularity, coeffs)
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


def save_seasonal_dataset(dataset: dict, output_dir: Path = DEFAULT_SEASONAL_OUTPUT_DIR) -> Path:
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


def plot_seasonal_debug(dataset: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = dataset["meta"]
    signal = dataset["signal"][0]
    context_len = int(meta["context_len"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(signal)), signal, linewidth=1.5)
    ax.axvline(context_len, color="black", linestyle="-", linewidth=1.2, label="boundary")
    ax.set_title(f"{meta['level']} {meta['granularity']} h={meta['horizon']}")
    ax.set_xlabel("t")
    ax.set_ylabel("signal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def max_break_discontinuity(meta: dict[str, Any]) -> float:
    max_jump = 0.0
    eps = 1e-7
    for sample in meta["samples"]:
        breakpoints = np.asarray(sample["breakpoints"], dtype=np.float64)
        slopes = np.asarray(sample["slopes"], dtype=np.float64)
        intercept = float(sample["intercept"])
        for breakpoint in breakpoints:
            left = piecewise_linear(
                np.array([breakpoint - eps]), breakpoints, slopes, intercept
            )[0]
            right = piecewise_linear(
                np.array([breakpoint + eps]), breakpoints, slopes, intercept
            )[0]
            max_jump = max(max_jump, float(abs(right - left)))
    return max_jump


def plot_debug(dataset: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = dataset["meta"]
    signal = dataset["signal"][0]
    breakpoints = meta["samples"][0]["breakpoints"]
    context_len = int(meta["context_len"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(signal)), signal, linewidth=1.5)
    ax.axvline(context_len, color="black", linestyle="-", linewidth=1.2, label="boundary")
    for idx, breakpoint in enumerate(breakpoints):
        ax.axvline(
            breakpoint,
            color="tab:red",
            linestyle="--",
            alpha=0.7,
            linewidth=0.9,
            label="breakpoint" if idx == 0 else None,
        )
    ax.set_title(f"{meta['level']} h={meta['horizon']}")
    ax.set_xlabel("t")
    ax.set_ylabel("signal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--debug_dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--levels", nargs="+", default=None, choices=LEVELS)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    global_cfg = cfg["global"]

    levels = args.levels or LEVELS
    horizons = args.horizons or list(global_cfg["horizons"])
    context_len = args.context_len or int(global_cfg["context_len"])
    n_samples = args.n_samples or int(global_cfg["n_samples"])
    seed = args.seed if args.seed is not None else int(global_cfg["seed"])

    for horizon in horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon {horizon}; expected one of {sorted(VALID_HORIZONS)}")

    for level in levels:
        for horizon in horizons:
            dataset = generate_trend_dataset(
                level=level,
                horizon=horizon,
                context_len=context_len,
                n_samples=n_samples,
                seed=seed,
                cfg=cfg,
            )
            if dataset["future_n"].shape != (n_samples, horizon):
                raise AssertionError(
                    f"future_n shape mismatch for {level}, h={horizon}: "
                    f"{dataset['future_n'].shape}"
                )
            if dataset["mu"].shape != (n_samples,) or dataset["sigma"].shape != (n_samples,):
                raise AssertionError(
                    f"mu/sigma shape mismatch for {level}, h={horizon}: "
                    f"{dataset['mu'].shape}, {dataset['sigma'].shape}"
                )
            discontinuity = max_break_discontinuity(dataset["meta"])
            if discontinuity >= 1e-4:
                raise AssertionError(
                    f"Continuity check failed for {level}, h={horizon}: {discontinuity}"
                )

            saved_path = save_trend_dataset(dataset, args.output_dir)
            print(
                f"saved={saved_path} future_n={dataset['future_n'].shape} "
                f"mu={dataset['mu'].shape} sigma={dataset['sigma'].shape} "
                f"max_discontinuity={discontinuity:.3e}"
            )

            if args.debug and horizon == 96:
                plot_path = args.debug_dir / f"trend_sanity_{level}_h96.png"
                plot_debug(dataset, plot_path)
                print(f"debug_plot={plot_path}")


if __name__ == "__main__":
    main()
