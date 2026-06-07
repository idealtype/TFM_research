#!/usr/bin/env python3
"""Non-Fourier seasonal synthetic generators for FuncDec evaluation."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import interp1d, make_interp_spline
from scipy.signal import sawtooth

try:
    import pywt
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    pywt = None


ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic_nonF"
DEFAULT_CONFIG = Path("/workspace/data/synthetic/synth_config.yaml")
DEFAULT_OUTPUT_ROOT = ROOT / "synth_eval_nonfourier" / "stage1_S_nonfourier"
VALID_HORIZONS = {96, 192, 336, 720}
EPS_SIGMA = 1e-6
NONFOURIER_MODELS = ["cyclic_spline", "sarima", "sawtooth", "daubechies", "symlet"]
SEASONALITY_PERIODS = {"daily": 1.0, "weekly": 7.0, "yearly": 365.25}
FREQ_DAYS = {"hourly": 1.0 / 24.0, "daily": 1.0, "weekly": 7.0}

# Granularity → active seasonality types.
# Period lengths at each granularity:
#   hourly : daily=24 steps, weekly=168 steps
#   daily  : weekly=7 steps, yearly=365 steps
#   weekly : yearly=~52 steps
GRANULARITIES: dict[str, list[str]] = {
    "hourly": ["daily", "weekly"],
    "daily":  ["weekly", "yearly"],
    "weekly": ["yearly"],
}


def period_steps(seasonality_type: str, granularity: str) -> float:
    if seasonality_type not in SEASONALITY_PERIODS:
        raise ValueError(f"Unknown seasonality_type={seasonality_type}")
    if granularity not in FREQ_DAYS:
        raise ValueError(f"Unknown granularity={granularity}")
    return SEASONALITY_PERIODS[seasonality_type] / FREQ_DAYS[granularity]


def _adjacent_diff_ok(values: np.ndarray, limit: float = 0.8) -> bool:
    wrapped = np.concatenate([values, values[:1]])
    return bool(np.max(np.abs(np.diff(wrapped))) <= limit)


def generate_cyclic_spline(
    period: float,
    n_knots: int,
    total_len: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    period = float(period)
    t_knots = np.linspace(0.0, period, int(n_knots), endpoint=False)

    for _attempt in range(100):
        diffs = rng.uniform(-0.4, 0.4, int(n_knots))
        y_knots = np.cumsum(diffs)
        y_knots = y_knots - y_knots.mean()
        y_knots = y_knots / (np.max(np.abs(y_knots)) + 1e-8)
        if _adjacent_diff_ok(y_knots):
            break
    else:
        y_knots = np.clip(y_knots, -0.8, 0.8)

    t_ext = np.append(t_knots, period)
    y_ext = np.append(y_knots, y_knots[0])
    spline = make_interp_spline(t_ext, y_ext, k=min(3, int(n_knots) - 1), bc_type="periodic")
    t_full = np.arange(total_len, dtype=np.float64)
    return np.asarray(spline(t_full % period), dtype=np.float64)


def generate_sarima_seasonal(
    period: int,
    phi_s: float,
    total_len: int,
    sigma: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    period = max(2, int(round(period)))
    y = np.zeros(int(total_len), dtype=np.float64)
    init_len = min(period, int(total_len))
    y[:init_len] = np.sin(2.0 * np.pi * np.arange(init_len) / period)
    for t in range(period, int(total_len)):
        y[t] = float(phi_s) * y[t - period] + rng.normal(0.0, float(sigma))
    return y


def generate_sawtooth(period: float, total_len: int, amplitude: float = 1.0, width: float = 1.0) -> np.ndarray:
    t = np.arange(total_len, dtype=np.float64)
    return float(amplitude) * sawtooth(2.0 * np.pi * t / float(period), width=float(width))


def _fallback_wavelet_period(period: int, family: str) -> np.ndarray:
    x = np.linspace(0.0, 1.0, int(period), endpoint=False)
    if family == "db":
        carrier = np.sin(2.0 * np.pi * x) + 0.55 * np.sin(4.0 * np.pi * x + 0.3)
        envelope = np.exp(-((x - 0.42) / 0.23) ** 2)
    else:
        carrier = np.cos(2.0 * np.pi * x - 0.2) - 0.45 * np.cos(6.0 * np.pi * x)
        envelope = np.exp(-((np.minimum(x, 1.0 - x)) / 0.32) ** 2)
    values = carrier * envelope
    values = values - values.mean()
    return values / (np.max(np.abs(values)) + 1e-8)


def generate_wavelet_seasonal(
    wavelet_name: str,
    period: int,
    total_len: int,
    level: int = 1,
) -> np.ndarray:
    period = max(8, int(round(period)))
    if pywt is not None:
        wavelet = pywt.Wavelet(wavelet_name)
        wavefun = wavelet.wavefun(level=level)
        psi = np.asarray(wavefun[1] if len(wavefun) == 5 else wavefun[0], dtype=np.float64)
        x_norm = np.linspace(0.0, 1.0, len(psi))
        t_period = np.linspace(0.0, 1.0, period)
        psi_resampled = interp1d(x_norm, psi, kind="cubic")(t_period)
        psi_resampled = psi_resampled - psi_resampled.mean()
        psi_resampled = psi_resampled / (np.max(np.abs(psi_resampled)) + 1e-8)
    else:
        family = "db" if wavelet_name.startswith("db") else "sym"
        psi_resampled = _fallback_wavelet_period(period, family)

    t_full = np.arange(total_len)
    return psi_resampled[t_full % period]


def _seed_for(base_seed: int, *parts: Any) -> int:
    value = int(base_seed)
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % (2**32 - 1)
    return value


def _normalize_component(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean()
    return values / (np.std(values) + EPS_SIGMA)


def _generate_component(
    model: str,
    seasonality_type: str,
    granularity: str,
    total_len: int,
    rng: np.random.Generator,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    period = period_steps(seasonality_type, granularity)
    if model == "cyclic_spline":
        n_knots = int(rng.integers(4, 9))
        values = generate_cyclic_spline(period, n_knots, total_len, seed=seed)
        params = {"period": float(period), "n_knots": n_knots}
    elif model == "sarima":
        phi_s = float(rng.uniform(0.7, 0.95))
        snr = float(rng.uniform(8.0, 16.0))
        sigma = 1.0 / snr
        values = generate_sarima_seasonal(int(round(period)), phi_s, total_len, sigma=sigma, seed=seed)
        params = {"period": int(round(period)), "phi_s": phi_s, "sigma": sigma, "snr": snr}
    elif model == "sawtooth":
        amplitude = float(rng.uniform(0.6, 1.2))
        width = float(rng.uniform(0.85, 1.0))
        values = generate_sawtooth(period, total_len, amplitude=amplitude, width=width)
        params = {"period": float(period), "amplitude": amplitude, "width": width}
    elif model in {"daubechies", "symlet"}:
        prefix = "db" if model == "daubechies" else "sym"
        order = int(rng.integers(4, 9))
        wavelet_name = f"{prefix}{order}"
        values = generate_wavelet_seasonal(wavelet_name, int(round(period)), total_len, level=1)
        params = {"period": int(round(period)), "wavelet": wavelet_name, "pywt_available": pywt is not None}
    else:
        raise ValueError(f"Unknown non-Fourier model: {model}")
    return _normalize_component(values), params


def generate_nonfourier_dataset(
    model: str,
    granularity: str,
    horizon: int,
    context_len: int,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    if model not in NONFOURIER_MODELS:
        raise ValueError(f"model must be one of {NONFOURIER_MODELS}, got {model}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {list(GRANULARITIES)}, got {granularity}")
    if int(horizon) not in VALID_HORIZONS:
        raise ValueError(f"horizon must be one of {sorted(VALID_HORIZONS)}, got {horizon}")

    active_types = GRANULARITIES[granularity]
    total_len = int(context_len) + int(horizon)
    signals = np.empty((int(n_samples), total_len), dtype=np.float32)
    future_n = np.empty((int(n_samples), int(horizon)), dtype=np.float32)
    mu = np.empty(int(n_samples), dtype=np.float32)
    sigma = np.empty(int(n_samples), dtype=np.float32)
    sample_meta: list[dict[str, Any]] = []

    for sample_idx in range(int(n_samples)):
        rng = np.random.default_rng(_seed_for(seed, model, granularity, horizon, sample_idx))
        components = []
        params = {}
        for seasonality_type in active_types:
            component, component_params = _generate_component(
                model,
                seasonality_type,
                granularity,
                total_len,
                rng,
                _seed_for(seed, model, granularity, horizon, sample_idx, seasonality_type),
            )
            scale = float(rng.uniform(0.7, 1.3))
            components.append(scale * component)
            params[seasonality_type] = {**component_params, "scale": scale}
        signal = np.sum(np.stack(components, axis=0), axis=0) / np.sqrt(len(components))
        signal = np.clip(signal, -4.0, 4.0)
        context = signal[: int(context_len)]
        sample_mu = float(np.mean(context))
        sample_sigma = float(np.std(context))
        denom = sample_sigma if sample_sigma >= EPS_SIGMA else 1.0
        signal_n = (signal - sample_mu) / denom

        signals[sample_idx] = signal.astype(np.float32)
        future_n[sample_idx] = signal_n[int(context_len) :].astype(np.float32)
        mu[sample_idx] = sample_mu
        sigma[sample_idx] = sample_sigma
        sample_meta.append(params)

    meta = {
        "category": "seasonal",
        "generator_family": "nonfourier",
        "model": model,
        "granularity": granularity,
        "horizon": int(horizon),
        "context_len": int(context_len),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "active_types": active_types,
        "samples": sample_meta,
    }
    return {
        "signal": signals,
        "future_n": future_n,
        "gt_seasonal_n": future_n.copy(),
        "mu": mu,
        "sigma": sigma,
        "active_types": active_types,
        "granularity": granularity,
        "meta": meta,
    }


def save_nonfourier_dataset(dataset: dict[str, Any], output_root: Path) -> Path:
    meta = dataset["meta"]
    out_dir = output_root / meta["model"] / "seasonal"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"{meta['granularity']}_seed{meta['seed']}_"
        f"c{meta['context_len']}_h{meta['horizon']}.npz"
    )
    np.savez_compressed(
        path,
        signal=dataset["signal"],
        future_n=dataset["future_n"],
        gt_seasonal_n=dataset["gt_seasonal_n"],
        mu=dataset["mu"],
        sigma=dataset["sigma"],
        active_types=np.asarray(dataset["active_types"], dtype=str),
        granularity=np.asarray(dataset["granularity"]),
        meta=json.dumps(meta),
    )
    return path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=NONFOURIER_MODELS, choices=NONFOURIER_MODELS)
    parser.add_argument("--granularities", nargs="+", default=list(GRANULARITIES), choices=list(GRANULARITIES))
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    global_cfg = cfg["global"]
    horizons = args.horizons or list(global_cfg["horizons"])
    context_len = args.context_len or int(global_cfg["context_len"])
    seed = args.seed if args.seed is not None else int(global_cfg["seed"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config),
        "output_root": str(args.output_root),
        "models": list(args.models),
        "granularities": list(args.granularities),
        "horizons": list(horizons),
        "context_len": int(context_len),
        "n_samples": int(args.n_samples),
        "seed": int(seed),
        "note": "Evaluation-only non-Fourier seasonal synthetic data.",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for model in args.models:
        for granularity in args.granularities:
            for horizon in horizons:
                dataset = generate_nonfourier_dataset(
                    model=model,
                    granularity=granularity,
                    horizon=int(horizon),
                    context_len=int(context_len),
                    n_samples=int(args.n_samples),
                    seed=int(seed),
                )
                saved_path = save_nonfourier_dataset(dataset, args.output_root)
                print(
                    f"nonfourier saved={saved_path} future_n={dataset['future_n'].shape} "
                    f"active={','.join(dataset['active_types'])}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
