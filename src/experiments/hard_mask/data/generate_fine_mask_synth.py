#!/usr/bin/env python3
"""Generate fine_mask additive complex synthetic datasets (SM1-SM10).

Produces train and eval splits under:
  /home/sia2/project/data/synthetic/func_dec_syn_cent_fine_mask_train/
  /home/sia2/project/data/synthetic/func_dec_syn_cent_fine_mask_eval/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(Path("/home/sia2/project/data/synthetic")))

from synth_generator import (  # noqa: E402
    EPS_SIGMA,
    LEVELS,
    VALID_HORIZONS,
    generate_trend_dataset,
    load_config,
)
from synth_generator_fine_mask import (  # noqa: E402
    SEASONAL_GRANULARITIES_FINE,
    SEASONAL_LEVELS_FINE,
    generate_fine_seasonal_dataset,
)


ROOT = Path("/home/sia2/project/data/synthetic")
DEFAULT_TRAIN_OUTPUT = ROOT / "func_dec_syn_cent_fine_mask_train"
DEFAULT_EVAL_OUTPUT = ROOT / "func_dec_syn_cent_fine_mask_eval"
DEFAULT_CONFIG = THIS_DIR / "synth_config_fine_mask.yaml"
COMPOSITIONS = ["A1", "A2", "A3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train_output_root", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval_output_root", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--trend_levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--seasonal_levels", nargs="+", default=SEASONAL_LEVELS_FINE,
                        choices=SEASONAL_LEVELS_FINE)
    parser.add_argument("--compositions", nargs="+", default=COMPOSITIONS, choices=COMPOSITIONS)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--amplitude_low", type=float, default=0.5)
    parser.add_argument("--amplitude_high", type=float, default=1.5)
    return parser.parse_args()


def _seasonal_combos(levels: list[str]) -> list[tuple[str, str]]:
    combos = []
    for level in levels:
        for granularity in SEASONAL_GRANULARITIES_FINE[level]:
            combos.append((level, granularity))
    return combos


def _seed_for(base_seed: int, *parts: Any) -> int:
    value = int(base_seed)
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % (2 ** 32 - 1)
    return value


def _coefficients_from_seasonal_meta(
    seasonal_meta: dict[str, Any],
    amplitude_scale: np.ndarray,
    seasonal_denom: np.ndarray,
    total_denom: np.ndarray,
) -> list[dict[str, list[dict[str, float]]]]:
    out = []
    for idx, sample in enumerate(seasonal_meta["samples"]):
        scaled: dict[str, list[dict[str, float]]] = {}
        for family, coeffs in sample["coeffs"].items():
            scaled[family] = []
            for coef in coeffs:
                scale = float(amplitude_scale[idx]) / (
                    float(seasonal_denom[idx]) * float(total_denom[idx])
                )
                scaled[family].append(
                    {
                        "k": int(coef["k"]),
                        "a": float(coef["a"]) * scale,
                        "b": float(coef["b"]) * scale,
                    }
                )
        out.append(scaled)
    return out


def build_complex_fine_dataset(
    composition: str,
    trend_level: str,
    seasonal_level: str,
    granularity: str,
    horizon: int,
    context_len: int,
    n_samples: int,
    seed: int,
    cfg: dict[str, Any],
    noise_scale: float,
    amplitude_low: float,
    amplitude_high: float,
) -> dict[str, Any]:
    trend = generate_trend_dataset(
        level=trend_level,
        horizon=horizon,
        context_len=context_len,
        n_samples=n_samples,
        seed=_seed_for(seed, composition, trend_level, horizon, "trend"),
        cfg=cfg,
    )
    seasonal = generate_fine_seasonal_dataset(
        level=seasonal_level,
        granularity=granularity,
        horizon=horizon,
        context_len=context_len,
        n_samples=n_samples,
        seed=_seed_for(seed, composition, seasonal_level, granularity, horizon, "seasonal"),
        cfg=cfg,
    )

    rng = np.random.default_rng(
        _seed_for(seed, composition, trend_level, seasonal_level, granularity, horizon, "mix")
    )
    L = context_len + horizon
    seasonal_scale = np.ones(n_samples, dtype=np.float32)
    if composition == "A3":
        seasonal_scale = rng.uniform(amplitude_low, amplitude_high, size=n_samples).astype(np.float32)

    trend_sigma = np.where(trend["sigma"] >= EPS_SIGMA, trend["sigma"], 1.0).astype(np.float32)
    seasonal_sigma = np.where(seasonal["sigma"] >= EPS_SIGMA, seasonal["sigma"], 1.0).astype(np.float32)
    trend_component = ((trend["signal"] - trend["mu"][:, None]) / trend_sigma[:, None]).astype(np.float32)
    seasonal_component = (
        seasonal["signal"] / seasonal_sigma[:, None] * seasonal_scale[:, None]
    ).astype(np.float32)

    residual_signal = np.zeros((n_samples, L), dtype=np.float32)
    if composition == "A2":
        base = trend_component + seasonal_component
        sample_std = np.std(base[:, :context_len], axis=1).astype(np.float32)
        noise_std = np.maximum(sample_std * float(noise_scale), 1e-3).astype(np.float32)
        residual_signal = rng.normal(0.0, noise_std[:, None], size=(n_samples, L)).astype(np.float32)

    signal = trend_component + seasonal_component + residual_signal
    context = signal[:, :context_len]
    mu = np.mean(context, axis=1).astype(np.float32)
    sigma = np.std(context, axis=1).astype(np.float32)
    denom = np.where(sigma >= EPS_SIGMA, sigma, 1.0).astype(np.float32)

    future_raw = signal[:, context_len:]
    future_n = ((future_raw - mu[:, None]) / denom[:, None]).astype(np.float32)
    gt_trend_n = ((trend_component[:, context_len:] - mu[:, None]) / denom[:, None]).astype(np.float32)
    gt_seasonal_n = (seasonal_component[:, context_len:] / denom[:, None]).astype(np.float32)
    gt_residual_n = (residual_signal[:, context_len:] / denom[:, None]).astype(np.float32)

    seasonal_coefficients_n = _coefficients_from_seasonal_meta(
        seasonal["meta"],
        seasonal_scale,
        seasonal_sigma,
        denom,
    )
    samples = []
    for idx in range(n_samples):
        samples.append(
            {
                "trend": trend["meta"]["samples"][idx],
                "seasonal_coefficients_n": seasonal_coefficients_n[idx],
                "seasonal_scale": float(seasonal_scale[idx]),
            }
        )

    meta = {
        "category": "complex",
        "composition": composition,
        "trend_level": trend_level,
        "seasonal_level": seasonal_level,
        "granularity": granularity,
        "active_types": list(seasonal["active_types"]),
        "n_terms": seasonal["meta"]["n_terms"],
        "horizon": int(horizon),
        "context_len": int(context_len),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "noise_scale": float(noise_scale) if composition == "A2" else 0.0,
        "amplitude_low": float(amplitude_low) if composition == "A3" else 1.0,
        "amplitude_high": float(amplitude_high) if composition == "A3" else 1.0,
        "component_scale_mode": "component_context_normalized",
        "samples": samples,
    }
    return {
        "signal": signal.astype(np.float32),
        "future_n": future_n,
        "gt_trend_n": gt_trend_n,
        "gt_seasonal_n": gt_seasonal_n,
        "gt_residual_n": gt_residual_n,
        "mu": mu,
        "sigma": sigma,
        "active_types": list(seasonal["active_types"]),
        "granularity": granularity,
        "meta": meta,
    }


def save_complex_fine_dataset(dataset: dict[str, Any], output_dir: Path) -> Path:
    meta = dataset["meta"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{meta['composition']}_{meta['trend_level']}_{meta['seasonal_level']}_"
        f"{meta['granularity']}_seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}.npz"
    )
    np.savez_compressed(
        path,
        signal=dataset["signal"],
        future_n=dataset["future_n"],
        gt_trend_n=dataset["gt_trend_n"],
        gt_seasonal_n=dataset["gt_seasonal_n"],
        gt_residual_n=dataset["gt_residual_n"],
        mu=dataset["mu"],
        sigma=dataset["sigma"],
        active_types=np.asarray(dataset["active_types"], dtype=str),
        granularity=np.asarray(dataset["granularity"]),
        meta=json.dumps(meta),
    )
    return path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    global_cfg = cfg["global"]
    horizons = args.horizons or list(global_cfg["horizons"])
    context_len = args.context_len or int(global_cfg["context_len"])
    n_samples = args.n_samples or int(global_cfg["n_samples"])
    seed = args.seed if args.seed is not None else int(global_cfg["seed"])

    for horizon in horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon {horizon}; expected one of {sorted(VALID_HORIZONS)}")

    combos = _seasonal_combos(args.seasonal_levels)

    for output_root, split_seed_offset in [
        (args.train_output_root, 0),
        (args.eval_output_root, 99991),
    ]:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config": str(args.config),
            "output_root": str(output_root),
            "trend_levels": list(args.trend_levels),
            "seasonal_combos": [{"level": lv, "granularity": gr} for lv, gr in combos],
            "compositions": list(args.compositions),
            "horizons": list(horizons),
            "context_len": int(context_len),
            "n_samples": int(n_samples),
            "seed": int(seed + split_seed_offset),
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

        out_dir = output_root / "complex"
        for composition in args.compositions:
            for trend_level in args.trend_levels:
                for seasonal_level, granularity in combos:
                    for horizon in horizons:
                        dataset = build_complex_fine_dataset(
                            composition,
                            trend_level,
                            seasonal_level,
                            granularity,
                            int(horizon),
                            int(context_len),
                            int(n_samples),
                            int(seed + split_seed_offset),
                            cfg,
                            float(args.noise_scale),
                            float(args.amplitude_low),
                            float(args.amplitude_high),
                        )
                        saved_path = save_complex_fine_dataset(dataset, out_dir)
                        print(
                            f"saved={saved_path.name} future_n={dataset['future_n'].shape} "
                            f"composition={composition} active={dataset['active_types']}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
