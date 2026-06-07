#!/usr/bin/env python3
"""Build evaluation-only T+S datasets from existing trend and non-Fourier seasonal data."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any

import numpy as np

from nonfourier_generator import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT as DEFAULT_STAGE1_ROOT,
    EPS_SIGMA,
    GRANULARITIES,
    NONFOURIER_MODELS,
    VALID_HORIZONS,
    load_config,
)


ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic_nonF"
DEFAULT_TREND_ROOT = Path("/workspace/data/synthetic/func_dec_syn_cent_simple_fixed_phase_scale")
DEFAULT_OUTPUT_ROOT = ROOT / "synth_eval_nonfourier" / "stage2_T_S_nonfourier"
TREND_LEVELS = [f"T{i}" for i in range(1, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trend_root", type=Path, default=DEFAULT_TREND_ROOT)
    parser.add_argument("--seasonal_root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=NONFOURIER_MODELS, choices=NONFOURIER_MODELS)
    parser.add_argument("--trend_levels", nargs="+", default=TREND_LEVELS, choices=TREND_LEVELS)
    parser.add_argument("--granularities", nargs="+", default=list(GRANULARITIES), choices=list(GRANULARITIES))
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def seed_for(base_seed: int, *parts: Any) -> int:
    value = int(base_seed)
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % (2**32 - 1)
    return value


def load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        out = {
            "signal": data["signal"].astype(np.float32),
            "future_n": data["future_n"].astype(np.float32),
            "mu": data["mu"].astype(np.float32),
            "sigma": data["sigma"].astype(np.float32),
            "meta": meta,
        }
        if "gt_seasonal_n" in data:
            out["gt_seasonal_n"] = data["gt_seasonal_n"].astype(np.float32)
        return out


def trend_path(trend_root: Path, level: str, seed: int, context_len: int, horizon: int) -> Path:
    return trend_root / "trend" / f"{level}_seed{seed}_c{context_len}_h{horizon}.npz"


def seasonal_path(seasonal_root: Path, model: str, granularity: str, seed: int, context_len: int, horizon: int) -> Path:
    return seasonal_root / model / "seasonal" / f"{granularity}_seed{seed}_c{context_len}_h{horizon}.npz"


def select_rows(data: dict[str, Any], n_samples: int, seed: int) -> dict[str, Any]:
    total = int(data["signal"].shape[0])
    if total < n_samples:
        raise ValueError(f"Not enough samples: requested={n_samples} available={total}")
    if total == n_samples:
        indices = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total, size=n_samples, replace=False))
    selected = dict(data)
    for key in ["signal", "future_n", "mu", "sigma", "gt_seasonal_n"]:
        if key in selected:
            selected[key] = selected[key][indices]
    meta = dict(data["meta"])
    samples = meta.get("samples")
    if isinstance(samples, list) and len(samples) == total:
        meta["samples"] = [samples[int(i)] for i in indices]
    meta["source_n_samples"] = total
    meta["selected_indices"] = [int(i) for i in indices]
    selected["meta"] = meta
    return selected


def build_nonfourier_ts_dataset(
    model: str,
    trend_level: str,
    granularity: str,
    horizon: int,
    context_len: int,
    n_samples: int,
    seed: int,
    trend_root: Path,
    seasonal_root: Path,
) -> dict[str, Any]:
    trend = select_rows(
        load_npz(trend_path(trend_root, trend_level, seed, context_len, horizon)),
        n_samples,
        seed_for(seed, model, trend_level, horizon, "trend_select"),
    )
    seasonal = select_rows(
        load_npz(seasonal_path(seasonal_root, model, granularity, seed, context_len, horizon)),
        n_samples,
        seed_for(seed, model, granularity, horizon, "seasonal_select"),
    )

    total_len = int(context_len) + int(horizon)
    trend_sigma = np.where(trend["sigma"] >= EPS_SIGMA, trend["sigma"], 1.0).astype(np.float32)
    seasonal_sigma = np.where(seasonal["sigma"] >= EPS_SIGMA, seasonal["sigma"], 1.0).astype(np.float32)
    trend_component = ((trend["signal"] - trend["mu"][:, None]) / trend_sigma[:, None]).astype(np.float32)
    seasonal_component = (seasonal["signal"] / seasonal_sigma[:, None]).astype(np.float32)

    signal = (trend_component + seasonal_component).astype(np.float32)
    context = signal[:, :context_len]
    mu = np.mean(context, axis=1).astype(np.float32)
    sigma = np.std(context, axis=1).astype(np.float32)
    denom = np.where(sigma >= EPS_SIGMA, sigma, 1.0).astype(np.float32)

    future_raw = signal[:, context_len:]
    future_n = ((future_raw - mu[:, None]) / denom[:, None]).astype(np.float32)
    gt_trend_n = ((trend_component[:, context_len:] - mu[:, None]) / denom[:, None]).astype(np.float32)
    gt_seasonal_n = (seasonal_component[:, context_len:] / denom[:, None]).astype(np.float32)
    gt_residual_n = np.zeros_like(future_n)

    samples = []
    trend_samples = trend["meta"].get("samples", [{} for _ in range(n_samples)])
    seasonal_samples = seasonal["meta"].get("samples", [{} for _ in range(n_samples)])
    for idx in range(n_samples):
        samples.append({"trend": trend_samples[idx], "seasonal": seasonal_samples[idx]})

    meta = {
        "category": "complex",
        "generator_family": "nonfourier",
        "model": model,
        "trend_level": trend_level,
        "granularity": granularity,
        "active_types": list(seasonal["meta"]["active_types"]),
        "horizon": int(horizon),
        "context_len": int(context_len),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "trend_source": str(trend_path(trend_root, trend_level, seed, context_len, horizon)),
        "seasonal_source": str(seasonal_path(seasonal_root, model, granularity, seed, context_len, horizon)),
        "samples": samples,
        "note": "T+S eval data built from existing trend npz and non-Fourier seasonal npz.",
    }
    return {
        "signal": signal.astype(np.float32),
        "future_n": future_n,
        "gt_trend_n": gt_trend_n,
        "gt_seasonal_n": gt_seasonal_n,
        "gt_residual_n": gt_residual_n,
        "mu": mu,
        "sigma": sigma,
        "active_types": list(seasonal["meta"]["active_types"]),
        "granularity": granularity,
        "meta": meta,
    }


def output_path(output_root: Path, meta: dict[str, Any]) -> Path:
    return output_root / meta["model"] / "complex" / (
        f"{meta['trend_level']}_{meta['granularity']}_"
        f"seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}.npz"
    )


def save_dataset(dataset: dict[str, Any], output_root: Path) -> Path:
    meta = dataset["meta"]
    path = output_path(output_root, meta)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    horizons = [int(h) for h in (args.horizons or list(global_cfg["horizons"]))]
    context_len = int(args.context_len or int(global_cfg["context_len"]))
    seed = int(args.seed if args.seed is not None else int(global_cfg["seed"]))
    for horizon in horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon={horizon}; expected one of {sorted(VALID_HORIZONS)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config),
        "trend_root": str(args.trend_root),
        "seasonal_root": str(args.seasonal_root),
        "output_root": str(args.output_root),
        "models": list(args.models),
        "trend_levels": list(args.trend_levels),
        "granularities": list(args.granularities),
        "horizons": list(horizons),
        "context_len": int(context_len),
        "n_samples": int(args.n_samples),
        "seed": int(seed),
        "note": "Evaluation-only T+S data using existing T and non-Fourier S.",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for model in args.models:
        for trend_level in args.trend_levels:
            for granularity in args.granularities:
                for horizon in horizons:
                    expected = args.output_root / model / "complex" / (
                        f"{trend_level}_{granularity}_seed{seed}_c{context_len}_h{horizon}.npz"
                    )
                    if args.skip_existing and expected.exists():
                        print(f"skip_existing={expected}", flush=True)
                        continue
                    dataset = build_nonfourier_ts_dataset(
                        model=model,
                        trend_level=trend_level,
                        granularity=granularity,
                        horizon=int(horizon),
                        context_len=int(context_len),
                        n_samples=int(args.n_samples),
                        seed=int(seed),
                        trend_root=args.trend_root,
                        seasonal_root=args.seasonal_root,
                    )
                    saved_path = save_dataset(dataset, args.output_root)
                    print(f"saved_ts={saved_path} future_n={dataset['future_n'].shape}", flush=True)


if __name__ == "__main__":
    main()
