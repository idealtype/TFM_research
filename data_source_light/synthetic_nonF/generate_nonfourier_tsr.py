#!/usr/bin/env python3
"""Build evaluation-only T+S+R datasets from Stage 2 non-Fourier T+S data."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any

import numpy as np

from generate_nonfourier_ts import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_STAGE2_ROOT,
    TREND_LEVELS,
    seed_for,
)
from nonfourier_generator import (
    DEFAULT_CONFIG,
    GRANULARITIES,
    NONFOURIER_MODELS,
    VALID_HORIZONS,
    load_config,
)
from residual_generator import RESIDUAL_DISTRIBUTIONS, generate_residual_batch


ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic_nonF"
DEFAULT_OUTPUT_ROOT = ROOT / "synth_eval_nonfourier" / "stage3_T_S_R_nonfourier"
EPS_SIGMA = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_STAGE2_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=NONFOURIER_MODELS, choices=NONFOURIER_MODELS)
    parser.add_argument("--residual_distributions", nargs="+", default=RESIDUAL_DISTRIBUTIONS, choices=RESIDUAL_DISTRIBUTIONS)
    parser.add_argument("--trend_levels", nargs="+", default=TREND_LEVELS, choices=TREND_LEVELS)
    parser.add_argument("--granularities", nargs="+", default=list(GRANULARITIES), choices=list(GRANULARITIES))
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=10, help="Subsample this many rows from each stage2 file. None = use all.")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return {
            "signal": data["signal"].astype(np.float32),
            "future_n": data["future_n"].astype(np.float32),
            "gt_trend_n": data["gt_trend_n"].astype(np.float32),
            "gt_seasonal_n": data["gt_seasonal_n"].astype(np.float32),
            "gt_residual_n": data["gt_residual_n"].astype(np.float32),
            "mu": data["mu"].astype(np.float32),
            "sigma": data["sigma"].astype(np.float32),
            "meta": meta,
        }


def input_path(
    input_root: Path,
    model: str,
    trend_level: str,
    granularity: str,
    seed: int,
    context_len: int,
    horizon: int,
) -> Path:
    return input_root / model / "complex" / (
        f"{trend_level}_{granularity}_seed{seed}_c{context_len}_h{horizon}.npz"
    )


def output_path(output_root: Path, meta: dict[str, Any]) -> Path:
    return output_root / meta["model"] / meta["residual_distribution"] / "complex" / (
        f"{meta['trend_level']}_{meta['granularity']}_"
        f"seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}.npz"
    )


def build_tsr_dataset(
    base: dict[str, Any],
    residual_distribution: str,
    seed: int,
) -> dict[str, Any]:
    meta = base["meta"]
    context_len = int(meta["context_len"])
    horizon = int(meta["horizon"])
    old_denom = np.where(base["sigma"] >= EPS_SIGMA, base["sigma"], 1.0).astype(np.float32)

    residual_signal, residual_params = generate_residual_batch(
        residual_distribution,
        base["signal"],
        seed_for(
            seed,
            meta["model"],
            residual_distribution,
            meta["trend_level"],
            meta["granularity"],
            horizon,
        ),
    )
    signal = (base["signal"] + residual_signal).astype(np.float32)
    context = signal[:, :context_len]
    mu = np.mean(context, axis=1).astype(np.float32)
    sigma = np.std(context, axis=1).astype(np.float32)
    denom = np.where(sigma >= EPS_SIGMA, sigma, 1.0).astype(np.float32)

    future_raw = signal[:, context_len:]
    future_n = ((future_raw - mu[:, None]) / denom[:, None]).astype(np.float32)

    trend_future_raw = (base["gt_trend_n"] * old_denom[:, None] + base["mu"][:, None]).astype(np.float32)
    seasonal_future_raw = (base["gt_seasonal_n"] * old_denom[:, None]).astype(np.float32)
    old_residual_future_raw = (base["gt_residual_n"] * old_denom[:, None]).astype(np.float32)
    new_residual_future_raw = (old_residual_future_raw + residual_signal[:, context_len:]).astype(np.float32)

    gt_trend_n = ((trend_future_raw - mu[:, None]) / denom[:, None]).astype(np.float32)
    gt_seasonal_n = (seasonal_future_raw / denom[:, None]).astype(np.float32)
    gt_residual_n = (new_residual_future_raw / denom[:, None]).astype(np.float32)

    samples = []
    base_samples = meta.get("samples", [{} for _ in range(signal.shape[0])])
    for idx, residual_meta in enumerate(residual_params):
        item = dict(base_samples[idx])
        item["residual"] = residual_meta
        samples.append(item)

    out_meta = dict(meta)
    out_meta.update(
        {
            "category": "complex",
            "generator_family": "nonfourier",
            "residual_distribution": residual_distribution,
            "stage2_source": str(input_path(
                Path(""),
                meta["model"],
                meta["trend_level"],
                meta["granularity"],
                int(meta["seed"]),
                context_len,
                horizon,
            )),
            "samples": samples,
            "note": "T+S+R eval data built by adding SNR-scaled residual noise to Stage 2 T+S data.",
        }
    )
    return {
        "signal": signal,
        "future_n": future_n,
        "gt_trend_n": gt_trend_n,
        "gt_seasonal_n": gt_seasonal_n,
        "gt_residual_n": gt_residual_n,
        "mu": mu,
        "sigma": sigma,
        "active_types": list(meta["active_types"]),
        "granularity": meta["granularity"],
        "meta": out_meta,
    }


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
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "models": list(args.models),
        "residual_distributions": list(args.residual_distributions),
        "trend_levels": list(args.trend_levels),
        "granularities": list(args.granularities),
        "horizons": list(horizons),
        "context_len": int(context_len),
        "seed": int(seed),
        "snr_choices": [2, 4, 8, 16],
        "note": "Evaluation-only T+S+R data using Stage 2 non-Fourier T+S plus SNR-scaled residuals.",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for model in args.models:
        for residual_distribution in args.residual_distributions:
            for trend_level in args.trend_levels:
                for granularity in args.granularities:
                    for horizon in horizons:
                        path = input_path(args.input_root, model, trend_level, granularity, seed, context_len, int(horizon))
                        base = load_npz(path)
                        if args.n_samples is not None and args.n_samples < base["signal"].shape[0]:
                            sub_rng = np.random.default_rng(seed_for(seed, model, residual_distribution, trend_level, granularity, horizon, "tsr_sub"))
                            idx = np.sort(sub_rng.choice(base["signal"].shape[0], size=args.n_samples, replace=False))
                            for k in ["signal", "future_n", "gt_trend_n", "gt_seasonal_n", "gt_residual_n", "mu", "sigma"]:
                                base[k] = base[k][idx]
                            sub_meta = dict(base["meta"])
                            if isinstance(sub_meta.get("samples"), list):
                                sub_meta["samples"] = [sub_meta["samples"][int(i)] for i in idx]
                            sub_meta["n_samples"] = args.n_samples
                            base["meta"] = sub_meta
                        expected_meta = dict(base["meta"])
                        expected_meta["residual_distribution"] = residual_distribution
                        expected = output_path(args.output_root, expected_meta)
                        if args.skip_existing and expected.exists():
                            print(f"skip_existing={expected}", flush=True)
                            continue
                        dataset = build_tsr_dataset(base, residual_distribution, seed)
                        saved_path = save_dataset(dataset, args.output_root)
                        print(f"saved_tsr={saved_path} future_n={dataset['future_n'].shape}", flush=True)


if __name__ == "__main__":
    main()
