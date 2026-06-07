#!/usr/bin/env python3
"""Build FuncDec caches for additive complex synthetic .npz files."""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prepare_synth_cache import (
    EMBED_DIM,
    REVIN_TOL,
    build_fourier_basis,
    load_backbone,
    save_backbone_cache,
    validate_cache,
    resolve_device,
)
from synth_generator import LEVELS, SEASONAL_GRANULARITIES, SEASONAL_LEVELS, VALID_HORIZONS


ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--trend_levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--seasonal_levels", nargs="+", default=SEASONAL_LEVELS, choices=SEASONAL_LEVELS)
    parser.add_argument("--compositions", nargs="+", default=["A1", "A2", "A3"], choices=["A1", "A2", "A3"])
    parser.add_argument("--horizons", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--n_fourier_daily", type=int, default=10)
    parser.add_argument("--n_fourier_weekly", type=int, default=4)
    parser.add_argument("--n_fourier_yearly", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--metadata_only", action="store_true")
    return parser.parse_args()


def seasonal_combos(levels: list[str]) -> list[tuple[str, str]]:
    return [
        (level, granularity)
        for level in levels
        for granularity in SEASONAL_GRANULARITIES[level]
    ]


def parse_complex_name(name: str) -> dict[str, Any]:
    match = re.fullmatch(r"(A\d+)_(T\d+)_(S\d+)_(\w+)_seed(\d+)_c(\d+)_h(\d+)", name)
    if not match:
        raise ValueError(f"Invalid complex cache name: {name}")
    composition, trend_level, seasonal_level, granularity, seed, context_len, horizon = match.groups()
    return {
        "category": "complex",
        "composition": composition,
        "trend_level": trend_level,
        "seasonal_level": seasonal_level,
        "granularity": granularity,
        "seed": int(seed),
        "context_len": int(context_len),
        "horizon": int(horizon),
    }


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
    composition: str,
    trend_level: str,
    seasonal_level: str,
    granularity: str,
    seed: int,
    horizon: int,
) -> Path:
    pattern = (
        f"{composition}_{trend_level}_{seasonal_level}_{granularity}_"
        f"seed{seed}_c*_h{horizon}.npz"
    )
    matches = sorted((input_root / "complex").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file for {pattern}, found {len(matches)}")
    return matches[0]


def cache_dir(output_root: Path, meta: dict[str, Any]) -> Path:
    return output_root / "complex" / (
        f"{meta['composition']}_{meta['trend_level']}_{meta['seasonal_level']}_"
        f"{meta['granularity']}_seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}"
    )


def save_raw_futures(npz_data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = npz_data["meta"]
    futures = torch.from_numpy(npz_data["future_n"]).float()
    torch.save(
        {
            "futures_n": futures,
            "valid_mask": torch.ones(futures.shape[0], dtype=torch.bool),
            "context_len": int(meta["context_len"]),
            "horizon": int(meta["horizon"]),
        },
        out_path,
    )


def save_component_targets(npz_data: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trend_n": torch.from_numpy(npz_data["gt_trend_n"]).float(),
            "seasonal_n": torch.from_numpy(npz_data["gt_seasonal_n"]).float(),
            "residual_n": torch.from_numpy(npz_data["gt_residual_n"]).float(),
        },
        out_path,
    )


def save_seasonal_coefficients(
    npz_data: dict[str, Any],
    n_fourier_terms: dict[str, int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = npz_data["meta"]
    samples = meta["samples"]
    if len(samples) != npz_data["future_n"].shape[0]:
        raise ValueError("Coefficient sample count mismatch")

    active = set(meta["active_types"])
    tensors: dict[str, torch.Tensor] = {}
    for family in ["daily", "weekly", "yearly"]:
        n_terms = int(n_fourier_terms[family])
        values = np.zeros((len(samples), 2 * n_terms), dtype=np.float32)
        if family in active:
            for row_idx, sample in enumerate(samples):
                coeffs = sample["seasonal_coefficients_n"].get(family, [])
                for coef in coeffs:
                    k_idx = int(coef["k"]) - 1
                    if k_idx >= n_terms:
                        raise ValueError(f"{family} k={k_idx + 1} exceeds cache width {n_terms}")
                    values[row_idx, 2 * k_idx] = float(coef["a"])
                    values[row_idx, 2 * k_idx + 1] = float(coef["b"])
        tensors[f"{family}_coefficients"] = torch.from_numpy(values)

    tensors["mask"] = {family: family in active for family in ["daily", "weekly", "yearly"]}
    tensors["n_fourier_terms"] = {key: int(value) for key, value in n_fourier_terms.items()}
    tensors["horizon"] = int(meta["horizon"])
    tensors["granularity"] = str(meta["granularity"])
    torch.save(tensors, out_path)


def validate_complex_cache(ds_dir: Path, horizon: int, active_types: list[str], backbone_path: Path | None) -> None:
    validate_cache(ds_dir, backbone_path, horizon, active_types)
    basis_path = ds_dir / f"fourier_basis_h{horizon}.pt"
    component_path = ds_dir / f"component_targets_h{horizon}.pt"
    coeff_path = ds_dir / f"seasonal_coefficients_h{horizon}.pt"
    basis = torch.load(basis_path, map_location="cpu", weights_only=False)
    components = torch.load(component_path, map_location="cpu", weights_only=False)
    coeffs = torch.load(coeff_path, map_location="cpu", weights_only=False)
    if components["trend_n"].shape != components["seasonal_n"].shape:
        raise AssertionError(f"component shape mismatch: {component_path}")
    if coeffs["daily_coefficients"].shape[1] != 20:
        raise AssertionError(f"unexpected daily coeff width: {coeff_path}")
    recon = torch.zeros_like(components["seasonal_n"], dtype=torch.float64)
    for family in ["daily", "weekly", "yearly"]:
        recon = recon + torch.matmul(
            coeffs[f"{family}_coefficients"].double(),
            basis[f"{family}_basis"].double().T,
        )
    abs_err = torch.abs(recon - components["seasonal_n"].double())
    max_err = torch.max(abs_err).item()
    target_scale = torch.max(torch.abs(components["seasonal_n"].double())).item()
    allowed_err = max(1e-3, 1e-3 * target_scale)
    if max_err > allowed_err:
        raise AssertionError(
            f"seasonal coefficient reconstruction mismatch: {coeff_path} "
            f"max_err={max_err:.6g} allowed_err={allowed_err:.6g}"
        )


def main() -> None:
    args = parse_args()
    for horizon in args.horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon={horizon}")

    n_fourier_terms = {
        "daily": int(args.n_fourier_daily),
        "weekly": int(args.n_fourier_weekly),
        "yearly": int(args.n_fourier_yearly),
    }
    device = None
    backbone = None
    revin_fn = None
    update_stats_fn = None
    if not args.metadata_only:
        device = resolve_device(args.device)
        backbone, revin_fn, update_stats_fn = load_backbone(device, args.hf_cache_dir)

    for composition in args.compositions:
        for trend_level in args.trend_levels:
            for seasonal_level, granularity in seasonal_combos(args.seasonal_levels):
                for horizon in args.horizons:
                    npz_path = input_path(
                        args.input_root,
                        composition,
                        trend_level,
                        seasonal_level,
                        granularity,
                        args.seed,
                        int(horizon),
                    )
                    npz_data = load_npz(npz_path)
                    meta = npz_data["meta"]
                    context_len = int(meta["context_len"])
                    active_types = list(meta["active_types"])
                    ds_dir = cache_dir(args.output_root, meta)
                    ds_dir.mkdir(parents=True, exist_ok=True)

                    backbone_path = ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt"
                    basis_path = ds_dir / f"fourier_basis_h{horizon}.pt"
                    raw_path = ds_dir / f"raw_futures_h{horizon}.pt"
                    component_path = ds_dir / f"component_targets_h{horizon}.pt"
                    coeff_path = ds_dir / f"seasonal_coefficients_h{horizon}.pt"

                    if (
                        args.skip_existing
                        and backbone_path.exists()
                        and basis_path.exists()
                        and raw_path.exists()
                        and component_path.exists()
                        and coeff_path.exists()
                    ):
                        validate_complex_cache(ds_dir, int(horizon), active_types, backbone_path)
                        print(f"skip_existing={ds_dir}", flush=True)
                        continue

                    torch.save(
                        build_fourier_basis(
                            int(horizon),
                            str(meta["granularity"]),
                            active_types,
                            n_fourier_terms=n_fourier_terms,
                            context_len=context_len,
                        ),
                        basis_path,
                    )
                    save_raw_futures(npz_data, raw_path)
                    save_component_targets(npz_data, component_path)
                    save_seasonal_coefficients(npz_data, n_fourier_terms, coeff_path)

                    if args.metadata_only:
                        backbone_path_or_none = None
                        print(f"metadata_only: skipped backbone={backbone_path}", flush=True)
                    else:
                        assert device is not None
                        assert backbone is not None and revin_fn is not None and update_stats_fn is not None
                        save_backbone_cache(
                            npz_data,
                            str(meta["granularity"]),
                            backbone_path,
                            args.batch_size,
                            backbone,
                            revin_fn,
                            update_stats_fn,
                            device,
                        )
                        backbone_path_or_none = backbone_path

                    validate_complex_cache(ds_dir, int(horizon), active_types, backbone_path_or_none)
                    print(f"saved_cache={ds_dir}", flush=True)


if __name__ == "__main__":
    main()
