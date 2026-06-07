#!/usr/bin/env python3
"""Build FuncDec caches for non-Fourier T+S evaluation datasets."""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic_nonF"
SRC_DIR = Path(os.environ.get("PROJECT_ROOT", "/workspace")) / "4.28basis" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_nonfourier_ts import DEFAULT_OUTPUT_ROOT, TREND_LEVELS  # noqa: E402
from nonfourier_generator import (  # noqa: E402
    DEFAULT_CONFIG,
    GRANULARITIES,
    NONFOURIER_MODELS,
    VALID_HORIZONS,
    load_config,
)
from prepare_nonfourier_synth_cache import (  # noqa: E402
    EMBED_DIM,
    N_FOURIER_TERMS,
    build_fourier_basis,
    load_backbone,
    resolve_device,
    save_backbone_cache,
    save_raw_futures,
)


DEFAULT_CACHE_ROOT = ROOT / "synth_eval_nonfourier" / "stage2_T_S_nonfourier_cache_10_4_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--models", nargs="+", default=NONFOURIER_MODELS, choices=NONFOURIER_MODELS)
    parser.add_argument("--trend_levels", nargs="+", default=TREND_LEVELS, choices=TREND_LEVELS)
    parser.add_argument("--granularities", nargs="+", default=list(GRANULARITIES), choices=list(GRANULARITIES))
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--n_fourier_daily", type=int, default=N_FOURIER_TERMS["daily"])
    parser.add_argument("--n_fourier_weekly", type=int, default=N_FOURIER_TERMS["weekly"])
    parser.add_argument("--n_fourier_yearly", type=int, default=N_FOURIER_TERMS["yearly"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Write raw/Fourier/component cache files without TimesFM backbone embeddings.",
    )
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


def cache_dir(output_root: Path, model: str, meta: dict[str, Any]) -> Path:
    return output_root / model / "complex" / (
        f"{meta['trend_level']}_{meta['granularity']}_"
        f"seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}"
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


def save_zero_seasonal_coefficients(
    npz_data: dict[str, Any],
    active_types: list[str],
    n_fourier_terms: dict[str, int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = npz_data["meta"]
    n_samples = int(npz_data["future_n"].shape[0])
    tensors: dict[str, Any] = {}
    for family in ["daily", "weekly", "yearly"]:
        tensors[f"{family}_coefficients"] = torch.zeros(
            n_samples,
            2 * int(n_fourier_terms[family]),
            dtype=torch.float32,
        )
    tensors["mask"] = {family: family in set(active_types) for family in ["daily", "weekly", "yearly"]}
    tensors["n_fourier_terms"] = {key: int(value) for key, value in n_fourier_terms.items()}
    tensors["horizon"] = int(meta["horizon"])
    tensors["granularity"] = str(meta["granularity"])
    tensors["note"] = "Non-Fourier T+S data has no ground-truth Fourier coefficients; zeros are placeholders."
    torch.save(tensors, out_path)


def validate_ts_cache(ds_dir: Path, horizon: int, active_types: list[str], backbone_path: Path | None) -> None:
    raw = torch.load(ds_dir / f"raw_futures_h{horizon}.pt", map_location="cpu", weights_only=False)
    basis = torch.load(ds_dir / f"fourier_basis_h{horizon}.pt", map_location="cpu", weights_only=False)
    components = torch.load(ds_dir / f"component_targets_h{horizon}.pt", map_location="cpu", weights_only=False)
    coeffs = torch.load(ds_dir / f"seasonal_coefficients_h{horizon}.pt", map_location="cpu", weights_only=False)
    if not bool(raw["valid_mask"].all().item()):
        raise AssertionError(f"valid_mask contains False: {ds_dir}")
    if raw["futures_n"].shape != components["trend_n"].shape:
        raise AssertionError(f"future/trend shape mismatch: {ds_dir}")
    if components["trend_n"].shape != components["seasonal_n"].shape:
        raise AssertionError(f"component shape mismatch: {ds_dir}")
    active = set(active_types)
    for family in ["daily", "weekly", "yearly"]:
        tensor = basis[f"{family}_basis"]
        nonzero = bool(torch.count_nonzero(tensor).item() > 0)
        if family in active and not nonzero:
            raise AssertionError(f"active basis is zero: {ds_dir} {family}")
        if family not in active and nonzero:
            raise AssertionError(f"inactive basis is non-zero: {ds_dir} {family}")
        expected_width = 2 * int(coeffs["n_fourier_terms"][family])
        if coeffs[f"{family}_coefficients"].shape[1] != expected_width:
            raise AssertionError(f"coefficient width mismatch: {ds_dir} {family}")
    if backbone_path is not None and backbone_path.exists():
        backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
        shape = tuple(backbone["embeddings"].shape)
        if len(shape) != 2 or shape[1] != EMBED_DIM:
            raise AssertionError(f"embedding shape mismatch: {backbone_path}: {shape}")


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

    for model in args.models:
        for trend_level in args.trend_levels:
            for granularity in args.granularities:
                for horizon in horizons:
                    npz_path = input_path(args.input_root, model, trend_level, granularity, seed, context_len, int(horizon))
                    npz_data = load_npz(npz_path)
                    meta = npz_data["meta"]
                    active_types = list(meta["active_types"])
                    ds_dir = cache_dir(args.output_root, model, meta)
                    ds_dir.mkdir(parents=True, exist_ok=True)

                    backbone_path = ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt"
                    basis_path = ds_dir / f"fourier_basis_h{horizon}.pt"
                    raw_path = ds_dir / f"raw_futures_h{horizon}.pt"
                    component_path = ds_dir / f"component_targets_h{horizon}.pt"
                    coeff_path = ds_dir / f"seasonal_coefficients_h{horizon}.pt"
                    if (
                        args.skip_existing
                        and raw_path.exists()
                        and basis_path.exists()
                        and component_path.exists()
                        and coeff_path.exists()
                        and (args.metadata_only or backbone_path.exists())
                    ):
                        validate_ts_cache(ds_dir, int(horizon), active_types, None if args.metadata_only else backbone_path)
                        print(f"skip_existing={ds_dir}", flush=True)
                        continue

                    torch.save(
                        build_fourier_basis(int(horizon), str(meta["granularity"]), active_types, n_fourier_terms, int(context_len)),
                        basis_path,
                    )
                    save_raw_futures(npz_data, raw_path)
                    save_component_targets(npz_data, component_path)
                    save_zero_seasonal_coefficients(npz_data, active_types, n_fourier_terms, coeff_path)
                    if args.metadata_only:
                        backbone_path_or_none = None
                        print(f"metadata_only: skipped backbone={backbone_path}", flush=True)
                    else:
                        assert device is not None
                        assert backbone is not None and revin_fn is not None and update_stats_fn is not None
                        save_backbone_cache(
                            npz_data, backbone_path, int(args.batch_size),
                            backbone, revin_fn, update_stats_fn, device,
                        )
                        backbone_path_or_none = backbone_path
                    validate_ts_cache(ds_dir, int(horizon), active_types, backbone_path_or_none)
                    print(f"saved_cache={ds_dir}", flush=True)


if __name__ == "__main__":
    main()
