#!/usr/bin/env python3
"""Build FuncDec caches for non-Fourier T+S+R evaluation datasets."""

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

from generate_nonfourier_ts import TREND_LEVELS  # noqa: E402
from generate_nonfourier_tsr import DEFAULT_OUTPUT_ROOT  # noqa: E402
from nonfourier_generator import DEFAULT_CONFIG, GRANULARITIES, NONFOURIER_MODELS, VALID_HORIZONS, load_config  # noqa: E402
from prepare_nonfourier_synth_cache import (  # noqa: E402
    EMBED_DIM,
    N_FOURIER_TERMS,
    build_fourier_basis,
    load_backbone,
    resolve_device,
    save_backbone_cache,
    save_raw_futures,
)
from prepare_nonfourier_ts_cache import (  # noqa: E402
    save_component_targets,
    save_zero_seasonal_coefficients,
    validate_ts_cache,
)
from residual_generator import RESIDUAL_DISTRIBUTIONS  # noqa: E402


DEFAULT_CACHE_ROOT = ROOT / "synth_eval_nonfourier" / "stage3_T_S_R_nonfourier_cache_10_4_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--models", nargs="+", default=NONFOURIER_MODELS, choices=NONFOURIER_MODELS)
    parser.add_argument("--residual_distributions", nargs="+", default=RESIDUAL_DISTRIBUTIONS, choices=RESIDUAL_DISTRIBUTIONS)
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
    residual_distribution: str,
    trend_level: str,
    granularity: str,
    seed: int,
    context_len: int,
    horizon: int,
) -> Path:
    return input_root / model / residual_distribution / "complex" / (
        f"{trend_level}_{granularity}_seed{seed}_c{context_len}_h{horizon}.npz"
    )


def cache_dir(output_root: Path, model: str, residual_distribution: str, meta: dict[str, Any]) -> Path:
    return output_root / model / residual_distribution / "complex" / (
        f"{meta['trend_level']}_{meta['granularity']}_"
        f"seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}"
    )


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
        for residual_distribution in args.residual_distributions:
            for trend_level in args.trend_levels:
                for granularity in args.granularities:
                    for horizon in horizons:
                        npz_path = input_path(
                            args.input_root, model, residual_distribution,
                            trend_level, granularity, seed, context_len, int(horizon),
                        )
                        npz_data = load_npz(npz_path)
                        meta = npz_data["meta"]
                        active_types = list(meta["active_types"])
                        ds_dir = cache_dir(args.output_root, model, residual_distribution, meta)
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
                            validate_ts_cache(
                                ds_dir, int(horizon), active_types,
                                None if args.metadata_only else backbone_path,
                            )
                            print(f"skip_existing={ds_dir}", flush=True)
                            continue

                        torch.save(
                            build_fourier_basis(
                                int(horizon), str(meta["granularity"]), active_types,
                                n_fourier_terms, int(context_len),
                            ),
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
