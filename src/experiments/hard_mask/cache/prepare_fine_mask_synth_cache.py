#!/usr/bin/env python3
"""Build FuncDec caches for SM1-SM10 fine_mask Fourier synthetic .npz files.

Produces:
  backbone_emb_c512_h{H}_stride1.pt     (GPU required)
  fourier_basis_fine_mask_h{H}.pt       (per-harmonic masking, 4 families)
  raw_futures_h{H}.pt
  component_targets_h{H}.pt
  seasonal_coefficients_fine_mask_h{H}.pt  (4-family coefficients)

Output suffix: 10_4_2_8 (daily:10, weekly:4, monthly:2, yearly:8)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = next(parent for parent in THIS_FILE.parents if (parent / "loader_utils.py").exists())
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import resolve_data_path, resolve_project_path  # noqa: E402
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(resolve_project_path("/home/sia2/project/4.28basis")))
sys.path.insert(0, str(resolve_project_path("/home/sia2/project/4.28basis/src")))

from common import build_fine_mask_basis, FREQ_DAYS  # noqa: E402


ROOT = resolve_data_path("/home/sia2/project/data/synthetic")
EMBED_DIM = 1280
PATCH_LEN = 32
REVIN_TOL = 1e-6

VALID_HORIZONS = {96, 192, 336, 720}
# SM granularities: SM1-SM8 daily, SM9-SM12 hourly, SM13-SM14 weekly
SM_GRANULARITIES = {
    "SM1":  "daily",  "SM2":  "daily",  "SM3":  "daily",  "SM4":  "daily",
    "SM5":  "daily",  "SM6":  "daily",  "SM7":  "daily",  "SM8":  "daily",
    "SM9":  "hourly", "SM10": "hourly", "SM11": "hourly", "SM12": "hourly",
    "SM13": "weekly", "SM14": "weekly",
}

N_FOURIER_TERMS = {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path,
                        default=ROOT / "func_dec_syn_cent_fine_mask_train")
    parser.add_argument("--output_root", type=Path,
                        default=ROOT / "func_dec_syn_cent_fine_mask_train_cache_10_4_2_8")
    parser.add_argument("--eval_input_root", type=Path,
                        default=ROOT / "func_dec_syn_cent_fine_mask_eval")
    parser.add_argument("--eval_output_root", type=Path,
                        default=ROOT / "func_dec_syn_cent_fine_mask_eval_cache_10_4_2_8")
    parser.add_argument("--seasonal_levels", nargs="+", default=list(SM_GRANULARITIES.keys()),
                        choices=list(SM_GRANULARITIES.keys()))
    parser.add_argument("--trend_levels", nargs="+", default=["T1", "T2", "T3", "T4", "T5", "T6"])
    parser.add_argument("--compositions", nargs="+", default=["A1", "A2", "A3"],
                        choices=["A1", "A2", "A3"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Skip backbone embedding (no GPU needed). Write all other cache files.")
    parser.add_argument("--eval", action="store_true", help="Also process eval set.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: requested {device_name} but CUDA not available. Using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_name)


def load_backbone(device: torch.device, hf_cache_dir: str | None):
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.torch.util import revin, update_running_stats
    model = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
        cache_dir=hf_cache_dir,
    )
    model.model.to(device).eval()
    return model.model, revin, update_running_stats


def save_backbone_cache(
    npz_data: dict,
    granularity: str,
    out_path: Path,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> None:
    signal = torch.from_numpy(npz_data["signal"]).float()
    n_samples, L = signal.shape
    context_len = int(npz_data["meta"]["context_len"])
    context = signal[:, :context_len]

    all_embs = []
    all_mu = []
    all_sigma = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        ctx = context[start:end].to(device)
        B = ctx.shape[0]
        patches = ctx.reshape(B, -1, PATCH_LEN)
        masks = torch.zeros_like(patches, dtype=torch.bool)
        num_patches = patches.shape[1]

        n = torch.zeros(B, device=device)
        mu = torch.zeros(B, device=device)
        sigma = torch.zeros(B, device=device)
        patch_mu = []
        patch_sigma = []
        for pi in range(num_patches):
            (n, mu, sigma), _ = update_stats_fn(n, mu, sigma, patches[:, pi], masks[:, pi])
            patch_mu.append(mu.clone())
            patch_sigma.append(sigma.clone())

        ctx_mu = torch.stack(patch_mu, dim=1)
        ctx_sigma = torch.stack(patch_sigma, dim=1)
        normed = revin_fn(patches, ctx_mu, ctx_sigma, reverse=False)
        normed = torch.where(masks, 0.0, normed)

        with torch.no_grad():
            (_, embs, _, _), _ = backbone(normed, masks)
        all_embs.append(embs[:, -1, :].cpu())
        all_mu.append(ctx_mu[:, -1:].cpu())
        all_sigma.append(ctx_sigma[:, -1:].cpu())

    embeddings = torch.cat(all_embs, dim=0)
    mu_out = torch.cat(all_mu, dim=0)
    sigma_out = torch.cat(all_sigma, dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "mu": mu_out,
            "sigma": sigma_out,
            "context_len": int(context_len),
            "granularity": granularity,
        },
        out_path,
    )


def load_npz(path: Path) -> dict:
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


def find_npz(input_root: Path, composition: str, trend_level: str,
             seasonal_level: str, granularity: str, seed: int, horizon: int) -> Path:
    pattern = (
        f"{composition}_{trend_level}_{seasonal_level}_{granularity}_"
        f"seed{seed}_c*_h{horizon}.npz"
    )
    matches = sorted((input_root / "complex").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 file for {pattern} in {input_root / 'complex'}, found {len(matches)}"
        )
    return matches[0]


def cache_dir_path(output_root: Path, meta: dict) -> Path:
    return output_root / "complex" / (
        f"{meta['composition']}_{meta['trend_level']}_{meta['seasonal_level']}_"
        f"{meta['granularity']}_seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}"
    )


def save_raw_futures(npz_data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    futures = torch.from_numpy(npz_data["future_n"]).float()
    meta = npz_data["meta"]
    torch.save(
        {
            "futures_n": futures,
            "valid_mask": torch.ones(futures.shape[0], dtype=torch.bool),
            "context_len": int(meta["context_len"]),
            "horizon": int(meta["horizon"]),
        },
        out_path,
    )


def save_component_targets(npz_data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trend_n": torch.from_numpy(npz_data["gt_trend_n"]).float(),
            "seasonal_n": torch.from_numpy(npz_data["gt_seasonal_n"]).float(),
            "residual_n": torch.from_numpy(npz_data["gt_residual_n"]).float(),
        },
        out_path,
    )


def save_seasonal_coefficients(npz_data: dict, out_path: Path) -> None:
    """Save 4-family seasonal coefficients (daily, weekly, monthly, yearly)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = npz_data["meta"]
    samples = meta["samples"]
    n_samples = len(samples)
    active = set(meta["active_types"])

    tensors: dict[str, Any] = {}
    for family in ["daily", "weekly", "monthly", "yearly"]:
        n_terms = int(N_FOURIER_TERMS[family])
        values = np.zeros((n_samples, 2 * n_terms), dtype=np.float32)
        if family in active:
            for row_idx, sample in enumerate(samples):
                coeffs = sample["seasonal_coefficients_n"].get(family, [])
                for coef in coeffs:
                    k_idx = int(coef["k"]) - 1
                    if k_idx >= n_terms:
                        continue  # skip if exceeds capacity
                    values[row_idx, 2 * k_idx] = float(coef["a"])
                    values[row_idx, 2 * k_idx + 1] = float(coef["b"])
        tensors[f"{family}_coefficients"] = torch.from_numpy(values)

    tensors["mask"] = {family: family in active for family in ["daily", "weekly", "monthly", "yearly"]}
    tensors["n_fourier_terms"] = {k: int(v) for k, v in N_FOURIER_TERMS.items()}
    tensors["horizon"] = int(meta["horizon"])
    tensors["granularity"] = str(meta["granularity"])
    torch.save(tensors, out_path)


def save_basis(npz_data: dict, granularity: str, out_path: Path) -> None:
    """Build and save fine_mask per-harmonic Fourier basis."""
    meta = npz_data["meta"]
    context_len = int(meta["context_len"])
    horizon = int(meta["horizon"])
    freq = granularity  # granularity IS the frequency key
    bases = build_fine_mask_basis(freq, context_len, horizon)
    save_data = {
        "freq": freq,
        "granularity": granularity,
        "context_len": context_len,
        "horizon": horizon,
        "daily_basis": bases["daily_basis"],
        "weekly_basis": bases["weekly_basis"],
        "monthly_basis": bases["monthly_basis"],
        "yearly_basis": bases["yearly_basis"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, out_path)


def process_one(
    npz_path: Path,
    output_root: Path,
    granularity: str,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device,
    metadata_only: bool,
    skip_existing: bool,
) -> str:
    npz_data = load_npz(npz_path)
    meta = npz_data["meta"]
    context_len = int(meta["context_len"])
    horizon = int(meta["horizon"])
    ds_dir = cache_dir_path(output_root, meta)
    ds_dir.mkdir(parents=True, exist_ok=True)

    backbone_path = ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt"
    basis_path = ds_dir / f"fourier_basis_fine_mask_h{horizon}.pt"
    raw_path = ds_dir / f"raw_futures_h{horizon}.pt"
    component_path = ds_dir / f"component_targets_h{horizon}.pt"
    coeff_path = ds_dir / f"seasonal_coefficients_fine_mask_h{horizon}.pt"

    if skip_existing:
        need_backbone = not metadata_only and not backbone_path.exists()
        need_others = not (basis_path.exists() and raw_path.exists() and
                           component_path.exists() and coeff_path.exists())
        if not need_backbone and not need_others:
            return f"skip={ds_dir.name}"

    save_basis(npz_data, granularity, basis_path)
    save_raw_futures(npz_data, raw_path)
    save_component_targets(npz_data, component_path)
    save_seasonal_coefficients(npz_data, coeff_path)

    if not metadata_only:
        save_backbone_cache(npz_data, granularity, backbone_path, batch_size,
                            backbone, revin_fn, update_stats_fn, device)

    return f"saved={ds_dir.name}"


def main() -> None:
    args = parse_args()
    for horizon in args.horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon={horizon}")

    device = None
    backbone = None
    revin_fn = None
    update_stats_fn = None
    if not args.metadata_only:
        device = resolve_device(args.device)
        backbone, revin_fn, update_stats_fn = load_backbone(device, args.hf_cache_dir)

    pairs = [
        (args.input_root, args.output_root),
    ]
    if args.eval:
        pairs.append((args.eval_input_root, args.eval_output_root))

    for input_root, output_root in pairs:
        if not input_root.exists():
            print(f"[skip] input_root does not exist: {input_root}", flush=True)
            continue
        output_root.mkdir(parents=True, exist_ok=True)

        for composition in args.compositions:
            for trend_level in args.trend_levels:
                for seasonal_level in args.seasonal_levels:
                    granularity = SM_GRANULARITIES[seasonal_level]
                    for horizon in args.horizons:
                        try:
                            npz_path = find_npz(
                                input_root, composition, trend_level,
                                seasonal_level, granularity, args.seed, horizon,
                            )
                        except FileNotFoundError as exc:
                            print(f"[skip] {exc}", flush=True)
                            continue
                        result = process_one(
                            npz_path, output_root, granularity, args.batch_size,
                            backbone, revin_fn, update_stats_fn, device,
                            args.metadata_only, args.skip_existing,
                        )
                        print(f"[{result}] {composition}/{trend_level}/{seasonal_level}/{granularity} h{horizon}", flush=True)


if __name__ == "__main__":
    main()
