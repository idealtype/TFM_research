#!/usr/bin/env python3
"""Build Monash-compatible FuncDec caches from synthetic .npz files."""

from __future__ import annotations

import argparse
import os
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic"
SRC_DIR = Path(os.environ.get("PROJECT_ROOT", "/workspace")) / "4.28basis" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
PATCH_LEN = 32
EMBED_DIM = 1280
REVIN_TOL = 1e-6
N_FOURIER_TERMS = {"daily": 2, "weekly": 2, "yearly": 8}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=["trend", "seasonal"], required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_root", type=Path, default=ROOT)
    parser.add_argument("--output_root", type=Path, default=ROOT / "cache")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--granularities", nargs="+", default=None)
    parser.add_argument("--n_fourier_daily", type=int, default=N_FOURIER_TERMS["daily"])
    parser.add_argument("--n_fourier_weekly", type=int, default=N_FOURIER_TERMS["weekly"])
    parser.add_argument("--n_fourier_yearly", type=int, default=N_FOURIER_TERMS["yearly"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Write Fourier/raw caches and validate paths without loading TimesFM.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is not available.")
    return torch.device(device_name)


def load_backbone(device: torch.device, hf_cache_dir: str | None):
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.torch.util import revin, update_running_stats

    pretrained = TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_ID,
        torch_compile=False,
        cache_dir=hf_cache_dir,
    )
    backbone = pretrained.model.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    if backbone.p != PATCH_LEN or backbone.md != EMBED_DIM:
        raise ValueError(
            f"Unexpected TimesFM shape: patch_len={backbone.p}, embed_dim={backbone.md}"
        )
    return backbone, revin, update_running_stats


def load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return {
            "signal": data["signal"].astype(np.float32),
            "future_n": data["future_n"].astype(np.float32),
            "mu": data["mu"].astype(np.float32),
            "sigma": data["sigma"].astype(np.float32),
            "meta": meta,
        }


def input_path(
    input_root: Path,
    category: str,
    level: str,
    granularity: str,
    seed: int,
    context_len: int,
    horizon: int,
) -> Path:
    if category == "trend":
        return input_root / "trend" / f"{level}_seed{seed}_c{context_len}_h{horizon}.npz"
    return (
        input_root
        / "seasonal"
        / f"{level}_{granularity}_seed{seed}_c{context_len}_h{horizon}.npz"
    )


def cache_dir(
    output_root: Path,
    category: str,
    level: str,
    granularity: str,
    seed: int,
    context_len: int,
    horizon: int,
) -> Path:
    return (
        output_root
        / category
        / f"{level}_{granularity}_seed{seed}_c{context_len}_h{horizon}"
    )


def build_fourier_basis(
    horizon: int,
    granularity: str,
    active_types: list[str],
    n_fourier_terms: dict[str, int] | None = None,
    context_len: int = 0,
) -> dict[str, Any]:
    t = torch.arange(int(context_len), int(context_len) + int(horizon), dtype=torch.float32)
    active = set(active_types)
    n_fourier_terms = n_fourier_terms or N_FOURIER_TERMS
    save_data: dict[str, Any] = {
        "mask": {stype: stype in active for stype in ["daily", "weekly", "yearly"]},
        "freq": granularity,
        "horizon": int(horizon),
        "context_len": int(context_len),
        "n_fourier_terms": {key: int(value) for key, value in n_fourier_terms.items()},
    }

    for stype in ["daily", "weekly", "yearly"]:
        n = int(n_fourier_terms[stype])
        basis = torch.zeros(horizon, 2 * n, dtype=torch.float32)
        if stype in active:
            if granularity not in FREQ_DAYS:
                raise ValueError(f"Cannot build active Fourier basis for granularity={granularity}")
            p_steps = SEASONALITY_PERIODS[stype] / FREQ_DAYS[granularity]
            for k in range(n):
                basis[:, 2 * k] = torch.sin(2 * math.pi * (k + 1) * t / p_steps)
                basis[:, 2 * k + 1] = torch.cos(2 * math.pi * (k + 1) * t / p_steps)
        save_data[f"{stype}_basis"] = basis
    return save_data


def save_raw_futures(npz_data: dict[str, Any], out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = npz_data["meta"]
    futures = torch.from_numpy(npz_data["future_n"]).float()
    raw = {
        "futures_n": futures,
        "valid_mask": torch.ones(futures.shape[0], dtype=torch.bool),
        "context_len": int(meta["context_len"]),
        "horizon": int(meta["horizon"]),
    }
    torch.save(raw, out_path)
    return raw


def save_seasonal_coefficients(
    npz_data: dict[str, Any],
    active_types: list[str],
    n_fourier_terms: dict[str, int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = npz_data["meta"]
    sigmas = npz_data["sigma"].astype(np.float32)
    denom = np.where(sigmas >= REVIN_TOL, sigmas, 1.0).astype(np.float32)
    samples = meta.get("samples", [])
    if len(samples) != len(denom):
        raise ValueError(
            f"Coefficient sample count mismatch: samples={len(samples)} sigma={len(denom)}"
        )

    tensors: dict[str, torch.Tensor] = {}
    active = set(active_types)
    for family in ["daily", "weekly", "yearly"]:
        n_terms = int(n_fourier_terms[family])
        values = np.zeros((len(samples), 2 * n_terms), dtype=np.float32)
        if family in active:
            for row_idx, sample in enumerate(samples):
                coeffs = sample["coeffs"].get(family, [])
                for coef in coeffs:
                    k_idx = int(coef["k"]) - 1
                    if k_idx >= n_terms:
                        raise ValueError(
                            f"{family} coefficient k={k_idx + 1} exceeds cache width {n_terms}"
                        )
                    values[row_idx, 2 * k_idx] = float(coef["a"]) / float(denom[row_idx])
                    values[row_idx, 2 * k_idx + 1] = float(coef["b"]) / float(denom[row_idx])
        tensors[f"{family}_coefficients"] = torch.from_numpy(values)

    tensors["mask"] = {family: family in active for family in ["daily", "weekly", "yearly"]}
    tensors["n_fourier_terms"] = {key: int(value) for key, value in n_fourier_terms.items()}
    tensors["horizon"] = int(meta["horizon"])
    tensors["granularity"] = str(meta["granularity"])
    torch.save(tensors, out_path)


def encode_batch(
    contexts: torch.Tensor,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_n, context_len = contexts.shape
    if context_len % PATCH_LEN != 0:
        raise ValueError(f"context_len={context_len} must be divisible by patch_len={PATCH_LEN}")

    contexts = contexts.to(device, non_blocking=True)
    masks = torch.zeros_like(contexts, dtype=torch.bool, device=device)
    patched_inputs = contexts.reshape(batch_n, -1, PATCH_LEN)
    patched_masks = masks.reshape(batch_n, -1, PATCH_LEN)
    n = torch.zeros(batch_n, device=device)
    mu = torch.zeros(batch_n, device=device)
    sigma = torch.zeros(batch_n, device=device)
    patch_mu = []
    patch_sigma = []

    for patch_idx in range(context_len // PATCH_LEN):
        (n, mu, sigma), _ = update_stats_fn(
            n,
            mu,
            sigma,
            patched_inputs[:, patch_idx],
            patched_masks[:, patch_idx],
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)

    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)
    normed_inputs = revin_fn(patched_inputs, context_mu, context_sigma, reverse=False)

    with torch.no_grad():
        if hasattr(backbone, "_encode"):
            encoded = backbone._encode(normed_inputs, patched_masks)
            output_embeddings = encoded[1] if isinstance(encoded, tuple) else encoded
        else:
            (_, output_embeddings, _, _), _ = backbone(normed_inputs, patched_masks)

    return (
        output_embeddings[:, -1, :].detach().float().cpu(),
        context_mu[:, -1:].detach().float().cpu(),
        context_sigma[:, -1:].detach().float().cpu(),
    )


def save_backbone_cache(
    npz_data: dict[str, Any],
    granularity: str,
    out_path: Path,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signal = npz_data["signal"]
    meta = npz_data["meta"]
    context_len = int(meta["context_len"])
    horizon = int(meta["horizon"])
    contexts = torch.from_numpy(signal[:, :context_len]).float()
    n_samples = contexts.shape[0]

    embeddings = torch.empty((n_samples, EMBED_DIM), dtype=torch.float32)
    mu = torch.empty((n_samples, 1), dtype=torch.float32)
    sigma = torch.empty((n_samples, 1), dtype=torch.float32)

    write_idx = 0
    for start in tqdm(range(0, n_samples, batch_size), desc=f"backbone {out_path.parent.name}"):
        end = min(start + batch_size, n_samples)
        emb_b, mu_b, sigma_b = encode_batch(
            contexts[start:end],
            backbone,
            revin_fn,
            update_stats_fn,
            device,
        )
        embeddings[write_idx : write_idx + emb_b.shape[0]] = emb_b
        mu[write_idx : write_idx + mu_b.shape[0]] = mu_b
        sigma[write_idx : write_idx + sigma_b.shape[0]] = sigma_b
        write_idx += emb_b.shape[0]

    cache = {
        "embeddings": embeddings,
        "mu": mu,
        "sigma": sigma,
        "win_starts": torch.zeros(n_samples, dtype=torch.long),
        "col_ids": [f"synth_{i}" for i in range(n_samples)],
        "context_len": context_len,
        "horizon": horizon,
        "stride": 1,
        "frequency": granularity,
    }
    torch.save(cache, out_path)
    return cache


def validate_cache(
    ds_cache_dir: Path,
    backbone_path: Path | None,
    horizon: int,
    active_types: list[str],
) -> None:
    raw = torch.load(ds_cache_dir / f"raw_futures_h{horizon}.pt", map_location="cpu", weights_only=False)
    basis = torch.load(ds_cache_dir / f"fourier_basis_h{horizon}.pt", map_location="cpu", weights_only=False)
    if not bool(raw["valid_mask"].all().item()):
        raise AssertionError(f"valid_mask contains False: {ds_cache_dir}")

    active = set(active_types)
    for stype in ["daily", "weekly", "yearly"]:
        tensor = basis[f"{stype}_basis"]
        nonzero = bool(torch.count_nonzero(tensor).item() > 0)
        if stype in active and not nonzero:
            raise AssertionError(f"active basis is zero: {ds_cache_dir} {stype}")
        if stype not in active and nonzero:
            raise AssertionError(f"inactive basis is non-zero: {ds_cache_dir} {stype}")
        if bool(basis["mask"][stype]) != (stype in active):
            raise AssertionError(f"mask mismatch: {ds_cache_dir} {stype}")

    if backbone_path is not None and backbone_path.exists():
        backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
        shape = tuple(backbone["embeddings"].shape)
        if len(shape) != 2 or shape[1] != EMBED_DIM:
            raise AssertionError(f"embedding shape mismatch: {backbone_path}: {shape}")


def print_manual_loader_check(
    ds_cache_dir: Path,
    backbone_path: Path,
    horizon: int,
    n_fourier_terms: dict[str, int],
) -> None:
    print("\nManual MonashFuncDecDataset loader check:")
    print("python - <<'PY'")
    print("import sys")
    print("sys.path.insert(0, '/workspace/4.28basis/basis_dec/experiment/func_dec_np_trend')")
    print("from train import MonashFuncDecDataset")
    print("payload = MonashFuncDecDataset._load_dataset_cache(")
    print(f"    dataset_name='{ds_cache_dir.name}',")
    print(f"    ds_cache_dir='{ds_cache_dir}',")
    print(f"    backbone_path='{backbone_path}',")
    print(f"    horizon={horizon},")
    print(f"    n_fourier_terms={n_fourier_terms!r},")
    print(")")
    print("print(payload['embeddings'].shape, payload['futures_n'].shape)")
    print("PY\n")


def requested_granularities(category: str, level: str, cli_values: list[str] | None) -> list[str]:
    if category == "trend":
        return ["none"]
    allowed = SEASONAL_GRANULARITIES[level]
    if not cli_values:
        return allowed
    values = [value for value in cli_values if value in allowed]
    if not values:
        raise ValueError(f"No requested granularity is valid for {level}; allowed={allowed}")
    return values


def main() -> None:
    args = parse_args()
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

    first_cache: tuple[Path, Path, int] | None = None
    for level in args.levels:
        for horizon in args.horizons:
            for granularity in requested_granularities(args.category, level, args.granularities):
                pattern = f"{level}_{granularity}_seed{args.seed}_c*_h{horizon}"
                candidates = sorted((args.input_root / args.category).glob(pattern + ".npz"))
                if len(candidates) != 1:
                    if args.category == "trend":
                        candidates = sorted(
                            (args.input_root / "trend").glob(
                                f"{level}_seed{args.seed}_c*_h{horizon}.npz"
                            )
                        )
                    if len(candidates) != 1:
                        raise FileNotFoundError(
                            f"Expected one input for {args.category} {level} {granularity} "
                            f"seed={args.seed} horizon={horizon}, found {len(candidates)}"
                        )
                npz_path = candidates[0]
                npz_data = load_npz(npz_path)
                meta = npz_data["meta"]
                context_len = int(meta["context_len"])
                active_types = [] if args.category == "trend" else list(meta["active_types"])
                ds_dir = cache_dir(
                    args.output_root,
                    args.category,
                    level,
                    granularity,
                    args.seed,
                    context_len,
                    horizon,
                )
                ds_dir.mkdir(parents=True, exist_ok=True)

                backbone_path = ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt"
                basis_path = ds_dir / f"fourier_basis_h{horizon}.pt"
                raw_path = ds_dir / f"raw_futures_h{horizon}.pt"
                coeff_path = ds_dir / f"seasonal_coefficients_h{horizon}.pt"

                coeff_exists = args.category != "seasonal" or coeff_path.exists()
                if (
                    args.skip_existing
                    and backbone_path.exists()
                    and basis_path.exists()
                    and raw_path.exists()
                    and coeff_exists
                ):
                    print(f"skip_existing={ds_dir}")
                    validate_cache(ds_dir, backbone_path, horizon, active_types)
                    continue

                torch.save(
                    build_fourier_basis(
                        horizon,
                        granularity,
                        active_types,
                        n_fourier_terms=n_fourier_terms,
                        context_len=context_len,
                    ),
                    basis_path,
                )
                save_raw_futures(npz_data, raw_path)
                if args.category == "seasonal":
                    save_seasonal_coefficients(
                        npz_data,
                        active_types,
                        n_fourier_terms,
                        coeff_path,
                    )
                if args.metadata_only:
                    backbone_path_or_none = None
                    print(f"metadata_only: skipped backbone={backbone_path}")
                else:
                    assert device is not None
                    assert backbone is not None and revin_fn is not None and update_stats_fn is not None
                    save_backbone_cache(
                        npz_data,
                        granularity,
                        backbone_path,
                        args.batch_size,
                        backbone,
                        revin_fn,
                        update_stats_fn,
                        device,
                    )
                    backbone_path_or_none = backbone_path

                validate_cache(ds_dir, backbone_path_or_none, horizon, active_types)
                print(f"saved_cache={ds_dir}")
                if first_cache is None:
                    first_cache = (ds_dir, backbone_path, horizon)

    if first_cache is not None:
        print_manual_loader_check(*first_cache, n_fourier_terms)


if __name__ == "__main__":
    main()
