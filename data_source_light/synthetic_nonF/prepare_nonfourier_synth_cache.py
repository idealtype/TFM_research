#!/usr/bin/env python3
"""Generate and cache non-Fourier seasonal synthetic evaluation datasets."""

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

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic_nonF"
SRC_DIR = Path(os.environ.get("PROJECT_ROOT", "/workspace")) / "4.28basis" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nonfourier_generator import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    GRANULARITIES,
    NONFOURIER_MODELS,
    VALID_HORIZONS,
    generate_nonfourier_dataset,
    load_config,
    save_nonfourier_dataset,
)


MODEL_ID = "google/timesfm-2.5-200m-pytorch"
PATCH_LEN = 32
EMBED_DIM = 1280
REVIN_TOL = 1e-6
N_FOURIER_TERMS = {"daily": 10, "weekly": 4, "yearly": 8}
SEASONALITY_PERIODS = {"daily": 1.0, "weekly": 7.0, "yearly": 365.25}
FREQ_DAYS = {"hourly": 1.0 / 24.0, "daily": 1.0, "weekly": 7.0}
DEFAULT_CACHE_ROOT = ROOT / "synth_eval_nonfourier" / "stage1_S_nonfourier_cache_10_4_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--models", nargs="+", default=NONFOURIER_MODELS, choices=NONFOURIER_MODELS)
    parser.add_argument("--granularities", nargs="+", default=list(GRANULARITIES), choices=list(GRANULARITIES))
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--n_fourier_daily", type=int, default=N_FOURIER_TERMS["daily"])
    parser.add_argument("--n_fourier_weekly", type=int, default=N_FOURIER_TERMS["weekly"])
    parser.add_argument("--n_fourier_yearly", type=int, default=N_FOURIER_TERMS["yearly"])
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Write raw/Fourier/component cache files without TimesFM backbone embeddings.",
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


def input_path(data_root: Path, model: str, granularity: str, seed: int, context_len: int, horizon: int) -> Path:
    return data_root / model / "seasonal" / f"{granularity}_seed{seed}_c{context_len}_h{horizon}.npz"


def cache_dir(cache_root: Path, model: str, granularity: str, seed: int, context_len: int, horizon: int) -> Path:
    return cache_root / model / "seasonal" / f"{granularity}_seed{seed}_c{context_len}_h{horizon}"


def load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return {
            "signal": data["signal"].astype(np.float32),
            "future_n": data["future_n"].astype(np.float32),
            "gt_seasonal_n": data["gt_seasonal_n"].astype(np.float32),
            "mu": data["mu"].astype(np.float32),
            "sigma": data["sigma"].astype(np.float32),
            "meta": meta,
        }


def build_fourier_basis(
    horizon: int,
    granularity: str,
    active_types: list[str],
    n_fourier_terms: dict[str, int],
    context_len: int,
) -> dict[str, Any]:
    t = torch.arange(int(context_len), int(context_len) + int(horizon), dtype=torch.float32)
    active = set(active_types)
    save_data: dict[str, Any] = {
        "mask": {stype: stype in active for stype in ["daily", "weekly", "yearly"]},
        "freq": granularity,
        "horizon": int(horizon),
        "context_len": int(context_len),
        "n_fourier_terms": {key: int(value) for key, value in n_fourier_terms.items()},
    }
    for stype in ["daily", "weekly", "yearly"]:
        n_terms = int(n_fourier_terms[stype])
        basis = torch.zeros(int(horizon), 2 * n_terms, dtype=torch.float32)
        if stype in active:
            p_steps = SEASONALITY_PERIODS[stype] / FREQ_DAYS[granularity]
            for k in range(n_terms):
                basis[:, 2 * k] = torch.sin(2.0 * math.pi * (k + 1) * t / p_steps)
                basis[:, 2 * k + 1] = torch.cos(2.0 * math.pi * (k + 1) * t / p_steps)
        save_data[f"{stype}_basis"] = basis
    return save_data


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
    seasonal = torch.from_numpy(npz_data["gt_seasonal_n"]).float()
    zeros = torch.zeros_like(seasonal)
    torch.save(
        {
            "trend_n": zeros,
            "seasonal_n": seasonal,
            "residual_n": zeros,
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
    tensors["note"] = "Non-Fourier generator has no ground-truth Fourier coefficients; zeros are placeholders."
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
            n, mu, sigma,
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
    out_path: Path,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> None:
    from tqdm import tqdm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    signal = npz_data["signal"]
    meta = npz_data["meta"]
    context_len = int(meta["context_len"])
    horizon = int(meta["horizon"])
    contexts = torch.from_numpy(signal[:, :context_len]).float()
    n_samples = int(contexts.shape[0])
    embeddings = torch.empty((n_samples, EMBED_DIM), dtype=torch.float32)
    mu = torch.empty((n_samples, 1), dtype=torch.float32)
    sigma = torch.empty((n_samples, 1), dtype=torch.float32)

    write_idx = 0
    for start in tqdm(range(0, n_samples, batch_size), desc=f"backbone {out_path.parent.name}"):
        end = min(start + batch_size, n_samples)
        emb_b, mu_b, sigma_b = encode_batch(
            contexts[start:end], backbone, revin_fn, update_stats_fn, device,
        )
        embeddings[write_idx : write_idx + emb_b.shape[0]] = emb_b
        mu[write_idx : write_idx + mu_b.shape[0]] = mu_b
        sigma[write_idx : write_idx + sigma_b.shape[0]] = sigma_b
        write_idx += emb_b.shape[0]

    torch.save(
        {
            "embeddings": embeddings,
            "mu": mu,
            "sigma": sigma,
            "win_starts": torch.zeros(n_samples, dtype=torch.long),
            "col_ids": [f"nonfourier_{meta['model']}_{i}" for i in range(n_samples)],
            "context_len": context_len,
            "horizon": horizon,
            "stride": 1,
            "frequency": str(meta["granularity"]),
            "generator_family": "nonfourier",
            "model": str(meta["model"]),
        },
        out_path,
    )


def validate_cache(ds_dir: Path, horizon: int, active_types: list[str], backbone_path: Path | None) -> None:
    raw = torch.load(ds_dir / f"raw_futures_h{horizon}.pt", map_location="cpu", weights_only=False)
    basis = torch.load(ds_dir / f"fourier_basis_h{horizon}.pt", map_location="cpu", weights_only=False)
    components = torch.load(ds_dir / f"component_targets_h{horizon}.pt", map_location="cpu", weights_only=False)
    if not bool(raw["valid_mask"].all().item()):
        raise AssertionError(f"valid_mask contains False: {ds_dir}")
    if raw["futures_n"].shape != components["seasonal_n"].shape:
        raise AssertionError(f"future/component shape mismatch: {ds_dir}")
    active = set(active_types)
    for stype in ["daily", "weekly", "yearly"]:
        tensor = basis[f"{stype}_basis"]
        nonzero = bool(torch.count_nonzero(tensor).item() > 0)
        if stype in active and not nonzero:
            raise AssertionError(f"active basis is zero: {ds_dir} {stype}")
        if stype not in active and nonzero:
            raise AssertionError(f"inactive basis is non-zero: {ds_dir} {stype}")
    if backbone_path is not None and backbone_path.exists():
        backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
        shape = tuple(backbone["embeddings"].shape)
        if len(shape) != 2 or shape[1] != EMBED_DIM:
            raise AssertionError(f"embedding shape mismatch: {backbone_path}: {shape}")


def generate_data(args: argparse.Namespace, cfg: dict[str, Any], horizons: list[int], context_len: int, seed: int) -> None:
    args.data_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config),
        "data_root": str(args.data_root),
        "models": list(args.models),
        "granularities": list(args.granularities),
        "horizons": list(horizons),
        "context_len": int(context_len),
        "n_samples": int(args.n_samples),
        "seed": int(seed),
        "note": "Evaluation-only non-Fourier seasonal synthetic data.",
    }
    (args.data_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for model in args.models:
        for granularity in args.granularities:
            for horizon in horizons:
                path = input_path(args.data_root, model, granularity, seed, context_len, int(horizon))
                if args.skip_existing and path.exists():
                    print(f"skip_existing_data={path}", flush=True)
                    continue
                dataset = generate_nonfourier_dataset(
                    model=model,
                    granularity=granularity,
                    horizon=int(horizon),
                    context_len=int(context_len),
                    n_samples=int(args.n_samples),
                    seed=int(seed),
                )
                saved_path = save_nonfourier_dataset(dataset, args.data_root)
                print(f"saved_data={saved_path}", flush=True)


def build_cache(args: argparse.Namespace, horizons: list[int], context_len: int, seed: int) -> None:
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
        for granularity in args.granularities:
            for horizon in horizons:
                npz_path = input_path(args.data_root, model, granularity, seed, context_len, int(horizon))
                npz_data = load_npz(npz_path)
                meta = npz_data["meta"]
                active_types = list(meta["active_types"])
                ds_dir = cache_dir(args.cache_root, model, granularity, seed, context_len, int(horizon))
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
                    validate_cache(ds_dir, int(horizon), active_types, None if args.metadata_only else backbone_path)
                    print(f"skip_existing_cache={ds_dir}", flush=True)
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
                validate_cache(ds_dir, int(horizon), active_types, backbone_path_or_none)
                print(f"saved_cache={ds_dir}", flush=True)


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
    if not args.skip_generation:
        generate_data(args, cfg, horizons, context_len, seed)
    if not args.skip_cache:
        build_cache(args, horizons, context_len, seed)


if __name__ == "__main__":
    main()
