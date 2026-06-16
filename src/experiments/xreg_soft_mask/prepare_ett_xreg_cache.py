#!/usr/bin/env python3
"""Build real-eval caches for ETT datasets with wide raw covariates.

The output matches the existing ``real_eval_lot_ett`` ETT cache layout:

  <real_root>/energy/<dataset>/<dataset>.csv
  <real_root>/energy/<dataset>/cache/backbone_emb_c512_stride512.pt
  <real_root>/energy/<dataset>/cache/futures_c512_<freq>_h<horizon>_ett.pt
  <real_root>/energy/<dataset>/cache/raw.parquet
  <real_root>/energy/<dataset>/cache/sample_indices.pt

Each target sample is one numeric column at one rolling window start. Other
numeric columns remain available in ``raw.parquet`` for XReg covariates.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


MODEL_ID = "google/timesfm-2.5-200m-pytorch"
EMBED_DIM = 1280
PATCH_LEN = 32
REVIN_TOL = 1e-6
DEFAULT_DATASETS = ["ETTh2", "ETTm1", "ETTm2"]
DEFAULT_HORIZONS = [96, 192, 336, 720]
DEFAULT_SOURCE_ROOT = Path("/workspace/data/external/ett")
DEFAULT_REAL_ROOT = Path("/workspace/data/real_eval_lot_ett")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--real_root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_backbone", action="store_true",
                        help="Only write raw/futures/manifest; requires an existing compatible backbone cache.")
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
        raise ValueError(f"Unexpected TimesFM shape: patch_len={backbone.p}, embed_dim={backbone.md}")
    return backbone, revin, update_running_stats


def source_csv(source_root: Path, dataset_name: str) -> Path:
    candidates = [
        source_root / dataset_name / f"{dataset_name}.csv",
        source_root / f"{dataset_name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No CSV found for {dataset_name}; checked: {candidates}")


def read_ett_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"No numeric columns in {path}")
    if df[numeric_cols].isna().any().any():
        df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both").ffill().bfill()
    return df


def infer_freq(dataset_name: str) -> str:
    return "H" if dataset_name.startswith("ETTh") else "15_minutes"


def build_samples(df: pd.DataFrame, context_len: int, max_horizon: int, stride: int) -> tuple[list[str], list[int]]:
    numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    last_start = len(df) - int(context_len) - int(max_horizon)
    if last_start < 0:
        raise ValueError(
            f"Dataset too short: rows={len(df)} context_len={context_len} max_horizon={max_horizon}"
        )
    starts = list(range(0, last_start + 1, int(stride)))
    col_ids: list[str] = []
    win_starts: list[int] = []
    for start in starts:
        for col in numeric_cols:
            col_ids.append(str(col))
            win_starts.append(int(start))
    return col_ids, win_starts


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


def context_matrix(df: pd.DataFrame, col_ids: list[str], win_starts: list[int], context_len: int) -> np.ndarray:
    values = []
    for col, start in zip(col_ids, win_starts, strict=True):
        arr = df[col].iloc[int(start): int(start) + int(context_len)].to_numpy(dtype=np.float32)
        if len(arr) != context_len:
            raise ValueError(f"Bad context length col={col} start={start} len={len(arr)}")
        values.append(arr)
    return np.stack(values, axis=0)


def save_backbone_cache(
    df: pd.DataFrame,
    out_path: Path,
    col_ids: list[str],
    win_starts: list[int],
    context_len: int,
    stride: int,
    freq: str,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> dict[str, Any]:
    contexts = torch.from_numpy(context_matrix(df, col_ids, win_starts, context_len)).float()
    n_samples = int(contexts.shape[0])
    embeddings = torch.empty((n_samples, EMBED_DIM), dtype=torch.float32)
    mu = torch.empty((n_samples, 1), dtype=torch.float32)
    sigma = torch.empty((n_samples, 1), dtype=torch.float32)

    write_idx = 0
    for start in tqdm(range(0, n_samples, batch_size), desc=f"backbone {out_path.parent.parent.name}"):
        end = min(start + batch_size, n_samples)
        emb_b, mu_b, sigma_b = encode_batch(
            contexts[start:end],
            backbone,
            revin_fn,
            update_stats_fn,
            device,
        )
        embeddings[write_idx: write_idx + emb_b.shape[0]] = emb_b
        mu[write_idx: write_idx + mu_b.shape[0]] = mu_b
        sigma[write_idx: write_idx + sigma_b.shape[0]] = sigma_b
        write_idx += int(emb_b.shape[0])

    payload = {
        "embeddings": embeddings,
        "mu": mu,
        "sigma": sigma,
        "win_starts": torch.tensor(win_starts, dtype=torch.long),
        "col_ids": list(col_ids),
        "freq": str(freq),
        "frequency": str(freq),
        "context_len": int(context_len),
        "stride": int(stride),
        "backbone": MODEL_ID,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload


def save_futures(
    df: pd.DataFrame,
    backbone: dict[str, Any],
    out_path: Path,
    horizon: int,
) -> None:
    col_ids = list(backbone["col_ids"])
    win_starts = backbone["win_starts"].cpu().tolist()
    context_len = int(backbone["context_len"])
    mu = backbone["mu"].float()
    sigma = backbone["sigma"].float()
    denom = torch.where(sigma >= REVIN_TOL, sigma, torch.ones_like(sigma))

    futures = []
    for col, start in zip(col_ids, win_starts, strict=True):
        begin = int(start) + context_len
        end = begin + int(horizon)
        arr = df[col].iloc[begin:end].to_numpy(dtype=np.float32)
        if len(arr) != horizon:
            raise ValueError(f"Bad future length col={col} start={start} horizon={horizon} len={len(arr)}")
        futures.append(arr)

    future_raw = torch.from_numpy(np.stack(futures, axis=0)).float()
    future_n = ((future_raw - mu) / denom).float()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "futures": future_n,
            "futures_n": future_n,
            "valid_mask": torch.ones(future_n.shape[0], dtype=torch.bool),
            "horizon": int(horizon),
            "context_len": context_len,
            "freq": str(backbone.get("freq") or backbone.get("frequency") or ""),
            "frequency": str(backbone.get("freq") or backbone.get("frequency") or ""),
        },
        out_path,
    )


def manifest_files(
    dataset_name: str,
    freq: str,
    horizons: list[int],
    context_len: int,
    stride: int,
) -> list[str]:
    files = [
        f"{dataset_name}.csv",
        f"cache/backbone_emb_c{int(context_len)}_stride{int(stride)}.pt",
    ]
    files.extend(f"cache/futures_c{int(context_len)}_{freq}_h{int(h)}_ett.pt" for h in sorted(horizons))
    files.extend(["cache/raw.parquet", "cache/sample_indices.pt"])
    return files


def update_manifest(
    real_root: Path,
    dataset_entries: list[dict[str, Any]],
    horizons: list[int],
    context_len: int,
) -> None:
    manifest_path = real_root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
    else:
        manifest = {
            "output_root": str(real_root),
            "max_windows": None,
            "seed": None,
            "min_context_sigma": 0.001,
            "horizons": sorted(horizons),
            "datasets": [],
        }

    if not isinstance(manifest, dict) or "datasets" not in manifest:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")
    manifest["output_root"] = str(real_root)
    manifest["horizons"] = sorted({*map(int, manifest.get("horizons", [])), *map(int, horizons)})
    existing = [
        item for item in manifest["datasets"]
        if str(item.get("dataset") or item.get("name")) not in {entry["dataset"] for entry in dataset_entries}
    ]
    manifest["datasets"] = existing + dataset_entries
    manifest["context_len"] = int(context_len)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_dataset(args: argparse.Namespace, dataset_name: str, model_pack) -> dict[str, Any]:
    csv_path = source_csv(args.source_root, dataset_name)
    df = read_ett_csv(csv_path)
    freq = infer_freq(dataset_name)
    max_horizon = max(int(h) for h in args.horizons)
    col_ids, win_starts = build_samples(df, args.context_len, max_horizon, args.stride)

    ds_root = args.real_root / "energy" / dataset_name
    cache_dir = ds_root / "cache"
    if args.overwrite and ds_root.exists():
        shutil.rmtree(ds_root)
    ds_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, ds_root / f"{dataset_name}.csv")
    df.to_parquet(cache_dir / "raw.parquet", index=False)

    backbone_path = cache_dir / f"backbone_emb_c{args.context_len}_stride{args.stride}.pt"
    if backbone_path.exists() and not args.overwrite:
        backbone_payload = torch.load(backbone_path, map_location="cpu", weights_only=False)
    else:
        if args.skip_backbone:
            raise FileNotFoundError(f"Missing backbone cache with --skip_backbone: {backbone_path}")
        backbone, revin_fn, update_stats_fn, device = model_pack
        backbone_payload = save_backbone_cache(
            df,
            backbone_path,
            col_ids,
            win_starts,
            args.context_len,
            args.stride,
            freq,
            args.batch_size,
            backbone,
            revin_fn,
            update_stats_fn,
            device,
        )

    torch.save(
        {
            "source_indices": torch.arange(len(col_ids), dtype=torch.long),
            "source_backbone": str(backbone_path),
            "dataset": dataset_name,
        },
        cache_dir / "sample_indices.pt",
    )
    for horizon in args.horizons:
        future_path = cache_dir / f"futures_c{args.context_len}_{freq}_h{int(horizon)}_ett.pt"
        if future_path.exists() and not args.overwrite:
            continue
        save_futures(df, backbone_payload, future_path, int(horizon))

    return {
        "domain": "energy",
        "dataset": dataset_name,
        "kind": "ett",
        "source": "ett_cache",
        "source_dir": str(csv_path.parent),
        "output_dir": str(ds_root),
        "context_len": int(args.context_len),
        "freq": freq,
        "source_windows": len(col_ids),
        "usable_windows": len(col_ids),
        "sampled_windows": len(col_ids),
        "horizons": sorted(int(h) for h in args.horizons),
        "files": manifest_files(
            dataset_name,
            freq,
            [int(h) for h in args.horizons],
            int(args.context_len),
            int(args.stride),
        ),
    }


def main() -> None:
    args = parse_args()
    if args.context_len % PATCH_LEN != 0:
        raise ValueError(f"--context_len must be divisible by {PATCH_LEN}")
    if args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("--stride and --batch_size must be positive")

    model_pack = None
    if not args.skip_backbone:
        device = resolve_device(args.device)
        backbone, revin_fn, update_stats_fn = load_backbone(device, args.hf_cache_dir)
        model_pack = (backbone, revin_fn, update_stats_fn, device)

    entries = []
    for dataset_name in args.datasets:
        entries.append(build_dataset(args, dataset_name, model_pack))
        print(f"[saved] {dataset_name}: windows={entries[-1]['sampled_windows']}", flush=True)
    update_manifest(args.real_root, entries, [int(h) for h in args.horizons], int(args.context_len))
    print(f"[manifest] {args.real_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
