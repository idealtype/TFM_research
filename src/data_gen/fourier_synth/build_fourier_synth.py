#!/usr/bin/env python3
"""Generate and cache the unified Fourier synthetic pool.

This replaces the split legacy flow:
  - old 3-family S1-S10 complex Fourier data
  - later fine-mask SM data with monthly support

Policy:
  - one 4-family decoder profile: daily=10, weekly=4, monthly=2, yearly=8
  - all canonical granularities that activate at least one hard-mask harmonic
  - all per-family active-order count cases under the hard-mask rule
  - random harmonic subsets and coefficients per sample for numerical diversity
  - train/eval raw npz plus TimesFM embedding caches in one command
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
EXPERIMENTS_ROOT = SRC_ROOT / "experiments"
SYNTH_SOURCE_ROOT = PROJECT_ROOT / "data_source_light" / "synthetic"

sys.path.insert(0, str(EXPERIMENTS_ROOT))
sys.path.insert(0, str(SYNTH_SOURCE_ROOT))

from loader_utils import resolve_data_path  # noqa: E402
from synth_generator import (  # noqa: E402
    EPS_SIGMA,
    LEVELS,
    VALID_HORIZONS,
    generate_trend_dataset,
    load_config,
)


DATA_ROOT = Path(os.environ.get("DATA_ROOT", os.environ.get("VESSL_DATA_ROOT", "/workspace/data")))
SYNTH_ROOT = DATA_ROOT / "synthetic"

DEFAULT_CONFIG = SRC_ROOT / "data_gen" / "fourier_synth" / "fourier_synth_config.yaml"
DEFAULT_TRAIN_RAW = SYNTH_ROOT / "func_dec_syn_cent_fourier_all_train"
DEFAULT_EVAL_RAW = SYNTH_ROOT / "func_dec_syn_cent_fourier_all_eval"
DEFAULT_TRAIN_CACHE = SYNTH_ROOT / "func_dec_syn_cent_fourier_all_train_cache_10_4_2_8"
DEFAULT_EVAL_CACHE = SYNTH_ROOT / "func_dec_syn_cent_fourier_all_eval_cache_10_4_2_8"

PERIODS = {"daily": 1.0, "weekly": 7.0, "monthly": 30.4375, "yearly": 365.25}
K_MAX = {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8}
FAMILIES = ["daily", "weekly", "monthly", "yearly"]

FREQ_DAYS = {
    "5_minutes": 1 / 288,
    "10_minutes": 1 / 144,
    "15_minutes": 1 / 96,
    "half_hourly": 1 / 48,
    "hourly": 1 / 24,
    "daily": 1.0,
    "weekly": 7.0,
    "monthly": 30.4375,
}
DEFAULT_GRANULARITIES = list(FREQ_DAYS.keys())
COMPOSITIONS = ["A1", "A2", "A3"]
PATCH_LEN = 32
SPLIT_SEED_OFFSET = {"train": 0, "eval": 99991}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train_raw_root", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument("--eval_raw_root", type=Path, default=DEFAULT_EVAL_RAW)
    parser.add_argument("--train_cache_root", type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--eval_cache_root", type=Path, default=DEFAULT_EVAL_CACHE)
    parser.add_argument("--splits", nargs="+", default=["train", "eval"], choices=["train", "eval"])
    parser.add_argument("--granularities", nargs="+", default=DEFAULT_GRANULARITIES,
                        choices=DEFAULT_GRANULARITIES)
    parser.add_argument("--trend_levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--compositions", nargs="+", default=COMPOSITIONS, choices=COMPOSITIONS)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--amplitude_low", type=float, default=0.5)
    parser.add_argument("--amplitude_high", type=float, default=1.5)
    parser.add_argument("--coefficient_low", type=float, default=0.3)
    parser.add_argument("--coefficient_high", type=float, default=1.7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--raw_only", action="store_true")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Build basis/targets/coefficients but skip TimesFM embeddings.")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def load_generation_config(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_config(path)
    fallback = SYNTH_SOURCE_ROOT / "synth_config.yaml"
    if fallback.exists():
        return load_config(fallback)
    raise FileNotFoundError(f"Missing config: {path} and fallback {fallback}")


def seed_for(base_seed: int, *parts: Any) -> int:
    value = int(base_seed)
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % (2**32 - 1)
    return value


def active_harmonics(granularity: str, context_len: int) -> dict[str, list[int]]:
    fd = FREQ_DAYS[granularity]
    context_span = float(context_len) * fd
    active: dict[str, list[int]] = {}
    for family, period in PERIODS.items():
        ks = []
        for k in range(1, K_MAX[family] + 1):
            harmonic_period = period / k
            if fd < harmonic_period and context_span >= harmonic_period:
                ks.append(k)
        active[family] = ks
    return active


def enumerate_count_cases(active: dict[str, list[int]]) -> list[dict[str, int]]:
    cases: list[dict[str, int]] = []

    def rec(idx: int, current: dict[str, int]) -> None:
        if idx == len(FAMILIES):
            if any(current.values()):
                cases.append(dict(current))
            return
        family = FAMILIES[idx]
        for count in range(0, len(active[family]) + 1):
            current[family] = count
            rec(idx + 1, current)

    rec(0, {})
    return cases


def build_fourier_basis(granularity: str, context_len: int, horizon: int) -> dict[str, torch.Tensor]:
    fd = FREQ_DAYS[granularity]
    context_span = float(context_len) * fd
    t = torch.arange(int(context_len), int(context_len) + int(horizon), dtype=torch.float32)
    out: dict[str, torch.Tensor] = {}
    for family, period in PERIODS.items():
        basis = torch.zeros(int(horizon), 2 * K_MAX[family], dtype=torch.float32)
        period_steps = period / fd
        for k in range(1, K_MAX[family] + 1):
            harmonic_period = period / k
            if fd < harmonic_period and context_span >= harmonic_period:
                basis[:, 2 * (k - 1)] = torch.sin(2 * math.pi * k * t / period_steps)
                basis[:, 2 * (k - 1) + 1] = torch.cos(2 * math.pi * k * t / period_steps)
        out[f"{family}_basis"] = basis
    return out


def seasonal_from_coeffs(
    t: np.ndarray,
    granularity: str,
    coeffs: dict[str, list[dict[str, float]]],
) -> np.ndarray:
    fd = FREQ_DAYS[granularity]
    y = np.zeros_like(t, dtype=np.float64)
    for family, family_coeffs in coeffs.items():
        period_steps = PERIODS[family] / fd
        for coef in family_coeffs:
            k = int(coef["k"])
            angle = 2.0 * np.pi * k * t / period_steps
            y += float(coef["a"]) * np.sin(angle) + float(coef["b"]) * np.cos(angle)
    return y


def sample_coefficients(
    active: dict[str, list[int]],
    case: dict[str, int],
    rng: np.random.Generator,
    coefficient_low: float,
    coefficient_high: float,
) -> dict[str, list[dict[str, float]]]:
    coeffs: dict[str, list[dict[str, float]]] = {}
    for family in FAMILIES:
        count = int(case[family])
        if count <= 0:
            continue
        choices = np.asarray(active[family], dtype=np.int64)
        selected = rng.choice(choices, size=count, replace=False)
        selected = sorted(int(k) for k in selected)
        coeffs[family] = []
        for k in selected:
            amp = float(rng.uniform(coefficient_low, coefficient_high)) / float(k)
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            coeffs[family].append({
                "k": int(k),
                "a": float(amp * np.cos(phase)),
                "b": float(amp * np.sin(phase)),
                "amplitude_k": float(amp),
            })
    return coeffs


def build_case_schedule(
    cases: list[dict[str, int]],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, int]]:
    if not cases:
        raise ValueError("No active Fourier cases are available")
    shuffled = [dict(case) for case in cases]
    rng.shuffle(shuffled)
    schedule = [dict(shuffled[i % len(shuffled)]) for i in range(n_samples)]
    rng.shuffle(schedule)
    return schedule


def scale_coefficients(
    coeffs: dict[str, list[dict[str, float]]],
    scale: float,
) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    for family, values in coeffs.items():
        out[family] = []
        for coef in values:
            out[family].append({
                "k": int(coef["k"]),
                "a": float(coef["a"]) * scale,
                "b": float(coef["b"]) * scale,
            })
    return out


def build_dataset(
    split: str,
    composition: str,
    trend_level: str,
    granularity: str,
    horizon: int,
    context_len: int,
    n_samples: int,
    seed: int,
    cfg: dict[str, Any],
    noise_scale: float,
    amplitude_low: float,
    amplitude_high: float,
    coefficient_low: float,
    coefficient_high: float,
) -> dict[str, Any]:
    active = active_harmonics(granularity, context_len)
    cases = enumerate_count_cases(active)
    if not cases:
        raise ValueError(f"No active hard-mask harmonics for granularity={granularity}")

    trend = generate_trend_dataset(
        level=trend_level,
        horizon=horizon,
        context_len=context_len,
        n_samples=n_samples,
        seed=seed_for(seed, split, composition, trend_level, granularity, horizon, "trend"),
        cfg=cfg,
    )
    rng = np.random.default_rng(seed_for(seed, split, composition, trend_level, granularity, horizon))
    schedule = build_case_schedule(cases, n_samples, rng)

    total_len = context_len + horizon
    t = np.arange(total_len, dtype=np.float64)
    seasonal_signal = np.empty((n_samples, total_len), dtype=np.float32)
    seasonal_meta: list[dict[str, Any]] = []
    for idx, case in enumerate(schedule):
        coeffs = sample_coefficients(active, case, rng, coefficient_low, coefficient_high)
        seasonal_signal[idx] = seasonal_from_coeffs(t, granularity, coeffs).astype(np.float32)
        seasonal_meta.append({"case": case, "coeffs": coeffs})

    seasonal_context = seasonal_signal[:, :context_len]
    seasonal_mu = np.mean(seasonal_context, axis=1).astype(np.float32)
    seasonal_sigma = np.std(seasonal_context, axis=1).astype(np.float32)
    seasonal_denom = np.where(seasonal_sigma >= EPS_SIGMA, seasonal_sigma, 1.0).astype(np.float32)

    trend_sigma = np.where(trend["sigma"] >= EPS_SIGMA, trend["sigma"], 1.0).astype(np.float32)
    trend_component = ((trend["signal"] - trend["mu"][:, None]) / trend_sigma[:, None]).astype(np.float32)

    seasonal_scale = np.ones(n_samples, dtype=np.float32)
    if composition == "A3":
        seasonal_scale = rng.uniform(amplitude_low, amplitude_high, size=n_samples).astype(np.float32)
    seasonal_component = (
        seasonal_signal / seasonal_denom[:, None] * seasonal_scale[:, None]
    ).astype(np.float32)

    residual_signal = np.zeros((n_samples, total_len), dtype=np.float32)
    if composition == "A2":
        base = trend_component + seasonal_component
        sample_std = np.std(base[:, :context_len], axis=1).astype(np.float32)
        noise_std = np.maximum(sample_std * float(noise_scale), 1e-3).astype(np.float32)
        residual_signal = rng.normal(0.0, noise_std[:, None], size=(n_samples, total_len)).astype(np.float32)

    signal = trend_component + seasonal_component + residual_signal
    context = signal[:, :context_len]
    mu = np.mean(context, axis=1).astype(np.float32)
    sigma = np.std(context, axis=1).astype(np.float32)
    denom = np.where(sigma >= EPS_SIGMA, sigma, 1.0).astype(np.float32)

    future_n = ((signal[:, context_len:] - mu[:, None]) / denom[:, None]).astype(np.float32)
    gt_trend_n = ((trend_component[:, context_len:] - mu[:, None]) / denom[:, None]).astype(np.float32)
    gt_seasonal_n = (seasonal_component[:, context_len:] / denom[:, None]).astype(np.float32)
    gt_residual_n = (residual_signal[:, context_len:] / denom[:, None]).astype(np.float32)

    samples = []
    for idx, item in enumerate(seasonal_meta):
        coef_scale = float(seasonal_scale[idx]) / (float(seasonal_denom[idx]) * float(denom[idx]))
        samples.append({
            "trend": trend["meta"]["samples"][idx],
            "seasonal_case": item["case"],
            "seasonal_coefficients_n": scale_coefficients(item["coeffs"], coef_scale),
            "seasonal_scale": float(seasonal_scale[idx]),
        })

    covered_cases = sorted({tuple(int(sample["seasonal_case"][f]) for f in FAMILIES) for sample in samples})
    meta = {
        "category": "complex",
        "generator": "fourier_all_v1",
        "split": split,
        "composition": composition,
        "trend_level": trend_level,
        "seasonal_level": "F1",
        "granularity": granularity,
        "active_harmonics": active,
        "case_families": FAMILIES,
        "n_cases": len(cases),
        "covered_cases": [list(case) for case in covered_cases],
        "horizon": int(horizon),
        "context_len": int(context_len),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "noise_scale": float(noise_scale) if composition == "A2" else 0.0,
        "amplitude_low": float(amplitude_low) if composition == "A3" else 1.0,
        "amplitude_high": float(amplitude_high) if composition == "A3" else 1.0,
        "coefficient_low": float(coefficient_low),
        "coefficient_high": float(coefficient_high),
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
        "active_types": [family for family, ks in active.items() if ks],
        "granularity": granularity,
        "meta": meta,
    }


def dataset_name(meta: dict[str, Any]) -> str:
    return (
        f"{meta['composition']}_{meta['trend_level']}_{meta['seasonal_level']}_"
        f"{meta['granularity']}_seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}"
    )


def save_npz(dataset: dict[str, Any], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    meta = dataset["meta"]
    path = output_root / "complex" / f"{dataset_name(meta)}.npz"
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


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return {
            "signal": data["signal"].astype(np.float32),
            "future_n": data["future_n"].astype(np.float32),
            "gt_trend_n": data["gt_trend_n"].astype(np.float32),
            "gt_seasonal_n": data["gt_seasonal_n"].astype(np.float32),
            "gt_residual_n": data["gt_residual_n"].astype(np.float32),
            "meta": meta,
        }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: requested {name} but CUDA is unavailable; using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(name)


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
    npz_data: dict[str, Any],
    out_path: Path,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> None:
    signal = torch.from_numpy(npz_data["signal"]).float()
    context_len = int(npz_data["meta"]["context_len"])
    context = signal[:, :context_len]
    if context_len % PATCH_LEN != 0:
        raise ValueError(f"context_len={context_len} must be divisible by PATCH_LEN={PATCH_LEN}")

    all_embs = []
    all_mu = []
    all_sigma = []
    for start in range(0, context.shape[0], batch_size):
        end = min(start + batch_size, context.shape[0])
        ctx = context[start:end].to(device)
        batch = ctx.shape[0]
        patches = ctx.reshape(batch, -1, PATCH_LEN)
        masks = torch.zeros_like(patches, dtype=torch.bool)

        n = torch.zeros(batch, device=device)
        mu = torch.zeros(batch, device=device)
        sigma = torch.zeros(batch, device=device)
        patch_mu = []
        patch_sigma = []
        for patch_idx in range(patches.shape[1]):
            (n, mu, sigma), _ = update_stats_fn(n, mu, sigma, patches[:, patch_idx], masks[:, patch_idx])
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.cat(all_embs, dim=0),
            "mu": torch.cat(all_mu, dim=0),
            "sigma": torch.cat(all_sigma, dim=0),
            "context_len": int(context_len),
            "granularity": str(npz_data["meta"]["granularity"]),
        },
        out_path,
    )


def save_targets(npz_data: dict[str, Any], ds_dir: Path) -> None:
    meta = npz_data["meta"]
    horizon = int(meta["horizon"])
    futures = torch.from_numpy(npz_data["future_n"]).float()
    torch.save(
        {
            "futures_n": futures,
            "valid_mask": torch.ones(futures.shape[0], dtype=torch.bool),
            "context_len": int(meta["context_len"]),
            "horizon": horizon,
        },
        ds_dir / f"raw_futures_h{horizon}.pt",
    )
    torch.save(
        {
            "trend_n": torch.from_numpy(npz_data["gt_trend_n"]).float(),
            "seasonal_n": torch.from_numpy(npz_data["gt_seasonal_n"]).float(),
            "residual_n": torch.from_numpy(npz_data["gt_residual_n"]).float(),
        },
        ds_dir / f"component_targets_h{horizon}.pt",
    )


def save_coefficients(npz_data: dict[str, Any], ds_dir: Path) -> None:
    meta = npz_data["meta"]
    horizon = int(meta["horizon"])
    samples = meta["samples"]
    n_samples = len(samples)
    tensors: dict[str, Any] = {}
    sample_masks: dict[str, torch.Tensor] = {}
    for family in FAMILIES:
        values = np.zeros((n_samples, 2 * K_MAX[family]), dtype=np.float32)
        active_mask = np.zeros(n_samples, dtype=np.bool_)
        for row_idx, sample in enumerate(samples):
            coeffs = sample["seasonal_coefficients_n"].get(family, [])
            if coeffs:
                active_mask[row_idx] = True
            for coef in coeffs:
                k_idx = int(coef["k"]) - 1
                if k_idx >= K_MAX[family]:
                    raise ValueError(f"{family} k={k_idx + 1} exceeds K_MAX={K_MAX[family]}")
                values[row_idx, 2 * k_idx] = float(coef["a"])
                values[row_idx, 2 * k_idx + 1] = float(coef["b"])
        tensors[f"{family}_coefficients"] = torch.from_numpy(values)
        sample_masks[family] = torch.from_numpy(active_mask)
    tensors["mask"] = {family: bool(sample_masks[family].any()) for family in FAMILIES}
    tensors["sample_mask"] = sample_masks
    tensors["n_fourier_terms"] = {family: int(K_MAX[family]) for family in FAMILIES}
    tensors["horizon"] = horizon
    tensors["granularity"] = str(meta["granularity"])
    torch.save(tensors, ds_dir / f"seasonal_coefficients_fine_mask_h{horizon}.pt")


def save_basis(npz_data: dict[str, Any], ds_dir: Path) -> None:
    meta = npz_data["meta"]
    horizon = int(meta["horizon"])
    basis = build_fourier_basis(str(meta["granularity"]), int(meta["context_len"]), horizon)
    payload = {
        "freq": str(meta["granularity"]),
        "granularity": str(meta["granularity"]),
        "context_len": int(meta["context_len"]),
        "horizon": horizon,
        **basis,
    }
    torch.save(payload, ds_dir / f"fourier_basis_fine_mask_h{horizon}.pt")


def cache_npz(
    npz_path: Path,
    output_root: Path,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device | None,
    metadata_only: bool,
    skip_existing: bool,
) -> None:
    npz_data = load_npz(npz_path)
    meta = npz_data["meta"]
    ds_dir = output_root / "complex" / dataset_name(meta)
    ds_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(meta["horizon"])
    context_len = int(meta["context_len"])
    backbone_path = ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt"
    expected = [
        ds_dir / f"raw_futures_h{horizon}.pt",
        ds_dir / f"component_targets_h{horizon}.pt",
        ds_dir / f"fourier_basis_fine_mask_h{horizon}.pt",
        ds_dir / f"seasonal_coefficients_fine_mask_h{horizon}.pt",
    ]
    if skip_existing and all(path.exists() for path in expected) and (metadata_only or backbone_path.exists()):
        print(f"skip_cache={ds_dir}", flush=True)
        return

    save_targets(npz_data, ds_dir)
    save_basis(npz_data, ds_dir)
    save_coefficients(npz_data, ds_dir)
    if not metadata_only:
        if device is None or backbone is None or revin_fn is None or update_stats_fn is None:
            raise RuntimeError("Backbone is required unless --metadata_only is set")
        save_backbone_cache(npz_data, backbone_path, batch_size, backbone, revin_fn, update_stats_fn, device)
    print(f"saved_cache={ds_dir}", flush=True)


def raw_root_for(split: str, args: argparse.Namespace) -> Path:
    return args.train_raw_root if split == "train" else args.eval_raw_root


def cache_root_for(split: str, args: argparse.Namespace) -> Path:
    return args.train_cache_root if split == "train" else args.eval_cache_root


def write_manifest(
    root: Path,
    split: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    context_len: int,
    n_samples: int,
    seed: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    granularity_cases = {}
    for granularity in args.granularities:
        active = active_harmonics(granularity, context_len)
        cases = enumerate_count_cases(active)
        granularity_cases[granularity] = {
            "active_harmonics": active,
            "n_cases": len(cases),
        }
    manifest = {
        "generator": "fourier_all_v1",
        "split": split,
        "config": str(args.config),
        "context_len": int(context_len),
        "horizons": list(args.horizons or cfg["global"]["horizons"]),
        "n_samples_per_file": int(n_samples),
        "seed": int(seed),
        "granularities": granularity_cases,
        "families": FAMILIES,
        "k_max": K_MAX,
        "compositions": list(args.compositions),
        "trend_levels": list(args.trend_levels),
        "note": (
            "Each file cycles through all active-order count cases for its granularity. "
            "For each sample, concrete harmonic indices and coefficients are randomized."
        ),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    cfg = load_generation_config(args.config)
    global_cfg = cfg["global"]
    horizons = args.horizons or list(global_cfg["horizons"])
    context_len = args.context_len or int(global_cfg["context_len"])
    seed = args.seed if args.seed is not None else int(global_cfg["seed"])
    train_samples = args.train_samples or int(global_cfg.get("train_samples", global_cfg["n_samples"]))
    eval_samples = args.eval_samples or int(global_cfg.get("eval_samples", max(512, global_cfg["n_samples"] // 2)))

    for horizon in horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon {horizon}; expected one of {sorted(VALID_HORIZONS)}")

    device = None
    backbone = None
    revin_fn = None
    update_stats_fn = None
    if not args.raw_only and not args.metadata_only:
        device = resolve_device(args.device)
        backbone, revin_fn, update_stats_fn = load_backbone(device, args.hf_cache_dir)

    for split in args.splits:
        n_samples = train_samples if split == "train" else eval_samples
        split_seed = seed + SPLIT_SEED_OFFSET[split]
        raw_root = raw_root_for(split, args)
        cache_root = cache_root_for(split, args)
        write_manifest(raw_root, split, args, cfg, context_len, n_samples, split_seed)
        cache_root.mkdir(parents=True, exist_ok=True)

        for composition in args.compositions:
            for trend_level in args.trend_levels:
                for granularity in args.granularities:
                    if not enumerate_count_cases(active_harmonics(granularity, context_len)):
                        print(f"[skip] no active hard-mask harmonics for {granularity}", flush=True)
                        continue
                    for horizon in horizons:
                        dataset = build_dataset(
                            split=split,
                            composition=composition,
                            trend_level=trend_level,
                            granularity=granularity,
                            horizon=int(horizon),
                            context_len=int(context_len),
                            n_samples=int(n_samples),
                            seed=int(split_seed),
                            cfg=cfg,
                            noise_scale=float(args.noise_scale),
                            amplitude_low=float(args.amplitude_low),
                            amplitude_high=float(args.amplitude_high),
                            coefficient_low=float(args.coefficient_low),
                            coefficient_high=float(args.coefficient_high),
                        )
                        npz_path = save_npz(dataset, raw_root)
                        print(
                            f"saved_npz={npz_path} samples={n_samples} "
                            f"cases={dataset['meta']['n_cases']} active={dataset['meta']['active_harmonics']}",
                            flush=True,
                        )
                        if not args.raw_only:
                            cache_npz(
                                npz_path=npz_path,
                                output_root=cache_root,
                                batch_size=int(args.batch_size),
                                backbone=backbone,
                                revin_fn=revin_fn,
                                update_stats_fn=update_stats_fn,
                                device=device,
                                metadata_only=bool(args.metadata_only),
                                skip_existing=bool(args.skip_existing),
                            )


if __name__ == "__main__":
    main()
