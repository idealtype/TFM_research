from __future__ import annotations

import argparse
import os
import math
import re
from pathlib import Path

import torch

from shared_utils import FREQ_DAYS

SEASONALITY_PERIODS = {"daily": 1.0, "weekly": 7.0, "yearly": 365.25}
N_FOURIER_TERMS = {"daily": 10, "weekly": 4, "yearly": 8}
BACKBONE_RE = re.compile(r"^backbone_emb_c(?P<context_len>\d+)_(?P<freq>.+?)_lotsa(?P<suffix>_test)?\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing seasonality mask and Fourier basis files.")
    return parser.parse_args()


def parse_backbone_name(path: Path) -> tuple[int, str, bool]:
    match = BACKBONE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected backbone cache filename: {path}")
    return int(match.group("context_len")), str(match.group("freq")), bool(match.group("suffix"))


def build_mask(freq: str, context_len: int) -> dict:
    if freq not in FREQ_DAYS:
        raise ValueError(f"Unsupported freq: {freq}")
    fd = FREQ_DAYS[freq]
    context_span = context_len * fd
    mask = {}
    for stype in ["daily", "weekly", "yearly"]:
        period = SEASONALITY_PERIODS[stype]
        mask[stype] = bool((fd < period) and (context_span >= period))
    return {
        "daily": mask["daily"],
        "weekly": mask["weekly"],
        "yearly": mask["yearly"],
        "freq": freq,
        "context_len": int(context_len),
    }


def build_basis(freq: str, context_len: int, horizon: int, mask_data: dict) -> dict:
    fd = FREQ_DAYS[freq]
    t = torch.arange(int(context_len), int(context_len) + int(horizon), dtype=torch.float32)
    save_data = {
        "mask": {
            "daily": bool(mask_data["daily"]),
            "weekly": bool(mask_data["weekly"]),
            "yearly": bool(mask_data["yearly"]),
        },
        "freq": freq,
        "horizon": int(horizon),
    }

    for stype in ["daily", "weekly", "yearly"]:
        n = N_FOURIER_TERMS[stype]
        basis = torch.zeros(horizon, 2 * n)
        if mask_data[stype]:
            p_steps = SEASONALITY_PERIODS[stype] / fd
            for k in range(n):
                basis[:, 2 * k] = torch.sin(2 * math.pi * (k + 1) * t / p_steps)
                basis[:, 2 * k + 1] = torch.cos(2 * math.pi * (k + 1) * t / p_steps)
        save_data[f"{stype}_basis"] = basis
    return save_data


def output_suffix(is_test: bool) -> str:
    return "_lotsa_test" if is_test else "_lotsa"


def main() -> None:
    args = parse_args()
    if not args.horizons:
        raise ValueError("--horizons must not be empty.")
    if any(h <= 0 for h in args.horizons):
        raise ValueError("All horizons must be positive.")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    backbone_files = sorted(cache_dir.glob("*/backbone_emb_c*_lotsa*.pt"))
    if not backbone_files:
        raise FileNotFoundError(f"No backbone cache files found under {cache_dir}")

    mask_created = 0
    mask_skipped = 0
    basis_created = 0
    basis_skipped = 0

    for backbone_path in backbone_files:
        context_len, freq, is_test = parse_backbone_name(backbone_path)
        suffix = output_suffix(is_test)
        parent = backbone_path.parent

        mask_path = parent / f"seasonality_mask_c{context_len}_{freq}{suffix}.pt"
        if mask_path.exists() and not args.overwrite:
            mask_data = torch.load(mask_path, weights_only=False)
            mask_skipped += 1
        else:
            mask_data = build_mask(freq, context_len)
            torch.save(mask_data, mask_path)
            mask_created += 1
            print(f"[saved] {mask_path}")

        for horizon in args.horizons:
            basis_path = parent / f"fourier_basis_c{context_len}_{freq}_h{horizon}{suffix}.pt"
            if basis_path.exists() and not args.overwrite:
                basis_skipped += 1
                continue
            basis_data = build_basis(freq, context_len, horizon, mask_data)
            torch.save(basis_data, basis_path)
            basis_created += 1
            print(f"[saved] {basis_path}")

    print("\nSeasonality cache summary:")
    print(f"  backbone_files: {len(backbone_files)}")
    print(f"  masks_created: {mask_created}")
    print(f"  masks_skipped: {mask_skipped}")
    print(f"  bases_created: {basis_created}")
    print(f"  bases_skipped: {basis_skipped}")


if __name__ == "__main__":
    main()
