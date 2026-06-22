#!/usr/bin/env python3
"""Validate a compact training pool for consistency.

For every synth ds_dir × horizon pair that has a backbone file, checks:
  - backbone_emb_c*_h{H}_stride1.pt   exists
  - raw_futures_h{H}.pt               exists
  - component_targets_h{H}.pt         exists
  - seasonal_coefficients*h{H}.pt     exists (either naming convention)
  - row counts match across all four
  - coeff families daily/weekly/monthly/yearly all present
  - coeff widths daily=20, weekly=8, monthly=4, yearly=16

Exits with code 1 on any failure (fail-fast).

Usage:
    python src/data_prep/validate_compact_pool.py <pool_root>
    python src/data_prep/validate_compact_pool.py /tmp/data
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


HORIZONS = [96, 192, 336, 720]
K_MAX = {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8}
COEFF_FAMILIES = ("daily", "weekly", "monthly", "yearly")
EXPECTED_WIDTHS = {f: 2 * k for f, k in K_MAX.items()}


def find_backbone_path(ds_dir: Path, horizon: int) -> Path | None:
    matches = sorted(ds_dir.glob(f"backbone_emb_c*_h{horizon}_stride1.pt"))
    return matches[0] if matches else None


def find_coeff_path(ds_dir: Path, horizon: int) -> Path | None:
    candidates = [
        ds_dir / f"seasonal_coefficients_fine_mask_h{horizon}.pt",
        ds_dir / f"seasonal_coefficients_h{horizon}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(ds_dir.glob(f"seasonal_coefficients*h{horizon}.pt"))
    return matches[0] if matches else None


def extract_coeff_family(payload: dict, family: str) -> torch.Tensor | None:
    for key in (
        family,
        f"{family}_coefficients",
        f"{family}_coeffs",
        f"{family}_coef",
        f"{family}_basis_coefficients",
    ):
        if key in payload:
            val = payload[key]
            if torch.is_tensor(val):
                return val
    for nested_key in ("seasonal_coefficients", "coefficients", "coeffs"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            for key in (
                family,
                f"{family}_coefficients",
                f"{family}_coeffs",
                f"{family}_coef",
                f"{family}_basis_coefficients",
            ):
                if key in nested and torch.is_tensor(nested[key]):
                    return nested[key]
    return None


def validate_pair(ds_dir: Path, horizon: int, errors: list[str]) -> None:
    label = f"{ds_dir.name}/h{horizon}"

    backbone_path = find_backbone_path(ds_dir, horizon)
    if backbone_path is None:
        errors.append(f"{label}: backbone_emb_c*_h{horizon}_stride1.pt missing")
        return

    raw_path = ds_dir / f"raw_futures_h{horizon}.pt"
    comp_path = ds_dir / f"component_targets_h{horizon}.pt"
    coeff_path = find_coeff_path(ds_dir, horizon)

    missing = []
    if not raw_path.exists():
        missing.append(f"raw_futures_h{horizon}.pt")
    if not comp_path.exists():
        missing.append(f"component_targets_h{horizon}.pt")
    if coeff_path is None:
        missing.append(f"seasonal_coefficients*h{horizon}.pt")
    if missing:
        errors.append(f"{label}: missing: {', '.join(missing)}")
        return

    backbone_data = torch.load(backbone_path, map_location="cpu", weights_only=False)
    n_backbone = int(backbone_data["embeddings"].shape[0])
    del backbone_data

    raw_data = torch.load(raw_path, map_location="cpu", weights_only=False)
    n_raw = int(raw_data["futures_n"].shape[0])
    del raw_data

    comp_data = torch.load(comp_path, map_location="cpu", weights_only=False)
    n_comp = int(comp_data["trend_n"].shape[0])
    del comp_data

    coeff_data = torch.load(coeff_path, map_location="cpu", weights_only=False)

    n_coeff: int | None = None
    for family in COEFF_FAMILIES:
        tensor = extract_coeff_family(coeff_data, family)
        if tensor is None:
            errors.append(f"{label}: coeff family '{family}' missing in {coeff_path.name}")
            continue
        tensor = tensor.float()
        if tensor.ndim != 2:
            errors.append(f"{label}: {family} coeff ndim={tensor.ndim} (expected 2)")
            continue
        expected_w = EXPECTED_WIDTHS[family]
        if tensor.shape[1] != expected_w:
            errors.append(
                f"{label}: {family} coeff width={tensor.shape[1]} expected={expected_w}"
            )
        if n_coeff is None:
            n_coeff = int(tensor.shape[0])
        elif int(tensor.shape[0]) != n_coeff:
            errors.append(
                f"{label}: {family} coeff rows={tensor.shape[0]} "
                f"inconsistent with previous={n_coeff}"
            )
    del coeff_data

    if n_coeff is None:
        return

    row_mismatches = []
    if n_raw != n_backbone:
        row_mismatches.append(f"raw={n_raw}")
    if n_comp != n_backbone:
        row_mismatches.append(f"comp={n_comp}")
    if n_coeff != n_backbone:
        row_mismatches.append(f"coeff={n_coeff}")
    if row_mismatches:
        errors.append(
            f"{label}: row count mismatch backbone={n_backbone} "
            + " ".join(row_mismatches)
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <compact_pool_root>", file=sys.stderr)
        sys.exit(1)

    pool_root = Path(sys.argv[1]).resolve()
    if not pool_root.exists():
        print(f"[validate] ERROR: pool root not found: {pool_root}", file=sys.stderr)
        sys.exit(1)

    ds_dirs: set[Path] = set()
    for h in HORIZONS:
        for p in pool_root.rglob(f"backbone_emb_c*_h{h}_stride1.pt"):
            ds_dirs.add(p.parent)

    if not ds_dirs:
        print(
            f"[validate] ERROR: no backbone_emb files found under {pool_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[validate] pool={pool_root}", flush=True)
    print(f"[validate] {len(ds_dirs)} synth ds_dirs", flush=True)

    errors: list[str] = []
    n_checked = 0

    for ds_dir in sorted(ds_dirs):
        for h in HORIZONS:
            if find_backbone_path(ds_dir, h) is None:
                continue
            n_checked += 1
            validate_pair(ds_dir, h, errors)

    print(f"[validate] checked {n_checked} (ds_dir, horizon) pairs", flush=True)

    if errors:
        print(
            f"[validate] FAILED — {len(errors)} error(s):",
            file=sys.stderr,
            flush=True,
        )
        for e in errors:
            print(f"  [validate] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[validate] ALL PASS", flush=True)


if __name__ == "__main__":
    main()
