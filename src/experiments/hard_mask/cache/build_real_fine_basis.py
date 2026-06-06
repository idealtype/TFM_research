#!/usr/bin/env python3
"""Build per-harmonic fine_mask Fourier basis files for real LOTSA data.

Scans futures_c{C}_{freq}_h{H}_*.pt files recursively under --cache_dir,
and outputs fourier_basis_c{C}_{freq}_h{H}_fine_mask_*.pt beside each future file.

Does NOT overwrite existing fine_mask files unless --overwrite is specified.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FREQ_DAYS, build_fine_mask_basis  # noqa: E402


FUTURE_RE = re.compile(
    r"^futures_c(?P<context_len>\d+)_(?P<freq>.+)_h(?P<horizon>\d+)_(?P<suffix>.+)\.pt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Root directory containing per-dataset subdirectories with backbone files.")
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing fine_mask basis files.")
    return parser.parse_args()


def parse_future_name(path: Path) -> tuple[int, str, int, str] | None:
    """Returns (context_len, freq, horizon, suffix) or None if the filename does not match."""
    match = FUTURE_RE.match(path.name)
    if match is None:
        return None
    return (
        int(match.group("context_len")),
        str(match.group("freq")),
        int(match.group("horizon")),
        str(match.group("suffix")),
    )


def build_and_save_basis(
    freq: str,
    context_len: int,
    horizon: int,
    output_path: Path,
) -> None:
    bases = build_fine_mask_basis(freq, context_len, horizon)
    save_data = {
        "freq": freq,
        "context_len": int(context_len),
        "horizon": int(horizon),
        "daily_basis": bases["daily_basis"],
        "weekly_basis": bases["weekly_basis"],
        "monthly_basis": bases["monthly_basis"],
        "yearly_basis": bases["yearly_basis"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path)


def main() -> None:
    args = parse_args()
    if not args.horizons:
        raise ValueError("--horizons must not be empty.")
    if any(h <= 0 for h in args.horizons):
        raise ValueError("All horizons must be positive.")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    future_files = sorted(cache_dir.rglob("futures_c*_h*_*.pt"))
    if not future_files:
        raise FileNotFoundError(f"No future cache files found under {cache_dir}")

    created = 0
    skipped = 0
    errors = 0

    wanted_horizons = {int(h) for h in args.horizons}
    seen: set[Path] = set()

    for future_path in future_files:
        parsed = parse_future_name(future_path)
        if parsed is None:
            print(f"[skip] unexpected filename: {future_path.name}", flush=True)
            continue
        context_len, freq, horizon, suffix = parsed
        if horizon not in wanted_horizons:
            continue

        if freq not in FREQ_DAYS:
            print(f"[skip] unknown freq={freq!r} in {future_path.name}", flush=True)
            continue

        parent = future_path.parent
        backbone_candidates = sorted(parent.glob(f"backbone_emb_c{context_len}_{freq}*.pt"))
        if not backbone_candidates:
            backbone_candidates = sorted(parent.glob(f"backbone_emb_c{context_len}*.pt"))
        if not backbone_candidates:
            print(f"[skip] no matching backbone for {future_path}", flush=True)
            continue

        output_name = f"fourier_basis_c{context_len}_{freq}_h{horizon}_fine_mask_{suffix}.pt"
        output_path = parent / output_name
        if output_path in seen:
            continue
        seen.add(output_path)

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            build_and_save_basis(freq, context_len, horizon, output_path)
            created += 1
            print(f"[saved] {output_path}", flush=True)
        except Exception as exc:
            errors += 1
            print(f"[error] {output_path}: {exc}", flush=True)

    print("\nFine-mask basis build summary:")
    print(f"  future_files:   {len(future_files)}")
    print(f"  bases_created:  {created}")
    print(f"  bases_skipped:  {skipped}")
    print(f"  errors:         {errors}")
    if errors:
        raise SystemExit(f"{errors} error(s) during basis construction.")


if __name__ == "__main__":
    main()
