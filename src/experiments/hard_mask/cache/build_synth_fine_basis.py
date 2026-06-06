#!/usr/bin/env python3
"""Build per-harmonic fine_mask Fourier basis for existing S1-S10 synthetic caches.

Scans existing synthetic cache directories for backbone_emb files and generates
fourier_basis_fine_mask_h{H}.pt alongside (does NOT overwrite fourier_basis_h{H}.pt).

GPU is NOT required; this script runs on CPU only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = next(parent for parent in THIS_FILE.parents if (parent / "loader_utils.py").exists())
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import resolve_data_path, resolve_project_path  # noqa: E402

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FREQ_DAYS, build_fine_mask_basis  # noqa: E402


# Backbone files in complex synth caches use granularity as frequency
GRANULARITY_TO_FREQ = {
    "hourly": "hourly",
    "daily": "daily",
    "weekly": "weekly",
}

# The backbone file name pattern for complex synth caches
BACKBONE_RE = re.compile(
    r"^backbone_emb_c(?P<context_len>\d+)_h(?P<horizon>\d+)_stride1\.pt$"
)

# Directory name pattern: {composition}_{trend}_{seasonal}_{granularity}_seed{seed}_c{C}_h{H}
CACHE_DIR_RE = re.compile(
    r"^(?:A\d+_)?(?:T\d+_)?(?:[A-Z]+\d+_)?(?P<granularity>\w+?)_seed\d+_c\d+_h\d+$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache_roots",
        type=Path,
        nargs="+",
        default=[
            resolve_data_path("/home/sia2/project/data/synthetic_nonF/synth_train_nonfourier"),
            resolve_data_path("/home/sia2/project/data/synthetic_nonF/synth_eval_nonfourier"),
            resolve_data_path("/home/sia2/project/data/synthetic/func_dec_syn_cent_complex_train_cache_10_4_8_fixed_phase_scale"),
            resolve_data_path("/home/sia2/project/data/synthetic/func_dec_syn_cent_complex_eval_cache_10_4_8_fixed_phase_scale"),
        ],
        help="Root directories containing synthetic cache subdirectories.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_cache_dirs(root: Path) -> list[Path]:
    """Find all leaf directories that contain at least one backbone_emb file."""
    found = []
    for backbone_file in sorted(root.rglob("backbone_emb_c*_h*_stride1.pt")):
        ds_dir = backbone_file.parent
        if ds_dir not in found:
            found.append(ds_dir)
    return found


def infer_granularity(ds_dir: Path) -> str | None:
    """Infer granularity from the directory name."""
    match = CACHE_DIR_RE.match(ds_dir.name)
    if match:
        return match.group("granularity")
    # Fallback: look for known granularity tokens in the name
    name = ds_dir.name.lower()
    for gran in ["hourly", "daily", "weekly"]:
        if gran in name:
            return gran
    return None


def build_and_save(ds_dir: Path, granularity: str, horizon: int, context_len: int, overwrite: bool) -> bool:
    out_path = ds_dir / f"fourier_basis_fine_mask_h{horizon}.pt"
    if out_path.exists() and not overwrite:
        return False
    freq = GRANULARITY_TO_FREQ.get(granularity, granularity)
    if freq not in FREQ_DAYS:
        print(f"[skip] unknown freq={freq!r} for {ds_dir.name}", flush=True)
        return False
    bases = build_fine_mask_basis(freq, context_len, horizon)
    save_data = {
        "freq": freq,
        "granularity": granularity,
        "context_len": int(context_len),
        "horizon": int(horizon),
        "daily_basis": bases["daily_basis"],
        "weekly_basis": bases["weekly_basis"],
        "monthly_basis": bases["monthly_basis"],
        "yearly_basis": bases["yearly_basis"],
    }
    torch.save(save_data, out_path)
    return True


def main() -> None:
    args = parse_args()

    created = 0
    skipped = 0
    errors = 0

    for root in args.cache_roots:
        if not root.exists():
            print(f"[skip] cache_root does not exist: {root}", flush=True)
            continue

        ds_dirs = discover_cache_dirs(root)
        print(f"[info] {root}: found {len(ds_dirs)} cache directories", flush=True)

        for ds_dir in ds_dirs:
            granularity = infer_granularity(ds_dir)
            if granularity is None:
                print(f"[skip] cannot infer granularity from {ds_dir.name}", flush=True)
                continue

            # Determine context_len from any existing backbone file
            backbone_files = sorted(ds_dir.glob("backbone_emb_c*_h*_stride1.pt"))
            if not backbone_files:
                continue
            match = BACKBONE_RE.match(backbone_files[0].name)
            if not match:
                continue
            context_len = int(match.group("context_len"))

            for horizon in args.horizons:
                try:
                    saved = build_and_save(ds_dir, granularity, horizon, context_len, args.overwrite)
                    if saved:
                        created += 1
                        print(f"[saved] {ds_dir / f'fourier_basis_fine_mask_h{horizon}.pt'}", flush=True)
                    else:
                        skipped += 1
                except Exception as exc:
                    errors += 1
                    print(f"[error] {ds_dir} h{horizon}: {exc}", flush=True)

    print("\nSynth fine-mask basis build summary:")
    print(f"  bases_created: {created}")
    print(f"  bases_skipped: {skipped}")
    print(f"  errors:        {errors}")
    if errors:
        raise SystemExit(f"{errors} error(s) during basis construction.")


if __name__ == "__main__":
    main()
