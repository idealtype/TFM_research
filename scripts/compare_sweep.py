#!/usr/bin/env python3
"""Compare coeff_l1_weight sweep results from nogate_softmask eval runs.

Reads real_eval_mae.csv from each sweep run directory and prints a comparison
table of total_mae by horizon and overall mean.

Usage:
    python scripts/compare_sweep.py
    python scripts/compare_sweep.py --sweep_dir ./results/nogate_softmask
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = REPO_ROOT / "results" / "nogate_softmask"
HORIZONS = [96, 192, 336, 720]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep_dir", type=Path, default=DEFAULT_SWEEP_DIR,
        help="Directory containing sweep run subdirectories (default: results/nogate_softmask/)",
    )
    return parser.parse_args()


def extract_l1_value(dir_name: str) -> str | None:
    marker = "_sweep_l1_"
    idx = dir_name.find(marker)
    if idx == -1:
        return None
    return dir_name[idx + len(marker):]


def read_mae_by_horizon(csv_path: Path) -> dict[int, list[float]]:
    by_horizon: dict[int, list[float]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                h = int(row["horizon"])
                mae = float(row["total_mae"])
            except (KeyError, ValueError):
                continue
            by_horizon.setdefault(h, []).append(mae)
    return by_horizon


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    sweep_dir: Path = args.sweep_dir

    if not sweep_dir.exists():
        print(f"Directory not found: {sweep_dir}")
        print("Download results first with vesslctl volume download.")
        return

    runs = sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and "_sweep_l1_" in d.name],
        key=lambda d: d.name,
    )
    if not runs:
        print(f"No sweep_l1_* directories found under {sweep_dir}")
        return

    results = []
    for run_dir in runs:
        csv_path = run_dir / "eval_real" / "real_eval_mae.csv"
        l1_val = extract_l1_value(run_dir.name) or run_dir.name
        if not csv_path.exists():
            print(f"  [missing] {run_dir.name}/eval_real/real_eval_mae.csv — skipping")
            results.append({"l1": l1_val, "run": run_dir.name, "by_h": {}, "mean": float("nan")})
            continue
        by_h = read_mae_by_horizon(csv_path)
        horizon_means = {h: mean(by_h.get(h, [])) for h in HORIZONS}
        all_vals = [v for vals in by_h.values() for v in vals]
        overall = mean(all_vals)
        results.append({"l1": l1_val, "run": run_dir.name, "by_h": horizon_means, "mean": overall})

    results.sort(key=lambda r: r["mean"])

    # ---------- print table ----------
    h_cols = [f"h{h}" for h in HORIZONS]
    col_w = 10
    header = f"{'coeff_l1_weight':>18}  " + "  ".join(f"{c:>{col_w}}" for c in h_cols) + f"  {'mean':>{col_w}}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        h_vals = "  ".join(
            f"{r['by_h'].get(h, float('nan')):>{col_w}.5f}" for h in HORIZONS
        )
        mean_str = f"{r['mean']:>{col_w}.5f}" if r["mean"] == r["mean"] else f"{'n/a':>{col_w}}"
        print(f"{r['l1']:>18}  {h_vals}  {mean_str}")
    print()

    # ---------- save CSV ----------
    out_csv = sweep_dir / "sweep_comparison.csv"
    fieldnames = ["coeff_l1_weight", "run_dir"] + h_cols + ["mean_mae"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict = {
                "coeff_l1_weight": r["l1"],
                "run_dir": r["run"],
                "mean_mae": f"{r['mean']:.6f}" if r["mean"] == r["mean"] else "",
            }
            for h in HORIZONS:
                row[f"h{h}"] = f"{r['by_h'].get(h, float('nan')):.6f}"
            writer.writerow(row)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
