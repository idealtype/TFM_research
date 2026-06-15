#!/usr/bin/env python3
"""Compare soft-mask warm-mix against hard-mask warm-mix and TimesFM.

This intentionally excludes non-Fourier synthetic evaluation.  The soft-mask
warm-mix run stopped before non-Fourier eval, and the comparison here preserves
the existing real/Fourier result layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOFT_ROOT = Path("/home/sia2/project/5.30soft_mask/results/fourier_warm_real_mix")
HARD_ROOT = Path("/home/sia2/project/5.30fine_mask/results/fourier_warm_real_mix")
OUT_ROOT = SOFT_ROOT / "analysis_tables"
PLOT_ROOT = OUT_ROOT / "performance_plots"

SOFT_REAL = SOFT_ROOT / "real_lot_ett" / "real_eval_mae.csv"
HARD_REAL = HARD_ROOT / "real_lot_ett" / "real_eval_mae.csv"
SOFT_FOURIER = SOFT_ROOT / "fourier_synth" / "fourier_synth_eval.csv"
HARD_FOURIER = HARD_ROOT / "fourier_synth" / "fourier_synth_eval.csv"

COLORS = {
    "soft": "#1f77b4",
    "hard": "#2ca02c",
    "timesfm": "#d95f02",
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    value_cols: list[tuple[str, str]],
    title: str,
    path: Path,
    rotate: int = 0,
) -> None:
    labels = df[x_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = min(0.8 / max(1, len(value_cols)), 0.24)
    fig_w = max(8.5, 0.55 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    offsets = (np.arange(len(value_cols)) - (len(value_cols) - 1) / 2) * width
    for offset, (col, label) in zip(offsets, value_cols, strict=True):
        ax.bar(x + offset, df[col], width, label=label, color=COLORS.get(col))
    ax.set_title(title)
    ax.set_ylabel("Normalized MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish(fig, path)


def line_by_horizon(df: pd.DataFrame, value_cols: list[tuple[str, str]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in value_cols:
        ax.plot(df["horizon"], df[col], marker="o", linewidth=2, label=label, color=COLORS.get(col))
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Normalized MAE")
    ax.set_xticks(df["horizon"].tolist())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    finish(fig, path)


def build_real() -> pd.DataFrame:
    key = ["domain", "dataset", "frequency", "horizon"]
    soft = pd.read_csv(SOFT_REAL)
    hard = pd.read_csv(HARD_REAL)
    out = soft[key + ["n_samples", "total_mae", "total_mse", "tfm_zeroshot_mae", "tfm_zeroshot_mse", "checkpoint"]].rename(
        columns={
            "total_mae": "soft",
            "total_mse": "soft_mse",
            "tfm_zeroshot_mae": "timesfm",
            "tfm_zeroshot_mse": "timesfm_mse",
            "checkpoint": "soft_checkpoint",
        }
    )
    out = out.merge(
        hard[key + ["total_mae", "total_mse", "checkpoint"]].rename(
            columns={"total_mae": "hard", "total_mse": "hard_mse", "checkpoint": "hard_checkpoint"}
        ),
        on=key,
        how="inner",
    )
    out["soft_vs_hard_pct"] = (out["soft"] / out["hard"] - 1.0) * 100.0
    out["soft_vs_timesfm_pct"] = (out["soft"] / out["timesfm"] - 1.0) * 100.0
    out["hard_vs_timesfm_pct"] = (out["hard"] / out["timesfm"] - 1.0) * 100.0
    out.to_csv(OUT_ROOT / "real_comparison.csv", index=False)
    return out


def build_fourier() -> pd.DataFrame:
    key = ["composition", "trend_level", "seasonal_level", "granularity", "horizon"]
    soft = pd.read_csv(SOFT_FOURIER)
    hard = pd.read_csv(HARD_FOURIER)
    out = soft[key + ["n_samples", "total_mae", "total_mse", "tfm_zeroshot_mae", "tfm_zeroshot_mse", "checkpoint"]].rename(
        columns={
            "total_mae": "soft",
            "total_mse": "soft_mse",
            "tfm_zeroshot_mae": "timesfm",
            "tfm_zeroshot_mse": "timesfm_mse",
            "checkpoint": "soft_checkpoint",
        }
    )
    out = out.merge(
        hard[key + ["total_mae", "total_mse", "checkpoint"]].rename(
            columns={"total_mae": "hard", "total_mse": "hard_mse", "checkpoint": "hard_checkpoint"}
        ),
        on=key,
        how="inner",
    )
    out["soft_vs_hard_pct"] = (out["soft"] / out["hard"] - 1.0) * 100.0
    out["soft_vs_timesfm_pct"] = (out["soft"] / out["timesfm"] - 1.0) * 100.0
    out["hard_vs_timesfm_pct"] = (out["hard"] / out["timesfm"] - 1.0) * 100.0
    out.to_csv(OUT_ROOT / "fourier_comparison.csv", index=False)
    return out


def plot_all(real: pd.DataFrame, fourier: pd.DataFrame) -> None:
    cols = [("soft", "soft-mask warm-mix"), ("hard", "hard-mask warm-mix"), ("timesfm", "TimesFM")]
    overall = pd.DataFrame(
        [
            {
                "split": "Real",
                "soft": real["soft"].mean(),
                "hard": real["hard"].mean(),
                "timesfm": real["timesfm"].mean(),
            },
            {
                "split": "Fourier synth",
                "soft": fourier["soft"].mean(),
                "hard": fourier["hard"].mean(),
                "timesfm": fourier["timesfm"].mean(),
            },
        ]
    )
    grouped_bar(overall, "split", cols, "Warm-Mix Overall MAE", PLOT_ROOT / "overall_mae_comparison.png")

    by_real_h = real.groupby("horizon")[["soft", "hard", "timesfm"]].mean().reset_index()
    line_by_horizon(by_real_h, cols, "Real MAE by Horizon", PLOT_ROOT / "real_mae_by_horizon.png")

    by_fourier_h = fourier.groupby("horizon")[["soft", "hard", "timesfm"]].mean().reset_index()
    line_by_horizon(by_fourier_h, cols, "Fourier Synthetic MAE by Horizon", PLOT_ROOT / "fourier_mae_by_horizon.png")

    by_dataset = real.groupby("dataset")[["soft", "hard", "timesfm"]].mean().reset_index()
    grouped_bar(by_dataset, "dataset", cols, "Real MAE by Dataset", PLOT_ROOT / "real_mae_by_dataset.png", rotate=35)

    by_level = fourier.groupby("seasonal_level")[["soft", "hard", "timesfm"]].mean().reset_index()
    by_level["sort_id"] = by_level["seasonal_level"].str.extract(r"(\d+)").astype(int)
    by_level = by_level.sort_values("sort_id").drop(columns=["sort_id"])
    grouped_bar(
        by_level,
        "seasonal_level",
        cols,
        "Fourier Synthetic MAE by Seasonal Level",
        PLOT_ROOT / "fourier_mae_by_seasonal_level.png",
    )

    by_comp = fourier.groupby("composition")[["soft", "hard", "timesfm"]].mean().reset_index()
    grouped_bar(
        by_comp,
        "composition",
        cols,
        "Fourier Synthetic MAE by Composition",
        PLOT_ROOT / "fourier_mae_by_composition.png",
    )

    summary = pd.DataFrame(
        [
            {
                "split": "real",
                "soft": real["soft"].mean(),
                "hard": real["hard"].mean(),
                "timesfm": real["timesfm"].mean(),
                "soft_vs_hard_pct": (real["soft"].mean() / real["hard"].mean() - 1.0) * 100.0,
                "soft_vs_timesfm_pct": (real["soft"].mean() / real["timesfm"].mean() - 1.0) * 100.0,
            },
            {
                "split": "fourier_synth",
                "soft": fourier["soft"].mean(),
                "hard": fourier["hard"].mean(),
                "timesfm": fourier["timesfm"].mean(),
                "soft_vs_hard_pct": (fourier["soft"].mean() / fourier["hard"].mean() - 1.0) * 100.0,
                "soft_vs_timesfm_pct": (fourier["soft"].mean() / fourier["timesfm"].mean() - 1.0) * 100.0,
            },
        ]
    )
    summary.to_csv(OUT_ROOT / "summary.csv", index=False)
    payload = {
        "roots": {"soft": str(SOFT_ROOT), "hard": str(HARD_ROOT)},
        "excluded": "nonfourier_synth",
        "summary": json.loads(summary.replace({np.nan: None}).to_json(orient="records")),
        "outputs": {
            "real_csv": str(OUT_ROOT / "real_comparison.csv"),
            "fourier_csv": str(OUT_ROOT / "fourier_comparison.csv"),
            "summary_csv": str(OUT_ROOT / "summary.csv"),
            "plot_dir": str(PLOT_ROOT),
        },
    }
    (OUT_ROOT / "comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    for path in [SOFT_REAL, HARD_REAL, SOFT_FOURIER, HARD_FOURIER]:
        require(path)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    real = build_real()
    fourier = build_fourier()
    plot_all(real, fourier)
    print(json.dumps({
        "real_rows": int(len(real)),
        "fourier_rows": int(len(fourier)),
        "output_root": str(OUT_ROOT),
        "plot_root": str(PLOT_ROOT),
        "excluded": "nonfourier_synth",
    }, indent=2))


if __name__ == "__main__":
    main()
