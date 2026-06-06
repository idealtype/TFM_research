#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/sia2/project/5.30fine_mask/results/analysis_tables")
OUT = ROOT / "performance_plots"
FINE_REAL_CSV = Path("/home/sia2/project/5.30fine_mask/results/real_lot_ett/real_eval_component_mae.csv")
PREVIOUS_REAL_CSV = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/"
    "all_domain_full_then_residual/eval/real_lot_ett_residual_extra/real_eval_component_mae.csv"
)
TIMESFM_REAL_CSV = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/"
    "real_lot_ett_single_model_phasefix/real_eval_component_mae.csv"
)
COLORS = {
    "fine_mask": "#2f6f9f",
    "previous": "#6f6f6f",
    "timesfm": "#c46a2b",
}


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grouped_bar(df: pd.DataFrame, x_col: str, title: str, path: Path, *, rotate: int = 0) -> None:
    labels = df[x_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.25
    fig_w = max(8.5, 0.55 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(x - width, df["fine_mask"], width, label="fine-mask", color=COLORS["fine_mask"])
    ax.bar(x, df["previous"], width, label="previous model", color=COLORS["previous"])
    ax.bar(x + width, df["timesfm"], width, label="TimesFM", color=COLORS["timesfm"])
    ax.set_title(title)
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _finish(fig, path)


def line_by_horizon(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [("fine_mask", "fine-mask"), ("previous", "previous model"), ("timesfm", "TimesFM")]:
        ax.plot(df["horizon"], df[key], marker="o", linewidth=2, label=label, color=COLORS[key])
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE")
    ax.set_xticks(df["horizon"].tolist())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finish(fig, path)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real = build_real_comparison()
    fourier = pd.read_csv(ROOT / "fourier_comparison.csv")
    nonf = pd.read_csv(ROOT / "nonfourier_comparison.csv")
    return real, fourier, nonf


def build_real_comparison() -> pd.DataFrame:
    fine = pd.read_csv(FINE_REAL_CSV)
    previous = pd.read_csv(PREVIOUS_REAL_CSV)
    timesfm = pd.read_csv(TIMESFM_REAL_CSV)
    key = ["domain", "dataset", "frequency", "horizon"]

    real = fine[
        key + [
            "n_samples",
            "model",
            "total_mae",
            "total_mse",
            "tfm_zeroshot_mae",
            "tfm_zeroshot_mse",
            "trend_mae",
            "seasonal_mae",
            "residual_mae",
            "checkpoint",
        ]
    ].rename(
        columns={
            "total_mae": "total_mae_fine",
            "tfm_zeroshot_mae": "tfm_zeroshot_mae_fine",
        }
    )
    real = real.merge(
        previous[key + ["total_mae", "total_mse", "checkpoint"]].rename(
            columns={
                "total_mae": "total_mae_old",
                "total_mse": "total_mse_old",
                "checkpoint": "checkpoint_old",
            }
        ),
        on=key,
        how="inner",
    )
    real = real.merge(
        timesfm[key + ["tfm_zeroshot_mae", "tfm_zeroshot_mse"]].rename(
            columns={
                "tfm_zeroshot_mae": "tfm_zeroshot_mae_old",
                "tfm_zeroshot_mse": "tfm_zeroshot_mse_old",
            }
        ),
        on=key,
        how="left",
    )
    real["tfm_zeroshot_mae_fine"] = real["tfm_zeroshot_mae_fine"].fillna(real["tfm_zeroshot_mae_old"])
    real["fine_vs_old_pct"] = (real["total_mae_fine"] / real["total_mae_old"] - 1.0) * 100.0
    real["fine_vs_tfm_pct"] = (real["total_mae_fine"] / real["tfm_zeroshot_mae_fine"] - 1.0) * 100.0
    real.to_csv(ROOT / "real_comparison.csv", index=False)
    return real


def plot_overall(real: pd.DataFrame, fourier: pd.DataFrame, nonf: pd.DataFrame) -> None:
    rows = [
        {
            "split": "Real",
            "fine_mask": real["total_mae_fine"].mean(),
            "previous": real["total_mae_old"].mean(),
            "timesfm": real["tfm_zeroshot_mae_fine"].mean(),
        },
        {
            "split": "Fourier synth",
            "fine_mask": fourier["total_mae"].mean(),
            "previous": fourier["old_best_mae"].mean(),
            "timesfm": fourier["tfm_zeroshot_mae"].mean(),
        },
        {
            "split": "non-Fourier synth",
            "fine_mask": nonf["total_mae"].mean(),
            "previous": nonf["old_best_mae"].mean(),
            "timesfm": nonf["tfm_zeroshot_mae"].mean(),
        },
    ]
    grouped_bar(pd.DataFrame(rows), "split", "Overall MAE Comparison", OUT / "overall_mae_comparison.png")


def plot_real(real: pd.DataFrame) -> None:
    by_h = real.groupby("horizon", as_index=False).agg(
        fine_mask=("total_mae_fine", "mean"),
        previous=("total_mae_old", "mean"),
        timesfm=("tfm_zeroshot_mae_fine", "mean"),
    )
    line_by_horizon(by_h, "Real Evaluation MAE by Horizon", OUT / "real_mae_by_horizon.png")

    by_ds = real.groupby("dataset", as_index=False).agg(
        fine_mask=("total_mae_fine", "mean"),
        previous=("total_mae_old", "mean"),
        timesfm=("tfm_zeroshot_mae_fine", "mean"),
    ).sort_values("fine_mask")
    grouped_bar(by_ds, "dataset", "Real Evaluation MAE by Dataset", OUT / "real_mae_by_dataset.png", rotate=35)


def plot_fourier(fourier: pd.DataFrame) -> None:
    by_h = fourier.groupby("horizon", as_index=False).agg(
        fine_mask=("total_mae", "mean"),
        previous=("old_best_mae", "mean"),
        timesfm=("tfm_zeroshot_mae", "mean"),
    )
    line_by_horizon(by_h, "Fourier Synthetic MAE by Horizon", OUT / "fourier_mae_by_horizon.png")

    order = sorted(fourier["seasonal_level"].dropna().unique(), key=lambda s: int(str(s).replace("S", "")))
    by_s = fourier.groupby("seasonal_level", as_index=False).agg(
        fine_mask=("total_mae", "mean"),
        previous=("old_best_mae", "mean"),
        timesfm=("tfm_zeroshot_mae", "mean"),
    )
    by_s["seasonal_level"] = pd.Categorical(by_s["seasonal_level"], categories=order, ordered=True)
    by_s = by_s.sort_values("seasonal_level")
    grouped_bar(by_s, "seasonal_level", "Fourier Synthetic MAE by Seasonal Type", OUT / "fourier_mae_by_seasonal.png")


def plot_nonf(nonf: pd.DataFrame) -> None:
    by_h = nonf.groupby("horizon", as_index=False).agg(
        fine_mask=("total_mae", "mean"),
        previous=("old_best_mae", "mean"),
        timesfm=("tfm_zeroshot_mae", "mean"),
    )
    line_by_horizon(by_h, "non-Fourier Synthetic MAE by Horizon", OUT / "nonfourier_mae_by_horizon.png")

    by_stage = nonf.groupby("stage", as_index=False).agg(
        fine_mask=("total_mae", "mean"),
        previous=("old_best_mae", "mean"),
        timesfm=("tfm_zeroshot_mae", "mean"),
    )
    grouped_bar(by_stage, "stage", "non-Fourier Synthetic MAE by Stage", OUT / "nonfourier_mae_by_stage.png", rotate=15)


def main() -> None:
    real, fourier, nonf = load_tables()
    plot_overall(real, fourier, nonf)
    plot_real(real)
    plot_fourier(fourier)
    plot_nonf(nonf)
    print(f"Saved plots to {OUT}")


if __name__ == "__main__":
    main()
