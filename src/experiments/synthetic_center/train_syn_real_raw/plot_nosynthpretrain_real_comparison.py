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


RESULT_ROOT = Path("/home/sia2/project/5.22syn_cent/train_syn_real_raw/results")
NOSYNTH_ROOT = RESULT_ROOT / "nosynthpretrain"
NOSYNTH_CSV = NOSYNTH_ROOT / "eval" / "real_lot_ett_residual_extra" / "real_eval_component_mae.csv"
SYNTH_CSV = (
    RESULT_ROOT
    / "all_domain_full_then_residual"
    / "eval"
    / "real_lot_ett_residual_extra"
    / "real_eval_component_mae.csv"
)
TFM_CSV = RESULT_ROOT / "real_lot_ett_single_model_phasefix" / "real_eval_component_mae.csv"

ANALYSIS_DIR = NOSYNTH_ROOT / "analysis_tables"
PLOT_DIR = ANALYSIS_DIR / "performance_plots"

COLORS = {
    "nosynth": "#2f6f9f",
    "synth_pretrain": "#6f6f6f",
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
    fig_w = max(8.5, 0.6 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(x - width, df["nosynth"], width, label="no synth pretrain", color=COLORS["nosynth"])
    ax.bar(x, df["synth_pretrain"], width, label="synth pretrain", color=COLORS["synth_pretrain"])
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
    for key, label in [
        ("nosynth", "no synth pretrain"),
        ("synth_pretrain", "synth pretrain"),
        ("timesfm", "TimesFM"),
    ]:
        ax.plot(df["horizon"], df[key], marker="o", linewidth=2, label=label, color=COLORS[key])
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE")
    ax.set_xticks(df["horizon"].tolist())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finish(fig, path)


def build_comparison() -> pd.DataFrame:
    nosynth = pd.read_csv(NOSYNTH_CSV)
    synth = pd.read_csv(SYNTH_CSV)
    tfm = pd.read_csv(TFM_CSV)

    key = ["domain", "dataset", "frequency", "horizon"]
    merged = nosynth[key + ["n_samples", "total_mae", "total_mse", "checkpoint"]].rename(
        columns={
            "total_mae": "nosynth_mae",
            "total_mse": "nosynth_mse",
            "checkpoint": "nosynth_checkpoint",
        }
    )
    merged = merged.merge(
        synth[key + ["total_mae", "total_mse", "checkpoint"]].rename(
            columns={
                "total_mae": "synth_pretrain_mae",
                "total_mse": "synth_pretrain_mse",
                "checkpoint": "synth_pretrain_checkpoint",
            }
        ),
        on=key,
        how="inner",
    )
    merged = merged.merge(
        tfm[key + ["tfm_zeroshot_mae", "tfm_zeroshot_mse"]].rename(
            columns={
                "tfm_zeroshot_mae": "timesfm_mae",
                "tfm_zeroshot_mse": "timesfm_mse",
            }
        ),
        on=key,
        how="left",
    )
    merged["nosynth_vs_synth_pct"] = (merged["nosynth_mae"] / merged["synth_pretrain_mae"] - 1.0) * 100.0
    merged["nosynth_vs_timesfm_pct"] = (merged["nosynth_mae"] / merged["timesfm_mae"] - 1.0) * 100.0
    return merged


def plot_all(comp: pd.DataFrame) -> None:
    overall = pd.DataFrame(
        [
            {
                "split": "Real LOTSA+ETT",
                "nosynth": comp["nosynth_mae"].mean(),
                "synth_pretrain": comp["synth_pretrain_mae"].mean(),
                "timesfm": comp["timesfm_mae"].mean(),
            }
        ]
    )
    grouped_bar(overall, "split", "Overall Real MAE Comparison", PLOT_DIR / "overall_mae_comparison.png")

    by_h = comp.groupby("horizon", as_index=False).agg(
        nosynth=("nosynth_mae", "mean"),
        synth_pretrain=("synth_pretrain_mae", "mean"),
        timesfm=("timesfm_mae", "mean"),
    )
    line_by_horizon(by_h, "Real MAE by Horizon", PLOT_DIR / "real_mae_by_horizon.png")

    by_ds = comp.groupby("dataset", as_index=False).agg(
        nosynth=("nosynth_mae", "mean"),
        synth_pretrain=("synth_pretrain_mae", "mean"),
        timesfm=("timesfm_mae", "mean"),
    ).sort_values("nosynth")
    grouped_bar(by_ds, "dataset", "Real MAE by Dataset", PLOT_DIR / "real_mae_by_dataset.png", rotate=35)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    comp = build_comparison()
    comp.to_csv(ANALYSIS_DIR / "real_comparison.csv", index=False)
    plot_all(comp)

    summary = {
        "rows": int(len(comp)),
        "nosynth_mean_mae": float(comp["nosynth_mae"].mean()),
        "synth_pretrain_mean_mae": float(comp["synth_pretrain_mae"].mean()),
        "timesfm_mean_mae": float(comp["timesfm_mae"].mean()),
        "nosynth_wins_vs_synth_pretrain": int((comp["nosynth_mae"] < comp["synth_pretrain_mae"]).sum()),
        "nosynth_wins_vs_timesfm": int((comp["nosynth_mae"] < comp["timesfm_mae"]).sum()),
    }
    pd.DataFrame([summary]).to_csv(ANALYSIS_DIR / "real_summary.csv", index=False)
    print(f"Saved comparison tables to {ANALYSIS_DIR}")
    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
