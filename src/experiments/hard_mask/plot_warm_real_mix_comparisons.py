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


ROOT = Path("/home/sia2/project/5.30fine_mask/results/fourier_warm_real_mix")
OUT_ROOT = ROOT / "analysis_tables"
OUT = OUT_ROOT / "performance_plots"

WARM_REAL = ROOT / "real_lot_ett" / "real_eval_component_mae.csv"
WARM_FOURIER = ROOT / "fourier_synth" / "fourier_synth_eval.csv"
WARM_NONF = ROOT / "nonfourier_synth" / "nonf_eval.csv"

FINE_REAL = Path("/home/sia2/project/5.30fine_mask/results/real_lot_ett/real_eval_component_mae.csv")
FINE_FOURIER_CMP = Path("/home/sia2/project/5.30fine_mask/results/analysis_tables/fourier_comparison.csv")
FINE_NONF_CMP = Path("/home/sia2/project/5.30fine_mask/results/analysis_tables/nonfourier_comparison.csv")
ALL_DOMAIN_REAL = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/"
    "all_domain_full_then_residual/eval/real_lot_ett_residual_extra/real_eval_component_mae.csv"
)
NOSYNTH_REAL = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/"
    "nosynthpretrain/eval/real_lot_ett_residual_extra/real_eval_component_mae.csv"
)
TIMESFM_REAL = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/"
    "real_lot_ett_single_model_phasefix/real_eval_component_mae.csv"
)

COLORS = {
    "warm": "#1f77b4",
    "fine": "#2ca02c",
    "all_domain": "#7f7f7f",
    "nosynth": "#9467bd",
    "old_best": "#7f7f7f",
    "timesfm": "#d95f02",
}


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grouped_bar(df: pd.DataFrame, x_col: str, value_cols: list[tuple[str, str]],
                title: str, path: Path, rotate: int = 0) -> None:
    labels = df[x_col].astype(str).tolist()
    x = np.arange(len(labels))
    n = len(value_cols)
    width = min(0.8 / max(n, 1), 0.22)
    fig_w = max(8.5, 0.6 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    offsets = (np.arange(n) - (n - 1) / 2) * width
    for offset, (col, label) in zip(offsets, value_cols, strict=True):
        ax.bar(x + offset, df[col], width, label=label, color=COLORS.get(col, None))
    ax.set_title(title)
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _finish(fig, path)


def line_by_horizon(df: pd.DataFrame, value_cols: list[tuple[str, str]],
                    title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in value_cols:
        ax.plot(df["horizon"], df[col], marker="o", linewidth=2,
                label=label, color=COLORS.get(col, None))
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE")
    ax.set_xticks(df["horizon"].tolist())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finish(fig, path)


def build_real() -> pd.DataFrame:
    key = ["domain", "dataset", "frequency", "horizon"]
    warm = pd.read_csv(WARM_REAL)
    fine = pd.read_csv(FINE_REAL)
    all_domain = pd.read_csv(ALL_DOMAIN_REAL)
    nosynth = pd.read_csv(NOSYNTH_REAL)
    timesfm = pd.read_csv(TIMESFM_REAL)
    out = warm[key + ["n_samples", "total_mae", "total_mse", "checkpoint"]].rename(
        columns={"total_mae": "warm", "total_mse": "warm_mse", "checkpoint": "warm_checkpoint"}
    )
    out = out.merge(
        fine[key + ["total_mae", "checkpoint"]].rename(
            columns={"total_mae": "fine", "checkpoint": "fine_checkpoint"}
        ),
        on=key,
        how="inner",
    )
    out = out.merge(
        all_domain[key + ["total_mae", "checkpoint"]].rename(
            columns={"total_mae": "all_domain", "checkpoint": "all_domain_checkpoint"}
        ),
        on=key,
        how="inner",
    )
    out = out.merge(
        nosynth[key + ["total_mae", "checkpoint"]].rename(
            columns={"total_mae": "nosynth", "checkpoint": "nosynth_checkpoint"}
        ),
        on=key,
        how="inner",
    )
    out = out.merge(
        timesfm[key + ["tfm_zeroshot_mae", "tfm_zeroshot_mse"]].rename(
            columns={"tfm_zeroshot_mae": "timesfm", "tfm_zeroshot_mse": "timesfm_mse"}
        ),
        on=key,
        how="left",
    )
    out["warm_vs_fine_pct"] = (out["warm"] / out["fine"] - 1.0) * 100.0
    out["warm_vs_all_domain_pct"] = (out["warm"] / out["all_domain"] - 1.0) * 100.0
    out["warm_vs_nosynth_pct"] = (out["warm"] / out["nosynth"] - 1.0) * 100.0
    out["warm_vs_timesfm_pct"] = (out["warm"] / out["timesfm"] - 1.0) * 100.0
    out.to_csv(OUT_ROOT / "real_comparison.csv", index=False)
    return out


def build_fourier() -> pd.DataFrame:
    key = ["composition", "trend_level", "seasonal_level", "granularity", "horizon"]
    warm = pd.read_csv(WARM_FOURIER)
    fine = pd.read_csv(FINE_FOURIER_CMP)
    out = warm[key + ["n_samples", "total_mae", "total_mse", "tfm_zeroshot_mae"]].rename(
        columns={"total_mae": "warm", "total_mse": "warm_mse", "tfm_zeroshot_mae": "timesfm"}
    )
    out = out.merge(
        fine[key + ["total_mae", "old_best_mae", "tfm_zeroshot_mae", "old_best_model"]].rename(
            columns={"total_mae": "fine", "old_best_mae": "old_best", "tfm_zeroshot_mae": "timesfm_ref"}
        ),
        on=key,
        how="inner",
    )
    out["timesfm"] = out["timesfm"].fillna(out["timesfm_ref"])
    out.drop(columns=["timesfm_ref"], inplace=True)
    out["warm_vs_fine_pct"] = (out["warm"] / out["fine"] - 1.0) * 100.0
    out["warm_vs_old_best_pct"] = (out["warm"] / out["old_best"] - 1.0) * 100.0
    out["warm_vs_timesfm_pct"] = (out["warm"] / out["timesfm"] - 1.0) * 100.0
    out.to_csv(OUT_ROOT / "fourier_comparison.csv", index=False)
    return out


def build_nonf() -> pd.DataFrame:
    key = [
        "stage", "generator", "residual_distribution", "composition",
        "trend_level", "seasonal_level", "granularity", "horizon",
    ]
    warm = pd.read_csv(WARM_NONF)
    fine = pd.read_csv(FINE_NONF_CMP)
    out = warm[key + ["n_samples", "total_mae", "total_mse", "tfm_zeroshot_mae"]].rename(
        columns={"total_mae": "warm", "total_mse": "warm_mse", "tfm_zeroshot_mae": "timesfm"}
    )
    out = out.merge(
        fine[key + ["total_mae", "old_best_mae", "tfm_zeroshot_mae", "old_best_model"]].rename(
            columns={"total_mae": "fine", "old_best_mae": "old_best", "tfm_zeroshot_mae": "timesfm_ref"}
        ),
        on=key,
        how="inner",
    )
    out["timesfm"] = out["timesfm"].fillna(out["timesfm_ref"])
    out.drop(columns=["timesfm_ref"], inplace=True)
    out["warm_vs_fine_pct"] = (out["warm"] / out["fine"] - 1.0) * 100.0
    out["warm_vs_old_best_pct"] = (out["warm"] / out["old_best"] - 1.0) * 100.0
    out["warm_vs_timesfm_pct"] = (out["warm"] / out["timesfm"] - 1.0) * 100.0
    out.to_csv(OUT_ROOT / "nonfourier_comparison.csv", index=False)
    return out


def plot_all(real: pd.DataFrame, fourier: pd.DataFrame, nonf: pd.DataFrame) -> None:
    real_cols = [
        ("warm", "warm real-mix"),
        ("fine", "fine-mask full"),
        ("all_domain", "all-domain synth"),
        ("nosynth", "no synth pretrain"),
        ("timesfm", "TimesFM"),
    ]
    synth_cols = [
        ("warm", "warm real-mix"),
        ("fine", "fine-mask full"),
        ("old_best", "synth-only best"),
        ("timesfm", "TimesFM"),
    ]

    overall = pd.DataFrame([
        {
            "split": "Real",
            "warm": real["warm"].mean(),
            "fine": real["fine"].mean(),
            "all_domain": real["all_domain"].mean(),
            "nosynth": real["nosynth"].mean(),
            "timesfm": real["timesfm"].mean(),
        },
        {
            "split": "Fourier synth",
            "warm": fourier["warm"].mean(),
            "fine": fourier["fine"].mean(),
            "all_domain": np.nan,
            "nosynth": np.nan,
            "timesfm": fourier["timesfm"].mean(),
        },
        {
            "split": "non-Fourier synth",
            "warm": nonf["warm"].mean(),
            "fine": nonf["fine"].mean(),
            "all_domain": np.nan,
            "nosynth": np.nan,
            "timesfm": nonf["timesfm"].mean(),
        },
    ])
    grouped_bar(overall, "split", real_cols, "Overall MAE Comparison", OUT / "overall_mae_comparison.png")
    real_overall = pd.DataFrame([{
        "split": "Real",
        "warm": real["warm"].mean(),
        "fine": real["fine"].mean(),
        "all_domain": real["all_domain"].mean(),
        "nosynth": real["nosynth"].mean(),
        "timesfm": real["timesfm"].mean(),
    }])
    grouped_bar(
        real_overall,
        "split",
        real_cols,
        "Real Overall MAE Comparison",
        OUT / "real_overall_mae_comparison.png",
    )
    synth_overall = pd.DataFrame([
        {
            "split": "Fourier synth",
            "warm": fourier["warm"].mean(),
            "fine": fourier["fine"].mean(),
            "old_best": fourier["old_best"].mean(),
            "timesfm": fourier["timesfm"].mean(),
        },
        {
            "split": "non-Fourier synth",
            "warm": nonf["warm"].mean(),
            "fine": nonf["fine"].mean(),
            "old_best": nonf["old_best"].mean(),
            "timesfm": nonf["timesfm"].mean(),
        },
    ])
    grouped_bar(
        synth_overall,
        "split",
        synth_cols,
        "Synthetic Overall MAE Comparison",
        OUT / "synthetic_overall_mae_comparison.png",
    )

    by_h = real.groupby("horizon", as_index=False).agg(
        warm=("warm", "mean"), fine=("fine", "mean"), all_domain=("all_domain", "mean"),
        nosynth=("nosynth", "mean"), timesfm=("timesfm", "mean"),
    )
    line_by_horizon(by_h, real_cols, "Real Evaluation MAE by Horizon", OUT / "real_mae_by_horizon.png")

    by_ds = real.groupby("dataset", as_index=False).agg(
        warm=("warm", "mean"), fine=("fine", "mean"), all_domain=("all_domain", "mean"),
        nosynth=("nosynth", "mean"), timesfm=("timesfm", "mean"),
    ).sort_values("warm")
    grouped_bar(by_ds, "dataset", real_cols, "Real Evaluation MAE by Dataset", OUT / "real_mae_by_dataset.png", rotate=35)

    f_h = fourier.groupby("horizon", as_index=False).agg(
        warm=("warm", "mean"), fine=("fine", "mean"), old_best=("old_best", "mean"), timesfm=("timesfm", "mean")
    )
    line_by_horizon(f_h, synth_cols, "Fourier Synthetic MAE by Horizon", OUT / "fourier_mae_by_horizon.png")

    order = sorted(fourier["seasonal_level"].dropna().unique(), key=lambda s: int(str(s).replace("S", "")))
    f_s = fourier.groupby("seasonal_level", as_index=False).agg(
        warm=("warm", "mean"), fine=("fine", "mean"), old_best=("old_best", "mean"), timesfm=("timesfm", "mean")
    )
    f_s["seasonal_level"] = pd.Categorical(f_s["seasonal_level"], categories=order, ordered=True)
    grouped_bar(f_s.sort_values("seasonal_level"), "seasonal_level", synth_cols,
                "Fourier Synthetic MAE by Seasonal Type", OUT / "fourier_mae_by_seasonal.png")

    nf_h = nonf.groupby("horizon", as_index=False).agg(
        warm=("warm", "mean"), fine=("fine", "mean"), old_best=("old_best", "mean"), timesfm=("timesfm", "mean")
    )
    line_by_horizon(nf_h, synth_cols, "non-Fourier Synthetic MAE by Horizon", OUT / "nonfourier_mae_by_horizon.png")

    nf_s = nonf.groupby("stage", as_index=False).agg(
        warm=("warm", "mean"), fine=("fine", "mean"), old_best=("old_best", "mean"), timesfm=("timesfm", "mean")
    )
    grouped_bar(nf_s, "stage", synth_cols, "non-Fourier Synthetic MAE by Stage",
                OUT / "nonfourier_mae_by_stage.png", rotate=15)


def write_summary(real: pd.DataFrame, fourier: pd.DataFrame, nonf: pd.DataFrame) -> None:
    rows = [
        {
            "split": "real",
            "warm": real["warm"].mean(),
            "fine": real["fine"].mean(),
            "all_domain": real["all_domain"].mean(),
            "nosynth": real["nosynth"].mean(),
            "timesfm": real["timesfm"].mean(),
            "warm_wins_vs_fine": int((real["warm"] < real["fine"]).sum()),
            "warm_wins_vs_timesfm": int((real["warm"] < real["timesfm"]).sum()),
            "n": int(len(real)),
        },
        {
            "split": "fourier",
            "warm": fourier["warm"].mean(),
            "fine": fourier["fine"].mean(),
            "old_best": fourier["old_best"].mean(),
            "timesfm": fourier["timesfm"].mean(),
            "warm_wins_vs_fine": int((fourier["warm"] < fourier["fine"]).sum()),
            "warm_wins_vs_timesfm": int((fourier["warm"] < fourier["timesfm"]).sum()),
            "n": int(len(fourier)),
        },
        {
            "split": "nonfourier",
            "warm": nonf["warm"].mean(),
            "fine": nonf["fine"].mean(),
            "old_best": nonf["old_best"].mean(),
            "timesfm": nonf["timesfm"].mean(),
            "warm_wins_vs_fine": int((nonf["warm"] < nonf["fine"]).sum()),
            "warm_wins_vs_timesfm": int((nonf["warm"] < nonf["timesfm"]).sum()),
            "n": int(len(nonf)),
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.csv", index=False)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    real = build_real()
    fourier = build_fourier()
    nonf = build_nonf()
    plot_all(real, fourier, nonf)
    write_summary(real, fourier, nonf)
    print(f"Saved comparison tables to {OUT_ROOT}")
    print(f"Saved plots to {OUT}")


if __name__ == "__main__":
    main()
