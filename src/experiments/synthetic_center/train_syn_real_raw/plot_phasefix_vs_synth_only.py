from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PHASEFIX_CSV = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/real_lot_ett_single_model_phasefix/real_eval_component_mae.csv"
)
DEFAULT_SYNTH_ONLY_CSV = Path(
    "/home/sia2/project/5.22syn_cent/train_nonF_rawtarget/results/real_lot_ett_single_model/real_eval_component_mae.csv"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PHASEFIX_CSV.parent / "comparison_vs_F+nonF"
)
HORIZONS = [96, 192, 336, 720]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot phasefix synthetic+real vs F+nonF synthetic-only real eval.")
    parser.add_argument("--phasefix_csv", type=Path, default=DEFAULT_PHASEFIX_CSV)
    parser.add_argument("--synth_only_csv", type=Path, default=DEFAULT_SYNTH_ONLY_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rows(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"domain", "dataset", "horizon", "total_mae", "tfm_zeroshot_mae"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["domain", "dataset", "frequency", "horizon", "total_mae", "tfm_zeroshot_mae"]].copy()
    out["horizon"] = out["horizon"].astype(int)
    out["total_mae"] = pd.to_numeric(out["total_mae"], errors="coerce")
    out["tfm_zeroshot_mae"] = pd.to_numeric(out["tfm_zeroshot_mae"], errors="coerce")
    out["run"] = label
    return out


def mean_by_horizon(df: pd.DataFrame, value: str) -> pd.Series:
    return df.groupby("horizon", sort=True)[value].mean().reindex(HORIZONS)


def plot_overall(phasefix: pd.DataFrame, synth: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(HORIZONS, mean_by_horizon(phasefix, "total_mae"), marker="o", linewidth=2.0, label="Synthetic+Real phasefix")
    ax.plot(HORIZONS, mean_by_horizon(synth, "total_mae"), marker="D", linewidth=2.0, label="F+nonF synthetic-only")
    ax.plot(HORIZONS, mean_by_horizon(phasefix, "tfm_zeroshot_mae"), color="black", linestyle="--", marker="s", linewidth=2.0, label="TimesFM zero-shot")
    ax.set_title("Overall MAE by horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Normalized MAE")
    ax.set_xticks(HORIZONS)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_mae_by_horizon.png", dpi=180)
    plt.close(fig)


def plot_dataset_grid(phasefix: pd.DataFrame, synth: pd.DataFrame, output_dir: Path) -> None:
    datasets = sorted(phasefix["dataset"].unique())
    ncols = 4
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        ax.set_visible(True)
        pf = phasefix[phasefix["dataset"] == dataset]
        sy = synth[synth["dataset"] == dataset]
        ax.plot(HORIZONS, mean_by_horizon(pf, "total_mae"), marker="o", linewidth=1.8, label="Synthetic+Real")
        ax.plot(HORIZONS, mean_by_horizon(sy, "total_mae"), marker="D", linewidth=1.8, label="F+nonF only")
        ax.plot(HORIZONS, mean_by_horizon(pf, "tfm_zeroshot_mae"), color="black", linestyle="--", marker="s", linewidth=1.5, label="TimesFM")
        ax.set_title(dataset)
        ax.set_xticks(HORIZONS)
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
    fig.suptitle("Dataset MAE by horizon", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "dataset_mae_grid_by_horizon.png", dpi=180)
    plt.close(fig)


def plot_each_dataset(phasefix: pd.DataFrame, synth: pd.DataFrame, output_dir: Path) -> None:
    for dataset in sorted(phasefix["dataset"].unique()):
        pf = phasefix[phasefix["dataset"] == dataset]
        sy = synth[synth["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(HORIZONS, mean_by_horizon(pf, "total_mae"), marker="o", linewidth=2.0, label="Synthetic+Real phasefix")
        ax.plot(HORIZONS, mean_by_horizon(sy, "total_mae"), marker="D", linewidth=2.0, label="F+nonF synthetic-only")
        ax.plot(HORIZONS, mean_by_horizon(pf, "tfm_zeroshot_mae"), color="black", linestyle="--", marker="s", linewidth=2.0, label="TimesFM zero-shot")
        ax.set_title(f"{dataset} MAE by horizon")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Normalized MAE")
        ax.set_xticks(HORIZONS)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"dataset_{dataset}_mae_by_horizon.png", dpi=180)
        plt.close(fig)


def write_comparison_csv(phasefix: pd.DataFrame, synth: pd.DataFrame, output_dir: Path) -> None:
    merged = phasefix.merge(
        synth[["dataset", "horizon", "total_mae"]].rename(columns={"total_mae": "synth_only_mae"}),
        on=["dataset", "horizon"],
        how="inner",
    ).rename(columns={"total_mae": "phasefix_mae", "tfm_zeroshot_mae": "timesfm_mae"})
    merged["delta_phasefix_minus_synth_only"] = merged["phasefix_mae"] - merged["synth_only_mae"]
    merged["phasefix_rel_improvement_pct"] = 100.0 * (merged["synth_only_mae"] - merged["phasefix_mae"]) / merged["synth_only_mae"]
    merged["phasefix_beats_synth_only"] = merged["phasefix_mae"] < merged["synth_only_mae"]
    merged["phasefix_beats_timesfm"] = merged["phasefix_mae"] < merged["timesfm_mae"]
    merged.to_csv(output_dir / "comparison_rows.csv", index=False)

    dataset_summary = (
        merged.groupby(["domain", "dataset"], as_index=False)
        .agg(
            phasefix_mae=("phasefix_mae", "mean"),
            synth_only_mae=("synth_only_mae", "mean"),
            timesfm_mae=("timesfm_mae", "mean"),
            delta_phasefix_minus_synth_only=("delta_phasefix_minus_synth_only", "mean"),
            phasefix_rel_improvement_pct=("phasefix_rel_improvement_pct", "mean"),
            phasefix_beats_synth_only=("phasefix_beats_synth_only", "sum"),
            phasefix_beats_timesfm=("phasefix_beats_timesfm", "sum"),
            n=("horizon", "count"),
        )
        .sort_values("delta_phasefix_minus_synth_only")
    )
    dataset_summary.to_csv(output_dir / "comparison_by_dataset.csv", index=False)

    horizon_summary = (
        merged.groupby("horizon", as_index=False)
        .agg(
            phasefix_mae=("phasefix_mae", "mean"),
            synth_only_mae=("synth_only_mae", "mean"),
            timesfm_mae=("timesfm_mae", "mean"),
            delta_phasefix_minus_synth_only=("delta_phasefix_minus_synth_only", "mean"),
            phasefix_rel_improvement_pct=("phasefix_rel_improvement_pct", "mean"),
            phasefix_beats_synth_only=("phasefix_beats_synth_only", "sum"),
            phasefix_beats_timesfm=("phasefix_beats_timesfm", "sum"),
            n=("dataset", "count"),
        )
        .sort_values("horizon")
    )
    horizon_summary.to_csv(output_dir / "comparison_by_horizon.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phasefix = load_rows(args.phasefix_csv, "Synthetic+Real phasefix")
    synth = load_rows(args.synth_only_csv, "F+nonF synthetic-only")
    plot_overall(phasefix, synth, args.output_dir)
    plot_dataset_grid(phasefix, synth, args.output_dir)
    plot_each_dataset(phasefix, synth, args.output_dir)
    write_comparison_csv(phasefix, synth, args.output_dir)
    print(f"wrote comparison plots and CSVs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
