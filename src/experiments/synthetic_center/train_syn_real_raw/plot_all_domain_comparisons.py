from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/all_domain_full_then_residual")
DEFAULT_PHASEFIX_REAL = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/real_lot_ett_single_model_phasefix/real_eval_component_mae.csv"
)
DEFAULT_F_BASE = Path(
    "/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent/results/simple_complex_on_complex_fixed_phase_scale/simple_complex_on_complex_component_mae.csv"
)
DEFAULT_NONF_BASE = Path(
    "/home/sia2/project/5.22syn_cent/train_nonF_rawtarget/results/nonfourier_single_model/nonfourier_component_mae.csv"
)
HORIZONS = [96, 192, 336, 720]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison plots for all-domain real fine-tune experiment.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phasefix_real_csv", type=Path, default=DEFAULT_PHASEFIX_REAL)
    parser.add_argument("--f_base_csv", type=Path, default=DEFAULT_F_BASE)
    parser.add_argument("--nonf_base_csv", type=Path, default=DEFAULT_NONF_BASE)
    parser.add_argument("--nosynth_root", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "horizon" in df.columns:
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    if "total_mae" in df.columns:
        df["total_mae"] = pd.to_numeric(df["total_mae"], errors="coerce")
    if "tfm_zeroshot_mae" in df.columns:
        df["tfm_zeroshot_mae"] = pd.to_numeric(df["tfm_zeroshot_mae"], errors="coerce")
    return df


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path)


def add_series(ax, df: pd.DataFrame, label: str, value_col: str = "total_mae", **style) -> None:
    if df.empty or value_col not in df.columns:
        return
    series = df.groupby("horizon")[value_col].mean().reindex(HORIZONS)
    if series.notna().any():
        ax.plot(HORIZONS, series.values, label=label, **style)


def finish(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Normalized MAE")
    ax.set_xticks(HORIZONS)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)


def save_line(out_path: Path, title: str, series: list[tuple[pd.DataFrame, str, str, dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for df, label, value_col, style in series:
        add_series(ax, df, label, value_col, **style)
    finish(ax, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def save_real_dataset_grid(
    out_path: Path,
    full: pd.DataFrame,
    residual: pd.DataFrame,
    phasefix: pd.DataFrame,
    nosynth_full: pd.DataFrame | None = None,
    nosynth_residual: pd.DataFrame | None = None,
) -> None:
    datasets = sorted(phasefix["dataset"].dropna().unique())
    ncols = 4
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        ax.set_visible(True)
        add_series(ax, full[full["dataset"] == dataset], "All-domain full", marker="o", linewidth=1.8)
        add_series(ax, residual[residual["dataset"] == dataset], "All-domain + residual", marker="D", linewidth=1.8)
        if nosynth_full is not None and not nosynth_full.empty:
            add_series(ax, nosynth_full[nosynth_full["dataset"] == dataset], "No-pretrain full", marker="v", linewidth=1.6)
        if nosynth_residual is not None and not nosynth_residual.empty:
            add_series(
                ax,
                nosynth_residual[nosynth_residual["dataset"] == dataset],
                "No-pretrain + residual",
                marker="P",
                linewidth=1.6,
            )
        add_series(ax, phasefix[phasefix["dataset"] == dataset], "Target-domain phasefix", marker="^", linewidth=1.8)
        add_series(
            ax,
            phasefix[phasefix["dataset"] == dataset],
            "TimesFM",
            "tfm_zeroshot_mae",
            color="black",
            linestyle="--",
            marker="s",
            linewidth=1.5,
        )
        ax.set_title(dataset)
        ax.set_xticks(HORIZONS)
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("Real target MAE by dataset", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def real_plots(root: Path, out_dir: Path, phasefix_csv: Path, nosynth_root: Path | None = None) -> pd.DataFrame:
    full = read_csv(root / "eval/real_lot_ett_full/real_eval_component_mae.csv")
    residual = read_csv(root / "eval/real_lot_ett_residual_extra/real_eval_component_mae.csv")
    phasefix = read_csv(phasefix_csv)
    nosynth_full = maybe_read_csv(nosynth_root / "eval/real_lot_ett_full/real_eval_component_mae.csv") if nosynth_root else pd.DataFrame()
    nosynth_residual = (
        maybe_read_csv(nosynth_root / "eval/real_lot_ett_residual_extra/real_eval_component_mae.csv") if nosynth_root else pd.DataFrame()
    )
    series = [
        (full, "All-domain full", "total_mae", {"marker": "o", "linewidth": 2.0}),
        (residual, "All-domain + residual", "total_mae", {"marker": "D", "linewidth": 2.0}),
    ]
    if not nosynth_full.empty:
        series.append((nosynth_full, "No-pretrain full", "total_mae", {"marker": "v", "linewidth": 1.8}))
    if not nosynth_residual.empty:
        series.append((nosynth_residual, "No-pretrain + residual", "total_mae", {"marker": "P", "linewidth": 1.8}))
    series.extend(
        [
            (phasefix, "Target-domain phasefix", "total_mae", {"marker": "^", "linewidth": 2.0}),
            (phasefix, "TimesFM", "tfm_zeroshot_mae", {"color": "black", "linestyle": "--", "marker": "s", "linewidth": 2.0}),
        ]
    )
    save_line(
        out_dir / "real_overall_mae_by_horizon.png",
        "Real target overall MAE",
        series,
    )
    save_real_dataset_grid(
        out_dir / "real_dataset_mae_grid_by_horizon.png",
        full,
        residual,
        phasefix,
        nosynth_full,
        nosynth_residual,
    )
    full_s = full.groupby("dataset", as_index=False)["total_mae"].mean().rename(columns={"total_mae": "all_domain_full"})
    residual_s = residual.groupby("dataset", as_index=False)["total_mae"].mean().rename(columns={"total_mae": "all_domain_residual_extra"})
    phase_s = phasefix.groupby("dataset", as_index=False).agg(
        phasefix=("total_mae", "mean"),
        timesfm=("tfm_zeroshot_mae", "mean"),
    )
    summary = full_s.merge(residual_s, on="dataset").merge(phase_s, on="dataset")
    if not nosynth_full.empty:
        ns_full_s = nosynth_full.groupby("dataset", as_index=False)["total_mae"].mean().rename(columns={"total_mae": "nosynth_full"})
        summary = summary.merge(ns_full_s, on="dataset", how="left")
    if not nosynth_residual.empty:
        ns_res_s = nosynth_residual.groupby("dataset", as_index=False)["total_mae"].mean().rename(columns={"total_mae": "nosynth_residual_extra"})
        summary = summary.merge(ns_res_s, on="dataset", how="left")
    return summary


def f_plots(root: Path, out_dir: Path, f_base_csv: Path, nosynth_root: Path | None = None) -> pd.DataFrame:
    cur = read_csv(root / "eval/synth_F/simple_on_complex_component_mae.csv")
    base = read_csv(f_base_csv)
    full = cur[cur["model"] == "all_domain_full"]
    residual = cur[cur["model"] == "all_domain_full_residual_extra"]
    ns_cur = maybe_read_csv(nosynth_root / "eval/synth_F/simple_on_complex_component_mae.csv") if nosynth_root else pd.DataFrame()
    ns_full = ns_cur[ns_cur["model"] == "all_domain_full"] if not ns_cur.empty else pd.DataFrame()
    ns_residual = ns_cur[ns_cur["model"] == "all_domain_full_residual_extra"] if not ns_cur.empty else pd.DataFrame()
    synth_only = base[base["model"] == "simple_complex_coeff_residual_tail"]
    base_tfm = base[base["model"] == "timesfm_zeroshot"]
    series = [
        (full, "All-domain full", "total_mae", {"marker": "o", "linewidth": 2.0}),
        (residual, "All-domain + residual", "total_mae", {"marker": "D", "linewidth": 2.0}),
    ]
    if not ns_full.empty:
        series.append((ns_full, "No-pretrain full", "total_mae", {"marker": "v", "linewidth": 1.8}))
    if not ns_residual.empty:
        series.append((ns_residual, "No-pretrain + residual", "total_mae", {"marker": "P", "linewidth": 1.8}))
    series.extend(
        [
            (synth_only, "F synthetic-only", "total_mae", {"marker": "^", "linewidth": 2.0}),
            (base_tfm, "TimesFM", "total_mae", {"color": "black", "linestyle": "--", "marker": "s", "linewidth": 2.0}),
        ]
    )
    save_line(
        out_dir / "synth_F_overall_mae_by_horizon.png",
        "F synthetic overall MAE",
        series,
    )
    summaries = []
    for name, df in [
        ("all_domain_full", full),
        ("all_domain_residual_extra", residual),
        ("nosynth_full", ns_full),
        ("nosynth_residual_extra", ns_residual),
        ("f_synthetic_only", synth_only),
        ("timesfm", base_tfm),
    ]:
        if df.empty:
            continue
        summaries.append({"model": name, "mean_mae": float(df["total_mae"].mean()), "n_rows": int(len(df))})
    return pd.DataFrame(summaries)


def nonf_plots(root: Path, out_dir: Path, nonf_base_csv: Path, nosynth_root: Path | None = None) -> pd.DataFrame:
    full = read_csv(root / "eval/synth_nonF_full/nonfourier_component_mae.csv")
    residual = read_csv(root / "eval/synth_nonF_residual_extra/nonfourier_component_mae.csv")
    base = read_csv(nonf_base_csv)
    ns_full = maybe_read_csv(nosynth_root / "eval/synth_nonF_full/nonfourier_component_mae.csv") if nosynth_root else pd.DataFrame()
    ns_residual = maybe_read_csv(nosynth_root / "eval/synth_nonF_residual_extra/nonfourier_component_mae.csv") if nosynth_root else pd.DataFrame()
    series = [
        (full, "All-domain full", "total_mae", {"marker": "o", "linewidth": 2.0}),
        (residual, "All-domain + residual", "total_mae", {"marker": "D", "linewidth": 2.0}),
    ]
    if not ns_full.empty:
        series.append((ns_full, "No-pretrain full", "total_mae", {"marker": "v", "linewidth": 1.8}))
    if not ns_residual.empty:
        series.append((ns_residual, "No-pretrain + residual", "total_mae", {"marker": "P", "linewidth": 1.8}))
    series.extend(
        [
            (base, "F+nonF synthetic-only", "total_mae", {"marker": "^", "linewidth": 2.0}),
            (base, "TimesFM", "tfm_zeroshot_mae", {"color": "black", "linestyle": "--", "marker": "s", "linewidth": 2.0}),
        ]
    )
    save_line(
        out_dir / "synth_nonF_overall_mae_by_horizon.png",
        "non-F synthetic overall MAE",
        series,
    )
    summaries = []
    for name, df, value_col in [
        ("all_domain_full", full, "total_mae"),
        ("all_domain_residual_extra", residual, "total_mae"),
        ("nosynth_full", ns_full, "total_mae"),
        ("nosynth_residual_extra", ns_residual, "total_mae"),
        ("F+nonF_synthetic_only", base, "total_mae"),
        ("timesfm", base, "tfm_zeroshot_mae"),
    ]:
        if df.empty:
            continue
        summaries.append({"model": name, "mean_mae": float(df[value_col].mean()), "n_rows": int(len(df))})
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.root / "eval/comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    real_summary = real_plots(args.root, out_dir, args.phasefix_real_csv, args.nosynth_root)
    f_summary = f_plots(args.root, out_dir, args.f_base_csv, args.nosynth_root)
    nonf_summary = nonf_plots(args.root, out_dir, args.nonf_base_csv, args.nosynth_root)
    real_summary.to_csv(out_dir / "real_summary_by_dataset.csv", index=False)
    f_summary.to_csv(out_dir / "synth_F_summary.csv", index=False)
    nonf_summary.to_csv(out_dir / "synth_nonF_summary.csv", index=False)
    print(f"wrote comparison summaries to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
