#!/usr/bin/env python3
"""Build soft_mask comparison files with fine_mask base and TimesFM rows."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOFT_ROOT = Path("/home/sia2/project/5.30soft_mask/results")
FINE_BASE_ROOT = Path("/home/sia2/project/5.30fine_mask/results/syn_and_alldata")

METRIC_COLS = {
    "model",
    "total_mae",
    "total_mse",
    "trend_mae",
    "seasonal_mae",
    "residual_mae",
    "tfm_zeroshot_mae",
    "tfm_zeroshot_mse",
    "checkpoint",
}


TASKS = [
    {
        "name": "fourier",
        "soft_rel": Path("fourier_synth/fourier_synth_eval.csv"),
        "fine_rel": Path("fourier_synth/fourier_synth_eval.csv"),
        "out_rel": Path("fourier_synth/fourier_synth_eval_with_baselines.csv"),
        "summary_rel": Path("fourier_synth/fourier_synth_summary_with_baselines.json"),
        "plot_rel": Path("fourier_synth/performance_by_horizon_with_baselines.png"),
        "extra_plot_rel": Path("fourier_synth/performance_by_seasonal_level_with_baselines.png"),
        "group_fields": ["seasonal_level", "granularity", "horizon", "model"],
        "extra_group": "seasonal_level",
    },
    {
        "name": "nonfourier",
        "soft_rel": Path("nonfourier_synth/nonf_eval.csv"),
        "fine_rel": Path("nonfourier_synth/nonf_eval.csv"),
        "out_rel": Path("nonfourier_synth/nonf_eval_with_baselines.csv"),
        "summary_rel": Path("nonfourier_synth/nonfourier_summary_with_baselines.json"),
        "plot_rel": Path("nonfourier_synth/performance_by_horizon_with_baselines.png"),
        "extra_plot_rel": Path("nonfourier_synth/performance_by_stage_with_baselines.png"),
        "group_fields": ["stage", "generator", "granularity", "horizon", "model"],
        "extra_group": "stage",
    },
    {
        "name": "real",
        "soft_rel": Path("real_lot_ett/real_eval_mae.csv"),
        "fine_rel": Path("real_lot_ett/real_eval_mae.csv"),
        "out_rel": Path("real_lot_ett/real_eval_mae_with_baselines.csv"),
        "summary_rel": Path("real_lot_ett/real_eval_summary_with_baselines.json"),
        "plot_rel": Path("real_lot_ett/performance_by_horizon_with_baselines.png"),
        "extra_plot_rel": Path("real_lot_ett/performance_by_dataset_with_baselines.png"),
        "group_fields": ["domain", "dataset", "horizon", "model"],
        "extra_group": "dataset",
    },
]


def key_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return [c for c in left.columns if c in right.columns and c not in METRIC_COLS]


def add_timesfm_rows(fine: pd.DataFrame) -> pd.DataFrame:
    tfm = fine.copy()
    tfm["model"] = "timesfm_zeroshot"
    tfm["total_mae"] = tfm["tfm_zeroshot_mae"]
    tfm["total_mse"] = tfm["tfm_zeroshot_mse"]
    for col in ["trend_mae", "seasonal_mae", "residual_mae"]:
        if col in tfm.columns:
            tfm[col] = np.nan
    tfm["checkpoint"] = "google/timesfm-2.5-200m-pytorch"
    return tfm[tfm["total_mae"].notna()].copy()


def fill_soft_tfm_columns(soft: pd.DataFrame, fine: pd.DataFrame) -> pd.DataFrame:
    keys = key_columns(soft, fine)
    tfm_cols = keys + ["tfm_zeroshot_mae", "tfm_zeroshot_mse"]
    merged = soft.drop(columns=["tfm_zeroshot_mae", "tfm_zeroshot_mse"], errors="ignore").merge(
        fine[tfm_cols],
        on=keys,
        how="left",
    )
    return merged[soft.columns]


def write_summary(path: Path, rows: pd.DataFrame, group_fields: list[str]) -> None:
    metrics = [
        "total_mae",
        "total_mse",
        "trend_mae",
        "seasonal_mae",
        "residual_mae",
        "tfm_zeroshot_mae",
        "tfm_zeroshot_mse",
    ]
    available = [m for m in metrics if m in rows.columns]
    grouped = (
        rows.groupby(group_fields, dropna=False)[available]
        .mean(numeric_only=True)
        .reset_index()
    )
    payload = {
        "source": {
            "soft_mask": str(SOFT_ROOT),
            "fine_mask_base": str(FINE_BASE_ROOT),
            "timesfm_source": "tfm_zeroshot_* columns from fine_mask base evaluation",
        },
        "n_rows": int(len(rows)),
        "group_fields": group_fields,
        "group_means": json.loads(grouped.replace({np.nan: None}).to_json(orient="records")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_grouped_bar(rows: pd.DataFrame, group_col: str, path: Path, title: str) -> None:
    order = ["timesfm_zeroshot", "fine_mask_base", "soft_mask"]
    grouped = (
        rows.groupby([group_col, "model"], dropna=False)["total_mae"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index=group_col, columns="model", values="total_mae")
    pivot = pivot[[m for m in order if m in pivot.columns]]
    if group_col == "horizon":
        pivot = pivot.sort_index(key=lambda idx: idx.astype(int))
    else:
        pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(max(8.0, 0.45 * len(pivot.index)), 4.8), width=0.78)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Normalized MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    fig = ax.get_figure()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_task(task: dict) -> dict:
    soft_path = SOFT_ROOT / task["soft_rel"]
    fine_path = FINE_BASE_ROOT / task["fine_rel"]
    soft = pd.read_csv(soft_path)
    fine = pd.read_csv(fine_path)

    soft_filled = fill_soft_tfm_columns(soft, fine)
    fine_base = fine.copy()
    fine_base["model"] = "fine_mask_base"
    timesfm = add_timesfm_rows(fine)

    combined = pd.concat([soft_filled, fine_base, timesfm], ignore_index=True, sort=False)
    out_path = SOFT_ROOT / task["out_rel"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    write_summary(SOFT_ROOT / task["summary_rel"], combined, task["group_fields"])
    plot_grouped_bar(combined, "horizon", SOFT_ROOT / task["plot_rel"], f"{task['name']} MAE by horizon")
    plot_grouped_bar(
        combined,
        task["extra_group"],
        SOFT_ROOT / task["extra_plot_rel"],
        f"{task['name']} MAE by {task['extra_group']}",
    )
    return {
        "task": task["name"],
        "rows": int(len(combined)),
        "output": str(out_path),
        "summary": str(SOFT_ROOT / task["summary_rel"]),
        "plots": [str(SOFT_ROOT / task["plot_rel"]), str(SOFT_ROOT / task["extra_plot_rel"])],
    }


def main() -> None:
    outputs = [build_task(task) for task in TASKS]
    report_path = SOFT_ROOT / "baseline_comparison_outputs.json"
    report_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
