#!/usr/bin/env python3
"""Build scratch warm-mix comparison plots without running TimesFM."""
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


SOFT_ROOT = Path("/home/sia2/project/5.30soft_mask/results/fourier_warm_real_mix_scratch")
BASE_ROOT = Path("/home/sia2/project/5.30soft_mask/results/syn_all_real")
TIMESFM_ROOT = Path("/home/sia2/project/5.30fine_mask/results/fourier_warm_real_mix")
OUT_ROOT = SOFT_ROOT / "analysis_tables"
PLOT_ROOT = OUT_ROOT / "performance_plots"

TASKS = {
    "real": {
        "rel": Path("real_lot_ett/real_eval_mae.csv"),
        "keys": ["domain", "dataset", "frequency", "horizon"],
        "groups": ["horizon", "dataset"],
    },
    "fourier": {
        "rel": Path("fourier_synth/fourier_synth_eval.csv"),
        "keys": ["composition", "trend_level", "seasonal_level", "granularity", "horizon"],
        "groups": ["horizon", "seasonal_level", "composition"],
    },
    "nonfourier": {
        "rel": Path("nonfourier_synth/nonf_eval.csv"),
        "keys": [
            "stage", "generator", "residual_distribution", "composition",
            "trend_level", "granularity", "horizon",
        ],
        "groups": ["horizon", "stage", "generator"],
    },
}

COLORS = {
    "warm_mix": "#1f77b4",
    "syn_all_real": "#9467bd",
    "timesfm": "#d95f02",
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def sorted_frame(df: pd.DataFrame, group: str) -> pd.DataFrame:
    out = df.copy()
    if group == "horizon":
        return out.sort_values(group)
    if group == "stage":
        order = {"stage1_S": 1, "stage2_T_S": 2, "stage3_T_S_R": 3}
        return out.assign(_sort=out[group].map(order).fillna(99)).sort_values("_sort").drop(columns=["_sort"])
    return out.sort_values(group)


def grouped_bar(df: pd.DataFrame, group: str, title: str, path: Path, rotate: int = 0) -> None:
    labels = df[group].astype(str).tolist()
    x = np.arange(len(labels))
    cols = [
        ("warm_mix", "soft warm-mix scratch"),
        ("syn_all_real", "soft syn_all_real"),
        ("timesfm", "TimesFM"),
    ]
    width = min(0.78 / len(cols), 0.24)
    fig_w = max(8.5, 0.55 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    offsets = (np.arange(len(cols)) - (len(cols) - 1) / 2) * width
    for offset, (col, label) in zip(offsets, cols, strict=True):
        if col in df.columns:
            ax.bar(x + offset, df[col], width, label=label, color=COLORS[col])
    ax.set_title(title)
    ax.set_ylabel("Normalized MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish(fig, path)


def line_by_horizon(df: pd.DataFrame, title: str, path: Path) -> None:
    cols = [
        ("warm_mix", "soft warm-mix scratch"),
        ("syn_all_real", "soft syn_all_real"),
        ("timesfm", "TimesFM"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in cols:
        if col in df.columns:
            ax.plot(df["horizon"], df[col], marker="o", linewidth=2, label=label, color=COLORS[col])
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Normalized MAE")
    ax.set_xticks(df["horizon"].tolist())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    finish(fig, path)


def attach_timesfm(rows: pd.DataFrame, tfm_source: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    tfm_cols = keys + ["tfm_zeroshot_mae", "tfm_zeroshot_mse"]
    if not set(tfm_cols).issubset(tfm_source.columns):
        rows["timesfm"] = np.nan
        rows["timesfm_mse"] = np.nan
        return rows
    tfm = tfm_source[tfm_cols].drop_duplicates(keys)
    out = rows.merge(tfm, on=keys, how="left")
    out = out.rename(columns={"tfm_zeroshot_mae": "timesfm", "tfm_zeroshot_mse": "timesfm_mse"})
    return out


def build_task(name: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    rel = cfg["rel"]
    keys = cfg["keys"]
    soft_path = SOFT_ROOT / rel
    base_path = BASE_ROOT / rel
    tfm_path = TIMESFM_ROOT / rel
    for path in [soft_path, base_path, tfm_path]:
        require(path)

    soft = pd.read_csv(soft_path)
    base = pd.read_csv(base_path)
    tfm_source = pd.read_csv(tfm_path)

    out = soft[keys + ["n_samples", "total_mae", "total_mse", "checkpoint"]].rename(
        columns={"total_mae": "warm_mix", "total_mse": "warm_mix_mse", "checkpoint": "warm_mix_checkpoint"}
    )
    out = out.merge(
        base[keys + ["total_mae", "total_mse", "checkpoint"]].rename(
            columns={"total_mae": "syn_all_real", "total_mse": "syn_all_real_mse", "checkpoint": "syn_all_real_checkpoint"}
        ),
        on=keys,
        how="inner",
    )
    out = attach_timesfm(out, tfm_source, keys)
    out["warm_vs_syn_pct"] = (out["warm_mix"] / out["syn_all_real"] - 1.0) * 100.0
    out["warm_vs_timesfm_pct"] = (out["warm_mix"] / out["timesfm"] - 1.0) * 100.0
    out.to_csv(OUT_ROOT / f"{name}_warm_vs_syn_all_real.csv", index=False)

    summary = {
        "split": name,
        "rows": int(len(out)),
        "warm_mix": float(out["warm_mix"].mean()),
        "syn_all_real": float(out["syn_all_real"].mean()),
        "timesfm": None if out["timesfm"].isna().all() else float(out["timesfm"].mean()),
        "warm_vs_syn_pct": float((out["warm_mix"].mean() / out["syn_all_real"].mean() - 1.0) * 100.0),
        "warm_vs_timesfm_pct": None if out["timesfm"].isna().all() else float(
            (out["warm_mix"].mean() / out["timesfm"].mean() - 1.0) * 100.0
        ),
        "warm_wins_vs_syn": int((out["warm_mix"] < out["syn_all_real"]).sum()),
    }
    return out, summary


def plot_task(name: str, df: pd.DataFrame, groups: list[str]) -> None:
    for group in groups:
        by_group = (
            df.groupby(group, dropna=False)[["warm_mix", "syn_all_real", "timesfm"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        by_group = sorted_frame(by_group, group)
        title = f"{name} MAE by {group}"
        path = PLOT_ROOT / f"{name}_mae_by_{group}.png"
        if group == "horizon":
            line_by_horizon(by_group, title, path)
        else:
            grouped_bar(by_group, group, title, path, rotate=35 if len(by_group) > 6 else 0)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, cfg in TASKS.items():
        df, summary = build_task(name, cfg)
        summaries.append(summary)
        plot_task(name, df, cfg["groups"])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_ROOT / "warm_mix_scratch_summary.csv", index=False)
    payload = {
        "roots": {
            "warm_mix": str(SOFT_ROOT),
            "syn_all_real": str(BASE_ROOT),
            "timesfm_source": str(TIMESFM_ROOT),
        },
        "summary": json.loads(summary_df.replace({np.nan: None}).to_json(orient="records")),
        "plot_root": str(PLOT_ROOT),
    }
    (OUT_ROOT / "warm_mix_scratch_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
