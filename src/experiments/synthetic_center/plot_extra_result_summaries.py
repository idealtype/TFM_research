from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


HORIZONS = [96, 192, 336, 720]
GENERATOR_ORDER = ["cyclic_spline", "sarima", "sawtooth", "daubechies", "symlet"]
STAGE_ORDER = ["stage1_S", "stage2_T_S", "stage3_T_S_R"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create extra summary plots for 5.22 synthetic-center results.")
    parser.add_argument("result_roots", nargs="+", type=Path, help="Result roots that contain nonfourier_single_model and/or real_lot_ett_single_model.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not np.isfinite(out):
        return None
    return out


def mean_by(rows: list[dict], keys: list[str], value_key: str) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(value_key))
        if value is None:
            continue
        grouped[tuple(row.get(key, "") for key in keys)].append(value)
    out = []
    for key, values in sorted(grouped.items()):
        item = {name: key[idx] for idx, name in enumerate(keys)}
        item[value_key] = float(np.mean(values))
        out.append(item)
    return out


def setup_ax(ax, title: str, ylabel: str = "MAE") -> None:
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.25)


def run_label(out_root: Path) -> str:
    name = out_root.name
    prefix = "real_lot_ett_single_model_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def metric_by_horizon(rows: list[dict], value_key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for horizon in HORIZONS:
        vals = [as_float(row.get(value_key)) for row in rows if int(row["horizon"]) == horizon]
        vals = [v for v in vals if v is not None]
        if vals:
            xs.append(horizon)
            ys.append(float(np.mean(vals)))
    return xs, ys


def compare_baselines_plot(
    rows: list[dict],
    phasefix_rows: list[dict],
    title: str,
    out_path: Path,
    current_label: str,
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs, ys = metric_by_horizon(rows, "total_mae")
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"Current: {current_label}")
    xs, ys = metric_by_horizon(phasefix_rows, "total_mae")
    if xs:
        ax.plot(xs, ys, color="#1f77b4", linestyle=":", marker="D", linewidth=2.0, label="Phasefix full fine-tune")
    xs, ys = metric_by_horizon(rows, "tfm_zeroshot_mae")
    if xs:
        ax.plot(xs, ys, color="black", linestyle="--", marker="s", linewidth=2.0, label="TimesFM zero-shot")
    setup_ax(ax, title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def line_plot(
    rows: list[dict],
    group_key: str,
    value_key: str,
    title: str,
    out_path: Path,
    preferred_order: list[str] | None = None,
    tfm_rows: list[dict] | None = None,
    phasefix_rows: list[dict] | None = None,
) -> None:
    if not rows:
        return
    groups = sorted({row[group_key] for row in rows})
    if preferred_order:
        groups = [g for g in preferred_order if g in groups] + [g for g in groups if g not in preferred_order]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for group in groups:
        xs, ys = [], []
        for horizon in HORIZONS:
            vals = [as_float(row.get(value_key)) for row in rows if row[group_key] == group and int(row["horizon"]) == horizon]
            vals = [v for v in vals if v is not None]
            if vals:
                xs.append(horizon)
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=group)
    if phasefix_rows:
        xs, ys = metric_by_horizon(phasefix_rows, "total_mae")
        if xs:
            ax.plot(xs, ys, color="#1f77b4", linestyle=":", marker="D", linewidth=2.0, label="Phasefix full fine-tune")
    if tfm_rows:
        xs, ys = metric_by_horizon(tfm_rows, "tfm_zeroshot_mae")
        if xs:
            ax.plot(xs, ys, color="black", linestyle="--", marker="s", linewidth=2.0, label="TimesFM zero-shot")
    setup_ax(ax, title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def panel_plot_by_stage(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, stage in zip(axes, STAGE_ORDER):
        stage_rows = [row for row in rows if row["stage"] == stage]
        for generator in GENERATOR_ORDER:
            xs, ys = [], []
            for horizon in HORIZONS:
                vals = [
                    as_float(row.get("total_mae"))
                    for row in stage_rows
                    if row["generator"] == generator and int(row["horizon"]) == horizon
                ]
                vals = [v for v in vals if v is not None]
                if vals:
                    xs.append(horizon)
                    ys.append(float(np.mean(vals)))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=generator)
        setup_ax(ax, stage)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
    fig.suptitle("Model MAE by generator and horizon")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_nonfourier(root: Path) -> None:
    out_root = root / "nonfourier_single_model"
    rows = read_csv(out_root / "nonfourier_component_mae.csv")
    if not rows:
        return

    panel_plot_by_stage(rows, out_root / "generator_model_mae_by_horizon_all_stages.png")
    for stage in STAGE_ORDER:
        stage_rows = [row for row in rows if row["stage"] == stage]
        if not stage_rows:
            continue
        stage_dir = out_root / stage
        line_plot(
            stage_rows,
            "generator",
            "total_mae",
            f"{stage} model MAE by generator",
            stage_dir / "generator_model_mae_by_horizon.png",
            GENERATOR_ORDER,
        )
        line_plot(
            stage_rows,
            "generator",
            "total_mae",
            f"{stage} model and TimesFM MAE by generator",
            stage_dir / "performance_by_horizon_generators_all.png",
            GENERATOR_ORDER,
            tfm_rows=stage_rows,
        )


def phasefix_csv_for(root: Path, out_root: Path) -> Path | None:
    candidates = [
        root / "real_lot_ett_single_model_phasefix" / "real_eval_component_mae.csv",
        root.parent / "real_lot_ett_single_model_phasefix" / "real_eval_component_mae.csv",
        out_root.parent / "real_lot_ett_single_model_phasefix" / "real_eval_component_mae.csv",
        out_root.parent.parent / "real_lot_ett_single_model_phasefix" / "real_eval_component_mae.csv",
    ]
    for path in candidates:
        if path.exists() and path.parent != out_root:
            return path
    return None


def attach_phasefix_tfm(rows: list[dict], phasefix_rows: list[dict]) -> list[dict]:
    tfm_by_key = {
        (row.get("dataset"), str(row.get("horizon"))): row.get("tfm_zeroshot_mae")
        for row in phasefix_rows
        if row.get("tfm_zeroshot_mae") not in {None, ""}
    }
    out = []
    for row in rows:
        item = dict(row)
        key = (item.get("dataset"), str(item.get("horizon")))
        if item.get("tfm_zeroshot_mae") in {None, ""} and key in tfm_by_key:
            item["tfm_zeroshot_mae"] = tfm_by_key[key]
        out.append(item)
    return out


def matching_phasefix_rows(rows: list[dict], phasefix_rows: list[dict]) -> list[dict]:
    keys = {(row.get("dataset"), str(row.get("horizon"))) for row in rows}
    return [row for row in phasefix_rows if (row.get("dataset"), str(row.get("horizon"))) in keys]


def plot_real_dir(root: Path, out_root: Path) -> None:
    rows = read_csv(out_root / "real_eval_component_mae.csv")
    if not rows:
        return
    label = run_label(out_root)
    for stale_name in ["dataset_model_mae_by_horizon.png", "performance_by_horizon_datasets_all.png"]:
        stale_path = out_root / stale_name
        if stale_path.exists():
            stale_path.unlink()
    phasefix_rows = []
    phasefix_csv = phasefix_csv_for(root, out_root)
    if phasefix_csv is not None:
        phasefix_rows = matching_phasefix_rows(rows, read_csv(phasefix_csv))
        rows = attach_phasefix_tfm(rows, phasefix_rows)
    compare_baselines_plot(
        rows,
        phasefix_rows,
        f"Overall MAE by horizon | current={label}",
        out_root / "current_vs_phasefix_timesfm_by_horizon.png",
        label,
    )
    line_plot(
        rows,
        "dataset",
        "total_mae",
        f"Current run MAE by dataset | {label}",
        out_root / "dataset_current_mae_by_horizon.png",
    )
    line_plot(
        rows,
        "dataset",
        "total_mae",
        f"Dataset detail with aggregate baselines | current={label}",
        out_root / "dataset_current_with_aggregate_baselines_by_horizon.png",
        tfm_rows=rows,
        phasefix_rows=phasefix_rows,
    )
    line_plot(
        rows,
        "dataset",
        "total_mae",
        f"Overall MAE by horizon | current={label}",
        out_root / "performance_by_horizon_all.png",
        tfm_rows=rows,
        phasefix_rows=phasefix_rows,
    )

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    by_phasefix_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row.get("dataset", "")].append(row)
    for row in phasefix_rows:
        by_phasefix_dataset[row.get("dataset", "")].append(row)
    for dataset, sub_rows in by_dataset.items():
        dataset_dir = out_root / dataset
        phasefix_sub_rows = by_phasefix_dataset.get(dataset, [])
        compare_baselines_plot(
            sub_rows,
            phasefix_sub_rows,
            f"{dataset} MAE by horizon | current={label}",
            dataset_dir / "current_vs_phasefix_timesfm_by_horizon.png",
            label,
        )
        compare_baselines_plot(
            sub_rows,
            phasefix_sub_rows,
            f"{dataset} MAE by horizon | current={label}",
            dataset_dir / "performance_by_horizon.png",
            label,
        )


def plot_real(root: Path) -> None:
    candidates = sorted(path for path in root.glob("real_lot_ett_single_model*") if path.is_dir())
    for out_root in candidates:
        plot_real_dir(root, out_root)


def main() -> None:
    args = parse_args()
    for root in args.result_roots:
        plot_nonfourier(root)
        plot_real(root)


if __name__ == "__main__":
    main()
