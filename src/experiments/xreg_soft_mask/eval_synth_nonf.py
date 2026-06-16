#!/usr/bin/env python3
"""Evaluate soft_mask model on non-Fourier synthetic datasets."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = next(
    parent for parent in THIS_DIR.parents if (parent / "loader_utils.py").exists()
)
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import add_runtime_args, dataloader_kwargs, resolve_data_path, resolve_project_path  # noqa: E402
PROJECT_ROOT_4_28 = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT_4_28 / "src"

for path in [SRC_DIR, PROJECT_ROOT_4_28, THIS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    FREQ_DAYS,
    HORIZONS,
    add_error,
    build_soft_mask_basis,
    expand_bases,
    finalize_mae,
    finalize_mse,
    
    load_single_model,
    load_tfm_zeroshot_model,
    metric_accumulator,
    plot_model_vs_tfm_by_horizon,
    plot_real_comparison_grid,
    select_indices,
    
    write_csv,
    write_summary,
)


DEFAULT_CHECKPOINT_ROOT = THIS_DIR / "results"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results" / "nonfourier_synth"
DEFAULT_NONF_EVAL_ROOT = resolve_data_path("/home/sia2/project/data/synthetic_nonF/synth_eval_nonfourier")
MODEL_NAME = "soft_mask"

STAGE_CACHE_DIRS = {
    "stage1_S": "stage1_S_nonfourier_cache_10_4_8",
    "stage2_T_S": "stage2_T_S_nonfourier_cache_10_4_8",
    "stage3_T_S_R": "stage3_T_S_R_nonfourier_cache_10_4_8",
}
STAGE1_RE = re.compile(r"(?P<granularity>\w+)_seed(?P<seed>\d+)_c(?P<context_len>\d+)_h(?P<horizon>\d+)$")
COMPLEX_RE = re.compile(r"(?P<trend_level>T\d+)_(?P<granularity>\w+)_seed(?P<seed>\d+)_c(?P<context_len>\d+)_h(?P<horizon>\d+)$")


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [nonf-eval] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--nonf_eval_root", type=Path, default=DEFAULT_NONF_EVAL_ROOT)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--stages", nargs="+", choices=list(STAGE_CACHE_DIRS.keys()),
                        default=list(STAGE_CACHE_DIRS.keys()))
    parser.add_argument("--samples_per_group", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_samples_per_group", type=int, default=3)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.set_defaults(skip_tfm=True)
    parser.add_argument("--run_tfm_zeroshot", dest="skip_tfm", action="store_false",
                        help="Explicitly run TimesFM during evaluation. Default is disabled.")
    parser.add_argument("--skip_tfm", dest="skip_tfm", action="store_true",
                        help="Do not run TimesFM during evaluation. This is the default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    add_runtime_args(parser)
    return parser.parse_args()


def source_npz_path(root: Path, meta: dict) -> Path:
    stem = f"{meta['granularity']}_seed{meta['seed']}_c{meta['context_len']}_h{meta['horizon']}.npz"
    if meta["stage"] == "stage1_S":
        return root / "stage1_S_nonfourier" / meta["generator"] / "seasonal" / stem
    stem = f"{meta['trend_level']}_{stem}"
    if meta["stage"] == "stage2_T_S":
        return root / "stage2_T_S_nonfourier" / meta["generator"] / "complex" / stem
    return (
        root
        / "stage3_T_S_R_nonfourier"
        / meta["generator"]
        / meta["residual_distribution"]
        / "complex"
        / stem
    )


def load_npz_dict(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        out = {
            key: data[key].astype(np.float32)
            for key in data.files
            if key != "meta" and data[key].dtype.kind in "fiu"
        }
        out["meta"] = json.loads(str(data["meta"]))
        return out


class NonFEvalDataset(Dataset):
    def __init__(self, ds_dir: Path, meta: dict, samples_per_group: int, seed: int,
                 nonf_eval_root: Path):
        self.ds_dir = ds_dir
        self.meta = meta
        self.horizon = int(meta["horizon"])
        self.context_len = int(meta["context_len"])
        self.granularity = meta["granularity"]
        self.plot_limit = 0

        backbone_path = ds_dir / f"backbone_emb_c{self.context_len}_h{self.horizon}_stride1.pt"
        raw_path = ds_dir / f"raw_futures_h{self.horizon}.pt"
        comp_path = ds_dir / f"component_targets_h{self.horizon}.pt"

        for p in [backbone_path, raw_path, comp_path]:
            if not p.exists():
                raise FileNotFoundError(p)

        backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        comp = torch.load(comp_path, map_location="cpu", weights_only=False)

        self.embeddings = backbone["embeddings"].float()
        self.mu = backbone["mu"].float()
        self.sigma = backbone["sigma"].float()
        self.future_n = raw["futures_n"].float()
        self.trend_n = comp["trend_n"].float()
        self.seasonal_n = comp["seasonal_n"].float()
        self.residual_n = comp["residual_n"].float()
        source = load_npz_dict(source_npz_path(nonf_eval_root, meta))
        self.context = torch.from_numpy(source["signal"][:, : self.context_len]).float()

        finite = (torch.isfinite(self.embeddings).all(dim=1) &
                  torch.isfinite(self.future_n).all(dim=1))
        valid_mask = raw.get("valid_mask")
        if valid_mask is not None:
            finite = finite & valid_mask.bool()
        self.indices = select_indices(finite, self.embeddings.shape[0], samples_per_group, seed)
        if not self.indices:
            raise ValueError(f"No valid samples: {ds_dir}")

        # Soft mask: physics-only basis computed on-the-fly (ignore cached files)
        freq = self.granularity if self.granularity in FREQ_DAYS else "hourly"
        self.bases = build_soft_mask_basis(freq, self.horizon)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        return {
            "emb": self.embeddings[i],
            "future_n": self.future_n[i],
            "trend_n": self.trend_n[i],
            "seasonal_n": self.seasonal_n[i],
            "residual_n": self.residual_n[i],
            "context": self.context[i],
            "mu": self.mu[i].view(1),
            "sigma": self.sigma[i].view(1),
            "source_idx": i,
        }


def collate(batch: list[dict]) -> dict:
    keys = ["emb", "future_n", "trend_n", "seasonal_n", "residual_n", "context", "mu", "sigma"]
    out = {k: torch.stack([item[k] for item in batch]) for k in keys}
    out["source_idx"] = torch.tensor([item["source_idx"] for item in batch])
    return out


def discover_groups(nonf_eval_root: Path, horizon: int,
                    stages: list[str]) -> list[tuple[Path, dict]]:
    groups = []
    for stage in stages:
        cache_dir_name = STAGE_CACHE_DIRS.get(stage)
        if not cache_dir_name:
            continue
        root = nonf_eval_root / cache_dir_name
        if not root.exists():
            continue
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if stage == "stage1_S":
                sub = gen_dir / "seasonal"
                if not sub.exists():
                    continue
                for ds_dir in sorted(sub.iterdir()):
                    m = STAGE1_RE.fullmatch(ds_dir.name)
                    if m and int(m.group("horizon")) == horizon:
                        meta = {
                            "stage": stage,
                            "generator": gen_dir.name,
                            "trend_level": "",
                            "granularity": m.group("granularity"),
                            "seed": int(m.group("seed")),
                            "horizon": horizon,
                            "context_len": int(m.group("context_len")),
                        }
                        groups.append((ds_dir, meta))
            else:
                sub = gen_dir / "complex" if stage == "stage2_T_S" else None
                # For stage3, there's an extra residual level dir
                if stage == "stage3_T_S_R":
                    for res_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
                        sub3 = res_dir / "complex"
                        if not sub3.exists():
                            continue
                        for ds_dir in sorted(sub3.iterdir()):
                            m = COMPLEX_RE.fullmatch(ds_dir.name)
                            if m and int(m.group("horizon")) == horizon:
                                meta = {
                                    "stage": stage,
                                    "generator": gen_dir.name,
                                    "residual_distribution": res_dir.name,
                                    "trend_level": m.group("trend_level"),
                                    "granularity": m.group("granularity"),
                                    "seed": int(m.group("seed")),
                                    "horizon": horizon,
                                    "context_len": int(m.group("context_len")),
                                }
                                groups.append((ds_dir, meta))
                elif sub and sub.exists():
                    for ds_dir in sorted(sub.iterdir()):
                        m = COMPLEX_RE.fullmatch(ds_dir.name)
                        if m and int(m.group("horizon")) == horizon:
                            meta = {
                                "stage": stage,
                                "generator": gen_dir.name,
                                "residual_distribution": "",
                                "trend_level": m.group("trend_level"),
                                "granularity": m.group("granularity"),
                                "seed": int(m.group("seed")),
                                "horizon": horizon,
                                "context_len": int(m.group("context_len")),
                            }
                            groups.append((ds_dir, meta))
    return groups


@torch.no_grad()
def evaluate_group(model, tfm_model, dataset: NonFEvalDataset, batch_size: int, device: torch.device, args: argparse.Namespace) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        **dataloader_kwargs(args, device),
    )
    acc = metric_accumulator()
    trend_acc = metric_accumulator()
    seasonal_acc = metric_accumulator()
    residual_acc = metric_accumulator()
    tfm_acc = metric_accumulator()
    plot_items = []

    for batch in loader:
        emb = batch["emb"].to(device)
        future_n = batch["future_n"].to(device)
        trend_n = batch["trend_n"].to(device)
        seasonal_n = batch["seasonal_n"].to(device)
        daily, weekly, monthly, yearly = expand_bases(dataset.bases, emb.shape[0], device)
        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        add_error(acc, pred, future_n)
        add_error(trend_acc, decomp["trend"], trend_n)
        add_error(seasonal_acc, decomp["seasonal"] + decomp["residual"], seasonal_n)
        add_error(residual_acc, decomp["residual"], batch["residual_n"].to(device))
        tfm_pred_n = None
        if tfm_model is not None:
            point_forecast, _ = tfm_model.forecast(
                dataset.horizon,
                [x for x in batch["context"].detach().cpu().numpy()],
            )
            tfm_pred = torch.as_tensor(point_forecast, dtype=torch.float32, device=device)
            mu = batch["mu"].to(device)
            sigma = batch["sigma"].to(device)
            denom = torch.where(sigma >= 1e-6, sigma, torch.ones_like(sigma))
            tfm_pred_n = (tfm_pred - mu) / denom
            add_error(tfm_acc, tfm_pred_n, future_n)
        if len(plot_items) < dataset.plot_limit:
            take = min(pred.shape[0], dataset.plot_limit - len(plot_items))
            for i in range(take):
                plot_items.append({
                    "source_idx": int(batch["source_idx"][i].item()),
                    "future": future_n[i].detach().cpu(),
                    "pred": pred[i].detach().cpu(),
                    "tfm_pred": None if tfm_pred_n is None else tfm_pred_n[i].detach().cpu(),
                    "decomp": {k: decomp[k][i].detach().cpu() for k in ["trend", "seasonal", "residual"]},
                })

    return {
        "total_mae": finalize_mae(acc),
        "total_mse": finalize_mse(acc),
        "trend_mae": finalize_mae(trend_acc),
        "seasonal_mae": finalize_mae(seasonal_acc),
        "residual_mae": finalize_mae(residual_acc),
        "tfm_zeroshot_mae": finalize_mae(tfm_acc) if tfm_acc["n"] > 0 else None,
        "tfm_zeroshot_mse": finalize_mse(tfm_acc) if tfm_acc["n"] > 0 else None,
    }, plot_items


def run(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    rows = []
    ckpt_by_horizon = {}
    plotted_keys = set()

    for horizon in args.horizons:
        model, ckpt_path, _cfg = load_single_model(args.checkpoint_root, int(horizon), device)
        ckpt_by_horizon[str(horizon)] = str(ckpt_path)
        log_progress(f"h{horizon}: loaded {ckpt_path.name}")
        tfm_model = None
        if not args.skip_tfm:
            log_progress(f"h{horizon}: loading TimesFM zeroshot")
            tfm_model = load_tfm_zeroshot_model(512, int(horizon), args.hf_cache_dir)

        groups = discover_groups(args.nonf_eval_root, horizon, args.stages)
        log_progress(f"h{horizon}: {len(groups)} eval groups")

        for ds_dir, meta in groups:
            try:
                dataset = NonFEvalDataset(ds_dir, meta, args.samples_per_group,
                                          args.seed + hash(ds_dir.name) % 10000,
                                          args.nonf_eval_root)
                dataset.plot_limit = int(args.plot_samples_per_group)
            except (FileNotFoundError, ValueError) as exc:
                log_progress(f"  skip {ds_dir.name}: {exc}")
                continue

            metrics, plot_items = evaluate_group(model, tfm_model, dataset, args.batch_size, device, args)
            row = {
                "stage": meta["stage"],
                "generator": meta.get("generator", ""),
                "residual_distribution": meta.get("residual_distribution", ""),
                "composition": "T_S_R" if meta["stage"] == "stage3_T_S_R" else ("T_S" if meta["stage"] == "stage2_T_S" else "S"),
                "trend_level": meta.get("trend_level", ""),
                "seasonal_level": "nonfourier",
                "granularity": meta["granularity"],
                "horizon": int(horizon),
                "n_samples": len(dataset),
                "model": MODEL_NAME,
                **metrics,
                "checkpoint": str(ckpt_path),
            }
            rows.append(row)
            log_progress(
                f"  {meta['stage']}/{meta.get('trend_level','')} "
                f"mae={metrics['total_mae']:.4g} tfm_mae={metrics['tfm_zeroshot_mae']}"
            )
            plot_key = (meta["stage"], meta.get("generator", ""), int(horizon))
            if plot_items and plot_key not in plotted_keys:
                plotted_keys.add(plot_key)
                plot_real_comparison_grid(
                    args.results_root / meta["stage"] / meta.get("generator", "") / "plots" / (
                        f"{meta.get('generator', '')}_{meta['stage']}_h{horizon}_pair1.png"
                    ),
                    f"{meta['stage']} | {meta.get('generator', '')} | h={horizon}",
                    plot_items,
                )

        del model
        if tfm_model is not None:
            del tfm_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fieldnames = [
        "stage", "generator", "residual_distribution", "composition", "trend_level",
        "seasonal_level", "granularity",
        "horizon", "n_samples", "model", "total_mae", "total_mse",
        "trend_mae", "seasonal_mae", "residual_mae",
        "tfm_zeroshot_mae", "tfm_zeroshot_mse", "checkpoint",
    ]
    write_csv(args.results_root / "nonfourier_component_mae.csv", rows, fieldnames)
    write_csv(args.results_root / "nonf_eval.csv", rows, fieldnames)
    plot_model_vs_tfm_by_horizon(
        rows,
        args.results_root / "performance_by_horizon_all.png",
        "Non-Fourier synthetic overall MAE by horizon",
    )
    by_stage: dict[str, list[dict]] = {}
    by_group: dict[Path, list[dict]] = {}
    for row in rows:
        by_stage.setdefault(row["stage"], []).append(row)
        if row["stage"] == "stage3_T_S_R":
            subdir = args.results_root / row["stage"] / row["generator"] / row["residual_distribution"]
        else:
            subdir = args.results_root / row["stage"] / row["generator"]
        by_group.setdefault(subdir, []).append(row)
    for stage, stage_rows in by_stage.items():
        plot_model_vs_tfm_by_horizon(
            stage_rows,
            args.results_root / stage / "performance_by_horizon.png",
            f"{stage} MAE by horizon",
        )
    for subdir, sub_rows in by_group.items():
        write_csv(subdir / "component_mae.csv", sub_rows, fieldnames)
        plot_model_vs_tfm_by_horizon(
            sub_rows,
            subdir / "performance_by_horizon.png",
            f"{subdir.relative_to(args.results_root)} MAE by horizon",
        )
        write_summary(
            subdir / "summary.json",
            {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
            sub_rows,
            ["horizon", "model"],
        )
    write_summary(
        args.results_root / "nonfourier_summary.json",
        {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
        rows,
        ["stage", "generator", "granularity", "horizon", "model"],
    )
    write_summary(
        args.results_root / "nonf_summary.json",
        {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
        rows,
        ["stage", "generator", "granularity", "horizon", "model"],
    )
    log_progress(f"complete output={args.results_root}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
