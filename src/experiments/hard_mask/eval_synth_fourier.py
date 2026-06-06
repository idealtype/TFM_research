#!/usr/bin/env python3
"""Evaluate fine_mask model on Fourier synthetic datasets (S1-S10 and SM1-SM10)."""
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
PROJECT_ROOT_4_28 = Path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT_4_28 / "src"

for path in [SRC_DIR, PROJECT_ROOT_4_28, THIS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    FREQ_DAYS,
    HORIZONS,
    add_error,
    build_fine_mask_basis,
    expand_bases,
    finalize_mae,
    finalize_mse,
    load_fine_mask_basis,
    load_single_model,
    load_tfm_zeroshot_model,
    metric_accumulator,
    plot_model_vs_tfm_by_horizon,
    plot_real_comparison_grid,
    select_indices,
    upgrade_legacy_basis,
    write_csv,
    write_summary,
)


DEFAULT_CHECKPOINT_ROOT = THIS_DIR / "results"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results" / "fourier_synth"
MODEL_NAME = "fine_mask"

EVAL_ROOTS = [
    Path("/home/sia2/project/data/synthetic/func_dec_syn_cent_complex_eval_cache_10_4_8_fixed_phase_scale"),
    Path("/home/sia2/project/data/synthetic/func_dec_syn_cent_fine_mask_eval_cache_10_4_2_8"),
]

CACHE_DIR_RE = re.compile(
    r"^(?P<composition>A\d+)_(?P<trend_level>T\d+)_(?P<seasonal_level>[A-Z]+\d+)_"
    r"(?P<granularity>\w+)_seed(?P<seed>\d+)_c(?P<context_len>\d+)_h(?P<horizon>\d+)$"
)


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [fourier-eval] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--eval_roots", type=Path, nargs="+", default=EVAL_ROOTS)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--seasonal_levels", nargs="+", default=None)
    parser.add_argument("--samples_per_group", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_samples_per_group", type=int, default=3)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--skip_tfm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    return parser.parse_args()


def source_root_for_cache(cache_root: Path) -> Path | None:
    name = cache_root.name
    if name.startswith("func_dec_syn_cent_complex_eval_cache"):
        return cache_root.parent / "func_dec_syn_cent_complex_eval_fixed_phase_scale"
    if name.startswith("func_dec_syn_cent_fine_mask_eval_cache"):
        return cache_root.parent / "func_dec_syn_cent_fine_mask_eval"
    return None


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


class FourierSynthDataset(Dataset):
    def __init__(self, ds_dir: Path, meta: dict, samples_per_group: int, seed: int):
        self.ds_dir = ds_dir
        self.meta = meta
        self.horizon = int(meta["horizon"])
        self.context_len = int(meta["context_len"])
        self.granularity = meta["granularity"]
        self.plot_limit = 0

        backbone_path = ds_dir / f"backbone_emb_c{self.context_len}_h{self.horizon}_stride1.pt"
        raw_path = ds_dir / f"raw_futures_h{self.horizon}.pt"
        comp_path = ds_dir / f"component_targets_h{self.horizon}.pt"

        for p in [backbone_path, raw_path]:
            if not p.exists():
                raise FileNotFoundError(p)

        backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)

        self.embeddings = backbone["embeddings"].float()
        self.mu = backbone["mu"].float()
        self.sigma = backbone["sigma"].float()
        self.future_n = raw["futures_n"].float()
        self.trend_n = None
        self.seasonal_n = None
        if comp_path.exists():
            comp = torch.load(comp_path, map_location="cpu", weights_only=False)
            self.trend_n = comp["trend_n"].float()
            self.seasonal_n = comp["seasonal_n"].float()

        finite = (torch.isfinite(self.embeddings).all(dim=1) &
                  torch.isfinite(self.future_n).all(dim=1))
        valid_mask = raw.get("valid_mask")
        if valid_mask is not None:
            finite = finite & valid_mask.bool()
        self.indices = select_indices(finite, self.embeddings.shape[0], samples_per_group, seed)
        if not self.indices:
            raise ValueError(f"No valid samples: {ds_dir}")

        self.context = None
        source_root = source_root_for_cache(ds_dir.parents[1])
        if source_root is not None:
            source_path = source_root / "complex" / f"{ds_dir.name}.npz"
            if source_path.exists():
                source = load_npz_dict(source_path)
                self.context = torch.from_numpy(source["signal"][:, : self.context_len]).float()

        # Load basis
        fine_mask_path = ds_dir / f"fourier_basis_fine_mask_h{self.horizon}.pt"
        legacy_path = ds_dir / f"fourier_basis_h{self.horizon}.pt"
        if fine_mask_path.exists():
            self.bases = load_fine_mask_basis(fine_mask_path)
        elif legacy_path.exists():
            self.bases = upgrade_legacy_basis(legacy_path)
        else:
            freq = self.granularity if self.granularity in FREQ_DAYS else "hourly"
            self.bases = build_fine_mask_basis(freq, self.context_len, self.horizon)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        item = {
            "emb": self.embeddings[i],
            "future_n": self.future_n[i],
            "mu": self.mu[i].view(1),
            "sigma": self.sigma[i].view(1),
            "source_idx": i,
        }
        if self.context is not None:
            item["context"] = self.context[i]
        if self.trend_n is not None:
            item["trend_n"] = self.trend_n[i]
            item["seasonal_n"] = self.seasonal_n[i]
        return item


def collate(batch: list[dict]) -> dict:
    keys = ["emb", "future_n", "mu", "sigma", "source_idx", "context"]
    out = {}
    for k in keys:
        if k in batch[0]:
            out[k] = torch.stack([item[k] for item in batch]) if k != "source_idx" else torch.tensor([item[k] for item in batch])
    for k in ["trend_n", "seasonal_n"]:
        if k in batch[0]:
            out[k] = torch.stack([item[k] for item in batch])
    return out


def discover_groups(eval_roots: list[Path], horizon: int,
                    seasonal_levels: list[str] | None) -> list[tuple[Path, dict]]:
    groups = []
    for root in eval_roots:
        if not root.exists():
            continue
        complex_dir = root / "complex"
        if not complex_dir.exists():
            continue
        for ds_dir in sorted(complex_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            match = CACHE_DIR_RE.fullmatch(ds_dir.name)
            if not match:
                continue
            meta = {
                "composition": match.group("composition"),
                "trend_level": match.group("trend_level"),
                "seasonal_level": match.group("seasonal_level"),
                "granularity": match.group("granularity"),
                "seed": int(match.group("seed")),
                "context_len": int(match.group("context_len")),
                "horizon": int(match.group("horizon")),
            }
            if int(meta["horizon"]) != horizon:
                continue
            if seasonal_levels and meta["seasonal_level"] not in seasonal_levels:
                continue
            if not (ds_dir / f"backbone_emb_c{meta['context_len']}_h{horizon}_stride1.pt").exists():
                continue
            groups.append((ds_dir, meta))
    return groups


@torch.no_grad()
def evaluate_group(model, tfm_model, dataset: FourierSynthDataset, batch_size: int,
                   device: torch.device) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    acc = metric_accumulator()
    trend_acc = metric_accumulator()
    seasonal_acc = metric_accumulator()
    tfm_acc = metric_accumulator()
    plot_items = []

    for batch in loader:
        emb = batch["emb"].to(device)
        future_n = batch["future_n"].to(device)
        daily, weekly, monthly, yearly = expand_bases(dataset.bases, emb.shape[0], device)
        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        add_error(acc, pred, future_n)
        tfm_pred_n = None
        if "trend_n" in batch:
            add_error(trend_acc, decomp["trend"], batch["trend_n"].to(device))
            add_error(seasonal_acc, decomp["seasonal"], batch["seasonal_n"].to(device))
        if tfm_model is not None and "context" in batch:
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
        "trend_mae": finalize_mae(trend_acc) if trend_acc["n"] > 0 else None,
        "seasonal_mae": finalize_mae(seasonal_acc) if seasonal_acc["n"] > 0 else None,
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

        groups = discover_groups(args.eval_roots, horizon, args.seasonal_levels)
        log_progress(f"h{horizon}: {len(groups)} eval groups")

        for ds_dir, meta in groups:
            try:
                dataset = FourierSynthDataset(ds_dir, meta, args.samples_per_group,
                                               args.seed + hash(ds_dir.name) % 10000)
                dataset.plot_limit = int(args.plot_samples_per_group)
            except (FileNotFoundError, ValueError) as exc:
                log_progress(f"  skip {ds_dir.name}: {exc}")
                continue

            metrics, plot_items = evaluate_group(model, tfm_model, dataset, args.batch_size, device)
            rows.append({
                "composition": meta["composition"],
                "trend_level": meta["trend_level"],
                "seasonal_level": meta["seasonal_level"],
                "granularity": meta["granularity"],
                "horizon": int(horizon),
                "n_samples": len(dataset),
                "model": MODEL_NAME,
                **metrics,
                "checkpoint": str(ckpt_path),
            })
            log_progress(
                f"  {meta['composition']}/{meta['trend_level']}/{meta['seasonal_level']} "
                f"mae={metrics['total_mae']:.4g} tfm_mae={metrics['tfm_zeroshot_mae']}"
            )
            plot_key = (int(horizon), meta["composition"], meta["seasonal_level"])
            if plot_items and plot_key not in plotted_keys:
                plotted_keys.add(plot_key)
                plot_real_comparison_grid(
                    args.results_root / "plots" / (
                        f"{meta['composition']}_{meta['trend_level']}_{meta['seasonal_level']}_"
                        f"{meta['granularity']}_h{horizon}.png"
                    ),
                    f"{meta['composition']} {meta['trend_level']} {meta['seasonal_level']} "
                    f"{meta['granularity']} h{horizon}",
                    plot_items,
                )

        del model
        if tfm_model is not None:
            del tfm_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fieldnames = [
        "composition", "trend_level", "seasonal_level", "granularity",
        "horizon", "n_samples", "model", "total_mae", "total_mse",
        "trend_mae", "seasonal_mae", "tfm_zeroshot_mae", "tfm_zeroshot_mse",
        "checkpoint",
    ]
    write_csv(args.results_root / "simple_on_complex_component_mae.csv", rows, fieldnames)
    write_csv(args.results_root / "fourier_synth_eval.csv", rows, fieldnames)
    plot_model_vs_tfm_by_horizon(
        rows,
        args.results_root / "performance_by_horizon_all.png",
        "Fourier synthetic overall MAE by horizon",
    )
    by_subdir: dict[Path, list[dict]] = {}
    for seasonal_level in sorted({row["seasonal_level"] for row in rows}):
        sub_rows = [row for row in rows if row["seasonal_level"] == seasonal_level]
        level_dir = args.results_root / seasonal_level
        write_csv(level_dir / "component_mae.csv", sub_rows, fieldnames)
        plot_model_vs_tfm_by_horizon(
            sub_rows,
            level_dir / "performance_by_horizon.png",
            f"{seasonal_level} MAE by horizon",
        )
        write_summary(
            level_dir / "summary.json",
            {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
            sub_rows,
            ["horizon", "model"],
        )
    for row in rows:
        subdir = args.results_root / row["composition"] / row["seasonal_level"]
        by_subdir.setdefault(subdir, []).append(row)
    for subdir, sub_rows in by_subdir.items():
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
        args.results_root / "simple_on_complex_summary.json",
        {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
        rows,
        ["seasonal_level", "granularity", "horizon", "model"],
    )
    write_summary(
        args.results_root / "fourier_synth_summary.json",
        {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
        rows,
        ["seasonal_level", "granularity", "horizon", "model"],
    )
    log_progress(f"complete output={args.results_root}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
