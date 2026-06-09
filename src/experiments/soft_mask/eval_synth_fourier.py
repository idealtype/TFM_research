#!/usr/bin/env python3
"""Evaluate soft_mask model on Fourier synthetic datasets (S1-S10 and SM1-SM10)."""
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
EXPERIMENTS_ROOT = THIS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import add_runtime_args, dataloader_kwargs, resolve_data_path, resolve_project_path  # noqa: E402
PROJECT_ROOT_4_28 = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT_4_28 / "src"

for path in [SRC_DIR, PROJECT_ROOT_4_28, THIS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    FAMILIES,
    FREQ_DAYS,
    HORIZONS,
    add_harmonic_activation_stats,
    add_error,
    build_soft_mask_basis,
    expand_bases,
    family_curves_from_coefficients,
    family_energy,
    family_share_from_energy,
    finalize_harmonic_activation_stats,
    finalize_mae,
    finalize_mse,
    harmonic_activation_fieldnames,
    init_harmonic_activation_accumulator,
    
    load_single_model,
    load_tfm_zeroshot_model,
    metric_accumulator,
    plot_model_vs_tfm_by_horizon,
    plot_real_comparison_grid,
    select_indices,
    soft_mask_family_contributions,
    
    write_csv,
    write_summary,
)


DEFAULT_CHECKPOINT_ROOT = THIS_DIR / "results"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results" / "fourier_synth"
MODEL_NAME = "soft_mask"

EVAL_ROOTS = [
    resolve_data_path("/home/sia2/project/data/synthetic/func_dec_syn_cent_fourier_all_eval_cache_10_4_2_8"),
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
    parser.add_argument("--gate_active_threshold", type=float, default=0.5,
                        help="Gate value threshold used for learned harmonic active-rate diagnostics.")
    parser.add_argument("--coeff_active_threshold", type=float, default=1e-6,
                        help="Absolute gated coefficient amplitude threshold used for active-rate diagnostics.")
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


def source_root_for_cache(cache_root: Path) -> Path | None:
    name = cache_root.name
    if name.startswith("func_dec_syn_cent_fourier_all_eval_cache"):
        return cache_root.parent / "func_dec_syn_cent_fourier_all_eval"
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
        coef_path = ds_dir / f"seasonal_coefficients_fine_mask_h{self.horizon}.pt"

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
        self.gt_coefficients = None
        if comp_path.exists():
            comp = torch.load(comp_path, map_location="cpu", weights_only=False)
            self.trend_n = comp["trend_n"].float()
            self.seasonal_n = comp["seasonal_n"].float()
        if coef_path.exists():
            coef_payload = torch.load(coef_path, map_location="cpu", weights_only=False)
            self.gt_coefficients = {
                family: coef_payload[f"{family}_coefficients"].float()
                for family in FAMILIES
                if f"{family}_coefficients" in coef_payload
            }

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

        # Soft mask: physics-only basis computed on-the-fly (ignore cached files)
        freq = self.granularity if self.granularity in FREQ_DAYS else "hourly"
        self.bases = build_soft_mask_basis(freq, self.horizon)

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
        if self.gt_coefficients is not None:
            for family in FAMILIES:
                if family in self.gt_coefficients:
                    item[f"gt_coef_{family}"] = self.gt_coefficients[family][i]
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
    for family in FAMILIES:
        key = f"gt_coef_{family}"
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
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


def build_plot_interpretability(
    future_n: torch.Tensor,
    pred_seasonal_n: torch.Tensor,
    pred_family_energy: dict[str, torch.Tensor],
    gt_family_energy: dict[str, torch.Tensor] | None,
    gt_seasonal_n: torch.Tensor | None,
) -> dict:
    future_energy = float((future_n.detach().float() ** 2).mean().item())
    pred_seasonal_energy = float((pred_seasonal_n.detach().float() ** 2).mean().item())
    pred_share = family_share_from_energy(pred_family_energy)
    out = {
        "pred_seasonal_future_energy_ratio": (
            None if future_energy <= 1e-12 else pred_seasonal_energy / future_energy
        ),
        "pred_family_share": pred_share,
    }
    if gt_seasonal_n is not None:
        gt_seasonal_energy = float((gt_seasonal_n.detach().float() ** 2).mean().item())
        out["gt_seasonal_future_energy_ratio"] = (
            None if future_energy <= 1e-12 else gt_seasonal_energy / future_energy
        )
    if gt_family_energy is not None:
        out["gt_family_share"] = family_share_from_energy(gt_family_energy)
    return out


@torch.no_grad()
def evaluate_group(model, tfm_model, dataset: FourierSynthDataset, batch_size: int,
                   device: torch.device, args: argparse.Namespace) -> dict:
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
    tfm_acc = metric_accumulator()
    family_sse = {family: 0.0 for family in FAMILIES}
    gt_family_energy_sum = {family: 0.0 for family in FAMILIES}
    pred_family_energy_sum = {family: 0.0 for family in FAMILIES}
    wrong_family_energy_sum = {family: 0.0 for family in FAMILIES}
    pred_seasonal_energy_sum = 0.0
    activation_acc = init_harmonic_activation_accumulator()
    plot_items = []

    for batch in loader:
        emb = batch["emb"].to(device)
        future_n = batch["future_n"].to(device)
        bases = expand_bases(dataset.bases, emb.shape[0], device)
        daily, weekly, monthly, yearly = bases
        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        add_error(acc, pred, future_n)
        tfm_pred_n = None
        pred_family_curves = soft_mask_family_contributions(decomp, bases)
        pred_family_energy = family_energy(pred_family_curves)
        batch_pred_seasonal_energy = (decomp["seasonal"].detach().float() ** 2).mean(dim=1)
        pred_seasonal_energy_sum += float(batch_pred_seasonal_energy.sum().item())
        for family in FAMILIES:
            pred_family_energy_sum[family] += float(pred_family_energy[family].sum().item())
        add_harmonic_activation_stats(
            activation_acc,
            decomp,
            dataset.granularity if dataset.granularity in FREQ_DAYS else "hourly",
            dataset.context_len,
            gate_threshold=args.gate_active_threshold,
            coeff_threshold=args.coeff_active_threshold,
        )

        gt_family_curves = None
        gt_family_energy = None
        if all(f"gt_coef_{family}" in batch for family in FAMILIES):
            gt_coefficients = {
                family: batch[f"gt_coef_{family}"].to(device)
                for family in FAMILIES
            }
            gt_family_curves = family_curves_from_coefficients(gt_coefficients, bases)
            gt_family_energy = family_energy(gt_family_curves)
            for family in FAMILIES:
                diff = pred_family_curves[family] - gt_family_curves[family]
                family_sse[family] += float((diff.detach().float() ** 2).sum().item())
                gt_energy = gt_family_energy[family]
                gt_family_energy_sum[family] += float(gt_energy.sum().item())
                gt_off = gt_energy <= 1e-10
                if bool(gt_off.any().item()):
                    wrong_family_energy_sum[family] += float(pred_family_energy[family][gt_off].sum().item())

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
                    "seasonal_families": {
                        family: pred_family_curves[family][i].detach().cpu()
                        for family in FAMILIES
                    },
                    "interpretability": build_plot_interpretability(
                        future_n[i],
                        decomp["seasonal"][i],
                        {family: pred_family_energy[family][i] for family in FAMILIES},
                        None if gt_family_energy is None else {family: gt_family_energy[family][i] for family in FAMILIES},
                        None if "seasonal_n" not in batch else batch["seasonal_n"][i].to(device),
                    ),
                })

    metrics = {
        "total_mae": finalize_mae(acc),
        "total_mse": finalize_mse(acc),
        "trend_mae": finalize_mae(trend_acc) if trend_acc["n"] > 0 else None,
        "seasonal_mae": finalize_mae(seasonal_acc) if seasonal_acc["n"] > 0 else None,
        "tfm_zeroshot_mae": finalize_mae(tfm_acc) if tfm_acc["n"] > 0 else None,
        "tfm_zeroshot_mse": finalize_mse(tfm_acc) if tfm_acc["n"] > 0 else None,
    }
    total_gt_family_energy = sum(gt_family_energy_sum.values())
    total_pred_family_energy = sum(pred_family_energy_sum.values())
    for family in FAMILIES:
        gt_energy = gt_family_energy_sum[family]
        pred_energy = pred_family_energy_sum[family]
        metrics[f"gt_explained_{family}"] = None if gt_energy <= 1e-10 else 1.0 - family_sse[family] / gt_energy
        metrics[f"gt_share_{family}"] = None if total_gt_family_energy <= 1e-10 else gt_energy / total_gt_family_energy
        metrics[f"pred_share_{family}"] = None if total_pred_family_energy <= 1e-10 else pred_energy / total_pred_family_energy
        metrics[f"wrong_family_leakage_{family}"] = (
            None if pred_seasonal_energy_sum <= 1e-10
            else wrong_family_energy_sum[family] / pred_seasonal_energy_sum
        )
    metrics.update(finalize_harmonic_activation_stats(activation_acc))
    return metrics, plot_items


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

            metrics, plot_items = evaluate_group(model, tfm_model, dataset, args.batch_size, device, args)
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
        *(f"gt_explained_{family}" for family in FAMILIES),
        *(f"gt_share_{family}" for family in FAMILIES),
        *(f"pred_share_{family}" for family in FAMILIES),
        *(f"wrong_family_leakage_{family}" for family in FAMILIES),
        *harmonic_activation_fieldnames(),
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
