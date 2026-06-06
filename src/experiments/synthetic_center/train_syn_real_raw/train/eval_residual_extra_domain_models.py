from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = next(parent for parent in THIS_DIR.parents if (parent / "loader_utils.py").exists())
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import add_runtime_args, dataloader_kwargs, resolve_data_path, resolve_project_path  # noqa: E402
EXP_DIR = THIS_DIR.parent
OLD_EXP_DIR = resolve_project_path("/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent")
PROJECT_ROOT = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT / "src"
for path in [EXP_DIR, PROJECT_ROOT, SRC_DIR, OLD_EXP_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_real_lot_ett_single_model import (  # noqa: E402
    DEFAULT_REAL_ROOT,
    RealCacheDataset,
    collate,
    dataset_result_dir,
    load_manifest,
    manifest_cache_dir,
)
from single_model_eval_common import (  # noqa: E402
    HORIZONS,
    add_error,
    expand_bases,
    finalize_mae,
    finalize_mse,
    load_single_model,
    load_tfm_zeroshot_model,
    metric_accumulator,
    plot_model_vs_tfm_by_horizon,
    plot_real_comparison_grid,
    write_csv,
    write_summary,
)


DEFAULT_CHECKPOINT_ROOT = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/full_resi_extra_train/train/residual_extra_from_full_phasefix"
)
DEFAULT_RESULTS_ROOT = Path("/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/full_resi_extra_train")
MODEL_NAME = "residual_extra_from_full_phasefix"


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [residual-extra-eval] {message}", flush=True)


def fmt_metric(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual-extra checkpoints.")
    parser.add_argument("--real_root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output_name", default="real_lot_ett_single_model")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--samples_per_dataset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_samples_per_dataset", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--skip_tfm", action="store_true")
    parser.add_argument(
        "--model_only",
        action="store_true",
        help="Save only the extra-trained model metrics. Skips TimesFM, no-residual ablation, and diagnostic stats.",
    )
    add_runtime_args(parser)
    return parser.parse_args()


@torch.no_grad()
def evaluate_dataset(
    model,
    tfm_zs,
    dataset: RealCacheDataset,
    batch_size: int,
    device: torch.device,
    plot_limit: int,
    model_only: bool,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, **dataloader_kwargs(args, device))
    full_acc = metric_accumulator()
    no_res_acc = metric_accumulator()
    tfm_acc = metric_accumulator()
    tfm_available = True
    plot_items = []
    abs_sums = {"trend": 0.0, "seasonal": 0.0, "residual": 0.0}
    std_sums = {"future": 0.0, "pred": 0.0, "trend": 0.0, "seasonal": 0.0, "residual": 0.0}
    std_n = 0
    corr_num = corr_den_r = corr_den_p = 0.0

    for batch in loader:
        emb = batch["emb"].to(device)
        future_n = batch["future_n"].to(device)
        daily, weekly, yearly = expand_bases(dataset.bases, emb.shape[0], device)
        pred, decomp = model(emb, daily, weekly, yearly)

        add_error(full_acc, pred, future_n)

        if not model_only:
            no_residual = decomp["trend"] + decomp["seasonal"]
            add_error(no_res_acc, no_residual, future_n)

            for key in ["trend", "seasonal", "residual"]:
                abs_sums[key] += float(decomp[key].detach().abs().sum().item())
                std_sums[key] += float(decomp[key].detach().float().std(dim=1).sum().item())
            std_sums["future"] += float(future_n.detach().float().std(dim=1).sum().item())
            std_sums["pred"] += float(pred.detach().float().std(dim=1).sum().item())
            std_n += int(future_n.shape[0])

            r = decomp["residual"].detach().flatten().float()
            p = pred.detach().flatten().float()
            r0 = r - r.mean()
            p0 = p - p.mean()
            corr_num += float((r0 * p0).sum().item())
            corr_den_r += float((r0 * r0).sum().item())
            corr_den_p += float((p0 * p0).sum().item())

        tfm_pred_n = None
        if model_only or tfm_zs is None:
            tfm_available = False
        else:
            contexts_np = dataset.raw_contexts_for_local_indices(batch["source_idx"])
            if contexts_np is None:
                tfm_available = False
            else:
                point_forecast, _ = tfm_zs.forecast(dataset.horizon, [x for x in contexts_np])
                tfm_pred = torch.as_tensor(point_forecast, dtype=torch.float32, device=device)
                mu = dataset.mu.index_select(0, batch["source_idx"]).to(device)
                sigma = dataset.sigma.index_select(0, batch["source_idx"]).to(device)
                denom = torch.where(sigma >= 1e-3, sigma, torch.ones_like(sigma))
                tfm_pred_n = (tfm_pred - mu) / denom
                add_error(tfm_acc, tfm_pred_n, future_n)

        if len(plot_items) < plot_limit:
            take = min(pred.shape[0], plot_limit - len(plot_items))
            for i in range(take):
                plot_items.append(
                    {
                        "future": future_n[i].detach().cpu(),
                        "pred": pred[i].detach().cpu(),
                        "tfm_pred": None if tfm_pred_n is None else tfm_pred_n[i].detach().cpu(),
                        "source_idx": int(batch["source_idx"][i].item()),
                        "decomp": {key: decomp[key][i].detach().cpu() for key in ["trend", "seasonal", "residual"]},
                    }
                )

    full_mae = finalize_mae(full_acc)
    if model_only:
        return {
            "total_mae": full_mae,
            "total_mse": finalize_mse(full_acc),
            "no_residual_mae": None,
            "no_residual_mse": None,
            "residual_gain_mae": None,
            "tfm_zeroshot_mae": None,
            "tfm_zeroshot_mse": None,
            "target_std": None,
            "pred_std_ratio": None,
            "trend_std_ratio": None,
            "seasonal_std_ratio": None,
            "residual_std_ratio": None,
            "residual_abs_share": None,
            "residual_pred_corr": None,
        }, plot_items

    no_res_mae = finalize_mae(no_res_acc)
    target_std = std_sums["future"] / max(1, std_n)
    abs_total = sum(abs_sums.values()) or 1.0
    return {
        "total_mae": full_mae,
        "total_mse": finalize_mse(full_acc),
        "no_residual_mae": no_res_mae,
        "no_residual_mse": finalize_mse(no_res_acc),
        "residual_gain_mae": no_res_mae - full_mae,
        "tfm_zeroshot_mae": None if not tfm_available or tfm_acc["n"] == 0 else finalize_mae(tfm_acc),
        "tfm_zeroshot_mse": None if not tfm_available or tfm_acc["n"] == 0 else finalize_mse(tfm_acc),
        "target_std": target_std,
        "pred_std_ratio": (std_sums["pred"] / max(1, std_n)) / max(target_std, 1e-9),
        "trend_std_ratio": (std_sums["trend"] / max(1, std_n)) / max(target_std, 1e-9),
        "seasonal_std_ratio": (std_sums["seasonal"] / max(1, std_n)) / max(target_std, 1e-9),
        "residual_std_ratio": (std_sums["residual"] / max(1, std_n)) / max(target_std, 1e-9),
        "residual_abs_share": abs_sums["residual"] / abs_total,
        "residual_pred_corr": corr_num / math.sqrt(max(corr_den_r * corr_den_p, 1e-12)),
    }, plot_items


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_root = args.results_root / args.output_name
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_filter = set(args.datasets or [])
    manifest = [
        item
        for item in load_manifest(args.real_root)
        if not dataset_filter or str(item.get("dataset") or item.get("name")) in dataset_filter
    ]
    if not manifest:
        raise FileNotFoundError("No matching evaluation datasets")
    log_progress(f"start device={device} output={out_root} datasets={len(manifest)}")

    rows: list[dict] = []
    ckpt_by_dataset_horizon: dict[str, dict[str, str]] = {}
    tfm_by_horizon = {}
    for horizon in args.horizons:
        if not args.skip_tfm and not args.model_only:
            log_progress(f"h{horizon}: loading TimesFM")
            tfm_by_horizon[horizon] = load_tfm_zeroshot_model(512, horizon, args.hf_cache_dir)
        for item_idx, item in enumerate(manifest):
            domain = str(item.get("domain") or item.get("group") or "")
            dataset_name = str(item.get("dataset") or item.get("name"))
            run_dir = args.checkpoint_root / dataset_name
            model, ckpt_path, _cfg = load_single_model(run_dir, int(horizon), device)
            ckpt_by_dataset_horizon.setdefault(dataset_name, {})[str(horizon)] = str(ckpt_path)
            cache_dir = manifest_cache_dir(args.real_root, item)
            log_progress(f"h{horizon}: [{item_idx + 1}/{len(manifest)}] evaluate {domain}/{dataset_name}")
            dataset = RealCacheDataset(
                cache_dir,
                int(horizon),
                args.samples_per_dataset,
                args.seed + item_idx,
                fallback_freq=str(item.get("freq") or item.get("frequency") or ""),
                dataset_name=dataset_name,
                real_root=args.real_root,
                hf_cache_dir=args.hf_cache_dir,
            )
            metrics, plot_items = evaluate_dataset(
                model,
                None if args.skip_tfm else tfm_by_horizon[horizon],
                dataset,
                args.batch_size,
                device,
                args.plot_samples_per_dataset,
                args.model_only,
            )
            row = {
                "domain": domain,
                "dataset": dataset_name,
                "frequency": dataset.freq,
                "horizon": int(horizon),
                "n_samples": len(dataset),
                "model": MODEL_NAME,
                "checkpoint": str(ckpt_path),
                **metrics,
            }
            rows.append(row)
            log_progress(
                f"h{horizon}: done {dataset_name} n={len(dataset)} "
                f"full_mae={fmt_metric(metrics['total_mae'])} no_res={fmt_metric(metrics['no_residual_mae'])} "
                f"gain={fmt_metric(metrics['residual_gain_mae'])} tfm_mae={fmt_metric(metrics['tfm_zeroshot_mae'])}"
            )
            dataset_out = dataset_result_dir(out_root, dataset_name)
            for plot_idx in range(0, len(plot_items), 3):
                plot_real_comparison_grid(
                    dataset_out / "plots" / f"h{horizon}_samples{plot_idx // 3 + 1}.png",
                    f"{domain}/{dataset_name} h{horizon}",
                    plot_items[plot_idx : plot_idx + 3],
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if horizon in tfm_by_horizon:
            del tfm_by_horizon[horizon]
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fieldnames = [
        "domain",
        "dataset",
        "frequency",
        "horizon",
        "n_samples",
        "model",
        "total_mae",
        "total_mse",
        "no_residual_mae",
        "no_residual_mse",
        "residual_gain_mae",
        "tfm_zeroshot_mae",
        "tfm_zeroshot_mse",
        "target_std",
        "pred_std_ratio",
        "trend_std_ratio",
        "seasonal_std_ratio",
        "residual_std_ratio",
        "residual_abs_share",
        "residual_pred_corr",
        "checkpoint",
    ]
    write_csv(out_root / "real_eval_component_mae.csv", rows, fieldnames)
    plot_model_vs_tfm_by_horizon(rows, out_root / "performance_by_horizon_all.png", "Residual-extra real eval MAE by horizon")
    write_summary(
        out_root / "real_eval_summary.json",
        {**vars(args), "checkpoint_by_dataset_horizon": ckpt_by_dataset_horizon},
        rows,
        ["domain", "dataset", "horizon", "model"],
    )

    by_dataset: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_dataset.setdefault((row["domain"], row["dataset"]), []).append(row)
    for (domain, dataset_name), sub_rows in by_dataset.items():
        dataset_out = dataset_result_dir(out_root, dataset_name)
        write_csv(dataset_out / "component_mae.csv", sub_rows, fieldnames)
        plot_model_vs_tfm_by_horizon(sub_rows, dataset_out / "performance_by_horizon.png", f"{domain}/{dataset_name} MAE by horizon")
        write_summary(
            dataset_out / "summary.json",
            {**vars(args), "checkpoint_by_dataset_horizon": ckpt_by_dataset_horizon},
            sub_rows,
            ["horizon", "model"],
        )
    log_progress(f"complete output={out_root}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
