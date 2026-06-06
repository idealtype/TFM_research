from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = next(parent for parent in THIS_DIR.parents if (parent / "loader_utils.py").exists())
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import add_runtime_args, dataloader_kwargs, resolve_data_path, resolve_project_path  # noqa: E402
OLD_EXP_DIR = resolve_project_path("/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent")
PROJECT_ROOT = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT / "src"
for path in [PROJECT_ROOT, SRC_DIR, OLD_EXP_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_real_lot_ett_single_model import (  # noqa: E402
    DEFAULT_REAL_ROOT,
    RealCacheDataset,
    collate,
    dataset_result_dir,
    evaluate_dataset,
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


DEFAULT_CHECKPOINT_ROOT = THIS_DIR / "results" / "all_domain_full_then_residual" / "train" / "all_domain_full_residual_extra"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results" / "all_domain_full_then_residual" / "eval"
MODEL_NAME = "all_domain_global"


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [global-real-eval] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one global all-domain FuncDec checkpoint on real targets.")
    parser.add_argument("--real_root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output_name", default="real_lot_ett")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--samples_per_dataset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_samples_per_dataset", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--skip_tfm", action="store_true")
    add_runtime_args(parser)
    return parser.parse_args()


@torch.no_grad()
def evaluate_without_tfm(model, dataset: RealCacheDataset, batch_size: int, device: torch.device, plot_limit: int, args: argparse.Namespace):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, **dataloader_kwargs(args, device))
    acc = metric_accumulator()
    plot_items = []
    for batch in loader:
        emb = batch["emb"].to(device)
        future_n = batch["future_n"].to(device)
        daily, weekly, yearly = expand_bases(dataset.bases, emb.shape[0], device)
        pred, decomp = model(emb, daily, weekly, yearly)
        add_error(acc, pred, future_n)
        if len(plot_items) < plot_limit:
            take = min(pred.shape[0], plot_limit - len(plot_items))
            for i in range(take):
                plot_items.append(
                    {
                        "future": future_n[i].detach().cpu(),
                        "pred": pred[i].detach().cpu(),
                        "tfm_pred": None,
                        "source_idx": int(batch["source_idx"][i].item()),
                        "decomp": {key: decomp[key][i].detach().cpu() for key in ["trend", "seasonal", "residual"]},
                    }
                )
    return {
        "total_mae": finalize_mae(acc),
        "total_mse": finalize_mse(acc),
        "tfm_zeroshot_mae": None,
        "tfm_zeroshot_mse": None,
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
    rows = []
    ckpt_by_horizon = {}
    tfm_by_horizon = {}
    log_progress(f"start device={device} output={out_root} datasets={len(manifest)}")
    for horizon in args.horizons:
        model, ckpt_path, _cfg = load_single_model(args.checkpoint_root, int(horizon), device)
        ckpt_by_horizon[str(horizon)] = str(ckpt_path)
        if not args.skip_tfm:
            log_progress(f"h{horizon}: loading TimesFM")
            tfm_by_horizon[horizon] = load_tfm_zeroshot_model(512, horizon, args.hf_cache_dir)
        for item_idx, item in enumerate(manifest):
            domain = str(item.get("domain") or item.get("group") or "")
            dataset_name = str(item.get("dataset") or item.get("name"))
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
            if args.skip_tfm:
                metrics, plot_items = evaluate_without_tfm(model, dataset, args.batch_size, device, args.plot_samples_per_dataset, args)
            else:
                metrics, plot_items = evaluate_dataset(
                    model, tfm_by_horizon[horizon], dataset, args.batch_size, device, args.plot_samples_per_dataset
                )
            rows.append(
                {
                    "domain": domain,
                    "dataset": dataset_name,
                    "frequency": dataset.freq,
                    "horizon": int(horizon),
                    "n_samples": len(dataset),
                    "model": MODEL_NAME,
                    "total_mae": metrics["total_mae"],
                    "total_mse": metrics["total_mse"],
                    "tfm_zeroshot_mae": metrics["tfm_zeroshot_mae"],
                    "tfm_zeroshot_mse": metrics["tfm_zeroshot_mse"],
                    "trend_mae": None,
                    "seasonal_mae": None,
                    "residual_mae": None,
                    "checkpoint": str(ckpt_path),
                }
            )
            log_progress(f"h{horizon}: done {dataset_name} model_mae={metrics['total_mae']:.6g}")
            dataset_out = dataset_result_dir(out_root, dataset_name)
            for plot_idx in range(0, len(plot_items), 3):
                plot_real_comparison_grid(
                    dataset_out / "plots" / f"h{horizon}_samples{plot_idx // 3 + 1}.png",
                    f"{domain}/{dataset_name} h{horizon}",
                    plot_items[plot_idx : plot_idx + 3],
                )
        del model
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
        "tfm_zeroshot_mae",
        "tfm_zeroshot_mse",
        "trend_mae",
        "seasonal_mae",
        "residual_mae",
        "checkpoint",
    ]
    write_csv(out_root / "real_eval_component_mae.csv", rows, fieldnames)
    plot_model_vs_tfm_by_horizon(rows, out_root / "performance_by_horizon_all.png", "Global all-domain real eval MAE")
    write_summary(out_root / "real_eval_summary.json", {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon}, rows, ["domain", "dataset", "horizon", "model"])

    by_dataset: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_dataset.setdefault((row["domain"], row["dataset"]), []).append(row)
    for (domain, dataset_name), sub_rows in by_dataset.items():
        dataset_out = dataset_result_dir(out_root, dataset_name)
        write_csv(dataset_out / "component_mae.csv", sub_rows, fieldnames)
        plot_model_vs_tfm_by_horizon(sub_rows, dataset_out / "performance_by_horizon.png", f"{domain}/{dataset_name} MAE")
        write_summary(dataset_out / "summary.json", {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon}, sub_rows, ["horizon", "model"])
    log_progress(f"complete output={out_root}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
