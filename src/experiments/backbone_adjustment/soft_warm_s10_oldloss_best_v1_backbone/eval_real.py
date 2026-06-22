#!/usr/bin/env python3
"""Evaluate soft_mask model on real LOTSA + ETT datasets."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = next(parent for parent in THIS_DIR.parents if (parent / "loader_utils.py").exists())
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import add_runtime_args, dataloader_kwargs, resolve_data_path, resolve_project_path  # noqa: E402
PROJECT_ROOT_4_28 = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT_4_28 / "src"
OLD_EXP_DIR = PROJECT_ROOT_4_28 / "basis_dec" / "experiment" / "func_dec_syn_cent"
DATA_LOTSA_DIR = resolve_data_path("/home/sia2/project/data/data_lotsa")

for path in [str(DATA_LOTSA_DIR), OLD_EXP_DIR, SRC_DIR, PROJECT_ROOT_4_28, THIS_DIR]:
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
from datasets import load_dataset  # noqa: E402
try:  # noqa: E402
    from eval_real_lot_ett_single_model import target_values  # type: ignore
except ModuleNotFoundError:  # noqa: E402
    def target_values(row: dict, variate_idx: int = 0) -> np.ndarray:
        target = row["target"]
        if len(target) == 0:
            return np.asarray([], dtype=np.float32)
        first = target[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            if len(target) <= variate_idx:
                raise ValueError(f"target has no variate index {variate_idx}")
            values = target[variate_idx]
        else:
            values = target
        return np.asarray(values, dtype=np.float32)


DEFAULT_REAL_ROOT = resolve_data_path("/home/sia2/project/data/real_eval_lot_ett")
DEFAULT_CHECKPOINT_ROOT = THIS_DIR / "results"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results" / "real_lot_ett"
DEFAULT_TIMESFM_METRICS_CSV = Path(
    "/home/sia2/project/5.30fine_mask/results/syn_and_alldata/real_lot_ett/real_eval_mae.csv"
)
MODEL_NAME = "soft_mask"


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [real-eval] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--samples_per_dataset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--plot_samples_per_dataset", type=int, default=3)
    parser.add_argument("--gate_active_threshold", type=float, default=0.5,
                        help="Gate value threshold used for learned harmonic active-rate diagnostics.")
    parser.add_argument("--coeff_active_threshold", type=float, default=1e-6,
                        help="Absolute gated coefficient amplitude threshold used for active-rate diagnostics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--timesfm_metrics_csv", type=Path, default=DEFAULT_TIMESFM_METRICS_CSV,
                        help="CSV with precomputed tfm_zeroshot_* columns to merge for plots/summaries. Use 'none' to disable.")
    parser.set_defaults(skip_tfm=True)
    parser.add_argument("--run_tfm_zeroshot", dest="skip_tfm", action="store_false",
                        help="Explicitly run TimesFM during evaluation. Default is disabled.")
    parser.add_argument("--skip_tfm", dest="skip_tfm", action="store_true",
                        help="Do not run TimesFM during evaluation. This is the default.")
    add_runtime_args(parser)
    return parser.parse_args()


def validate_same_rows(path_a: Path | str, name_a: str, tensor_a: torch.Tensor,
                       path_b: Path | str, name_b: str, tensor_b: torch.Tensor) -> None:
    rows_a = int(tensor_a.shape[0])
    rows_b = int(tensor_b.shape[0])
    if rows_a != rows_b:
        raise ValueError(
            f"Row mismatch between cached tensors: {name_a} rows={rows_a} ({path_a}) "
            f"vs {name_b} rows={rows_b} ({path_b}). "
            "This usually means a partial/stale v1 cache mixed compact backbone files "
            "with full-size target files. Reuse a matching compact cache or recache this dataset."
        )


class FineMaskRealDataset(Dataset):
    def __init__(self, cache_dir: Path, horizon: int, samples_per_dataset: int, seed: int,
                 fallback_freq: str = "", dataset_name: str = "", real_root: Path = DEFAULT_REAL_ROOT,
                 hf_cache_dir: str | None = None):
        self.cache_dir = cache_dir
        self.real_root = real_root
        self.dataset_name = dataset_name
        self.hf_cache_dir = hf_cache_dir
        self.horizon = int(horizon)

        backbone_paths = sorted(cache_dir.glob("backbone_emb*.pt"))
        future_paths = sorted(cache_dir.glob(f"futures*_h{self.horizon}_*.pt"))
        if not backbone_paths or not future_paths:
            raise FileNotFoundError(
                f"Missing backbone or future files in {cache_dir} for h{horizon}"
            )
        self.backbone_path = backbone_paths[0]
        self.future_path = future_paths[0]

        self.backbone = torch.load(self.backbone_path, map_location="cpu", weights_only=False)
        self.futures = torch.load(self.future_path, map_location="cpu", weights_only=False)
        self.embeddings = self.backbone["embeddings"].float()
        self.mu = self.backbone["mu"].float()
        self.sigma = self.backbone["sigma"].float()
        self.future_n = self.futures["futures_n"].float()
        validate_same_rows(self.backbone_path, "embeddings", self.embeddings,
                           self.future_path, "futures_n", self.future_n)
        validate_same_rows(self.backbone_path, "embeddings", self.embeddings,
                           self.backbone_path, "mu", self.mu)
        validate_same_rows(self.backbone_path, "embeddings", self.embeddings,
                           self.backbone_path, "sigma", self.sigma)
        self.context_len = int(self.backbone.get("context_len", 512))
        self.freq = str(
            self.backbone.get("frequency")
            or self.backbone.get("freq")
            or self.futures.get("frequency")
            or self.futures.get("freq")
            or fallback_freq
            or "hourly"
        )

        valid_mask = self.futures.get("valid_mask")
        finite = torch.isfinite(self.embeddings).all(dim=1) & torch.isfinite(self.future_n).all(dim=1)
        if valid_mask is not None:
            validate_same_rows(self.backbone_path, "embeddings", self.embeddings,
                               self.future_path, "valid_mask", valid_mask)
            finite = finite & valid_mask.bool()
        self.indices = select_indices(finite, self.embeddings.shape[0], samples_per_dataset, seed)
        if not self.indices:
            raise ValueError(f"No valid samples: {cache_dir} h{horizon}")

        # Soft mask: always compute physics-only basis on-the-fly (ignore cached basis files)
        freq_key = self.freq if self.freq in FREQ_DAYS else "hourly"
        self.bases = build_soft_mask_basis(freq_key, self.horizon)

        self._raw_df = None
        self._hf_dataset = None
        self._source_backbone = None
        self._sample_indices = None
        self._cloudops_index = None
        self._raw_contexts = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        return {
            "emb": self.embeddings[i],
            "future_n": self.future_n[i],
            "source_idx": i,
        }

    def _load_raw_df(self):
        if self._raw_df is None:
            raw_path = self.cache_dir / "raw.parquet"
            if not raw_path.exists():
                return None
            self._raw_df = pd.read_parquet(raw_path)
        return self._raw_df

    def _load_raw_contexts(self):
        if self._raw_contexts is None:
            path = self.cache_dir / f"raw_contexts_c{self.context_len}.pt"
            if not path.exists():
                return None
            payload = torch.load(path, map_location="cpu", weights_only=False)
            contexts = payload["contexts"] if isinstance(payload, dict) else payload
            self._raw_contexts = contexts.float() if torch.is_tensor(contexts) else torch.as_tensor(contexts).float()
        return self._raw_contexts

    def _load_hf_dataset(self):
        if self._hf_dataset is None:
            try:
                self._hf_dataset = load_dataset(
                    "Salesforce/lotsa_data",
                    self.dataset_name,
                    split="train",
                    streaming=False,
                    cache_dir=self.hf_cache_dir,
                )
            except Exception as exc:
                log_progress(
                    f"  TimesFM context unavailable for {self.dataset_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return None
        return self._hf_dataset

    def _load_sample_indices(self):
        if self._sample_indices is None:
            path = self.cache_dir / "sample_indices.pt"
            if not path.exists():
                return None
            self._sample_indices = torch.load(path, map_location="cpu", weights_only=False)
        return self._sample_indices

    def _load_source_backbone(self, source_backbone_path: Path):
        if self._source_backbone is None:
            self._source_backbone = torch.load(source_backbone_path, map_location="cpu", weights_only=False)
        return self._source_backbone

    def _load_cloudops_index(self):
        if self._cloudops_index is None:
            path = self.real_root / "cloudops_index.parquet"
            if not path.exists():
                return None
            self._cloudops_index = pd.read_parquet(path)
        return self._cloudops_index

    def raw_contexts_for_local_indices(self, local_indices: torch.Tensor) -> np.ndarray | None:
        raw_contexts = self._load_raw_contexts()
        if raw_contexts is not None:
            if int(raw_contexts.shape[0]) < int(local_indices.max().item()) + 1:
                return None
            return raw_contexts.index_select(0, local_indices.long()).numpy()

        raw_df = self._load_raw_df()
        if raw_df is not None:
            contexts = []
            numeric_cols = [c for c in raw_df.columns if c != "date"]
            if not numeric_cols:
                return None
            col_ids = self.backbone.get("col_ids", numeric_cols)
            win_starts = self.backbone.get("win_starts")
            if win_starts is None:
                return None
            for local_idx in local_indices.tolist():
                col = col_ids[int(local_idx)]
                if col not in raw_df.columns:
                    col = numeric_cols[int(local_idx) % len(numeric_cols)]
                start = int(win_starts[int(local_idx)])
                values = raw_df[col].iloc[start: start + self.context_len].to_numpy(dtype=np.float32)
                if len(values) != self.context_len:
                    return None
                contexts.append(values)
            return np.stack(contexts, axis=0)

        sample_info = self._load_sample_indices()
        if sample_info is not None:
            source_indices = sample_info["source_indices"].long()
            source_backbone = self._load_source_backbone(Path(sample_info["source_backbone"]))
            series_ids = source_backbone["series_ids"]
            win_starts = source_backbone["win_starts"]
            dataset = self._load_hf_dataset()
            if dataset is None:
                return None
            series_cache = {}
            contexts = []
            for local_idx in local_indices.tolist():
                source_idx = int(source_indices[int(local_idx)])
                series_id = int(series_ids[source_idx])
                if series_id not in series_cache:
                    series_cache[series_id] = target_values(dataset[series_id])
                start = int(win_starts[source_idx])
                values = series_cache[series_id][start: start + self.context_len]
                if len(values) != self.context_len:
                    return None
                contexts.append(values.astype(np.float32, copy=False))
            return np.stack(contexts, axis=0)

        cloudops_index = self._load_cloudops_index()
        if cloudops_index is not None and self.dataset_name == "alibaba_cluster_trace_2018":
            dataset = self._load_hf_dataset()
            if dataset is None:
                return None
            series_cache = {}
            contexts = []
            for local_idx in local_indices.tolist():
                row = cloudops_index.iloc[int(local_idx)]
                series_id = int(row["series_id"])
                if series_id not in series_cache:
                    series_cache[series_id] = target_values(dataset[series_id])
                start = int(row["win_start"])
                values = series_cache[series_id][start: start + self.context_len]
                if len(values) != self.context_len:
                    return None
                contexts.append(values.astype(np.float32, copy=False))
            return np.stack(contexts, axis=0)
        return None


def collate(batch: list[dict]) -> dict:
    return {
        "emb": torch.stack([item["emb"] for item in batch]),
        "future_n": torch.stack([item["future_n"] for item in batch]),
        "source_idx": torch.tensor([item["source_idx"] for item in batch]),
    }


def load_manifest(real_root: Path) -> list[dict]:
    import json
    manifest_path = real_root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "datasets" in payload:
            return list(payload["datasets"])
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported manifest format: {manifest_path}")
    # Auto-discover from subdirectories
    items = []
    for d in sorted(real_root.iterdir()):
        if d.is_dir() and any(d.glob("backbone_emb*.pt")):
            items.append({"dataset": d.name, "domain": "", "cache_dir": str(d), "freq": ""})
    return items


def manifest_cache_dir(real_root: Path, item: dict) -> Path:
    if "cache_dir" in item:
        return resolve_data_path(item["cache_dir"])
    if "output_dir" in item:
        base = resolve_data_path(item["output_dir"])
    else:
        domain = item.get("domain") or item.get("group")
        name = item.get("dataset") or item.get("name")
        base = real_root / str(domain) / str(name)
    if (base / "cache").exists():
        return base / "cache"
    return base


def build_real_plot_interpretability(
    future_n: torch.Tensor,
    pred_seasonal_n: torch.Tensor,
    pred_family_energy: dict[str, torch.Tensor],
) -> dict:
    future_energy = float((future_n.detach().float() ** 2).mean().item())
    pred_seasonal_energy = float((pred_seasonal_n.detach().float() ** 2).mean().item())
    return {
        "pred_seasonal_future_energy_ratio": (
            None if future_energy <= 1e-12 else pred_seasonal_energy / future_energy
        ),
        "pred_family_share": family_share_from_energy(pred_family_energy),
    }


@torch.no_grad()
def evaluate_dataset(model, tfm_model, dataset: FineMaskRealDataset, batch_size: int,
                     device: torch.device, plot_limit: int, args: argparse.Namespace):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        **dataloader_kwargs(args, device),
    )
    acc = metric_accumulator()
    tfm_acc = metric_accumulator()
    no_res_acc = metric_accumulator()
    residual_sum = 0.0
    residual_sumsq = 0.0
    residual_n = 0
    residual_abs_sum = 0.0
    pred_abs_sum = 0.0
    pred_family_energy_sum = {family: 0.0 for family in FAMILIES}
    activation_acc = init_harmonic_activation_accumulator()
    plot_items = []

    for batch in loader:
        emb = batch["emb"].to(device)
        future_n_dev = batch["future_n"].to(device)
        bases = expand_bases(dataset.bases, emb.shape[0], device)
        daily, weekly, monthly, yearly = bases
        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        add_error(acc, pred, future_n_dev)
        no_residual_pred = decomp["trend"] + decomp["seasonal"]
        add_error(no_res_acc, no_residual_pred, future_n_dev)
        pred_family_curves = soft_mask_family_contributions(decomp, bases)
        pred_family_energy = family_energy(pred_family_curves)
        for family in FAMILIES:
            pred_family_energy_sum[family] += float(pred_family_energy[family].sum().item())
        add_harmonic_activation_stats(
            activation_acc,
            decomp,
            dataset.freq if dataset.freq in FREQ_DAYS else "hourly",
            dataset.context_len,
            gate_threshold=args.gate_active_threshold,
            coeff_threshold=args.coeff_active_threshold,
        )

        residual = decomp["residual"].detach().float()
        pred_detached = pred.detach().float()
        residual_sum += float(residual.sum().item())
        residual_sumsq += float((residual ** 2).sum().item())
        residual_n += int(residual.numel())
        residual_abs_sum += float(residual.abs().sum().item())
        pred_abs_sum += float(pred_detached.abs().sum().item())

        if tfm_model is not None:
            contexts_np = dataset.raw_contexts_for_local_indices(batch["source_idx"])
            if contexts_np is not None:
                point_forecast, _ = tfm_model.forecast(dataset.horizon, [x for x in contexts_np], dataset.freq)
                tfm_pred = torch.as_tensor(point_forecast, dtype=torch.float32, device=device)
                mu = dataset.mu.index_select(0, batch["source_idx"]).to(device)
                sigma = dataset.sigma.index_select(0, batch["source_idx"]).to(device)
                denom = torch.where(sigma >= 1e-3, sigma, torch.ones_like(sigma))
                tfm_pred_n = (tfm_pred - mu) / denom
                add_error(tfm_acc, tfm_pred_n, future_n_dev)
            else:
                tfm_pred_n = None
        else:
            tfm_pred_n = None

        if len(plot_items) < plot_limit:
            take = min(pred.shape[0], plot_limit - len(plot_items))
            for i in range(take):
                plot_items.append({
                    "future": batch["future_n"][i].cpu(),
                    "pred": pred[i].detach().cpu(),
                    "tfm_pred": None if tfm_pred_n is None else tfm_pred_n[i].detach().cpu(),
                    "source_idx": int(batch["source_idx"][i].item()),
                    "decomp": {k: decomp[k][i].detach().cpu() for k in ["trend", "seasonal", "residual"]},
                    "seasonal_families": {
                        family: pred_family_curves[family][i].detach().cpu()
                        for family in FAMILIES
                    },
                    "interpretability": build_real_plot_interpretability(
                        future_n_dev[i],
                        decomp["seasonal"][i],
                        {family: pred_family_energy[family][i] for family in FAMILIES},
                    ),
                })

    total_mae = finalize_mae(acc)
    no_residual_mae = finalize_mae(no_res_acc)
    residual_mean = residual_sum / max(1, residual_n)
    residual_var = residual_sumsq / max(1, residual_n) - residual_mean ** 2
    residual_std = max(0.0, residual_var) ** 0.5
    residual_abs_mean = residual_abs_sum / max(1, residual_n)
    total_pred_abs_mean = pred_abs_sum / max(1, residual_n)

    metrics = {
        "total_mae": total_mae,
        "total_mse": finalize_mse(acc),
        "no_residual_mae": no_residual_mae,
        "residual_gain": no_residual_mae - total_mae,
        "residual_std": residual_std,
        "total_pred_abs_mean": total_pred_abs_mean,
        "residual_abs_mean": residual_abs_mean,
        "residual_total_abs_ratio": residual_abs_mean / (total_pred_abs_mean + 1e-8),
        "tfm_zeroshot_mae": None if tfm_acc["n"] == 0 else finalize_mae(tfm_acc),
        "tfm_zeroshot_mse": None if tfm_acc["n"] == 0 else finalize_mse(tfm_acc),
    }
    total_pred_family_energy = sum(pred_family_energy_sum.values())
    for family in FAMILIES:
        metrics[f"pred_share_{family}"] = (
            None if total_pred_family_energy <= 1e-10
            else pred_family_energy_sum[family] / total_pred_family_energy
        )
    metrics.update(finalize_harmonic_activation_stats(activation_acc))
    return metrics, plot_items


def dataset_result_dir(out_root: Path, dataset_name: str) -> Path:
    return out_root / dataset_name.replace("/", "_")


def attach_precomputed_timesfm(rows: list[dict], metrics_csv: Path | None) -> list[dict]:
    if not rows or metrics_csv is None or str(metrics_csv).lower() == "none":
        return rows
    if not metrics_csv.exists():
        log_progress(f"TimesFM metrics CSV not found, leaving tfm columns empty: {metrics_csv}")
        return rows

    source = pd.read_csv(metrics_csv)
    required = {"tfm_zeroshot_mae", "tfm_zeroshot_mse"}
    if not required.issubset(source.columns):
        log_progress(f"TimesFM metrics CSV has no tfm_zeroshot columns: {metrics_csv}")
        return rows

    row_df = pd.DataFrame(rows)
    key_candidates = [
        ["domain", "dataset", "frequency", "horizon"],
        ["domain", "dataset", "horizon"],
        ["dataset", "frequency", "horizon"],
        ["dataset", "horizon"],
    ]
    keys = next(
        (cols for cols in key_candidates if all(c in row_df.columns and c in source.columns for c in cols)),
        None,
    )
    if keys is None:
        log_progress(f"No shared keys for TimesFM metrics merge: {metrics_csv}")
        return rows

    tfm = source[keys + ["tfm_zeroshot_mae", "tfm_zeroshot_mse"]].drop_duplicates(keys)
    merged = row_df.drop(columns=["tfm_zeroshot_mae", "tfm_zeroshot_mse"], errors="ignore").merge(
        tfm,
        on=keys,
        how="left",
    )
    matched = int(merged["tfm_zeroshot_mae"].notna().sum())
    log_progress(f"merged precomputed TimesFM metrics from {metrics_csv} matched={matched}/{len(merged)} keys={keys}")
    return merged.to_dict(orient="records")


EVAL_FIELDNAMES = [
    "domain", "dataset", "frequency", "horizon", "n_samples", "model",
    "total_mae", "total_mse", "tfm_zeroshot_mae", "tfm_zeroshot_mse",
    "no_residual_mae", "residual_gain", "residual_std",
    "total_pred_abs_mean", "residual_abs_mean", "residual_total_abs_ratio",
    "trend_mae", "seasonal_mae", "residual_mae",
    *(f"pred_share_{family}" for family in FAMILIES),
    *harmonic_activation_fieldnames(),
    "checkpoint",
]


def eval_horizon(
    horizon: int,
    args: argparse.Namespace,
    manifest: list[dict],
    device: torch.device,
) -> tuple[list[dict], str]:
    """Evaluate a single horizon across all datasets. Returns (rows, ckpt_path_str)."""
    out_root = args.results_root
    model, ckpt_path, _cfg = load_single_model(args.checkpoint_root, int(horizon), device)
    log_progress(f"h{horizon}: loaded checkpoint {ckpt_path.name}")
    tfm_model = None
    if not args.skip_tfm:
        log_progress(f"h{horizon}: loading TimesFM")
        tfm_model = load_tfm_zeroshot_model(512, int(horizon), args.hf_cache_dir)

    rows = []
    for item_idx, item in enumerate(manifest):
        domain = str(item.get("domain") or item.get("group") or "")
        dataset_name = str(item.get("dataset") or item.get("name"))
        cache_dir = manifest_cache_dir(args.real_root, item)
        fallback_freq = str(item.get("freq") or item.get("frequency") or "")

        log_progress(f"h{horizon}: [{item_idx + 1}/{len(manifest)}] {domain}/{dataset_name}")
        try:
            dataset = FineMaskRealDataset(
                cache_dir, int(horizon), args.samples_per_dataset,
                args.seed + item_idx, fallback_freq=fallback_freq,
                dataset_name=dataset_name, real_root=args.real_root,
                hf_cache_dir=args.hf_cache_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            log_progress(f"  skip {dataset_name}: {exc}")
            continue

        metrics, plot_items = evaluate_dataset(
            model, tfm_model, dataset, args.batch_size, device, args.plot_samples_per_dataset, args
        )
        row = {
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
            "no_residual_mae": metrics["no_residual_mae"],
            "residual_gain": metrics["residual_gain"],
            "residual_std": metrics["residual_std"],
            "total_pred_abs_mean": metrics["total_pred_abs_mean"],
            "residual_abs_mean": metrics["residual_abs_mean"],
            "residual_total_abs_ratio": metrics["residual_total_abs_ratio"],
            "trend_mae": None,
            "seasonal_mae": None,
            "residual_mae": None,
            "checkpoint": str(ckpt_path),
        }
        for family in FAMILIES:
            row[f"pred_share_{family}"] = metrics.get(f"pred_share_{family}")
        for key in harmonic_activation_fieldnames():
            row[key] = metrics.get(key)
        rows.append(row)
        log_progress(f"  done model_mae={metrics['total_mae']:.6g}")

        dataset_out = dataset_result_dir(out_root, dataset_name)
        for plot_idx in range(0, len(plot_items), 3):
            plot_real_comparison_grid(
                dataset_out / "plots" / f"h{horizon}_samples{plot_idx // 3 + 1}.png",
                f"{domain}/{dataset_name} h{horizon}",
                plot_items[plot_idx: plot_idx + 3],
            )

    del model
    if tfm_model is not None:
        del tfm_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows, str(ckpt_path)


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict],
    ckpt_by_horizon: dict[str, str],
    out_root: Path,
) -> None:
    rows = attach_precomputed_timesfm(rows, args.timesfm_metrics_csv)
    write_csv(out_root / "real_eval_component_mae.csv", rows, EVAL_FIELDNAMES)
    write_csv(out_root / "real_eval_mae.csv", rows, EVAL_FIELDNAMES)
    plot_model_vs_tfm_by_horizon(rows, out_root / "performance_by_horizon_all.png",
                                  "Soft-mask real eval MAE by horizon")
    plot_model_vs_tfm_by_horizon(rows, out_root / "performance_by_horizon.png",
                                  "Soft-mask real eval MAE by horizon")
    write_summary(out_root / "real_eval_summary.json",
                  {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
                  rows, ["domain", "dataset", "horizon", "model"])
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset_name, sub_rows in by_dataset.items():
        dataset_out = dataset_result_dir(out_root, dataset_name)
        write_csv(dataset_out / "component_mae.csv", sub_rows, EVAL_FIELDNAMES)
        plot_model_vs_tfm_by_horizon(
            sub_rows,
            dataset_out / "performance_by_horizon.png",
            f"{dataset_name} MAE by horizon",
        )
        write_summary(
            dataset_out / "summary.json",
            {**vars(args), "checkpoint_by_horizon": ckpt_by_horizon},
            sub_rows,
            ["horizon", "model"],
        )


def run(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    out_root = args.results_root
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.real_root)
    if args.datasets:
        dataset_filter = set(args.datasets)
        manifest = [
            item for item in manifest
            if str(item.get("dataset") or item.get("name")) in dataset_filter
        ]
    if not manifest:
        raise FileNotFoundError(f"No matching evaluation datasets in {args.real_root}")

    log_progress(f"start device={device} output={out_root} datasets={len(manifest)}")
    if args.skip_tfm:
        log_progress("TimesFM execution disabled; tfm columns will be filled from precomputed metrics CSV")

    rows = []
    ckpt_by_horizon = {}
    for horizon in args.horizons:
        horizon_rows, ckpt_path_str = eval_horizon(horizon, args, manifest, device)
        rows.extend(horizon_rows)
        ckpt_by_horizon[str(horizon)] = ckpt_path_str

    write_outputs(args, rows, ckpt_by_horizon, out_root)
    log_progress(f"complete output={out_root}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
