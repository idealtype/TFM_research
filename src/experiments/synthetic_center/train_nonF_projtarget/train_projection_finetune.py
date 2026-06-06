from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


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

from hpo_space import N_KNOTS_FAMILIES  # noqa: E402
from model.decomp_funcdec import FuncDecModel  # noqa: E402
from model.decoder_seasonal import N_FOURIER_TERMS as HEAD_FOURIER_TERMS  # noqa: E402
from train_synth_only import to_jsonable  # noqa: E402


DEFAULT_TRAIN_ROOT = resolve_data_path("/home/sia2/project/data/synthetic_nonF/synth_train_nonfourier")
DEFAULT_PROJECTION_ROOT = THIS_DIR / "projection_targets"
DEFAULT_INIT_CHECKPOINT_DIR = (
    OLD_EXP_DIR
    / "results"
    / "simple_complex_synth_fixed_phase_scale"
    / "train"
    / "simple_complex_coeff_residual_tail"
)
DEFAULT_RESULTS_DIR = THIS_DIR / "results" / "train"
HORIZONS = [96, 192, 336, 720]
FOURIER_PROFILE = {"daily": 10, "weekly": 4, "yearly": 8}
FAMILIES = ["daily", "weekly", "yearly"]
STAGE_CACHE_DIRS = {
    "stage1_S": "stage1_S_nonfourier_cache_10_4_8",
    "stage2_T_S": "stage2_T_S_nonfourier_cache_10_4_8",
    "stage3_T_S_R": "stage3_T_S_R_nonfourier_cache_10_4_8",
}
STAGE1_RE = re.compile(r"(?P<granularity>\w+)_seed(?P<seed>\d+)_c(?P<context_len>\d+)_h(?P<horizon>\d+)$")
COMPLEX_RE = re.compile(r"(?P<trend_level>T\d+)_(?P<granularity>\w+)_seed(?P<seed>\d+)_c(?P<context_len>\d+)_h(?P<horizon>\d+)$")


def _pad_width(tensor: torch.Tensor, family: str) -> torch.Tensor:
    target_cols = 2 * int(HEAD_FOURIER_TERMS[family])
    tensor = tensor.float()
    if tensor.shape[1] == target_cols:
        return tensor
    if tensor.shape[1] > target_cols:
        return tensor[:, :target_cols]
    pad = torch.zeros(tensor.shape[0], target_cols - tensor.shape[1], dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=1)


def build_config(horizon: int, args: argparse.Namespace, ckpt_cfg: dict | None = None) -> dict:
    cfg = {
        "context_len": 512,
        "embed_dim": 1280,
        "horizon": int(horizon),
        "n_knots": N_KNOTS_FAMILIES[args.n_knots_family],
        "n_fourier_terms": dict(FOURIER_PROFILE),
        "mlp_units": {"trend": [1280, 1280], "seasonal": [1280, 1280], "residual": [1280, 1280]},
        "activation": "ReLU",
        "dropout": 0.0,
    }
    if ckpt_cfg:
        for key in ["context_len", "embed_dim", "n_knots", "n_fourier_terms", "mlp_units", "activation", "dropout"]:
            if key in ckpt_cfg:
                cfg[key] = ckpt_cfg[key]
    cfg["horizon"] = int(horizon)
    return cfg


def checkpoint_candidates(run_dir: Path, horizon: int) -> list[Path]:
    return [
        run_dir / "checkpoints" / f"funcdec_h{horizon}.pt",
        run_dir / "checkpoints" / f"simple_complex_synth_h{horizon}.pt",
    ]


def load_initial_model(horizon: int, args: argparse.Namespace, device: torch.device) -> tuple[FuncDecModel, Path, dict]:
    found = next((path for path in checkpoint_candidates(args.init_checkpoint_dir, horizon) if path.exists()), None)
    if found is None:
        tried = "\n".join(str(path) for path in checkpoint_candidates(args.init_checkpoint_dir, horizon))
        raise FileNotFoundError(f"No initial checkpoint for h{horizon}. Tried:\n{tried}")
    payload = torch.load(found, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        ckpt_cfg = payload.get("config", {})
    else:
        state = payload
        ckpt_cfg = {}
    cfg = build_config(horizon, args, ckpt_cfg)
    model = FuncDecModel(cfg, load_backbone=False).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if key.startswith("decoder_")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("backbone.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: {found} missing={missing[:8]} unexpected={unexpected[:8]}")
    return model, found, cfg


def parse_stage1(path: Path, generator: str) -> dict | None:
    match = STAGE1_RE.fullmatch(path.name)
    if not match:
        return None
    meta = match.groupdict()
    return {
        "stage": "stage1_S",
        "generator": generator,
        "residual_distribution": "",
        "trend_level": "",
        "granularity": meta["granularity"],
        "seed": int(meta["seed"]),
        "context_len": int(meta["context_len"]),
        "horizon": int(meta["horizon"]),
    }


def parse_complex(path: Path, stage: str, generator: str, residual_distribution: str = "") -> dict | None:
    match = COMPLEX_RE.fullmatch(path.name)
    if not match:
        return None
    meta = match.groupdict()
    return {
        "stage": stage,
        "generator": generator,
        "residual_distribution": residual_distribution,
        "trend_level": meta["trend_level"],
        "granularity": meta["granularity"],
        "seed": int(meta["seed"]),
        "context_len": int(meta["context_len"]),
        "horizon": int(meta["horizon"]),
    }


def discover_groups(args: argparse.Namespace, horizon: int) -> list[tuple[Path, dict]]:
    groups: list[tuple[Path, dict]] = []
    stages = set(args.stages)
    generators = set(args.generators or [])
    residuals = set(args.residual_distributions or [])
    trend_levels = set(args.trend_levels or [])

    if "stage1_S" in stages:
        root = args.train_root / STAGE_CACHE_DIRS["stage1_S"]
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            if generators and gen_dir.name not in generators:
                continue
            for ds_dir in sorted((gen_dir / "seasonal").iterdir()) if (gen_dir / "seasonal").exists() else []:
                meta = parse_stage1(ds_dir, gen_dir.name)
                if meta and meta["horizon"] == horizon:
                    groups.append((ds_dir, meta))

    if "stage2_T_S" in stages:
        root = args.train_root / STAGE_CACHE_DIRS["stage2_T_S"]
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            if generators and gen_dir.name not in generators:
                continue
            for ds_dir in sorted((gen_dir / "complex").iterdir()) if (gen_dir / "complex").exists() else []:
                meta = parse_complex(ds_dir, "stage2_T_S", gen_dir.name)
                if meta and meta["horizon"] == horizon and (not trend_levels or meta["trend_level"] in trend_levels):
                    groups.append((ds_dir, meta))

    if "stage3_T_S_R" in stages:
        root = args.train_root / STAGE_CACHE_DIRS["stage3_T_S_R"]
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            if generators and gen_dir.name not in generators:
                continue
            for res_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
                if residuals and res_dir.name not in residuals:
                    continue
                for ds_dir in sorted((res_dir / "complex").iterdir()) if (res_dir / "complex").exists() else []:
                    meta = parse_complex(ds_dir, "stage3_T_S_R", gen_dir.name, res_dir.name)
                    if meta and meta["horizon"] == horizon and (not trend_levels or meta["trend_level"] in trend_levels):
                        groups.append((ds_dir, meta))
    return groups


def projection_dir(args: argparse.Namespace, ds_dir: Path) -> Path:
    return args.projection_root / ds_dir.relative_to(args.train_root)


class PayloadCache:
    cache: dict[str, dict] = {}

    @classmethod
    def clear(cls) -> None:
        cls.cache.clear()
        gc.collect()

    @classmethod
    def load(cls, ds_dir: Path, meta: dict, args: argparse.Namespace) -> dict:
        key = str(ds_dir)
        if key in cls.cache:
            return cls.cache[key]
        horizon = int(meta["horizon"])
        context_len = int(meta["context_len"])
        proj_dir = projection_dir(args, ds_dir)
        backbone = torch.load(ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt", map_location="cpu", weights_only=False)
        basis = torch.load(ds_dir / f"fourier_basis_h{horizon}.pt", map_location="cpu", weights_only=False)
        proj_coeffs = torch.load(proj_dir / f"projected_seasonal_coefficients_h{horizon}.pt", map_location="cpu", weights_only=False)
        proj_targets = torch.load(proj_dir / f"projected_component_targets_h{horizon}.pt", map_location="cpu", weights_only=False)
        payload = {
            "meta": meta,
            "embeddings": backbone["embeddings"].float(),
            "trend_n": proj_targets["trend_n"].float(),
            "seasonal_projection_n": proj_targets["seasonal_projection_n"].float(),
            "seasonal_remainder_n": proj_targets["seasonal_remainder_n"].float(),
            "clean_total_n": proj_targets["clean_total_n"].float(),
            "noise_metric_only_n": proj_targets["noise_metric_only_n"].float(),
            "daily_basis": _pad_width(basis["daily_basis"], "daily"),
            "weekly_basis": _pad_width(basis["weekly_basis"], "weekly"),
            "yearly_basis": _pad_width(basis["yearly_basis"], "yearly"),
            "daily_coefficients": _pad_width(proj_coeffs["daily_coefficients"], "daily"),
            "weekly_coefficients": _pad_width(proj_coeffs["weekly_coefficients"], "weekly"),
            "yearly_coefficients": _pad_width(proj_coeffs["yearly_coefficients"], "yearly"),
        }
        cls.cache[key] = payload
        return payload


class RandomProjectionDataset(Dataset):
    def __init__(self, items: list[tuple[Path, dict]], split: str, val_split: float, length: int, seed: int, args: argparse.Namespace):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.groups = []
        self.length = int(length)
        self.seed = int(seed)
        for ds_dir, meta in items:
            payload = PayloadCache.load(ds_dir, meta, args)
            n = int(payload["clean_total_n"].shape[0])
            split_at = int(n * (1.0 - val_split))
            split_at = max(1, min(split_at, n - 1))
            indices = np.arange(0, split_at) if split == "train" else np.arange(split_at, n)
            if len(indices) > 0:
                self.groups.append((payload, indices))
        if not self.groups:
            raise ValueError(f"No {split} samples")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + int(idx))
        payload, indices = self.groups[int(rng.integers(0, len(self.groups)))]
        sample_idx = int(indices[int(rng.integers(0, len(indices)))])
        meta = payload["meta"]
        return {
            "emb": payload["embeddings"][sample_idx],
            "trend_n": payload["trend_n"][sample_idx],
            "seasonal_projection_n": payload["seasonal_projection_n"][sample_idx],
            "seasonal_remainder_n": payload["seasonal_remainder_n"][sample_idx],
            "clean_total_n": payload["clean_total_n"][sample_idx],
            "noise_metric_only_n": payload["noise_metric_only_n"][sample_idx],
            "daily_basis": payload["daily_basis"],
            "weekly_basis": payload["weekly_basis"],
            "yearly_basis": payload["yearly_basis"],
            "daily_coefficients": payload["daily_coefficients"][sample_idx],
            "weekly_coefficients": payload["weekly_coefficients"][sample_idx],
            "yearly_coefficients": payload["yearly_coefficients"][sample_idx],
            "stage": meta["stage"],
            "generator": meta["generator"],
            "residual_distribution": meta["residual_distribution"],
            "trend_level": meta["trend_level"],
            "granularity": meta["granularity"],
        }


def collate(batch: list[dict]) -> dict:
    tensor_keys = [
        "emb",
        "trend_n",
        "seasonal_projection_n",
        "seasonal_remainder_n",
        "clean_total_n",
        "noise_metric_only_n",
        "daily_basis",
        "weekly_basis",
        "yearly_basis",
        "daily_coefficients",
        "weekly_coefficients",
        "yearly_coefficients",
    ]
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ["stage", "generator", "residual_distribution", "trend_level", "granularity"]:
        out[key] = [item[key] for item in batch]
    return out


def batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["emb"].to(device, non_blocking=True),
        batch["daily_basis"].to(device, non_blocking=True),
        batch["weekly_basis"].to(device, non_blocking=True),
        batch["yearly_basis"].to(device, non_blocking=True),
    )


def set_trainable(model: FuncDecModel, args: argparse.Namespace) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    modules = []
    if args.train_trend:
        modules.append(model.decoder_t)
    if args.train_seasonal:
        modules.append(model.decoder_s)
    if args.train_residual:
        modules.append(model.decoder_r)
    params = []
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
            params.append(param)
    if not params:
        raise ValueError("No trainable decoder selected")
    return params


def coefficient_loss(decomp: dict, batch: dict, device: torch.device) -> torch.Tensor:
    coeffs = decomp["seasonal_coefficients"]
    return (
        F.l1_loss(coeffs["daily"], batch["daily_coefficients"].to(device, non_blocking=True))
        + F.l1_loss(coeffs["weekly"], batch["weekly_coefficients"].to(device, non_blocking=True))
        + F.l1_loss(coeffs["yearly"], batch["yearly_coefficients"].to(device, non_blocking=True))
    )


def training_loss(pred: torch.Tensor, decomp: dict, batch: dict, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    trend_n = batch["trend_n"].to(device, non_blocking=True)
    clean_total_n = batch["clean_total_n"].to(device, non_blocking=True)
    seasonal_remainder_n = batch["seasonal_remainder_n"].to(device, non_blocking=True)

    clean_pred = decomp["trend"] + decomp["seasonal"] + decomp["residual"]
    loss = args.lambda_total * F.l1_loss(clean_pred, clean_total_n)
    loss = loss + args.lambda_trend * F.l1_loss(decomp["trend"], trend_n)
    loss = loss + args.lambda_seasonal_coefficients * coefficient_loss(decomp, batch, device)
    loss = loss + args.lambda_residual_remainder * F.l1_loss(decomp["residual"], seasonal_remainder_n)
    return loss


@torch.no_grad()
def evaluate(model: FuncDecModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals: dict[str, dict[str, float]] = {}
    for batch in loader:
        emb, db, wb, yb = batch_to_device(batch, device)
        trend_n = batch["trend_n"].to(device, non_blocking=True)
        projection_n = batch["seasonal_projection_n"].to(device, non_blocking=True)
        remainder_n = batch["seasonal_remainder_n"].to(device, non_blocking=True)
        clean_total_n = batch["clean_total_n"].to(device, non_blocking=True)
        pred, decomp = model(emb, db, wb, yb)
        clean_pred = decomp["trend"] + decomp["seasonal"] + decomp["residual"]
        errors = {
            "clean_total_mae": torch.abs(clean_pred - clean_total_n).detach().cpu(),
            "trend_mae": torch.abs(decomp["trend"] - trend_n).detach().cpu(),
            "seasonal_projection_mae": torch.abs(decomp["seasonal"] - projection_n).detach().cpu(),
            "residual_remainder_mae": torch.abs(decomp["residual"] - remainder_n).detach().cpu(),
            "seasonal_reconstructed_mae": torch.abs((decomp["seasonal"] + decomp["residual"]) - (projection_n + remainder_n)).detach().cpu(),
        }
        for i, stage in enumerate(batch["stage"]):
            key = "/".join(
                part
                for part in [
                    stage,
                    batch["generator"][i],
                    batch["residual_distribution"][i],
                    batch["trend_level"][i],
                    batch["granularity"][i],
                ]
                if part
            )
            group = totals.setdefault(key, {name: 0.0 for name in errors} | {"n_values": 0.0})
            for name, err in errors.items():
                group[name] += float(err[i].sum().item())
            group["n_values"] += float(clean_total_n.shape[1])
    metric_names = [
        "clean_total_mae",
        "trend_mae",
        "seasonal_projection_mae",
        "residual_remainder_mae",
        "seasonal_reconstructed_mae",
    ]
    return {
        key: {name: value[name] / max(1.0, value["n_values"]) for name in metric_names}
        | {"n_values": int(value["n_values"])}
        for key, value in sorted(totals.items())
    }


def train_horizon(horizon: int, items: list[tuple[Path, dict]], args: argparse.Namespace, device: torch.device) -> dict:
    model, init_ckpt, cfg = load_initial_model(horizon, args, device)
    params = set_trainable(model, args)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    train_ds = RandomProjectionDataset(
        items, "train", args.val_split,
        length=args.max_steps * args.batch_size,
        seed=args.seed + horizon * 17,
        args=args,
    )
    val_ds = RandomProjectionDataset(
        items, "val", args.val_split,
        length=max(args.eval_batches * args.batch_size, args.batch_size),
        seed=args.seed + horizon * 31,
        args=args,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, **dataloader_kwargs(args, device))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, **dataloader_kwargs(args, device))

    history = []
    started = time.perf_counter()
    model.train()
    step = 0
    for batch in train_loader:
        emb, db, wb, yb = batch_to_device(batch, device)
        pred, decomp = model(emb, db, wb, yb)
        loss = training_loss(pred, decomp, batch, args, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            value = float(loss.item())
            history.append({"step": step, "loss": value})
            print(f"h{horizon} step {step}/{args.max_steps} loss={value:.6g}", flush=True)
        if step >= args.max_steps:
            break

    eval_rows = evaluate(model, val_loader, device)
    ckpt_path = None
    if args.save_checkpoint:
        ckpt_dir = Path(args.results_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"nonfourier_projection_finetune_h{horizon}.pt"
        torch.save({"config": cfg, "args": to_jsonable(vars(args)), "state_dict": model.state_dict()}, ckpt_path)
        eval_ckpt_path = ckpt_dir / f"funcdec_h{horizon}.pt"
        if eval_ckpt_path.exists() or eval_ckpt_path.is_symlink():
            eval_ckpt_path.unlink()
        eval_ckpt_path.symlink_to(ckpt_path.name)

    elapsed = time.perf_counter() - started
    del model, train_loader, val_loader, train_ds, val_ds
    PayloadCache.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "horizon": int(horizon),
        "initial_checkpoint": str(init_ckpt),
        "num_cache_groups": len(items),
        "history": history,
        "eval": eval_rows,
        "checkpoint_path": None if ckpt_path is None else str(ckpt_path),
        "elapsed_sec": elapsed,
        "loss_policy": {
            "trend": "trend decoder vs trend_n",
            "seasonal": "seasonal decoder coefficient outputs vs projected Fourier coefficients",
            "residual": "residual decoder vs seasonal_n minus projected seasonal_n",
            "total": "trend+seasonal+residual vs trend_n+seasonal_n",
            "distribution_noise": "excluded from all training targets",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FuncDec with projected Fourier coefficient targets for non-Fourier seasonal data.")
    parser.add_argument("--train_root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--projection_root", type=Path, default=DEFAULT_PROJECTION_ROOT)
    parser.add_argument("--init_checkpoint_dir", type=Path, default=DEFAULT_INIT_CHECKPOINT_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run_name", default="nonfourier_projection_from_simple_complex")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--stages", nargs="+", choices=list(STAGE_CACHE_DIRS), default=list(STAGE_CACHE_DIRS))
    parser.add_argument("--generators", nargs="+", default=None)
    parser.add_argument("--residual_distributions", nargs="+", default=None)
    parser.add_argument("--trend_levels", nargs="+", default=None)
    parser.add_argument("--n_knots_family", choices=sorted(N_KNOTS_FAMILIES), default="dense")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--lambda_total", type=float, default=1.0)
    parser.add_argument("--lambda_trend", type=float, default=1.0)
    parser.add_argument("--lambda_seasonal_coefficients", type=float, default=1.0)
    parser.add_argument("--lambda_residual_remainder", type=float, default=1.0)
    parser.add_argument("--train_trend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_seasonal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--save_checkpoint", action="store_true")
    add_runtime_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.results_dir = Path(args.results_dir) / args.run_name
    args.results_dir.mkdir(parents=True, exist_ok=True)

    result = {"run_name": args.run_name, "args": to_jsonable(vars(args)), "per_horizon": {}}
    print(f"[train-proj] results_dir={args.results_dir}", flush=True)
    print(f"[train-proj] train_root={args.train_root}", flush=True)
    print(f"[train-proj] projection_root={args.projection_root}", flush=True)
    print(f"[train-proj] init_checkpoint_dir={args.init_checkpoint_dir}", flush=True)
    print("[train-proj] seasonal learns projected coefficients; residual learns projection remainder; distribution noise excluded", flush=True)

    for horizon in args.horizons:
        items = discover_groups(args, int(horizon))
        if not items:
            raise FileNotFoundError(f"No non-Fourier train cache groups for h={horizon} under {args.train_root}")
        print(f"=== horizon {horizon}: groups={len(items)} ===", flush=True)
        result["per_horizon"][str(horizon)] = train_horizon(int(horizon), items, args, device)
        with (args.results_dir / "result_partial.json").open("w") as f:
            json.dump(to_jsonable(result), f, indent=2)

    with (args.results_dir / "result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    print(f"saved result={args.results_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
