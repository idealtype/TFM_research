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


class PayloadCache:
    cache: dict[str, dict] = {}

    @classmethod
    def clear(cls) -> None:
        cls.cache.clear()
        gc.collect()

    @classmethod
    def load(cls, ds_dir: Path, meta: dict) -> dict:
        key = str(ds_dir)
        if key in cls.cache:
            return cls.cache[key]
        horizon = int(meta["horizon"])
        context_len = int(meta["context_len"])
        backbone = torch.load(ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt", map_location="cpu", weights_only=False)
        raw = torch.load(ds_dir / f"raw_futures_h{horizon}.pt", map_location="cpu", weights_only=False)
        basis = torch.load(ds_dir / f"fourier_basis_h{horizon}.pt", map_location="cpu", weights_only=False)
        components = torch.load(ds_dir / f"component_targets_h{horizon}.pt", map_location="cpu", weights_only=False)
        payload = {
            "meta": meta,
            "embeddings": backbone["embeddings"].float(),
            "future_n": raw["futures_n"].float(),
            "trend_n": components["trend_n"].float(),
            "seasonal_n": components["seasonal_n"].float(),
            # Kept for metrics only. Distribution residuals are not used as a direct training target.
            "residual_n": components["residual_n"].float(),
            "daily_basis": _pad_width(basis["daily_basis"], "daily"),
            "weekly_basis": _pad_width(basis["weekly_basis"], "weekly"),
            "yearly_basis": _pad_width(basis["yearly_basis"], "yearly"),
        }
        cls.cache[key] = payload
        return payload


class RandomNonFourierDataset(Dataset):
    def __init__(self, items: list[tuple[Path, dict]], split: str, val_split: float, length: int, seed: int):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.groups = []
        self.length = int(length)
        self.seed = int(seed)
        for ds_dir, meta in items:
            payload = PayloadCache.load(ds_dir, meta)
            n = int(payload["future_n"].shape[0])
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
            "future_n": payload["future_n"][sample_idx],
            "trend_n": payload["trend_n"][sample_idx],
            "seasonal_n": payload["seasonal_n"][sample_idx],
            "residual_n": payload["residual_n"][sample_idx],
            "daily_basis": payload["daily_basis"],
            "weekly_basis": payload["weekly_basis"],
            "yearly_basis": payload["yearly_basis"],
            "stage": meta["stage"],
            "generator": meta["generator"],
            "residual_distribution": meta["residual_distribution"],
            "trend_level": meta["trend_level"],
            "granularity": meta["granularity"],
        }


def collate(batch: list[dict]) -> dict:
    tensor_keys = ["emb", "future_n", "trend_n", "seasonal_n", "residual_n", "daily_basis", "weekly_basis", "yearly_basis"]
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


def training_loss(pred: torch.Tensor, decomp: dict, batch: dict, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    trend_n = batch["trend_n"].to(device, non_blocking=True)
    seasonal_n = batch["seasonal_n"].to(device, non_blocking=True)

    # Stage3 distribution noise is intentionally excluded from supervision.
    # The residual decoder is allowed to help represent non-Fourier seasonality,
    # but it is not trained to reconstruct random distribution noise.
    clean_total = trend_n + seasonal_n
    seasonal_plus_residual = decomp["seasonal"] + decomp["residual"]
    clean_pred = decomp["trend"] + seasonal_plus_residual

    loss = args.lambda_total * F.l1_loss(clean_pred, clean_total)
    loss = loss + args.lambda_trend * F.l1_loss(decomp["trend"], trend_n)
    loss = loss + args.lambda_seasonal * F.l1_loss(seasonal_plus_residual, seasonal_n)
    return loss


@torch.no_grad()
def evaluate(model: FuncDecModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals: dict[str, dict[str, float]] = {}
    for batch in loader:
        emb, db, wb, yb = batch_to_device(batch, device)
        future_n = batch["future_n"].to(device, non_blocking=True)
        trend_n = batch["trend_n"].to(device, non_blocking=True)
        seasonal_n = batch["seasonal_n"].to(device, non_blocking=True)
        residual_n = batch["residual_n"].to(device, non_blocking=True)
        pred, decomp = model(emb, db, wb, yb)
        errors = {
            "total_mae": torch.abs(pred - future_n).detach().cpu(),
            "clean_total_mae": torch.abs((decomp["trend"] + decomp["seasonal"] + decomp["residual"]) - (trend_n + seasonal_n)).detach().cpu(),
            "trend_mae": torch.abs(decomp["trend"] - trend_n).detach().cpu(),
            "seasonal_plus_residual_mae": torch.abs((decomp["seasonal"] + decomp["residual"]) - seasonal_n).detach().cpu(),
            "seasonal_decoder_mae": torch.abs(decomp["seasonal"] - seasonal_n).detach().cpu(),
            "residual_mae_metric_only": torch.abs(decomp["residual"] - residual_n).detach().cpu(),
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
            group["n_values"] += float(future_n.shape[1])
    return {
        key: {
            name: value[name] / max(1.0, value["n_values"])
            for name in [
                "total_mae",
                "clean_total_mae",
                "trend_mae",
                "seasonal_plus_residual_mae",
                "seasonal_decoder_mae",
                "residual_mae_metric_only",
            ]
        }
        | {"n_values": int(value["n_values"])}
        for key, value in sorted(totals.items())
    }


def train_horizon(horizon: int, items: list[tuple[Path, dict]], args: argparse.Namespace, device: torch.device) -> dict:
    model, init_ckpt, cfg = load_initial_model(horizon, args, device)
    params = set_trainable(model, args)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    train_ds = RandomNonFourierDataset(
        items, "train", args.val_split,
        length=args.max_steps * args.batch_size,
        seed=args.seed + horizon * 17,
    )
    val_ds = RandomNonFourierDataset(
        items, "val", args.val_split,
        length=max(args.eval_batches * args.batch_size, args.batch_size),
        seed=args.seed + horizon * 31,
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
        ckpt_path = ckpt_dir / f"nonfourier_finetune_h{horizon}.pt"
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
            "total": "trend+seasonal+residual vs trend_n+seasonal_n; distribution noise excluded",
            "trend": "trend decoder vs trend_n",
            "seasonal": "seasonal decoder + residual decoder vs seasonal_n raw non-Fourier component",
            "residual": "no direct residual target; distribution noise is not supervised",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Fourier pretrained FuncDec on non-Fourier synthetic caches.")
    parser.add_argument("--train_root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--init_checkpoint_dir", type=Path, default=DEFAULT_INIT_CHECKPOINT_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run_name", default="nonfourier_finetune_from_simple_complex")
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
    parser.add_argument("--lambda_seasonal", type=float, default=1.0)
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
    print(f"[train-nonF] results_dir={args.results_dir}", flush=True)
    print(f"[train-nonF] train_root={args.train_root}", flush=True)
    print(f"[train-nonF] init_checkpoint_dir={args.init_checkpoint_dir}", flush=True)
    print("[train-nonF] residual distribution noise is not used as direct target", flush=True)

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
