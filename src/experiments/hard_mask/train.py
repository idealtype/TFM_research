#!/usr/bin/env python3
"""Full training pipeline for 5.30fine_mask.

Training order:
  1. Fourier synthetic (existing S1-S10 + new SM1-SM10)
  2. non-Fourier synthetic (existing nonF caches)
  3. Real data (domain real finetune)

Initial checkpoint: 5.22syn_cent/train_nonF_rawtarget results (loaded with strict=False).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THIS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import add_runtime_args, dataloader_kwargs, resolve_data_path, resolve_project_path  # noqa: E402
PROJECT_ROOT_4_28 = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT_4_28 / "src"
OLD_EXP_DIR = PROJECT_ROOT_4_28 / "basis_dec" / "experiment" / "func_dec_syn_cent"
NONF_TRAIN_DIR = resolve_project_path("/home/sia2/project/5.22syn_cent/train_nonF_rawtarget")
REAL_TRAIN_DIR = resolve_project_path("/home/sia2/project/5.22syn_cent/train_syn_real_raw")

# Keep THIS_DIR first even when Python already inserted the script directory.
# Otherwise older model/ packages on sys.path can shadow the fine_mask model.
for path in [THIS_DIR, PROJECT_ROOT_4_28, SRC_DIR, OLD_EXP_DIR, NONF_TRAIN_DIR, REAL_TRAIN_DIR]:
    path_s = str(path)
    while path_s in sys.path:
        sys.path.remove(path_s)
for path in [REAL_TRAIN_DIR, NONF_TRAIN_DIR, OLD_EXP_DIR, SRC_DIR, PROJECT_ROOT_4_28, THIS_DIR]:
    sys.path.insert(0, str(path))

from model.decomp_funcdec import FuncDecModel  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_CONFIG,
    FREQ_DAYS,
    K_MAX,
    adapt_state_dict_for_fine_mask,
    build_fine_mask_basis,
    expand_bases,
    load_fine_mask_basis,
    upgrade_legacy_basis,
    to_jsonable,
)


HORIZONS = [96, 192, 336, 720]

DEFAULT_INIT_CHECKPOINT_DIR = Path(
    "/home/sia2/project/5.22syn_cent/train_nonF_rawtarget/results/train/"
    "nonfourier_finetune_from_simple_complex"
)
DEFAULT_RESULTS_ROOT = THIS_DIR / "results"
DEFAULT_LOTSA_CACHE_ROOT = resolve_data_path("/home/sia2/project/data/data_lotsa/lotsa_cache")

# Synth Fourier cache roots (old S1-S10 + new SM1-SM10)
FOURIER_SYNTH_CACHE_ROOTS = [
    resolve_data_path("/home/sia2/project/data/synthetic/func_dec_syn_cent_complex_train_cache_10_4_8_fixed_phase_scale"),
    resolve_data_path("/home/sia2/project/data/synthetic/func_dec_syn_cent_fine_mask_train_cache_10_4_2_8"),
]

# nonF cache roots
NONF_STAGE_CACHE_DIRS = {
    "stage1_S": "stage1_S_nonfourier_cache_10_4_8",
    "stage2_T_S": "stage2_T_S_nonfourier_cache_10_4_8",
    "stage3_T_S_R": "stage3_T_S_R_nonfourier_cache_10_4_8",
}
NONF_TRAIN_ROOT = resolve_data_path("/home/sia2/project/data/synthetic_nonF/synth_train_nonfourier")

COMPLEX_CACHE_RE = re.compile(
    r"^(?:A\d+_)?(?:T\d+_)?(?:\w+\d+_)?(?P<granularity>\w+?)_seed\d+_c\d+_h\d+$"
)
FUTURE_RE = re.compile(r"futures_c(?P<context>\d+)_(?P<freq>.+)_h(?P<horizon>\d+)_.+\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint_dir", type=Path, default=DEFAULT_INIT_CHECKPOINT_DIR)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--lotsa_cache_root", type=Path, default=DEFAULT_LOTSA_CACHE_ROOT)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    # Fourier synth training
    parser.add_argument("--fourier_steps", type=int, default=5000)
    parser.add_argument("--fourier_batch_size", type=int, default=256)
    # nonF training
    parser.add_argument("--nonf_steps", type=int, default=3000)
    parser.add_argument("--nonf_batch_size", type=int, default=256)
    parser.add_argument("--nonf_stages", nargs="+",
                        choices=list(NONF_STAGE_CACHE_DIRS.keys()),
                        default=list(NONF_STAGE_CACHE_DIRS.keys()))
    # Real training
    parser.add_argument("--real_steps", type=int, default=10000)
    parser.add_argument("--real_batch_size", type=int, default=256)
    parser.add_argument("--real_group_chunk_steps", type=int, default=250)
    parser.add_argument("--domain_config", type=Path,
                        default=REAL_TRAIN_DIR / "domain_config.json")
    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_fourier", action="store_true")
    parser.add_argument("--skip_nonf", action="store_true")
    parser.add_argument("--skip_real", action="store_true")
    add_runtime_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_initial_model(horizon: int, args: argparse.Namespace, device: torch.device):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["horizon"] = int(horizon)
    if args.init_checkpoint_dir == Path("none"):
        model = FuncDecModel(cfg, load_backbone=False).to(device)
        print("  [info] training from scratch (random init)", flush=True)
        return model, "scratch", cfg

    candidates = [
        args.init_checkpoint_dir / "checkpoints" / f"funcdec_h{horizon}.pt",
        args.init_checkpoint_dir / "checkpoints" / f"nonfourier_finetune_h{horizon}.pt",
        args.init_checkpoint_dir / "checkpoints" / f"simple_complex_synth_h{horizon}.pt",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No initial checkpoint for h{horizon}. Tried:\n{tried}")

    payload = torch.load(found, map_location=device, weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    ckpt_cfg = dict(payload.get("config", {})) if isinstance(payload, dict) else {}
    ckpt_cfg["horizon"] = int(horizon)

    for key in DEFAULT_CONFIG:
        if key in ckpt_cfg:
            if isinstance(cfg.get(key), dict) and isinstance(ckpt_cfg[key], dict):
                merged = dict(cfg[key])
                merged.update(ckpt_cfg[key])
                cfg[key] = merged
            else:
                cfg[key] = ckpt_cfg[key]
    cfg["horizon"] = int(horizon)

    model = FuncDecModel(cfg, load_backbone=False).to(device)
    # Adapt state dict to handle size mismatches (legacy HEAD capacity 10,10,10 vs new 10,4,2,8)
    adapted_state = adapt_state_dict_for_fine_mask(state, model)
    incompatible = model.load_state_dict(adapted_state, strict=False)
    # monthly-related missing keys are expected and OK
    missing = [k for k in incompatible.missing_keys
               if k.startswith("decoder_") and "monthly" not in k]
    unexpected = [k for k in incompatible.unexpected_keys
                  if not k.startswith("backbone.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: {found}\nmissing={missing[:8]}\nunexpected={unexpected[:8]}"
        )
    monthly_missing = [k for k in incompatible.missing_keys if "monthly" in k]
    if monthly_missing:
        print(f"  [info] monthly missing keys (expected): {monthly_missing[:4]}", flush=True)
    return model, found, cfg


def save_checkpoint(model: FuncDecModel, cfg: dict, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict(), **payload}, path)
    # Also create a symlink funcdec_h{H}.pt -> actual checkpoint
    eval_path = path.parent / f"funcdec_h{int(cfg['horizon'])}.pt"
    if eval_path == path:
        return
    if eval_path.exists() or eval_path.is_symlink():
        eval_path.unlink()
    eval_path.symlink_to(path.name)


# ---------------------------------------------------------------------------
# Basis resolution
# ---------------------------------------------------------------------------

def resolve_basis_for_dir(ds_dir: Path, granularity: str, horizon: int, context_len: int) -> dict:
    """Load fine_mask basis from file, or compute on-the-fly."""
    fine_mask_path = ds_dir / f"fourier_basis_fine_mask_h{horizon}.pt"
    legacy_path = ds_dir / f"fourier_basis_h{horizon}.pt"
    if fine_mask_path.exists():
        return load_fine_mask_basis(fine_mask_path)
    if legacy_path.exists():
        return upgrade_legacy_basis(legacy_path)
    freq = granularity  # granularity == freq for synth data
    return build_fine_mask_basis(freq, context_len, horizon)


def resolve_basis_for_real(group: dict) -> dict:
    """Load fine_mask basis for real data, falling back to on-the-fly computation."""
    context_len = int(group["context_len"])
    horizon = int(group["horizon"])
    freq = str(group["freq"])
    cache_dir = Path(group["cache_dir"])

    # Prefer fine_mask basis file
    fine_mask_files = sorted(cache_dir.glob(
        f"fourier_basis_c{context_len}_{freq}_h{horizon}_fine_mask_*.pt"
    ))
    if fine_mask_files:
        return load_fine_mask_basis(fine_mask_files[0])

    # Fall back to legacy basis (upgrade by adding zeros for monthly)
    legacy_files = sorted(cache_dir.glob(
        f"fourier_basis_c{context_len}_{freq}_h{horizon}_*.pt"
    ))
    if legacy_files:
        return upgrade_legacy_basis(legacy_files[0])

    # Compute on-the-fly
    if freq in FREQ_DAYS:
        return build_fine_mask_basis(freq, context_len, horizon)
    return build_fine_mask_basis("hourly", context_len, horizon)


# ---------------------------------------------------------------------------
# Fourier synth training
# ---------------------------------------------------------------------------

def infer_granularity(ds_dir: Path) -> str | None:
    match = COMPLEX_CACHE_RE.match(ds_dir.name)
    if match:
        g = match.group("granularity")
        if g in FREQ_DAYS:
            return g
    for g in ["hourly", "daily", "weekly"]:
        if g in ds_dir.name:
            return g
    return None


def discover_fourier_groups(horizon: int) -> list[dict]:
    groups = []
    for root in FOURIER_SYNTH_CACHE_ROOTS:
        if not root.exists():
            continue
        for backbone_file in sorted(root.rglob(f"backbone_emb_c*_h{horizon}_stride1.pt")):
            ds_dir = backbone_file.parent
            granularity = infer_granularity(ds_dir)
            if granularity is None:
                continue
            raw_path = ds_dir / f"raw_futures_h{horizon}.pt"
            if not raw_path.exists():
                continue
            # Extract context_len from filename
            m = re.match(r"backbone_emb_c(\d+)_h\d+_stride1\.pt", backbone_file.name)
            context_len = int(m.group(1)) if m else 512
            groups.append({
                "ds_dir": str(ds_dir),
                "granularity": granularity,
                "horizon": int(horizon),
                "context_len": int(context_len),
                "backbone_path": str(backbone_file),
                "raw_path": str(raw_path),
            })
    return groups


class FourierSynthPayloadCache:
    cache: dict[str, dict] = {}

    @classmethod
    def clear(cls) -> None:
        cls.cache.clear()
        gc.collect()

    @classmethod
    def load(cls, group: dict) -> dict:
        key = group["ds_dir"]
        if key in cls.cache:
            return cls.cache[key]
        ds_dir = Path(group["ds_dir"])
        horizon = int(group["horizon"])
        context_len = int(group["context_len"])
        granularity = group["granularity"]

        backbone = torch.load(group["backbone_path"], map_location="cpu", weights_only=False)
        raw = torch.load(group["raw_path"], map_location="cpu", weights_only=False)
        embeddings = backbone["embeddings"].float()
        future_n = raw["futures_n"].float()
        finite = torch.isfinite(embeddings).all(dim=1) & torch.isfinite(future_n).all(dim=1)
        valid_mask = raw.get("valid_mask")
        if valid_mask is not None:
            finite = finite & valid_mask.bool()
        indices = finite.nonzero(as_tuple=True)[0].cpu().numpy()
        if len(indices) == 0:
            raise ValueError(f"No valid samples in {ds_dir}")

        # Try to load component targets (may not exist)
        comp_path = ds_dir / f"component_targets_h{horizon}.pt"
        trend_n = None
        seasonal_n = None
        if comp_path.exists():
            comp = torch.load(comp_path, map_location="cpu", weights_only=False)
            trend_n = comp["trend_n"].float()
            seasonal_n = comp["seasonal_n"].float()

        bases = resolve_basis_for_dir(ds_dir, granularity, horizon, context_len)
        payload = {
            "embeddings": embeddings,
            "future_n": future_n,
            "trend_n": trend_n,
            "seasonal_n": seasonal_n,
            "indices": indices,
            "bases": bases,
        }
        cls.cache[key] = payload
        return payload


def sample_fourier_batch(payload: dict, batch_size: int, seed: int, step: int, device: torch.device):
    rng = np.random.default_rng(seed + step * 1009)
    chosen = rng.choice(payload["indices"], size=int(batch_size),
                        replace=len(payload["indices"]) < batch_size)
    chosen_t = torch.as_tensor(chosen, dtype=torch.long)
    emb = payload["embeddings"].index_select(0, chosen_t).to(device, non_blocking=True)
    future_n = payload["future_n"].index_select(0, chosen_t).to(device, non_blocking=True)
    trend_n = None
    if payload.get("trend_n") is not None:
        trend_n = payload["trend_n"].index_select(0, chosen_t).to(device, non_blocking=True)
    seasonal_n = None
    if payload.get("seasonal_n") is not None:
        seasonal_n = payload["seasonal_n"].index_select(0, chosen_t).to(device, non_blocking=True)
    bases = payload["bases"]
    daily, weekly, monthly, yearly = expand_bases(bases, emb.shape[0], device)
    return emb, future_n, trend_n, seasonal_n, daily, weekly, monthly, yearly


def train_fourier_synth(model: FuncDecModel, groups: list[dict], steps: int,
                        args: argparse.Namespace, horizon: int, device: torch.device) -> list[dict]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(steps))

    history = []
    model.train()
    active_groups = list(groups)
    for step in range(1, int(steps) + 1):
        group = active_groups[(step - 1) % len(active_groups)]
        try:
            payload = FourierSynthPayloadCache.load(group)
        except ValueError as exc:
            print(f"[fourier] skip h{horizon} {group['ds_dir']}: {exc}", flush=True)
            active_groups = [g for g in active_groups if g["ds_dir"] != group["ds_dir"]]
            if not active_groups:
                raise
            continue

        emb, future_n, trend_n, seasonal_n, daily, weekly, monthly, yearly = sample_fourier_batch(
            payload, args.fourier_batch_size, args.seed + horizon * 7, step, device
        )
        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        pred_loss = F.l1_loss(pred, future_n)
        loss_terms = []
        if trend_n is not None:
            loss_terms.append(F.l1_loss(decomp["trend"], trend_n))
        if seasonal_n is not None:
            loss_terms.append(F.l1_loss(decomp["seasonal"], seasonal_n))
        loss = sum(loss_terms) if loss_terms else pred_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == int(steps):
            value = float(loss.item())
            history.append({"phase": "fourier_synth", "step": step, "loss": value})
            print(f"[fourier] h{horizon} step {step}/{steps} loss={value:.6g}", flush=True)

    FourierSynthPayloadCache.clear()
    return history


# ---------------------------------------------------------------------------
# non-Fourier training
# ---------------------------------------------------------------------------

STAGE1_RE = re.compile(r"(?P<granularity>\w+)_seed(?P<seed>\d+)_c(?P<context_len>\d+)_h(?P<horizon>\d+)$")
COMPLEX_NONF_RE = re.compile(r"(?P<trend_level>T\d+)_(?P<granularity>\w+)_seed(?P<seed>\d+)_c(?P<context_len>\d+)_h(?P<horizon>\d+)$")


def discover_nonf_groups(horizon: int, stages: list[str]) -> list[tuple[Path, dict]]:
    groups = []
    for stage in stages:
        cache_dir_name = NONF_STAGE_CACHE_DIRS[stage]
        root = NONF_TRAIN_ROOT / cache_dir_name
        if not root.exists():
            continue
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if stage == "stage1_S":
                sub_dir = gen_dir / "seasonal"
                if not sub_dir.exists():
                    continue
                for ds_dir in sorted(sub_dir.iterdir()):
                    m = STAGE1_RE.fullmatch(ds_dir.name)
                    if m and int(m.group("horizon")) == horizon:
                        meta = {
                            "stage": stage,
                            "granularity": m.group("granularity"),
                            "horizon": horizon,
                            "context_len": int(m.group("context_len")),
                        }
                        groups.append((ds_dir, meta))
            elif stage == "stage2_T_S":
                sub_dir = gen_dir / "complex"
                if not sub_dir.exists():
                    continue
                for ds_dir in sorted(sub_dir.iterdir()):
                    m = COMPLEX_NONF_RE.fullmatch(ds_dir.name)
                    if m and int(m.group("horizon")) == horizon:
                        meta = {
                            "stage": stage,
                            "granularity": m.group("granularity"),
                            "horizon": horizon,
                            "context_len": int(m.group("context_len")),
                        }
                        groups.append((ds_dir, meta))
            else:
                for res_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
                    sub_dir = res_dir / "complex"
                    if not sub_dir.exists():
                        continue
                    for ds_dir in sorted(sub_dir.iterdir()):
                        m = COMPLEX_NONF_RE.fullmatch(ds_dir.name)
                        if m and int(m.group("horizon")) == horizon:
                            meta = {
                                "stage": stage,
                                "generator": gen_dir.name,
                                "residual_distribution": res_dir.name,
                                "granularity": m.group("granularity"),
                                "horizon": horizon,
                                "context_len": int(m.group("context_len")),
                            }
                            groups.append((ds_dir, meta))
    return groups


def _pad_width(tensor: torch.Tensor, family: str) -> torch.Tensor:
    """Pad/truncate basis tensor to match K_MAX capacity."""
    target_cols = 2 * int(K_MAX[family])
    tensor = tensor.float()
    if tensor.shape[1] == target_cols:
        return tensor
    if tensor.shape[1] > target_cols:
        return tensor[:, :target_cols]
    pad = torch.zeros(tensor.shape[0], target_cols - tensor.shape[1], dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=1)


class NonFPayloadCache:
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
        granularity = meta["granularity"]

        backbone_path = ds_dir / f"backbone_emb_c{context_len}_h{horizon}_stride1.pt"
        raw_path = ds_dir / f"raw_futures_h{horizon}.pt"
        comp_path = ds_dir / f"component_targets_h{horizon}.pt"

        for p in [backbone_path, raw_path, comp_path]:
            if not p.exists():
                raise ValueError(f"Missing file: {p}")

        backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        comp = torch.load(comp_path, map_location="cpu", weights_only=False)

        embeddings = backbone["embeddings"].float()
        future_n = raw["futures_n"].float()
        trend_n = comp["trend_n"].float()
        seasonal_n = comp["seasonal_n"].float()

        # Load fine_mask basis (or on-the-fly)
        bases = resolve_basis_for_dir(ds_dir, granularity, horizon, context_len)

        # Existing nonF caches have 3-family basis files; monthly may have been added via upgrade
        payload = {
            "embeddings": embeddings,
            "future_n": future_n,
            "trend_n": trend_n,
            "seasonal_n": seasonal_n,
            "daily_basis": _pad_width(bases["daily"], "daily"),
            "weekly_basis": _pad_width(bases["weekly"], "weekly"),
            "monthly_basis": _pad_width(bases["monthly"], "monthly"),
            "yearly_basis": _pad_width(bases["yearly"], "yearly"),
        }
        cls.cache[key] = payload
        return payload


class NonFDataset(Dataset):
    def __init__(self, items: list[tuple[Path, dict]], split: str, val_split: float,
                 length: int, seed: int):
        self.groups = []
        self.length = int(length)
        self.seed = int(seed)
        for ds_dir, meta in items:
            try:
                payload = NonFPayloadCache.load(ds_dir, meta)
            except (ValueError, FileNotFoundError) as exc:
                print(f"[nonf] skip {ds_dir.name}: {exc}", flush=True)
                continue
            n = int(payload["future_n"].shape[0])
            split_at = max(1, min(int(n * (1.0 - val_split)), n - 1))
            indices = np.arange(0, split_at) if split == "train" else np.arange(split_at, n)
            if len(indices) > 0:
                self.groups.append((payload, indices))
        if not self.groups:
            raise ValueError(f"No {split} samples found")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + int(idx))
        payload, indices = self.groups[int(rng.integers(0, len(self.groups)))]
        i = int(indices[int(rng.integers(0, len(indices)))])
        return {
            "emb": payload["embeddings"][i],
            "future_n": payload["future_n"][i],
            "trend_n": payload["trend_n"][i],
            "seasonal_n": payload["seasonal_n"][i],
            "daily_basis": payload["daily_basis"],
            "weekly_basis": payload["weekly_basis"],
            "monthly_basis": payload["monthly_basis"],
            "yearly_basis": payload["yearly_basis"],
        }


def nonf_collate(batch: list[dict]) -> dict:
    keys = ["emb", "future_n", "trend_n", "seasonal_n",
            "daily_basis", "weekly_basis", "monthly_basis", "yearly_basis"]
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def train_nonf(model: FuncDecModel, items: list[tuple[Path, dict]], steps: int,
               args: argparse.Namespace, horizon: int, device: torch.device) -> list[dict]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(steps))

    ds = NonFDataset(items, "train", 0.1,
                     length=int(steps) * args.nonf_batch_size,
                     seed=args.seed + horizon * 17)
    loader = DataLoader(
        ds,
        batch_size=args.nonf_batch_size,
        shuffle=False,
        collate_fn=nonf_collate,
        **dataloader_kwargs(args, device),
    )

    history = []
    model.train()
    step = 0
    for batch in loader:
        emb = batch["emb"].to(device, non_blocking=True)
        daily = batch["daily_basis"].to(device, non_blocking=True)
        weekly = batch["weekly_basis"].to(device, non_blocking=True)
        monthly = batch["monthly_basis"].to(device, non_blocking=True)
        yearly = batch["yearly_basis"].to(device, non_blocking=True)
        future_n = batch["future_n"].to(device, non_blocking=True)
        trend_n = batch["trend_n"].to(device, non_blocking=True)
        seasonal_n = batch["seasonal_n"].to(device, non_blocking=True)

        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        clean_total = trend_n + seasonal_n
        clean_pred = decomp["trend"] + decomp["seasonal"] + decomp["residual"]
        loss = (
            F.l1_loss(clean_pred, clean_total)
            + F.l1_loss(decomp["trend"], trend_n)
            + F.l1_loss(decomp["seasonal"] + decomp["residual"], seasonal_n)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        if step == 1 or step % args.log_every == 0 or step == int(steps):
            value = float(loss.item())
            history.append({"phase": "nonf", "step": step, "loss": value})
            print(f"[nonf] h{horizon} step {step}/{steps} loss={value:.6g}", flush=True)
        if step >= int(steps):
            break

    NonFPayloadCache.clear()
    return history


# ---------------------------------------------------------------------------
# Real data training
# ---------------------------------------------------------------------------

def load_domain_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def discover_real_group(cache_root: Path, subset: str, horizon: int) -> dict | None:
    cache_dir = cache_root / subset
    if not cache_dir.exists():
        return None
    future_paths = sorted(cache_dir.glob(f"futures_c*_h{horizon}_*.pt"))
    if not future_paths:
        return None
    future_path = future_paths[0]
    match = FUTURE_RE.match(future_path.name)
    if not match:
        return None
    context_len = int(match.group("context"))
    freq = match.group("freq")
    backbone_paths = sorted(cache_dir.glob(f"backbone_emb_c{context_len}_*.pt"))
    if not backbone_paths:
        return None
    future_payload = torch.load(future_path, map_location="cpu", weights_only=False)
    valid_mask = future_payload.get("valid_mask")
    if valid_mask is not None and not bool(valid_mask.bool().any()):
        return None
    return {
        "subset": subset,
        "cache_dir": str(cache_dir),
        "context_len": context_len,
        "freq": freq,
        "horizon": int(horizon),
        "backbone_path": str(backbone_paths[0]),
        "future_path": str(future_path),
    }


class RealPayloadCache:
    def __init__(self, max_items: int = 1):
        self.max_items = int(max_items)
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def clear(self) -> None:
        self.cache.clear()
        gc.collect()

    def load(self, group: dict) -> dict:
        key = f"{group['cache_dir']}::h{group['horizon']}"
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        backbone = torch.load(group["backbone_path"], map_location="cpu", weights_only=False)
        futures = torch.load(group["future_path"], map_location="cpu", weights_only=False)
        embeddings = backbone["embeddings"].float()
        future_n = futures["futures_n"].float()
        finite = torch.isfinite(embeddings).all(dim=1) & torch.isfinite(future_n).all(dim=1)
        valid_mask = futures.get("valid_mask")
        if valid_mask is not None:
            finite = finite & valid_mask.bool()
        indices = finite.nonzero(as_tuple=True)[0].cpu().numpy()
        if len(indices) == 0:
            raise ValueError(f"No valid samples in {group['cache_dir']} h{group['horizon']}")

        bases = resolve_basis_for_real(group)
        payload = {"embeddings": embeddings, "future_n": future_n,
                   "indices": indices, "bases": bases}
        self.cache[key] = payload
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
            gc.collect()
        return payload


def sample_real_batch(payload: dict, batch_size: int, seed: int, step: int, device: torch.device):
    rng = np.random.default_rng(seed + step * 1009)
    chosen = rng.choice(payload["indices"], size=int(batch_size),
                        replace=len(payload["indices"]) < batch_size)
    chosen_t = torch.as_tensor(chosen, dtype=torch.long)
    emb = payload["embeddings"].index_select(0, chosen_t).to(device, non_blocking=True)
    future_n = payload["future_n"].index_select(0, chosen_t).to(device, non_blocking=True)
    daily, weekly, monthly, yearly = expand_bases(payload["bases"], emb.shape[0], device)
    return emb, future_n, daily, weekly, monthly, yearly


EVAL_TARGETS = {
    "elecdemand", "ETTh1", "oikolab_weather", "saugeenday", "us_births",
    "PEMS03", "pedestrian_counts", "alibaba_cluster_trace_2018",
}


def train_real(model: FuncDecModel, groups: list[dict], steps: int,
               args: argparse.Namespace, horizon: int, device: torch.device) -> list[dict]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(steps))
    payload_cache = RealPayloadCache(max_items=1)
    chunk_steps = args.real_group_chunk_steps
    history = []
    model.train()
    active_groups = list(groups)

    for step in range(1, int(steps) + 1):
        group = active_groups[((step - 1) // max(1, chunk_steps)) % len(active_groups)]
        try:
            payload = payload_cache.load(group)
        except ValueError as exc:
            print(f"[real] skip h{horizon} subset={group['subset']}: {exc}", flush=True)
            active_groups = [g for g in active_groups if g["subset"] != group["subset"]]
            if not active_groups:
                raise
            continue

        emb, future_n, daily, weekly, monthly, yearly = sample_real_batch(
            payload, args.real_batch_size, args.seed + horizon * 13, step, device
        )
        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        loss = F.l1_loss(pred, future_n)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == int(steps):
            no_res = float(F.l1_loss(decomp["trend"] + decomp["seasonal"], future_n).item())
            value = float(loss.item())
            history.append({
                "phase": "real", "step": step, "loss": value,
                "subset": group["subset"], "no_residual_mae": no_res,
            })
            print(
                f"[real] h{horizon} step {step}/{steps} subset={group['subset']} "
                f"loss={value:.6g} no_res={no_res:.6g}",
                flush=True,
            )

    payload_cache.clear()
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_horizon(horizon: int, args: argparse.Namespace, device: torch.device) -> dict:
    ckpt_dir = args.results_root / "checkpoints"
    final_path = ckpt_dir / f"funcdec_h{horizon}.pt"
    if args.skip_existing and final_path.exists():
        return {"horizon": horizon, "skipped": True}

    print(f"\n=== h{horizon} loading initial model ===", flush=True)
    model, init_ckpt, cfg = load_initial_model(horizon, args, device)

    # Enable all decoders for training
    for param in model.parameters():
        param.requires_grad = True
    # Freeze backbone if present
    if model.backbone is not None:
        for param in model.backbone.parameters():
            param.requires_grad = False

    history = {}
    started = time.perf_counter()

    # Phase 1: Fourier synth
    if not args.skip_fourier and args.fourier_steps > 0:
        groups = discover_fourier_groups(horizon)
        if groups:
            print(f"[fourier] h{horizon}: {len(groups)} groups, {args.fourier_steps} steps", flush=True)
            history["fourier"] = train_fourier_synth(model, groups, args.fourier_steps,
                                                      args, horizon, device)
            ckpt_fourier = ckpt_dir / f"fourier_synth_h{horizon}.pt"
            save_checkpoint(model, cfg, ckpt_fourier, {
                "phase": "fourier_synth",
                "args": to_jsonable(vars(args)),
                "initial_checkpoint": str(init_ckpt),
            })
        else:
            print(f"[fourier] h{horizon}: no groups found, skipping", flush=True)

    # Phase 2: non-Fourier synth
    if not args.skip_nonf and args.nonf_steps > 0:
        nonf_items = discover_nonf_groups(horizon, args.nonf_stages)
        if nonf_items:
            print(f"[nonf] h{horizon}: {len(nonf_items)} groups, {args.nonf_steps} steps", flush=True)
            history["nonf"] = train_nonf(model, nonf_items, args.nonf_steps, args, horizon, device)
            ckpt_nonf = ckpt_dir / f"nonf_h{horizon}.pt"
            save_checkpoint(model, cfg, ckpt_nonf, {
                "phase": "nonf",
                "args": to_jsonable(vars(args)),
            })
        else:
            print(f"[nonf] h{horizon}: no groups found, skipping", flush=True)

    # Phase 3: Real data
    if not args.skip_real and args.real_steps > 0 and args.domain_config.exists():
        domain_cfg = load_domain_config(args.domain_config)["targets"]
        all_subsets = sorted({
            subset
            for target_cfg in domain_cfg.values()
            for subset in target_cfg.get("train_subsets", [])
        } - EVAL_TARGETS)
        real_groups = [
            g for subset in all_subsets
            if (g := discover_real_group(args.lotsa_cache_root, subset, horizon)) is not None
        ]
        if real_groups:
            print(f"[real] h{horizon}: {len(real_groups)} groups, {args.real_steps} steps", flush=True)
            history["real"] = train_real(model, real_groups, args.real_steps, args, horizon, device)
        else:
            print(f"[real] h{horizon}: no groups found, skipping", flush=True)

    save_checkpoint(model, cfg, final_path, {
        "phase": "complete",
        "args": to_jsonable(vars(args)),
        "initial_checkpoint": str(init_ckpt),
        "history": to_jsonable(history),
        "elapsed_sec": time.perf_counter() - started,
    })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "horizon": int(horizon),
        "initial_checkpoint": str(init_ckpt),
        "checkpoint": str(final_path),
        "elapsed_sec": time.perf_counter() - started,
        "history": to_jsonable(history),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.results_root.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} results={args.results_root}", flush=True)
    print(f"[train] init_checkpoint_dir={args.init_checkpoint_dir}", flush=True)

    result = {"args": to_jsonable(vars(args)), "per_horizon": {}}
    for horizon in args.horizons:
        try:
            result["per_horizon"][str(horizon)] = train_horizon(int(horizon), args, device)
        except Exception as exc:
            print(f"[train] ERROR h{horizon}: {exc}", flush=True)
            result["per_horizon"][str(horizon)] = {"error": str(exc)}
        with (args.results_root / "train_result_partial.json").open("w") as f:
            json.dump(to_jsonable(result), f, indent=2)

    with (args.results_root / "train_result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    print(f"[train] complete. Saved {args.results_root / 'train_result.json'}", flush=True)


if __name__ == "__main__":
    main()
