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

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = next(parent for parent in THIS_FILE.parents if (parent / "loader_utils.py").exists())
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import resolve_data_path, resolve_project_path  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


THIS_DIR = Path(__file__).resolve().parent
OLD_EXP_DIR = resolve_project_path("/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent")
PROJECT_ROOT = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT / "src"
for path in [PROJECT_ROOT, SRC_DIR, OLD_EXP_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model.decomp_funcdec import FuncDecModel  # noqa: E402
from single_model_eval_common import load_basis_file, build_fourier_basis, to_jsonable  # noqa: E402


DEFAULT_INIT_CHECKPOINT_DIR = Path(
    "/home/sia2/project/5.22syn_cent/train_nonF_rawtarget/results/train/nonfourier_finetune_from_simple_complex"
)
DEFAULT_LOTSA_CACHE_ROOT = resolve_data_path("/home/sia2/project/data/data_lotsa/lotsa_cache")
DEFAULT_RESULTS_DIR = THIS_DIR / "results" / "train" / "domain_real_finetune"
DEFAULT_DOMAIN_CONFIG = THIS_DIR / "domain_config.json"
HORIZONS = [96, 192, 336, 720]
FUTURE_RE = re.compile(r"futures_c(?P<context>\d+)_(?P<freq>.+)_h(?P<horizon>\d+)_.+\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune synthetic-pretrained FuncDec on same-domain real LOTSA caches.")
    parser.add_argument("--domain_config", type=Path, default=DEFAULT_DOMAIN_CONFIG)
    parser.add_argument("--lotsa_cache_root", type=Path, default=DEFAULT_LOTSA_CACHE_ROOT)
    parser.add_argument("--init_checkpoint_dir", type=Path, default=DEFAULT_INIT_CHECKPOINT_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--group_chunk_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def checkpoint_candidates(run_dir: Path, horizon: int) -> list[Path]:
    return [
        run_dir / "checkpoints" / f"funcdec_h{horizon}.pt",
        run_dir / "checkpoints" / f"nonfourier_finetune_h{horizon}.pt",
    ]


def load_initial_model(horizon: int, args: argparse.Namespace, device: torch.device) -> tuple[FuncDecModel, Path, dict]:
    found = next((path for path in checkpoint_candidates(args.init_checkpoint_dir, horizon) if path.exists()), None)
    if found is None:
        tried = "\n".join(str(path) for path in checkpoint_candidates(args.init_checkpoint_dir, horizon))
        raise FileNotFoundError(f"No initial checkpoint for h{horizon}. Tried:\n{tried}")
    payload = torch.load(found, map_location=device, weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    cfg = dict(payload.get("config", {})) if isinstance(payload, dict) else {}
    cfg["horizon"] = int(horizon)
    model = FuncDecModel(cfg, load_backbone=False).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if key.startswith("decoder_")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("backbone.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: {found} missing={missing[:8]} unexpected={unexpected[:8]}")
    return model, found, cfg


def discover_group(cache_root: Path, subset: str, horizon: int) -> dict | None:
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
    basis_paths = sorted(cache_dir.glob(f"fourier_basis_c{context_len}_{freq}_h{horizon}_*.pt"))
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
        "basis_path": None if not basis_paths else str(basis_paths[0]),
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
        if group.get("basis_path"):
            bases = load_basis_file(Path(group["basis_path"]))
        else:
            bases = build_fourier_basis(str(group["freq"]), int(group["context_len"]), int(group["horizon"]))
        payload = {"embeddings": embeddings, "future_n": future_n, "indices": indices, "bases": bases}
        self.cache[key] = payload
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
            gc.collect()
        return payload


def sample_batch(payload: dict, batch_size: int, seed: int, step: int, device: torch.device):
    rng = np.random.default_rng(seed + step * 1009)
    chosen = rng.choice(payload["indices"], size=int(batch_size), replace=len(payload["indices"]) < batch_size)
    chosen_t = torch.as_tensor(chosen, dtype=torch.long)
    emb = payload["embeddings"].index_select(0, chosen_t).to(device, non_blocking=True)
    future_n = payload["future_n"].index_select(0, chosen_t).to(device, non_blocking=True)
    bases = payload["bases"]
    daily = bases["daily"].to(device).unsqueeze(0).expand(emb.shape[0], -1, -1)
    weekly = bases["weekly"].to(device).unsqueeze(0).expand(emb.shape[0], -1, -1)
    yearly = bases["yearly"].to(device).unsqueeze(0).expand(emb.shape[0], -1, -1)
    return emb, future_n, daily, weekly, yearly


def train_one(target: str, target_cfg: dict, horizon: int, groups: list[dict], args: argparse.Namespace, device: torch.device) -> dict:
    model, init_ckpt, model_cfg = load_initial_model(horizon, args, device)
    for param in model.parameters():
        param.requires_grad = False
    params = []
    for module in [model.decoder_t, model.decoder_s, model.decoder_r]:
        for param in module.parameters():
            param.requires_grad = True
            params.append(param)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    payload_cache = RealPayloadCache(max_items=1)
    history = []
    started = time.perf_counter()
    model.train()

    active_groups = list(groups)
    step = 1
    while step <= int(args.max_steps):
        group = active_groups[((step - 1) // max(1, args.group_chunk_steps)) % len(active_groups)]
        try:
            payload = payload_cache.load(group)
        except ValueError as exc:
            print(f"[train-real] skip_invalid target={target} h{horizon} subset={group['subset']} reason={exc}", flush=True)
            active_groups = [item for item in active_groups if item["subset"] != group["subset"]]
            if not active_groups:
                raise
            continue
        emb, future_n, daily, weekly, yearly = sample_batch(
            payload, args.batch_size, args.seed + horizon * 17 + len(target), step, device
        )
        pred, _decomp = model(emb, daily, weekly, yearly)
        loss = F.l1_loss(pred, future_n)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            value = float(loss.item())
            history.append({"step": step, "loss": value, "subset": group["subset"]})
            print(f"[train-real] {target} h{horizon} step {step}/{args.max_steps} subset={group['subset']} loss={value:.6g}", flush=True)
        step += 1

    ckpt_path = None
    if args.save_checkpoint:
        ckpt_dir = args.results_dir / target / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"domain_real_h{horizon}.pt"
        torch.save(
            {
                "config": model_cfg,
                "args": to_jsonable(vars(args)),
                "target_dataset": target,
                "target_domain": target_cfg.get("domain"),
                "train_groups": groups,
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        eval_ckpt = ckpt_dir / f"funcdec_h{horizon}.pt"
        if eval_ckpt.exists() or eval_ckpt.is_symlink():
            eval_ckpt.unlink()
        eval_ckpt.symlink_to(ckpt_path.name)

    elapsed = time.perf_counter() - started
    payload_cache.clear()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "target": target,
        "domain": target_cfg.get("domain"),
        "horizon": int(horizon),
        "initial_checkpoint": str(init_ckpt),
        "checkpoint_path": None if ckpt_path is None else str(ckpt_path),
        "train_groups": active_groups,
        "history": history,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = parse_args()
    if args.max_steps >= 3000:
        raise ValueError("--max_steps must be below 3000 for this experiment")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(args.domain_config)["targets"]
    targets = args.targets or list(config)
    result = {"args": to_jsonable(vars(args)), "per_target": {}}

    for target in targets:
        if target not in config:
            raise KeyError(f"Unknown target in domain_config: {target}")
        target_cfg = config[target]
        result["per_target"].setdefault(target, {})
        for horizon in args.horizons:
            existing_ckpt = args.results_dir / target / "checkpoints" / f"funcdec_h{int(horizon)}.pt"
            if args.skip_existing and existing_ckpt.exists():
                print(f"skip_existing target={target} h{horizon} checkpoint={existing_ckpt}", flush=True)
                continue
            groups = [
                group
                for subset in target_cfg["train_subsets"]
                if (group := discover_group(args.lotsa_cache_root, subset, int(horizon))) is not None
            ]
            if not groups:
                raise FileNotFoundError(f"No train groups for {target} h{horizon}")
            print(f"=== target={target} domain={target_cfg.get('domain')} h{horizon} groups={len(groups)} ===", flush=True)
            result["per_target"][target][str(horizon)] = train_one(target, target_cfg, int(horizon), groups, args, device)
            with (args.results_dir / "result_partial.json").open("w") as f:
                json.dump(to_jsonable(result), f, indent=2)

    with (args.results_dir / "result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    print(f"[train-real] saved {args.results_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
