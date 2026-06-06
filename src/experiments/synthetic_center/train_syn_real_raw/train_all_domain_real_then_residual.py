from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


THIS_DIR = Path(__file__).resolve().parent
OLD_EXP_DIR = Path("/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent")
PROJECT_ROOT = Path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT / "src"
for path in [THIS_DIR, PROJECT_ROOT, SRC_DIR, OLD_EXP_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model.decomp_funcdec import FuncDecModel  # noqa: E402
from single_model_eval_common import DEFAULT_CONFIG, to_jsonable  # noqa: E402
from train_domain_real_finetune import (  # noqa: E402
    DEFAULT_DOMAIN_CONFIG,
    DEFAULT_LOTSA_CACHE_ROOT,
    HORIZONS,
    RealPayloadCache,
    discover_group,
    load_json,
    sample_batch,
)


DEFAULT_INIT_CHECKPOINT_DIR = Path(
    "/home/sia2/project/5.22syn_cent/train_nonF_rawtarget/results/train/nonfourier_finetune_from_simple_complex"
)
DEFAULT_RESULTS_ROOT = THIS_DIR / "results" / "all_domain_full_then_residual"
EVAL_TARGETS = {
    "elecdemand",
    "ETTh1",
    "oikolab_weather",
    "saugeenday",
    "us_births",
    "PEMS03",
    "pedestrian_counts",
    "alibaba_cluster_trace_2018",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one all-domain FuncDec model, then residual-only extra train.")
    parser.add_argument("--domain_config", type=Path, default=DEFAULT_DOMAIN_CONFIG)
    parser.add_argument("--lotsa_cache_root", type=Path, default=DEFAULT_LOTSA_CACHE_ROOT)
    parser.add_argument("--init_checkpoint_dir", type=Path, default=DEFAULT_INIT_CHECKPOINT_DIR)
    parser.add_argument("--random_init", action="store_true", help="Start from freshly initialized decoders instead of a synthetic-pretrained checkpoint.")
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--full_steps", type=int, default=10000)
    parser.add_argument("--residual_steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--full_group_chunk_steps", type=int, default=250)
    parser.add_argument("--residual_group_chunk_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def checkpoint_candidates(run_dir: Path, horizon: int) -> list[Path]:
    return [
        run_dir / "checkpoints" / f"funcdec_h{horizon}.pt",
        run_dir / "checkpoints" / f"nonfourier_finetune_h{horizon}.pt",
        run_dir / "checkpoints" / f"simple_complex_synth_h{horizon}.pt",
    ]


def load_model_from_dir(run_dir: Path, horizon: int, device: torch.device) -> tuple[FuncDecModel, Path, dict]:
    found = next((path for path in checkpoint_candidates(run_dir, horizon) if path.exists()), None)
    if found is None:
        tried = "\n".join(str(path) for path in checkpoint_candidates(run_dir, horizon))
        raise FileNotFoundError(f"No checkpoint for h{horizon}. Tried:\n{tried}")
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


def init_random_model(horizon: int, args: argparse.Namespace, device: torch.device) -> tuple[FuncDecModel, str, dict]:
    torch.manual_seed(int(args.seed) + int(horizon))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + int(horizon))
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["horizon"] = int(horizon)
    model = FuncDecModel(cfg, load_backbone=False).to(device)
    return model, "random_init", cfg


def save_checkpoint(model: FuncDecModel, cfg: dict, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict(), **payload}, path)
    eval_path = path.parent / f"funcdec_h{int(cfg['horizon'])}.pt"
    if eval_path.exists() or eval_path.is_symlink():
        eval_path.unlink()
    eval_path.symlink_to(path.name)


def all_train_subsets(args: argparse.Namespace) -> list[str]:
    cfg = load_json(args.domain_config)["targets"]
    subsets = []
    for target_cfg in cfg.values():
        subsets.extend(target_cfg["train_subsets"])
    return sorted(set(subsets) - EVAL_TARGETS)


def discover_groups(args: argparse.Namespace, horizon: int) -> list[dict]:
    groups = [
        group
        for subset in all_train_subsets(args)
        if (group := discover_group(args.lotsa_cache_root, subset, int(horizon))) is not None
    ]
    if not groups:
        raise FileNotFoundError(f"No train groups for h{horizon}")
    return groups


def set_trainable(model: FuncDecModel, residual_only: bool) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    modules = [model.decoder_r] if residual_only else [model.decoder_t, model.decoder_s, model.decoder_r]
    params = []
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
            params.append(param)
    return params


def train_loop(
    model: FuncDecModel,
    groups: list[dict],
    steps: int,
    chunk_steps: int,
    residual_only: bool,
    args: argparse.Namespace,
    horizon: int,
    device: torch.device,
) -> list[dict]:
    params = set_trainable(model, residual_only)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(steps))
    payload_cache = RealPayloadCache(max_items=1)
    active_groups = list(groups)
    history = []
    phase = "residual" if residual_only else "full"
    model.train()
    step = 1
    while step <= int(steps):
        group = active_groups[((step - 1) // max(1, int(chunk_steps))) % len(active_groups)]
        try:
            payload = payload_cache.load(group)
        except ValueError as exc:
            print(f"[all-domain-{phase}] skip_invalid h{horizon} subset={group['subset']} reason={exc}", flush=True)
            active_groups = [item for item in active_groups if item["subset"] != group["subset"]]
            if not active_groups:
                raise
            continue
        emb, future_n, daily, weekly, yearly = sample_batch(
            payload, args.batch_size, args.seed + horizon * 17 + (1 if residual_only else 0), step, device
        )
        pred, decomp = model(emb, daily, weekly, yearly)
        loss = F.l1_loss(pred, future_n)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == int(steps):
            no_res = float(F.l1_loss(decomp["trend"] + decomp["seasonal"], future_n).item())
            value = float(loss.item())
            row = {
                "phase": phase,
                "step": step,
                "loss": value,
                "subset": group["subset"],
                "no_residual_mae": no_res,
                "residual_gain": no_res - value,
                "residual_std": float(decomp["residual"].detach().float().std().item()),
            }
            history.append(row)
            print(
                f"[all-domain-{phase}] h{horizon} step {step}/{steps} subset={group['subset']} "
                f"loss={value:.6g} no_res={no_res:.6g} gain={row['residual_gain']:.6g} "
                f"res_std={row['residual_std']:.6g}",
                flush=True,
            )
        step += 1
    payload_cache.clear()
    return history


def train_horizon(horizon: int, args: argparse.Namespace, device: torch.device) -> dict:
    groups = discover_groups(args, horizon)
    full_dir = args.results_root / "train" / "all_domain_full"
    residual_dir = args.results_root / "train" / "all_domain_full_residual_extra"
    residual_ckpt = residual_dir / "checkpoints" / f"funcdec_h{horizon}.pt"
    if args.skip_existing and residual_ckpt.exists():
        return {"horizon": horizon, "skipped": True, "checkpoint_path": str(residual_ckpt)}

    if args.random_init:
        model, init_ckpt, cfg = init_random_model(horizon, args, device)
    else:
        model, init_ckpt, cfg = load_model_from_dir(args.init_checkpoint_dir, horizon, device)
    started = time.perf_counter()
    full_history = train_loop(
        model,
        groups,
        args.full_steps,
        args.full_group_chunk_steps,
        False,
        args,
        horizon,
        device,
    )
    full_path = full_dir / "checkpoints" / f"all_domain_full_h{horizon}.pt"
    save_checkpoint(
        model,
        cfg,
        full_path,
        {
            "args": to_jsonable(vars(args)),
            "train_groups": groups,
            "initial_checkpoint": str(init_ckpt),
            "trainable": "trend_seasonal_residual_decoders",
        },
    )

    residual_history = train_loop(
        model,
        groups,
        args.residual_steps,
        args.residual_group_chunk_steps,
        True,
        args,
        horizon,
        device,
    )
    residual_path = residual_dir / "checkpoints" / f"all_domain_full_residual_extra_h{horizon}.pt"
    save_checkpoint(
        model,
        cfg,
        residual_path,
        {
            "args": to_jsonable(vars(args)),
            "train_groups": groups,
            "initial_checkpoint": str(full_path),
            "trainable": "residual_decoder_only_after_full",
        },
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "horizon": int(horizon),
        "num_groups": len(groups),
        "groups": [group["subset"] for group in groups],
        "initial_checkpoint": str(init_ckpt),
        "full_checkpoint": str(full_path),
        "residual_checkpoint": str(residual_path),
        "full_history": full_history,
        "residual_history": residual_history,
        "elapsed_sec": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.results_root.mkdir(parents=True, exist_ok=True)
    result = {"args": to_jsonable(vars(args)), "excluded_eval_targets": sorted(EVAL_TARGETS), "per_horizon": {}}
    print(f"[all-domain] train_subsets={len(all_train_subsets(args))} excluded={sorted(EVAL_TARGETS)}", flush=True)
    for horizon in args.horizons:
        print(f"=== all-domain h{horizon} ===", flush=True)
        result["per_horizon"][str(horizon)] = train_horizon(int(horizon), args, device)
        with (args.results_root / "result_partial.json").open("w") as f:
            json.dump(to_jsonable(result), f, indent=2)
    with (args.results_root / "result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    print(f"[all-domain] saved {args.results_root / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
