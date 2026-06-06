from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from train_domain_real_finetune import (  # noqa: E402
    DEFAULT_DOMAIN_CONFIG,
    DEFAULT_LOTSA_CACHE_ROOT,
    HORIZONS,
    RealPayloadCache,
    discover_group,
    load_initial_model,
    load_json,
    sample_batch,
)

OLD_EXP_DIR = Path("/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent")
if str(OLD_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_EXP_DIR))
from single_model_eval_common import to_jsonable  # noqa: E402


DEFAULT_INIT_CHECKPOINT_DIR = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/train/domain_real_finetune_phasefix"
)
DEFAULT_RESULTS_DIR = Path(
    "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/full_resi_extra_train/train/residual_extra_from_full_phasefix"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only residual decoder on top of full real-adapted checkpoints.")
    parser.add_argument("--domain_config", type=Path, default=DEFAULT_DOMAIN_CONFIG)
    parser.add_argument("--lotsa_cache_root", type=Path, default=DEFAULT_LOTSA_CACHE_ROOT)
    parser.add_argument("--init_checkpoint_dir", type=Path, default=DEFAULT_INIT_CHECKPOINT_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_steps_limit", type=int, default=0, help="Optional safety limit. 0 disables the limit.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--group_chunk_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def init_from_target_checkpoint(horizon: int, target: str, args: argparse.Namespace, device: torch.device):
    target_args = argparse.Namespace(init_checkpoint_dir=args.init_checkpoint_dir / target)
    return load_initial_model(horizon, target_args, device)


def train_one(target: str, target_cfg: dict, horizon: int, groups: list[dict], args: argparse.Namespace, device: torch.device) -> dict:
    model, init_ckpt, model_cfg = init_from_target_checkpoint(horizon, target, args, device)
    for param in model.parameters():
        param.requires_grad = False
    params = []
    for param in model.decoder_r.parameters():
        param.requires_grad = True
        params.append(param)
    if not params:
        raise ValueError("No residual parameters selected")

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    payload_cache = RealPayloadCache(max_items=1)
    active_groups = list(groups)
    history = []
    started = time.perf_counter()
    model.train()

    step = 1
    while step <= int(args.max_steps):
        group = active_groups[((step - 1) // max(1, args.group_chunk_steps)) % len(active_groups)]
        try:
            payload = payload_cache.load(group)
        except ValueError as exc:
            print(f"[train-residual-extra] skip_invalid target={target} h{horizon} subset={group['subset']} reason={exc}", flush=True)
            active_groups = [item for item in active_groups if item["subset"] != group["subset"]]
            if not active_groups:
                raise
            continue

        emb, future_n, daily, weekly, yearly = sample_batch(
            payload, args.batch_size, args.seed + horizon * 17 + len(target), step, device
        )
        pred, decomp = model(emb, daily, weekly, yearly)
        no_residual = decomp["trend"] + decomp["seasonal"]
        loss = F.l1_loss(pred, future_n)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            value = float(loss.item())
            no_residual_mae = float(F.l1_loss(no_residual, future_n).item())
            residual_gain = no_residual_mae - value
            residual_std = float(decomp["residual"].detach().float().std().item())
            history.append(
                {
                    "step": step,
                    "loss": value,
                    "no_residual_mae": no_residual_mae,
                    "residual_gain": residual_gain,
                    "residual_std": residual_std,
                    "subset": group["subset"],
                }
            )
            print(
                f"[train-residual-extra] {target} h{horizon} step {step}/{args.max_steps} "
                f"subset={group['subset']} loss={value:.6g} no_res={no_residual_mae:.6g} "
                f"gain={residual_gain:.6g} res_std={residual_std:.6g}",
                flush=True,
            )
        step += 1

    ckpt_path = None
    if args.save_checkpoint:
        ckpt_dir = args.results_dir / target / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"residual_extra_h{horizon}.pt"
        torch.save(
            {
                "config": model_cfg,
                "args": to_jsonable(vars(args)),
                "target_dataset": target,
                "target_domain": target_cfg.get("domain"),
                "train_groups": active_groups,
                "init_checkpoint": str(init_ckpt),
                "trainable": "residual_decoder_only",
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        eval_ckpt = ckpt_dir / f"funcdec_h{horizon}.pt"
        if eval_ckpt.exists() or eval_ckpt.is_symlink():
            eval_ckpt.unlink()
        eval_ckpt.symlink_to(ckpt_path.name)

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
        "elapsed_sec": time.perf_counter() - started,
        "trainable": "residual_decoder_only",
    }


def main() -> None:
    args = parse_args()
    if args.max_steps_limit > 0 and args.max_steps > args.max_steps_limit:
        raise ValueError(f"--max_steps must be <= {args.max_steps_limit} for this experiment")
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
            print(f"=== residual-extra target={target} domain={target_cfg.get('domain')} h{horizon} groups={len(groups)} ===", flush=True)
            result["per_target"][target][str(horizon)] = train_one(target, target_cfg, int(horizon), groups, args, device)
            with (args.results_dir / "result_partial.json").open("w") as f:
                json.dump(to_jsonable(result), f, indent=2)

    with (args.results_dir / "result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    print(f"[train-residual-extra] saved {args.results_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
