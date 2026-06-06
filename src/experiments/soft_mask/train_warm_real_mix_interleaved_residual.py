#!/usr/bin/env python3
"""Fourier-warm + interleaved residual training for soft_mask.

Training order:
  1. Short Fourier-only warmup.
  2. Full-decoder mixed burn-in with real batches and sparse Fourier batches.
  3. Alternate full-decoder mixed training and residual-only real training.

This intentionally excludes non-Fourier synthetic data from training.  The
trend-seasonal correlation penalty is kept as an optional term, while all
residual correlation penalties are removed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import train as base
from common import HORIZONS, to_jsonable
from model.decomp_funcdec import FuncDecModel


DEFAULT_INIT_CHECKPOINT_DIR = Path("none")
DEFAULT_RESULTS_ROOT = base.resolve_project_path("/home/sia2/project/5.30soft_mask/results/interleaved_residual_mix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint_dir", type=Path, default=DEFAULT_INIT_CHECKPOINT_DIR)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--lotsa_cache_root", type=Path, default=base.DEFAULT_LOTSA_CACHE_ROOT)
    parser.add_argument("--domain_config", type=Path,
                        default=base.REAL_TRAIN_DIR / "domain_config.json")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)

    parser.add_argument("--fourier_warmup_steps", type=int, default=500)
    parser.add_argument("--mixed_steps", type=int, default=10000)
    parser.add_argument("--full_burnin_steps", type=int, default=1500,
                        help="Run this many full mixed steps before residual-only cycles.")
    parser.add_argument("--cycle_full_steps", type=int, default=1000,
                        help="Full mixed steps per interleaved cycle after burn-in.")
    parser.add_argument("--cycle_residual_steps", type=int, default=250,
                        help="Residual-only real steps per interleaved cycle.")
    parser.add_argument("--synth_interval", type=int, default=10,
                        help="Use one Fourier batch every N mixed steps. N=10 gives real:synth=9:1.")
    parser.add_argument("--real_group_chunk_steps", type=int, default=250)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fourier_batch_size", type=int, default=None,
                        help="Defaults to --batch_size; kept for compatibility with base Fourier trainer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--residual_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ts_corr_weight", type=float, default=0.01,
                        help="Optional trend-seasonal squared-correlation penalty for real batches only.")
    parser.add_argument("--gate_l1_weight", type=float, default=1e-3,
                        help="Soft-mask gate sparsity regularization.")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--skip_existing", action="store_true")
    base.add_runtime_args(parser)
    parser.add_argument("--allow_checkpoint_init", action="store_true",
                        help="Allow warm-starting from --init_checkpoint_dir. Disabled by default for fair scratch comparisons.")
    return parser.parse_args()


def save_checkpoint(model: FuncDecModel, cfg: dict, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    torch.save({"config": cfg, "state_dict": model.state_dict(), **payload}, path)


def component_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_c = x - x.mean(dim=1, keepdim=True)
    y_c = y - y.mean(dim=1, keepdim=True)
    r = (x_c * y_c).sum(dim=1) / (x_c.norm(dim=1) * y_c.norm(dim=1) + 1e-8)
    return (r ** 2).mean()


def set_all_decoders_trainable(model: FuncDecModel) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if model.backbone is not None:
        for param in model.backbone.parameters():
            param.requires_grad = False


def set_residual_only_trainable(model: FuncDecModel) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.decoder_r.parameters():
        param.requires_grad = True


def trainable_params(model: FuncDecModel) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def discover_real_groups(args: argparse.Namespace, horizon: int) -> list[dict]:
    domain_cfg = base.load_domain_config(args.domain_config)["targets"]
    all_subsets = sorted({
        subset
        for target_cfg in domain_cfg.values()
        for subset in target_cfg.get("train_subsets", [])
    } - base.EVAL_TARGETS)
    return [
        group for subset in all_subsets
        if (group := base.discover_real_group(args.lotsa_cache_root, subset, horizon)) is not None
    ]


def train_mixed_full(model: FuncDecModel, real_groups: list[dict], synth_groups: list[dict],
                     steps: int, args: argparse.Namespace, horizon: int,
                     device: torch.device, start_step: int = 0,
                     phase_prefix: str = "mixed") -> list[dict]:
    set_all_decoders_trainable(model)
    params = trainable_params(model)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(steps)))
    real_cache = base.RealPayloadCache(max_items=1)
    history = []
    active_real = list(real_groups)
    active_synth = list(synth_groups)
    model.train()

    for local_step in range(1, int(steps) + 1):
        global_step = int(start_step) + local_step
        use_synth = bool(active_synth) and args.synth_interval > 0 and global_step % args.synth_interval == 0
        phase = f"{phase_prefix}_synth" if use_synth else f"{phase_prefix}_real"

        if use_synth:
            group = active_synth[((global_step // args.synth_interval) - 1) % len(active_synth)]
            try:
                payload = base.FourierSynthPayloadCache.load(group)
            except ValueError as exc:
                print(f"[mixed-synth] skip h{horizon} {group['ds_dir']}: {exc}", flush=True)
                active_synth = [g for g in active_synth if g["ds_dir"] != group["ds_dir"]]
                base.FourierSynthPayloadCache.clear()
                continue
            emb, future_n, trend_n, seasonal_n, daily, weekly, monthly, yearly = base.sample_fourier_batch(
                payload, args.batch_size, args.seed + horizon * 19, global_step, device
            )
            pred, decomp = model(emb, daily, weekly, monthly, yearly)
            pred_loss = F.l1_loss(pred, future_n)
            loss_terms = []
            if trend_n is not None:
                loss_terms.append(F.l1_loss(decomp["trend"], trend_n))
            if seasonal_n is not None:
                loss_terms.append(F.l1_loss(decomp["seasonal"], seasonal_n))
            loss = sum(loss_terms) if loss_terms else pred_loss
            loss = loss + float(args.gate_l1_weight) * decomp["gates"].mean()
            ts_corr = component_corr(decomp["trend"], decomp["seasonal"]).detach()
            subset = Path(group["ds_dir"]).name
        else:
            group = active_real[((global_step - 1) // max(1, args.real_group_chunk_steps)) % len(active_real)]
            try:
                payload = real_cache.load(group)
            except ValueError as exc:
                print(f"[mixed-real] skip h{horizon} subset={group['subset']}: {exc}", flush=True)
                active_real = [g for g in active_real if g["subset"] != group["subset"]]
                if not active_real:
                    raise
                continue
            emb, future_n, daily, weekly, monthly, yearly = base.sample_real_batch(
                payload, args.batch_size, args.seed + horizon * 23, global_step, device
            )
            pred, decomp = model(emb, daily, weekly, monthly, yearly)
            pred_loss = F.l1_loss(pred, future_n)
            ts_corr = component_corr(decomp["trend"], decomp["seasonal"])
            loss = (
                pred_loss
                + float(args.ts_corr_weight) * ts_corr
                + float(args.gate_l1_weight) * decomp["gates"].mean()
            )
            subset = group["subset"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        should_log = (
            local_step == 1
            or local_step == int(steps)
            or global_step % args.log_every == 0
            or (not use_synth and (global_step - 1) % args.log_every == 0)
        )
        if should_log:
            with torch.no_grad():
                no_res = F.l1_loss(decomp["trend"] + decomp["seasonal"], future_n)
                residual_std = decomp["residual"].std()
                trend_std = decomp["trend"].std()
            row = {
                "phase": phase,
                "step": global_step,
                "local_step": local_step,
                "subset": subset,
                "loss": float(loss.item()),
                "pred_loss": float(pred_loss.item()),
                "no_residual_mae": float(no_res.item()),
                "gain": float(no_res.item() - pred_loss.item()),
                "trend_std": float(trend_std.item()),
                "residual_std": float(residual_std.item()),
                "trend_seasonal_corr": float(ts_corr.item()),
            }
            history.append(row)
            print(
                f"[{phase}] h{horizon} step {global_step} local={local_step}/{steps} subset={subset} "
                f"loss={row['loss']:.6g} pred={row['pred_loss']:.6g} "
                f"no_res={row['no_residual_mae']:.6g} gain={row['gain']:.6g} "
                f"res_std={row['residual_std']:.6g} "
                f"ts_corr={row['trend_seasonal_corr']:.6g}",
                flush=True,
            )

        if use_synth and len(base.FourierSynthPayloadCache.cache) > 2:
            base.FourierSynthPayloadCache.clear()

    real_cache.clear()
    base.FourierSynthPayloadCache.clear()
    return history


def train_residual_only(model: FuncDecModel, real_groups: list[dict], steps: int,
                        args: argparse.Namespace, horizon: int,
                        device: torch.device, start_step: int = 0,
                        phase: str = "residual_only") -> list[dict]:
    set_residual_only_trainable(model)
    params = trainable_params(model)
    optimizer = torch.optim.AdamW(params, lr=args.residual_learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(steps)))
    real_cache = base.RealPayloadCache(max_items=1)
    history = []
    active_real = list(real_groups)
    model.train()

    for local_step in range(1, int(steps) + 1):
        global_step = int(start_step) + local_step
        group = active_real[((global_step - 1) // max(1, args.real_group_chunk_steps)) % len(active_real)]
        try:
            payload = real_cache.load(group)
        except ValueError as exc:
            print(f"[residual-only] skip h{horizon} subset={group['subset']}: {exc}", flush=True)
            active_real = [g for g in active_real if g["subset"] != group["subset"]]
            if not active_real:
                raise
            continue

        emb, future_n, daily, weekly, monthly, yearly = base.sample_real_batch(
            payload, args.batch_size, args.seed + horizon * 29, global_step, device
        )
        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        pred_loss = F.l1_loss(pred, future_n)

        optimizer.zero_grad(set_to_none=True)
        pred_loss.backward()
        clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if local_step == 1 or global_step % args.log_every == 0 or local_step == int(steps):
            with torch.no_grad():
                no_res = F.l1_loss(decomp["trend"] + decomp["seasonal"], future_n)
                residual_std = decomp["residual"].std()
                trend_std = decomp["trend"].std()
                ts_corr = component_corr(decomp["trend"], decomp["seasonal"])
            row = {
                "phase": phase,
                "step": global_step,
                "local_step": local_step,
                "subset": group["subset"],
                "loss": float(pred_loss.item()),
                "no_residual_mae": float(no_res.item()),
                "gain": float(no_res.item() - pred_loss.item()),
                "trend_std": float(trend_std.item()),
                "residual_std": float(residual_std.item()),
                "trend_seasonal_corr": float(ts_corr.item()),
            }
            history.append(row)
            print(
                f"[{phase}] h{horizon} step {global_step} local={local_step}/{steps} subset={group['subset']} "
                f"loss={row['loss']:.6g} no_res={row['no_residual_mae']:.6g} "
                f"gain={row['gain']:.6g} res_std={row['residual_std']:.6g} "
                f"ts_corr={row['trend_seasonal_corr']:.6g}",
                flush=True,
            )

    real_cache.clear()
    return history


def train_interleaved_mixed_residual(
    model: FuncDecModel,
    real_groups: list[dict],
    synth_groups: list[dict],
    args: argparse.Namespace,
    horizon: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    """Run full mixed training with residual-only blocks inserted midstream.

    The full-decoder mixed optimizer is re-created per full block. This is
    deliberate: after each residual-only correction, the next full block starts
    with a fresh schedule instead of inheriting a partially consumed LR cycle.
    The final block always remains full-decoder mixed training; the last
    residual-only block is skipped so the final checkpoint is not residual-only
    biased.
    """
    history: dict[str, list[dict]] = {}
    full_done = 0
    residual_done = 0
    remaining_full = int(args.mixed_steps)

    burnin_steps = min(max(0, int(args.full_burnin_steps)), remaining_full)
    if burnin_steps > 0:
        print(f"[interleaved] h{horizon}: full burn-in {burnin_steps} steps", flush=True)
        history["full_burnin"] = train_mixed_full(
            model, real_groups, synth_groups, burnin_steps, args, horizon, device,
            start_step=full_done, phase_prefix="burnin_full",
        )
        full_done += burnin_steps
        remaining_full -= burnin_steps

    cycle_idx = 0
    while remaining_full > 0:
        cycle_idx += 1
        full_steps = min(max(1, int(args.cycle_full_steps)), remaining_full)
        print(
            f"[interleaved] h{horizon}: cycle {cycle_idx} full={full_steps} "
            f"residual={int(args.cycle_residual_steps)}",
            flush=True,
        )
        history[f"cycle_{cycle_idx:02d}_full"] = train_mixed_full(
            model, real_groups, synth_groups, full_steps, args, horizon, device,
            start_step=full_done, phase_prefix=f"cycle{cycle_idx:02d}_full",
        )
        full_done += full_steps
        remaining_full -= full_steps

        is_final_full_block = remaining_full <= 0
        if int(args.cycle_residual_steps) > 0 and not is_final_full_block:
            history[f"cycle_{cycle_idx:02d}_residual"] = train_residual_only(
                model, real_groups, int(args.cycle_residual_steps), args, horizon, device,
                start_step=residual_done, phase=f"cycle{cycle_idx:02d}_residual",
            )
            residual_done += int(args.cycle_residual_steps)
        elif int(args.cycle_residual_steps) > 0:
            print(
                f"[interleaved] h{horizon}: cycle {cycle_idx} final residual-only skipped; "
                "ending after full-decoder mixed training",
                flush=True,
            )

    history["step_counts"] = [{
        "phase": "step_counts",
        "full_steps": full_done,
        "residual_steps": residual_done,
        "full_burnin_steps": burnin_steps,
        "cycle_full_steps": int(args.cycle_full_steps),
        "cycle_residual_steps": int(args.cycle_residual_steps),
        "final_residual_only_skipped": bool(int(args.cycle_residual_steps) > 0),
    }]
    return history


def train_horizon(horizon: int, args: argparse.Namespace, device: torch.device) -> dict:
    ckpt_dir = args.results_root / "checkpoints"
    final_path = ckpt_dir / f"funcdec_h{horizon}.pt"
    if args.skip_existing and final_path.exists():
        return {"horizon": horizon, "skipped": True, "checkpoint": str(final_path)}

    print(f"\n=== warm-real-mix h{horizon} loading initial model ===", flush=True)
    model, cfg, init_source = base.load_initial_model(horizon, args, device)
    if init_source != "scratch" and not args.allow_checkpoint_init:
        raise RuntimeError(
            f"h{horizon} loaded checkpoint init ({init_source}). "
            "Use --init_checkpoint_dir none for scratch training, or pass "
            "--allow_checkpoint_init only for intentional warm-start experiments."
        )
    set_all_decoders_trainable(model)
    if model.backbone is not None:
        for param in model.backbone.parameters():
            param.requires_grad = False

    started = time.perf_counter()
    history: dict[str, list[dict]] = {}

    synth_groups = base.discover_fourier_groups(horizon)
    real_groups = discover_real_groups(args, horizon)
    print(
        f"[warm-real-mix] h{horizon}: synth_groups={len(synth_groups)} real_groups={len(real_groups)}",
        flush=True,
    )

    if synth_groups and args.fourier_warmup_steps > 0:
        print(f"[warmup-fourier] h{horizon}: {args.fourier_warmup_steps} steps", flush=True)
        history["fourier_warmup"] = base.train_fourier_synth(
            model, synth_groups, args.fourier_warmup_steps, args, horizon, device
        )
        save_checkpoint(model, cfg, ckpt_dir / f"fourier_warm_h{horizon}.pt", {
            "phase": "fourier_warmup",
            "args": to_jsonable(vars(args)),
            "initial_checkpoint": init_source,
        })

    if not real_groups:
        raise RuntimeError(f"No real training groups found for h{horizon}")

    if args.mixed_steps > 0:
        print(
            f"[interleaved] h{horizon}: mixed_steps={args.mixed_steps} "
            f"burnin={args.full_burnin_steps} cycle_full={args.cycle_full_steps} "
            f"cycle_residual={args.cycle_residual_steps} synth_interval={args.synth_interval}",
            flush=True,
        )
        history["interleaved_mixed_residual"] = train_interleaved_mixed_residual(
            model, real_groups, synth_groups, args, horizon, device
        )
        save_checkpoint(model, cfg, ckpt_dir / f"mixed_full_h{horizon}.pt", {
            "phase": "interleaved_mixed_residual",
            "args": to_jsonable(vars(args)),
            "initial_checkpoint": init_source,
        })

    save_checkpoint(model, cfg, final_path, {
        "phase": "complete",
        "args": to_jsonable(vars(args)),
        "initial_checkpoint": init_source,
        "history": to_jsonable(history),
        "elapsed_sec": time.perf_counter() - started,
    })

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "horizon": int(horizon),
        "initial_checkpoint": init_source,
        "checkpoint": str(final_path),
        "elapsed_sec": time.perf_counter() - started,
        "history": to_jsonable(history),
    }


def main() -> None:
    args = parse_args()
    if args.fourier_batch_size is None:
        args.fourier_batch_size = args.batch_size
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.results_root.mkdir(parents=True, exist_ok=True)
    print(f"[warm-real-mix] device={device} results={args.results_root}", flush=True)
    print(f"[warm-real-mix] init_checkpoint_dir={args.init_checkpoint_dir}", flush=True)
    print(
        f"[warm-real-mix] fourier_warmup={args.fourier_warmup_steps} "
        f"mixed={args.mixed_steps} full_burnin={args.full_burnin_steps} "
        f"cycle_full={args.cycle_full_steps} cycle_residual={args.cycle_residual_steps} "
        f"synth_interval={args.synth_interval} ts_corr_weight={args.ts_corr_weight}",
        flush=True,
    )

    result = {"args": to_jsonable(vars(args)), "per_horizon": {}}
    for horizon in args.horizons:
        try:
            result["per_horizon"][str(horizon)] = train_horizon(int(horizon), args, device)
        except Exception as exc:
            print(f"[warm-real-mix] ERROR h{horizon}: {exc}", flush=True)
            result["per_horizon"][str(horizon)] = {"error": str(exc)}
        with (args.results_root / "train_result_partial.json").open("w") as f:
            json.dump(to_jsonable(result), f, indent=2)

    with (args.results_root / "train_result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    print(f"[warm-real-mix] complete. Saved {args.results_root / 'train_result.json'}", flush=True)


if __name__ == "__main__":
    main()
