#!/usr/bin/env python3
"""Parallel wrapper for the unified Fourier synthetic generator.

The single-process generator is intentionally simple. This wrapper keeps the
same outputs and policies, but splits work by (split, granularity) so one VESSL
GPU job can overlap raw generation, compression, volume writes, and backbone
inference.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from build_fourier_synth import (
    COMPOSITIONS,
    DEFAULT_CONFIG,
    DEFAULT_EVAL_CACHE,
    DEFAULT_EVAL_RAW,
    DEFAULT_GRANULARITIES,
    DEFAULT_TRAIN_CACHE,
    DEFAULT_TRAIN_RAW,
    LEVELS,
    SPLIT_SEED_OFFSET,
    VALID_HORIZONS,
    active_harmonics,
    build_dataset,
    cache_npz,
    cache_root_for,
    enumerate_count_cases,
    load_backbone,
    load_generation_config,
    raw_root_for,
    resolve_device,
    save_npz,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train_raw_root", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument("--eval_raw_root", type=Path, default=DEFAULT_EVAL_RAW)
    parser.add_argument("--train_cache_root", type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--eval_cache_root", type=Path, default=DEFAULT_EVAL_CACHE)
    parser.add_argument("--splits", nargs="+", default=["train", "eval"], choices=["train", "eval"])
    parser.add_argument("--granularities", nargs="+", default=DEFAULT_GRANULARITIES,
                        choices=DEFAULT_GRANULARITIES)
    parser.add_argument("--trend_levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--compositions", nargs="+", default=COMPOSITIONS, choices=COMPOSITIONS)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--amplitude_low", type=float, default=0.5)
    parser.add_argument("--amplitude_high", type=float, default=1.5)
    parser.add_argument("--coefficient_low", type=float, default=0.3)
    parser.add_argument("--coefficient_high", type=float, default=1.7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--raw_only", action="store_true")
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--parallel_workers", type=int,
                        default=int(os.environ.get("FOURIER_SYNTH_WORKERS", "4")))
    return parser.parse_args()


def _namespace_from_dict(values: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**values)


def _worker(worker_idx: int, task_keys: list[tuple[str, str]], args_dict: dict[str, Any]) -> list[str]:
    args = _namespace_from_dict(args_dict)
    cfg = load_generation_config(args.config)
    global_cfg = cfg["global"]
    horizons = args.horizons or list(global_cfg["horizons"])
    context_len = args.context_len or int(global_cfg["context_len"])
    seed = args.seed if args.seed is not None else int(global_cfg["seed"])
    train_samples = args.train_samples or int(global_cfg.get("train_samples", global_cfg["n_samples"]))
    eval_samples = args.eval_samples or int(global_cfg.get("eval_samples", max(512, global_cfg["n_samples"] // 2)))

    device = None
    backbone = None
    revin_fn = None
    update_stats_fn = None
    if not args.raw_only and not args.metadata_only:
        device = resolve_device(args.device)
        backbone, revin_fn, update_stats_fn = load_backbone(device, args.hf_cache_dir)

    completed: list[str] = []
    for split, granularity in task_keys:
        n_samples = train_samples if split == "train" else eval_samples
        split_seed = seed + SPLIT_SEED_OFFSET[split]
        raw_root = raw_root_for(split, args)
        cache_root = cache_root_for(split, args)
        cache_root.mkdir(parents=True, exist_ok=True)

        if not enumerate_count_cases(active_harmonics(granularity, context_len)):
            print(f"[worker {worker_idx}] skip_no_active split={split} granularity={granularity}", flush=True)
            continue

        for composition in args.compositions:
            for trend_level in args.trend_levels:
                for horizon in horizons:
                    dataset = build_dataset(
                        split=split,
                        composition=composition,
                        trend_level=trend_level,
                        granularity=granularity,
                        horizon=int(horizon),
                        context_len=int(context_len),
                        n_samples=int(n_samples),
                        seed=int(split_seed),
                        cfg=cfg,
                        noise_scale=float(args.noise_scale),
                        amplitude_low=float(args.amplitude_low),
                        amplitude_high=float(args.amplitude_high),
                        coefficient_low=float(args.coefficient_low),
                        coefficient_high=float(args.coefficient_high),
                    )
                    npz_path = save_npz(dataset, raw_root)
                    print(
                        f"[worker {worker_idx}] saved_npz={npz_path} samples={n_samples} "
                        f"cases={dataset['meta']['n_cases']}",
                        flush=True,
                    )
                    if not args.raw_only:
                        cache_npz(
                            npz_path=npz_path,
                            output_root=cache_root,
                            batch_size=int(args.batch_size),
                            backbone=backbone,
                            revin_fn=revin_fn,
                            update_stats_fn=update_stats_fn,
                            device=device,
                            metadata_only=bool(args.metadata_only),
                            skip_existing=bool(args.skip_existing),
                        )
        completed.append(f"{split}:{granularity}")
    return completed


def main() -> None:
    args = parse_args()
    cfg = load_generation_config(args.config)
    global_cfg = cfg["global"]
    horizons = args.horizons or list(global_cfg["horizons"])
    context_len = args.context_len or int(global_cfg["context_len"])
    seed = args.seed if args.seed is not None else int(global_cfg["seed"])
    train_samples = args.train_samples or int(global_cfg.get("train_samples", global_cfg["n_samples"]))
    eval_samples = args.eval_samples or int(global_cfg.get("eval_samples", max(512, global_cfg["n_samples"] // 2)))

    for horizon in horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon {horizon}; expected one of {sorted(VALID_HORIZONS)}")

    for split in args.splits:
        n_samples = train_samples if split == "train" else eval_samples
        split_seed = seed + SPLIT_SEED_OFFSET[split]
        write_manifest(raw_root_for(split, args), split, args, cfg, context_len, n_samples, split_seed)
        cache_root_for(split, args).mkdir(parents=True, exist_ok=True)

    tasks = [(split, granularity) for split in args.splits for granularity in args.granularities]
    workers = max(1, min(int(args.parallel_workers), len(tasks)))
    shards = [[] for _ in range(workers)]
    for idx, task in enumerate(tasks):
        shards[idx % workers].append(task)

    args_dict = vars(args).copy()
    print(f"parallel_workers={workers} tasks={tasks}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, idx, shard, args_dict) for idx, shard in enumerate(shards) if shard]
        for future in as_completed(futures):
            completed = future.result()
            print(f"completed_tasks={completed}", flush=True)


if __name__ == "__main__":
    main()
