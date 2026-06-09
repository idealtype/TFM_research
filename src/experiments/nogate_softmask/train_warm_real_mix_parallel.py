#!/usr/bin/env python3
"""Parallel horizon runner for nogate_softmask warm real-mix training.

This module leaves ``train_warm_real_mix.py`` unchanged and only parallelizes
the top-level horizon loop. Each worker process calls the original
``train_horizon`` implementation and writes horizon-specific checkpoints into
the same results root.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import argparse
import sys
import time
from argparse import Namespace

import torch

from common import to_jsonable
from train_warm_real_mix import parse_args, train_horizon


def parse_parallel_processes() -> int | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parallel_processes", type=int, default=None)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known.parallel_processes


def train_horizon_wrapper(horizon: int, args: Namespace, device_str: str) -> tuple[int, dict]:
    args.device = device_str
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    try:
        result = train_horizon(int(horizon), args, device)
    except Exception as exc:
        print(f"[warm-real-mix-parallel] ERROR h{horizon}: {exc}", flush=True)
        result = {"error": str(exc)}
    return int(horizon), to_jsonable(result)


def main() -> None:
    parallel_processes = parse_parallel_processes()
    args = parse_args()
    if args.fourier_batch_size is None:
        args.fourier_batch_size = args.batch_size
    args.parallel_processes = parallel_processes or len(args.horizons)

    device_str = str(args.device)
    args.results_root.mkdir(parents=True, exist_ok=True)

    print(f"[warm-real-mix-parallel] device={device_str} results={args.results_root}", flush=True)
    print(f"[warm-real-mix-parallel] init_checkpoint_dir={args.init_checkpoint_dir}", flush=True)
    print(
        f"[warm-real-mix-parallel] horizons={args.horizons} "
        f"parallel_processes={args.parallel_processes}",
        flush=True,
    )
    print(
        f"[warm-real-mix-parallel] fourier_warmup={args.fourier_warmup_steps} "
        f"mixed={args.mixed_steps} residual={args.residual_steps} "
        f"synth_interval={args.synth_interval} ts_corr_weight={args.ts_corr_weight}",
        flush=True,
    )

    started = time.perf_counter()
    worker_args = [(int(h), args, device_str) for h in args.horizons]
    process_count = max(1, min(int(args.parallel_processes), len(worker_args)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=process_count) as pool:
        results = pool.starmap(train_horizon_wrapper, worker_args)

    result = {"args": to_jsonable(vars(args)), "per_horizon": {}}
    for horizon, horizon_result in sorted(results, key=lambda item: item[0]):
        result["per_horizon"][str(horizon)] = horizon_result

    result["elapsed_sec"] = time.perf_counter() - started
    with (args.results_root / "train_result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    print(f"[warm-real-mix-parallel] complete. Saved {args.results_root / 'train_result.json'}", flush=True)


if __name__ == "__main__":
    main()
