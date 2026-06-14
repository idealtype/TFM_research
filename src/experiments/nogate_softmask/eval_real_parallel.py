#!/usr/bin/env python3
"""Parallel horizon runner for nogate_softmask real evaluation.

Runs each horizon in a separate process sharing the same GPU (cuda:0).
Results are collected in the main process and written once after all
horizons complete.

Usage:
    python eval_real_parallel.py [eval_real args...] [--parallel_processes N]

--parallel_processes defaults to the number of horizons (len(--horizons)).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import torch

from eval_real import (
    eval_horizon,
    load_manifest,
    log_progress,
    parse_args,
    write_outputs,
)


def _parse_parallel_processes() -> int | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parallel_processes", type=int, default=None)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known.parallel_processes


def _eval_horizon_worker(
    horizon: int,
    args: argparse.Namespace,
    manifest: list[dict],
    device_str: str,
) -> tuple[int, list[dict], str]:
    device = torch.device(
        device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    try:
        rows, ckpt_path_str = eval_horizon(horizon, args, manifest, device)
    except Exception as exc:
        log_progress(f"[parallel] ERROR h{horizon}: {exc}")
        rows, ckpt_path_str = [], ""
    return horizon, rows, ckpt_path_str


def main() -> None:
    parallel_processes = _parse_parallel_processes()
    args = parse_args()
    n_parallel = parallel_processes or len(args.horizons)

    device_str = str(args.device)
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

    log_progress(
        f"start device={device_str} output={out_root} "
        f"datasets={len(manifest)} horizons={args.horizons} parallel={n_parallel}"
    )
    if args.skip_tfm:
        log_progress("TimesFM execution disabled; tfm columns will be filled from precomputed metrics CSV")

    worker_args = [(h, args, manifest, device_str) for h in args.horizons]
    with mp.Pool(processes=n_parallel) as pool:
        results = pool.starmap(_eval_horizon_worker, worker_args)

    rows = []
    ckpt_by_horizon = {}
    for horizon, horizon_rows, ckpt_path_str in sorted(results, key=lambda x: x[0]):
        rows.extend(horizon_rows)
        ckpt_by_horizon[str(horizon)] = ckpt_path_str

    write_outputs(args, rows, ckpt_by_horizon, out_root)
    log_progress(f"complete output={out_root}")


if __name__ == "__main__":
    main()
