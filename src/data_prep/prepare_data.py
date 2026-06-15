#!/usr/bin/env python3
"""Build compact training caches for nogate_softmask warm real-mix training.

The script simulates the deterministic group/row sampling schedule from
``src/experiments/nogate_softmask/train_warm_real_mix.py`` and writes a smaller
DATA_ROOT tree containing only rows that can be sampled by that schedule.
Existing experiment training/evaluation files are intentionally not modified.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
NOGATE_DIR = REPO_ROOT / "src" / "experiments" / "nogate_softmask"
DEFAULT_SRC_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data"))
DEFAULT_DST_DATA_ROOT = Path("/tmp/data")
DEFAULT_DOMAIN_CONFIG = (
    REPO_ROOT
    / "src"
    / "experiments"
    / "synthetic_center"
    / "train_syn_real_raw"
    / "domain_config.json"
)
DEFAULT_HORIZONS = [96, 192, 336, 720]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_data_root", type=Path, default=DEFAULT_SRC_DATA_ROOT)
    parser.add_argument("--dst_data_root", type=Path, default=DEFAULT_DST_DATA_ROOT)
    parser.add_argument("--lotsa_cache_root", type=Path, default=None)
    parser.add_argument("--domain_config", type=Path, default=DEFAULT_DOMAIN_CONFIG)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fourier_batch_size", type=int, default=None)
    parser.add_argument("--fourier_warmup_steps", type=int, default=125)
    parser.add_argument("--mixed_steps", type=int, default=2500)
    parser.add_argument("--synth_interval", type=int, default=13)
    parser.add_argument("--residual_steps", type=int, default=500)
    parser.add_argument("--real_group_chunk_steps", type=int, default=63)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def import_nogate_train(src_data_root: Path):
    """Import nogate_softmask train.py after binding DATA_ROOT to src_data_root."""
    os.environ["DATA_ROOT"] = str(src_data_root)
    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))

    nogate_s = str(NOGATE_DIR)
    if nogate_s not in sys.path:
        sys.path.insert(0, nogate_s)
    repo_s = str(REPO_ROOT)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)

    try:
        import train as base  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Failed to import src/experiments/nogate_softmask/train.py. "
            "Run from the repository environment with the project dependencies installed."
        ) from exc
    return base


def path_under(path: str | Path, root: Path) -> Path:
    p = Path(path).resolve()
    r = root.resolve()
    try:
        return p.relative_to(r)
    except ValueError as exc:
        raise ValueError(f"Path is not under src_data_root: path={p} root={r}") from exc


def dst_for(src_path: str | Path, src_root: Path, dst_root: Path) -> Path:
    return dst_root / path_under(src_path, src_root)


def tensor_index_payload(payload: dict[str, Any], indices: torch.Tensor, keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in payload:
            value = payload[key]
            if torch.is_tensor(value):
                out[key] = value.index_select(0, indices).contiguous()
            else:
                out[key] = value
    return out


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def copy_file(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return file_size(src), file_size(dst)


def save_torch(payload: dict[str, Any], dst: Path, src_size: int) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
    return src_size, file_size(dst)


@dataclass
class Plan:
    real_needed: dict[tuple[str, int], set[int]]
    synth_needed: dict[str, set[int]]
    real_groups_seen: set[tuple[str, int]]
    synth_groups_seen: set[tuple[str, int]]
    real_groups_total: set[tuple[str, int]]
    synth_groups_total: set[tuple[str, int]]
    real_group_by_key: dict[tuple[str, int], dict]
    synth_group_by_key: dict[tuple[str, int], dict]
    real_before_rows: int
    real_after_rows: int
    synth_before_rows: int
    synth_after_rows: int


def discover_real_groups(base, args: argparse.Namespace, horizon: int) -> list[dict]:
    domain_cfg = base.load_domain_config(args.domain_config)["targets"]
    all_subsets = sorted(
        {
            subset
            for target_cfg in domain_cfg.values()
            for subset in target_cfg.get("train_subsets", [])
        }
        - base.EVAL_TARGETS
    )
    skip = set(getattr(args, "skip_lotsa_subsets", None) or [])
    return [
        group
        for subset in all_subsets
        if subset not in skip
        and (group := base.discover_real_group(args.lotsa_cache_root, subset, horizon)) is not None
    ]


def update_choice(target, key, indices: np.ndarray) -> None:
    target[key].update(int(x) for x in np.asarray(indices).tolist())


def simulate_schedule(base, args: argparse.Namespace) -> Plan:
    real_needed: dict[tuple[str, int], set[int]] = defaultdict(set)
    synth_needed: dict[str, set[int]] = defaultdict(set)
    real_seen: set[tuple[str, int]] = set()
    synth_seen: set[tuple[str, int]] = set()
    real_total: set[tuple[str, int]] = set()
    synth_total: set[tuple[str, int]] = set()
    real_group_by_key: dict[tuple[str, int], dict] = {}
    synth_group_by_key: dict[tuple[str, int], dict] = {}

    all_real_groups_by_horizon: dict[int, list[dict]] = {}
    all_synth_groups_by_horizon: dict[int, list[dict]] = {}
    for horizon in args.horizons:
        real_groups = discover_real_groups(base, args, int(horizon))
        synth_groups = base.discover_fourier_groups(int(horizon))
        all_real_groups_by_horizon[int(horizon)] = real_groups
        all_synth_groups_by_horizon[int(horizon)] = synth_groups
        for group in real_groups:
            key = (group["cache_dir"], int(horizon))
            real_total.add(key)
            real_group_by_key[key] = group
        for group in synth_groups:
            key = (group["ds_dir"], int(horizon))
            synth_total.add(key)
            synth_group_by_key[key] = group
        print(
            f"[plan] h{horizon}: real_groups={len(real_groups)} synth_groups={len(synth_groups)}",
            flush=True,
        )

    real_cache = base.RealPayloadCache(max_items=max(1, len(real_total)))

    for horizon in args.horizons:
        horizon = int(horizon)
        real_groups = all_real_groups_by_horizon[horizon]
        synth_groups = all_synth_groups_by_horizon[horizon]

        if synth_groups and args.fourier_warmup_steps > 0:
            for step in range(1, int(args.fourier_warmup_steps) + 1):
                group = synth_groups[(step - 1) % len(synth_groups)]
                payload = base.FourierSynthPayloadCache.load(group)
                seed = args.seed + horizon * 7
                rng = np.random.default_rng(seed + step * 1009)
                chosen = rng.choice(
                    payload["indices"],
                    size=int(args.fourier_batch_size),
                    replace=len(payload["indices"]) < int(args.fourier_batch_size),
                )
                update_choice(synth_needed, group["ds_dir"], chosen)
                synth_seen.add((group["ds_dir"], horizon))

        if args.mixed_steps > 0:
            active_real = list(real_groups)
            active_synth = list(synth_groups)
            plan_loaded: set[str] = set()
            print(f"[plan] h{horizon} mixed: {int(args.mixed_steps)} steps, {len(active_real)} real groups", flush=True)
            for step in range(1, int(args.mixed_steps) + 1):
                use_synth = bool(active_synth) and args.synth_interval > 0 and step % args.synth_interval == 0
                if use_synth:
                    group = active_synth[((step // args.synth_interval) - 1) % len(active_synth)]
                    payload = base.FourierSynthPayloadCache.load(group)
                    seed = args.seed + horizon * 19
                    rng = np.random.default_rng(seed + step * 1009)
                    chosen = rng.choice(
                        payload["indices"],
                        size=int(args.batch_size),
                        replace=len(payload["indices"]) < int(args.batch_size),
                    )
                    update_choice(synth_needed, group["ds_dir"], chosen)
                    synth_seen.add((group["ds_dir"], horizon))
                else:
                    if not active_real:
                        continue
                    group = active_real[
                        ((step - 1) // max(1, int(args.real_group_chunk_steps))) % len(active_real)
                    ]
                    cache_key = f"{group['subset']}::h{horizon}"
                    if cache_key not in plan_loaded:
                        print(f"[plan] loading mixed real h{horizon} subset={group['subset']} (step {step})", flush=True)
                    try:
                        payload = real_cache.load(group)
                    except Exception as exc:
                        print(f"[plan] skip mixed real h{horizon} subset={group['subset']}: {type(exc).__name__}: {exc}", flush=True)
                        active_real = [g for g in active_real if g["subset"] != group["subset"]]
                        plan_loaded.discard(cache_key)
                        continue
                    plan_loaded.add(cache_key)
                    seed = args.seed + horizon * 23
                    rng = np.random.default_rng(seed + step * 1009)
                    chosen = rng.choice(
                        payload["indices"],
                        size=int(args.batch_size),
                        replace=len(payload["indices"]) < int(args.batch_size),
                    )
                    update_choice(real_needed, (group["cache_dir"], horizon), chosen)
                    real_seen.add((group["cache_dir"], horizon))
            print(f"[plan] h{horizon} mixed done: {len(plan_loaded)} subsets loaded", flush=True)

        if args.residual_steps > 0:
            active_real = list(real_groups)
            plan_loaded_res: set[str] = set()
            print(f"[plan] h{horizon} residual: {int(args.residual_steps)} steps, {len(active_real)} real groups", flush=True)
            for step in range(1, int(args.residual_steps) + 1):
                if not active_real:
                    continue
                group = active_real[
                    ((step - 1) // max(1, int(args.real_group_chunk_steps))) % len(active_real)
                ]
                cache_key = f"{group['subset']}::h{horizon}"
                if cache_key not in plan_loaded_res:
                    print(f"[plan] loading residual real h{horizon} subset={group['subset']} (step {step})", flush=True)
                try:
                    payload = real_cache.load(group)
                except Exception as exc:
                    print(f"[plan] skip residual real h{horizon} subset={group['subset']}: {type(exc).__name__}: {exc}", flush=True)
                    active_real = [g for g in active_real if g["subset"] != group["subset"]]
                    plan_loaded_res.discard(cache_key)
                    continue
                plan_loaded_res.add(cache_key)
                seed = args.seed + horizon * 29
                rng = np.random.default_rng(seed + step * 1009)
                chosen = rng.choice(
                    payload["indices"],
                    size=int(args.batch_size),
                    replace=len(payload["indices"]) < int(args.batch_size),
                )
                update_choice(real_needed, (group["cache_dir"], horizon), chosen)
                real_seen.add((group["cache_dir"], horizon))
            print(f"[plan] h{horizon} residual done: {len(plan_loaded_res)} subsets loaded", flush=True)

    real_before_rows = 0
    representative_real: dict[str, tuple[str, int]] = {}
    for key in real_seen:
        representative_real.setdefault(key[0], key)
    for key in representative_real.values():
        payload = real_cache.load(real_group_by_key[key])
        real_before_rows += int(len(payload["indices"]))
    synth_before_rows = 0
    for key in synth_seen:
        payload = base.FourierSynthPayloadCache.load(synth_group_by_key[key])
        synth_before_rows += int(len(payload["indices"]))

    real_after_rows = 0
    real_needed_by_cache: dict[str, set[int]] = defaultdict(set)
    for (cache_dir, _horizon), indices in real_needed.items():
        real_needed_by_cache[cache_dir].update(indices)
    real_after_rows = sum(len(v) for v in real_needed_by_cache.values())
    synth_after_rows = sum(len(v) for v in synth_needed.values())

    real_cache.clear()
    base.FourierSynthPayloadCache.clear()

    return Plan(
        real_needed=dict(real_needed),
        synth_needed=dict(synth_needed),
        real_groups_seen=real_seen,
        synth_groups_seen=synth_seen,
        real_groups_total=real_total,
        synth_groups_total=synth_total,
        real_group_by_key=real_group_by_key,
        synth_group_by_key=synth_group_by_key,
        real_before_rows=real_before_rows,
        real_after_rows=real_after_rows,
        synth_before_rows=synth_before_rows,
        synth_after_rows=synth_after_rows,
    )


FUTURES_HORIZON_RE = re.compile(r"^futures_c\d+_.+_h(?P<horizon>\d+)_.+\.pt$")


def horizon_from_future_path(path: Path) -> int | None:
    match = FUTURES_HORIZON_RE.match(path.name)
    return int(match.group("horizon")) if match else None


def grouped_real_needed(real_needed: dict[tuple[str, int], set[int]]) -> dict[str, dict[int, set[int]]]:
    grouped: dict[str, dict[int, set[int]]] = defaultdict(dict)
    for (cache_dir, horizon), indices in real_needed.items():
        grouped[cache_dir][int(horizon)] = set(indices)
    return dict(grouped)


def copy_real_group(
    cache_dir_s: str,
    indices_by_horizon: dict[int, set[int]],
    src_root: Path,
    dst_root: Path,
) -> tuple[int, int]:
    cache_dir = Path(cache_dir_s)
    union_indices = sorted({idx for indices in indices_by_horizon.values() for idx in indices})
    if not union_indices:
        return 0, 0
    union_idx = torch.as_tensor(union_indices, dtype=torch.long)
    compact_pos = {original_idx: pos for pos, original_idx in enumerate(union_indices)}
    src_bytes = 0
    dst_bytes = 0

    backbone_files = sorted(cache_dir.glob("backbone_emb_c*_*.pt"))
    future_files = sorted(cache_dir.glob("futures_c*_h*_*.pt"))
    slice_files = set(backbone_files + future_files)

    for path in backbone_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out = tensor_index_payload(payload, union_idx, ["embeddings"])
        s, d = save_torch(out, dst_for(path, src_root, dst_root), file_size(path))
        src_bytes += s
        dst_bytes += d

    for path in future_files:
        horizon = horizon_from_future_path(path)
        if horizon is None or horizon not in indices_by_horizon:
            s, d = copy_file(path, dst_for(path, src_root, dst_root))
            src_bytes += s
            dst_bytes += d
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        futures = payload["futures_n"]
        selected = sorted(indices_by_horizon[horizon])
        selected_idx = torch.as_tensor(selected, dtype=torch.long)
        selected_futures = futures.index_select(0, selected_idx).contiguous()
        compact_futures = torch.zeros(
            (len(union_indices), *selected_futures.shape[1:]),
            dtype=selected_futures.dtype,
        )
        compact_valid = torch.zeros(len(union_indices), dtype=torch.bool)
        compact_rows = torch.as_tensor([compact_pos[i] for i in selected], dtype=torch.long)
        compact_futures.index_copy_(0, compact_rows, selected_futures)
        compact_valid.index_fill_(0, compact_rows, True)
        out = {"futures_n": compact_futures, "valid_mask": compact_valid}
        s, d = save_torch(out, dst_for(path, src_root, dst_root), file_size(path))
        src_bytes += s
        dst_bytes += d

    for path in sorted(p for p in cache_dir.iterdir() if p.is_file() and p not in slice_files):
        if path.name == "valid_mask.pt":
            continue
        s, d = copy_file(path, dst_for(path, src_root, dst_root))
        src_bytes += s
        dst_bytes += d

    return src_bytes, dst_bytes


def copy_synth_group(ds_dir_s: str, indices_s: set[int], src_root: Path, dst_root: Path) -> tuple[int, int]:
    ds_dir = Path(ds_dir_s)
    idx = torch.as_tensor(sorted(indices_s), dtype=torch.long)
    src_bytes = 0
    dst_bytes = 0

    backbone_files = sorted(ds_dir.glob("backbone_emb_c*_h*_stride1.pt"))
    raw_files = sorted(ds_dir.glob("raw_futures_h*.pt"))
    component_files = sorted(ds_dir.glob("component_targets_h*.pt"))
    slice_files = set(backbone_files + raw_files + component_files)

    for path in backbone_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out = tensor_index_payload(payload, idx, ["embeddings"])
        s, d = save_torch(out, dst_for(path, src_root, dst_root), file_size(path))
        src_bytes += s
        dst_bytes += d

    for path in raw_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out = tensor_index_payload(payload, idx, ["futures_n"])
        s, d = save_torch(out, dst_for(path, src_root, dst_root), file_size(path))
        src_bytes += s
        dst_bytes += d

    for path in component_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out = tensor_index_payload(payload, idx, ["trend_n", "seasonal_n"])
        s, d = save_torch(out, dst_for(path, src_root, dst_root), file_size(path))
        src_bytes += s
        dst_bytes += d

    for path in sorted(p for p in ds_dir.iterdir() if p.is_file() and p not in slice_files):
        s, d = copy_file(path, dst_for(path, src_root, dst_root))
        src_bytes += s
        dst_bytes += d

    return src_bytes, dst_bytes


def run_copy_jobs(
    label: str,
    items: dict[str, Any],
    worker_fn,
    src_root: Path,
    dst_root: Path,
    num_workers: int,
) -> tuple[int, int]:
    if not items:
        return 0, 0
    src_total = 0
    dst_total = 0
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as pool:
        futures = {
            pool.submit(worker_fn, path, rows, src_root, dst_root): path
            for path, rows in sorted(items.items())
            if rows
        }
        done = 0
        for future in as_completed(futures):
            path = futures[future]
            try:
                src_bytes, dst_bytes = future.result()
            except Exception as exc:
                raise RuntimeError(f"{label} copy failed for {path}: {exc}") from exc
            src_total += src_bytes
            dst_total += dst_bytes
            done += 1
            print(
                f"[copy:{label}] {done}/{len(futures)} {Path(path).name} "
                f"{gb(src_bytes):.3f}GB -> {gb(dst_bytes):.3f}GB",
                flush=True,
            )
    return src_total, dst_total


def print_row_summary(plan: Plan) -> None:
    def reduction(before: int, after: int) -> float:
        if before <= 0:
            return 0.0
        return 100.0 * (1.0 - after / before)

    real_seen_dirs = {cache_dir for cache_dir, _horizon in plan.real_groups_seen}
    real_total_dirs = {cache_dir for cache_dir, _horizon in plan.real_groups_total}
    print(f"Real groups: {len(real_seen_dirs)} / {len(real_total_dirs)} accessed", flush=True)
    print(f"Synth groups: {len(plan.synth_groups_seen)} / {len(plan.synth_groups_total)} accessed", flush=True)
    print(
        f"Real total rows: before={plan.real_before_rows}, after={plan.real_after_rows} "
        f"({reduction(plan.real_before_rows, plan.real_after_rows):.2f}% reduced)",
        flush=True,
    )
    print(
        f"Synth total rows: before={plan.synth_before_rows}, after={plan.synth_after_rows} "
        f"({reduction(plan.synth_before_rows, plan.synth_after_rows):.2f}% reduced)",
        flush=True,
    )


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    args.src_data_root = args.src_data_root.resolve()
    args.dst_data_root = args.dst_data_root.resolve()
    if args.fourier_batch_size is None:
        args.fourier_batch_size = args.batch_size
    if args.lotsa_cache_root is None:
        args.lotsa_cache_root = args.src_data_root / "data_lotsa" / "lotsa_cache"
    else:
        args.lotsa_cache_root = args.lotsa_cache_root.resolve()
    args.domain_config = args.domain_config.resolve()

    if args.src_data_root == args.dst_data_root:
        raise SystemExit("--src_data_root and --dst_data_root must be different")
    if not args.src_data_root.exists():
        raise SystemExit(f"src_data_root not found: {args.src_data_root}")
    if not args.lotsa_cache_root.exists():
        raise SystemExit(f"lotsa_cache_root not found: {args.lotsa_cache_root}")
    if not args.domain_config.exists():
        raise SystemExit(f"domain_config not found: {args.domain_config}")

    print(f"[prepare] src={args.src_data_root}", flush=True)
    print(f"[prepare] dst={args.dst_data_root}", flush=True)
    print(f"[prepare] horizons={args.horizons}", flush=True)
    print(
        f"[prepare] batch={args.batch_size} fourier_batch={args.fourier_batch_size} "
        f"warmup={args.fourier_warmup_steps} mixed={args.mixed_steps} "
        f"residual={args.residual_steps} synth_interval={args.synth_interval} "
        f"real_chunk={args.real_group_chunk_steps}",
        flush=True,
    )

    base = import_nogate_train(args.src_data_root)
    plan = simulate_schedule(base, args)
    print_row_summary(plan)

    args.dst_data_root.mkdir(parents=True, exist_ok=True)
    real_src_bytes, real_dst_bytes = run_copy_jobs(
        "real",
        grouped_real_needed(plan.real_needed),
        copy_real_group,
        args.src_data_root,
        args.dst_data_root,
        args.num_workers,
    )
    gc.collect()
    synth_src_bytes, synth_dst_bytes = run_copy_jobs(
        "synth",
        plan.synth_needed,
        copy_synth_group,
        args.src_data_root,
        args.dst_data_root,
        args.num_workers,
    )

    elapsed = time.perf_counter() - started
    print(f"준비 완료. dst: {args.dst_data_root}", flush=True)
    print(f"총 소요 시간: {elapsed:.1f}초", flush=True)
    print(f"Real: {gb(real_src_bytes):.3f} GB -> {gb(real_dst_bytes):.3f} GB", flush=True)
    print(f"Synth: {gb(synth_src_bytes):.3f} GB -> {gb(synth_dst_bytes):.3f} GB", flush=True)


if __name__ == "__main__":
    main()
