from __future__ import annotations

import argparse
import os
import gc
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from shared_utils import IndexRow, format_bytes, group_sorted_rows, load_sorted_index_rows, target_values


REVIN_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="lotsa_index.parquet")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="lotsa_cache/")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    return parser.parse_args()


def load_subset(subset_name: str, hf_cache_dir: str | None):
    return load_dataset(
        "Salesforce/lotsa_data",
        subset_name,
        split="train",
        streaming=False,
        cache_dir=hf_cache_dir,
    )


def backbone_path(output_dir: Path, subset_name: str, freq: str, context_len: int) -> Path:
    return output_dir / subset_name / f"backbone_emb_c{context_len}_{freq}_lotsa.pt"


def futures_path(output_dir: Path, subset_name: str, freq: str, context_len: int, horizon: int) -> Path:
    return output_dir / subset_name / f"futures_c{context_len}_{freq}_h{horizon}_lotsa.pt"


def slice_future(series: np.ndarray, win_start: int, context_len: int, horizon: int) -> np.ndarray | None:
    start = win_start + context_len
    end = start + horizon
    if end > len(series):
        return None
    future = series[start:end]
    if len(future) != horizon:
        return None
    return future.astype(np.float32, copy=False)


def validate_row_order(rows: list[IndexRow], backbone_cache: dict[str, Any], cache_file: Path) -> None:
    cached_series_ids = list(backbone_cache["series_ids"])
    cached_win_starts = backbone_cache["win_starts"].cpu().tolist()
    cached_freq = str(backbone_cache["freq"])
    if len(cached_series_ids) != len(rows) or len(cached_win_starts) != len(rows):
        raise ValueError(
            f"Row count mismatch for {cache_file}: index={len(rows)} "
            f"series_ids={len(cached_series_ids)} win_starts={len(cached_win_starts)}"
        )
    for idx, row in enumerate(rows):
        if row.freq != cached_freq:
            raise ValueError(f"Freq mismatch for {cache_file}: index={row.freq} cache={cached_freq}")
        if cached_series_ids[idx] != row.series_id or cached_win_starts[idx] != row.win_start:
            raise ValueError(
                f"Row order mismatch for {cache_file} at row {idx}: "
                f"index=({row.series_id}, {row.win_start}) "
                f"cache=({cached_series_ids[idx]}, {cached_win_starts[idx]})"
            )


def build_group_futures(
    subset_name: str,
    context_len: int,
    rows: list[IndexRow],
    dataset,
    horizon: int,
    output_path: Path,
    cache_file: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"[skip] {output_path}")
        return 0
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing backbone cache: {cache_file}")

    backbone_cache = torch.load(cache_file, map_location="cpu")
    validate_row_order(rows, backbone_cache, cache_file)
    mu = backbone_cache["mu"].float()
    sigma = backbone_cache["sigma"].float()
    denom = torch.where(sigma >= REVIN_TOL, sigma, torch.ones_like(sigma))

    futures: list[np.ndarray] = []
    current_series_id: int | None = None
    current_series: np.ndarray | None = None
    skipped = 0

    for row in tqdm(rows, desc=f"{subset_name} c{context_len} h{horizon}", leave=False):
        if row.series_id != current_series_id:
            current_series_id = row.series_id
            current_series = target_values(dataset[row.series_id])
        assert current_series is not None
        future = slice_future(current_series, row.win_start, context_len, horizon)
        if future is None:
            skipped += 1
            continue
        futures.append(future)

    if skipped:
        raise ValueError(f"{subset_name} c{context_len} h{horizon}: skipped={skipped}; index is not valid for horizon")

    future_tensor = torch.from_numpy(np.stack(futures, axis=0)).float() if futures else torch.empty((0, horizon))
    futures_n = ((future_tensor - mu) / denom).float()
    torch.save(
        {
            "futures": futures_n,
            "futures_n": futures_n,
            "valid_mask": torch.ones(len(futures), dtype=torch.bool),
            "horizon": int(horizon),
            "context_len": int(context_len),
        },
        output_path,
    )
    print(f"[saved] {output_path} windows={len(futures)}")
    return len(futures)


def total_future_bytes(output_dir: Path, horizon: int) -> int:
    return sum(path.stat().st_size for path in output_dir.rglob(f"futures_c*_*_h{horizon}_lotsa.pt"))


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")

    index_path = Path(args.index)
    output_dir = Path(args.output_dir)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    rows = load_sorted_index_rows(index_path)
    if not rows:
        raise ValueError(f"No index rows found in {index_path}")
    for row in rows:
        if args.horizon > row.max_horizon:
            raise ValueError(f"horizon={args.horizon} exceeds index max_horizon={row.max_horizon}")
    grouped = group_sorted_rows(rows)

    windows_written = 0
    indexed_by_subset: Counter[str] = Counter(row.subset_name for row in rows)
    written_by_subset: Counter[str] = Counter()
    subsets = sorted({subset_name for subset_name, _freq, _context_len in grouped})

    for subset_name in tqdm(subsets, desc="subsets"):
        subset_groups = {
            (freq, context_len): group_rows
            for (group_subset, freq, context_len), group_rows in grouped.items()
            if group_subset == subset_name
        }
        pending = {
            group_key: group_rows
            for group_key, group_rows in subset_groups.items()
            if not futures_path(output_dir, subset_name, group_key[0], group_key[1], args.horizon).exists()
        }
        if not pending:
            print(f"[skip] {subset_name}: futures exist")
            continue

        dataset = load_subset(subset_name, args.hf_cache_dir)
        try:
            for (freq, context_len), group_rows in sorted(pending.items()):
                out_path = futures_path(output_dir, subset_name, freq, context_len, args.horizon)
                cache_file = backbone_path(output_dir, subset_name, freq, context_len)
                count = build_group_futures(
                    subset_name,
                    context_len,
                    group_rows,
                    dataset,
                    args.horizon,
                    out_path,
                    cache_file,
                )
                windows_written += count
                written_by_subset[subset_name] += count
        finally:
            del dataset
            gc.collect()

    total_size = total_future_bytes(output_dir, args.horizon)
    print("\nFuture build summary:")
    print(f"  index_windows: {len(rows)}")
    print(f"  windows_written: {windows_written}")
    print(f"  future_size_h{args.horizon}: {format_bytes(total_size)}")
    print("  subset_index_windows:")
    for subset_name, count in sorted(indexed_by_subset.items()):
        written = written_by_subset[subset_name]
        print(f"    {subset_name}: indexed={count} written={written}")


if __name__ == "__main__":
    main()
