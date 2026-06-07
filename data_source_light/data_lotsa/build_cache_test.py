from __future__ import annotations

import argparse
import os
import gc
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from build_cache import (
    load_backbone,
    load_subset,
    process_context_group,
    resolve_device,
)
from shared_utils import format_bytes, group_sorted_rows, load_sorted_index_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="lotsa_index_test.parquet")
    parser.add_argument("--output_dir", type=str, default="lotsa_cache_test/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    return parser.parse_args()


def cache_path(output_dir: Path, subset_name: str, freq: str, context_len: int) -> Path:
    return output_dir / subset_name / f"backbone_emb_c{context_len}_{freq}_lotsa_test.pt"


def total_cache_bytes(output_dir: Path) -> int:
    return sum(path.stat().st_size for path in output_dir.rglob("backbone_emb_c*_lotsa_test.pt"))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    index_path = Path(args.index)
    output_dir = Path(args.output_dir)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    rows = load_sorted_index_rows(index_path)
    if not rows:
        raise ValueError(f"No index rows found in {index_path}")
    grouped = group_sorted_rows(rows)

    device = resolve_device(args.device)
    backbone, revin_fn, update_stats_fn = load_backbone(device, args.hf_cache_dir)

    total_index_windows = len(rows)
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
            if not cache_path(output_dir, subset_name, group_key[0], group_key[1]).exists()
        }
        if not pending:
            print(f"[skip] {subset_name}: cache exists")
            continue

        dataset = load_subset(subset_name, args.hf_cache_dir)
        try:
            for (freq, context_len), group_rows in sorted(pending.items()):
                out_path = cache_path(output_dir, subset_name, freq, context_len)
                count = process_context_group(
                    subset_name,
                    freq,
                    context_len,
                    group_rows,
                    dataset,
                    out_path,
                    args.batch_size,
                    backbone,
                    revin_fn,
                    update_stats_fn,
                    device,
                )
                windows_written += count
                written_by_subset[subset_name] += count
        finally:
            del dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    total_size = total_cache_bytes(output_dir)
    print("\nTest cache build summary:")
    print(f"  index_windows: {total_index_windows}")
    print(f"  windows_written: {windows_written}")
    print(f"  cache_size: {format_bytes(total_size)}")
    print("  subset_index_windows:")
    for subset_name, count in sorted(indexed_by_subset.items()):
        written = written_by_subset[subset_name]
        print(f"    {subset_name}: indexed={count} written={written}")


if __name__ == "__main__":
    main()
