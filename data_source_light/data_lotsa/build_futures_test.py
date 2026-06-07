from __future__ import annotations

import argparse
import os
import gc
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from build_futures import build_group_futures
from shared_utils import format_bytes, group_sorted_rows, load_sorted_index_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="lotsa_index_test.parquet")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="lotsa_cache_test/")
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
    return output_dir / subset_name / f"backbone_emb_c{context_len}_{freq}_lotsa_test.pt"


def futures_path(output_dir: Path, subset_name: str, freq: str, context_len: int, horizon: int) -> Path:
    return output_dir / subset_name / f"futures_c{context_len}_{freq}_h{horizon}_lotsa_test.pt"


def total_future_bytes(output_dir: Path, horizon: int) -> int:
    return sum(path.stat().st_size for path in output_dir.rglob(f"futures_c*_*_h{horizon}_lotsa_test.pt"))


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
    print("\nTest future build summary:")
    print(f"  index_windows: {len(rows)}")
    print(f"  windows_written: {windows_written}")
    print(f"  future_size_h{args.horizon}: {format_bytes(total_size)}")
    print("  subset_index_windows:")
    for subset_name, count in sorted(indexed_by_subset.items()):
        written = written_by_subset[subset_name]
        print(f"    {subset_name}: indexed={count} written={written}")


if __name__ == "__main__":
    main()
