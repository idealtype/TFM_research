from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

from shared_utils import CONTEXT_BY_FREQ, row_freq, rows_to_table, series_len


TEST_SUBSETS = {
    "energy": "elecdemand",
    "transport": "uber_tlc_daily",
    "climate": "oikolab_weather",
    "cloudops": "alibaba_cluster_trace_2018",
    "web": "kaggle_web_traffic_weekly",
    "sales": "restaurant",
    "econfin": "nn5_weekly",
    "nature": "saugeenday",
    "healthcare": "us_births",
}

@dataclass(frozen=True)
class SeriesInfo:
    series_id: int
    freq: str
    context_len: int
    n_windows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_horizon", type=int, required=True)
    parser.add_argument("--target_n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="lotsa_index_test.parquet")
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


def scan_domain(domain: str, subset_name: str, max_horizon: int, hf_cache_dir: str | None) -> list[SeriesInfo]:
    dataset = load_subset(subset_name, hf_cache_dir)
    usable: list[SeriesInfo] = []
    for series_id, row in enumerate(tqdm(dataset, desc=f"scan {domain}/{subset_name}", leave=False)):
        freq = row_freq(row)
        if freq not in CONTEXT_BY_FREQ:
            continue
        context_len = CONTEXT_BY_FREQ[freq]
        length = series_len(row)
        if length < context_len + max_horizon:
            continue
        n_windows = max(0, length - context_len - max_horizon + 1)
        if n_windows <= 0:
            continue
        usable.append(
            SeriesInfo(
                series_id=series_id,
                freq=freq,
                context_len=context_len,
                n_windows=n_windows,
            )
        )
    return usable


def weighted_choice(rng: random.Random, infos: list[SeriesInfo], remaining_by_series: dict[int, int]) -> SeriesInfo:
    candidates = [info for info in infos if remaining_by_series[info.series_id] > 0]
    weights = [remaining_by_series[info.series_id] for info in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def save_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    pq.write_table(rows_to_table(rows), output_path)


def print_pool_stats(pool_sizes: dict[str, int], target_n: int) -> None:
    total_pool = sum(pool_sizes.values())
    print("Available window pool:")
    for domain in TEST_SUBSETS:
        print(f"  {domain}: {pool_sizes[domain]}")
    print(f"  total: {total_pool}")
    if total_pool < target_n:
        print(f"  warning: target_n={target_n} exceeds available unique windows={total_pool}")


def print_result_stats(domain_counts: Counter[str], target_n: int) -> None:
    total = sum(domain_counts.values())
    per_domain_target = target_n / len(TEST_SUBSETS)
    print("\nTest index summary:")
    print(f"  total_collected: {total}")
    print("  domain_distribution:")
    for domain in TEST_SUBSETS:
        count = domain_counts[domain]
        ratio = count / per_domain_target if per_domain_target > 0 else 0.0
        mark = " warning" if count < per_domain_target else ""
        print(f"    {domain}: {count} ({ratio:.2%} of per-domain target){mark}")


def main() -> None:
    args = parse_args()
    if args.max_horizon <= 0:
        raise ValueError("--max_horizon must be positive.")
    if args.target_n <= 0:
        raise ValueError("--target_n must be positive.")

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)

    rng = random.Random(args.seed)
    domain_infos: dict[str, list[SeriesInfo]] = {}
    pool_sizes: dict[str, int] = {}

    for domain, subset_name in TEST_SUBSETS.items():
        infos = scan_domain(domain, subset_name, args.max_horizon, args.hf_cache_dir)
        domain_infos[domain] = infos
        pool_sizes[domain] = sum(info.n_windows for info in infos)

    print_pool_stats(pool_sizes, args.target_n)

    selected: dict[str, set[tuple[int, int]]] = defaultdict(set)
    selected_by_series: dict[str, Counter[int]] = defaultdict(Counter)
    domain_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    progress = tqdm(total=args.target_n, desc="sampling test windows")
    try:
        while len(rows) < args.target_n:
            candidate_domains = [
                domain
                for domain, infos in domain_infos.items()
                if domain_counts[domain] < pool_sizes[domain] and infos
            ]
            if not candidate_domains:
                break

            domain = rng.choice(candidate_domains)
            subset_name = TEST_SUBSETS[domain]
            remaining_by_series = {
                info.series_id: info.n_windows - selected_by_series[domain][info.series_id]
                for info in domain_infos[domain]
            }
            info = weighted_choice(rng, domain_infos[domain], remaining_by_series)
            win_start = rng.randint(0, info.n_windows - 1)
            key = (info.series_id, win_start)
            if key in selected[domain]:
                continue

            selected[domain].add(key)
            selected_by_series[domain][info.series_id] += 1
            domain_counts[domain] += 1
            rows.append(
                {
                    "subset_name": subset_name,
                    "series_id": info.series_id,
                    "win_start": win_start,
                    "freq": info.freq,
                    "context_len": info.context_len,
                    "max_horizon": args.max_horizon,
                }
            )
            progress.update(1)
    finally:
        progress.close()

    save_parquet(rows, output_path)
    print(f"\nSaved: {output_path}")
    print_result_stats(domain_counts, args.target_n)


if __name__ == "__main__":
    main()
