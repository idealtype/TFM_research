from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

from shared_utils import CONTEXT_BY_FREQ, index_schema, row_freq, rows_to_table, series_len


DOMAIN_MAP = {
    "energy": [
        "bdg-2_panther",
        "bdg-2_fox",
        "bdg-2_rat",
        "bdg-2_bear",
        "lcl",
        "smart",
        "ideal",
        "sceaux",
        "borealis",
        "covid19_energy",
        "gfc12_load",
        "gfc14_load",
        "gfc17_load",
        "pdb",
        "spain",
        "hog",
        "bull",
        "cockatoo",
        "elf",
        "kdd2022",
        "residential_load_power",
        "residential_pv_power",
        "wind_power",
        "solar_power",
        "london_smart_meters_with_missing",
        "wind_farms_with_missing",
        "australian_electricity_demand",
        "elecdemand",
    ],
    "transport": [
        "PEMS03",
        "PEMS04",
        "PEMS07",
        "PEMS08",
        "PEMS_BAY",
        "LOS_LOOP",
        "LOOP_SEATTLE",
        "SZ_TAXI",
        "BEIJING_SUBWAY_30MIN",
        "SHMETRO",
        "HZMETRO",
        "M_DENSE",
        "Q-TRAFFIC",
        "taxi_30min",
        "uber_tlc_daily",
        "uber_tlc_hourly",
        "rideshare_with_missing",
        "covid_mobility",
        "traffic_hourly",
        "traffic_weekly",
    ],
    "climate": [
        "oikolab_weather",
        "subseasonal",
        "subseasonal_precip",
        "china_air_quality",
        "beijing_air_quality",
        "temperature_rain_with_missing",
        "kdd_cup_2018_with_missing",
    ],
    # cloudops is intentionally excluded from train. The only remaining small
    # subset, alibaba_cluster_trace_2018, is reserved for the test holdout.
    "web": [
        "wiki-rolling_nips",
        "kaggle_web_traffic_weekly",
        "extended_web_traffic_with_missing",
    ],
    "sales": [
        "m5",
        "favorita_sales",
        "favorita_transactions",
        "restaurant",
        "hierarchical_sales",
        "godaddy",
        "car_parts_with_missing",
    ],
    "econfin": [
        "m1_yearly",
        "m1_quarterly",
        "m1_monthly",
        "monash_m3_yearly",
        "monash_m3_quarterly",
        "monash_m3_monthly",
        "monash_m3_other",
        "m4_yearly",
        "m4_quarterly",
        "m4_monthly",
        "m4_weekly",
        "m4_daily",
        "m4_hourly",
        "nn5_daily_with_missing",
        "nn5_weekly",
        "tourism_yearly",
        "tourism_quarterly",
        "tourism_monthly",
        "cif_2016_6",
        "cif_2016_12",
        "fred_md",
        "bitcoin_with_missing",
        "pedestrian_counts",
    ],
    "nature": [
        "saugeenday",
        "sunspot_with_missing",
        "weather",
        "oikolab_weather",
        "vehicle_trips_with_missing",
    ],
    "healthcare": [
        "hospital",
        "covid_deaths",
        "cdc_fluview_ilinet",
        "cdc_fluview_who_nrevss",
        "project_tycho",
        "us_births",
    ],
}

BATCH_SIZE = 100_000


@dataclass(frozen=True)
class SeriesInfo:
    series_id: int
    freq: str
    context_len: int
    n_windows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_horizon", type=int, required=True)
    parser.add_argument("--target_n", type=int, default=16_000_000)
    parser.add_argument("--cap_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="lotsa_index.parquet")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def subset_to_domains() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for domain, subsets in DOMAIN_MAP.items():
        for subset_name in subsets:
            mapping[subset_name].add(domain)
    return dict(mapping)


def load_subset(subset_name: str, hf_cache_dir: str | None):
    return load_dataset(
        "Salesforce/lotsa_data",
        subset_name,
        split="train",
        streaming=False,
        cache_dir=hf_cache_dir,
    )


def scan_subset(subset_name: str, max_horizon: int, hf_cache_dir: str | None) -> list[SeriesInfo]:
    dataset = load_subset(subset_name, hf_cache_dir)
    usable: list[SeriesInfo] = []
    skipped_freq: Counter[str] = Counter()
    skipped_short = 0
    read_errors = 0

    for series_id, row in enumerate(tqdm(dataset, desc=f"scan {subset_name}", leave=False)):
        try:
            freq = row_freq(row)
            if freq not in CONTEXT_BY_FREQ:
                skipped_freq[str(freq)] += 1
                continue
            context_len = CONTEXT_BY_FREQ[freq]
            length = series_len(row)
        except Exception:
            read_errors += 1
            continue

        if length < context_len + max_horizon:
            skipped_short += 1
            continue

        n_windows = length - context_len - max_horizon + 1
        if n_windows <= 0:
            skipped_short += 1
            continue
        usable.append(
            SeriesInfo(
                series_id=series_id,
                freq=freq,
                context_len=context_len,
                n_windows=n_windows,
            )
        )

    pool_windows = sum(info.n_windows for info in usable)
    skip_freq_msg = ", ".join(f"{freq}:{count}" for freq, count in sorted(skipped_freq.items()))
    if not skip_freq_msg:
        skip_freq_msg = "none"
    print(
        f"[pool] {subset_name}: usable_series={len(usable)} pool_windows={pool_windows} "
        f"skipped_short={skipped_short} skipped_freq={skip_freq_msg} read_errors={read_errors}",
        flush=True,
    )
    return usable


def build_pool(max_horizon: int, hf_cache_dir: str | None) -> dict[str, list[SeriesInfo]]:
    pool: dict[str, list[SeriesInfo]] = {}
    for subset_name in sorted(all_subset_names()):
        try:
            infos = scan_subset(subset_name, max_horizon, hf_cache_dir)
        except Exception as exc:
            print(f"[warning] failed to scan {subset_name}: {exc}", flush=True)
            infos = []
        if infos:
            pool[subset_name] = infos
    return pool


def output_counts(output_path: Path) -> tuple[int, Counter[str], Counter[str]]:
    subset_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    total = 0
    subset_domains = subset_to_domains()

    parquet_file = pq.ParquetFile(output_path)
    for batch in parquet_file.iter_batches(columns=["subset_name"]):
        names = batch.column(0).to_pylist()
        total += len(names)
        subset_counts.update(names)
        for subset_name in names:
            for domain in subset_domains.get(subset_name, ()):
                domain_counts[domain] += 1
    return total, subset_counts, domain_counts


def all_subset_names() -> set[str]:
    return {subset_name for subsets in DOMAIN_MAP.values() for subset_name in subsets}


def copy_existing_rows(src: Path, writer: pq.ParquetWriter) -> None:
    parquet_file = pq.ParquetFile(src)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        writer.write_table(rows_to_table(batch.to_pylist()))


def choose_subset(
    rng: random.Random,
    domain: str,
    subset_counts: Counter[str],
    pool: dict[str, list[SeriesInfo]],
    cap_per_subset: float,
) -> str | None:
    active = [
        subset_name
        for subset_name in DOMAIN_MAP[domain]
        if subset_name in pool
    ]
    under_cap = [
        subset_name
        for subset_name in active
        if subset_counts[subset_name] < cap_per_subset
    ]
    candidates = under_cap if under_cap else active
    if not candidates:
        return None
    return rng.choice(candidates)


def choose_series(rng: random.Random, infos: list[SeriesInfo]) -> SeriesInfo:
    weights = [info.n_windows for info in infos]
    return rng.choices(infos, weights=weights, k=1)[0]


def domain_has_pool(domain: str, pool: dict[str, list[SeriesInfo]]) -> bool:
    return any(subset_name in pool for subset_name in DOMAIN_MAP[domain])


def choose_domain(
    rng: random.Random,
    domains: list[str],
    pool: dict[str, list[SeriesInfo]],
) -> str | None:
    candidates = [domain for domain in domains if domain_has_pool(domain, pool)]
    if not candidates:
        return None
    return rng.choice(candidates)


def print_stats(subset_counts: Counter[str], domain_counts: Counter[str]) -> None:
    print("\nDomain contribution:")
    for domain in sorted(DOMAIN_MAP):
        print(f"  {domain}: {domain_counts[domain]}")

    print("\nSub-dataset contribution:")
    for subset_name, count in sorted(subset_counts.items()):
        print(f"  {subset_name}: {count}")


def main() -> None:
    args = parse_args()
    if args.max_horizon <= 0:
        raise ValueError("--max_horizon must be positive.")
    if args.target_n <= 0:
        raise ValueError("--target_n must be positive.")
    if args.cap_ratio <= 0:
        raise ValueError("--cap_ratio must be positive.")

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_count = 0
    subset_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()

    if args.resume and output_path.exists():
        total_count, subset_counts, domain_counts = output_counts(output_path)
        print(f"[resume] existing rows: {total_count}")
        if total_count >= args.target_n:
            print_stats(subset_counts, domain_counts)
            return
    elif output_path.exists():
        raise FileExistsError(f"{output_path} already exists. Use --resume to continue from it.")

    cap_per_subset = (args.target_n / len(DOMAIN_MAP)) * args.cap_ratio
    domains = sorted(DOMAIN_MAP)
    subset_domains = subset_to_domains()
    rows: list[dict[str, Any]] = []
    pool = build_pool(args.max_horizon, args.hf_cache_dir)
    total_pool_windows = sum(info.n_windows for infos in pool.values() for info in infos)
    print(f"\nPool summary: subsets={len(pool)} total_windows={total_pool_windows}", flush=True)
    if not pool:
        raise RuntimeError("No eligible windows found in configured sub-datasets.")

    if tmp_path.exists():
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path)
        else:
            tmp_path.unlink()

    writer = pq.ParquetWriter(tmp_path, index_schema())
    try:
        if args.resume and output_path.exists():
            copy_existing_rows(output_path, writer)

        progress = tqdm(total=args.target_n, initial=total_count, desc="sampling windows")
        try:
            while total_count < args.target_n:
                domain = choose_domain(rng, domains, pool)
                if domain is None:
                    raise RuntimeError(
                        f"No eligible sub-datasets remain before target_n={args.target_n}; "
                        f"collected={total_count}."
                    )
                subset_name = choose_subset(rng, domain, subset_counts, pool, cap_per_subset)
                assert subset_name is not None
                info = choose_series(rng, pool[subset_name])
                win_start = rng.randint(0, info.n_windows - 1)
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
                subset_counts[subset_name] += 1
                for owning_domain in subset_domains.get(subset_name, (domain,)):
                    domain_counts[owning_domain] += 1
                total_count += 1
                progress.update(1)

                if len(rows) >= BATCH_SIZE:
                    writer.write_table(rows_to_table(rows))
                    rows.clear()

            if rows:
                writer.write_table(rows_to_table(rows))
                rows.clear()
        finally:
            progress.close()
    finally:
        writer.close()

    os.replace(tmp_path, output_path)
    print(f"\nSaved: {output_path}")
    print_stats(subset_counts, domain_counts)


if __name__ == "__main__":
    main()
