#!/usr/bin/env python3
"""Generate calibration synthetic datasets without touching eval artifacts."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from synth_generator import (
    LEVELS,
    SEASONAL_GRANULARITIES,
    SEASONAL_LEVELS,
    VALID_HORIZONS,
    generate_seasonal_dataset,
    generate_trend_dataset,
    load_config,
    max_break_discontinuity,
    save_seasonal_dataset,
    save_trend_dataset,
)


ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "synthetic"
DEFAULT_OUTPUT_ROOT = ROOT / "calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "synth_config.yaml")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--category", choices=["trend", "seasonal", "all"], default="all")
    parser.add_argument("--trend_levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument(
        "--seasonal_levels",
        nargs="+",
        default=SEASONAL_LEVELS,
        choices=SEASONAL_LEVELS,
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    global_cfg = cfg["global"]

    horizons = args.horizons or list(global_cfg["horizons"])
    context_len = args.context_len or int(global_cfg["context_len"])
    seed = args.seed if args.seed is not None else int(global_cfg["seed"])

    for horizon in horizons:
        if horizon not in VALID_HORIZONS:
            raise ValueError(
                f"Invalid horizon {horizon}; expected one of {sorted(VALID_HORIZONS)}"
            )

    trend_dir = args.output_root / "trend"
    seasonal_dir = args.output_root / "seasonal"
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config),
        "output_root": str(args.output_root),
        "trend_levels": list(args.trend_levels),
        "seasonal_levels": list(args.seasonal_levels),
        "horizons": list(horizons),
        "context_len": int(context_len),
        "n_samples": int(args.n_samples),
        "seed": int(seed),
        "note": "Calibration synthetic data generated with the existing synth_generator level definitions.",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.category in {"trend", "all"}:
        for level in args.trend_levels:
            for horizon in horizons:
                dataset = generate_trend_dataset(
                    level=level,
                    horizon=horizon,
                    context_len=context_len,
                    n_samples=args.n_samples,
                    seed=seed,
                    cfg=cfg,
                )
                discontinuity = max_break_discontinuity(dataset["meta"])
                if discontinuity >= 1e-4:
                    raise AssertionError(
                        f"Continuity check failed for {level}, h={horizon}: {discontinuity}"
                    )
                saved_path = save_trend_dataset(dataset, trend_dir)
                print(
                    f"trend saved={saved_path} future_n={dataset['future_n'].shape} "
                    f"max_discontinuity={discontinuity:.3e}",
                    flush=True,
                )

    if args.category in {"seasonal", "all"}:
        for level in args.seasonal_levels:
            for granularity in SEASONAL_GRANULARITIES[level]:
                for horizon in horizons:
                    dataset = generate_seasonal_dataset(
                        level=level,
                        granularity=granularity,
                        horizon=horizon,
                        context_len=context_len,
                        n_samples=args.n_samples,
                        seed=seed,
                        cfg=cfg,
                    )
                    saved_path = save_seasonal_dataset(dataset, seasonal_dir)
                    print(
                        f"seasonal saved={saved_path} future_n={dataset['future_n'].shape} "
                        f"active={','.join(dataset['active_types'])}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
