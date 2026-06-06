from __future__ import annotations

import argparse
import json
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = next(parent for parent in THIS_FILE.parents if (parent / "loader_utils.py").exists())
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import resolve_data_path, resolve_project_path  # noqa: E402

import torch


DEFAULT_TRAIN_ROOT = resolve_data_path("/home/sia2/project/data/synthetic_nonF/synth_train_nonfourier")
DEFAULT_OUTPUT_ROOT = Path("/home/sia2/project/5.22syn_cent/train_nonF_projtarget/projection_targets")
STAGE_CACHE_DIRS = [
    "stage1_S_nonfourier_cache_10_4_8",
    "stage2_T_S_nonfourier_cache_10_4_8",
    "stage3_T_S_R_nonfourier_cache_10_4_8",
]
FAMILIES = ["daily", "weekly", "yearly"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project non-Fourier seasonal targets onto the Fourier basis.")
    parser.add_argument("--train_root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stages", nargs="+", default=STAGE_CACHE_DIRS, choices=STAGE_CACHE_DIRS)
    parser.add_argument("--rcond", type=float, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def basis_and_slices(basis_payload: dict) -> tuple[torch.Tensor, dict[str, slice]]:
    tensors = []
    slices: dict[str, slice] = {}
    start = 0
    for family in FAMILIES:
        basis = basis_payload[f"{family}_basis"].double()
        width = int(basis.shape[1])
        tensors.append(basis)
        slices[family] = slice(start, start + width)
        start += width
    return torch.cat(tensors, dim=1), slices


def project_targets(ds_dir: Path, horizon: int, rcond: float | None) -> dict:
    basis_payload = torch.load(ds_dir / f"fourier_basis_h{horizon}.pt", map_location="cpu", weights_only=False)
    component_payload = torch.load(ds_dir / f"component_targets_h{horizon}.pt", map_location="cpu", weights_only=False)

    seasonal = component_payload["seasonal_n"].double()
    trend = component_payload["trend_n"].float()
    noise_metric_only = component_payload["residual_n"].float()
    basis, slices = basis_and_slices(basis_payload)

    active_cols = torch.linalg.vector_norm(basis, dim=0) > 0
    coeff_all = torch.zeros(seasonal.shape[0], basis.shape[1], dtype=torch.float64)
    projected = torch.zeros_like(seasonal)
    if bool(active_cols.any()):
        active_basis = basis[:, active_cols]
        solution = torch.linalg.lstsq(active_basis, seasonal.T, rcond=rcond).solution
        coeff_all[:, active_cols] = solution.T
        projected = (active_basis @ solution).T

    remainder = seasonal - projected
    coeffs = {
        f"{family}_coefficients": coeff_all[:, sl].float()
        for family, sl in slices.items()
    }
    coeffs["mask"] = dict(basis_payload.get("mask", {}))
    coeffs["n_fourier_terms"] = dict(basis_payload.get("n_fourier_terms", {}))
    coeffs["horizon"] = int(horizon)
    coeffs["granularity"] = str(basis_payload.get("freq") or basis_payload.get("granularity") or "")
    coeffs["note"] = "Least-squares projection of non-Fourier seasonal_n onto the cached Fourier basis."

    targets = {
        "trend_n": trend,
        "seasonal_projection_n": projected.float(),
        "seasonal_remainder_n": remainder.float(),
        "clean_total_n": (trend.double() + seasonal).float(),
        "noise_metric_only_n": noise_metric_only,
        "note": (
            "Training target policy: trend decoder learns trend_n; seasonal decoder learns projection "
            "coefficients; residual decoder learns seasonal_remainder_n. Distribution noise is not supervised."
        ),
    }
    err = remainder.float()
    stats = {
        "n_samples": int(seasonal.shape[0]),
        "horizon": int(horizon),
        "basis_cols": int(basis.shape[1]),
        "active_cols": int(active_cols.sum().item()),
        "seasonal_std": float(seasonal.float().std().item()),
        "projection_mae": float(err.abs().mean().item()),
        "projection_rmse": float(torch.sqrt((err * err).mean()).item()),
    }
    return {"coefficients": coeffs, "targets": targets, "stats": stats}


def output_dir(output_root: Path, train_root: Path, ds_dir: Path) -> Path:
    return output_root / ds_dir.relative_to(train_root)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_root": str(args.train_root),
        "output_root": str(args.output_root),
        "stages": list(args.stages),
        "rcond": args.rcond,
        "policy": {
            "trend": "trend_n",
            "seasonal": "projected Fourier coefficients",
            "residual": "seasonal_n minus projected seasonal_n",
            "distribution_noise": "excluded from training target",
        },
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows = []
    for stage in args.stages:
        stage_root = args.train_root / stage
        for raw_path in sorted(stage_root.rglob("raw_futures_h*.pt")):
            ds_dir = raw_path.parent
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            horizon = int(raw["horizon"])
            out_dir = output_dir(args.output_root, args.train_root, ds_dir)
            coeff_path = out_dir / f"projected_seasonal_coefficients_h{horizon}.pt"
            target_path = out_dir / f"projected_component_targets_h{horizon}.pt"
            if args.skip_existing and coeff_path.exists() and target_path.exists():
                print(f"skip_existing={out_dir}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            result = project_targets(ds_dir, horizon, args.rcond)
            torch.save(result["coefficients"], coeff_path)
            torch.save(result["targets"], target_path)
            rows.append({"source": str(ds_dir), "output": str(out_dir), **result["stats"]})
            print(
                f"saved_projection={out_dir} n={result['stats']['n_samples']} "
                f"h={horizon} active_cols={result['stats']['active_cols']} "
                f"remainder_mae={result['stats']['projection_mae']:.6g}",
                flush=True,
            )

    if rows:
        summary = {
            "num_groups": len(rows),
            "total_samples": int(sum(row["n_samples"] for row in rows)),
            "mean_projection_mae": float(sum(row["projection_mae"] for row in rows) / len(rows)),
            "rows": rows,
        }
        (args.output_root / "projection_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"projection_summary groups={summary['num_groups']} "
            f"samples={summary['total_samples']} mean_mae={summary['mean_projection_mae']:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
