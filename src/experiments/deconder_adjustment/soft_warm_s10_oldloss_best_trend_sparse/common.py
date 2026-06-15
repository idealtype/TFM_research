from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = next(
    parent for parent in THIS_DIR.parents if (parent / "loader_utils.py").exists()
)
sys.path.insert(0, str(EXPERIMENTS_ROOT))
from loader_utils import resolve_project_path  # noqa: E402
PROJECT_ROOT_4_28 = resolve_project_path("/home/sia2/project/4.28basis")
SRC_DIR = PROJECT_ROOT_4_28 / "src"
for path in [THIS_DIR, PROJECT_ROOT_4_28, SRC_DIR]:
    path_s = str(path)
    while path_s in sys.path:
        sys.path.remove(path_s)
for path in [SRC_DIR, PROJECT_ROOT_4_28, THIS_DIR]:
    sys.path.insert(0, str(path))

from model.decomp_funcdec import FuncDecModel  # noqa: E402
from model.decoder_seasonal import N_FOURIER_TERMS, N_GATES  # noqa: E402


HORIZONS = [96, 192, 336, 720]

# Periods in days
PERIODS = {"daily": 1.0, "weekly": 7.0, "monthly": 30.4375, "yearly": 365.25}
# Maximum harmonic order per family
K_MAX = {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8}
FAMILIES = ["daily", "weekly", "monthly", "yearly"]
GATE_SLICES = {
    "daily": (0, 10),
    "weekly": (10, 14),
    "monthly": (14, 16),
    "yearly": (16, 24),
}

FREQ_DAYS = {
    "5_minutes": 1 / 288,
    "10_minutes": 1 / 144,
    "15_minutes": 1 / 96,
    "half_hourly": 1 / 48,
    "hourly": 1 / 24,
    "H": 1 / 24,
    "D": 1.0,
    "daily": 1.0,
    "weekly": 7.0,
    "monthly": 30.4375,
    "yearly": 365.25,
}

DEFAULT_CONFIG = {
    "context_len": 512,
    "embed_dim": 1280,
    "n_knots": {"96": 10, "192": 20, "336": 40, "720": 80},
    "n_fourier_terms": {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8},
    "mlp_units": {
        "trend": [1280, 1280],
        "seasonal": [1280, 1280],
        "residual": [1280, 1280],
    },
    "activation": "ReLU",
    "dropout": 0.0,
}

MODEL_NAME = os.environ.get("FUNCDEC_MODEL_NAME", "soft_mask")
DEFAULT_RESULTS_ROOT = THIS_DIR / "results"
DEFAULT_CHECKPOINT_RUN_DIR = THIS_DIR / "results"


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def build_config(horizon: int, run_cfg: dict | None = None, ckpt_cfg: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for source in (run_cfg or {}, ckpt_cfg or {}):
        for key in DEFAULT_CONFIG:
            if key in source:
                if isinstance(cfg.get(key), dict) and isinstance(source[key], dict):
                    merged = dict(cfg[key])
                    merged.update(source[key])
                    cfg[key] = merged
                else:
                    cfg[key] = source[key]
    cfg["context_len"] = int(cfg["context_len"])
    cfg["horizon"] = int(horizon)
    if str(horizon) not in cfg["n_knots"]:
        raise ValueError(f"n_knots does not include horizon={horizon}")
    return cfg


def build_fine_mask_basis(freq: str, context_len: int, horizon: int) -> dict[str, torch.Tensor]:
    """Build per-harmonic masked Fourier basis for all 4 periods.

    Per-harmonic activation rule:
        active = (fd < P/k) AND (context_span >= P/k)
    where fd = FREQ_DAYS[freq], context_span = context_len * fd

    t starts from context_len (not 0) for phase alignment.
    """
    if freq not in FREQ_DAYS:
        raise KeyError(f"Unknown frequency for Fourier basis: {freq}")
    fd = FREQ_DAYS[freq]
    context_span = context_len * fd
    # Critical: t starts at context_len, not 0
    t = torch.arange(int(context_len), int(context_len) + int(horizon), dtype=torch.float32)

    result = {}
    for family, P in PERIODS.items():
        k_max = K_MAX[family]
        basis = torch.zeros(horizon, 2 * k_max, dtype=torch.float32)
        p_steps = P / fd  # period in steps
        for k in range(1, k_max + 1):
            harmonic_period = P / k  # harmonic period in days
            # Per-harmonic activation: fd must be finer than harmonic, and context must span it
            if fd < harmonic_period and context_span >= harmonic_period:
                basis[:, 2 * (k - 1)] = torch.sin(2 * math.pi * k * t / p_steps)
                basis[:, 2 * (k - 1) + 1] = torch.cos(2 * math.pi * k * t / p_steps)
        # Return with bare key (consistent with load_fine_mask_basis output)
        result[family] = basis
        # Also include _basis suffix key for file-save compatibility
        result[f"{family}_basis"] = basis
    return result


def build_soft_mask_basis(freq: str, horizon: int) -> dict[str, torch.Tensor]:
    """Build Fourier basis with PHYSICS-ONLY masking (no context_span condition).

    Activation rule: active = (fd < P/k)
    The context_span condition from fine_mask is intentionally removed.
    The soft gating network (HarmonicGatingNetwork) learns this discrimination.

    Note: context_len is NOT an argument — the basis depends only on freq + horizon.
    This makes on-the-fly computation trivial and avoids per-context-len caching.

    t starts from 512 (default context_len) for phase alignment with cached embeddings.
    If context_len differs from 512, the phase is slightly off, but the gate can compensate.
    """
    if freq not in FREQ_DAYS:
        raise KeyError(f"Unknown frequency for soft basis: {freq}")
    fd = FREQ_DAYS[freq]
    # Phase reference at context_len=512 (matches all backbone_emb caches)
    PHASE_CONTEXT_LEN = 512
    t = torch.arange(PHASE_CONTEXT_LEN, PHASE_CONTEXT_LEN + int(horizon), dtype=torch.float32)

    result = {}
    for family, P in PERIODS.items():
        k_max = K_MAX[family]
        basis = torch.zeros(horizon, 2 * k_max, dtype=torch.float32)
        p_steps = P / fd
        for k in range(1, k_max + 1):
            harmonic_period = P / k
            # Physics-only: only check if the data granularity can resolve this harmonic
            if fd < harmonic_period:
                basis[:, 2 * (k - 1)]     = torch.sin(2 * math.pi * k * t / p_steps)
                basis[:, 2 * (k - 1) + 1] = torch.cos(2 * math.pi * k * t / p_steps)
            # else: fd >= harmonic_period → physically unresolvable → stays 0
        result[family] = basis
        result[f"{family}_basis"] = basis   # _basis suffix for file-save compatibility
    return result


def limit_or_pad_basis(basis: torch.Tensor, family: str) -> torch.Tensor:
    """Truncate or zero-pad basis to match HEAD capacity from N_FOURIER_TERMS."""
    target = 2 * int(N_FOURIER_TERMS[family])
    basis = basis.float()
    if basis.shape[1] == target:
        return basis
    if basis.shape[1] > target:
        return basis[:, :target]
    pad = torch.zeros(basis.shape[0], target - basis.shape[1], dtype=basis.dtype)
    return torch.cat([basis, pad], dim=1)


def load_fine_mask_basis(path: Path) -> dict[str, torch.Tensor]:
    """Load a fine_mask basis file and return standardized dict with 4 families."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "daily": limit_or_pad_basis(payload["daily_basis"], "daily"),
        "weekly": limit_or_pad_basis(payload["weekly_basis"], "weekly"),
        "monthly": limit_or_pad_basis(payload["monthly_basis"], "monthly"),
        "yearly": limit_or_pad_basis(payload["yearly_basis"], "yearly"),
    }


def load_basis_file(path: Path) -> dict[str, torch.Tensor]:
    """Load any basis file (legacy 3-family or new 4-family). Returns 4-family dict."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    for family in ["daily", "weekly", "monthly", "yearly"]:
        key = f"{family}_basis"
        if key in payload:
            result[family] = limit_or_pad_basis(payload[key], family)
        else:
            # Missing family (e.g., monthly in legacy files): return zeros
            k_max = K_MAX[family]
            # We need horizon from one of the existing tensors
            existing = next(iter(v for k, v in payload.items() if isinstance(v, torch.Tensor) and v.ndim == 2), None)
            if existing is not None:
                h = existing.shape[0]
            else:
                h = 96  # fallback
            result[family] = torch.zeros(h, 2 * int(N_FOURIER_TERMS[family]), dtype=torch.float32)
    return result


def upgrade_legacy_basis(path: Path) -> dict[str, torch.Tensor]:
    """Load a legacy 3-family basis file and add zeros for monthly."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    existing_tensor = None
    for family in ["daily", "weekly", "yearly"]:
        key = f"{family}_basis"
        if key in payload:
            t = limit_or_pad_basis(payload[key], family)
            result[family] = t
            existing_tensor = t
    # Add monthly with zeros
    if existing_tensor is not None:
        h = existing_tensor.shape[0]
    else:
        h = 96
    result["monthly"] = torch.zeros(h, 2 * int(N_FOURIER_TERMS["monthly"]), dtype=torch.float32)
    return result


def expand_bases(bases: dict[str, torch.Tensor], batch_size: int, device: torch.device):
    """Expand 4-family basis dict to batch tensors. Returns (daily, weekly, monthly, yearly)."""
    daily = bases["daily"].to(device).unsqueeze(0).expand(batch_size, -1, -1)
    weekly = bases["weekly"].to(device).unsqueeze(0).expand(batch_size, -1, -1)
    monthly = bases["monthly"].to(device).unsqueeze(0).expand(batch_size, -1, -1)
    yearly = bases["yearly"].to(device).unsqueeze(0).expand(batch_size, -1, -1)
    return daily, weekly, monthly, yearly


def split_gates_by_family(gates: torch.Tensor) -> dict[str, torch.Tensor]:
    return {family: gates[:, start:end] for family, (start, end) in GATE_SLICES.items()}


def harmonic_rule_masks(freq: str, context_len: int = 512) -> dict[str, dict[str, torch.Tensor]]:
    """Return per-family hard-rule and physics-only harmonic masks.

    hard_rule matches fine_mask:
        fd < P/k and context_len * fd >= P/k
    physics_only matches soft_mask basis support:
        fd < P/k
    """
    if freq not in FREQ_DAYS:
        raise KeyError(f"Unknown frequency for harmonic rule masks: {freq}")
    fd = FREQ_DAYS[freq]
    context_span = int(context_len) * fd
    masks = {}
    for family, period in PERIODS.items():
        hard_values = []
        physics_values = []
        for k in range(1, K_MAX[family] + 1):
            harmonic_period = period / k
            physics_on = fd < harmonic_period
            hard_values.append(bool(physics_on and context_span >= harmonic_period))
            physics_values.append(bool(physics_on))
        masks[family] = {
            "hard": torch.tensor(hard_values, dtype=torch.bool),
            "physics": torch.tensor(physics_values, dtype=torch.bool),
        }
    return masks


def family_curves_from_coefficients(
    coefficients: dict[str, torch.Tensor],
    bases: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return per-family seasonal curves from coefficient tensors.

    Coefficients are expected in [sin_k1, cos_k1, ...] layout and bases are
    expanded batch tensors in daily/weekly/monthly/yearly order.
    """
    result = {}
    for family, basis in zip(FAMILIES, bases):
        coef = coefficients[family].to(device=basis.device, dtype=basis.dtype)
        result[family] = torch.bmm(basis, coef.unsqueeze(-1)).squeeze(-1)
    return result


def soft_mask_family_contributions(
    decomp: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    bases: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    gates = split_gates_by_family(decomp["gates"])
    coefficients = decomp["seasonal_coefficients"]
    gated = {
        family: coefficients[family] * gates[family].repeat_interleave(2, dim=1)
        for family in FAMILIES
    }
    return family_curves_from_coefficients(gated, bases)


def family_energy(curves: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {family: (curve.detach().float() ** 2).mean(dim=1) for family, curve in curves.items()}


def family_share_from_energy(energy: dict[str, torch.Tensor | float]) -> dict[str, float]:
    values = {
        family: float(torch.as_tensor(energy[family]).detach().float().mean().item())
        for family in FAMILIES
    }
    total = sum(values.values())
    if total <= 1e-12:
        return {family: 0.0 for family in FAMILIES}
    return {family: values[family] / total for family in FAMILIES}


def init_harmonic_activation_accumulator() -> dict[str, list[float]]:
    return {}


def _add_weighted_stat(acc: dict[str, list[float]], key: str, value: torch.Tensor, mask: torch.Tensor) -> None:
    mask = mask.to(device=value.device, dtype=torch.bool)
    if value.ndim == 2 and mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(value.shape[0], -1)
    if not bool(mask.any().item()):
        return
    selected = value.detach().float()[mask]
    if selected.numel() == 0:
        return
    slot = acc.setdefault(key, [0.0, 0.0])
    slot[0] += float(selected.sum().item())
    slot[1] += float(selected.numel())


def add_harmonic_activation_stats(
    acc: dict[str, list[float]],
    decomp: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    freq: str,
    context_len: int,
    gate_threshold: float = 0.5,
    coeff_threshold: float = 1e-6,
) -> None:
    """Accumulate learned harmonic activity against hard-rule masks."""
    masks = harmonic_rule_masks(freq, context_len)
    gates = split_gates_by_family(decomp["gates"])
    coefficients = decomp["seasonal_coefficients"]
    for family in FAMILIES:
        gate = gates[family].detach().float()
        raw_coef = coefficients[family].detach().float()
        coef_amp = raw_coef.view(raw_coef.shape[0], -1, 2).norm(dim=2)
        gated_amp = coef_amp * gate
        gate_on = (gate >= float(gate_threshold)).float()
        raw_on = (coef_amp > float(coeff_threshold)).float()
        gated_on = (gated_amp > float(coeff_threshold)).float()

        hard_mask = masks[family]["hard"].to(gate.device)
        physics_mask = masks[family]["physics"].to(gate.device)
        extra_mask = physics_mask & ~hard_mask

        for prefix, mask in [
            ("soft_support", physics_mask),
            ("hard_rule", hard_mask),
            ("soft_extra", extra_mask),
        ]:
            _add_weighted_stat(acc, f"{prefix}_gate_active_rate_{family}", gate_on, mask)
            _add_weighted_stat(acc, f"{prefix}_raw_coeff_active_rate_{family}", raw_on, mask)
            _add_weighted_stat(acc, f"{prefix}_gated_coeff_active_rate_{family}", gated_on, mask)
            _add_weighted_stat(acc, f"{prefix}_mean_gate_{family}", gate, mask)
            _add_weighted_stat(acc, f"{prefix}_mean_raw_coeff_amp_{family}", coef_amp, mask)
            _add_weighted_stat(acc, f"{prefix}_mean_gated_coeff_amp_{family}", gated_amp, mask)

        _add_weighted_stat(acc, f"hard_rule_support_rate_{family}", hard_mask.float(), physics_mask)
        _add_weighted_stat(acc, f"soft_extra_support_rate_{family}", extra_mask.float(), physics_mask)
        _add_weighted_stat(acc, "hard_rule_support_rate_all", hard_mask.float(), physics_mask)
        _add_weighted_stat(acc, "soft_extra_support_rate_all", extra_mask.float(), physics_mask)
        _add_weighted_stat(acc, "soft_support_gate_active_rate_all", gate_on, physics_mask)
        _add_weighted_stat(acc, "hard_rule_gate_active_rate_all", gate_on, hard_mask)
        _add_weighted_stat(acc, "soft_extra_gate_active_rate_all", gate_on, extra_mask)
        _add_weighted_stat(acc, "soft_support_gated_coeff_active_rate_all", gated_on, physics_mask)
        _add_weighted_stat(acc, "hard_rule_gated_coeff_active_rate_all", gated_on, hard_mask)
        _add_weighted_stat(acc, "soft_extra_gated_coeff_active_rate_all", gated_on, extra_mask)


def finalize_harmonic_activation_stats(acc: dict[str, list[float]]) -> dict[str, float | None]:
    return {
        key: (None if count <= 0 else total / count)
        for key, (total, count) in sorted(acc.items())
    }


def harmonic_activation_fieldnames() -> list[str]:
    fields = []
    for prefix in ["soft_support", "hard_rule", "soft_extra"]:
        for metric in [
            "gate_active_rate",
            "raw_coeff_active_rate",
            "gated_coeff_active_rate",
            "mean_gate",
            "mean_raw_coeff_amp",
            "mean_gated_coeff_amp",
        ]:
            fields.extend(f"{prefix}_{metric}_{family}" for family in FAMILIES)
    fields.extend(f"hard_rule_support_rate_{family}" for family in FAMILIES)
    fields.extend(f"soft_extra_support_rate_{family}" for family in FAMILIES)
    fields.extend([
        "hard_rule_support_rate_all",
        "soft_extra_support_rate_all",
        "soft_support_gate_active_rate_all",
        "hard_rule_gate_active_rate_all",
        "soft_extra_gate_active_rate_all",
        "soft_support_gated_coeff_active_rate_all",
        "hard_rule_gated_coeff_active_rate_all",
        "soft_extra_gated_coeff_active_rate_all",
    ])
    return fields


def checkpoint_candidates(run_dir: Path, horizon: int) -> list[Path]:
    return [
        run_dir / "checkpoints" / f"funcdec_h{horizon}.pt",
        run_dir / "checkpoints" / f"nonfourier_finetune_h{horizon}.pt",
        run_dir / "checkpoints" / f"simple_complex_synth_h{horizon}.pt",
    ]


def load_checkpoint_payload(path: Path, device: torch.device) -> tuple[dict, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("config", {})
    return payload, {}


def adapt_state_dict_for_fine_mask(state: dict, model: FuncDecModel) -> dict:
    """Kept for compatibility. Delegates to adapt_state_dict_for_soft_mask."""
    return adapt_state_dict_for_soft_mask(state, model)


def adapt_state_dict_for_soft_mask(state: dict, model: FuncDecModel) -> dict:
    """Adapt a checkpoint to the soft_mask model.

    Handles:
    - backbone.* keys: skipped (frozen, not saved in decoder-only checkpoints)
    - decoder_s.gating_network.* keys missing in old ckpt: allowed (random init)
    - Size mismatch for HEAD weights: truncate/pad as in fine_mask
    - Unknown keys: skipped silently
    """
    model_state = model.state_dict()
    adapted = {}
    for key, param in state.items():
        if key.startswith("backbone."):
            continue
        if key not in model_state:
            continue
        target_shape = model_state[key].shape
        if param.shape == target_shape:
            adapted[key] = param
        elif param.ndim == len(target_shape):
            slices = tuple(slice(0, s) for s in target_shape)
            if all(param.shape[i] >= target_shape[i] for i in range(len(target_shape))):
                adapted[key] = param[slices]
            else:
                t = torch.zeros(target_shape, dtype=param.dtype)
                src_slices = tuple(
                    slice(0, min(param.shape[i], target_shape[i]))
                    for i in range(len(target_shape))
                )
                t[src_slices] = param[src_slices]
                adapted[key] = t
    return adapted


def load_single_model(run_dir: Path, horizon: int, device: torch.device) -> tuple[FuncDecModel, Path, dict]:
    """Load a soft_mask model checkpoint. Uses strict=False + weight adaptation for legacy checkpoints."""
    found = next((path for path in checkpoint_candidates(run_dir, horizon) if path.exists()), None)
    if found is None:
        tried = "\n".join(str(path) for path in checkpoint_candidates(run_dir, horizon))
        raise FileNotFoundError(f"No checkpoint for h{horizon}. Tried:\n{tried}")

    state, ckpt_cfg = load_checkpoint_payload(found, device)
    cfg = build_config(horizon, read_json(run_dir.parent / "config.json" if (run_dir.parent / "config.json").exists() else run_dir / "config.json"), ckpt_cfg)
    model = FuncDecModel(cfg, load_backbone=False).to(device)

    # First adapt the state dict to handle size mismatches (legacy HEAD capacity)
    adapted_state = adapt_state_dict_for_soft_mask(state, model)
    incompatible = model.load_state_dict(adapted_state, strict=False)

    # Allowed missing: gating_network (new in soft_mask), monthly keys from old ckpts
    allowed_missing_prefixes = ("decoder_s.gating_network.", "decoder_s.mlp_monthly.",
                                "decoder_s.forecast_head_monthly.")
    missing = [
        key for key in incompatible.missing_keys
        if key.startswith("decoder_") and not any(key.startswith(p) for p in allowed_missing_prefixes)
    ]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("backbone.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {found}: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    newly_init = [k for k in incompatible.missing_keys
                  if any(k.startswith(p) for p in allowed_missing_prefixes)]
    if newly_init:
        print(f"  [info] randomly initialized: {len(newly_init)} keys "
              f"(gating_network or monthly)", flush=True)
    model.eval()
    return model, found, cfg


def load_tfm_zeroshot_model(context_len: int, horizon: int, hf_cache_dir: str | None = None):
    from timesfm.configs import ForecastConfig
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    max_horizon = int(math.ceil(max(int(horizon), 128) / 128) * 128)
    model = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
        cache_dir=hf_cache_dir,
    )
    model.compile(
        ForecastConfig(
            max_context=int(context_len),
            max_horizon=max_horizon,
            per_core_batch_size=64,
            force_flip_invariance=True,
            normalize_inputs=True,
            infer_is_positive=False,
        )
    )
    model.model.eval()
    return model


def select_indices(valid_mask: torch.Tensor | None, n_rows: int, limit: int, seed: int) -> list[int]:
    if valid_mask is None:
        valid = np.arange(n_rows)
    else:
        valid = valid_mask.bool().nonzero(as_tuple=True)[0].cpu().numpy()
    if limit > 0 and len(valid) > limit:
        rng = np.random.default_rng(seed)
        valid = np.sort(rng.choice(valid, size=limit, replace=False))
    return [int(i) for i in valid]


def metric_accumulator() -> dict:
    return {"abs_sum": 0.0, "sq_sum": 0.0, "n": 0}


def add_error(acc: dict, pred: torch.Tensor, target: torch.Tensor) -> None:
    diff = (pred - target).detach().float().cpu()
    acc["abs_sum"] += float(diff.abs().sum().item())
    acc["sq_sum"] += float((diff * diff).sum().item())
    acc["n"] += int(diff.numel())


def add_optional_error(acc: dict, pred: torch.Tensor | None, target: torch.Tensor) -> None:
    if pred is None:
        return
    add_error(acc, pred, target)


def finalize_mae(acc: dict) -> float:
    return acc["abs_sum"] / max(1, acc["n"])


def finalize_mse(acc: dict) -> float:
    return acc["sq_sum"] / max(1, acc["n"])


def mean_or_empty(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, args: dict, rows: list[dict], group_fields: list[str]) -> None:
    metric_fields = [
        "total_mae",
        "total_mse",
        "trend_mae",
        "seasonal_mae",
        "residual_mae",
        "tfm_zeroshot_mae",
        "tfm_zeroshot_mse",
    ]
    grouped: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        bucket = grouped.setdefault(key, {field: row.get(field, "") for field in group_fields})
        for metric in metric_fields:
            bucket.setdefault(metric, []).append(row.get(metric))

    summary_rows = []
    for bucket in grouped.values():
        summary_row = {field: bucket[field] for field in group_fields}
        for metric in metric_fields:
            summary_row[metric] = mean_or_empty(bucket.get(metric, []))
        summary_rows.append(summary_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {
                "args": to_jsonable(args),
                "n_rows": len(rows),
                "group_fields": group_fields,
                "group_means": to_jsonable(summary_rows),
            },
            f,
            indent=2,
        )


def plot_forecast_sample(
    path: Path,
    title: str,
    future: torch.Tensor,
    pred: torch.Tensor,
    decomp: dict[str, torch.Tensor],
    target_components: dict[str, torch.Tensor] | None = None,
) -> None:
    x = np.arange(int(future.numel()))
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    axes[0].plot(x, future.detach().cpu(), color="black", linestyle="--", label="target")
    axes[0].plot(x, pred.detach().cpu(), label=MODEL_NAME)
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, decomp["trend"].detach().cpu(), label="pred trend")
    axes[1].plot(x, decomp["seasonal"].detach().cpu(), label="pred seasonal")
    axes[1].plot(x, decomp["residual"].detach().cpu(), label="pred residual")
    if target_components is not None:
        axes[1].plot(x, target_components["trend"].detach().cpu(), linestyle="--", alpha=0.65, label="target trend")
        axes[1].plot(
            x, target_components["seasonal"].detach().cpu(),
            linestyle="--", alpha=0.65, label="target seasonal",
        )
        axes[1].plot(
            x, target_components["residual"].detach().cpu(),
            linestyle="--", alpha=0.65, label="target residual",
        )
    axes[1].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_real_comparison_grid(path: Path, title: str, items: list[dict]) -> None:
    if not items:
        return
    shown = items[:3]
    fig, axes = plt.subplots(len(shown), 3, figsize=(16.5, 3.6 * len(shown)), squeeze=False)
    family_colors = {
        "daily": "#1f77b4",
        "weekly": "#ff7f0e",
        "monthly": "#9467bd",
        "yearly": "#8c564b",
    }
    for row_idx, item in enumerate(shown):
        x = np.arange(int(item["future"].numel()))
        ax = axes[row_idx][0]
        ax.plot(x, item["future"], color="black", linestyle="--", label="GT")
        if item.get("tfm_pred") is not None:
            ax.plot(x, item["tfm_pred"], color="#2ca02c", label="TimesFM ZS")
        ax.set_title(f"TimesFM sample={item.get('source_idx', row_idx)}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        ax = axes[row_idx][1]
        ax.plot(x, item["future"], color="black", linestyle="--", label="GT")
        ax.plot(x, item["pred"], color="#d62728", label=MODEL_NAME)
        ax.plot(x, item["decomp"]["trend"], label="trend", alpha=0.8)
        ax.plot(x, item["decomp"]["seasonal"], label="seasonal", alpha=0.8)
        ax.plot(x, item["decomp"]["residual"], label="residual", alpha=0.8)
        interp = item.get("interpretability")
        if interp:
            lines = []
            gt_ratio = interp.get("gt_seasonal_future_energy_ratio")
            pred_ratio = interp.get("pred_seasonal_future_energy_ratio")
            if gt_ratio is not None:
                lines.append(f"GT seasonal/GT: {100.0 * float(gt_ratio):.1f}%")
            if pred_ratio is not None:
                lines.append(f"Pred seasonal/GT: {100.0 * float(pred_ratio):.1f}%")
            shares = interp.get("pred_family_share") or {}
            if shares:
                lines.append(
                    "Pred D/W/M/Y: "
                    + "/".join(f"{100.0 * float(shares.get(f, 0.0)):.0f}" for f in FAMILIES)
                    + "%"
                )
            gt_shares = interp.get("gt_family_share") or {}
            if gt_shares:
                lines.append(
                    "GT D/W/M/Y: "
                    + "/".join(f"{100.0 * float(gt_shares.get(f, 0.0)):.0f}" for f in FAMILIES)
                    + "%"
                )
            if lines:
                ax.text(
                    0.01, 0.98, "\n".join(lines),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
                )
        ax.set_title(f"FuncDec components sample={item.get('source_idx', row_idx)}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        ax = axes[row_idx][2]
        family_curves = item.get("seasonal_families") or {}
        if family_curves:
            ax.plot(x, item["decomp"]["seasonal"], color="black", linewidth=1.4, label="seasonal total")
            for family in FAMILIES:
                curve = family_curves.get(family)
                if curve is not None:
                    ax.plot(x, curve, color=family_colors[family], alpha=0.9, label=family)
        else:
            ax.plot(x, item["decomp"]["seasonal"], color="black", linewidth=1.4, label="seasonal total")
        ax.axhline(0.0, color="0.65", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Seasonal families sample={item.get('source_idx', row_idx)}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_vs_tfm_by_horizon(rows: list[dict], path: Path, title: str) -> None:
    horizons = sorted({int(row["horizon"]) for row in rows if row.get("horizon") is not None})
    if not horizons:
        return

    def _mean_metric(rows_sub, metric):
        vals = []
        for r in rows_sub:
            v = r.get(metric)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    model_values = [_mean_metric([r for r in rows if int(r["horizon"]) == h], "total_mae") for h in horizons]
    tfm_values = [_mean_metric([r for r in rows if int(r["horizon"]) == h], "tfm_zeroshot_mae") for h in horizons]
    if all(v is None for v in model_values) and all(v is None for v in tfm_values):
        return

    x = np.arange(len(horizons))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    model_plot = [np.nan if v is None else v for v in model_values]
    tfm_plot = [np.nan if v is None else v for v in tfm_values]
    ax.bar(x - width / 2, tfm_plot, width=width, label="TimesFM ZS", color="#2ca02c")
    ax.bar(x + width / 2, model_plot, width=width, label=MODEL_NAME, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Normalized MAE")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
