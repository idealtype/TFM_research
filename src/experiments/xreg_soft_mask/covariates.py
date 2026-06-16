from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class CovariateWindow:
    target_context: np.ndarray
    covariates_context: np.ndarray
    covariates_future: np.ndarray
    covariate_names: list[str]
    target_name: str


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c != "date" and pd.api.types.is_numeric_dtype(df[c])
    ]


def _as_col_ids(raw_value, fallback: list[str]) -> list[str]:
    if raw_value is None:
        return fallback
    if torch.is_tensor(raw_value):
        return [str(x) for x in raw_value.cpu().tolist()]
    return [str(x) for x in raw_value]


def raw_dataframe_covariate_window(
    *,
    raw_df: pd.DataFrame,
    backbone: dict,
    local_idx: int,
    context_len: int,
    horizon: int,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> CovariateWindow | None:
    """Extract target and same-dataset numeric covariates from a wide raw parquet.

    The selected target column follows the cache's ``col_ids`` mapping when
    present. Candidate covariates are all other numeric columns unless an
    explicit include/exclude list is provided.
    """
    numeric_cols = _numeric_columns(raw_df)
    if not numeric_cols:
        return None

    col_ids = _as_col_ids(backbone.get("col_ids"), numeric_cols)
    target_col = col_ids[int(local_idx)] if int(local_idx) < len(col_ids) else numeric_cols[int(local_idx) % len(numeric_cols)]
    if target_col not in raw_df.columns:
        target_col = numeric_cols[int(local_idx) % len(numeric_cols)]

    win_starts = backbone.get("win_starts")
    if win_starts is None:
        return None
    start = int(win_starts[int(local_idx)])
    stop = start + int(context_len) + int(horizon)
    if stop > len(raw_df):
        return None

    if include_columns:
        cov_cols = [c for c in include_columns if c in raw_df.columns and c != target_col]
    else:
        cov_cols = [c for c in numeric_cols if c != target_col]
    if exclude_columns:
        excluded = set(exclude_columns)
        cov_cols = [c for c in cov_cols if c not in excluded]
    if not cov_cols:
        return None

    target = raw_df[target_col].iloc[start: start + context_len].to_numpy(dtype=np.float32)
    cov = raw_df[cov_cols].iloc[start: stop].to_numpy(dtype=np.float32)
    if len(target) != context_len or cov.shape != (context_len + horizon, len(cov_cols)):
        return None
    if not np.isfinite(target).all():
        return None

    return CovariateWindow(
        target_context=target,
        covariates_context=cov[:context_len],
        covariates_future=cov[context_len:],
        covariate_names=list(cov_cols),
        target_name=str(target_col),
    )


def batch_raw_dataframe_covariates(
    *,
    raw_df: pd.DataFrame | None,
    backbone: dict,
    local_indices: torch.Tensor,
    context_len: int,
    horizon: int,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> list[CovariateWindow | None]:
    if raw_df is None:
        return [None for _ in local_indices.tolist()]
    return [
        raw_dataframe_covariate_window(
            raw_df=raw_df,
            backbone=backbone,
            local_idx=int(local_idx),
            context_len=int(context_len),
            horizon=int(horizon),
            include_columns=include_columns,
            exclude_columns=exclude_columns,
        )
        for local_idx in local_indices.tolist()
    ]


def parse_column_list(value: str | None) -> list[str] | None:
    if value is None or str(value).strip().lower() in {"", "none"}:
        return None
    path = Path(value)
    if path.exists():
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [item.strip() for item in str(value).split(",") if item.strip()]
