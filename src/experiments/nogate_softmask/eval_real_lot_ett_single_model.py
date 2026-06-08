"""Utilities for eval_real.py — single-model evaluation on LOTSA + ETT datasets."""
from __future__ import annotations

from typing import Any

import numpy as np


def target_values(row: dict[str, Any], variate_idx: int = 0) -> np.ndarray:
    target = row["target"]
    if len(target) == 0:
        return np.asarray([], dtype=np.float32)
    first = target[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        if len(target) <= variate_idx:
            raise ValueError(f"target has no variate index {variate_idx}")
        values = target[variate_idx]
    else:
        values = target
    return np.asarray(values, dtype=np.float32)
