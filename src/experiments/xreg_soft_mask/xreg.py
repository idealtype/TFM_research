from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TOL = 1e-6


@dataclass(frozen=True)
class XRegResult:
    forecast: np.ndarray
    fitted_context: np.ndarray
    n_features: int
    train_len: int
    status: str


def _normalize_target(y: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std <= _TOL:
        std = 1.0
    return (y - mean) / std, (mean, std)


def _renormalize(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    mean, std = stats
    return values * std + mean


def _feature_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std > _TOL, std, 1.0)
    return mean, std


def _apply_feature_stats(x: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = stats
    return (x - mean) / std


def fit_linear_xreg(
    target_context: np.ndarray,
    covariates_context: np.ndarray,
    covariates_future: np.ndarray,
    *,
    ridge: float = 0.0,
    normalize_target: bool = True,
    use_intercept: bool = True,
) -> XRegResult:
    """Fit one in-context linear XReg model and forecast the horizon.

    This mirrors the TimesFM XReg core policy for numeric covariates:
    normalize covariate columns on the context, optionally normalize the target
    per input, add an intercept, then solve OLS/ridge in closed form.
    """
    y = np.asarray(target_context, dtype=np.float64)
    x_ctx = np.asarray(covariates_context, dtype=np.float64)
    x_fut = np.asarray(covariates_future, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError(f"target_context must be 1D, got {y.shape}")
    if x_ctx.ndim != 2 or x_fut.ndim != 2:
        raise ValueError(f"covariates must be 2D, got {x_ctx.shape=} {x_fut.shape=}")
    if len(y) != x_ctx.shape[0]:
        raise ValueError(f"target/covariate context length mismatch: {len(y)} vs {x_ctx.shape[0]}")
    if x_ctx.shape[1] != x_fut.shape[1]:
        raise ValueError(f"covariate feature mismatch: {x_ctx.shape[1]} vs {x_fut.shape[1]}")
    if x_ctx.shape[1] == 0:
        return XRegResult(
            forecast=np.zeros(x_fut.shape[0], dtype=np.float32),
            fitted_context=np.zeros(x_ctx.shape[0], dtype=np.float32),
            n_features=0,
            train_len=int(len(y)),
            status="no_covariates",
        )

    finite_train = np.isfinite(y) & np.isfinite(x_ctx).all(axis=1)
    if int(finite_train.sum()) <= x_ctx.shape[1] + int(use_intercept):
        return XRegResult(
            forecast=np.zeros(x_fut.shape[0], dtype=np.float32),
            fitted_context=np.zeros(x_ctx.shape[0], dtype=np.float32),
            n_features=int(x_ctx.shape[1]),
            train_len=int(finite_train.sum()),
            status="insufficient_rows",
        )

    y_fit = y[finite_train]
    x_fit = x_ctx[finite_train]
    feature_stats = _feature_stats(x_fit)
    x_ctx_norm = _apply_feature_stats(x_fit, feature_stats)
    x_ctx_all_norm = _apply_feature_stats(x_ctx, feature_stats)
    x_fut_norm = _apply_feature_stats(x_fut, feature_stats)

    stats = (0.0, 1.0)
    if normalize_target:
        y_fit, stats = _normalize_target(y_fit)

    if use_intercept:
        x_ctx_norm = np.pad(x_ctx_norm, ((0, 0), (1, 0)), constant_values=1.0)
        x_ctx_all_norm = np.pad(x_ctx_all_norm, ((0, 0), (1, 0)), constant_values=1.0)
        x_fut_norm = np.pad(x_fut_norm, ((0, 0), (1, 0)), constant_values=1.0)

    penalty = float(ridge) * np.eye(x_ctx_norm.shape[1], dtype=np.float64)
    beta = np.linalg.pinv(x_ctx_norm.T @ x_ctx_norm + penalty, hermitian=True) @ x_ctx_norm.T @ y_fit
    forecast = x_fut_norm @ beta
    fitted_context = x_ctx_all_norm @ beta

    if normalize_target:
        forecast = _renormalize(forecast, stats)
        fitted_context = _renormalize(fitted_context, stats)

    return XRegResult(
        forecast=forecast.astype(np.float32, copy=False),
        fitted_context=fitted_context.astype(np.float32, copy=False),
        n_features=int(x_ctx.shape[1]),
        train_len=int(finite_train.sum()),
        status="ok",
    )
