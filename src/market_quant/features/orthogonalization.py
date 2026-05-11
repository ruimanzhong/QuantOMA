"""Train-only Gram-Schmidt style feature orthogonalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrthogonalizationModel:
    alpha_columns: list[str]
    risk_columns: list[str]
    coefficients: dict[str, np.ndarray]
    risk_means: pd.Series
    risk_stds: pd.Series
    alpha_means: pd.Series
    alpha_stds: pd.Series


def _safe_standardize(frame: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    safe_stds = stds.replace([np.inf, -np.inf], np.nan).replace(0.0, np.nan).fillna(1.0)
    standardized = (frame.fillna(means).fillna(0.0) - means.fillna(0.0)) / safe_stds
    return standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)


def fit_orthogonalization_model(
    train: pd.DataFrame,
    alpha_columns: list[str],
    risk_columns: list[str],
    ridge: float = 1e-6,
) -> OrthogonalizationModel:
    """Fit train-only linear projections that remove macro beta from alpha factors."""
    alpha_columns = [col for col in alpha_columns if col in train]
    risk_columns = [col for col in risk_columns if col in train]
    if not alpha_columns or not risk_columns:
        return OrthogonalizationModel(
            alpha_columns,
            risk_columns,
            {},
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )
    risk = train[risk_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    alpha = train[alpha_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    risk_means = risk.mean()
    alpha_means = alpha.mean()
    risk_stds = risk.std(ddof=0).replace([np.inf, -np.inf], np.nan).replace(0.0, np.nan).fillna(1.0)
    alpha_stds = alpha.std(ddof=0).replace([np.inf, -np.inf], np.nan).replace(0.0, np.nan).fillna(1.0)
    risk_z = _safe_standardize(risk, risk_means, risk_stds)
    alpha_z = _safe_standardize(alpha, alpha_means, alpha_stds)
    x = risk_z.to_numpy(dtype=float)
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1])
    coefficients: dict[str, np.ndarray] = {}
    for col in alpha_columns:
        y = alpha_z[col].to_numpy(dtype=float)
        rhs = np.einsum("ij,i->j", x, y, optimize=True)
        try:
            beta = np.linalg.solve(xtx, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx) @ rhs
        coefficients[col] = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    return OrthogonalizationModel(alpha_columns, risk_columns, coefficients, risk_means, risk_stds, alpha_means, alpha_stds)


def apply_orthogonalization_model(df: pd.DataFrame, model: OrthogonalizationModel, suffix: str = "_orth") -> tuple[pd.DataFrame, list[str]]:
    """Append orthogonalized alpha residual columns using a train-fitted model."""
    if not model.alpha_columns or not model.risk_columns:
        return df.copy(), []
    base = df.copy()
    risk = base[model.risk_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    risk_z = _safe_standardize(risk, model.risk_means, model.risk_stds)
    x = risk_z.to_numpy(dtype=float)
    residual_cols = {}
    created = []
    for col in model.alpha_columns:
        if col not in base or col not in model.coefficients:
            continue
        alpha = pd.to_numeric(base[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        alpha_frame = alpha.to_frame(name=col)
        alpha_centered = _safe_standardize(alpha_frame, model.alpha_means[[col]], model.alpha_stds[[col]])[col].to_numpy(dtype=float)
        projection = np.einsum("ij,j->i", x, model.coefficients[col], optimize=True)
        residual = alpha_centered - projection
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        new_col = f"{col}{suffix}"
        residual_cols[new_col] = residual
        created.append(new_col)
    if not residual_cols:
        return base, []
    residual_frame = pd.DataFrame(residual_cols, index=base.index)
    return pd.concat([base, residual_frame], axis=1), created


def orthogonalize_training_frames(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    """Fit Gram-Schmidt residualization on train only, then apply out-of-sample."""
    cfg = config or {}
    if not cfg.get("enabled", False):
        return train, calibration, test, feature_columns, pd.DataFrame()
    alpha_prefix = cfg.get("alpha_prefix", "alpha158_")
    suffix = cfg.get("suffix", "_orth")
    alpha_columns = [col for col in feature_columns if col.startswith(alpha_prefix) and not col.endswith(suffix)]
    risk_columns = [col for col in cfg.get("risk_features", []) if col in feature_columns and col in train]
    model = fit_orthogonalization_model(train, alpha_columns, risk_columns, ridge=float(cfg.get("ridge", 1e-6)))
    train_out, created = apply_orthogonalization_model(train, model, suffix=suffix)
    calibration_out, _ = apply_orthogonalization_model(calibration, model, suffix=suffix) if not calibration.empty else (calibration.copy(), [])
    test_out, _ = apply_orthogonalization_model(test, model, suffix=suffix)
    keep_original = bool(cfg.get("keep_original", False))
    if keep_original:
        feature_out = feature_columns + [col for col in created if col not in feature_columns]
    else:
        feature_out = [col for col in feature_columns if col not in model.alpha_columns] + created
    report = pd.DataFrame(
        {
            "feature": model.alpha_columns,
            "orthogonalized_feature": [f"{col}{suffix}" for col in model.alpha_columns],
            "n_risk_features": len(model.risk_columns),
            "selection_stage": "gram_schmidt_orthogonalization",
            "selected": True,
        }
    )
    return train_out, calibration_out, test_out, feature_out, report
