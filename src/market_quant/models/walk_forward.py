"""Purged walk-forward model training for tabular research baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_name_used: str
    selected_features: pd.DataFrame


@dataclass(frozen=True)
class OrthogonalizationModel:
    alpha_columns: list[str]
    risk_columns: list[str]
    coefficients: dict[str, np.ndarray]
    risk_means: pd.Series
    risk_stds: pd.Series
    alpha_means: pd.Series
    alpha_stds: pd.Series


def make_purged_walk_forward_folds(
    data: pd.DataFrame,
    initial_train_size: int = 756,
    test_size: int = 63,
    step_size: int = 63,
    label_end_column: str = "label_end_date",
    min_train_samples: int = 300,
    min_test_samples: int = 20,
) -> list[dict[str, Any]]:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")
    df = data.sort_index()
    folds = []
    fold_id = 0
    test_start_pos = initial_train_size
    while test_start_pos < len(df):
        test_end_pos = min(test_start_pos + test_size, len(df))
        test_index = df.index[test_start_pos:test_end_pos]
        train_index = df.index[:test_start_pos]
        if label_end_column in df:
            label_end = pd.to_datetime(df.loc[train_index, label_end_column])
            train_index = train_index[label_end < test_index.min()]
        if len(train_index) >= min_train_samples and len(test_index) >= min_test_samples:
            folds.append(
                {
                    "fold_id": fold_id,
                    "train_index": train_index,
                    "test_index": test_index,
                    "train_start": train_index.min(),
                    "train_end": train_index.max(),
                    "test_start": test_index.min(),
                    "test_end": test_index.max(),
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                }
            )
        test_start_pos += step_size
        fold_id += 1
    return folds


def make_classifier(model_name: str, parameters: dict[str, Any] | None = None):
    params = parameters or {}
    try:
        if model_name == "catboost":
            try:
                from catboost import CatBoostClassifier
            except (ImportError, OSError):
                return make_classifier("hist_gradient_boosting", params)
            else:
                return CatBoostClassifier(
                    iterations=params.get("iterations", params.get("n_estimators", 300)),
                    learning_rate=params.get("learning_rate", 0.03),
                    depth=params.get("depth", params.get("max_depth", 4)),
                    l2_leaf_reg=params.get("l2_leaf_reg", 6.0),
                    loss_function=params.get("loss_function", "Logloss"),
                    eval_metric=params.get("eval_metric", "Logloss"),
                    random_seed=params.get("random_seed", params.get("random_state", 42)),
                    verbose=params.get("verbose", False),
                    allow_writing_files=params.get("allow_writing_files", False),
                )
        if model_name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except (ImportError, OSError):
                return make_classifier("hist_gradient_boosting", params)
            else:
                return LGBMClassifier(**params)
        if model_name == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                learning_rate=params.get("learning_rate", 0.05),
                max_iter=params.get("n_estimators", 200),
                random_state=params.get("random_state", 42),
            )
        if model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=params.get("random_state", 42)))
    except ImportError as exc:
        raise ImportError("scikit-learn is required for gold model training. Install with: python -m pip install scikit-learn") from exc
    raise ValueError(f"Unsupported classifier: {model_name}")


def make_regressor(model_name: str, parameters: dict[str, Any] | None = None):
    params = parameters or {}
    try:
        if model_name == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except (ImportError, OSError):
                return make_regressor("hist_gradient_boosting", params)
            else:
                return LGBMRegressor(**params)
        if model_name == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                learning_rate=params.get("learning_rate", 0.05),
                max_iter=params.get("n_estimators", 200),
                random_state=params.get("random_state", 42),
            )
        if model_name == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), Ridge(alpha=params.get("alpha", 1.0), random_state=params.get("random_state", 42)))
    except ImportError as exc:
        raise ImportError("scikit-learn is required for gold model training. Install with: python -m pip install scikit-learn") from exc
    raise ValueError(f"Unsupported regressor: {model_name}")


def _predict_probability(model, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 1:
            return np.repeat(float(proba[:, 0][0]), len(x_test))
        return proba[:, 1]
    scores = model.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-scores))


def predict_classifier_probability(model, x: pd.DataFrame) -> np.ndarray:
    """Public probability helper for live inference scripts."""
    return _predict_probability(model, x)


def fit_probability_calibrator(raw_probability: np.ndarray, y: pd.Series | np.ndarray, method: str = "platt"):
    """Fit a one-dimensional probability calibration model."""
    method = method.lower()
    x = np.asarray(raw_probability, dtype=float).reshape(-1, 1)
    target = np.asarray(y, dtype=int)
    if method in {"none", "off", ""}:
        return None
    if method in {"platt", "sigmoid"}:
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(x, target)
        return calibrator
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(x.ravel(), target)
        return calibrator
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_probability_calibrator(calibrator, raw_probability: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(raw_probability, dtype=float)
    x = np.asarray(raw_probability, dtype=float).reshape(-1, 1)
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(x)[:, 1]
    return calibrator.predict(x.ravel())


def _binary_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(y_true, dtype=float)
    return float(((p - y) ** 2).mean())


def split_train_calibration_index(train_index: pd.Index, calibration_size: int, min_fit_samples: int) -> tuple[pd.Index, pd.Index]:
    if calibration_size <= 0 or len(train_index) < min_fit_samples + calibration_size:
        return train_index, pd.Index([])
    return train_index[:-calibration_size], train_index[-calibration_size:]


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


def select_alpha158_features_by_train_ic(
    train: pd.DataFrame,
    base_features: list[str],
    target_column: str,
    alpha_prefix: str = "alpha158_",
    max_alpha_features: int = 20,
    min_abs_ic: float = 0.02,
    max_missing_rate: float = 0.30,
) -> tuple[list[str], pd.DataFrame]:
    """Select Alpha158 columns using training-only rank IC style correlation."""
    alpha_cols = [col for col in base_features if col.startswith(alpha_prefix)]
    non_alpha = [col for col in base_features if not col.startswith(alpha_prefix)]
    rows = []
    y = train[target_column].astype(float)
    for col in alpha_cols:
        x = pd.to_numeric(train[col], errors="coerce")
        missing_rate = float(x.isna().mean())
        valid = x.notna() & y.notna()
        if valid.sum() < 50 or missing_rate > max_missing_rate or x[valid].nunique() < 3:
            ic = np.nan
        else:
            ic = float(x[valid].rank().corr(y[valid].rank()))
        rows.append({"feature": col, "ic": ic, "abs_ic": abs(ic) if pd.notna(ic) else np.nan, "missing_rate": missing_rate})
    score = pd.DataFrame(rows)
    if score.empty:
        return non_alpha, score
    selected = (
        score.dropna(subset=["abs_ic"])
        .query("abs_ic >= @min_abs_ic and missing_rate <= @max_missing_rate")
        .sort_values(["abs_ic", "missing_rate"], ascending=[False, True])
        .head(max_alpha_features)
    )
    return non_alpha + selected["feature"].tolist(), score


def select_features_by_correlation_clusters(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_abs_spearman: float = 0.85,
    max_features: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Greedy hierarchical-style de-redundancy using Spearman correlation clusters.

    This is a lightweight approximation of hierarchical feature clustering:
    rank features by target rank-IC, then keep the best representative while
    rejecting later features that are too correlated with an existing cluster.
    """
    if not feature_columns:
        return [], pd.DataFrame()
    y = train[target_column].astype(float)
    usable = []
    rows = []
    for col in feature_columns:
        x = pd.to_numeric(train[col], errors="coerce")
        valid = x.notna() & y.notna()
        missing_rate = float(x.isna().mean())
        if valid.sum() < 50 or x[valid].nunique() < 2:
            ic = np.nan
        else:
            ic = float(x[valid].rank().corr(y[valid].rank()))
            usable.append(col)
        rows.append({"feature": col, "ic": ic, "abs_ic": abs(ic) if pd.notna(ic) else np.nan, "missing_rate": missing_rate})
    score = pd.DataFrame(rows)
    if not usable:
        return feature_columns, score

    ranked = score.dropna(subset=["abs_ic"]).sort_values(["abs_ic", "missing_rate"], ascending=[False, True])["feature"].tolist()
    rank_df = train[ranked].apply(pd.to_numeric, errors="coerce").rank()
    selected: list[str] = []
    cluster_id = 0
    cluster_rows = []
    for feature in ranked:
        if max_features is not None and len(selected) >= max_features:
            break
        if not selected:
            selected.append(feature)
            cluster_rows.append({"feature": feature, "cluster_id": cluster_id, "selected": True, "max_abs_corr_to_selected": 0.0})
            cluster_id += 1
            continue
        corr = rank_df[selected].corrwith(rank_df[feature]).abs().max()
        max_corr = float(corr) if pd.notna(corr) else 0.0
        if max_corr < max_abs_spearman:
            selected.append(feature)
            cluster_rows.append({"feature": feature, "cluster_id": cluster_id, "selected": True, "max_abs_corr_to_selected": max_corr})
            cluster_id += 1
        else:
            cluster_rows.append({"feature": feature, "cluster_id": None, "selected": False, "max_abs_corr_to_selected": max_corr})
    cluster_report = score.merge(pd.DataFrame(cluster_rows), on="feature", how="left")
    cluster_report["selected"] = cluster_report["selected"].where(cluster_report["selected"].notna(), False).astype(bool)
    return selected, cluster_report


def select_features_for_training(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    selection_cfg: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    selected_features = feature_columns
    reports = []
    if selection_cfg.get("enabled", False):
        selected_features, alpha_score = select_alpha158_features_by_train_ic(
            train,
            feature_columns,
            target_column,
            alpha_prefix=selection_cfg.get("alpha_prefix", "alpha158_"),
            max_alpha_features=int(selection_cfg.get("max_alpha_features", 20)),
            min_abs_ic=float(selection_cfg.get("min_abs_ic", 0.02)),
            max_missing_rate=float(selection_cfg.get("max_missing_rate", 0.30)),
        )
        if not alpha_score.empty:
            alpha_score["selection_stage"] = "alpha_rank_ic"
            reports.append(alpha_score)
    cluster_cfg = selection_cfg.get("correlation_clustering", {})
    if cluster_cfg.get("enabled", False):
        selected_features, cluster_score = select_features_by_correlation_clusters(
            train,
            selected_features,
            target_column,
            max_abs_spearman=float(cluster_cfg.get("max_abs_spearman", 0.90)),
            max_features=cluster_cfg.get("max_features"),
        )
        if not cluster_score.empty:
            cluster_score["selection_stage"] = "correlation_cluster"
            reports.append(cluster_score)
    report = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return selected_features, report


def run_walk_forward_classifier(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str = "lightgbm",
    model_parameters: dict[str, Any] | None = None,
    fold_config: dict[str, Any] | None = None,
    feature_selection_config: dict[str, Any] | None = None,
) -> WalkForwardResult:
    cfg = fold_config or {}
    folds = make_purged_walk_forward_folds(
        data,
        initial_train_size=int(cfg.get("initial_train_size", 756)),
        test_size=int(cfg.get("test_size", 63)),
        step_size=int(cfg.get("step_size", 63)),
        min_train_samples=int(cfg.get("min_train_samples", 300)),
        min_test_samples=int(cfg.get("min_test_samples", 20)),
    )
    if not folds:
        raise ValueError("No walk-forward folds generated. Check dataset length and fold configuration.")

    pred_frames = []
    metric_rows = []
    selected_rows = []
    model_name_used = model_name
    selection_cfg = feature_selection_config or {}
    orthogonalization_cfg = selection_cfg.get("orthogonalization", {})
    calibration_cfg = (fold_config or {}).get("calibration", {})
    calibration_method = calibration_cfg.get("method", "none")
    calibration_size = int(calibration_cfg.get("size", 0))
    min_fit_samples = int(calibration_cfg.get("min_fit_samples", 300))
    for fold in folds:
        fit_index, calibration_index = split_train_calibration_index(fold["train_index"], calibration_size, min_fit_samples)
        train = data.loc[fit_index]
        calibration = data.loc[calibration_index] if len(calibration_index) else pd.DataFrame()
        test = data.loc[fold["test_index"]]
        train_model, calibration_model, test_model, model_feature_columns, orth_report = orthogonalize_training_frames(
            train,
            calibration,
            test,
            feature_columns,
            orthogonalization_cfg,
        )
        selected_features = model_feature_columns
        score = pd.DataFrame()
        if selection_cfg.get("enabled", False) or selection_cfg.get("correlation_clustering", {}).get("enabled", False):
            selected_features, score = select_features_for_training(train_model, model_feature_columns, target_column, selection_cfg)
            report_parts = [df for df in [orth_report, score] if df is not None and not df.empty]
            full_score = pd.concat(report_parts, ignore_index=True, sort=False) if report_parts else pd.DataFrame()
            if not full_score.empty:
                top = full_score.sort_values("abs_ic", ascending=False, na_position="last").head(50).copy()
                top.insert(0, "fold_id", fold["fold_id"])
                top["selected"] = top["feature"].isin(selected_features)
                selected_rows.append(top)
        elif not orth_report.empty:
            top = orth_report.head(50).copy()
            top.insert(0, "fold_id", fold["fold_id"])
            selected_rows.append(top)
        x_train = train_model[selected_features].astype(float)
        y_train = train_model[target_column].astype(int)
        x_test = test_model[selected_features].astype(float)
        y_test = test_model[target_column].astype(int)
        model = make_classifier(model_name, model_parameters)
        model_name_used = type(model).__name__
        model.fit(x_train, y_train)
        raw_probability = _predict_probability(model, x_test)
        probability = raw_probability
        calibration_used = "none"
        if calibration_method not in {"none", "off", "", None} and not calibration.empty and calibration[target_column].nunique() > 1:
            x_calibration = calibration_model[selected_features].astype(float)
            y_calibration = calibration_model[target_column].astype(int)
            calibration_raw = _predict_probability(model, x_calibration)
            calibrator = fit_probability_calibrator(calibration_raw, y_calibration, str(calibration_method))
            probability = apply_probability_calibrator(calibrator, raw_probability)
            calibration_used = str(calibration_method)
        predicted = (probability >= 0.5).astype(int)
        pred_frames.append(
            pd.DataFrame(
                {
                    "date": test.index,
                    "fold_id": fold["fold_id"],
                    "raw_probability": raw_probability,
                    "probability": probability,
                    "prediction": predicted,
                    "target": y_test.to_numpy(),
                    "primary_etf": test["primary_etf"].to_numpy(),
                }
            )
        )
        metric_rows.append(
            {
                "fold_id": fold["fold_id"],
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "n_train": len(train),
                "n_test": len(test),
                "accuracy": float((predicted == y_test.to_numpy()).mean()),
                "brier_score": _brier_score(y_test.to_numpy(), probability),
                "log_loss": _binary_log_loss(y_test.to_numpy(), probability),
                "raw_brier_score": _brier_score(y_test.to_numpy(), raw_probability),
                "raw_log_loss": _binary_log_loss(y_test.to_numpy(), raw_probability),
                "positive_rate": float(predicted.mean()),
                "target_positive_rate": float(y_test.mean()),
                "n_features": len(selected_features),
                "n_alpha158_features": sum(col.startswith(selection_cfg.get("alpha_prefix", "alpha158_")) for col in selected_features),
                "calibration_method": calibration_used,
                "n_calibration": int(len(calibration)),
            }
        )
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return WalkForwardResult(pd.concat(pred_frames, ignore_index=True), pd.DataFrame(metric_rows), model_name_used, selected)


def run_walk_forward_regressor(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str = "lightgbm",
    model_parameters: dict[str, Any] | None = None,
    fold_config: dict[str, Any] | None = None,
) -> WalkForwardResult:
    cfg = fold_config or {}
    folds = make_purged_walk_forward_folds(
        data,
        initial_train_size=int(cfg.get("initial_train_size", 756)),
        test_size=int(cfg.get("test_size", 63)),
        step_size=int(cfg.get("step_size", 63)),
        min_train_samples=int(cfg.get("min_train_samples", 300)),
        min_test_samples=int(cfg.get("min_test_samples", 20)),
    )
    if not folds:
        raise ValueError("No walk-forward folds generated. Check dataset length and fold configuration.")

    pred_frames = []
    metric_rows = []
    model_name_used = model_name
    for fold in folds:
        train = data.loc[fold["train_index"]]
        test = data.loc[fold["test_index"]]
        x_train = train[feature_columns].astype(float)
        y_train = train[target_column].astype(float)
        x_test = test[feature_columns].astype(float)
        y_test = test[target_column].astype(float)
        model = make_regressor(model_name, model_parameters)
        model_name_used = type(model).__name__
        model.fit(x_train, y_train)
        predicted = pd.Series(model.predict(x_test), index=test.index).clip(0.0, 1.0)
        error = predicted.to_numpy() - y_test.to_numpy()
        pred_frames.append(
            pd.DataFrame(
                {
                    "date": test.index,
                    "fold_id": fold["fold_id"],
                    "position": predicted.to_numpy(),
                    "target_position": y_test.to_numpy(),
                    "target_forward_return": test["target_forward_return"].to_numpy() if "target_forward_return" in test else np.nan,
                    "primary_etf": test["primary_etf"].to_numpy(),
                }
            )
        )
        metric_rows.append(
            {
                "fold_id": fold["fold_id"],
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "n_train": len(train),
                "n_test": len(test),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "target_mean_position": float(y_test.mean()),
                "predicted_mean_position": float(predicted.mean()),
                "n_features": len(feature_columns),
            }
        )
    return WalkForwardResult(pd.concat(pred_frames, ignore_index=True), pd.DataFrame(metric_rows), model_name_used, pd.DataFrame())
