"""Diagnostics for Qlib Alpha158 feature matrices."""

from __future__ import annotations

import pandas as pd


def _validate_alpha_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Alpha158 feature DataFrame is empty")
    required = {"date", "asset_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Alpha158 feature DataFrame missing required columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["asset_id"] = out["asset_id"].astype(str)
    return out


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in {"date", "asset_id"}]


def summarize_alpha158_features(df: pd.DataFrame) -> pd.DataFrame:
    alpha = _validate_alpha_df(df)
    features = _feature_columns(alpha)
    duplicate_count = int(alpha.duplicated(["asset_id", "date"]).sum())
    missing = alpha[features].isna() if features else pd.DataFrame(index=alpha.index)
    return pd.DataFrame(
        [
            {
                "n_rows": len(alpha),
                "n_assets": alpha["asset_id"].nunique(),
                "first_date": alpha["date"].min(),
                "last_date": alpha["date"].max(),
                "n_feature_columns": len(features),
                "missing_rate_avg": float(missing.mean().mean()) if features else 0.0,
                "missing_rate_max": float(missing.mean().max()) if features else 0.0,
                "duplicate_asset_date_count": duplicate_count,
            }
        ]
    )


def alpha158_asset_coverage(df: pd.DataFrame) -> pd.DataFrame:
    alpha = _validate_alpha_df(df)
    features = _feature_columns(alpha)
    base = alpha.groupby("asset_id", sort=True).agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_rows=("date", "size"),
    )
    duplicate_assets = alpha.loc[alpha.duplicated(["asset_id", "date"], keep=False), "asset_id"].value_counts()
    if features:
        missing = alpha[features].isna()
        missing_any = missing.any(axis=1).groupby(alpha["asset_id"]).sum()
        missing_rate_avg = missing.groupby(alpha["asset_id"]).mean().mean(axis=1)
    else:
        missing_any = pd.Series(0, index=base.index)
        missing_rate_avg = pd.Series(0.0, index=base.index)

    out = base.reset_index()
    out["n_missing_any_feature"] = out["asset_id"].map(missing_any).fillna(0).astype(int)
    out["missing_rate_avg"] = out["asset_id"].map(missing_rate_avg).fillna(0.0).astype(float)
    out["_duplicate_count"] = out["asset_id"].map(duplicate_assets).fillna(0).astype(int)
    out["coverage_ok"] = (out["n_rows"] > 100) & (out["missing_rate_avg"] < 0.5) & (out["_duplicate_count"] == 0)
    return out.drop(columns=["_duplicate_count"])


def compare_alpha158_with_price_universe(
    alpha_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    alpha = _validate_alpha_df(alpha_df)
    if price_df.empty:
        raise ValueError("price DataFrame is empty")
    missing = {"date", "asset_id"}.difference(price_df.columns)
    if missing:
        raise ValueError(f"price DataFrame missing required columns: {sorted(missing)}")

    price = price_df.copy()
    price["date"] = pd.to_datetime(price["date"])
    price["asset_id"] = price["asset_id"].astype(str)
    price_dates_df = price[["asset_id", "date"]].drop_duplicates()
    alpha_dates_df = alpha[["asset_id", "date"]].drop_duplicates()
    price_stats = price_dates_df.groupby("asset_id").agg(
        price_first_date=("date", "min"),
        price_last_date=("date", "max"),
        price_n_dates=("date", "size"),
    )
    alpha_stats = alpha_dates_df.groupby("asset_id").agg(
        alpha_first_date=("date", "min"),
        alpha_last_date=("date", "max"),
        alpha_n_dates=("date", "size"),
    )
    overlap_counts = (
        price_dates_df.merge(alpha_dates_df, on=["asset_id", "date"], how="inner")
        .groupby("asset_id")
        .size()
    )

    rows = []
    for asset_id in sorted(set(price_stats.index) | set(alpha_stats.index)):
        in_price = asset_id in price_stats.index
        in_alpha = asset_id in alpha_stats.index
        overlap = int(overlap_counts.get(asset_id, 0))
        price_n_dates = int(price_stats.loc[asset_id, "price_n_dates"]) if in_price else 0
        rows.append(
            {
                "asset_id": asset_id,
                "in_price_data": in_price,
                "in_alpha158": in_alpha,
                "price_first_date": price_stats.loc[asset_id, "price_first_date"] if in_price else pd.NaT,
                "price_last_date": price_stats.loc[asset_id, "price_last_date"] if in_price else pd.NaT,
                "alpha_first_date": alpha_stats.loc[asset_id, "alpha_first_date"] if in_alpha else pd.NaT,
                "alpha_last_date": alpha_stats.loc[asset_id, "alpha_last_date"] if in_alpha else pd.NaT,
                "overlapping_rows": overlap,
                "alpha_coverage_ratio_on_price_dates": overlap / price_n_dates if price_n_dates else 0.0,
            }
        )
    return pd.DataFrame(rows)
