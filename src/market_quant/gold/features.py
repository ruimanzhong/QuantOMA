"""Gold-specific feature engineering and dataset assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TROY_OUNCE_GRAMS = 31.1034768


@dataclass(frozen=True)
class GoldDatasetBundle:
    data: pd.DataFrame
    feature_columns: list[str]
    target_column: str


@dataclass(frozen=True)
class GoldFeatureFrame:
    data: pd.DataFrame
    feature_columns: list[str]


def daily_series_to_wide(series_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "series_id", "value"}
    missing = required.difference(series_df.columns)
    if missing:
        raise ValueError(f"gold daily series missing columns: {sorted(missing)}")
    df = series_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="series_id", values="value").sort_index()


def append_preopen_primary_row(
    daily_series: pd.DataFrame,
    primary_series_id: str,
    signal_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Add a pre-open decision row using the last known primary ETF close.

    This is for cases such as China holidays: external markets trade during the
    break, but the China ETF has no close yet on the re-open date. The synthetic
    row makes those external moves available for a pre-open signal without
    inventing a new ETF return.
    """
    required = {"date", "series_id", "value"}
    missing = required.difference(daily_series.columns)
    if missing:
        raise ValueError(f"daily series missing columns: {sorted(missing)}")

    signal_ts = pd.Timestamp(signal_date).normalize()
    out = daily_series.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    primary = out[out["series_id"].eq(primary_series_id)].dropna(subset=["value"]).sort_values("date")
    if primary.empty:
        raise ValueError(f"primary series has no observed values: {primary_series_id}")
    if ((primary["date"] == signal_ts)).any():
        return out
    previous = primary[primary["date"] < signal_ts].tail(1)
    if previous.empty:
        raise ValueError(f"primary series has no value before signal_date={signal_ts.date()}: {primary_series_id}")

    template = {col: np.nan for col in out.columns}
    template.update(
        {
            "date": signal_ts,
            "series_id": primary_series_id,
            "value": float(previous["value"].iloc[0]),
            "source": "synthetic_preopen_last_close",
        }
    )
    return pd.concat([out, pd.DataFrame([template])], ignore_index=True)


def add_price_features(close: pd.Series, prefix: str, windows: list[int], volatility_windows: list[int]) -> pd.DataFrame:
    px = close.astype(float).sort_index()
    out = pd.DataFrame(index=px.index)
    returns_1d = px.pct_change(fill_method=None)
    for window in windows:
        out[f"{prefix}_return_{window}d"] = px.pct_change(window, fill_method=None)
        out[f"{prefix}_ma_distance_{window}d"] = px / px.rolling(window).mean() - 1
        out[f"{prefix}_momentum_{window}d"] = px.diff(window)
    for window in volatility_windows:
        out[f"{prefix}_volatility_{window}d"] = returns_1d.rolling(window).std(ddof=0)
    out[f"{prefix}_drawdown"] = px / px.cummax() - 1
    return out


def build_macro_features(wide: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=wide.index)
    for col in ["dxy", "sp500", "oil"]:
        if col in wide:
            series = wide[col].astype(float)
            out[f"{col}_return"] = series.pct_change(fill_method=None)
            out[f"{col}_return_5d"] = series.pct_change(5, fill_method=None)
            out[f"{col}_return_20d"] = series.pct_change(20, fill_method=None)
            out[f"{col}_trend_20d"] = series / series.rolling(20).mean() - 1
    if "vix" in wide:
        vix = wide["vix"].astype(float)
        out["vix_level"] = vix
        out["vix_change"] = vix.diff()
        out["vix_change_5d"] = vix.diff(5)
        out["vix_trend_20d"] = vix / vix.rolling(20).mean() - 1
    if "treasury_10y" in wide:
        out["yield_10y_change"] = wide["treasury_10y"].astype(float).diff()
    if "tips_10y" in wide:
        tips = wide["tips_10y"].astype(float)
        out["tips_10y_change"] = tips.diff()
        out["real_yield_10y_level"] = tips
        out["real_yield_10y_change"] = tips.diff()
        out["real_yield_trend_20d"] = tips - tips.rolling(20).mean()
    if "treasury_10y" in wide and "tips_10y" in wide:
        breakeven = wide["treasury_10y"].astype(float) - wide["tips_10y"].astype(float)
        out["breakeven_10y_level"] = breakeven
        out["breakeven_10y_change"] = breakeven.diff()
        out["breakeven_trend_20d"] = breakeven - breakeven.rolling(20).mean()
    return out


def build_fx_features(wide: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=wide.index)
    if "usd_cny_close" in wide:
        out["usd_cny_return"] = wide["usd_cny_close"].astype(float).pct_change(fill_method=None)
    if "usd_cnh_close" in wide:
        out["usd_cnh_return"] = wide["usd_cnh_close"].astype(float).pct_change(fill_method=None)
    fx_col = "usd_cny_close" if "usd_cny_close" in wide else "usd_cnh_close"
    if "xauusd_close" in wide and fx_col in wide:
        out["implied_gold_cny_per_g"] = wide["xauusd_close"].astype(float) * wide[fx_col].astype(float) / TROY_OUNCE_GRAMS
        out["implied_gold_cny_return"] = out["implied_gold_cny_per_g"].pct_change(fill_method=None)
    for window in windows:
        if "usd_cny_return" in out:
            out[f"usd_cny_volatility_{window}d"] = out["usd_cny_return"].rolling(window).std(ddof=0)
        if "usd_cnh_return" in out:
            out[f"usd_cnh_volatility_{window}d"] = out["usd_cnh_return"].rolling(window).std(ddof=0)
    return out


def build_china_gold_features(wide: pd.DataFrame, primary_col: str = "518880_close") -> pd.DataFrame:
    out = pd.DataFrame(index=wide.index)
    if primary_col not in wide:
        return out
    etf = wide[primary_col].astype(float)
    out["primary_etf"] = etf
    out["china_gold_etf_return"] = etf.pct_change(fill_method=None)
    out["tracking_error_proxy"] = out["china_gold_etf_return"].rolling(20).std(ddof=0)
    if "implied_gold_cny_per_g" in wide:
        implied_return = wide["implied_gold_cny_per_g"].astype(float).pct_change(fill_method=None)
        out["residual_return"] = out["china_gold_etf_return"] - implied_return
    if "au9999_sge_gold_cny_per_g" in wide and "implied_gold_cny_per_g" in wide:
        out["shanghai_gold_premium"] = wide["au9999_sge_gold_cny_per_g"].astype(float) / wide["implied_gold_cny_per_g"].astype(float) - 1
    volume_col = primary_col.replace("_close", "_volume")
    if volume_col in wide:
        out["china_gold_etf_volume"] = wide[volume_col].astype(float)
        out["china_gold_etf_turnover_proxy"] = out["china_gold_etf_volume"] * etf
    return out


def build_external_gap_features(
    wide: pd.DataFrame,
    primary_col: str = "518880_close",
    external_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Capture external market moves accumulated while the primary ETF is shut.

    For a China gold ETF trading date T, the feature compares the latest
    external observation before T with the external observation on the previous
    China ETF trading date. This keeps May Day / Golden Week price discovery
    available on the re-open row without forward-filling the ETF itself.
    """
    if primary_col not in wide:
        return pd.DataFrame(index=wide.index)

    external_cols = external_cols or [
        "xauusd_close",
        "implied_gold_cny_per_g",
        "dxy",
        "usd_cny_close",
        "usd_cnh_close",
        "tips_10y",
        "treasury_10y",
        "vix",
    ]
    external_cols = [col for col in external_cols if col in wide]
    if not external_cols:
        return pd.DataFrame(index=wide.index)

    primary_dates = pd.DatetimeIndex(wide.index[wide[primary_col].notna()])
    out = pd.DataFrame(index=wide.index)
    out["primary_trading_gap_calendar_days"] = np.nan
    out["primary_trading_gap_weekdays"] = np.nan
    out["external_gap_observation_count"] = np.nan

    for col in external_cols:
        feature_prefix = col.replace("_close", "")
        out[f"external_gap_{feature_prefix}_return"] = np.nan
        out[f"external_gap_{feature_prefix}_change"] = np.nan

    for idx in range(1, len(primary_dates)):
        prev_date = primary_dates[idx - 1]
        date = primary_dates[idx]
        external_window_index = wide.loc[(wide.index > prev_date) & (wide.index < date), external_cols].dropna(how="all").index
        out.at[date, "primary_trading_gap_calendar_days"] = float((date - prev_date).days)
        out.at[date, "primary_trading_gap_weekdays"] = float(len(pd.bdate_range(prev_date, date)) - 1)
        out.at[date, "external_gap_observation_count"] = float(len(external_window_index))

        if len(external_window_index) == 0:
            continue
        latest_external_date = external_window_index.max()
        for col in external_cols:
            start_value = wide.at[prev_date, col] if prev_date in wide.index else np.nan
            end_value = wide.at[latest_external_date, col]
            if pd.isna(start_value) or pd.isna(end_value):
                continue
            feature_prefix = col.replace("_close", "")
            if start_value != 0:
                out.at[date, f"external_gap_{feature_prefix}_return"] = float(end_value / start_value - 1)
            out.at[date, f"external_gap_{feature_prefix}_change"] = float(end_value - start_value)

    return out


def align_to_primary_trading_calendar(df: pd.DataFrame, primary_col: str) -> pd.DataFrame:
    """Keep only rows where the tradable gold ETF has an observed close."""
    if primary_col not in df.columns:
        raise ValueError(f"primary gold ETF series missing: {primary_col}")
    aligned = df[df[primary_col].notna()].copy()
    if aligned.empty:
        raise ValueError(f"primary gold ETF series has no tradable rows: {primary_col}")
    return aligned


def load_optional_feature_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    target = Path(path).expanduser()
    if not target.exists():
        return pd.DataFrame()
    df = pd.read_csv(target)
    if "date" not in df.columns:
        raise ValueError(f"optional feature CSV missing date column: {target}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_alpha158_features(path: str | Path | None, asset_id: str) -> pd.DataFrame:
    """Load Alpha158 features for one gold ETF asset and prefix columns."""
    if path is None:
        return pd.DataFrame()
    target = Path(path).expanduser()
    if not target.exists():
        return pd.DataFrame()
    if target.suffix.lower() == ".parquet":
        alpha = pd.read_parquet(target)
    else:
        alpha = pd.read_csv(target)
    required = {"date", "asset_id"}
    missing = required.difference(alpha.columns)
    if missing:
        raise ValueError(f"Alpha158 feature file missing columns: {sorted(missing)}")
    alpha = alpha.copy()
    alpha["date"] = pd.to_datetime(alpha["date"])
    alpha["asset_id"] = alpha["asset_id"].astype(str)
    part = alpha[alpha["asset_id"] == str(asset_id)].copy()
    if part.empty:
        raise ValueError(f"Alpha158 feature file has no rows for asset_id={asset_id}")
    feature_cols = [col for col in part.columns if col not in {"date", "asset_id"}]
    part = part[["date", *feature_cols]].set_index("date").sort_index()
    part = part.apply(pd.to_numeric, errors="coerce")
    part.columns = [f"alpha158_{col}" for col in part.columns]
    return part


def build_news_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    required = {"published_at", "gold_direction", "confidence", "surprise"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"news events missing columns: {sorted(missing)}")
    df = events.copy()
    df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()
    signed = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0) * pd.to_numeric(df["surprise"], errors="coerce").fillna(0.0)
    df["bullish"] = signed.where(df["gold_direction"].astype(str).str.lower() == "bullish", 0.0)
    df["bearish"] = signed.where(df["gold_direction"].astype(str).str.lower() == "bearish", 0.0)
    df["event_type"] = df["event_type"].astype(str).str.lower() if "event_type" in df else "other"
    df["affected_channels"] = df["affected_channels"].astype(str) if "affected_channels" in df else ""
    daily = pd.DataFrame(index=pd.DatetimeIndex(sorted(df["date"].unique())))
    daily["news_bullish_score"] = df.groupby("date")["bullish"].sum()
    daily["news_bearish_score"] = df.groupby("date")["bearish"].sum()
    daily["fed_dovish_score"] = df[df["event_type"].eq("fed") & df["gold_direction"].astype(str).str.lower().eq("bullish")].groupby("date")["bullish"].sum()
    daily["geopolitical_risk_score"] = df[df["event_type"].eq("geopolitics")].groupby("date")["bullish"].sum()
    daily["fiscal_risk_score"] = df[df["event_type"].eq("fiscal_risk")].groupby("date")["bullish"].sum()
    daily["usd_pressure_score"] = df[df["affected_channels"].str.contains("usd", case=False, na=False)].groupby("date")["bearish"].sum()
    daily["inflation_risk_score"] = df[df["event_type"].eq("inflation")].groupby("date")["bullish"].sum()
    daily["central_bank_buying_news_score"] = df[df["event_type"].eq("central_bank_buying")].groupby("date")["bullish"].sum()
    liquidity_mask = df["event_type"].eq("market_stress") | df["affected_channels"].str.contains("liquidity|stress|funding", case=False, regex=True, na=False)
    daily["liquidity_stress_news_score"] = df[liquidity_mask].groupby("date")["bullish"].sum()
    daily = daily.fillna(0.0)
    for window in [1, 3, 5]:
        rolled = daily.rolling(window).sum()
        rolled.columns = [f"{col}_sum_{window}d" for col in daily.columns]
        daily = daily.join(rolled)
    for half_life in [3, 5]:
        ewm = daily[[
            "news_bullish_score",
            "news_bearish_score",
            "fed_dovish_score",
            "geopolitical_risk_score",
            "usd_pressure_score",
            "liquidity_stress_news_score",
        ]].ewm(halflife=half_life, adjust=False).mean()
        ewm.columns = [f"{col}_ewm_hl_{half_life}d" for col in ewm.columns]
        daily = daily.join(ewm)
    return daily


def build_polymarket_features(markets: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        return pd.DataFrame()
    required = {"timestamp", "market_group", "probability"}
    missing = required.difference(markets.columns)
    if missing:
        raise ValueError(f"polymarket data missing columns: {sorted(missing)}")
    df = markets.copy()
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    frames = []
    for group, part in df.groupby("market_group"):
        prob = pd.to_numeric(part["probability"], errors="coerce")
        part = part.assign(probability=prob).dropna(subset=["probability"])
        daily_prob = part.groupby("date")["probability"].mean()
        out = pd.DataFrame(index=daily_prob.index)
        out[f"{group}_probability_level"] = daily_prob
        for window in [1, 3, 5]:
            out[f"{group}_probability_change_{window}d"] = daily_prob.diff(window)
        frames.append(out)
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()


def add_fixed_horizon_target(
    df: pd.DataFrame,
    price_col: str,
    horizon_days: int,
    threshold: float,
    full_position_return: float = 0.02,
) -> pd.DataFrame:
    out = df.copy()
    forward_return = out[price_col].shift(-horizon_days) / out[price_col] - 1
    out[f"{price_col}_forward_return_{horizon_days}d"] = forward_return
    out["target_forward_return"] = forward_return
    out["target_direction"] = (forward_return > threshold).astype(float)
    out.loc[forward_return.isna(), "target_direction"] = np.nan
    scale = max(full_position_return - threshold, 1e-9)
    out["target_position"] = ((forward_return - threshold) / scale).clip(0.0, 1.0)
    out.loc[forward_return.isna(), "target_position"] = np.nan
    out["label_end_date"] = pd.NaT
    if len(out) > horizon_days:
        out.iloc[:-horizon_days, out.columns.get_loc("label_end_date")] = out.index[horizon_days:]
    return out


def apply_feature_availability_lag(df: pd.DataFrame, feature_columns: list[str], lag_rows: int) -> pd.DataFrame:
    out = df.copy()
    if lag_rows <= 0:
        return out
    out[feature_columns] = out[feature_columns].shift(lag_rows)
    return out


def apply_selective_feature_availability_lag(
    df: pd.DataFrame,
    feature_columns: list[str],
    lag_rows: int,
    no_lag_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Lag regular close-based features while preserving pre-open gap features."""
    no_lag_prefixes = no_lag_prefixes or []
    lagged_columns = [
        col
        for col in feature_columns
        if not any(col.startswith(prefix) for prefix in no_lag_prefixes)
    ]
    return apply_feature_availability_lag(df, lagged_columns, lag_rows)


def fill_model_feature_nans(model_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = model_df.copy()
    for col in feature_columns:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
                out[col] = out[col].fillna(0.0)
    return out


def build_gold_feature_frame(
    daily_series: pd.DataFrame,
    config: dict[str, Any] | None = None,
    news_events: pd.DataFrame | None = None,
    polymarket_data: pd.DataFrame | None = None,
    alpha158_features: pd.DataFrame | None = None,
) -> GoldFeatureFrame:
    """Build model-ready gold features without requiring future labels."""
    cfg = config or {}
    windows = cfg.get("price_windows", [1, 5, 10, 20, 60])
    volatility_windows = cfg.get("volatility_windows", [5, 20, 60])
    fx_windows = cfg.get("fx_windows", [1, 5, 20])
    primary_col = cfg.get("primary_etf_series", "518880_close")

    wide = daily_series_to_wide(daily_series)
    frames = [wide]
    if "xauusd_close" in wide:
        frames.append(add_price_features(wide["xauusd_close"], "xauusd", windows, volatility_windows))
    macro_features = build_macro_features(wide)
    fx_features = build_fx_features(wide, fx_windows)
    wide_with_fx = wide.join(fx_features[["implied_gold_cny_per_g"]] if "implied_gold_cny_per_g" in fx_features else pd.DataFrame(index=wide.index))
    china_features = build_china_gold_features(wide_with_fx, primary_col=primary_col)
    gap_features = build_external_gap_features(wide_with_fx, primary_col=primary_col, external_cols=cfg.get("external_gap_columns"))
    frames.extend([macro_features, fx_features, china_features, gap_features])
    if news_events is not None and not news_events.empty:
        frames.append(build_news_features(news_events))
    if polymarket_data is not None and not polymarket_data.empty:
        frames.append(build_polymarket_features(polymarket_data))
    if cfg.get("include_alpha158", False) and alpha158_features is not None and not alpha158_features.empty:
        frames.append(alpha158_features)

    df = pd.concat(frames, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    prealign_live_overlay_prefixes = tuple(
        cfg.get(
            "live_overlay_feature_prefixes",
            [
                "news_",
                "fed_dovish",
                "geopolitical_risk_score",
                "fiscal_risk_score",
                "usd_pressure",
                "inflation_risk",
                "central_bank_buying_news",
                "liquidity_stress_news",
            ],
        )
    )
    prealign_live_overlay_probability_patterns = tuple(cfg.get("live_overlay_probability_patterns", ["_probability_level", "_probability_change_"]))
    prealign_live_overlay_cols = [
        col
        for col in df.columns
        if col.startswith(prealign_live_overlay_prefixes)
        or any(pattern in col for pattern in prealign_live_overlay_probability_patterns)
    ]
    if prealign_live_overlay_cols:
        df[prealign_live_overlay_cols] = df[prealign_live_overlay_cols].ffill()
    df = align_to_primary_trading_calendar(df, primary_col)
    if "primary_etf" not in df:
        if primary_col not in df:
            raise ValueError(f"primary gold ETF series missing: {primary_col}")
        df["primary_etf"] = df[primary_col]

    horizon_days = int(cfg.get("horizon_days", 5))
    exclude = set(cfg.get("exclude_feature_columns", [])) | {
        "target_direction",
        "target_position",
        "target_forward_return",
        "label_end_date",
        f"primary_etf_forward_return_{horizon_days}d",
    }
    raw_prefixes = tuple(cfg.get("raw_feature_exclude_prefixes", ["159934_", "518800_", "518880_", "gld_", "usd_cny_", "usd_cnh_"]))
    live_overlay_prefixes = prealign_live_overlay_prefixes
    live_overlay_probability_patterns = prealign_live_overlay_probability_patterns
    exclude_live_overlay_from_model = bool(cfg.get("exclude_live_overlay_from_model_features", True))
    feature_columns = [
        col
        for col in df.columns
        if col not in exclude
        and col != "primary_etf"
        and not col.startswith(raw_prefixes)
        and not (
            exclude_live_overlay_from_model
            and (
                col.startswith(live_overlay_prefixes)
                or any(pattern in col for pattern in live_overlay_probability_patterns)
            )
        )
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    df = apply_selective_feature_availability_lag(
        df,
        feature_columns,
        int(cfg.get("feature_lag_rows", 1)),
        list(
            cfg.get(
                "no_lag_feature_prefixes",
                [
                    "external_gap_",
                    "primary_trading_gap_",
                    "news_",
                    "fed_dovish",
                    "geopolitical_risk",
                    "fiscal_risk",
                    "usd_pressure",
                    "inflation_risk",
                    "central_bank_buying_news",
                    "liquidity_stress_news",
                    "fed_rate_cut_probability",
                    "us_recession_probability",
                    "geopolitical_risk_probability",
                    "tariff_trade_risk_probability",
                    "gold_upside_probability",
                ],
            )
        ),
    )
    required = list(cfg.get("required_features", ["xauusd_return_1d", "dxy_return", "real_yield_10y_level", "vix_level"]))
    required = [col for col in required if col in df.columns]
    model_df = df.dropna(subset=["primary_etf"] + required).copy()
    live_overlay_cols = [
        col
        for col in df.columns
        if col.startswith(live_overlay_prefixes)
        or any(pattern in col for pattern in live_overlay_probability_patterns)
    ]
    if live_overlay_cols:
        model_df[live_overlay_cols] = df[live_overlay_cols].ffill().reindex(model_df.index)
    model_df = fill_model_feature_nans(model_df, feature_columns)
    feature_columns = [col for col in feature_columns if col in model_df and model_df[col].notna().any()]
    return GoldFeatureFrame(model_df, feature_columns)


def keep_rows_with_label_end_in_index(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively keep rows whose label_end_date remains in the final sample index."""
    out = df.copy()
    while True:
        before = len(out)
        out = out[pd.to_datetime(out["label_end_date"]).isin(set(out.index))].copy()
        if len(out) == before:
            return out


def build_gold_dataset(
    daily_series: pd.DataFrame,
    config: dict[str, Any] | None = None,
    news_events: pd.DataFrame | None = None,
    polymarket_data: pd.DataFrame | None = None,
    alpha158_features: pd.DataFrame | None = None,
) -> GoldDatasetBundle:
    cfg = config or {}
    horizon_days = int(cfg.get("horizon_days", 5))
    threshold = float(cfg.get("return_threshold", 0.0))
    full_position_return = float(cfg.get("position_full_return_threshold", 0.02))
    feature_frame = build_gold_feature_frame(daily_series, cfg, news_events, polymarket_data, alpha158_features)
    model_df = feature_frame.data
    feature_columns = feature_frame.feature_columns
    model_df = add_fixed_horizon_target(model_df, "primary_etf", horizon_days, threshold, full_position_return)
    model_df = model_df.dropna(subset=["target_direction", "label_end_date"]).copy()
    return GoldDatasetBundle(
        model_df[feature_columns + ["primary_etf", "label_end_date", "target_direction", "target_position", "target_forward_return"]],
        feature_columns,
        "target_direction",
    )
