import pandas as pd
import pytest

from market_quant.gold.backtest import build_gold_positions, run_gold_long_flat_backtest, run_gold_position_backtest
from market_quant.gold.features import (
    align_to_primary_trading_calendar,
    append_preopen_primary_row,
    build_external_gap_features,
    build_gold_feature_frame,
    build_gold_dataset,
    build_news_features,
)
from market_quant.gold.live_signal import build_live_gold_signal
from market_quant.diagnostics.gold_pipeline_audit import audit_gold_dataset
from market_quant.models.walk_forward import make_purged_walk_forward_folds
from market_quant.models.walk_forward import fit_probability_calibrator
from market_quant.models.walk_forward import fit_orthogonalization_model
from market_quant.models.walk_forward import apply_orthogonalization_model
from market_quant.models.walk_forward import select_alpha158_features_by_train_ic
from market_quant.models.walk_forward import select_features_by_correlation_clusters


def _series_frame(n=900):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    values = {
        "518880_close": 3.0 + pd.Series(range(n), dtype=float) * 0.001,
        "xauusd_close": 1500.0 + pd.Series(range(n), dtype=float),
        "usd_cny_close": 7.0 + pd.Series(range(n), dtype=float) * 0.0001,
        "dxy": 100.0 + pd.Series(range(n), dtype=float) * 0.001,
        "vix": 20.0 + pd.Series(range(n), dtype=float) * 0.001,
        "tips_10y": 1.0 + pd.Series(range(n), dtype=float) * 0.0001,
        "treasury_10y": 2.0 + pd.Series(range(n), dtype=float) * 0.0001,
    }
    for series_id, series in values.items():
        for date, value in zip(dates, series, strict=False):
            rows.append({"date": date, "series_id": series_id, "value": value, "source": "test"})
    return pd.DataFrame(rows)


def test_build_gold_dataset_applies_feature_lag_and_target():
    bundle = build_gold_dataset(
        _series_frame(900),
        {
            "horizon_days": 5,
            "feature_lag_rows": 1,
            "primary_etf_series": "518880_close",
            "required_features": ["xauusd_return_1d", "dxy_return", "real_yield_10y_level", "vix_level"],
        },
    )

    assert "target_direction" in bundle.data.columns
    assert "target_position" in bundle.data.columns
    assert bundle.data["target_position"].between(0.0, 1.0).all()
    assert "xauusd_return_1d" in bundle.feature_columns
    assert bundle.data.index.max() < pd.bdate_range("2020-01-01", periods=900).max()
    audit = audit_gold_dataset(bundle.data.reset_index(names="date"), 5, bundle.feature_columns)
    assert audit["passed"].all()


def test_align_to_primary_trading_calendar_drops_macro_only_rows():
    df = pd.DataFrame(
        {
            "518880_close": [1.0, None, 1.1],
            "dxy": [100.0, 101.0, 102.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    out = align_to_primary_trading_calendar(df, "518880_close")

    assert out.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]


def test_external_gap_features_capture_holiday_price_discovery():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    wide = pd.DataFrame(
        {
            "518880_close": [1.0, None, 1.02],
            "xauusd_close": [2000.0, 2040.0, 2050.0],
            "usd_cny_close": [7.10, 7.12, 7.13],
        },
        index=dates,
    )

    out = build_external_gap_features(wide, "518880_close")

    assert out.loc[pd.Timestamp("2024-01-04"), "primary_trading_gap_calendar_days"] == 2
    assert out.loc[pd.Timestamp("2024-01-04"), "external_gap_observation_count"] == 1
    assert out.loc[pd.Timestamp("2024-01-04"), "external_gap_xauusd_return"] == pytest.approx(0.02)
    assert out.loc[pd.Timestamp("2024-01-04"), "external_gap_usd_cny_return"] == pytest.approx(7.12 / 7.10 - 1)


def test_append_preopen_primary_row_uses_last_known_close():
    series = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "series_id": ["518880_close", "xauusd_close"],
            "value": [1.0, 2040.0],
            "source": ["test", "test"],
        }
    )

    out = append_preopen_primary_row(series, "518880_close", "2024-01-04")

    row = out[(pd.to_datetime(out["date"]).eq(pd.Timestamp("2024-01-04"))) & out["series_id"].eq("518880_close")].iloc[0]
    assert row["value"] == 1.0
    assert row["source"] == "synthetic_preopen_last_close"


def test_build_gold_dataset_keeps_external_gap_features_unlagged():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
    values = {
        "518880_close": [1.0, None, 1.02, 1.03, 1.04, 1.05, 1.06],
        "xauusd_close": [2000.0, 2040.0, 2050.0, 2060.0, 2070.0, 2080.0, 2090.0],
        "usd_cny_close": [7.10, 7.12, 7.13, 7.14, 7.15, 7.16, 7.17],
        "dxy": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6],
        "vix": [20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6],
        "tips_10y": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06],
    }
    rows = []
    for series_id, series in values.items():
        for date, value in zip(dates, series, strict=False):
            if pd.notna(value):
                rows.append({"date": date, "series_id": series_id, "value": value, "source": "test"})

    bundle = build_gold_dataset(
        pd.DataFrame(rows),
        {
            "horizon_days": 1,
            "feature_lag_rows": 1,
            "primary_etf_series": "518880_close",
            "price_windows": [1],
            "volatility_windows": [2],
            "required_features": [],
        },
    )

    row = bundle.data.loc[pd.Timestamp("2024-01-04")]
    assert "external_gap_xauusd_return" in bundle.feature_columns
    assert row["external_gap_xauusd_return"] == pytest.approx(0.02)


def test_live_gold_signal_converts_probability_to_position():
    dates = pd.bdate_range("2024-01-01", periods=80)
    training = pd.DataFrame(
        {
            "primary_etf": 1.0 + pd.Series(range(80), dtype=float).to_numpy() * 0.001,
            "feature_a": [0.0, 1.0] * 40,
            "feature_b": [1.0, 0.0] * 40,
            "target_direction": [0, 1] * 40,
        },
        index=dates,
    )
    feature_frame = pd.DataFrame(
        {
            "primary_etf": [1.08],
            "feature_a": [1.0],
            "feature_b": [0.0],
            "xauusd_momentum_20d": [1.0],
            "xauusd_momentum_60d": [1.0],
            "xauusd_ma_distance_60d": [1.0],
            "dxy_trend_20d": [-0.01],
            "real_yield_trend_20d": [-0.01],
            "vix_change_5d": [1.0],
        },
        index=[pd.Timestamp("2024-04-22")],
    )

    signal, _ = build_live_gold_signal(
        training,
        feature_frame,
        ["feature_a", "feature_b"],
        "2024-04-22",
        model_name="logistic_regression",
        model_parameters={"random_state": 42},
        calibration_config={"method": "platt", "size": 20, "min_fit_samples": 40},
        sizing_config={"strategy_type": "sigmoid_bet_sizing", "volatility_targeting": False},
        feature_selection_config={
            "enabled": True,
            "alpha_prefix": "alpha158_",
            "orthogonalization": {
                "enabled": True,
                "alpha_prefix": "alpha158_",
                "risk_features": ["feature_b"],
                "keep_original": False,
            },
        },
    )

    assert 0.0 <= signal.probability <= 1.0
    assert 0.0 <= signal.raw_probability <= 1.0
    assert 0.0 <= signal.sigmoid_position <= 1.0
    assert 0.0 <= signal.final_position <= 1.0
    assert signal.feature_date == pd.Timestamp("2024-04-22")


def test_orthogonalization_removes_linear_macro_beta():
    risk = pd.Series(range(120), dtype=float)
    alpha = 2.0 * risk + pd.Series([0.0, 1.0] * 60)
    df = pd.DataFrame({"risk": risk, "alpha158_beta": alpha})

    model = fit_orthogonalization_model(df, ["alpha158_beta"], ["risk"])
    out, created = apply_orthogonalization_model(df, model)

    assert created == ["alpha158_beta_orth"]
    assert abs(out["alpha158_beta_orth"].corr(df["risk"])) < 1e-5


def test_build_gold_dataset_includes_alpha158_features():
    dates = pd.bdate_range("2020-01-01", periods=900)
    alpha = pd.DataFrame({"alpha158_RESI5": range(900)}, index=dates)
    bundle = build_gold_dataset(
        _series_frame(900),
        {
            "horizon_days": 5,
            "feature_lag_rows": 1,
            "primary_etf_series": "518880_close",
            "include_alpha158": True,
            "required_features": ["xauusd_return_1d", "dxy_return", "real_yield_10y_level", "vix_level"],
        },
        alpha158_features=alpha,
    )

    assert "alpha158_RESI5" in bundle.feature_columns


def test_build_news_features_includes_event_channel_scores():
    events = pd.DataFrame(
        {
            "published_at": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "event_type": ["fed", "geopolitics", "usd"],
            "gold_direction": ["bullish", "bullish", "bearish"],
            "confidence": [0.7, 0.5, 0.6],
            "surprise": [0.5, 0.2, 0.3],
            "affected_channels": ["real_yield,usd", "safe_haven", "usd"],
        }
    )

    out = build_news_features(events)

    assert out["news_bullish_score"].iloc[0] == pytest.approx(0.45)
    assert out["fed_dovish_score"].iloc[0] == pytest.approx(0.35)
    assert out["geopolitical_risk_score"].iloc[0] == pytest.approx(0.10)
    assert out["usd_pressure_score"].iloc[0] == pytest.approx(0.18)
    assert "fed_dovish_score_ewm_hl_3d" in out.columns


def test_short_history_external_features_stay_out_of_model_candidates_by_default():
    dates = pd.bdate_range("2024-01-01", periods=80)
    series = []
    for series_id, base in {
        "518880_close": 1.0,
        "xauusd_close": 2000.0,
        "dxy": 100.0,
        "vix": 20.0,
        "tips_10y": 1.0,
    }.items():
        for idx, date in enumerate(dates):
            series.append({"date": date, "series_id": series_id, "value": base + idx * 0.01, "source": "test"})
    news = pd.DataFrame(
        {
            "published_at": [dates[-1]],
            "event_type": ["fed"],
            "gold_direction": ["bullish"],
            "confidence": [0.8],
            "surprise": [0.5],
            "affected_channels": ["usd"],
        }
    )
    polymarket = pd.DataFrame({"timestamp": [dates[-1]], "market_group": ["fed_rate_cut"], "probability": [0.4]})

    frame = build_gold_feature_frame(
        pd.DataFrame(series),
        {
            "feature_lag_rows": 1,
            "primary_etf_series": "518880_close",
            "price_windows": [1],
            "volatility_windows": [2],
            "required_features": [],
            "exclude_live_overlay_from_model_features": True,
        },
        news_events=news,
        polymarket_data=polymarket,
    )

    assert "news_bullish_score" in frame.data.columns
    assert "fed_rate_cut_probability_level" in frame.data.columns
    assert "news_bullish_score" not in frame.feature_columns
    assert "fed_rate_cut_probability_level" not in frame.feature_columns


def test_make_purged_walk_forward_folds_respects_label_end_date():
    dates = pd.bdate_range("2020-01-01", periods=120)
    df = pd.DataFrame(index=dates)
    df["label_end_date"] = pd.Series(dates, index=dates).shift(-5)
    folds = make_purged_walk_forward_folds(
        df,
        initial_train_size=60,
        test_size=20,
        step_size=20,
        min_train_samples=10,
        min_test_samples=10,
    )

    first = folds[0]
    assert pd.to_datetime(df.loc[first["train_index"], "label_end_date"]).max() < first["test_start"]


def test_gold_backtest_outputs_metrics():
    predictions = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "probability": [0.6, 0.6, 0.4, 0.7, 0.7],
            "primary_etf": [1.0, 1.01, 1.0, 1.02, 1.03],
        }
    )

    daily, metrics = run_gold_long_flat_backtest(predictions, threshold=0.55, transaction_cost_bps=2)

    assert "strategy_return" in daily.columns
    assert "sharpe" in metrics


def test_gold_position_backtest_uses_direct_position():
    predictions = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "position": [0.0, 0.25, 0.5, 0.75, 1.0],
            "primary_etf": [1.0, 1.01, 1.0, 1.02, 1.03],
        }
    )

    daily, metrics = run_gold_position_backtest(predictions, transaction_cost_bps=2)

    assert daily["position"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert "strategy_return" in daily.columns
    assert metrics["strategy_type"] == "direct_position"


def test_gold_position_variants():
    probability = pd.Series([0.3, 0.5, 0.7])
    threshold = build_gold_positions(probability, strategy_type="threshold", threshold=0.55)
    scaled = build_gold_positions(probability, strategy_type="probability_scaled", lower_threshold=0.4, threshold=0.6)
    overlay = build_gold_positions(probability, strategy_type="defensive_overlay", lower_threshold=0.4, defensive_position=0.3)
    sigmoid = build_gold_positions(probability, strategy_type="sigmoid_bet_sizing", base_position=0.1, sigmoid_threshold=0.55, sigmoid_sigma=0.08)

    assert threshold.tolist() == [0.0, 0.0, 1.0]
    assert scaled.round(2).tolist() == [0.0, 0.5, 1.0]
    assert overlay.tolist() == [0.3, 1.0, 1.0]
    assert sigmoid.is_monotonic_increasing
    assert sigmoid.between(0.0, 1.0).all()


def test_platt_calibrator_outputs_valid_probabilities():
    raw = pd.Series([0.2, 0.3, 0.4, 0.7, 0.8, 0.9])
    target = pd.Series([0, 0, 0, 1, 1, 1])

    calibrator = fit_probability_calibrator(raw.to_numpy(), target, "platt")
    calibrated = calibrator.predict_proba(raw.to_frame())[:, 1]

    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0
    assert calibrated[-1] > calibrated[0]


def test_correlation_cluster_feature_selection_deduplicates_similar_features():
    target = [0, 1] * 60
    df = pd.DataFrame(
        {
            "feature_a": target,
            "feature_a_clone": target,
            "feature_b": [0, 0, 1, 1] * 30,
            "target_direction": target,
        }
    )

    selected, report = select_features_by_correlation_clusters(
        df,
        ["feature_a", "feature_a_clone", "feature_b"],
        "target_direction",
        max_abs_spearman=0.95,
    )

    assert len({"feature_a", "feature_a_clone"}.intersection(selected)) == 1
    assert "feature_b" in selected
    assert not report.empty


def test_alpha158_feature_selection_uses_training_data_only():
    target = [0, 1, 0, 1, 0, 1] * 20
    df = pd.DataFrame(
        {
            "macro": [0, 1, 0, 1, 0, 1] * 20,
            "alpha158_good": [value + idx * 0.001 for idx, value in enumerate(target)],
            "alpha158_bad": [1, 1, 1, 1, 1, 1] * 20,
            "target_direction": target,
        }
    )

    selected, score = select_alpha158_features_by_train_ic(
        df,
        ["macro", "alpha158_good", "alpha158_bad"],
        "target_direction",
        max_alpha_features=1,
        min_abs_ic=0.1,
    )

    assert selected == ["macro", "alpha158_good"]
    assert set(score["feature"]) == {"alpha158_good", "alpha158_bad"}
