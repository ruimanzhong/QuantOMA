import pandas as pd

from market_quant.gold.layered import (
    build_gold_regime_scores,
    build_hmm_style_regime_prior,
    build_layered_gold_positions,
    run_layered_gold_backtest,
)


def feature_frame(n=80, include_external=False):
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "date": dates,
            "xauusd_momentum_20d": [0.02] * n,
            "xauusd_momentum_60d": [0.01] * n,
            "xauusd_ma_distance_60d": [0.03] * n,
            "dxy_trend_20d": [-0.01] * n,
            "real_yield_trend_20d": [-0.01] * n,
            "vix_change_5d": [0.2] * n,
            "vix_level": [15.0 if idx < n // 2 else 35.0 for idx in range(n)],
            "sp500_return_20d": [0.02 if idx < n // 2 else -0.08 for idx in range(n)],
            "xauusd_volatility_20d": [0.01 if idx < n // 2 else 0.04 for idx in range(n)],
        }
    )
    if include_external:
        df["news_bullish_score_sum_3d"] = [1.0] * n
        df["news_bearish_score_sum_3d"] = [0.0] * n
        df["fed_rate_cut_probability_change_3d"] = [0.01] * n
    return df


def prediction_frame(n=80):
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "probability": [0.62 if idx % 2 else 0.48 for idx in range(n)],
            "primary_etf": [100 + idx for idx in range(n)],
        }
    )


def test_regime_scores_use_external_features_when_present():
    without_external = build_gold_regime_scores(feature_frame(include_external=False))
    with_external = build_gold_regime_scores(feature_frame(include_external=True))

    assert without_external["news_score"].iloc[0] == 0.5
    assert with_external["news_score"].iloc[0] == 1.0
    assert with_external["polymarket_score"].iloc[0] == 1.0
    assert with_external["regime_score"].iloc[0] > without_external["regime_score"].iloc[0]


def test_layered_positions_respect_bounds_and_regime_columns():
    predictions = prediction_frame()
    regime = build_gold_regime_scores(feature_frame())

    out = build_layered_gold_positions(predictions, regime, min_position=0.2, max_position=1.0)

    assert {"regime_score", "regime_label", "position", "raw_position"}.issubset(out.columns)
    assert out["position"].between(0.2, 1.0).all()


def test_hmm_style_regime_prior_outputs_adaptive_base_position():
    out = build_hmm_style_regime_prior(feature_frame(120))

    assert out["crisis_probability"].between(0.0, 1.0).all()
    assert out["adaptive_base_position"].between(0.1, 0.6).all()
    assert out["crisis_probability"].iloc[-1] > out["crisis_probability"].iloc[0]


def test_layered_positions_use_adaptive_base_position():
    predictions = prediction_frame(5)
    regime = build_gold_regime_scores(feature_frame(5))
    regime["adaptive_base_position"] = 0.6
    regime["crisis_probability"] = 1.0
    regime["hmm_regime_label"] = "crisis"

    out = build_layered_gold_positions(predictions, regime, min_position=0.1, max_position=1.0)

    assert (out["position"] >= 0.6).all()


def test_layered_backtest_outputs_metrics():
    daily, metrics = run_layered_gold_backtest(prediction_frame(), feature_frame(), transaction_cost_bps=2.0)

    assert "strategy_return" in daily.columns
    assert metrics["strategy_type"] == "layered_regime_exposure"
    assert "avg_regime_score" in metrics
