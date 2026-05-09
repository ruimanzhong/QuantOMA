import importlib.util
from pathlib import Path

import pandas as pd

from market_quant.diagnostics.gold_regime import (
    add_price_regime_labels,
    build_buy_hold_daily,
    summarize_by_price_regime,
    summarize_by_subperiod,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_test_module"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_frame(n=170):
    dates = pd.bdate_range("2024-01-01", periods=n)
    prices = [100 + idx * 0.2 for idx in range(n)]
    return pd.DataFrame(
        {
            "date": dates,
            "probability": [0.6 if idx % 3 else 0.4 for idx in range(n)],
            "primary_etf": prices,
        }
    )


def test_gold_subperiod_summary_uses_same_oos_window():
    daily = build_buy_hold_daily(prediction_frame(20))
    periods = [{"name": "jan_2024", "start_date": "2024-01-01", "end_date": "2024-01-31"}]

    out = summarize_by_subperiod(daily, periods)

    assert out["subperiod"].tolist() == ["jan_2024"]
    assert out["strategy_name"].tolist() == ["buy_and_hold"]
    assert out["n_trading_days"].iloc[0] == 20


def test_price_regime_labels_are_observable_and_summarized():
    daily = build_buy_hold_daily(prediction_frame(170))

    labelled = add_price_regime_labels(daily, short_window=20, long_window=60, momentum_window=20)
    out = summarize_by_price_regime(labelled)

    assert "trend_up" in set(labelled["price_regime"])
    assert set(out["strategy_name"]) == {"buy_and_hold"}
    assert out["n_trading_days"].sum() == len(daily)


def test_external_feature_status_detects_news_and_polymarket(tmp_path):
    module = load_script("run_gold_regime_diagnostics.py")
    feature_path = tmp_path / "gold_model_features.txt"
    feature_path.write_text("xauusd_return_1d\nnews_bullish_score_sum_3d\nfed_rate_cut_probability_level\n")

    out = module._feature_source_status(feature_path)

    assert out.set_index("source").loc["news", "available_in_model_features"]
    assert out.set_index("source").loc["polymarket", "available_in_model_features"]
