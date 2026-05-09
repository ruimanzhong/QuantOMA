import pandas as pd

from market_quant.data.polymarket_data import append_polymarket_cache, extract_outcome_probability, normalize_polymarket_market
from market_quant.gold.live_overlay import build_live_overlay_result


def test_extract_outcome_probability_handles_json_strings():
    market = {"outcomes": '["Yes", "No"]', "outcomePrices": '["0.37", "0.63"]'}

    assert extract_outcome_probability(market, "Yes") == 0.37
    assert extract_outcome_probability(market, "No") == 0.63


def test_normalize_polymarket_market_outputs_feature_schema():
    market = {
        "id": "123",
        "slug": "fed-rate-cut",
        "question": "Will the Fed cut rates?",
        "outcomes": ["Yes", "No"],
        "outcomePrices": [0.42, 0.58],
        "volume": "10000",
        "liquidity": "5000",
        "endDate": "2026-12-31",
    }

    out = normalize_polymarket_market(market, "fed_rate_cut")

    assert out["market_group"] == "fed_rate_cut"
    assert out["probability"] == 0.42
    assert out["volume"] == 10000.0


def test_append_polymarket_cache_deduplicates_snapshots():
    rows = pd.DataFrame(
        {
            "timestamp": ["2026-05-05"],
            "market_group": ["fed_rate_cut"],
            "market_id": ["123"],
            "slug": ["fed-rate-cut"],
            "question": ["Will the Fed cut rates?"],
            "probability": [0.42],
            "volume": [10000.0],
            "liquidity": [5000.0],
            "end_date": ["2026-12-31"],
            "source": ["test"],
        }
    )

    out = append_polymarket_cache(rows, rows)

    assert len(out) == 1


def test_live_overlay_penalizes_high_polymarket_risk():
    row = pd.DataFrame(
        {
            "geopolitical_risk_probability_level": [0.8],
            "gold_upside_probability_level": [0.4],
            "news_bullish_score_sum_3d": [0.0],
            "news_bearish_score_sum_3d": [0.0],
        }
    )

    out = build_live_overlay_result(row, {"enabled": True, "polymarket": {"enabled": True, "extreme_risk_threshold": 0.75}})

    assert out.multiplier < 1.0
    assert out.reasons
