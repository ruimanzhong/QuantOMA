import pandas as pd

from market_quant.options.live import build_live_option_plan
from market_quant.options.strategies import generate_option_candidates


def option_chain() -> pd.DataFrame:
    rows = []
    for dte, expiry in [(12, "2026-05-29"), (35, "2026-06-30")]:
        for strike, call_delta, put_delta in [(980, 0.65, -0.25), (1000, 0.50, -0.50), (1020, 0.25, -0.65)]:
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-12"),
                    "symbol": f"AU{expiry}C{strike}",
                    "underlying_symbol": "AU9999",
                    "option_type": "call",
                    "expiry": pd.Timestamp(expiry),
                    "strike": strike,
                    "dte": dte,
                    "underlying_price": 1000.0,
                    "mid": 8.0 if strike == 1000 else 5.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "iv": 0.18,
                    "multiplier": 100.0,
                    "delta": call_delta,
                }
            )
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-12"),
                    "symbol": f"AU{expiry}P{strike}",
                    "underlying_symbol": "AU9999",
                    "option_type": "put",
                    "expiry": pd.Timestamp(expiry),
                    "strike": strike,
                    "dte": dte,
                    "underlying_price": 1000.0,
                    "mid": 8.0 if strike == 1000 else 5.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "iv": 0.18,
                    "multiplier": 100.0,
                    "delta": put_delta,
                }
            )
    return pd.DataFrame(rows)


def test_directional_option_candidates_use_trend_dte_bucket():
    candidates = generate_option_candidates(
        option_chain(),
        p_up=0.62,
        vol_state={"volatility_state": "neutral_iv"},
        event_risk=False,
        config={
            "expiry_selection": {"trend_min_dte": 25, "trend_max_dte": 60, "event_min_dte": 10, "event_max_dte": 30},
            "signal_horizon_days": 5,
        },
    )

    spread = next(c for c in candidates if c.strategy_type == "bull_call_spread")

    assert spread.diagnostics["expiry_bucket"] == "trend"
    assert {leg.contract.dte for leg in spread.legs} == {35}


def test_event_option_candidates_use_event_dte_bucket():
    candidates = generate_option_candidates(
        option_chain(),
        p_up=0.50,
        vol_state={"volatility_state": "low_iv"},
        event_risk=True,
        config={
            "expiry_selection": {"trend_min_dte": 25, "trend_max_dte": 60, "event_min_dte": 10, "event_max_dte": 30},
            "signal_horizon_days": 5,
        },
    )

    combo = next(c for c in candidates if c.strategy_type == "long_straddle")

    assert combo.diagnostics["expiry_bucket"] == "event"
    assert {leg.contract.dte for leg in combo.legs} == {12}


def test_bearish_option_candidates_use_hedge_dte_bucket():
    candidates = generate_option_candidates(
        option_chain(),
        p_up=0.38,
        vol_state={"volatility_state": "neutral_iv"},
        event_risk=False,
        config={
            "expiry_selection": {
                "trend_min_dte": 50,
                "trend_max_dte": 60,
                "event_min_dte": 10,
                "event_max_dte": 30,
                "hedge_min_dte": 20,
                "hedge_max_dte": 45,
            },
            "signal_horizon_days": 5,
        },
    )

    spread = next(c for c in candidates if c.strategy_type == "bear_put_spread")

    assert spread.diagnostics["expiry_bucket"] == "hedge"
    assert {leg.contract.dte for leg in spread.legs} == {35}


def test_live_option_plan_serializes_holding_and_exit_rules():
    out = build_live_option_plan(
        gold_signal={"probability": 0.62, "feature_date": "2026-05-12", "final_position": 0.8},
        feature_frame=pd.DataFrame({"primary_etf": [990.0, 995.0, 1000.0]}, index=pd.date_range("2026-05-08", periods=3)),
        option_chain=option_chain(),
        config={
            "enabled": True,
            "account_equity": 100000.0,
            "holding": {"max_holding_days": 5, "reevaluate_daily": True},
            "exits": {"spread_take_profit_of_max_profit": 0.5},
            "selector": {"min_edge_score": -999},
            "risk": {"max_single_trade_loss_pct": 0.10, "max_premium_risk_pct": 0.10},
        },
    )

    assert out["signal_horizon_days"] == 5
    assert out["holding"]["max_holding_days"] == 5
    assert out["exits"]["spread_take_profit_of_max_profit"] == 0.5


def test_live_option_plan_stays_disabled_when_config_disabled():
    out = build_live_option_plan(
        gold_signal={"probability": 0.62, "feature_date": "2026-05-12"},
        feature_frame=pd.DataFrame(),
        option_chain=option_chain(),
        config={"enabled": False, "holding": {"max_holding_days": 5}},
    )

    assert out == {"enabled": False, "reason": "disabled"}
