#!/usr/bin/env python
"""Summarize Phase 1 rule robustness diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_REPORTS = {
    "rule_vs_buy_hold": PROJECT_ROOT / "data" / "reports" / "rule_vs_buy_hold_comparison.csv",
    "stability": PROJECT_ROOT / "data" / "reports" / "strategy_stability_summary.csv",
    "annual": PROJECT_ROOT / "data" / "reports" / "rule_annual_performance.csv",
    "subperiod": PROJECT_ROOT / "data" / "reports" / "rule_subperiod_performance.csv",
    "parameter": PROJECT_ROOT / "data" / "reports" / "rule_parameter_sensitivity.csv",
    "cost": PROJECT_ROOT / "data" / "reports" / "transaction_cost_sensitivity.csv",
}


def require_report(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required robustness input missing: {path}")


def _family_for_strategy(strategy_name: str) -> str | None:
    if strategy_name.startswith("momentum_"):
        return "momentum"
    if strategy_name.startswith("donchian_breakout_"):
        return "donchian_breakout"
    if strategy_name.startswith("ma_trend"):
        return "ma_trend"
    if strategy_name.startswith("support_pullback"):
        return "support_pullback"
    return None


def _score_ratio(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return max(0.0, min(float(value), 1.5)) / 1.5


def build_robustness_summary(report_paths: dict[str, Path] | None = None) -> pd.DataFrame:
    paths = report_paths or REQUIRED_REPORTS
    for path in paths.values():
        require_report(path)

    stability = pd.read_csv(paths["stability"])
    annual = pd.read_csv(paths["annual"])
    subperiod = pd.read_csv(paths["subperiod"])
    parameter = pd.read_csv(paths["parameter"])
    cost = pd.read_csv(paths["cost"])

    max_stability = stability["stability_score"].max() or 1.0
    stability_score = stability.set_index("strategy_name")["stability_score"] / max_stability
    annual_ratio = (annual["cumulative_return"] > 0).groupby(annual["strategy_name"]).mean()
    subperiod_ratio = (subperiod["cumulative_return"] > 0).groupby(subperiod["strategy_name"]).mean()

    family_param = parameter.groupby(["strategy_family", "strategy_variant"])["sharpe"].mean().reset_index()
    family_param["rank_pct"] = family_param.groupby("strategy_family")["sharpe"].rank(pct=True)
    family_scores = family_param.groupby("strategy_family")["rank_pct"].mean()

    cost_pivot = cost.pivot_table(
        index="strategy_name",
        columns="transaction_cost_bps",
        values="sharpe",
        aggfunc="mean",
    )
    cost_scores = {}
    for strategy_name, row in cost_pivot.iterrows():
        zero = row.get(0, float("nan"))
        ten = row.get(10, float("nan"))
        if pd.isna(zero) or abs(zero) < 1e-12:
            cost_scores[strategy_name] = float("nan")
        else:
            cost_scores[strategy_name] = _score_ratio(ten / zero)

    rows = []
    strategy_names = sorted(
        set(stability["strategy_name"])
        | set(annual["strategy_name"])
        | set(subperiod["strategy_name"])
        | set(cost["strategy_name"])
    )
    for strategy_name in strategy_names:
        family = _family_for_strategy(strategy_name)
        parameter_score = family_scores.get(family, float("nan")) if family else float("nan")
        scores = [
            stability_score.get(strategy_name, float("nan")),
            annual_ratio.get(strategy_name, float("nan")),
            subperiod_ratio.get(strategy_name, float("nan")),
            parameter_score,
            cost_scores.get(strategy_name, float("nan")),
        ]
        score_series = pd.Series(scores, dtype="float64")
        notes = []
        if annual_ratio.get(strategy_name, 0.0) < 0.5:
            notes.append("weak_annual_consistency")
        if subperiod_ratio.get(strategy_name, 0.0) < 0.5:
            notes.append("subperiod_dependent")
        if pd.notna(cost_scores.get(strategy_name, float("nan"))) and cost_scores[strategy_name] < 0.7:
            notes.append("cost_sensitive")
        if pd.isna(parameter_score):
            notes.append("no_parameter_grid")
        rows.append(
            {
                "strategy_name": strategy_name,
                "cross_asset_stability_score": stability_score.get(strategy_name, float("nan")),
                "positive_year_ratio": annual_ratio.get(strategy_name, float("nan")),
                "positive_subperiod_ratio": subperiod_ratio.get(strategy_name, float("nan")),
                "parameter_robustness_score": parameter_score,
                "cost_robustness_score": cost_scores.get(strategy_name, float("nan")),
                "overall_robustness_score": score_series.mean(skipna=True),
                "notes": ";".join(notes),
            }
        )
    return pd.DataFrame(rows).sort_values("overall_robustness_score", ascending=False).reset_index(drop=True)


def main() -> None:
    try:
        summary = build_robustness_summary()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    out = PROJECT_ROOT / "data" / "reports" / "rule_robustness_summary.csv"
    summary.to_csv(out, index=False)
    print(summary.head(10).to_string(index=False))
    print(f"\nsaved rule robustness summary to {out}")


if __name__ == "__main__":
    main()
