#!/usr/bin/env python
"""Diagnose Phase 1 sample coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.sample_attrition import summarize_sample_attrition


def main() -> None:
    price_path = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    df = pd.read_csv(price_path)
    report = summarize_sample_attrition(df)
    out = PROJECT_ROOT / "data" / "reports" / "sample_attrition.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"saved sample attrition report to {out}")


if __name__ == "__main__":
    main()
