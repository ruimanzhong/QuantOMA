#!/usr/bin/env python
"""Build Qlib Alpha158 features for Phase 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.features.qlib_alpha158_adapter import build_qlib_alpha158_features


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--feature-set", default="alpha158")
    pre_args, _ = pre_parser.parse_known_args()
    config = yaml.safe_load((PROJECT_ROOT / "config" / "feature_sets.yaml").read_text())[pre_args.feature_set]

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--provider-uri", default=config["provider_uri"])
    parser.add_argument("--instruments", nargs="+", default=config["instruments"])
    parser.add_argument("--start-date", default=config["start_time"])
    parser.add_argument("--end-date", default=config["end_time"])
    parser.add_argument("--fit-start-date", default=config.get("fit_start_time"))
    parser.add_argument("--fit-end-date", default=config.get("fit_end_time"))
    parser.add_argument("--freq", default=config.get("freq", "day"))
    parser.add_argument("--normalize", action="store_true", default=bool(config.get("normalize", False)))
    parser.add_argument("--output", default=config["output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    instruments = args.instruments
    if isinstance(instruments, list) and len(instruments) == 1 and instruments[0].lower() == "all":
        instruments = "all"
    df = build_qlib_alpha158_features(
        provider_uri=args.provider_uri,
        instruments=instruments,
        start_time=args.start_date,
        end_time=args.end_date,
        fit_start_time=args.fit_start_date,
        fit_end_time=args.fit_end_date,
        freq=args.freq,
        normalize=args.normalize,
        output_path=str(output),
    )
    feature_cols = [col for col in df.columns if col not in {"date", "asset_id"}]
    print(f"output path: {output}")
    print(f"n rows: {len(df)}")
    print(f"n assets: {df['asset_id'].nunique() if 'asset_id' in df.columns else 0}")
    print(f"first date: {df['date'].min() if 'date' in df.columns and not df.empty else None}")
    print(f"last date: {df['date'].max() if 'date' in df.columns and not df.empty else None}")
    print(f"n feature columns: {len(feature_cols)}")


if __name__ == "__main__":
    main()
