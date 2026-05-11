#!/usr/bin/env python
"""Run the Phase 1 research baseline verification gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Step:
    title: str
    command: list[str]
    skip_when: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 verification commands.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching A-share ETF data.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after failures, then return non-zero if any step failed.",
    )
    return parser.parse_args()


def verification_steps() -> list[Step]:
    py = sys.executable
    return [
        Step("Unit tests", [py, "-B", "-m", "pytest", "-p", "no:cacheprovider"], skip_when="skip_tests"),
        Step("Fetch A-share ETF data", [py, "scripts/data/fetch_a_share_data.py"], skip_when="skip_fetch"),
        Step("Check A-share data coverage", [py, "scripts/diagnostics/check_data_coverage.py"]),
        Step("Run rule baselines", [py, "scripts/strategies/run_rule_baselines.py"]),
        Step("Evaluate rule baselines", [py, "scripts/strategies/evaluate_rule_baselines.py"]),
        Step("Diagnose sample attrition", [py, "scripts/diagnostics/diagnose_sample_attrition.py"]),
        Step("Diagnose annual rule performance", [py, "scripts/diagnostics/diagnose_rule_annual_performance.py"]),
        Step("Diagnose subperiod rule performance", [py, "scripts/diagnostics/diagnose_rule_subperiod_performance.py"]),
        Step("Run rule parameter sensitivity", [py, "scripts/strategies/run_rule_parameter_sensitivity.py"]),
        Step("Run transaction cost sensitivity", [py, "scripts/backtests/run_transaction_cost_sensitivity.py"]),
        Step("Summarize rule robustness", [py, "scripts/diagnostics/summarize_rule_robustness.py"]),
    ]


def should_skip(step: Step, args: argparse.Namespace) -> bool:
    return bool(step.skip_when and getattr(args, step.skip_when))


def run_step(step: Step) -> int:
    print(f"\n=== {step.title} ===", flush=True)
    print("$ " + " ".join(step.command), flush=True)
    completed = subprocess.run(step.command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    failures: list[tuple[str, int]] = []

    for step in verification_steps():
        if should_skip(step, args):
            print(f"\n=== {step.title} ===", flush=True)
            print("skipped", flush=True)
            continue

        returncode = run_step(step)
        if returncode == 0:
            continue

        failures.append((step.title, returncode))
        print(f"FAILED: {step.title} exited with code {returncode}", flush=True)
        if not args.continue_on_error:
            return returncode

    if failures:
        print("\nPhase 1 verification completed with failures:", flush=True)
        for title, returncode in failures:
            print(f"- {title}: exit code {returncode}", flush=True)
        return 1

    print("\nPhase 1 verification completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
