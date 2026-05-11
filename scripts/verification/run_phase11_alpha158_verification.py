#!/usr/bin/env python
"""Run the Phase 1.1 local ETF Alpha158 verification gate."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local ETF Qlib provider and Alpha158 features.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after failures, then return non-zero if any step failed.",
    )
    return parser.parse_args()


def verification_steps() -> list[Step]:
    py = sys.executable
    return [
        Step("Check A-share ETF data coverage", [py, "scripts/diagnostics/check_data_coverage.py"]),
        Step("Build local ETF Qlib provider", [py, "scripts/features/build_etf_qlib_provider.py"]),
        Step(
            "Inspect local ETF Qlib provider",
            [py, "scripts/diagnostics/inspect_qlib_provider.py", "--feature-set", "alpha158_etf"],
        ),
        Step(
            "Build ETF Alpha158 features",
            [py, "scripts/features/build_qlib_alpha158_features.py", "--feature-set", "alpha158_etf"],
        ),
        Step(
            "Diagnose ETF Alpha158 features",
            [py, "scripts/diagnostics/diagnose_alpha158_features.py", "--feature-set", "alpha158_etf"],
        ),
    ]


def run_step(step: Step) -> int:
    print(f"\n=== {step.title} ===", flush=True)
    print("$ " + " ".join(step.command), flush=True)
    completed = subprocess.run(step.command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    failures: list[tuple[str, int]] = []

    for step in verification_steps():
        returncode = run_step(step)
        if returncode == 0:
            continue
        failures.append((step.title, returncode))
        print(f"FAILED: {step.title} exited with code {returncode}", flush=True)
        if not args.continue_on_error:
            return returncode

    if failures:
        print("\nPhase 1.1 Alpha158 verification completed with failures:", flush=True)
        for title, returncode in failures:
            print(f"- {title}: exit code {returncode}", flush=True)
        return 1

    print("\nPhase 1.1 Alpha158 verification completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
