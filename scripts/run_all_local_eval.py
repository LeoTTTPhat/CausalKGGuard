#!/usr/bin/env python3
"""Run the complete local reproducibility pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "synthetic" / "trace_cases_extended.jsonl"
DEFAULT_POLICY = ROOT / "configs" / "policies.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    commands: List[List[str]] = []
    if not args.skip_generate:
        commands.append([sys.executable, "scripts/generate_extended_synthetic.py"])
    commands.extend(
        [
            [sys.executable, "scripts/validate_trace_cases.py", "--data", str(args.data)],
            [
                sys.executable,
                "scripts/run_synthetic_eval.py",
                "--data",
                str(args.data),
                "--prefix",
                "extended_synthetic_eval",
                "--policy-config",
                str(args.policy_config),
            ],
            [
                sys.executable,
                "scripts/run_ablation_eval.py",
                "--data",
                str(args.data),
                "--policy-config",
                str(args.policy_config),
            ],
            [
                sys.executable,
                "scripts/run_explanation_eval.py",
                "--data",
                str(args.data),
                "--policy-config",
                str(args.policy_config),
            ],
            [
                sys.executable,
                "scripts/run_statistical_analysis.py",
                "--data",
                str(args.data),
                "--policy-config",
                str(args.policy_config),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
            ],
            [
                sys.executable,
                "scripts/export_quarantine_plans.py",
                "--data",
                str(args.data),
                "--policy-config",
                str(args.policy_config),
            ],
        ]
    )
    if not args.skip_runtime:
        commands.append([sys.executable, "scripts/run_runtime_eval.py"])

    for command in commands:
        print("+ " + " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
