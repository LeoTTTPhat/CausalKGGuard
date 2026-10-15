#!/usr/bin/env python3
"""Export graph-guided quarantine plans for a trace benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import CausalKGGuard, TraceCase, quarantine_plans


def load_cases(path: Path) -> List[TraceCase]:
    return [
        TraceCase.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "trace_cases_extended.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "quarantine_plans.jsonl")
    parser.add_argument("--policy-config", type=Path, default=None)
    args = parser.parse_args()

    guard = CausalKGGuard(policy_path=args.policy_config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for case in load_cases(args.data):
            findings = guard.evaluate(case)
            for plan in quarantine_plans(case, findings):
                handle.write(json.dumps(asdict(plan), sort_keys=True) + "\n")
                count += 1
    print(f"Wrote {count} quarantine plans to {args.out}")


if __name__ == "__main__":
    main()
