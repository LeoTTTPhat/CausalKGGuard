#!/usr/bin/env python3
"""Summarize AgentDojo utility and security from raw JSON logs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "tables"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-logdir", type=Path, required=True)
    parser.add_argument("--noattack-logdir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="agentdojo_model")
    parser.add_argument("--attack-name", default="important_instructions")
    parser.add_argument("--include-injection-utility-noattack", action="store_true")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    attack_records = [
        record
        for record in read_records(args.attack_logdir)
        if record.get("injection_task_id") is not None
    ]
    rows.extend(summarize_records(args.attack_name, attack_records, include_security=True))

    if args.noattack_logdir is not None:
        noattack_records = [
            record
            for record in read_records(args.noattack_logdir)
            if record.get("injection_task_id") is None
            and (
                args.include_injection_utility_noattack
                or not str(record.get("user_task_id") or "").startswith("injection_task_")
            )
        ]
        rows.extend(summarize_records("no_attack", noattack_records, include_security=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_summary.csv", rows)


def read_records(logdir: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    for path in sorted(logdir.rglob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def summarize_records(run_type: str, records: Iterable[Mapping[str, Any]], include_security: bool) -> List[Dict[str, object]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("suite_name") or "unknown")].append(record)

    rows: List[Dict[str, object]] = []
    for suite, suite_records in sorted(groups.items()):
        rows.append(summary_row(run_type, suite, suite_records, include_security))
    if groups:
        all_records = [record for suite_records in groups.values() for record in suite_records]
        rows.append(summary_row(run_type, "all", all_records, include_security))
    return rows


def summary_row(
    run_type: str,
    suite: str,
    records: List[Mapping[str, Any]],
    include_security: bool,
) -> Dict[str, object]:
    utility_values = [bool(record.get("utility")) for record in records if record.get("utility") is not None]
    security_values = [bool(record.get("security")) for record in records if record.get("security") is not None]
    agentdojo_security_rate = rate(security_values) if include_security else ""
    # AgentDojo injection tasks often return True when the injection goal was
    # achieved. Keep the raw field and expose explicit names to avoid inverting
    # the security semantics in downstream result tables.
    attack_success_rate = agentdojo_security_rate if include_security else ""
    defense_success_rate = (1.0 - agentdojo_security_rate) if include_security else ""
    return {
        "run_type": run_type,
        "suite": suite,
        "cases": len(records),
        "utility_rate": rate(utility_values),
        "agentdojo_security_rate": agentdojo_security_rate,
        "attack_success_rate": attack_success_rate,
        "defense_success_rate": defense_success_rate,
    }


def rate(values: List[bool]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
