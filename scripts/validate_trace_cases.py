#!/usr/bin/env python3
"""Validate TraceCase JSONL files before evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import TraceCase


KNOWN_ROLES = {"system", "developer", "user", "retrieved", "tool", "memory", "action", "output", "event"}
KNOWN_LABELS = {"benign", "attack"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--agentdojo", action="store_true", help="Require AgentDojo metadata fields")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    cases, load_errors = load_cases(args.data)
    errors: List[str] = list(load_errors)
    warnings: List[str] = []
    case_ids = Counter(case.case_id for case in cases)

    for case_id, count in case_ids.items():
        if count > 1:
            errors.append(f"duplicate case_id: {case_id}")

    for case in cases:
        case_errors, case_warnings = validate_case(case, agentdojo=args.agentdojo)
        errors.extend(f"{case.case_id}: {message}" for message in case_errors)
        warnings.extend(f"{case.case_id}: {message}" for message in case_warnings)

    labels = Counter(case.label for case in cases)
    attacks = Counter(case.attack_type for case in cases)
    print(f"cases={len(cases)} labels={dict(labels)} attack_types={dict(attacks)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors or (args.strict and warnings):
        raise SystemExit(1)


def load_cases(path: Path) -> Tuple[List[TraceCase], List[str]]:
    cases: List[TraceCase] = []
    errors: List[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(TraceCase.from_dict(json.loads(line)))
        except Exception as exc:
            errors.append(f"line {line_no}: {exc}")
    return cases, errors


def validate_case(case: TraceCase, agentdojo: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if case.label not in KNOWN_LABELS:
        errors.append(f"unknown label {case.label!r}")
    if not case.events:
        errors.append("trace has no events")

    event_ids = Counter(event.event_id for event in case.events)
    for event_id, count in event_ids.items():
        if count > 1:
            errors.append(f"duplicate event_id {event_id!r}")

    for event in case.events:
        if event.role not in KNOWN_ROLES:
            warnings.append(f"unknown role {event.role!r} in {event.event_id}")
        if event.role == "action" and event.tool and not event.permissions:
            warnings.append(f"action {event.event_id} calls {event.tool} without permissions")
        if event.sensitivity < 0:
            errors.append(f"negative sensitivity in {event.event_id}")

    if case.label == "attack":
        if case.attack_type == "none":
            errors.append("attack label with attack_type=none")
        risky_events = [
            event
            for event in case.events
            if event.role in {"action", "output"} and (event.sink or event.tool or event.sensitivity > 0)
        ]
        if not risky_events:
            warnings.append("attack trace has no obvious action/output risk event")
    elif case.attack_type != "none":
        warnings.append(f"benign trace has non-none attack_type {case.attack_type!r}")

    if agentdojo:
        for field in ("suite", "task_id", "model"):
            if field not in case.metadata:
                warnings.append(f"AgentDojo metadata missing {field!r}")
    return errors, warnings


if __name__ == "__main__":
    main()
