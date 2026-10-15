#!/usr/bin/env python3
"""Analyze CausalKGGuard false negatives by suite, attack, tool, and feature gap."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import CausalKGGuard, TraceCase


OUT_DIR = ROOT / "results" / "tables"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="false_negative_analysis")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--policy-config", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.data)
    guard = CausalKGGuard(threshold=args.threshold, finding_threshold=0.0, policy_path=args.policy_config)

    fn_rows: List[Dict[str, object]] = []
    summary_counter: Counter[tuple[str, str, str]] = Counter()
    tool_counter: Counter[tuple[str, str, str]] = Counter()

    for case in cases:
        if case.label != "attack":
            continue
        findings = guard.evaluate(case)
        pred = any(finding.risk_score >= args.threshold for finding in findings)
        if pred:
            continue
        best = max(findings, key=lambda finding: finding.risk_score, default=None)
        features = dict(best.evidence.get("features", {}) if best is not None else {})
        source_events = dict(best.evidence.get("source_events", {}) if best is not None else {})
        taxonomy = classify_false_negative(features, best.risk_score if best is not None else 0.0, args.threshold)
        suite = str(case.metadata.get("suite", "unknown"))
        attack = str(case.metadata.get("attack_type") or case.attack_type or "unknown")
        tool = str(source_events.get("tool") or "")
        sink = str(source_events.get("sink") or "")
        row: Dict[str, object] = {
            "case_id": case.case_id,
            "suite": suite,
            "agentdojo_attack_type": attack,
            "user_task_id": case.metadata.get("user_task_id", ""),
            "injection_task_id": case.metadata.get("injection_task_id", ""),
            "taxonomy": taxonomy,
            "max_risk_score": best.risk_score if best is not None else 0.0,
            "causal_lift": best.causal_lift if best is not None else 0.0,
            "tool": tool,
            "sink": sink,
            "has_untrusted_injection": features.get("has_untrusted_injection", 0.0),
            "has_untrusted_exfiltration_request": features.get("has_untrusted_exfiltration_request", 0.0),
            "sensitive_upstream": features.get("sensitive_upstream", 0.0),
            "untrusted_sensitive_context": features.get("untrusted_sensitive_context", 0.0),
            "high_impact_tool": features.get("high_impact_tool", 0.0),
            "external_sink": features.get("external_sink", 0.0),
            "unauthorized_external_sink": features.get("unauthorized_external_sink", 0.0),
            "excessive_agency": features.get("excessive_agency", 0.0),
            "reasons": " | ".join(best.reasons if best is not None else []),
        }
        fn_rows.append(row)
        summary_counter[(suite, attack, taxonomy)] += 1
        tool_counter[(suite, tool, taxonomy)] += 1

    summary_rows = [
        {"suite": suite, "agentdojo_attack_type": attack, "taxonomy": taxonomy, "false_negatives": count}
        for (suite, attack, taxonomy), count in sorted(summary_counter.items())
    ]
    tool_rows = [
        {"suite": suite, "tool": tool, "taxonomy": taxonomy, "false_negatives": count}
        for (suite, tool, taxonomy), count in sorted(tool_counter.items(), key=lambda item: (-item[1], item[0]))
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_cases.csv", fn_rows)
    write_csv(args.out_dir / f"{args.prefix}_summary.csv", summary_rows)
    write_csv(args.out_dir / f"{args.prefix}_tools.csv", tool_rows)
    write_report(args.out_dir / f"{args.prefix}_report.md", fn_rows, summary_rows, tool_rows)


def load_cases(paths: Iterable[Path]) -> List[TraceCase]:
    cases: List[TraceCase] = []
    for path in paths:
        cases.extend(
            TraceCase.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return cases


def classify_false_negative(features: Dict[str, float], score: float, threshold: float) -> str:
    if not features:
        return "no_action_or_output_evidence"
    if float(features.get("has_untrusted_injection", 0.0)) <= 0.0:
        return "missing_untrusted_instruction_signal"
    if (
        float(features.get("sensitive_upstream", 0.0)) <= 0.0
        and float(features.get("untrusted_sensitive_context", 0.0)) <= 0.0
    ):
        return "missing_sensitive_context"
    if (
        float(features.get("high_impact_tool", 0.0)) <= 0.0
        and float(features.get("external_sink", 0.0)) <= 0.0
        and float(features.get("excessive_agency", 0.0)) <= 0.0
    ):
        return "no_high_impact_or_external_action"
    if (
        float(features.get("unauthorized_external_sink", 0.0)) <= 0.0
        and float(features.get("excessive_agency", 0.0)) <= 0.0
        and float(features.get("high_impact_tool", 0.0)) > 0.0
    ):
        return "authorized_or_required_high_impact_tool"
    if score >= threshold - 0.10:
        return "near_threshold"
    return "low_score_despite_risk_signals"


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    fn_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    tool_rows: List[Dict[str, object]],
) -> None:
    lines = [
        "# False Negative Analysis",
        "",
        f"- False negatives: {len(fn_rows)}",
        "",
        "## By Suite, Attack, And Taxonomy",
        "",
        "| Suite | Attack | Taxonomy | FNs |",
        "|---|---|---|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['suite']} | {row['agentdojo_attack_type']} | {row['taxonomy']} | "
            f"{row['false_negatives']} |"
        )
    lines.extend(["", "## Top Tool/Taxonomy Buckets", "", "| Suite | Tool | Taxonomy | FNs |", "|---|---|---|---:|"])
    for row in tool_rows[:30]:
        lines.append(f"| {row['suite']} | {row['tool']} | {row['taxonomy']} | {row['false_negatives']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

