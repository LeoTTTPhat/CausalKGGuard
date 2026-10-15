#!/usr/bin/env python3
"""Analyze counterfactual risk lift as an attribution signal."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
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
    parser.add_argument("--prefix", default="counterfactual_attribution")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--policy-config", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.data)
    guard = CausalKGGuard(threshold=args.threshold, finding_threshold=0.0, policy_path=args.policy_config)

    case_rows: List[Dict[str, object]] = []
    grouped = defaultdict(list)
    for case in cases:
        findings = guard.evaluate(case)
        best = max(findings, key=lambda finding: finding.risk_score, default=None)
        actual_score = best.risk_score if best is not None else 0.0
        counterfactual_score = (
            float(best.evidence.get("counterfactual_without_untrusted_instruction", 0.0))
            if best is not None
            else 0.0
        )
        lift = best.causal_lift if best is not None else 0.0
        truth = case.label == "attack"
        pred = actual_score >= args.threshold
        cf_pred = counterfactual_score >= args.threshold
        bucket = confusion_bucket(truth, pred)
        row: Dict[str, object] = {
            "case_id": case.case_id,
            "suite": case.metadata.get("suite", "unknown"),
            "agentdojo_attack_type": case.metadata.get("attack_type") or case.attack_type or "none",
            "label": case.label,
            "prediction": int(pred),
            "counterfactual_prediction": int(cf_pred),
            "actual_score": actual_score,
            "counterfactual_score": counterfactual_score,
            "causal_lift": lift,
            "counterfactual_crosses_threshold": int(pred and not cf_pred),
            "bucket": bucket,
        }
        case_rows.append(row)
        grouped[bucket].append(row)
        grouped[f"label:{case.label}"].append(row)
        grouped[f"attack:{row['agentdojo_attack_type']}"].append(row)

    summary_rows = [summary_row(group, rows) for group, rows in sorted(grouped.items())]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_cases.csv", case_rows)
    write_csv(args.out_dir / f"{args.prefix}_summary.csv", summary_rows)
    write_report(args.out_dir / f"{args.prefix}_report.md", summary_rows)


def load_cases(paths: Iterable[Path]) -> List[TraceCase]:
    cases: List[TraceCase] = []
    for path in paths:
        cases.extend(
            TraceCase.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return cases


def confusion_bucket(truth: bool, pred: bool) -> str:
    if truth and pred:
        return "tp"
    if truth and not pred:
        return "fn"
    if not truth and pred:
        return "fp"
    return "tn"


def summary_row(group: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    lifts = [float(row["causal_lift"]) for row in rows]
    actual_scores = [float(row["actual_score"]) for row in rows]
    cf_scores = [float(row["counterfactual_score"]) for row in rows]
    crossing = [int(row["counterfactual_crosses_threshold"]) for row in rows]
    return {
        "group": group,
        "cases": len(rows),
        "mean_actual_score": mean(actual_scores),
        "mean_counterfactual_score": mean(cf_scores),
        "mean_causal_lift": mean(lifts),
        "median_causal_lift": statistics.median(lifts) if lifts else 0.0,
        "lift_ge_0_20_rate": mean([1.0 if value >= 0.20 else 0.0 for value in lifts]),
        "counterfactual_cross_threshold_rate": mean(crossing),
    }


def mean(values: List[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Counterfactual Attribution Analysis",
        "",
        "| Group | Cases | Mean score | Mean CF score | Mean lift | Median lift | Lift >= 0.20 | CF crosses threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {row['cases']} | {row['mean_actual_score']:.3f} | "
            f"{row['mean_counterfactual_score']:.3f} | {row['mean_causal_lift']:.3f} | "
            f"{row['median_causal_lift']:.3f} | {row['lift_ge_0_20_rate']:.3f} | "
            f"{row['counterfactual_cross_threshold_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

