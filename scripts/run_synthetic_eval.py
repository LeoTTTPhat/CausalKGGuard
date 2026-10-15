#!/usr/bin/env python3
"""Run synthetic evaluations for CausalKGGuard and trace-level baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import CausalKGGuard, TraceCase
from causalkgguard.baselines import baseline_predictors


DATA_PATH = ROOT / "data" / "synthetic" / "trace_cases.jsonl"
OUT_DIR = ROOT / "results" / "tables"


def load_cases(path: Path) -> List[TraceCase]:
    cases: List[TraceCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(TraceCase.from_dict(json.loads(line)))
    return cases


def metrics(rows: Iterable[Tuple[bool, bool]]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for truth, pred in rows:
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1

    positives = tp + fn
    negatives = fp + tn
    total = positives + negatives
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    fpr = fp / negatives if negatives else 0.0
    attack_success_rate = fn / positives if positives else 0.0
    intervention_rate = (tp + fp) / total if total else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "attack_success_rate": attack_success_rate,
        "intervention_rate": intervention_rate,
    }


def dataset_stats(cases: List[TraceCase]) -> Dict[str, object]:
    labels = Counter(case.label for case in cases)
    attack_types = Counter(case.attack_type for case in cases)
    events = sum(len(case.events) for case in cases)
    return {
        "cases": len(cases),
        "events": events,
        "benign": labels.get("benign", 0),
        "attacks": labels.get("attack", 0),
        "attack_types": dict(sorted(attack_types.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="JSONL trace benchmark")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory for tables and report")
    parser.add_argument("--prefix", default="synthetic_eval", help="Output filename prefix")
    parser.add_argument("--threshold", type=float, default=0.60, help="CausalKGGuard attack threshold")
    parser.add_argument("--policy-config", type=Path, default=None, help="Optional JSON policy registry")
    args = parser.parse_args()

    cases = load_cases(args.data)
    guard = CausalKGGuard(threshold=args.threshold, policy_path=args.policy_config)
    predictors: Dict[str, Callable[[TraceCase], bool]] = baseline_predictors()
    predictors["causal_kgguard"] = guard.predict_attack

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_rows: List[Dict[str, object]] = []
    finding_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    by_attack_type_rows: List[Dict[str, object]] = []

    predictions_by_method: Dict[str, List[Tuple[bool, bool]]] = {}
    predictions_by_type: Dict[str, Dict[str, List[Tuple[bool, bool]]]] = defaultdict(lambda: defaultdict(list))

    for method, predictor in predictors.items():
        pairs: List[Tuple[bool, bool]] = []
        for case in cases:
            truth = case.label == "attack"
            pred = predictor(case)
            pairs.append((truth, pred))
            predictions_by_type[method][case.attack_type].append((truth, pred))
            case_rows.append(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "attack_type": case.attack_type,
                    "method": method,
                    "prediction": int(pred),
                    "truth": int(truth),
                }
            )
        predictions_by_method[method] = pairs
        summary_rows.append({"method": method, **metrics(pairs)})

    for method, attack_type_map in predictions_by_type.items():
        for attack_type, pairs in sorted(attack_type_map.items()):
            if attack_type == "none":
                continue
            by_attack_type_rows.append({"method": method, "attack_type": attack_type, **metrics(pairs)})

    for case in cases:
        findings = guard.evaluate(case)
        if not findings:
            finding_rows.append(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "attack_type": case.attack_type,
                    "event_id": "",
                    "risk_score": "0.000",
                    "causal_lift": "0.000",
                    "decision": "allow",
                    "severity": "none",
                    "policy_violations": "",
                    "top_reasons": "",
                    "source_events": "",
                    "evidence_path_count": 0,
                }
            )
            continue
        for finding in findings:
            violations = finding.evidence.get("policy_violations", [])
            finding_rows.append(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "attack_type": case.attack_type,
                    "event_id": finding.event_id,
                    "risk_score": f"{finding.risk_score:.3f}",
                    "causal_lift": f"{finding.causal_lift:.3f}",
                    "decision": finding.decision,
                    "severity": finding.severity,
                    "policy_violations": ";".join(
                        str(item.get("policy_id", "")) for item in violations if isinstance(item, dict)
                    ),
                    "top_reasons": " | ".join(finding.reasons),
                    "source_events": json.dumps(finding.evidence.get("source_events", {}), sort_keys=True),
                    "evidence_path_count": len(finding.evidence.get("evidence_paths", [])),
                }
            )

    write_csv(args.out_dir / f"{args.prefix}_summary.csv", summary_rows)
    write_csv(args.out_dir / f"{args.prefix}_cases.csv", case_rows)
    write_csv(args.out_dir / f"{args.prefix}_by_attack_type.csv", by_attack_type_rows)
    write_csv(args.out_dir / f"{args.prefix}_guard_findings.csv", finding_rows)
    write_report(
        args.out_dir / f"{args.prefix}_report.md",
        stats=dataset_stats(cases),
        summary_rows=summary_rows,
        by_attack_type_rows=by_attack_type_rows,
        finding_rows=finding_rows,
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    stats: Dict[str, object],
    summary_rows: List[Dict[str, object]],
    by_attack_type_rows: List[Dict[str, object]],
    finding_rows: List[Dict[str, object]],
) -> None:
    lines = [
        "# Synthetic Evaluation Report",
        "",
        "This report is generated from trace-level synthetic cases.",
        "",
        "## Dataset",
        "",
        f"- Cases: {stats['cases']}",
        f"- Events: {stats['events']}",
        f"- Benign: {stats['benign']}",
        f"- Attacks: {stats['attacks']}",
        f"- Attack types: `{json.dumps(stats['attack_types'], sort_keys=True)}`",
        "",
        "## Main Summary",
        "",
        "| Method | Precision | Recall | F1 | Accuracy | FPR | ASR | Intervention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['accuracy']:.3f} | {row['false_positive_rate']:.3f} | "
            f"{row['attack_success_rate']:.3f} | {row['intervention_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Recall By Attack Type",
            "",
            "| Method | Attack Type | Recall | ASR | Cases |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in by_attack_type_rows:
        total = int(row["tp"] + row["fn"])
        lines.append(
            f"| {row['method']} | {row['attack_type']} | {row['recall']:.3f} | "
            f"{row['attack_success_rate']:.3f} | {total} |"
        )

    lines.extend(
        [
            "",
            "## CausalKGGuard Findings",
            "",
            "| Case | Label | Attack Type | Event | Risk | Lift | Decision | Policies |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in finding_rows[:80]:
        lines.append(
            f"| {row['case_id']} | {row['label']} | {row['attack_type']} | {row['event_id']} | "
            f"{row['risk_score']} | {row['causal_lift']} | {row['decision']} | {row['policy_violations']} |"
        )
    if len(finding_rows) > 80:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | {len(finding_rows) - 80} more rows omitted |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
