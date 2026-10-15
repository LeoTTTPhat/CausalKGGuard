#!/usr/bin/env python3
"""Evaluate CausalKGGuard on AgentDojo-converted TraceCase JSONL files."""

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
from causalkgguard.adapters import agentdojo_available
from causalkgguard.baselines import baseline_predictors


OUT_DIR = ROOT / "results" / "tables"


def load_cases(paths: Iterable[Path]) -> List[TraceCase]:
    cases: List[TraceCase] = []
    for path in paths:
        cases.extend(
            TraceCase.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
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
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    asr = fn / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "attack_success_rate": asr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        required=True,
        action="append",
        help="AgentDojo traces converted to TraceCase JSONL. Pass multiple times to combine datasets.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="agentdojo_eval")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--require-agentdojo-installed", action="store_true")
    args = parser.parse_args()

    if args.require_agentdojo_installed and not agentdojo_available():
        raise SystemExit("AgentDojo is not installed. Install it on the evaluation machine first.")

    cases = load_cases(args.trace_jsonl)
    guard = CausalKGGuard(threshold=args.threshold, policy_path=args.policy_config)
    predictors: Dict[str, Callable[[TraceCase], bool]] = baseline_predictors()
    predictors["causal_kgguard"] = guard.predict_attack
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    by_suite_rows: List[Dict[str, object]] = []
    by_attack_rows: List[Dict[str, object]] = []
    case_rows: List[Dict[str, object]] = []
    by_suite = defaultdict(lambda: defaultdict(list))
    by_attack = defaultdict(lambda: defaultdict(list))

    for method, predictor in predictors.items():
        pairs: List[Tuple[bool, bool]] = []
        for case in cases:
            truth = case.label == "attack"
            pred = predictor(case)
            suite = str(case.metadata.get("suite", "unknown"))
            attack_type = str(case.metadata.get("attack_type") or case.attack_type or "none")
            pairs.append((truth, pred))
            by_suite[method][suite].append((truth, pred))
            by_attack[method][attack_type].append((truth, pred))
            case_rows.append(
                {
                    "case_id": case.case_id,
                    "suite": suite,
                    "agentdojo_attack_type": attack_type,
                    "label": case.label,
                    "attack_type": case.attack_type,
                    "method": method,
                    "prediction": int(pred),
                    "truth": int(truth),
                }
            )
        summary_rows.append({"method": method, **metrics(pairs)})

    for method, suite_map in by_suite.items():
        for suite, pairs in sorted(suite_map.items()):
            by_suite_rows.append({"method": method, "suite": suite, **metrics(pairs)})
    for method, attack_map in by_attack.items():
        for attack_type, pairs in sorted(attack_map.items()):
            by_attack_rows.append({"method": method, "agentdojo_attack_type": attack_type, **metrics(pairs)})

    write_csv(args.out_dir / f"{args.prefix}_summary.csv", summary_rows)
    write_csv(args.out_dir / f"{args.prefix}_by_suite.csv", by_suite_rows)
    write_csv(args.out_dir / f"{args.prefix}_by_attack.csv", by_attack_rows)
    write_csv(args.out_dir / f"{args.prefix}_cases.csv", case_rows)
    write_report(args.out_dir / f"{args.prefix}_report.md", cases, summary_rows, by_suite_rows, by_attack_rows)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    cases: List[TraceCase],
    summary_rows: List[Dict[str, object]],
    by_suite_rows: List[Dict[str, object]],
    by_attack_rows: List[Dict[str, object]],
) -> None:
    suites = Counter(str(case.metadata.get("suite", "unknown")) for case in cases)
    lines = [
        "# AgentDojo Evaluation Report",
        "",
        f"- Cases: {len(cases)}",
        f"- Suites: `{json.dumps(dict(sorted(suites.items())), sort_keys=True)}`",
        "",
        "## Summary",
        "",
        "| Method | Precision | Recall | F1 | FPR | ASR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['false_positive_rate']:.3f} | {row['attack_success_rate']:.3f} |"
        )
    lines.extend(["", "## By Suite", "", "| Method | Suite | Recall | FPR | ASR |", "|---|---|---:|---:|---:|"])
    for row in by_suite_rows:
        lines.append(
            f"| {row['method']} | {row['suite']} | {row['recall']:.3f} | "
            f"{row['false_positive_rate']:.3f} | {row['attack_success_rate']:.3f} |"
        )
    lines.extend(["", "## By Attack", "", "| Method | Attack | Recall | FPR | ASR |", "|---|---|---:|---:|---:|"])
    for row in by_attack_rows:
        lines.append(
            f"| {row['method']} | {row['agentdojo_attack_type']} | {row['recall']:.3f} | "
            f"{row['false_positive_rate']:.3f} | {row['attack_success_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
