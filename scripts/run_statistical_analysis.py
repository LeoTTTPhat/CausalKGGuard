#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired tests for trace-level predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import CausalKGGuard, TraceCase
from causalkgguard.baselines import baseline_predictors


DATA_PATH = ROOT / "data" / "synthetic" / "trace_cases_extended.jsonl"
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
    return {"precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fpr, "attack_success_rate": asr}


def bootstrap_ci(
    rows: List[Tuple[bool, bool]],
    metric_name: str,
    samples: int,
    rng: random.Random,
) -> Tuple[float, float]:
    values: List[float] = []
    n = len(rows)
    for _ in range(samples):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        values.append(metrics(sample)[metric_name])
    values.sort()
    low_index = max(0, int(0.025 * samples) - 1)
    high_index = min(samples - 1, int(0.975 * samples))
    return values[low_index], values[high_index]


def mcnemar(reference: List[bool], contender: List[bool], truth: List[bool]) -> Dict[str, float]:
    ref_correct = [pred == label for pred, label in zip(reference, truth)]
    contender_correct = [pred == label for pred, label in zip(contender, truth)]
    b = sum(ref and not other for ref, other in zip(ref_correct, contender_correct))
    c = sum((not ref) and other for ref, other in zip(ref_correct, contender_correct))
    n = b + c
    if n == 0:
        p_value = 1.0
    elif n <= 200:
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
        p_value = min(1.0, 2 * tail)
    else:
        z = (abs(b - c) - 1) / math.sqrt(n)
        p_value = math.erfc(z / math.sqrt(2))
    return {"b_ref_only_correct": b, "c_contender_only_correct": c, "mcnemar_p": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="statistical")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data_paths = args.data or [DATA_PATH]
    cases = load_cases(data_paths)
    truth = [case.label == "attack" for case in cases]
    predictors: Dict[str, Callable[[TraceCase], bool]] = baseline_predictors()
    predictors["causal_kgguard"] = CausalKGGuard(
        threshold=args.threshold,
        policy_path=args.policy_config,
    ).predict_attack
    predictions = {name: [predictor(case) for case in cases] for name, predictor in predictors.items()}
    rng = random.Random(args.seed)

    metric_rows: List[Dict[str, object]] = []
    for method, preds in predictions.items():
        pairs = list(zip(truth, preds))
        point = metrics(pairs)
        row: Dict[str, object] = {"method": method}
        for metric_name, value in point.items():
            low, high = bootstrap_ci(pairs, metric_name, args.bootstrap_samples, rng)
            row[metric_name] = value
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        metric_rows.append(row)

    test_rows: List[Dict[str, object]] = []
    reference = predictions["causal_kgguard"]
    for method, preds in predictions.items():
        if method == "causal_kgguard":
            continue
        test_rows.append({"reference": "causal_kgguard", "contender": method, **mcnemar(reference, preds, truth)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_bootstrap.csv", metric_rows)
    write_csv(args.out_dir / f"{args.prefix}_mcnemar.csv", test_rows)
    write_report(args.out_dir / f"{args.prefix}_report.md", metric_rows, test_rows)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, metrics_rows: List[Dict[str, object]], test_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Statistical Analysis",
        "",
        "## Bootstrap Metrics",
        "",
        "| Method | F1 | F1 95% CI | FPR | FPR 95% CI | ASR | ASR 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['method']} | {row['f1']:.3f} | [{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}] | "
            f"{row['false_positive_rate']:.3f} | [{row['false_positive_rate_ci_low']:.3f}, {row['false_positive_rate_ci_high']:.3f}] | "
            f"{row['attack_success_rate']:.3f} | [{row['attack_success_rate_ci_low']:.3f}, {row['attack_success_rate_ci_high']:.3f}] |"
        )
    lines.extend(["", "## McNemar Tests", "", "| Reference | Contender | b | c | p |", "|---|---|---:|---:|---:|"])
    for row in test_rows:
        lines.append(
            f"| {row['reference']} | {row['contender']} | {row['b_ref_only_correct']} | "
            f"{row['c_contender_only_correct']} | {row['mcnemar_p']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
