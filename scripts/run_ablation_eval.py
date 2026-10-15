#!/usr/bin/env python3
"""Run component ablations for CausalKGGuard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import CausalKGGuard, TraceCase
from causalkgguard.baselines import flow_policy, text_only_detector


DATA_PATH = ROOT / "data" / "synthetic" / "trace_cases_extended.jsonl"
OUT_DIR = ROOT / "results" / "tables"

VARIANT_NAMES = [
    "full_causal_kgguard",
    "no_semantic_detector",
    "no_sensitive_flow",
    "no_untrusted_sensitive_context",
    "no_permission_policy",
    "no_system_prompt_policy",
    "no_counterfactual_scoring",
    "text_only_no_graph",
    "flow_only_policy",
]


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
        "--data",
        type=Path,
        action="append",
        default=None,
        help="TraceCase JSONL input. Pass multiple times to combine datasets.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="ablation")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel worker processes for variant-level ablation.",
    )
    args = parser.parse_args()

    data_paths = args.data if args.data is not None else [DATA_PATH]
    cases = load_cases(data_paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    by_attack_rows: List[Dict[str, object]] = []
    tasks = [
        (
            variant,
            args.threshold,
            str(args.policy_config) if args.policy_config is not None else None,
            cases,
        )
        for variant in VARIANT_NAMES
    ]
    if args.jobs <= 1:
        results = [_evaluate_variant(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(_evaluate_variant, tasks))
    results_by_variant = {row["variant"]: (row, attack_rows) for row, attack_rows in results}
    for variant in VARIANT_NAMES:
        row, attack_rows = results_by_variant[variant]
        rows.append(row)
        by_attack_rows.extend(attack_rows)

    write_csv(args.out_dir / f"{args.prefix}_summary.csv", rows)
    write_csv(args.out_dir / f"{args.prefix}_by_attack.csv", by_attack_rows)
    write_report(args.out_dir / f"{args.prefix}_report.md", rows, by_attack_rows, cases, data_paths)


def _evaluate_variant(
    task: Tuple[str, float, str | None, List[TraceCase]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    variant, threshold, policy_config, cases = task
    policy_path = Path(policy_config) if policy_config is not None else None
    pairs: List[Tuple[bool, bool]] = []
    by_attack = defaultdict(list)

    if variant.startswith("full") or variant.startswith("no_"):
        guard = _guard_for_variant(variant, threshold, policy_path)
        lifts = []
        violation_counts = []
        for case in cases:
            findings = guard.evaluate(case)
            pred = any(finding.risk_score >= threshold for finding in findings)
            truth = case.label == "attack"
            attack_type = str(case.metadata.get("attack_type") or case.attack_type or "none")
            pairs.append((truth, pred))
            by_attack[attack_type].append((truth, pred))
            lifts.append(max((finding.causal_lift for finding in findings), default=0.0))
            violation_counts.append(
                max(
                    (
                        len(finding.evidence.get("policy_violations", []))
                        for finding in findings
                    ),
                    default=0,
                )
            )
        row: Dict[str, object] = {"variant": variant, **metrics(pairs)}
        row["mean_max_causal_lift"] = sum(lifts) / len(lifts) if lifts else 0.0
        row["mean_policy_violations"] = sum(violation_counts) / len(violation_counts) if violation_counts else 0.0
    else:
        predictor = _baseline_for_variant(variant)
        for case in cases:
            pred = predictor(case)
            truth = case.label == "attack"
            attack_type = str(case.metadata.get("attack_type") or case.attack_type or "none")
            pairs.append((truth, pred))
            by_attack[attack_type].append((truth, pred))
        row = {
            "variant": variant,
            **metrics(pairs),
            "mean_max_causal_lift": 0.0,
            "mean_policy_violations": 0.0,
        }

    by_attack_rows = [
        {
            "variant": variant,
            "attack_type": attack_type,
            **metrics(attack_pairs),
        }
        for attack_type, attack_pairs in sorted(by_attack.items())
    ]
    return row, by_attack_rows


def _guard_for_variant(variant: str, threshold: float, policy_config: Path | None) -> CausalKGGuard:
    disabled = {
        "no_semantic_detector": ["semantic_detector"],
        "no_sensitive_flow": ["sensitive_flow"],
        "no_untrusted_sensitive_context": ["untrusted_sensitive_context"],
        "no_permission_policy": ["permission_policy"],
        "no_system_prompt_policy": ["system_prompt_policy"],
        "no_counterfactual_scoring": ["counterfactual_scoring"],
    }.get(variant, [])
    return CausalKGGuard(threshold=threshold, disabled_components=disabled, policy_path=policy_config)


def _baseline_for_variant(variant: str) -> Callable[[TraceCase], bool]:
    baselines: Dict[str, Callable[[TraceCase], bool]] = {
        "text_only_no_graph": text_only_detector,
        "flow_only_policy": flow_policy,
    }
    return baselines[variant]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: List[Dict[str, object]],
    by_attack_rows: List[Dict[str, object]],
    cases: List[TraceCase],
    data_paths: List[Path],
) -> None:
    labels = Counter(case.label for case in cases)
    attack_types = Counter(str(case.metadata.get("attack_type") or case.attack_type or "none") for case in cases)
    lines = [
        "# Ablation Report",
        "",
        f"- Cases: {len(cases)}",
        f"- Labels: `{json.dumps(dict(sorted(labels.items())), sort_keys=True)}`",
        f"- Attack types: `{json.dumps(dict(sorted(attack_types.items())), sort_keys=True)}`",
        f"- Inputs: `{json.dumps([str(path) for path in data_paths])}`",
        "",
        "| Variant | Precision | Recall | F1 | FPR | ASR | Mean Lift | Mean Violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['false_positive_rate']:.3f} | "
            f"{row['attack_success_rate']:.3f} | {row['mean_max_causal_lift']:.3f} | "
            f"{row['mean_policy_violations']:.3f} |"
        )
    lines.extend(["", "## By Attack", "", "| Variant | Attack | Recall | FPR | ASR | Total |", "|---|---|---:|---:|---:|---:|"])
    for row in by_attack_rows:
        total = int(row["tp"]) + int(row["fp"]) + int(row["tn"]) + int(row["fn"])
        lines.append(
            f"| {row['variant']} | {row['attack_type']} | {row['recall']:.3f} | "
            f"{row['false_positive_rate']:.3f} | {row['attack_success_rate']:.3f} | {total} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
