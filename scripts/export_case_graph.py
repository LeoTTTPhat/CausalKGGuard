#!/usr/bin/env python3
"""Export graph, evidence paths, and quarantine plan for one trace case."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import CausalKGGuard, TraceCase, graph_from_trace, graph_to_dot, quarantine_plans
from causalkgguard.explanations import highlighted_elements


def load_case(path: Path, case_id: str) -> TraceCase:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            if data["case_id"] == case_id:
                return TraceCase.from_dict(data)
    raise SystemExit(f"Case not found: {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic" / "trace_cases_extended.jsonl")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "graphs")
    parser.add_argument("--policy-config", type=Path, default=None)
    args = parser.parse_args()

    trace = load_case(args.data, args.case_id)
    guard = CausalKGGuard(policy_path=args.policy_config)
    graph = graph_from_trace(trace)
    findings = guard.evaluate(trace)
    paths = []
    for finding in findings:
        paths.extend(finding.evidence.get("evidence_paths", []))
    highlighted_nodes, highlighted_edges = highlighted_elements(paths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = args.out_dir / f"{trace.case_id}.dot"
    json_path = args.out_dir / f"{trace.case_id}_evidence.json"
    dot_path.write_text(graph_to_dot(graph, highlighted_nodes, highlighted_edges), encoding="utf-8")

    payload = {
        "case_id": trace.case_id,
        "findings": [
            {
                "event_id": finding.event_id,
                "risk_score": finding.risk_score,
                "causal_lift": finding.causal_lift,
                "decision": finding.decision,
                "severity": finding.severity,
                "reasons": finding.reasons,
                "evidence": finding.evidence,
            }
            for finding in findings
        ],
        "quarantine_plans": [asdict(plan) for plan in quarantine_plans(trace, findings)],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {dot_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
