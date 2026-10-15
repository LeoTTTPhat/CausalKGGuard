#!/usr/bin/env python3
"""Measure graph construction and decision latency."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard import CausalKGGuard, Event, TraceCase, graph_from_trace


OUT_DIR = ROOT / "results" / "tables"


def make_trace(length: int) -> TraceCase:
    events: List[Event] = [
        Event(
            event_id="e1",
            role="user",
            content="Summarize internal customer status without sending anything.",
            trusted=True,
        )
    ]
    for index in range(2, length + 1):
        if index % 4 == 0:
            events.append(
                Event(
                    event_id=f"e{index}",
                    role="tool",
                    content=f"Internal note {index}. Contact: user{index}@example.com. Tier: gold.",
                    trusted=False,
                    sensitivity=1,
                )
            )
        elif index % 4 == 1:
            events.append(
                Event(
                    event_id=f"e{index}",
                    role="output",
                    content=f"Internal summary checkpoint {index}.",
                    trusted=True,
                )
            )
        else:
            events.append(
                Event(
                    event_id=f"e{index}",
                    role="action",
                    content=f"Read internal record {index}.",
                    trusted=True,
                    tool="crm.read",
                    permissions=["crm.read"],
                )
            )
    return TraceCase(
        case_id=f"runtime_trace_{length}",
        label="benign",
        user_goal="Summarize internal customer status without sending anything.",
        required_permissions=["crm.read"],
        events=events,
    )


def measure(length: int, repeats: int) -> Dict[str, object]:
    trace = make_trace(length)
    guard = CausalKGGuard()
    graph_times: List[float] = []
    decision_times: List[float] = []
    peak_bytes = 0

    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        graph = graph_from_trace(trace)
        graph_times.append((time.perf_counter() - start) * 1000)
        _, peak = tracemalloc.get_traced_memory()
        peak_bytes = max(peak_bytes, peak)
        tracemalloc.stop()

        start = time.perf_counter()
        guard.evaluate(trace)
        decision_times.append((time.perf_counter() - start) * 1000)

    action_count = sum(1 for event in trace.events if event.role in {"action", "output"})
    return {
        "events": length,
        "actions_or_outputs": action_count,
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "graph_ms_mean": sum(graph_times) / repeats,
        "decision_ms_mean": sum(decision_times) / repeats,
        "decision_ms_per_action": (sum(decision_times) / repeats) / action_count if action_count else 0.0,
        "peak_kb": peak_bytes / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="runtime")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [measure(length, args.repeats) for length in (10, 50, 100, 500, 1000)]
    write_csv(args.out_dir / f"{args.prefix}_summary.csv", rows)
    write_report(args.out_dir / f"{args.prefix}_report.md", rows)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Runtime Report",
        "",
        "| Events | Actions/Outputs | Nodes | Edges | Graph ms | Decision ms | Decision ms/action | Peak KB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['events']} | {row['actions_or_outputs']} | {row['graph_nodes']} | {row['graph_edges']} | "
            f"{row['graph_ms_mean']:.3f} | {row['decision_ms_mean']:.3f} | "
            f"{row['decision_ms_per_action']:.3f} | {row['peak_kb']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
