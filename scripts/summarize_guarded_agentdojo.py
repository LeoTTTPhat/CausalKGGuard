#!/usr/bin/env python3
"""Summarize end-to-end AgentDojo runs with CausalKGGuard block statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "tables"
BLOCK_RE = re.compile(r"CausalKGGuard blocked unsafe tool call\(s\): ([^.]+)\.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, action="append")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="agentdojo_guarded")
    args = parser.parse_args()

    records = read_records(args.logdir)
    rows = []
    rows.extend(summarize("all", records))
    rows.extend(summarize_by("run_type", records, run_type))
    rows.extend(summarize_by("suite", records, lambda record: str(record.get("suite_name") or "unknown")))
    rows.extend(
        summarize_by(
            "suite_run_type",
            records,
            lambda record: f"{record.get('suite_name') or 'unknown'}:{run_type(record)}",
        )
    )

    block_rows = block_tool_rows(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_summary.csv", rows)
    write_csv(args.out_dir / f"{args.prefix}_blocked_tools.csv", block_rows)
    write_report(args.out_dir / f"{args.prefix}_report.md", rows, block_rows)


def read_records(logdirs: Iterable[Path]) -> List[Mapping[str, Any]]:
    records = []
    for logdir in logdirs:
        for path in sorted(logdir.rglob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["_path"] = str(path)
            records.append(record)
    return records


def run_type(record: Mapping[str, Any]) -> str:
    if record.get("injection_task_id") is not None:
        return str(record.get("attack_type") or "attack")
    if str(record.get("user_task_id") or "").startswith("injection_task_"):
        return "no_attack_injection_utility"
    return "no_attack_user_task"


def summarize_by(group_type: str, records: List[Mapping[str, Any]], key_fn) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    rows = []
    for group, group_records in sorted(groups.items()):
        for row in summarize(group_type, group_records):
            row["group"] = group
            rows.append(row)
    return rows


def summarize(group_type: str, records: List[Mapping[str, Any]]) -> List[Dict[str, object]]:
    if not records:
        return []
    utility_values = [bool(record.get("utility")) for record in records if record.get("utility") is not None]
    attack_records = [record for record in records if record.get("injection_task_id") is not None]
    attack_success_values = [
        bool(record.get("security")) for record in attack_records if record.get("security") is not None
    ]
    secure_utility_values = [
        bool(record.get("utility")) and not bool(record.get("security"))
        for record in attack_records
        if record.get("utility") is not None and record.get("security") is not None
    ]
    blocked_counts = [count_blocks(record) for record in records]
    utility_rate = rate(utility_values)
    attack_success_rate = rate(attack_success_values) if attack_records else ""
    secure_utility_rate = rate(secure_utility_values) if attack_records else ""
    blocked_case_values = [count > 0 for count in blocked_counts]
    row = {
        "group_type": group_type,
        "group": "all",
        "cases": len(records),
        "attack_cases": len(attack_records),
        "utility_rate": utility_rate,
        "utility_ci95_low": wilson_low(sum(utility_values), len(utility_values)) if utility_values else "",
        "utility_ci95_high": wilson_high(sum(utility_values), len(utility_values)) if utility_values else "",
        "attack_success_rate": attack_success_rate,
        "attack_success_ci95_low": (
            wilson_low(sum(attack_success_values), len(attack_success_values)) if attack_success_values else ""
        ),
        "attack_success_ci95_high": (
            wilson_high(sum(attack_success_values), len(attack_success_values)) if attack_success_values else ""
        ),
        "defense_success_rate": (1.0 - attack_success_rate) if attack_records else "",
        "secure_utility_rate": secure_utility_rate,
        "secure_utility_ci95_low": (
            wilson_low(sum(secure_utility_values), len(secure_utility_values)) if secure_utility_values else ""
        ),
        "secure_utility_ci95_high": (
            wilson_high(sum(secure_utility_values), len(secure_utility_values)) if secure_utility_values else ""
        ),
        "blocked_case_rate": rate(blocked_case_values),
        "blocked_case_ci95_low": wilson_low(sum(blocked_case_values), len(blocked_case_values)),
        "blocked_case_ci95_high": wilson_high(sum(blocked_case_values), len(blocked_case_values)),
        "mean_blocks_per_case": sum(blocked_counts) / len(blocked_counts),
    }
    return [row]


def block_tool_rows(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, object]]:
    counter: Dict[tuple[str, str, str], int] = defaultdict(int)
    for record in records:
        suite = str(record.get("suite_name") or "unknown")
        rtype = run_type(record)
        for tool in blocked_tools(record):
            counter[(suite, rtype, tool)] += 1
    return [
        {"suite": suite, "run_type": rtype, "tool": tool, "blocks": count}
        for (suite, rtype, tool), count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def count_blocks(record: Mapping[str, Any]) -> int:
    return len(blocked_tools(record))


def blocked_tools(record: Mapping[str, Any]) -> List[str]:
    tools = []
    for message in record.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        content = content_to_text(message.get("content"))
        for match in BLOCK_RE.finditer(content):
            tools.extend(tool.strip() for tool in match.group(1).split(",") if tool.strip())
    return tools


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def rate(values: List[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def wilson_low(successes: int, total: int, z: float = 1.96) -> float:
    return wilson_interval(successes, total, z)[0]


def wilson_high(successes: int, total: int, z: float = 1.96) -> float:
    return wilson_interval(successes, total, z)[1]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt((p * (1.0 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: List[Dict[str, object]], block_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Guarded AgentDojo Summary",
        "",
        "| Group type | Group | Cases | Attack cases | Utility | ASR | Defense success | Secure utility | Blocked cases | Blocks/case |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group_type']} | {row['group']} | {row['cases']} | {row['attack_cases']} | "
            f"{_fmt(row['utility_rate'])} | {_fmt(row['attack_success_rate'])} | "
            f"{_fmt(row['defense_success_rate'])} | {_fmt(row['secure_utility_rate'])} | "
            f"{_fmt(row['blocked_case_rate'])} | "
            f"{_fmt(row['mean_blocks_per_case'])} |"
        )
    lines.extend(["", "## Blocked Tools", "", "| Suite | Run type | Tool | Blocks |", "|---|---|---|---:|"])
    for row in block_rows[:50]:
        lines.append(f"| {row['suite']} | {row['run_type']} | {row['tool']} | {row['blocks']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
