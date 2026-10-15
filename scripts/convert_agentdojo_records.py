#!/usr/bin/env python3
"""Convert generic AgentDojo result records into CausalKGGuard TraceCase JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard.adapters import AgentDojoTraceAdapter


def iter_records(path: Path) -> Iterable[Mapping[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="AgentDojo generic result JSON or JSONL")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "agentdojo" / "agentdojo_traces.jsonl")
    args = parser.parse_args()

    adapter = AgentDojoTraceAdapter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in iter_records(args.input):
            trace = adapter.from_agentdojo_result(record)
            handle.write(json.dumps(_trace_to_dict(trace), sort_keys=True) + "\n")
            count += 1
    print(f"Wrote {count} traces to {args.output}")


def _trace_to_dict(trace: object) -> dict:
    data = asdict(trace)
    return data


if __name__ == "__main__":
    main()
