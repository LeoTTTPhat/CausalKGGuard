#!/usr/bin/env python3
"""Runtime of in-loop decisions on real AgentDojo traces and on synthetic long traces.

For every tool call of every recorded trace, the monitor input that the in-loop
executor would receive (the trace prefix ending with the candidate call) is
rebuilt and the decision is timed:
  R1: graph over the prefix with prefix dependencies; every event is scored and
      the maximum is compared with theta (behaviour of the R1 executor);
  R2: graph over the prefix with provenance dependencies; only the candidate is
      scored (attribution rule).
Timings include trace parsing, lexical detection, graph construction, feature
extraction, counterfactual scoring and the decision rule. They exclude the LLM
and the optional neural detector, whose latency is reported separately.

Writes results/runtime/<model>.json and results/runtime/scaling.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np

import r2lib
from causalkgguard import Event, TraceCase, graph_from_trace


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def summarize(xs):
    return {"n": len(xs), "mean": float(np.mean(xs)), "median": pct(xs, 50), "p95": pct(xs, 95),
            "p99": pct(xs, 99), "max": float(np.max(xs))}


def real_traces(model: str, limit: int | None):
    cases = [r2lib.reannotate(c) for c in r2lib.load_cases(r2lib.model_files(model))]
    if limit:
        cases = cases[:limit]
    g1 = r2lib.make_guard("prefix")
    g2 = r2lib.make_guard("provenance", decision_rule="attribution", lift_threshold=0.2)
    lat1, lat2, edges1, edges2, plen = [], [], [], [], []
    n_events = [len(c["events"]) for c in cases]
    n_tools = [sum(e["role"] == "action" for e in c["events"]) for c in cases]
    for c in cases:
        for i, e in enumerate(c["events"]):
            if e["role"] != "action":
                continue
            pref = dict(c)
            pref["events"] = c["events"][: i + 1]
            t0 = time.perf_counter()
            tc = TraceCase.from_dict(pref)
            fs = g1.evaluate(tc)
            _ = max((f.risk_score for f in fs), default=0.0) >= 0.60
            lat1.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            tc = TraceCase.from_dict(pref)
            f2 = g2.decide_candidate(tc)
            _ = bool(f2 and f2.evidence.get("intercept"))
            lat2.append((time.perf_counter() - t0) * 1000)
            plen.append(i + 1)
            if len(plen) % 7 == 0:  # graph sizes on a subsample
                edges1.append(len(graph_from_trace(tc, "prefix").edges))
                edges2.append(len(graph_from_trace(tc, "provenance").edges))
    return {
        "traces": len(cases),
        "events_per_trace": summarize(n_events),
        "tool_calls_per_trace": summarize(n_tools),
        "prefix_length_at_decision": summarize(plen),
        "r1_decision_ms": summarize(lat1),
        "r2_decision_ms": summarize(lat2),
        "r1_edges_sample": summarize(edges1),
        "r2_edges_sample": summarize(edges2),
    }


def synthetic_trace(n: int) -> TraceCase:
    events = [Event("e1", "system", "You are a helpful assistant.", trusted=True),
              Event("e2", "user", "Summarise the customer records and e-mail the summary to manager@corp.example.", trusted=True)]
    for i in range(3, n + 1):
        k = i % 4
        if k == 0:
            events.append(Event(f"e{i}", "tool", f"record {i}: customer{i}@example.com tier gold balance {1000 + i}",
                                tool="crm.read", trusted=False, permissions=["crm.read"], sensitivity=1))
        elif k == 1:
            events.append(Event(f"e{i}", "output", f"checkpoint {i}", trusted=True))
        else:
            events.append(Event(f"e{i}", "action", json.dumps({"record_id": f"R{i - 1}"}), tool="crm.read",
                                trusted=True, permissions=["crm.read"], metadata={"args": {"record_id": f"R{i - 1}"}}))
    events.append(Event(f"e{n + 1}", "action", json.dumps({"to": "manager@corp.example", "body": "summary"}),
                        tool="email.send", sink="email:external", trusted=True, permissions=["email.send"],
                        metadata={"args": {"to": "manager@corp.example", "body": "summary"}}))
    return TraceCase(f"synthetic_{n}", "benign", "summarise and e-mail", events,
                     required_permissions=["crm.read", "email.send"], metadata={"allowed_sinks": [], "allow_sensitive_external": True})


def scaling(repeats: int):
    g1 = r2lib.make_guard("prefix")
    g2 = r2lib.make_guard("provenance", decision_rule="attribution", lift_threshold=0.2)
    rows = []
    for n in (10, 50, 100, 500, 1000, 2000):
        tc = synthetic_trace(n)
        e1 = len(graph_from_trace(tc, "prefix").edges)
        e2 = len(graph_from_trace(tc, "provenance").edges)
        t1, t2 = [], []
        for _ in range(repeats):
            t0 = time.perf_counter(); fs = g1.evaluate(tc); _ = max(f.risk_score for f in fs); t1.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter(); g2.decide_candidate(tc); t2.append((time.perf_counter() - t0) * 1000)
        rows.append({"events": n + 1, "prefix_edges": e1, "provenance_edges": e2,
                     "r1_decision_ms": statistics.median(t1), "r2_decision_ms": statistics.median(t2)})
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor()}
    out = r2lib.RESULTS / "runtime"
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_real:
        res = real_traces(args.model, args.limit)
        res["env"] = env
        (out / f"{args.model}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "env"}, indent=1))
    sc = scaling(args.repeats)
    (out / "scaling.json").write_text(json.dumps({"rows": sc, "env": env}, indent=1))


if __name__ == "__main__":
    main()
