#!/usr/bin/env python3
"""Verification checks quoted in Section 6.4; writes results/verification.json.

(1) Replay equals the in-loop decision. For a random sample of GPT-4o-mini traces (seed 11), the
    decision on every tool call computed from the full trace, as replay_actions.py does, equals the
    decision of ``decide_candidate`` on the prefix that ends with that call, which is what the in-loop
    monitor sees. Checked for the CKG-A configuration (provenance, attribution rule) and for the
    candidate-only score rule over prefix dependencies.
(2) The replay engine (this artifact) and the frozen package used in the live CKG-A runs
    (/Users/sonha/CausalKGGuard_eval/frozen_r2) take the same decision on every tool call of the
    GPT-4o-mini traces.
"""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
import time
from pathlib import Path

import r2lib

FROZEN_R2 = "/Users/sonha/CausalKGGuard_eval/frozen_r2"
ARTIFACT = str(r2lib.ART)
CKG_A = dict(dependency_mode="provenance", decision_rule="attribution", lift_threshold=0.20)


def load_package(root: str):
    for m in [m for m in sys.modules if m == "causalkgguard" or m.startswith("causalkgguard.")]:
        del sys.modules[m]
    sys.path.insert(0, root)
    mod = importlib.import_module("causalkgguard")
    sys.path.pop(0)
    return mod


def guard_of(pkg, **kw):
    return pkg.CausalKGGuard(threshold=0.60, finding_threshold=0.0, policy_path=str(r2lib.POLICY), **kw)


def prefix_case(case: dict, i: int) -> dict:
    c = copy.copy(case)
    c["events"] = case["events"][: i + 1]
    return c


def candidate_decision(pkg, guard, case: dict, i: int):
    f = guard.decide_candidate(pkg.TraceCase.from_dict(prefix_case(case, i)))
    return bool(f and f.evidence.get("intercept")), (round(f.risk_score, 9) if f else None)


def main() -> None:
    cases = [r2lib.reannotate(c, "gt") for c in r2lib.load_cases(r2lib.model_files("gpt-4o-mini"))]
    out = {}

    # (1) replay (full-trace evaluation) versus in-loop candidate decision on the prefix
    pkg = load_package(ARTIFACT)
    sample = random.Random(11).sample(cases, 300)
    for name, kw in (("CKG-A", CKG_A), ("prefix-score-candidate", dict(dependency_mode="prefix", decision_rule="score"))):
        guard = guard_of(pkg, **kw)
        n = mism = 0
        for c in sample:
            tc = pkg.TraceCase.from_dict(c)
            full = {f.event_id: f for f in guard.evaluate(tc)}
            for i, ev in enumerate(c["events"]):
                if ev["role"] != "action":
                    continue
                f = full.get(ev["event_id"])
                rep = (bool(f and f.evidence.get("intercept")), round(f.risk_score, 9) if f else None)
                n += 1
                mism += rep != candidate_decision(pkg, guard, c, i)
        out[f"replay_vs_inloop|{name}"] = {"traces": len(sample), "tool_calls": n, "mismatches": mism}
        print(name, "replay vs in-loop:", n, "calls,", mism, "mismatches", flush=True)

    # (2) artifact engine versus the frozen package of the live CKG-A runs, every tool call
    decisions = {}
    for label, root in (("artifact", ARTIFACT), ("frozen_live", FROZEN_R2)):
        pkg = load_package(root)
        guard = guard_of(pkg, **CKG_A)  # candidate scope: decide_candidate on each prefix, as the live executor
        t0 = time.time()
        res = []
        for c in cases:
            for i, ev in enumerate(c["events"]):
                if ev["role"] == "action":
                    res.append(candidate_decision(pkg, guard, c, i)[0])
        decisions[label] = res
        print(label, len(res), "tool calls in", round(time.time() - t0, 1), "s", flush=True)
    a, b = decisions["artifact"], decisions["frozen_live"]
    out["artifact_vs_frozen_live_CKG-A"] = {"tool_calls": len(a), "disagreements": sum(x != y for x, y in zip(a, b)),
                                            "intercepted_artifact": sum(a), "intercepted_frozen": sum(b)}
    print(out["artifact_vs_frozen_live_CKG-A"])
    (r2lib.RESULTS / "verification.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
