#!/usr/bin/env python3
"""Controlled and adaptive benchmarks for R1/R2 configurations, detectors and baselines.

Benchmarks are the R1 generators exported by export_synthetic_suite.py and the
R2 adaptive benchmark. A configuration "flags" a trace if any action or output
is intercepted (R1 convention). On the adaptive benchmark we report whether the
*injected* action is intercepted and whether any action of the benign twin is.
Neural configurations are evaluated only when every untrusted observation has
a cached detector score.

Writes results/synthetic/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import r2lib
from causalkgguard import TraceCase
from causalkgguard.baselines import baseline_predictors
from trace_level_eval import camel_capability

sys.path.insert(0, str(r2lib.ART / "scripts"))
import run_review_response_eval as rre  # noqa: E402  (R1 normalisation stage)

SYN = r2lib.ART / "data" / "r2_synthetic"
ADAPT = r2lib.ART / "data" / "adaptive" / "adaptive_benchmark.jsonl"


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def guards(utexts):
    g = {
        "R1": r2lib.make_guard("prefix"),
        "R1_norm": r2lib.make_guard("prefix", rre.normalized_inspect),
        "R2": r2lib.make_guard("provenance", decision_rule="attribution"),
        "R2_norm": r2lib.make_guard("provenance", rre.normalized_inspect, decision_rule="attribution"),
    }
    try:
        for t in utexts:
            r2lib.neural_score(t)
        insp = r2lib.make_neural_inspector(utexts)
        g["R1_neural"] = r2lib.make_guard("prefix", insp)
        g["R2_neural"] = r2lib.make_guard("provenance", insp, decision_rule="attribution")
        g["detector_only_neural"] = None  # handled separately
    except r2lib.MissingNeuralScore:
        pass
    return g


def flag(guard, case):
    return any(f.evidence.get("intercept") for f in guard.evaluate(TraceCase.from_dict(case)))


def prf(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    tn = sum((not p) and (not l) for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"recall": rec, "fpr": fp / (fp + tn) if fp + tn else 0.0, "precision": prec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    files = sorted(SYN.glob("*.jsonl")) + [ADAPT]
    allcases = [c for f in files for c in load(f)]
    utexts = r2lib.untrusted_texts(allcases)
    G = guards(utexts)
    B = baseline_predictors()
    B["camel_capability"] = camel_capability
    base_names = ["keyword_filter", "text_only_detector", "output_sensitive_filter", "flow_policy",
                  "permission_gate", "information_flow", "camel_capability"]

    def predictors():
        out = {n: (lambda c, f=B[n]: bool(f(TraceCase.from_dict(c)))) for n in base_names}
        for n, g in G.items():
            if g is None:
                out[n] = lambda c: any(r2lib.neural_score(e["content"]) >= 0.5 for e in c["events"]
                                       if e["role"] in {"tool", "retrieved", "memory"})
            else:
                out[n] = lambda c, g=g: flag(g, c)
        return out

    P = predictors()
    res = {}
    ext = load(SYN / "extended.jsonl")
    labels = [c["label"] == "attack" for c in ext]
    res["extended"] = {n: prf([p(c) for c in ext], labels) for n, p in P.items()}
    res["obfuscation"] = {}
    for f in sorted(SYN.glob("obfuscation_*.jsonl")):
        cs = load(f)
        res["obfuscation"][f.stem.replace("obfuscation_", "")] = {n: sum(p(c) for c in cs) / len(cs) for n, p in P.items()}
    priv = load(SYN / "privacy.jsonl")
    res["privacy"] = {n: prf([p(c) for c in priv], [c["label"] == "attack" for c in priv]) for n, p in P.items()}
    res["manifest"] = {}
    for m in ("full", "partial", "refused"):
        cs = load(SYN / f"privacy_manifest_{m}.jsonl")
        res["manifest"][m] = {n: sum(p(c) for c in cs) / len(cs) for n, p in P.items() if n.startswith("R")}
    ovr = load(SYN / "overrefusal.jsonl")
    cats = {"B1": ovr[0:20], "B2": ovr[20:40], "B3": ovr[40:60], "B4": ovr[60:80]}
    res["overrefusal"] = {n: {k: sum(p(c) for c in v) / len(v) for k, v in cats.items()} for n, p in P.items()}
    for n in res["overrefusal"]:
        res["overrefusal"][n]["overall"] = sum(P[n](c) for c in ovr) / len(ovr)
    dec = load(SYN / "decoy.jsonl")
    res["decoy"] = {}
    for n, g in G.items():
        if g is None:
            continue
        lifts = {"attack": [], "benign": []}
        for c in dec:
            fs = [f for f in g.evaluate(TraceCase.from_dict(c)) if f.evidence.get("role") == "action"]
            best = max(fs, key=lambda f: f.risk_score) if fs else None
            lifts[c["label"]].append(best.causal_lift if best else 0.0)
        pr = prf([flag(g, c) for c in dec], [c["label"] == "attack" for c in dec])
        pr.update({"mean_lift_attack": sum(lifts["attack"]) / 30, "mean_lift_decoy": sum(lifts["benign"]) / 30})
        res["decoy"][n] = pr
    # adaptive benchmark: injected action interception and benign-twin false positives
    ad = load(ADAPT)
    strategies = sorted({c["metadata"]["strategy"] for c in ad})
    res["adaptive"] = {}
    for n, p in P.items():
        g = G.get(n)
        row = {}
        for s in strategies:
            att = [c for c in ad if c["metadata"]["strategy"] == s and c["label"] == "attack"]
            ben = [c for c in ad if c["metadata"]["strategy"] == s and c["label"] == "benign"]
            if g is not None:
                tp = 0
                for c in att:
                    fs = {f.event_id: f for f in g.evaluate(TraceCase.from_dict(c))}
                    f = fs.get(c["metadata"]["injected_event"])
                    tp += bool(f and f.evidence.get("intercept"))
                fp = sum(any(f.evidence.get("intercept") for f in g.evaluate(TraceCase.from_dict(c))
                             if f.evidence.get("role") == "action") for c in ben)
            else:
                tp = sum(p(c) for c in att)
                fp = sum(p(c) for c in ben)
            row[s] = {"tp": tp, "fp": fp, "n": len(att)}
        row["all"] = {"tp": sum(v["tp"] for v in row.values()), "fp": sum(v["fp"] for v in row.values()),
                      "n": sum(v["n"] for v in row.values())}
        res["adaptive"][n] = row
    out = r2lib.RESULTS / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("extended:", {n: (round(v["recall"], 2), round(v["fpr"], 2), round(v["f1"], 2)) for n, v in res["extended"].items()})
    print("obfuscation:")
    for o, v in res["obfuscation"].items():
        print("  ", o, {n: round(x, 2) for n, x in v.items()})
    print("privacy:", {n: (round(v["recall"], 2), round(v["fpr"], 2)) for n, v in res["privacy"].items()})
    print("manifest:", res["manifest"])
    print("overrefusal:", {n: round(v["overall"], 2) for n, v in res["overrefusal"].items()})
    print("decoy:", {n: (round(v["recall"], 2), round(v["fpr"], 2), round(v["mean_lift_attack"], 2), round(v["mean_lift_decoy"], 2)) for n, v in res["decoy"].items()})
    print("adaptive:")
    for n, v in res["adaptive"].items():
        print("  ", n, {s: f"{x['tp']}/{x['n']} fp{x['fp']}" for s, x in v.items()})


if __name__ == "__main__":
    main()
