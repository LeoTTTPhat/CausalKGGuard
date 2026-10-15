#!/usr/bin/env python3
"""Consolidate replay-based results (RQ3, RQ5) with task-cluster bootstrap intervals.

Configurations (all at theta=0.60, delta=0.20):
  R1            prefix dependencies, score rule, trace-wide maximum (original executor)
  pre-score     prefix, score rule, candidate only
  pre-attr      prefix, attribution-gated rule, candidate only
  prov-score    provenance, score rule, candidate only
  R2            provenance, attribution-gated rule, candidate only
Writes results/replay_summary.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import analyze_replay as ar
import r2lib

CONFIGS = {
    "R1": ("prefix", "score", "sticky"),
    "pre-score": ("prefix", "score", "candidate"),
    "pre-attr": ("prefix", "attr", "candidate"),
    "prov-score": ("provenance", "score", "candidate"),
    "R2": ("provenance", "attr", "candidate"),
}
KEYS = ["succ_attack_prevention", "replay_residual_asr", "inj_interception", "attack_prevention",
        "leg_overblock_attack", "leg_overblock_attack_pre", "leg_overblock_benign", "tool_trace_fpr",
        "att_traces_with_legit_block", "spurious_attr_legit", "inj_lift_pos", "trace_recall", "trace_fpr",
        "trace_f1", "trace_recall_exposed", "succ_attack_prevention_exact", "succ_attack_premature",
        "first_int_injected", "first_int_legit", "first_int_other", "first_int_none",
        "noinj_sessions_intercepted", "leg_overblock_attack_exact"]
MODELS = ["gpt-4o-mini", "llama-3.3-70b-instruct", "gemini-2.5-flash-lite"]


# per-task count vector layout
C_SUCC, C_PREV, C_LEG, C_LEGB, C_PREVX, C_ATT, C_FILEG, C_FINONINJ, C_NOINJ, C_NOINJI, C_LEGX, C_LEGXB = range(12)


def per_task_counts(traces, rule, scope):
    """Per user task sufficient statistics for the replay measures (see the C_* layout)."""
    out = defaultdict(lambda: np.zeros(12))
    for t in traces:
        if t["label"] != "attack":
            continue
        dec = ar.decide(t["rows"], rule, 0.60, 0.20, scope)
        acts = [(r, d) for r, d in zip(t["rows"], dec) if r["role"] == "action"]
        first = next((i for i, (r, _) in enumerate(acts) if r["kind"] == "injected"), None)
        fi = next((i for i, (_, d) in enumerate(acts) if d), None)
        v = out[t["user_task"]]
        if first is not None and t["security"] == "True":
            v[C_SUCC] += 1
            v[C_PREV] += acts[first][1]
            v[C_PREVX] += fi == first
        for r, d in acts:
            if r["kind"] == "legitimate":
                v[C_LEG] += 1
                v[C_LEGB] += d
        # exact under replay: decisions up to and including the first interception
        v[C_ATT] += 1
        kind = acts[fi][0]["kind"] if fi is not None else "none"
        v[C_FILEG] += kind == "legitimate"
        v[C_FINONINJ] += kind in ("legitimate", "other")
        if first is None:
            v[C_NOINJ] += 1
            v[C_NOINJI] += fi is not None
        last = fi if fi is not None else len(acts) - 1
        for r, _ in acts[: last + 1]:
            if r["kind"] == "legitimate":
                v[C_LEGX] += 1
        v[C_LEGXB] += kind == "legitimate"
    return out


def boot_ci(counts_by_task, fn, B=10000, seed=0):
    keys = sorted(counts_by_task)
    arr = np.array([counts_by_task[k] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    vals = np.array([fn(arr[i].sum(axis=0)) for i in idx])
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_diff_ci(c1, c2, fn, B=10000, seed=1):
    keys = sorted(set(c1) & set(c2))
    a1 = np.array([c1[k] for k in keys])
    a2 = np.array([c2[k] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    vals = np.array([fn(a2[i].sum(axis=0)) - fn(a1[i].sum(axis=0)) for i in idx])
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _ratio(num, den):
    return lambda s: s[num] / s[den] if s[den] else np.nan


prevention = _ratio(C_PREV, C_SUCC)
overblock = _ratio(C_LEGB, C_LEG)
EXACT = {  # name -> (statistic, key in metrics())
    "prevention_exact": (_ratio(C_PREVX, C_SUCC), "succ_attack_prevention_exact"),
    "first_int_legit": (_ratio(C_FILEG, C_ATT), "first_int_legit"),
    "first_int_noninjected": (_ratio(C_FINONINJ, C_ATT), None),
    "noinj_intercepted": (_ratio(C_NOINJI, C_NOINJ), "noinj_sessions_intercepted"),
    "overblock_exact": (_ratio(C_LEGXB, C_LEGX), "leg_overblock_attack_exact"),
}


def main() -> None:
    res = {}
    for model in MODELS:
        for det in ("lexical", "generic", "neural"):
            try:
                pre = ar.load_replay(model, "prefix", det, "all", "gt")
                pro = ar.load_replay(model, "provenance", det, "all", "gt")
            except FileNotFoundError:
                continue
            splits = {"all": lambda t: True}
            if model == "gpt-4o-mini":
                splits.update({"calib": lambda t: ar.split_of(t["user_task"]) == "calib",
                               "test": lambda t: ar.split_of(t["user_task"]) == "test"})
            for split, keep in splits.items():
                block = {}
                counts = {}
                for name, (dep, rule, scope) in CONFIGS.items():
                    tr = [t for t in (pre if dep == "prefix" else pro) if keep(t)]
                    m = ar.metrics(tr, rule, 0.60, 0.20, scope)
                    block[name] = {k: m[k] for k in KEYS}
                    block[name]["n"] = {k[2:]: v for k, v in m.items() if k.startswith("n_")}
                    counts[name] = per_task_counts(tr, rule, scope)
                    block[name]["ci_prevention"] = boot_ci(counts[name], prevention)
                    block[name]["ci_overblock"] = boot_ci(counts[name], overblock, seed=2)
                    block[name]["first_int_noninjected"] = block[name]["first_int_legit"] + block[name]["first_int_other"]
                block["R2_minus_R1"] = {
                    "prevention_diff": block["R2"]["succ_attack_prevention"] - block["R1"]["succ_attack_prevention"],
                    "prevention_diff_ci": paired_diff_ci(counts["R1"], counts["R2"], prevention),
                    "overblock_diff": block["R2"]["leg_overblock_attack"] - block["R1"]["leg_overblock_attack"],
                    "overblock_diff_ci": paired_diff_ci(counts["R1"], counts["R2"], overblock, seed=3),
                }
                block["R2_minus_R1_exact"] = {}
                for j, (ename, (fn, _)) in enumerate(EXACT.items()):
                    a, b = fn(sum(counts["R1"].values())), fn(sum(counts["R2"].values()))
                    block["R2_minus_R1_exact"][ename] = {
                        "R1": float(a), "R2": float(b), "diff": float(b - a),
                        "diff_ci": paired_diff_ci(counts["R1"], counts["R2"], fn, seed=10 + j)}
                block["attr_minus_score_prov"] = {
                    "prevention_diff": block["R2"]["succ_attack_prevention"] - block["prov-score"]["succ_attack_prevention"],
                    "prevention_diff_ci": paired_diff_ci(counts["prov-score"], counts["R2"], prevention, seed=4),
                    "overblock_diff": block["R2"]["leg_overblock_attack"] - block["prov-score"]["leg_overblock_attack"],
                    "overblock_diff_ci": paired_diff_ci(counts["prov-score"], counts["R2"], overblock, seed=5),
                }
                res[f"{model}|{det}|{split}"] = block
    # LLM detector used alone as an in-loop gate: intercept every tool call that follows an untrusted
    # observation flagged by the detector (prefix dependencies make every earlier observation a parent)
    for model in MODELS:
        try:
            pre = ar.load_replay(model, "prefix", "neural", "all", "gt")
        except FileNotFoundError:
            continue
        c = defaultdict(float)
        for t in pre:
            acts = [r for r in t["rows"] if r["role"] == "action"]
            dec = [bool(r["feat"].get("has_untrusted_injection")) for r in acts]
            first = next((i for i, r in enumerate(acts) if r["kind"] == "injected"), None)
            fi = next((i for i, d in enumerate(dec) if d), None)
            if t["label"] != "attack":
                c["ben"] += 1
                c["ben_int"] += fi is not None
                continue
            c["att"] += 1
            if first is None:
                c["noinj"] += 1
                c["noinj_int"] += fi is not None
            elif t["security"] == "True":
                c["succ"] += 1
                c["prev"] += dec[first]
                c["prevx"] += fi == first
            for r, d in zip(acts, dec):
                if r["kind"] == "legitimate":
                    c["leg"] += 1
                    c["leg_int"] += d
            c["fi_noninj"] += fi is not None and acts[fi]["kind"] in ("legitimate", "other")
        q = lambda a, b: c[a] / c[b] if c[b] else float("nan")
        res[f"{model}|neural_detector_gate"] = {
            "succ_attack_prevention": q("prev", "succ"), "succ_attack_prevention_exact": q("prevx", "succ"),
            "leg_overblock_attack": q("leg_int", "leg"), "noinj_sessions_intercepted": q("noinj_int", "noinj"),
            "first_int_noninjected": q("fi_noninj", "att"), "tool_trace_fpr": q("ben_int", "ben"), "n": dict(c)}
    # component ablations (GPT-4o-mini, lexical)
    abl = {}
    for comp in ("", "semantic_detector", "sensitive_flow", "untrusted_sensitive_context", "permission_policy", "system_prompt_policy"):
        row = {}
        for name, (dep, rule, scope) in (("R1", CONFIGS["R1"]), ("R2", CONFIGS["R2"])):
            try:
                tr = ar.load_replay("gpt-4o-mini", dep, "lexical", "all", "gt", comp)
            except FileNotFoundError:
                continue
            m = ar.metrics(tr, rule, 0.60, 0.20, scope)
            row[name] = {k: m[k] for k in KEYS}
        # removing counterfactual attribution: R2 graph with the score rule
        if comp == "":
            tr = ar.load_replay("gpt-4o-mini", "provenance", "lexical", "all", "gt")
            m = ar.metrics(tr, "score", 0.60, 0.20, "candidate")
            abl["minus_counterfactual"] = {"R2": {k: m[k] for k in KEYS}}
        abl[comp or "full"] = row
    res["ablation"] = abl
    out = r2lib.RESULTS / "replay_summary.json"
    out.write_text(json.dumps(res, indent=1))
    for key, block in res.items():
        if key == "ablation" or key.endswith("neural_detector_gate"):
            if key.endswith("neural_detector_gate"):
                print(key, {k: round(v, 3) for k, v in block.items() if k != "n"})
            continue
        print(key)
        for name in CONFIGS:
            b = block[name]
            print(f"  {name:10s} prev {b['succ_attack_prevention']:.3f} {b['ci_prevention']} resASR {b['replay_residual_asr']:.4f} "
                  f"OB {b['leg_overblock_attack']:.3f} {b['ci_overblock']} OBben {b['leg_overblock_benign']:.3f} "
                  f"toolFPR {b['tool_trace_fpr']:.3f} spur {b['spurious_attr_legit']:.3f}")
        d = block["R2_minus_R1"]
        print(f"  R2-R1: prev {d['prevention_diff']:+.3f} {d['prevention_diff_ci']}  OB {d['overblock_diff']:+.3f} {d['overblock_diff_ci']}")
        for ename, d in block["R2_minus_R1_exact"].items():
            print(f"  exact {ename:22s} R1 {d['R1']:.3f} R2 {d['R2']:.3f} diff {d['diff']:+.3f} {[round(x, 3) for x in d['diff_ci']]}")
        d = block["attr_minus_score_prov"]
        print(f"  attr-score(prov): prev {d['prevention_diff']:+.3f} {d['prevention_diff_ci']}  OB {d['overblock_diff']:+.3f} {d['overblock_diff_ci']}")
    print("ablation")
    for comp, row in abl.items():
        print(" ", comp, {n: (round(v["trace_f1"], 3), round(v["trace_fpr"], 3), round(v["succ_attack_prevention"], 3), round(v["leg_overblock_attack"], 3)) for n, v in row.items()})


if __name__ == "__main__":
    main()
