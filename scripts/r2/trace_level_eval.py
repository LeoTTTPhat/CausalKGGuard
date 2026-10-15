#!/usr/bin/env python3
"""Trace-level comparison with baselines on the corrected (label-independent) protocol.

Labels follow R1 (every injected case is an attack attempt). All methods see the
same re-annotated traces (task allowlist = user-task ground-truth tools for
attack and benign traces alike; sinks from the live adapter mapping).
Guard configurations are read from the replay files so that trace-level and
action-level results come from identical risk computations.

Writes results/trace_level/<model>/{cases.csv, summary.json, summary.md}.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import analyze_replay as ar
import r2lib
import stats_utils as su
from causalkgguard import TraceCase
from causalkgguard.baselines import baseline_predictors

GUARDS = {
    # name: (dependency, detector, rule, delta)
    "ckg_r1": ("prefix", "lexical", "score", 0.2),
    "ckg_r2": ("provenance", "lexical", "attr", 0.2),
    "ckg_r1_generic": ("prefix", "generic", "score", 0.2),
    "ckg_r2_generic": ("provenance", "generic", "attr", 0.2),
    "ckg_r1_neural": ("prefix", "neural", "score", 0.2),
    "ckg_r2_neural": ("provenance", "neural", "attr", 0.2),
}


def camel_capability(case: TraceCase) -> bool:
    """CaMeL-style capability + data-flow rule (our re-implementation, as in R1)."""
    from causalkgguard.detectors import inspect_text
    from causalkgguard.reasoning import is_high_impact_tool
    from causalkgguard.schema import DATA_SOURCE_ROLES

    allowed_sinks = set(case.metadata.get("allowed_sinks", []))
    required = set(case.required_permissions)
    tainted = sensitive = False
    for e in case.events:
        sig = inspect_text(e.content)
        if (not e.trusted) or e.role in DATA_SOURCE_ROLES:
            tainted = tainted or bool(sig.injection or sig.exfiltration or sig.system_prompt_request)
            sensitive = sensitive or e.sensitivity > 0 or sig.sensitive > 0
        if e.role in {"action", "output"}:
            ext = bool(e.sink) and any(h in e.sink.lower() for h in ("external", "webhook", "email:", "http", "slack:", "public"))
            authorized_sink = (e.sink in allowed_sinks) if e.sink else True
            if e.permissions and required and not set(e.permissions).issubset(required):
                return True
            if is_high_impact_tool(e.tool) and e.tool not in required and not authorized_sink:
                return True
            if (sensitive or tainted) and ext and not authorized_sink:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    raw = r2lib.load_cases(r2lib.model_files(args.model))
    cases = [r2lib.reannotate(c) for c in raw]
    meta = {}
    for c in cases:
        md = c["metadata"]
        lab = r2lib.label_actions(c)
        meta[c["case_id"]] = {
            "suite": md["suite"], "user_task": r2lib.user_task_key(c), "inj_task": r2lib.injection_task_key(c),
            "family": r2lib.family_of(c), "attack": c["label"] == "attack",
            "exposed": any(v["injection_bearing"] for v in lab.values()),
        }
    preds = defaultdict(dict)
    base = baseline_predictors()
    base["camel_capability"] = camel_capability
    for name, fn in base.items():
        for c in cases:
            preds[name][c["case_id"]] = bool(fn(TraceCase.from_dict(c)))
    # detector-only neural baseline (Qwen2.5-7B zero-shot on untrusted observations)
    try:
        flags = {}
        for c in cases:
            flags[c["case_id"]] = any(r2lib.neural_score(e["content"]) >= 0.5
                                      for e in c["events"] if e["role"] in {"tool", "retrieved", "memory"})
        preds["llm_detector_only"] = flags
    except r2lib.MissingNeuralScore:
        print("neural scores incomplete: skipping llm_detector_only")
    for name, (dep, det, rule, delta) in GUARDS.items():
        try:
            traces = ar.load_replay(args.model, dep, det, "all", "gt")
        except FileNotFoundError:
            continue
        if len(traces) != len(cases):
            print(f"{name}: replay incomplete ({len(traces)}/{len(cases)}), skipped")
            continue
        for t in traces:
            preds[name][t["case_id"]] = any(ar.decide(t["rows"], rule, 0.60, delta, "candidate"))

    out = r2lib.RESULTS / "trace_level" / args.model
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "suite", "user_task", "injection_task", "family", "label", "exposed", "method", "pred"])
        for name, p in preds.items():
            for cid, v in p.items():
                m = meta[cid]
                w.writerow([cid, m["suite"], m["user_task"], m["inj_task"], m["family"],
                            "attack" if m["attack"] else "benign", int(m["exposed"]), name, int(v)])

    ids = [c["case_id"] for c in cases]
    users = sorted({meta[i]["user_task"] for i in ids})
    templates = sorted({meta[i]["inj_task"] for i in ids if meta[i]["attack"]})
    uidx = {u: k for k, u in enumerate(users)}
    summary = {}
    for name, p in preds.items():
        tp = sum(p[i] for i in ids if meta[i]["attack"])
        fn = sum(1 - p[i] for i in ids if meta[i]["attack"])
        fp = sum(p[i] for i in ids if not meta[i]["attack"])
        tn = sum(1 - p[i] for i in ids if not meta[i]["attack"])
        exp_ids = [i for i in ids if meta[i]["attack"] and meta[i]["exposed"]]
        tp_e = sum(p[i] for i in exp_ids)
        res = su.prf(tp, fn, fp, tn)
        res.update({"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall_exposed": tp_e / len(exp_ids)})
        res["recall_cp"] = su.clopper_pearson(tp, tp + fn)
        res["fpr_cp"] = su.clopper_pearson(fp, fp + tn)
        # cluster bootstrap over user tasks: per-cluster (tp, fn, fp, tn)
        cc = np.zeros((len(users), 4))
        for i in ids:
            k = uidx[meta[i]["user_task"]]
            if meta[i]["attack"]:
                cc[k, 0 if p[i] else 1] += 1
            else:
                cc[k, 2 if p[i] else 3] += 1
        res["recall_ci_task"] = su.cluster_bootstrap(cc, lambda s: s[0] / (s[0] + s[1]) if s[0] + s[1] else np.nan, args.B)
        res["f1_ci_task"] = su.cluster_bootstrap(cc, lambda s: su.prf(*s)["f1"], args.B, seed=1)
        res["fpr_ci_task"] = su.cluster_bootstrap(cc, lambda s: s[2] / (s[2] + s[3]) if s[2] + s[3] else np.nan, args.B, seed=2)
        # cluster bootstrap over attack templates (suite/injection task) for recall
        tidx = {t: k for k, t in enumerate(templates)}
        ct = np.zeros((len(templates), 2))
        for i in ids:
            if meta[i]["attack"]:
                ct[tidx[meta[i]["inj_task"]], 0 if p[i] else 1] += 1
        res["recall_ci_template"] = su.cluster_bootstrap(ct, lambda s: s[0] / (s[0] + s[1]), args.B, seed=3)
        res["task_level_recall_mean"] = float(np.nanmean([cc[k, 0] / (cc[k, 0] + cc[k, 1]) if cc[k, 0] + cc[k, 1] else np.nan for k in range(len(users))]))
        summary[name] = res

    # paired comparisons against the two guard configurations
    comparisons = {}
    for ref in ("ckg_r2", "ckg_r1"):
        if ref not in preds:
            continue
        for name in preds:
            if name == ref:
                continue
            att = [i for i in ids if meta[i]["attack"]]
            b = sum(preds[ref][i] and not preds[name][i] for i in att)
            c = sum(preds[name][i] and not preds[ref][i] for i in att)
            per_task_ref, per_task_other = [], []
            diffs = np.zeros((len(users), 2))
            for u in users:
                a_ids = [i for i in att if meta[i]["user_task"] == u]
                if not a_ids:
                    continue
                r1 = np.mean([preds[ref][i] for i in a_ids])
                r2 = np.mean([preds[name][i] for i in a_ids])
                per_task_ref.append(r1)
                per_task_other.append(r2)
            dd = np.array(per_task_ref) - np.array(per_task_other)
            rng = np.random.default_rng(7)
            boots = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(args.B)]
            comparisons[f"{ref}_vs_{name}"] = {
                "ref_only": b, "other_only": c, "mcnemar_p": su.mcnemar_exact(b, c),
                "task_mean_recall_diff": float(dd.mean()),
                "task_diff_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                "wilcoxon_p": su.wilcoxon_paired(per_task_ref, per_task_other),
                "n_tasks": len(dd),
            }
    (out / "summary.json").write_text(json.dumps({"methods": summary, "comparisons": comparisons}, indent=1))
    lines = ["| method | P | R | F1 | FPR | FP/97 | R exposed | R 95% CI (task) | R 95% CI (template) | F1 95% CI (task) | FPR 95% CI (exact) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in summary.items():
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['fpr']:.3f} | {r['fp']} | "
                     f"{r['recall_exposed']:.3f} | [{r['recall_ci_task'][0]:.3f}, {r['recall_ci_task'][1]:.3f}] | "
                     f"[{r['recall_ci_template'][0]:.3f}, {r['recall_ci_template'][1]:.3f}] | "
                     f"[{r['f1_ci_task'][0]:.3f}, {r['f1_ci_task'][1]:.3f}] | [{r['fpr_cp'][0]:.3f}, {r['fpr_cp'][1]:.3f}] |")
    lines += ["", "| comparison | ref-only | other-only | McNemar p | task-level mean diff [95% CI] | Wilcoxon p (tasks) |", "|---|---|---|---|---|---|"]
    for k, v in comparisons.items():
        lines.append(f"| {k} | {v['ref_only']} | {v['other_only']} | {v['mcnemar_p']:.2e} | {v['task_mean_recall_diff']:+.3f} "
                     f"[{v['task_diff_ci'][0]:+.3f}, {v['task_diff_ci'][1]:+.3f}] | {v['wilcoxon_p']:.2e} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
