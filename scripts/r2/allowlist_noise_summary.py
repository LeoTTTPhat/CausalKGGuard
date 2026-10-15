#!/usr/bin/env python3
"""Summarise the annotation-noise (task allowlist) experiment for R1 and R2 configurations."""
import json
import numpy as np
import analyze_replay as ar
import r2lib

MODES = [("gt", ["gt"]), ("drop0.25", [f"drop0.25_s{s}" for s in range(3)]), ("drop0.5", [f"drop0.5_s{s}" for s in range(3)]),
         ("add0.25", [f"add0.25_s{s}" for s in range(3)]), ("add0.5", [f"add0.5_s{s}" for s in range(3)]),
         ("heuristic", ["heuristic"]), ("suite", ["suite"]), ("none", ["none"])]
KEYS = ["succ_attack_prevention", "replay_residual_asr", "inj_interception", "leg_overblock_attack", "leg_overblock_benign", "tool_trace_fpr", "trace_recall", "trace_fpr"]
CFG = {"R1": ("prefix", "score", "sticky"), "R1cand": ("prefix", "score", "candidate"), "R2": ("provenance", "attr", "candidate")}
out = {}
for mode, tags in MODES:
    for name, (dep, rule, scope) in CFG.items():
        vals = []
        for tag in tags:
            tr = ar.load_replay("gpt-4o-mini", dep, "lexical", "all", tag)
            vals.append(ar.metrics(tr, rule, 0.60, 0.20, scope))
        out[f"{mode}|{name}"] = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))) for k in KEYS}
(r2lib.RESULTS / "allowlist_noise_gpt-4o-mini.json").write_text(json.dumps(out, indent=1))
print("mode       cfg     " + " ".join(f"{k[:12]:>13s}" for k in KEYS))
for k, v in out.items():
    m, c = k.split("|")
    print(f"{m:10s} {c:7s} " + " ".join(f"{v[x][0]:8.3f}±{v[x][1]:.3f}" for x in KEYS))
