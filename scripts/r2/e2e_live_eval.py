#!/usr/bin/env python3
"""End-to-end (live, in-loop) AgentDojo results with case- and task-level statistics.

Reads raw AgentDojo JSON logs. For each (model, configuration):
  attack runs : ASR = mean(security), utility under attack = mean(utility),
                share of cases with at least one blocked call;
  benign runs : utility on the 97 canonical user tasks (and on the expanded set).
Intervals: Clopper-Pearson (cases) and cluster bootstrap over user tasks.
Paired tests (guarded vs no defense, matched by suite/user task/injection task):
exact McNemar on cases, Wilcoxon signed-rank on per-user-task ASR, and a
cluster-bootstrap CI of the ASR difference.

Writes results/e2e/summary.json and summary.md
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import r2lib
import stats_utils as su

RUNS = Path("/Users/sonha/CausalKGGuard_eval/results/agentdojo_runs")
FAMILIES = ("direct", "ignore_previous", "important_instructions", "system_message", "tool_knowledge")

CONFIGS = {
    ("gpt-4o-mini", "none"): [RUNS / f"full_gpt4omini_{f}" for f in ("direct", "ignore_previous", "important", "system_message", "tool_knowledge", "noattack")],
    ("gpt-4o-mini", "r1"): [RUNS / f"guarded_causalkgguard_{f}_sinkfix" for f in ("direct", "ignore_previous", "important", "system_message", "tool_knowledge", "noattack_expanded")],
    ("gpt-4o-mini", "r2"): [RUNS / "r2_gpt-4o-mini-2024-07-18_guard_r2"],
    ("llama-3.3-70b", "none"): [RUNS / "r2_llama-3.3-70b-instruct_nodef"],
    ("llama-3.3-70b", "r1"): [RUNS / "r2_llama-3.3-70b-instruct_guard_r1"],
    ("llama-3.3-70b", "r2"): [RUNS / "r2_llama-3.3-70b-instruct_guard_r2"],
    ("gemini-2.5-flash-lite", "none"): [RUNS / "r2_gemini-2.5-flash-lite_nodef"],
    ("gemini-2.5-flash-lite", "r1"): [RUNS / "r2_gemini-2.5-flash-lite_guard_r1"],
    ("gemini-2.5-flash-lite", "r2"): [RUNS / "r2_gemini-2.5-flash-lite_guard_r2"],
}


def read(dirs):
    recs = {}
    for d in dirs:
        for f in glob.glob(str(d / "**" / "*.json"), recursive=True):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            suite, ut, it, at = r.get("suite_name"), r.get("user_task_id"), r.get("injection_task_id"), r.get("attack_type")
            key = (suite, ut, str(it), str(at))
            blocked = sum("CausalKGGuard blocked" in json.dumps(m.get("content")) for m in r.get("messages", []))
            recs[key] = {"suite": suite, "user_task": f"{suite}/{ut}", "inj": it, "attack": at,
                         "utility": bool(r.get("utility")), "security": bool(r.get("security")),
                         "blocked": blocked > 0, "duration": r.get("duration") or 0.0,
                         "canonical_benign": it is None and str(ut).startswith("user_task"),
                         # expanded no-attack set: injection goals issued as direct user requests
                         "injection_goal_benign": it is None and str(ut).startswith("injection_task")}
    return recs


def rate_block(recs, field):
    k = sum(r[field] for r in recs)
    n = len(recs)
    users = sorted({r["user_task"] for r in recs})
    idx = {u: i for i, u in enumerate(users)}
    cc = np.zeros((len(users), 2))
    for r in recs:
        cc[idx[r["user_task"]], 0 if r[field] else 1] += 1
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "cp": su.clopper_pearson(k, n),
            "task_ci": su.cluster_bootstrap(cc, lambda s: s[0] / (s[0] + s[1]) if s[0] + s[1] else np.nan, 10000)}


def main() -> None:
    data = {k: read(v) for k, v in CONFIGS.items()}
    out = {}
    lines = ["| model | config | attack cases | ASR [CP 95%] | ASR task-cluster 95% | utility under attack | blocked cases | benign utility (97) | benign blocked |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (model, cfg), recs in data.items():
        att = [r for (s, u, i, a), r in recs.items() if a not in ("none", "None") and i not in ("None",)]
        ben = [r for r in recs.values() if r["canonical_benign"]]
        if not att and not ben:
            continue
        res = {"attack_n": len(att), "benign_n": len(ben)}
        if att:
            res["asr"] = rate_block(att, "security")
            res["utility_attack"] = rate_block(att, "utility")
            res["blocked_attack"] = rate_block(att, "blocked")
            res["by_family"] = {f: rate_block([r for r in att if r["attack"] == f], "security") for f in FAMILIES if any(r["attack"] == f for r in att)}
        if ben:
            res["utility_benign"] = rate_block(ben, "utility")
            res["blocked_benign"] = rate_block(ben, "blocked")
        ig = [r for r in recs.values() if r["injection_goal_benign"]]
        if ig:
            res["injgoal_n"] = len(ig)
            res["utility_injgoal"] = rate_block(ig, "utility")
            res["blocked_injgoal"] = rate_block(ig, "blocked")
        out[f"{model}|{cfg}"] = res
        a = res.get("asr")
        lines.append(
            f"| {model} | {cfg} | {len(att)} | "
            + (f"{a['rate']:.3f} [{a['cp'][0]:.3f}, {a['cp'][1]:.3f}] | [{a['task_ci'][0]:.3f}, {a['task_ci'][1]:.3f}] | "
               f"{res['utility_attack']['rate']:.3f} | {res['blocked_attack']['rate']:.3f} | " if a else "- | - | - | - | ")
            + (f"{res['utility_benign']['rate']:.3f} ({res['utility_benign']['k']}/{len(ben)}) | {res['blocked_benign']['rate']:.3f} |" if ben else "- | - |"))
    # paired comparisons on matched cases
    comp = {}
    for model in ("gpt-4o-mini", "llama-3.3-70b", "gemini-2.5-flash-lite"):
        for a_cfg, b_cfg in (("none", "r1"), ("none", "r2"), ("r1", "r2")):
            A, B = data.get((model, a_cfg), {}), data.get((model, b_cfg), {})
            keys = [k for k in A if k in B and k[3] not in ("none", "None") and k[2] != "None"]
            if len(keys) < 20:
                continue
            for field in ("security", "utility"):
                b = sum(A[k][field] and not B[k][field] for k in keys)
                c = sum(B[k][field] and not A[k][field] for k in keys)
                per = defaultdict(lambda: [0, 0, 0])
                for k in keys:
                    u = A[k]["user_task"]
                    per[u][0] += A[k][field]; per[u][1] += B[k][field]; per[u][2] += 1
                ra = [v[0] / v[2] for v in per.values()]
                rb = [v[1] / v[2] for v in per.values()]
                d = np.array(ra) - np.array(rb)
                rng = np.random.default_rng(11)
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
                comp[f"{model}|{a_cfg}_vs_{b_cfg}|{field}"] = {
                    "n_cases": len(keys), "a_rate": float(np.mean([A[k][field] for k in keys])),
                    "b_rate": float(np.mean([B[k][field] for k in keys])), "a_only": b, "b_only": c,
                    "mcnemar_p": su.mcnemar_exact(b, c), "task_mean_diff": float(d.mean()),
                    "task_diff_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                    "wilcoxon_p": su.wilcoxon_paired(ra, rb), "n_tasks": len(d)}
    lines += ["", "| comparison | field | matched cases | rate A | rate B | McNemar p | task-level diff [95% CI] | Wilcoxon p |", "|---|---|---|---|---|---|---|---|"]
    for k, v in comp.items():
        m, pair, field = k.split("|")
        lines.append(f"| {m} {pair} | {field} | {v['n_cases']} | {v['a_rate']:.3f} | {v['b_rate']:.3f} | {v['mcnemar_p']:.2e} | "
                     f"{v['task_mean_diff']:+.3f} [{v['task_diff_ci'][0]:+.3f}, {v['task_diff_ci'][1]:+.3f}] | {v['wilcoxon_p']:.2e} |")
    dest = r2lib.RESULTS / "e2e"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps({"configs": out, "comparisons": comp}, indent=1))
    (dest / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
