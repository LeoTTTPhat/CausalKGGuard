#!/usr/bin/env python3
"""Generate the LaTeX tables and figure data of the R2 manuscript from the result files.

Every number in the generated tables comes from ManuscriptR2/results/*.json; the
manuscript inputs paper/generated/*.tex. Re-run after any experiment changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import r2lib

RES = r2lib.RESULTS
GEN = r2lib.R2 / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def f3(x):
    return "--" if x is None or x != x else f"{x:.3f}"


def pct(x, d=1):
    return "--" if x is None or x != x else f"{100 * x:.{d}f}"


def ci(c, d=3, scale=1.0):
    if not c or c[0] != c[0]:
        return "--"
    return f"[{scale * c[0]:.{d}f}, {scale * c[1]:.{d}f}]"


def tex_p(p):
    if p == 0 or p < 1e-300:
        return "$<10^{-300}$"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def sg(x, d=3):
    """Signed number with a typographic minus."""
    if x is None or x != x:
        return "--"
    return f"${x:+.{d}f}$"


def sci(c, d=3):
    if not c or c[0] != c[0]:
        return "--"
    return f"[${c[0]:.{d}f}$, ${c[1]:.{d}f}$]"


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")
    print("wrote", name)


MODEL_LABEL = {"gpt-4o-mini": "GPT-4o-mini", "llama-3.3-70b": "Llama-3.3-70B", "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
               "llama-3.3-70b-instruct": "Llama-3.3-70B"}


# ------------------------------------------------------------------ RQ1 live end-to-end
def tab_e2e():
    d = load("e2e/summary.json")
    if not d:
        return
    rows = []
    for model in ("gpt-4o-mini", "llama-3.3-70b", "gemini-2.5-flash-lite"):
        first = True
        for cfg, label in (("none", "no monitor"), ("r1", "\\cfgS{}"), ("r2", "\\cfgA{}")):
            r = d["configs"].get(f"{model}|{cfg}")
            name = MODEL_LABEL[model] if first else ""
            first = False
            if not r:
                rows.append(f"{name} & {label} & \\multicolumn{{6}}{{l}}{{not available}} \\\\")
                continue
            n = r.get("attack_n", 0)
            if n:
                a = r["asr"]
                cint = ci(a['task_ci'], 1, 100)
                if a["k"] == 0:  # degenerate cluster interval: report the exact interval instead
                    cint = ci(a["cp"], 1, 100) + "\\textsuperscript{a}"
                asr = f"{pct(a['rate'])} & {cint} & {pct(r['utility_attack']['rate'])} & {pct(r['blocked_attack']['rate'])}"
            else:
                asr = "-- & -- & -- & --"
            if r.get("benign_n"):
                ub = r["utility_benign"]
                bb = r["blocked_benign"]
                ben = f"{pct(ub['rate'])} ({ub['k']}/{ub['n']}) & {bb['k']}/{bb['n']}"
            else:
                ben = "-- & --"
            n_str = f"{n:,}".replace(",", "{,}")
            rows.append(f"{name} & {label} & {n_str} & {asr} & {ben} \\\\")
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    write("tab_e2e.tex", f"""\\begin{{tabular}}{{@{{}}l l r r r r r r r@{{}}}}
\\toprule
Agent & Monitor & Attack & ASR & ASR 95\\% CI & Utility & Blocked & Benign & Benign \\\\
 & & cases & (\\%) & (task cluster) & under & attack & utility & blocked \\\\
 & & & & & attack (\\%) & cases (\\%) & (\\%) & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
""")


def tab_e2e_family():
    d = load("e2e/summary.json")
    if not d:
        return
    fams = ("direct", "ignore_previous", "important_instructions", "system_message", "tool_knowledge")
    short = {f: "\\textsf{" + f.replace("_", "\\_") + "}" for f in fams}
    cfgs = [(c, l) for c, l in (("none", "No monitor"), ("r1", "\\cfgS{}"), ("r2", "\\cfgA{}"))
            if (d["configs"].get(f"gpt-4o-mini|{c}") or {}).get("by_family")]
    rows = []
    for f in fams:
        cells = []
        for cfg, _ in cfgs:
            r = d["configs"].get(f"gpt-4o-mini|{cfg}")
            bf = (r or {}).get("by_family", {}).get(f)
            cells.append(f"{pct(bf['rate'])}" if bf else "--")
        rows.append(f"{short[f]} & " + " & ".join(cells) + " \\\\")
    write("tab_e2e_family.tex", "\\begin{tabular}{@{}l" + "r" * len(cfgs) + "@{}}\n\\toprule\nInjection family & " + " & ".join(l for _, l in cfgs)
          + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ RQ2 trace level
METHOD_LABEL = {
    "no_defense": "No defense", "keyword_filter": "Keyword filter", "text_only_detector": "Text-only detector",
    "output_sensitive_filter": "Output sensitive filter", "flow_policy": "Flow policy", "permission_gate": "Permission gate",
    "information_flow": "Information flow", "camel_capability": "CaMeL-style rule", "llm_detector_only": "LLM detector only",
    "ckg_r1": "\\method{} \\cfgS{} (lexical)", "ckg_r2": "\\method{} \\cfgA{} (lexical)", "ckg_r1_generic": "\\method{} \\cfgS{} (template-agnostic)",
    "ckg_r2_generic": "\\method{} \\cfgA{} (template-agnostic)", "ckg_r1_neural": "\\method{} \\cfgS{} (LLM detector)",
    "ckg_r2_neural": "\\method{} \\cfgA{} (LLM detector)",
}


def tab_trace(model="gpt-4o-mini"):
    d = load(f"trace_level/{model}/summary.json")
    if not d:
        return
    order = ["no_defense", "keyword_filter", "text_only_detector", "output_sensitive_filter", "flow_policy", "permission_gate",
             "information_flow", "camel_capability", "llm_detector_only", "ckg_r1", "ckg_r2", "ckg_r1_neural", "ckg_r2_neural"]
    rows = []
    for m in order:
        r = d["methods"].get(m)
        if not r:
            continue
        hi = "\\hirow\n" if m in ("ckg_r2", "ckg_r2_neural") else ""
        prec = "--" if r["precision"] != r["precision"] else f"{r['precision']:.3f}"
        rows.append(f"{hi}{METHOD_LABEL[m]} & {prec} & {r['recall']:.3f} & {ci(r['recall_ci_task'])} & {r['f1']:.3f} & {ci(r['f1_ci_task'])} & "
                    f"{r['fpr']:.3f} ({r['fp']}) & {ci(r['fpr_cp'])} & {1 - r['recall']:.3f} \\\\")
    write(f"tab_trace_{model}.tex", f"""\\begin{{tabular}}{{@{{}}l r r c r c r c r@{{}}}}
\\toprule
Method & Prec. & Recall & Recall 95\\% CI & F1 & F1 95\\% CI & FPR (FP) & FPR 95\\% CI & Miss \\\\
 & & & (task cluster) & & (task cluster) & of 97 & (exact) & rate \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
""")
    # paired tests against R2
    rows = []
    for m in ["output_sensitive_filter", "flow_policy", "permission_gate", "information_flow", "camel_capability", "text_only_detector",
              "keyword_filter", "llm_detector_only", "ckg_r1"]:
        c = d["comparisons"].get(f"ckg_r2_vs_{m}")
        if not c:
            continue
        disc = f"{c['ref_only']:,} / {c['other_only']:,}".replace(",", "{,}")
        rows.append(f"{METHOD_LABEL[m]} & {disc} & {tex_p(c['mcnemar_p'])} & {sg(c['task_mean_recall_diff'])} {sci(c['task_diff_ci'])} & {tex_p(c['wilcoxon_p'])} \\\\")
    write(f"tab_trace_tests_{model}.tex", f"""\\begin{{tabular}}{{@{{}}l r r r r@{{}}}}
\\toprule
Baseline & Discordant & McNemar & Task-level recall & Wilcoxon \\\\
 & (\\cfgA{{}} / other) & $p$ (cases) & difference [95\\% CI] & $p$ (tasks) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
""")


# ------------------------------------------------------------------ RQ3 replay / attribution
CFG_LABEL = {"R1": "\\cfgS{}: prefix, score rule, trace-wide", "pre-score": "prefix, score rule", "pre-attr": "prefix, attribution rule",
             "prov-score": "provenance, score rule", "R2": "\\cfgA{}: provenance, attribution rule"}


def tab_replay():
    d = load("replay_summary.json")
    if not d:
        return
    sets = [("gpt-4o-mini|lexical|all", "GPT-4o-mini (all)"), ("gpt-4o-mini|lexical|test", "GPT-4o-mini (test)"),
            ("llama-3.3-70b-instruct|lexical|all", "Llama-3.3-70B"), ("gemini-2.5-flash-lite|lexical|all", "Gemini-2.5-Flash-Lite")]
    sets = [(k, l) for k, l in sets if k in d]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{l}}}" for _, l in sets)
    cm = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(sets)))
    rows = []
    for cfg in ("R1", "pre-score", "pre-attr", "prov-score", "R2"):
        cells = []
        for k, _ in sets:
            b = d[k][cfg]
            cells += [f"{b['succ_attack_prevention']:.3f}", f"{b['leg_overblock_attack']:.3f}"]
        hi = "\\hirow\n" if cfg == "R2" else ""
        rows.append(f"{hi}{CFG_LABEL[cfg]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for key, label in (("R2_minus_R1", "\\cfgA{} $-$ \\cfgS{}"), ("attr_minus_score_prov", "attribution term (provenance)")):
        cells = []
        for k, _ in sets:
            v = d[k][key]
            cells += [sg(v['prevention_diff']), sg(v['overblock_diff'])]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        cells = []
        for k, _ in sets:
            v = d[k][key]
            cells += [f"\\scriptsize{sci(v['prevention_diff_ci'], 3)}", f"\\scriptsize{sci(v['overblock_diff_ci'], 3)}"]
        rows.append(" & " + " & ".join(cells) + " \\\\")
    cols = "r" * (2 * len(sets))
    write("tab_replay.tex", f"""\\begin{{tabular}}{{@{{}}l{cols}@{{}}}}
\\toprule
 & {head} \\\\
{cm}
Configuration & {' & '.join(['Prev.', 'OB'] * len(sets))} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
""")
    # extra numbers used in the text
    extra = {}
    for k, _ in sets:
        for cfg in ("R1", "R2", "prov-score", "pre-attr", "pre-score"):
            b = d[k][cfg]
            extra[f"{k}|{cfg}"] = {x: b[x] for x in ("replay_residual_asr", "inj_interception", "leg_overblock_benign", "tool_trace_fpr",
                                                      "spurious_attr_legit", "ci_prevention", "ci_overblock", "att_traces_with_legit_block")}
    (GEN / "replay_extra.json").write_text(json.dumps(extra, indent=1))


EXACT_ROWS = (("prevention_exact", "successful attacks stopped at the injected call"),
              ("first_int_noninjected", "attacked sessions whose first interception is not injected"),
              ("first_int_legit", "\\quad of which a legitimate call"),
              ("noinj_intercepted", "attacked sessions without injected call, intercepted"),
              ("overblock_exact", "legitimate calls intercepted up to the first interception"))


def tab_replay_exact():
    d = load("replay_summary.json")
    if not d:
        return
    sets = [("gpt-4o-mini|lexical|all", "GPT-4o-mini (all)"), ("gpt-4o-mini|lexical|test", "GPT-4o-mini (test)"),
            ("llama-3.3-70b-instruct|lexical|all", "Llama-3.3-70B"), ("gemini-2.5-flash-lite|lexical|all", "Gemini-2.5-Flash-Lite")]
    sets = [(k, l) for k, l in sets if k in d and "R2_minus_R1_exact" in d[k]]
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{l}}}" for _, l in sets)
    cm = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(sets)))
    rows = []
    for key, lab in EXACT_ROWS:
        cells, cis = [], []
        for k, _ in sets:
            v = d[k]["R2_minus_R1_exact"][key]
            cells += [f"{v['R1']:.3f}", f"{v['R2']:.3f}", sg(v['diff'])]
            cis += ["", "", f"\\scriptsize{sci(v['diff_ci'], 3)}"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        rows.append(" & " + " & ".join(cis) + " \\\\")
    cols = "rrr" * len(sets)
    hdr = " & ".join(["\\cfgS{}", "\\cfgA{}", "difference"] * len(sets))
    write("tab_replay_exact.tex", f"""\\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{4.4cm}}{cols}@{{}}}}
\\toprule
 & {head} \\\\
{cm}
Measure & {hdr} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
""")


def tab_ablation():
    d = load("replay_summary.json")
    if not d or "ablation" not in d:
        return
    a = d["ablation"]
    lab = {"full": "Full monitor", "minus_counterfactual": "$-$ counterfactual attribution (score rule)",
           "semantic_detector": "$-$ semantic detector", "sensitive_flow": "$-$ sensitive-flow edges",
           "untrusted_sensitive_context": "$-$ untrusted-sensitive-context policy", "permission_policy": "$-$ permission policy",
           "system_prompt_policy": "$-$ system-prompt policy"}
    rows = []
    for k in ("full", "minus_counterfactual", "semantic_detector", "sensitive_flow", "untrusted_sensitive_context", "permission_policy", "system_prompt_policy"):
        row = a.get(k, {})
        cells = []
        for cfg in ("R1", "R2"):
            v = row.get(cfg)
            if v is None:
                cells += ["--"] * 4
            else:
                cells += [f"{v['trace_f1']:.3f}", f"{v['trace_fpr']:.3f}", f"{v['succ_attack_prevention']:.3f}", f"{v['leg_overblock_attack']:.3f}"]
        hi = "\\hirow\n" if k == "full" else ""
        rows.append(f"{hi}{lab[k]} & " + " & ".join(cells) + " \\\\")
    write("tab_ablation.tex", f"""\\begin{{tabular}}{{@{{}}l rrrr rrrr@{{}}}}
\\toprule
 & \\multicolumn{{4}}{{c}}{{\\cfgS{{}}}} & \\multicolumn{{4}}{{c}}{{\\cfgA{{}}}} \\\\
\\cmidrule(lr){{2-5}}\\cmidrule(lr){{6-9}}
Variant & F1 & FPR & Prev. & OB & F1 & FPR & Prev. & OB \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
""")


def tab_detectors():
    d = load("replay_summary.json")
    if not d:
        return
    rows = []
    for model in ("gpt-4o-mini", "llama-3.3-70b-instruct", "gemini-2.5-flash-lite"):
        first = True
        for det, dl in (("lexical", "lexical (template-aware)"), ("generic", "template-agnostic lexical"), ("neural", "LLM-based (Qwen2.5-7B)")):
            b = d.get(f"{model}|{det}|all")
            if not b:
                continue
            name = MODEL_LABEL[model] if first else ""
            first = False
            cells = []
            for cfg in ("R1", "R2"):
                x = b[cfg]
                cells += [f"{x['trace_recall']:.3f}", f"{x['trace_fpr']:.3f}", f"{x['succ_attack_prevention']:.3f}", f"{x['leg_overblock_attack']:.3f}"]
            rows.append(f"{name} & {dl} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    write("tab_detectors.tex", f"""\\begin{{tabular}}{{@{{}}l l rrrr rrrr@{{}}}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{\\cfgS{{}}}} & \\multicolumn{{4}}{{c}}{{\\cfgA{{}}}} \\\\
\\cmidrule(lr){{3-6}}\\cmidrule(lr){{7-10}}
Agent & Semantic detector & Recall & FPR & Prev. & OB & Recall & FPR & Prev. & OB \\\\
\\midrule
{chr(10).join(rows[:-1])}
\\bottomrule
\\end{{tabular}}
""")


def tab_noise():
    d = load("allowlist_noise_gpt-4o-mini.json")
    if not d:
        return
    lab = {"gt": "oracle (ground-truth tools)", "drop0.25": "25\\% of required tools dropped", "drop0.5": "50\\% of required tools dropped",
           "add0.25": "25\\% of other tools added", "add0.5": "50\\% of other tools added", "heuristic": "prompt-derived (no oracle)",
           "suite": "all tools of the suite", "none": "empty allowlist"}
    rows = []
    for mode in ("gt", "drop0.25", "drop0.5", "add0.25", "add0.5", "heuristic", "suite", "none"):
        cells = []
        for cfg in ("R1", "R2"):
            v = d.get(f"{mode}|{cfg}")
            if not v:
                cells += ["--"] * 3
                continue
            cells += [f"{v['succ_attack_prevention'][0]:.3f}", f"{v['leg_overblock_attack'][0]:.3f}", f"{v['tool_trace_fpr'][0]:.3f}"]
        rows.append(f"{lab[mode]} & " + " & ".join(cells) + " \\\\")
    write("tab_noise.tex", f"""\\begin{{tabular}}{{@{{}}l rrr rrr@{{}}}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{\\cfgS{{}}}} & \\multicolumn{{3}}{{c}}{{\\cfgA{{}}}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}
Task allowlist & Prev. & OB & FPR & Prev. & OB & FPR \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
""")


CFGN = {"r1": "prefix, score", "r2": "\\cfgA{}"}


def tab_sensitivity():
    d = load("sensitivity/gpt-4o-mini.json")
    if not d:
        return
    rows = []
    for spread in ("pm25", "pm50"):
        w = d["weights"][spread]
        for cfg in ("r1", "r2"):
            v = w[cfg]
            rows.append(f"$\\pm{spread[2:]}\\%$ weights & {CFGN[cfg]} & {v['succ_attack_prevention'][0]:.3f} [{v['succ_attack_prevention'][1]:.3f}, {v['succ_attack_prevention'][2]:.3f}] & "
                        f"{v['leg_overblock_attack'][0]:.3f} [{v['leg_overblock_attack'][1]:.3f}, {v['leg_overblock_attack'][2]:.3f}] & "
                        f"{v['tool_trace_fpr'][0]:.3f} [{v['tool_trace_fpr'][1]:.3f}, {v['tool_trace_fpr'][2]:.3f}] \\\\")
    for cfg in ("r1", "r2"):
        v = d["uniform"][cfg]
        rows.append(f"uniform weights & {CFGN[cfg]} & {v['succ_attack_prevention']:.3f} & {v['leg_overblock_attack']:.3f} & {v['tool_trace_fpr']:.3f} \\\\")
    for dep, cfg in (("prefix", CFGN["r1"]), ("provenance", CFGN["r2"])):
        v = d["learned"][dep]
        rows.append(f"fitted on calibration split (test) & {cfg} & {v['test_learned']['succ_attack_prevention']:.3f} & {v['test_learned']['leg_overblock_attack']:.3f} & {v['test_learned']['tool_trace_fpr']:.3f} \\\\")
        rows.append(f"hand-set weights (test) & {cfg} & {v['test_hand']['succ_attack_prevention']:.3f} & {v['test_hand']['leg_overblock_attack']:.3f} & {v['test_hand']['tool_trace_fpr']:.3f} \\\\")
    write("tab_sensitivity.tex", f"""\\begin{{tabular}}{{@{{}}l l r r r@{{}}}}
\\toprule
Weights & Config. & Prevention & Over-blocking & Benign FPR \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
""")
    # theta / delta sweep figure data
    th = d["theta_sweep"]
    r1 = " ".join(f"({x['theta']},{100 * x['r1_tracewide']['replay_residual_asr']:.3f})" for x in th)
    r1c = " ".join(f"({x['theta']},{100 * x['r1']['replay_residual_asr']:.3f})" for x in th)
    f1 = " ".join(f"({x['theta']},{100 * x['r1_tracewide']['tool_trace_fpr']:.2f})" for x in th)
    f2 = " ".join(f"({x['theta']},{100 * x['r2']['tool_trace_fpr']:.2f})" for x in th)
    r2 = " ".join(f"({x['theta']},{100 * x['r2']['replay_residual_asr']:.3f})" for x in th)
    o1 = " ".join(f"({x['theta']},{100 * x['r1_tracewide']['leg_overblock_attack']:.2f})" for x in th)
    o2 = " ".join(f"({x['theta']},{100 * x['r2']['leg_overblock_attack']:.2f})" for x in th)
    write("fig_theta_data.tex", f"\\def\\thetaRone{{{r1}}}\n\\def\\thetaRonec{{{r1c}}}\n\\def\\thetaRtwo{{{r2}}}\n\\def\\thetaOone{{{o1}}}\n\\def\\thetaOtwo{{{o2}}}\n"
          f"\\def\\thetaFone{{{f1}}}\n\\def\\thetaFtwo{{{f2}}}\n")
    dl = d["delta_sweep"]
    write("fig_delta_data.tex", "\\def\\deltaRes{" + " ".join(f"({x['delta']},{100 * x['r2']['replay_residual_asr']:.3f})" for x in dl) + "}\n"
          + "\\def\\deltaOB{" + " ".join(f"({x['delta']},{100 * x['r2']['leg_overblock_attack']:.2f})" for x in dl) + "}\n"
          + "\\def\\deltaFPR{" + " ".join(f"({x['delta']},{100 * x['r2']['tool_trace_fpr']:.2f})" for x in dl) + "}\n")


# ------------------------------------------------------------------ explanations, runtime
def tab_expl():
    d = load("explanations_r2.json")
    if not d:
        return
    c = d["controlled"]
    rows = []
    for k, lab in (("intercepted", "attacks intercepted"), ("policy_recall", "policy recall"), ("policy_precision_strict", "policy precision (expected set only)"),
                   ("policy_validity", "policy precision (mechanically valid)"), ("source_hit", "source hit"), ("sink_hit", "sink hit"), ("path_hit", "evidence path")):
        rows.append(f"{lab} & {c['R1'][k]:.3f} & {c['R2'][k]:.3f} \\\\")
    rows.append(f"extra policy firings (valid / total) & {c['R1']['extra_firings_valid']} / {c['R1']['extra_firings']} & {c['R2']['extra_firings_valid']} / {c['R2']['extra_firings']} \\\\")
    write("tab_expl_controlled.tex", "\\begin{tabular}{@{}l rr@{}}\n\\toprule\nMeasure (60 attacks) & prefix, score & \\cfgA{} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    a = d["agentdojo"]
    rows = []
    for k, lab in (("intercepted_injected", "intercepted injected calls"), ("source_hit", "cites an injected observation"),
                   ("evidence_specificity", "evidence specificity"), ("mean_untrusted_deps", "untrusted dependencies cited"),
                   ("mean_deps", "dependencies cited"), ("spurious_attribution_legit", "spurious attribution (legitimate calls)")):
        cells = []
        for m in ("gpt-4o-mini", "gemini-2.5-flash-lite"):
            for cfg in ("R1", "R2"):
                v = a.get(f"{m}|{cfg}", {}).get(k)
                cells.append("--" if v is None else (f"{v:,.0f}".replace(",", "{,}") if k == "intercepted_injected" else f"{v:.2f}" if "mean" in k else f"{v:.3f}"))
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_expl_agentdojo.tex", "\\begin{tabular}{@{}l rr rr@{}}\n\\toprule\n & \\multicolumn{2}{c}{GPT-4o-mini} & \\multicolumn{2}{c}{Gemini-2.5-Flash-Lite} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nMeasure & prefix, score & \\cfgA{} & prefix, score & \\cfgA{} \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_runtime():
    d = load("runtime/gpt-4o-mini.json")
    s = load("runtime/scaling.json")
    if d:
        rows = []
        for k, lab in (("events_per_trace", "events per trace"), ("tool_calls_per_trace", "tool calls per trace"),
                       ("prefix_length_at_decision", "prefix length at a decision")):
            v = d[k]
            rows.append(f"{lab} & {v['median']:.0f} & {v['p95']:.0f} & {v['p99']:.0f} & {v['max']:.0f} \\\\")
        rows.append("\\midrule")
        for k, lab in (("r1_decision_ms", "\\cfgS{} decision (ms)"), ("r2_decision_ms", "\\cfgA{} decision (ms)")):
            v = d[k]
            rows.append(f"{lab} & {v['median']:.2f} & {v['p95']:.1f} & {v['p99']:.1f} & {v['max']:.1f} \\\\")
        for k, lab in (("r1_edges_sample", "\\cfgS{} graph edges"), ("r2_edges_sample", "\\cfgA{} graph edges")):
            v = d[k]
            rows.append(f"{lab} & {v['median']:.0f} & {v['p95']:.0f} & {v['p99']:.0f} & {v['max']:.0f} \\\\")
        n = d["r1_decision_ms"]["n"]
        write("tab_runtime_real.tex", "\\begin{tabular}{@{}l rrrr@{}}\n\\toprule\nQuantity & Median & 95th pct. & 99th pct. & Max \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    if s:
        rows = [f"{r['events']:,} & {r['prefix_edges']:,} & {r['provenance_edges']:,} & {r['r1_decision_ms']:.1f} & {r['r2_decision_ms']:.1f} \\\\".replace(",", "{,}") for r in s["rows"]]
        write("tab_runtime_scaling.tex", "\\begin{tabular}{@{}rrrrr@{}}\n\\toprule\nEvents & Edges (prefix) & Edges (prov.) & \\cfgS{} (ms) & \\cfgA{} (ms) \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        write("fig_scaling_data.tex", "\\def\\scaleRone{" + " ".join(f"({r['events']},{r['r1_decision_ms']:.3f})" for r in s["rows"]) + "}\n"
              + "\\def\\scaleRtwo{" + " ".join(f"({r['events']},{r['r2_decision_ms']:.3f})" for r in s["rows"]) + "}\n")


# ------------------------------------------------------------------ robustness (synthetic)
SYN_LABEL = {"keyword_filter": "Keyword", "text_only_detector": "Text-only", "information_flow": "Info-flow", "permission_gate": "Perm.\\ gate",
             "camel_capability": "CaMeL-style", "R1": "\\cfgS{}", "R1_norm": "\\cfgS{}$^{\\dagger}$", "R2": "\\cfgA{}", "R2_norm": "\\cfgA{}$^{\\dagger}$",
             "R1_neural": "\\cfgS{}-LLM", "R2_neural": "\\cfgA{}-LLM", "detector_only_neural": "LLM only", "output_sensitive_filter": "Output", "flow_policy": "Flow"}


def tab_synthetic():
    d = load("synthetic/summary.json")
    if not d:
        return
    obf = d["obfuscation"]
    cols = [c for c in ("keyword_filter", "text_only_detector", "information_flow", "permission_gate", "camel_capability", "detector_only_neural",
                        "R1", "R1_norm", "R2", "R2_norm", "R1_neural", "R2_neural") if c in obf["clean"]]
    order = ("clean", "zero_width", "leetspeak", "homoglyph", "base64", "spaced", "paraphrase")
    lab = {"clean": "Clean", "zero_width": "Zero-width", "leetspeak": "Leetspeak", "homoglyph": "Homoglyph", "base64": "Base64", "spaced": "Spacing", "paraphrase": "Paraphrase"}
    rows = [f"{lab[o]} & " + " & ".join(f"{obf[o][c]:.2f}" for c in cols) + " \\\\" for o in order]
    write("tab_obf.tex", "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\nObfuscation & " + " & ".join(SYN_LABEL[c] for c in cols)
          + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    ad = d["adaptive"]
    meths = [m for m in ("permission_gate", "information_flow", "camel_capability", "text_only_detector", "detector_only_neural", "R1", "R2", "R1_neural", "R2_neural") if m in ad]
    strat = ("S1_multistep", "S2_delayed", "S3_laundering", "S4_intermediate_tool", "S5_memory", "S6_colluding", "S7_lexical_evasion")
    slab = {"S1_multistep": "S1 multi-step exfiltration", "S2_delayed": "S2 delayed execution", "S3_laundering": "S3 data laundering",
            "S4_intermediate_tool": "S4 benign intermediate tool", "S5_memory": "S5 memory-mediated", "S6_colluding": "S6 colluding sources",
            "S7_lexical_evasion": "S7 lexical evasion"}
    rows = []
    for s in strat + ("all",):
        cells = []
        for m in meths:
            x = ad[m][s]
            cells.append(f"{x['tp']}/{x['fp']}")
        rows.append(("\\midrule\n" if s == "all" else "") + f"{slab.get(s, 'All (140 attacks / 140 benign)')} & " + " & ".join(cells) + " \\\\")
    write("tab_adaptive.tex", "\\begin{tabular}{@{}l" + "c" * len(meths) + "@{}}\n\\toprule\nStrategy & " + " & ".join(SYN_LABEL[m] for m in meths)
          + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    pv = d["privacy"]
    meths = [m for m in ("keyword_filter", "text_only_detector", "output_sensitive_filter", "flow_policy", "information_flow", "permission_gate",
                         "camel_capability", "detector_only_neural", "R1", "R2", "R2_neural") if m in pv]
    rows = [f"{SYN_LABEL[m]} & {pv[m]['recall']:.2f} & {pv[m]['fpr']:.2f} ({pv[m]['fp']}/{pv[m]['fp'] + pv[m]['tn']}) & {pv[m]['f1']:.2f} \\\\" for m in meths]
    write("tab_privacy.tex", "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nMethod & Recall & FPR (FP) & F1 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    ov = d["overrefusal"]
    meths = [m for m in ("keyword_filter", "text_only_detector", "output_sensitive_filter", "flow_policy", "permission_gate", "information_flow",
                         "camel_capability", "detector_only_neural", "R1", "R2", "R2_neural") if m in ov]
    rows = [f"{SYN_LABEL[m]} & {ov[m]['B1']:.2f} & {ov[m]['B2']:.2f} & {ov[m]['B3']:.2f} & {ov[m]['B4']:.2f} & {ov[m]['overall']:.2f} \\\\" for m in meths]
    write("tab_overrefusal.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nMethod & B1 auth.\\ sink & B2 sec.\\ keywords & B3 high-impact & B4 sens.\\ read & Overall \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    ex = d["extended"]
    meths = [m for m in ("keyword_filter", "text_only_detector", "output_sensitive_filter", "flow_policy", "permission_gate", "information_flow",
                         "camel_capability", "detector_only_neural", "R1", "R2", "R1_neural", "R2_neural") if m in ex]
    rows = [f"{SYN_LABEL[m]} & {ex[m]['recall']:.2f} & {ex[m]['fpr']:.2f} ({ex[m]['fp']}/{ex[m]['fp'] + ex[m]['tn']}) & {ex[m]['f1']:.2f} \\\\" for m in meths]
    write("tab_extended.tex", "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nMethod & Recall & FPR (FP) & F1 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    tab_e2e()
    tab_e2e_family()
    tab_trace()
    tab_replay()
    tab_replay_exact()
    tab_ablation()
    tab_detectors()
    tab_noise()
    tab_sensitivity()
    tab_expl()
    tab_runtime()
    tab_synthetic()
