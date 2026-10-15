#!/usr/bin/env python3
"""Threshold and weight sensitivity of the risk score (R1 comment 5, R5 comment 8).

Risk scores are recomputed analytically from the per-action feature vectors
stored in the replay files (verified to reproduce the engine exactly), so the
sweeps do not re-run the monitor.

Writes results/sensitivity/<model>.json
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np

import analyze_replay as ar
import r2lib
from causalkgguard.reasoning import DEFAULT_INTERACTIONS, DEFAULT_LINEAR_WEIGHTS, U_DEPENDENT_FEATURES

FEATURES = [k for k, _ in DEFAULT_LINEAR_WEIGHTS]
INTER = [(a, b) for a, b, _ in DEFAULT_INTERACTIONS]


def score(feat, lw, iw):
    s = 0.0
    for k, w in lw:
        s += w * feat.get(k, 0.0)
    for a, b, w in iw:
        if feat.get(a) and feat.get(b):
            s += w
    return min(1.0, s)


def rescore(traces, lw, iw):
    out = []
    for t in traces:
        rows = []
        for r in t["rows"]:
            f = r["feat"]
            R = score(f, lw, iw)
            Rcf = score({k: v for k, v in f.items() if k not in U_DEPENDENT_FEATURES}, lw, iw)
            rows.append({**r, "R": R, "Rcf": Rcf, "lift": max(0.0, R - Rcf)})
        out.append({**t, "rows": rows})
    return out


KEYS = ["trace_recall", "trace_fpr", "trace_f1", "inj_interception", "succ_attack_prevention",
        "replay_residual_asr", "leg_overblock_attack", "leg_overblock_benign", "tool_trace_fpr"]


def pick(m):
    return {k: m[k] for k in KEYS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--parts", default="all",
                    help="comma-separated subset of theta,delta,weights,learned; other parts are kept from the existing file")
    args = ap.parse_args()
    parts = {"theta", "delta", "weights", "learned"} if args.parts == "all" else set(args.parts.split(","))
    pre = ar.load_replay(args.model, "prefix", "lexical", "all", "gt")
    pro = ar.load_replay(args.model, "provenance", "lexical", "all", "gt")
    out_file = r2lib.RESULTS / "sensitivity" / f"{args.model}.json"
    res = {"theta_sweep": [], "delta_sweep": [], "weights": {}, "learned": {}}
    if args.parts != "all" and out_file.exists():
        res = json.loads(out_file.read_text())
    if "theta" in parts:
        res["theta_sweep"] = []
        for theta in [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]:
            # "r1": prefix + score rule with candidate-only scoring; "r1_tracewide": the original
            # executor (trace-wide maximum), i.e. the R1 configuration itself
            res["theta_sweep"].append({"theta": theta,
                                       "r1": pick(ar.metrics(pre, "score", theta, 0.2, "candidate")),
                                       "r1_tracewide": pick(ar.metrics(pre, "score", theta, 0.2, "sticky")),
                                       "r2": pick(ar.metrics(pro, "attr", theta, 0.2, "candidate"))})
    if "delta" in parts:
        res["delta_sweep"] = []
        res["delta_sweep_calibration"] = []  # the check made before freezing CKG-A (40 calibration tasks)
        cal_pro = [t for t in pro if ar.split_of(t["user_task"]) == "calib"]
        for delta in [0.10, 0.14, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
            res["delta_sweep"].append({"delta": delta, "r2": pick(ar.metrics(pro, "attr", 0.60, delta, "candidate"))})
            m = ar.metrics(cal_pro, "attr", 0.60, delta, "candidate")
            res["delta_sweep_calibration"].append({"delta": delta, "r2": pick(m), "benign_tasks_intercepted": m["n_ben_flag_tool"],
                                                   "benign_tasks": m["n_ben"]})

    lw0, iw0 = DEFAULT_LINEAR_WEIGHTS, DEFAULT_INTERACTIONS
    for spread in ((0.25, 0.50) if "weights" in parts else ()):
        rng = random.Random(int(spread * 100))
        draws = []
        for _ in range(args.draws):
            lw = tuple((k, w * rng.uniform(1 - spread, 1 + spread)) for k, w in lw0)
            iw = tuple((a, b, w * rng.uniform(1 - spread, 1 + spread)) for a, b, w in iw0)
            m1 = ar.metrics(rescore(pre, lw, iw), "score", 0.60, 0.2, "candidate")
            m2 = ar.metrics(rescore(pro, lw, iw), "attr", 0.60, 0.2, "candidate")
            draws.append({"r1": pick(m1), "r2": pick(m2)})
        summ = {}
        for cfg in ("r1", "r2"):
            summ[cfg] = {k: (float(np.median([d[cfg][k] for d in draws])),
                             float(np.percentile([d[cfg][k] for d in draws], 2.5)),
                             float(np.percentile([d[cfg][k] for d in draws], 97.5))) for k in KEYS}
        summ["r2_dominates_frac"] = float(np.mean([
            d["r2"]["replay_residual_asr"] <= d["r1"]["replay_residual_asr"] and
            d["r2"]["leg_overblock_attack"] <= d["r1"]["leg_overblock_attack"] for d in draws]))
        res["weights"][f"pm{int(spread * 100)}"] = summ
    # uniform weights (same total mass)
    mu_l = float(np.mean([w for _, w in lw0]))
    mu_i = float(np.mean([w for _, _, w in iw0]))
    lw_u = tuple((k, mu_l) for k, _ in lw0)
    iw_u = tuple((a, b, mu_i) for a, b, _ in iw0)
    if "weights" in parts:
        res["uniform"] = {"r1": pick(ar.metrics(rescore(pre, lw_u, iw_u), "score", 0.60, 0.2, "candidate")),
                          "r2": pick(ar.metrics(rescore(pro, lw_u, iw_u), "attr", 0.60, 0.2, "candidate"))}

    # data-calibrated alternative: logistic regression on the calibration split
    from sklearn.linear_model import LogisticRegression

    def xy(traces):
        X, y = [], []
        for t in traces:
            for r in t["rows"]:
                if r["role"] != "action" or r["kind"] not in ("injected", "legitimate"):
                    continue
                f = r["feat"]
                X.append([f.get(k, 0.0) for k in FEATURES] + [float(bool(f.get(a) and f.get(b))) for a, b in INTER])
                y.append(1 if r["kind"] == "injected" else 0)
        return np.array(X), np.array(y)

    def fit_nonnegative(X, y, C=1.0):
        """Class-balanced, L2-penalized logistic regression with coefficients constrained to be
        nonnegative (intercept free), so that the fitted score stays monotone (Proposition 1)."""
        from scipy.optimize import minimize

        cw = {c: len(y) / (2.0 * np.sum(y == c)) for c in (0, 1)}
        sw = np.array([cw[v] for v in y])
        ys = 2.0 * y - 1.0

        def obj(p):
            w, b = p[:-1], p[-1]
            m = ys * (X @ w + b)
            loss = np.sum(sw * np.logaddexp(0.0, -m))
            g = -sw * ys / (1.0 + np.exp(m))
            return 0.5 * w @ w + C * loss, np.append(w + C * (X.T @ g), C * g.sum())

        p0 = np.zeros(X.shape[1] + 1)
        bounds = [(0.0, None)] * X.shape[1] + [(None, None)]
        r = minimize(obj, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
        return r.x[:-1]

    for dep, traces in ((("prefix", pre), ("provenance", pro)) if "learned" in parts else ()):
        cal = [t for t in traces if ar.split_of(t["user_task"]) == "calib"]
        test = [t for t in traces if ar.split_of(t["user_task"]) == "test"]
        X, y = xy(cal)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)
        coef_unconstrained = clf.coef_[0]
        coef = fit_nonnegative(X, y)
        # rescale the nonnegative coefficients to a score on [0,1]
        pos = np.clip(coef, 0, None)
        scale = pos.sum() if pos.sum() > 0 else 1.0
        lw = tuple((k, float(pos[i] / scale)) for i, k in enumerate(FEATURES))
        iw = tuple((a, b, float(pos[len(FEATURES) + j] / scale)) for j, (a, b) in enumerate(INTER))
        # choose theta on calibration to match the hand-weighted benign/legit over-blocking on calibration
        target = ar.metrics(cal, "score" if dep == "prefix" else "attr", 0.60, 0.2, "candidate")["leg_overblock_attack"]
        best = None
        resc_cal = rescore(cal, lw, iw)
        for th in np.arange(0.02, 0.6, 0.01):
            m = ar.metrics(resc_cal, "score" if dep == "prefix" else "attr", th, min(0.2, th), "candidate")
            if m["leg_overblock_attack"] <= target and (best is None or m["succ_attack_prevention"] > best[1]):
                best = (float(th), m["succ_attack_prevention"])
        th = best[0] if best else 0.6
        m_test_learned = ar.metrics(rescore(test, lw, iw), "score" if dep == "prefix" else "attr", th, min(0.2, th), "candidate")
        m_test_hand = ar.metrics(test, "score" if dep == "prefix" else "attr", 0.60, 0.2, "candidate")
        names = FEATURES + [f"{a}*{b}" for a, b in INTER]
        res["learned"][dep] = {"theta": th, "test_learned": pick(m_test_learned), "test_hand": pick(m_test_hand),
                               "fit": "nonnegative logistic regression (L-BFGS-B), coefficients rescaled to sum to 1",
                               "coef": {k: float(coef[i]) for i, k in enumerate(names)},
                               "coef_unconstrained": {k: float(coef_unconstrained[i]) for i, k in enumerate(names)},
                               "weights_used": {k: float(pos[i] / scale) for i, k in enumerate(names)}}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=1))
    print("theta sweep (R1 score | R2 attr): theta, trace F1, trace FPR, residual ASR, legit OB")
    for row in res["theta_sweep"]:
        a, b = row["r1"], row["r2"]
        print(f"  {row['theta']:.2f}  R1 {a['trace_f1']:.3f} {a['trace_fpr']:.3f} {a['replay_residual_asr']:.4f} {a['leg_overblock_attack']:.3f}"
              f" | R2 {b['trace_f1']:.3f} {b['trace_fpr']:.3f} {b['replay_residual_asr']:.4f} {b['leg_overblock_attack']:.3f}")
    print("delta sweep (R2): delta, residual ASR, legit OB attack, benign tool FPR")
    for row in res["delta_sweep"]:
        b = row["r2"]
        print(f"  {row['delta']:.2f}  {b['replay_residual_asr']:.4f} {b['leg_overblock_attack']:.3f} {b['tool_trace_fpr']:.3f}")
    for k, v in res.get("weights", {}).items():
        print(k, "R2 dominates R1 in", v["r2_dominates_frac"], "of draws;",
              "R1 resASR", [round(x, 4) for x in v["r1"]["replay_residual_asr"]], "R2 resASR", [round(x, 4) for x in v["r2"]["replay_residual_asr"]],
              "R1 OB", [round(x, 3) for x in v["r1"]["leg_overblock_attack"]], "R2 OB", [round(x, 3) for x in v["r2"]["leg_overblock_attack"]],
              "R2 FPR", [round(x, 3) for x in v["r2"]["tool_trace_fpr"]])
    print("uniform", json.dumps(res.get("uniform"), indent=0)[:600])
    print("learned", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'coef'} for k, v in res["learned"].items()}, indent=0)[:1500])


if __name__ == "__main__":
    main()
