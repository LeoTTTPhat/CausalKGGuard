#!/usr/bin/env python3
"""Replay the monitor over recorded AgentDojo traces and store per-action risk decompositions.

For each trace and configuration (dependency construction x semantic detector)
the monitor scores every action and output from its own prefix, which is what
the in-loop monitor would see at that point. Decisions under different
enforcement rules are computed later from the stored (R, R^{-U}, Delta_U)
triples by ``analyze_replay.py``.

Output: results/replay/<model>/<dependency>__<detector>.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import time

import r2lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--deps", default="prefix,provenance")
    ap.add_argument("--detectors", default="lexical,generic")
    ap.add_argument("--max-parents", type=int, default=8)
    ap.add_argument("--allowlist", default="gt", help="gt | drop:p | add:q | suite | none | heuristic | r1label")
    ap.add_argument("--excess", default="all,consequential")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disable", default="", help="comma-separated components to ablate (feature masks)")
    args = ap.parse_args()

    import random

    rng = random.Random(args.seed)
    cases = r2lib.load_cases(r2lib.model_files(args.model))
    if args.allowlist != "r1label":  # r1label keeps R1's label-dependent metadata (for comparison only)
        cases = [r2lib.reannotate(c, args.allowlist, rng) for c in cases]
    labels = {c["case_id"]: r2lib.label_actions(c) for c in cases}
    utexts = r2lib.untrusted_texts(cases)
    tag = args.allowlist.replace(":", "") + (f"_s{args.seed}" if args.allowlist.startswith(("drop", "add")) else "")
    outdir = r2lib.RESULTS / "replay" / args.model / tag
    outdir.mkdir(parents=True, exist_ok=True)
    for det in args.detectors.split(","):
        if det == "lexical":
            insp = r2lib.inspect_text
        elif det == "generic":
            insp = r2lib.inspect_text_generic
        elif det == "neural":
            insp = r2lib.make_neural_inspector(utexts)
        elif det == "neural_lexical":
            insp = r2lib.make_neural_inspector(utexts, union_with_lexical=True)
        else:
            raise SystemExit(f"unknown detector {det}")
        for dep in args.deps.split(","):
          for excess in args.excess.split(","):
            dis = [d for d in args.disable.split(",") if d]
            guard = r2lib.make_guard(dep, insp, max_parents=args.max_parents, excess_scope=excess,
                                     disabled_components=dis)
            t0 = time.time()
            suffix = ("__minus-" + "-".join(dis)) if dis else ""
            path = outdir / f"{dep}__{det}__ex-{excess}{suffix}.jsonl.gz"
            missing = 0
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                for c in cases:
                    md = c.get("metadata", {})
                    try:
                        rows = r2lib.action_rows(c, guard, labels[c["case_id"]])
                    except r2lib.MissingNeuralScore:
                        missing += 1
                        continue
                    lab = labels[c["case_id"]]
                    fh.write(json.dumps({
                        "case_id": c["case_id"],
                        "suite": md.get("suite"),
                        "user_task": r2lib.user_task_key(c),
                        "injection_task": r2lib.injection_task_key(c),
                        "family": r2lib.family_of(c),
                        "label": c.get("label"),
                        "utility": str(md.get("utility")),
                        "security": str(md.get("security")),
                        "n_events": len(c["events"]),
                        "inj_bearing": [e for e, v in lab.items() if v.get("injection_bearing")],
                        "rows": rows,
                    }) + "\n")
            print(f"{args.model} [{tag}] {dep} {det} ex-{excess}: {len(cases)} traces in {time.time() - t0:.1f}s"
                  f"{' (missing neural scores: %d)' % missing if missing else ''}", flush=True)


if __name__ == "__main__":
    main()
