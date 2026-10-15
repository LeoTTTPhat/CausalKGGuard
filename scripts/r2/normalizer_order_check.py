#!/usr/bin/env python3
"""Explain the base64 result of Table (obfuscation): the normalizer of the original artifact
(run_review_response_eval.normalize_text) applies the leetspeak mapping before base64 decoding,
which corrupts encoded payloads. This script evaluates both configurations with a normalizer
that decodes base64 first and writes results/synthetic/normalizer_order.json."""

from __future__ import annotations

import json
import re
import sys
import unicodedata

import r2lib
from causalkgguard import TraceCase
from causalkgguard.detectors import inspect_text

sys.path.insert(0, str(r2lib.ART / "scripts"))
import run_review_response_eval as rre  # noqa: E402


def normalize_decode_first(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[​-‏﻿⁠]", "", t)
    inv = {v: k for k, v in rre.HOMOGLYPHS.items()}
    t = "".join(inv.get(c, c) for c in t)
    t = rre._decode_b64_segments(t)  # decode before the leetspeak mapping
    t = rre._collapse_spaced_letters(t)
    return rre._deleet(t)


def main() -> None:
    insp = lambda s: inspect_text(normalize_decode_first(s))  # noqa: E731
    guards = {"R1_norm_decode_first": r2lib.make_guard("prefix", insp),
              "R2_norm_decode_first": r2lib.make_guard("provenance", insp, decision_rule="attribution")}
    out = {}
    for name in ("clean", "zero_width", "leetspeak", "homoglyph", "base64", "spaced", "paraphrase"):
        cases = [json.loads(l) for l in (r2lib.ART / "data" / "r2_synthetic" / f"obfuscation_{name}.jsonl").read_text().splitlines() if l.strip()]
        att = [c for c in cases if c["label"] == "attack"]
        out[name] = {g: sum(any(f.evidence.get("intercept") for f in guard.evaluate(TraceCase.from_dict(c))) for c in att) / len(att)
                     for g, guard in guards.items()}
    dest = r2lib.RESULTS / "synthetic" / "normalizer_order.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
