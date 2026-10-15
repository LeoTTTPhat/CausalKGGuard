#!/usr/bin/env python3
"""Export the controlled benchmarks of R1 (and the R2 adaptive benchmark) as JSONL.

The generators are those of ``scripts/run_review_response_eval.py`` (R1) and
are reused unchanged, so that R1 and R2 configurations, baselines and the
neural detector are evaluated on identical cases.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ART = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ART))
sys.path.insert(0, str(ART / "scripts"))

import run_review_response_eval as rre  # noqa: E402

OUT = ART / "data" / "r2_synthetic"


def dump(name, cases):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(asdict(c)) + "\n")
    print(name, len(cases))


def main() -> None:
    ext = rre.load_extended()
    dump("extended", ext)
    attacks = [c for c in ext if c.label == "attack"]
    for obf, fn in rre.OBFUSCATIONS.items():
        dump(f"obfuscation_{obf}", [rre.obfuscate_case(c, fn) for c in attacks])
    dump("privacy", [rre.privacy_attack(i, "full") for i in range(30)]
         + [rre.privacy_benign(i, "authorized_send") for i in range(15)]
         + [rre.privacy_benign(i, "internal") for i in range(15)])
    for manifest in ("full", "partial", "refused"):
        dump(f"privacy_manifest_{manifest}", [rre.privacy_attack(i, manifest) for i in range(30)])
    dump("overrefusal", [rre.benign_b1(i) for i in range(20)] + [rre.benign_b2(i) for i in range(20)]
         + [rre.benign_b3(i) for i in range(20)] + [rre.benign_b4(i) for i in range(20)])
    dump("decoy", [rre.cf_causal_case(i) for i in range(30)] + [rre.cf_decoy_case(i) for i in range(30)])


if __name__ == "__main__":
    main()
