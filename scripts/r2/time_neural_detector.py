#!/usr/bin/env python3
"""Latency of the LLM-based detector per window and per observation (R5, comment 10).

Run when no other job uses the GPU. Scores a fixed random sample of untrusted
observations from the GPT-4o-mini AgentDojo traces with batch size 1 (the
in-loop setting: one new observation at a time) and reports median, P95, and
P99 latency per window and per observation.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neural_detector_qwen as nd  # noqa: E402
import r2lib  # noqa: E402


def main(n_obs: int = 150) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = sorted(nd.collect_texts([str(p) for p in r2lib.model_files("gpt-4o-mini")]))
    rng = random.Random(7)
    sample = rng.sample(texts, n_obs)
    tok = AutoTokenizer.from_pretrained(nd.MODEL)
    model = AutoModelForCausalLM.from_pretrained(nd.MODEL, dtype=torch.bfloat16, attn_implementation="eager").to("mps")
    model.eval()
    yes = {tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")}
    no = {tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")}

    def score(w: str) -> float:
        prompt = tok.apply_chat_template([{"role": "system", "content": nd.SYSTEM},
                                          {"role": "user", "content": nd.USER.format(text=w)}],
                                         tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to("mps")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        pr = torch.softmax(logits, -1)[0]
        torch.mps.synchronize()
        py = float(sum(pr[j] for j in yes))
        pn = float(sum(pr[j] for j in no))
        return py / (py + pn)

    score("warm-up text")  # load kernels
    per_window, per_obs, n_windows = [], [], []
    for t in sample:
        t0 = time.perf_counter()
        ws = nd.windows(t)
        for w in ws:
            t1 = time.perf_counter()
            score(w)
            per_window.append((time.perf_counter() - t1) * 1000)
        per_obs.append((time.perf_counter() - t0) * 1000)
        n_windows.append(len(ws))

    def summ(x):
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
                "mean": float(np.mean(x)), "n": len(x)}

    # lexical detector latency on the same observations
    lex = []
    for t in sample:
        t0 = time.perf_counter()
        r2lib.inspect_text(t)
        lex.append((time.perf_counter() - t0) * 1000)
    out = {"llm_ms_per_window": summ(per_window), "llm_ms_per_observation": summ(per_obs),
           "windows_per_observation": summ(n_windows), "lexical_ms_per_observation": summ(lex),
           "setup": "Qwen2.5-7B-Instruct, bfloat16, eager attention, PyTorch MPS, batch size 1, Apple M3 Ultra"}
    (r2lib.RESULTS / "runtime" / "neural_detector.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
