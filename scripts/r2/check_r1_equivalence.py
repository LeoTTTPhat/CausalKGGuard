"""Check that the R2 engine with default settings reproduces the frozen R1 engine exactly."""
import importlib, json, sys, time
from pathlib import Path

DATA = Path("/Users/sonha/CausalKGGuard_eval/data/agentdojo")
FILES = [f"agentdojo_full_gpt4omini_{a}_attempt_noinjutil_traces.jsonl" for a in
         ("direct", "ignore_previous", "important", "system_message", "tool_knowledge")] + ["agentdojo_full_gpt4omini_noattack_traces.jsonl"]

def load(pkg_root):
    for m in [m for m in sys.modules if m == "causalkgguard" or m.startswith("causalkgguard.")]:
        del sys.modules[m]
    sys.path.insert(0, pkg_root)
    mod = importlib.import_module("causalkgguard")
    sys.path.pop(0)
    return mod

rows = [json.loads(l) for f in FILES for l in (DATA / f).read_text().splitlines() if l.strip()]
out = {}
for name, root in (("r1", "/Users/sonha/CausalKGGuard_eval/frozen_r1"), ("r2", "/Users/sonha/Documents/SonHa/CausalKGGuard/ManuscriptR2/artifact")):
    ck = load(root)
    guard = ck.CausalKGGuard(threshold=0.60, finding_threshold=0.0, policy_path="/Users/sonha/CausalKGGuard_eval/frozen_r1/policies.json")
    t = time.time(); res = []
    for r in rows:
        tc = ck.TraceCase.from_dict(r)
        fs = guard.evaluate(tc)
        res.append((guard.predict_attack(tc), tuple((f.event_id, round(f.risk_score, 12), round(f.causal_lift, 12)) for f in fs)))
    out[name] = res
    print(name, "done", round(time.time() - t, 1), "s", "attacks flagged", sum(p for p, _ in res))
diff = sum(a != b for a, b in zip(out["r1"], out["r2"]))
print("traces:", len(rows), "differences:", diff)
