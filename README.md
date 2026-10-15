# CausalKGGuard

CausalKGGuard is a lightweight Python prototype for causal security-graph reasoning over agent execution traces. It models instructions, data, tool calls, permissions, trust boundaries, and external sinks, then applies symbolic policies and counterfactual-style scoring to flag unsafe actions.

## Structure

- `causalkgguard/` - trace schema, detectors, graph-grounded risk scoring, policy decisions, and baselines.
- `configs/` - symbolic policy configuration.
- `scripts/` - evaluation, summarization, threshold sweep, ablation, and runtime scripts.
- `tests/` - unit tests.
- `Exp_Results/` - aggregate evaluation tables and findings, without raw trace transcripts.

## Run the pilot evaluation

```bash
python3 scripts/run_synthetic_eval.py
```

This writes aggregate tables under `results/tables/`:

- `results/tables/synthetic_eval_summary.csv`
- `results/tables/synthetic_eval_cases.csv`
- `results/tables/synthetic_eval_by_attack_type.csv`
- `results/tables/synthetic_eval_guard_findings.csv`
- `results/tables/synthetic_eval_report.md`

## Run the extended local evaluation

```bash
python3 scripts/run_all_local_eval.py
```

Or run each stage manually:

```bash
python3 scripts/generate_extended_synthetic.py
python3 scripts/run_synthetic_eval.py \
  --data data/synthetic/trace_cases_extended.jsonl \
  --prefix extended_synthetic_eval \
  --policy-config configs/policies.json
python3 scripts/run_ablation_eval.py \
  --data data/synthetic/trace_cases_extended.jsonl \
  --policy-config configs/policies.json
python3 scripts/run_explanation_eval.py \
  --data data/synthetic/trace_cases_extended.jsonl \
  --policy-config configs/policies.json
python3 scripts/run_statistical_analysis.py \
  --data data/synthetic/trace_cases_extended.jsonl \
  --policy-config configs/policies.json
python3 scripts/run_runtime_eval.py
```

This writes extended benchmark, baseline, ablation, explanation, statistical, and runtime reports under `results/tables/`.

## Validate the artifact

```bash
python3 -m compileall causalkgguard scripts tests
python3 -m unittest discover -s tests
python3 scripts/validate_trace_cases.py --data data/synthetic/trace_cases_extended.jsonl
```

## Notes

Raw execution transcripts and local environment files are intentionally excluded from the public artifact. The included `Exp_Results/` folder contains summarized tables and findings only.

## Revision R2 (ASOC-D-26-09638): what was added

The engine keeps the original behaviour by default (verified bit-exact on all 4,842
GPT-4o-mini AgentDojo traces by `scripts/r2/check_r1_equivalence.py`). New, opt-in options:

* `causalkgguard/provenance.py` and `graph_from_trace(..., dependency_mode="provenance")`:
  provenance-aware dependencies (value-level data provenance, inverted index, at most
  `max_parents=8` data parents per tool call; responses keep prefix dependencies).
* `CausalKGGuard(decision_rule="attribution", lift_threshold=0.20)`: attribution-gated rule
  (intercept iff R(a|tau^-U) >= theta, or consequential and Delta_U(a) >= delta);
  `decide_candidate()` scores only the candidate call; `RiskWeights` exposes all 22 coefficients.
* `adapters/agentdojo_defense.py`: `decision_scope="candidate"` for the live executor.

Configurations used in the paper (theta = 0.60):

| Name | dependency_mode | decision_rule | decision scope | live package |
|---|---|---|---|---|
| R1 (original) | prefix | score | trace-wide maximum | `frozen_r1` |
| R2 (revised)  | provenance | attribution (delta = 0.20) | candidate | `frozen_r2` (hash in FROZEN_HASH.txt) |

### Scripts (scripts/r2)

| Script | Produces |
|---|---|
| `run_agentdojo_openrouter.py` | live AgentDojo runs for any OpenAI-compatible model, with or without the monitor |
| `build_agentdojo_ground_truth.py` | AgentDojo task ground truth (needs `agentdojo`) used only to label actions |
| `replay_actions.py` | exact replay of in-loop decisions; per-action risk decompositions |
| `analyze_replay.py`, `results_summary.py` | action-level metrics, ablations, task-cluster bootstrap CIs (Tables 13, 17, 20) |
| `trace_level_eval.py` | label-independent trace-level comparison with baselines (Tables 11, 12) |
| `e2e_live_eval.py` | live end-to-end statistics (Tables 9, 10) |
| `sensitivity.py` | threshold, delta, and weight sensitivity (Figure 8, Table 18) |
| `run_allowlist_noise.sh`, `allowlist_noise_summary.py` | annotation-noise experiment (Table 19) |
| `explanation_eval_r2.py` | explanation quality against programmatic ground truth (Table 14) |
| `runtime_r2.py`, `time_neural_detector.py` | runtime on real and synthetic traces, detector latency (Tables 15, 16) |
| `neural_detector_qwen.py` | zero-shot Qwen2.5-7B-Instruct prompt-injection detector (cached window scores) |
| `generate_adaptive_benchmark.py`, `export_synthetic_suite.py`, `synthetic_eval_r2.py` | adaptive and controlled benchmarks (Tables 21-25) |
| `make_tables.py` | all LaTeX tables and figure data of the manuscript from the result files |

Order: build ground truth -> replay (lexical, generic, neural) -> analyses -> `make_tables.py`.
Random seeds are fixed in every script (bootstrap 0-11, noise 0-2, perturbations 25/50,
adaptive generator 2026). Live agents use provider-default sampling and cannot be seeded.

### Corrections relative to the original evaluation

* The original log converter derived the task allowlist from the trace label
  (`_required_permissions`); `r2lib.reannotate` replaces it with the label-independent
  allowlist of the live runs (user-task ground-truth tools) and the live sink map.
* Explanation ground truth no longer uses the monitor's own detector.
* Runtime is reported per in-loop decision, not amortized per action.
