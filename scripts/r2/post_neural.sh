#!/bin/zsh
# Run after the LLM-detector scores are cached for a model: replay with the neural detector,
# then refresh every summary and table. Usage: ./post_neural.sh gpt-4o-mini [gemini-2.5-flash-lite ...]
cd "$(dirname "$0")"
for model in "$@"; do
  /usr/bin/python3 replay_actions.py --model "$model" --deps prefix,provenance --detectors neural --allowlist gt --excess all
done
/usr/bin/python3 trace_level_eval.py --model gpt-4o-mini > ../../../ops/logs/trace_level.out 2>&1
/usr/bin/python3 results_summary.py > ../../../ops/logs/results_summary.out 2>&1
/usr/bin/python3 make_tables.py
echo POST_NEURAL_DONE
