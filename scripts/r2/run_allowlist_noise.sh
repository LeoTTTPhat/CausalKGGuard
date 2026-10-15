#!/bin/zsh
# Annotation-noise experiment (R5 comment 6): replay with degraded task allowlists.
cd "$(dirname "$0")"
for mode in heuristic suite none; do
  /usr/bin/python3 replay_actions.py --model gpt-4o-mini --deps prefix,provenance --detectors lexical --allowlist $mode --excess all
done
for mode in drop:0.25 drop:0.5 add:0.25 add:0.5; do
  for seed in 0 1 2; do
    /usr/bin/python3 replay_actions.py --model gpt-4o-mini --deps prefix,provenance --detectors lexical --allowlist $mode --seed $seed --excess all
  done
done
echo ALLOWLIST_NOISE_DONE
