#!/bin/zsh
# Regenerate all lexical/generic replay files with the current engine (R2 final).
cd "$(dirname "$0")"
/usr/bin/python3 replay_actions.py --model gpt-4o-mini --deps prefix,provenance --detectors lexical,generic --allowlist gt --excess all
/usr/bin/python3 replay_actions.py --model gemini-2.5-flash-lite --deps prefix,provenance --detectors lexical,generic --allowlist gt --excess all
for comp in semantic_detector sensitive_flow untrusted_sensitive_context permission_policy system_prompt_policy; do
  /usr/bin/python3 replay_actions.py --model gpt-4o-mini --deps prefix,provenance --detectors lexical --allowlist gt --excess all --disable $comp
done
for mode in heuristic suite none; do
  /usr/bin/python3 replay_actions.py --model gpt-4o-mini --deps prefix,provenance --detectors lexical --allowlist $mode --excess all
done
for mode in drop:0.25 drop:0.5 add:0.25 add:0.5; do
  for seed in 0 1 2; do
    /usr/bin/python3 replay_actions.py --model gpt-4o-mini --deps prefix,provenance --detectors lexical --allowlist $mode --seed $seed --excess all
  done
done
echo REGEN_DONE
