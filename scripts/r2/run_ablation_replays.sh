#!/bin/zsh
cd "$(dirname "$0")"
for comp in semantic_detector sensitive_flow untrusted_sensitive_context permission_policy system_prompt_policy; do
  /usr/bin/python3 replay_actions.py --model gpt-4o-mini --deps prefix,provenance --detectors lexical --allowlist gt --excess all --disable $comp
done
echo ABLATION_DONE
