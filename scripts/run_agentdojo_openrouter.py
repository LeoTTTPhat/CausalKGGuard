#!/usr/bin/env python3
"""Run AgentDojo (v1.2.2) with any OpenAI-compatible model, with or without CausalKGGuard.

Added for the R2 revision so that the end-to-end evaluation can be repeated on
agent models that are not in AgentDojo's built-in model list (for example
Llama-3.3-70B-Instruct or Gemini-2.5-Flash-Lite served through OpenRouter).

The pipeline is the standard AgentDojo tool-calling pipeline
(system message -> user query -> LLM -> tool-execution loop). With
``--defense causalkgguard`` the tool executor is replaced by the CausalKGGuard
executor, which evaluates every candidate tool call before it is dispatched.

Work is split into one subprocess per (suite, user task); AgentDojo skips
cases whose log already exists, so the runner can be stopped and restarted at
any time without repeating finished cases.

Example (no defense, important_instructions):
  .venv/bin/python run_agentdojo_openrouter.py --model-id meta-llama/llama-3.3-70b-instruct \
      --model-tag llama-3.3-70b-instruct --model-prose "AI assistant" \
      --attack important_instructions --logdir results/agentdojo_runs/llama33_none --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SUITES = ("workspace", "travel", "banking", "slack")
EVAL_ROOT = Path("/Users/sonha/CausalKGGuard_eval")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True, help="Model id sent to the OpenAI-compatible endpoint.")
    p.add_argument("--model-tag", required=True, help="Short model tag used in the pipeline name and log path.")
    p.add_argument("--model-prose", default="AI assistant",
                   help="How the important_instructions attack addresses the model (AgentDojo MODEL_NAMES).")
    p.add_argument("--defense", choices=("none", "causalkgguard"), default="none")
    p.add_argument("--guard-kwargs", default="{}", help="JSON kwargs for the CausalKGGuard executor.")
    p.add_argument("--guard-tag", default="r1", help="Tag of the guard configuration (pipeline name suffix).")
    p.add_argument("--package-root", type=Path, default=EVAL_ROOT / "frozen_r1",
                   help="Directory containing the causalkgguard package to import.")
    p.add_argument("--policy-config", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--benchmark-version", default="v1.2.2")
    p.add_argument("--attack", default="important_instructions",
                   help="Attack name, 'none' for benign utility runs, or a comma-separated list run in order.")
    p.add_argument("--include-injection-utility", action="store_true")
    p.add_argument("--suite", action="append", choices=DEFAULT_SUITES, dest="suites")
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--shard-logdir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--env-file", type=Path, default=EVAL_ROOT / ".env")
    p.add_argument("--max-user-tasks", type=int, default=None, help="Smoke-test limit per suite.")
    p.add_argument("--max-injection-tasks", type=int, default=None, help="Smoke-test limit per user task.")
    p.add_argument("--run-suite", default=None, help=argparse.SUPPRESS)
    p.add_argument("--run-task", default=None, help=argparse.SUPPRESS)
    p.add_argument("--run-task-kind", choices=("user", "injection_utility"), default="user", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.run_suite and args.run_task:
        run_one(args)
        return
    attacks = [a.strip() for a in args.attack.split(",") if a.strip()]
    failed = False
    for attack in attacks:
        sub = argparse.Namespace(**vars(args))
        sub.attack = attack
        try:
            orchestrate(sub)
        except SystemExit as exc:
            failed = failed or bool(exc.code)
    if failed:
        raise SystemExit(1)


def pipeline_name(args) -> str:
    suffix = "none" if args.defense == "none" else f"causalkgguard-{args.guard_tag}"
    return f"{args.model_tag}-{suffix}"


def orchestrate(args) -> None:
    sys.path.insert(0, str(args.package_root))
    from agentdojo.task_suite.load_suites import get_suite

    attack = norm_attack(args.attack)
    shard_dir = args.shard_logdir or (args.logdir.parent / f"{args.logdir.name}_shards" / attack)
    shard_dir.mkdir(parents=True, exist_ok=True)
    name = pipeline_name(args)
    shards = []
    for suite_name in args.suites or DEFAULT_SUITES:
        suite = get_suite(args.benchmark_version, suite_name)
        n_inj = len(suite.injection_tasks) if args.max_injection_tasks is None else min(
            args.max_injection_tasks, len(suite.injection_tasks))
        user_ids = sorted(suite.user_tasks, key=task_key)
        if args.max_user_tasks is not None:
            user_ids = user_ids[: args.max_user_tasks]
        kinds = [(uid, "user") for uid in user_ids]
        if attack == "none" and args.include_injection_utility:
            kinds += [(iid, "injection_utility") for iid in sorted(suite.injection_tasks, key=task_key)]
        for task_id, kind in kinds:
            if attack == "none":
                expected = 1
                have = int((args.logdir / name / suite_name / task_id / "none" / "none.json").exists())
            else:
                expected = n_inj
                d = args.logdir / name / suite_name / task_id / attack
                have = sum(1 for _ in d.glob("*.json")) if d.exists() else 0
            if have < expected:
                shards.append((suite_name, task_id, kind))
    print(f"[{time.strftime('%H:%M:%S')}] {name} attack={attack}: {len(shards)} incomplete shards", flush=True)
    if not shards:
        return
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(run_shard, s, args, shard_dir) for s in shards]
        for fut in concurrent.futures.as_completed(futs):
            shard, code = fut.result()
            print(f"[{time.strftime('%H:%M:%S')}] {'ok' if code == 0 else 'FAILED'} {shard}", flush=True)
            if code != 0:
                failures.append(shard)
    if failures:
        print(f"{len(failures)} shards failed", flush=True)
        raise SystemExit(1)


def run_shard(shard, args, shard_dir: Path):
    suite_name, task_id, kind = shard
    cmd = [sys.executable, str(Path(__file__).resolve())]
    for key in ("model_id", "model_tag", "model_prose", "defense", "guard_kwargs", "guard_tag",
                "package_root", "threshold", "benchmark_version", "attack", "logdir", "env_file"):
        cmd += [f"--{key.replace('_', '-')}", str(getattr(args, key))]
    if args.policy_config:
        cmd += ["--policy-config", str(args.policy_config)]
    if args.max_injection_tasks is not None:
        cmd += ["--max-injection-tasks", str(args.max_injection_tasks)]
    cmd += ["--run-suite", suite_name, "--run-task", task_id, "--run-task-kind", kind]
    log_path = shard_dir / f"{suite_name}_{kind}_{task_id}.log"
    code = 1
    with log_path.open("a", encoding="utf-8") as fh:
        for attempt in range(1, args.attempts + 1):
            fh.write(f"\n== attempt {attempt} {time.strftime('%Y-%m-%d %H:%M:%S')} ==\n")
            fh.flush()
            code = subprocess.run(cmd, cwd=str(EVAL_ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
            if code == 0:
                break
            time.sleep(10 * attempt)
    return shard, code


def _patch_agentdojo(model_tag: str, model_prose: str) -> None:
    """Make AgentDojo accept an arbitrary OpenAI-compatible model."""
    import agentdojo.models as models
    from agentdojo.agent_pipeline.llms import openai_llm

    models.MODEL_NAMES.setdefault(model_tag, model_prose)

    def robust_tool_call(tool_call):
        raw = tool_call.function.arguments or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Some providers emit trailing text after the JSON object; keep the first object.
            try:
                parsed, _ = json.JSONDecoder().raw_decode(raw.strip())
            except json.JSONDecodeError:
                parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        from agentdojo.functions_runtime import FunctionCall
        return FunctionCall(function=tool_call.function.name, args=parsed, id=tool_call.id)

    openai_llm._openai_to_tool_call = robust_tool_call


def build_pipeline(args, allowed_tools):
    import openai
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

    client = openai.OpenAI(timeout=180.0, max_retries=4)
    llm = OpenAILLM(client, args.model_id)
    if args.defense == "none":
        executor = ToolsExecutor()
    else:
        from causalkgguard.adapters.agentdojo_defense import CausalKGGuardToolsExecutor

        kwargs = json.loads(args.guard_kwargs or "{}")
        executor = CausalKGGuardToolsExecutor(
            allowed_tools=allowed_tools,
            threshold=args.threshold,
            policy_config=args.policy_config,
            **kwargs,
        )
    loop = ToolsExecutionLoop([executor, llm])
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm, loop])
    pipeline.name = pipeline_name(args)
    return pipeline


def run_one(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(str(args.env_file), override=True)
    sys.path.insert(0, str(args.package_root))
    _patch_agentdojo(args.model_tag, args.model_prose)

    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite

    from causalkgguard.adapters.agentdojo_defense import allowed_tools_from_task

    suite = get_suite(args.benchmark_version, args.run_suite)
    task = (suite.get_injection_task_by_id(args.run_task) if args.run_task_kind == "injection_utility"
            else suite.get_user_task_by_id(args.run_task))
    allowed = allowed_tools_from_task(suite, task)
    pipeline = build_pipeline(args, allowed)
    args.logdir.mkdir(parents=True, exist_ok=True)
    attack = norm_attack(args.attack)
    with OutputLogger(str(args.logdir)):
        if attack == "none":
            u, s = run_task_without_injection_tasks(suite, pipeline, task, args.logdir, False,
                                                    args.benchmark_version)
            print(f"{args.run_suite}/{args.run_task}: utility={u} security={s}")
        else:
            attacker = load_attack(attack, suite, pipeline)
            inj = None
            if args.max_injection_tasks is not None:
                inj = sorted(suite.injection_tasks, key=task_key)[: args.max_injection_tasks]
            u, s = run_task_with_injection_tasks(suite, pipeline, task, attacker, args.logdir, False,
                                                 injection_tasks=inj,
                                                 benchmark_version=args.benchmark_version)
            print(f"{args.run_suite}/{args.run_task}: utility={sum(u.values())}/{len(u)} "
                  f"security={sum(s.values())}/{len(s)}")


def task_key(task_id: str):
    m = re.search(r"(\d+)$", task_id)
    return (task_id.rsplit("_", 1)[0], int(m.group(1)) if m else 0)


def norm_attack(value: str | None) -> str:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return "none"
    return value.strip()


if __name__ == "__main__":
    main()
