#!/usr/bin/env python3
"""Run AgentDojo with CausalKGGuard as an end-to-end runtime defense."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITES = ("workspace", "travel", "banking", "slack")


@dataclass(frozen=True)
class Shard:
    suite: str
    task_id: str
    task_kind: str
    existing_logs: int
    expected_logs: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="GPT_4O_MINI_2024_07_18")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--pipeline-name", default="gpt-4o-mini-2024-07-18-causalkgguard")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--attack", default="important_instructions", help="Use 'none' for no-attack utility runs.")
    parser.add_argument("--logdir", type=Path, default=ROOT / "results" / "agentdojo_runs" / "guarded_causalkgguard")
    parser.add_argument("--shard-logdir", type=Path, default=ROOT / "results" / "agentdojo_runs" / "guarded_sharded_logs")
    parser.add_argument("--suite", action="append", choices=DEFAULT_SUITES, dest="suites")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--policy-config", type=Path, default=ROOT / "configs" / "policies.json")
    parser.add_argument("--include-injection-utility", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-suite", choices=DEFAULT_SUITES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-task", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-task-kind", choices=("user", "injection_utility"), default="user", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_suite and args.run_task:
        run_one(args)
        return

    suites = tuple(args.suites or DEFAULT_SUITES)
    args.shard_logdir.mkdir(parents=True, exist_ok=True)
    shards = discover_shards(
        suites=suites,
        logdir=args.logdir,
        pipeline_name=args.pipeline_name,
        benchmark_version=args.benchmark_version,
        attack=normalize_attack(args.attack),
        include_injection_utility=args.include_injection_utility,
    )
    print(f"Queued {len(shards)} incomplete CausalKGGuard shards across {', '.join(suites)}")
    for shard in shards:
        print(
            f"- {shard.suite}/{shard.task_kind}/{shard.task_id}: "
            f"{shard.existing_logs}/{shard.expected_logs} logs"
        )

    if args.dry_run or not shards:
        return

    failures: List[tuple[Shard, int]] = []
    worker_count = min(args.workers, len(shards))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_shard, shard, args) for shard in shards]
        for future in concurrent.futures.as_completed(futures):
            shard, returncode = future.result()
            status = "ok" if returncode == 0 else f"failed:{returncode}"
            print(f"{status} {shard.suite}/{shard.task_kind}/{shard.task_id}")
            if returncode != 0:
                failures.append((shard, returncode))

    if failures:
        print("Failed shards:")
        for shard, returncode in failures:
            print(f"- {shard.suite}/{shard.task_kind}/{shard.task_id}: {returncode}")
        raise SystemExit(1)


def discover_shards(
    suites: Iterable[str],
    logdir: Path,
    pipeline_name: str,
    benchmark_version: str,
    attack: str,
    include_injection_utility: bool,
) -> List[Shard]:
    sys.path.insert(0, str(ROOT))
    try:
        from agentdojo.task_suite.load_suites import get_suite
    except ImportError as exc:
        raise SystemExit("agentdojo is required; install it in the active environment first") from exc

    shards: List[Shard] = []
    for suite_name in suites:
        suite = get_suite(benchmark_version, suite_name)
        for task_id in sorted(suite.user_tasks):
            shards.append(
                shard_for_task(logdir, pipeline_name, suite_name, task_id, "user", attack, len(suite.injection_tasks))
            )
        if attack == "none" and include_injection_utility:
            for task_id in sorted(suite.injection_tasks):
                shards.append(
                    shard_for_task(
                        logdir,
                        pipeline_name,
                        suite_name,
                        task_id,
                        "injection_utility",
                        attack,
                        len(suite.injection_tasks),
                    )
                )
    return [shard for shard in shards if shard.existing_logs < shard.expected_logs]


def shard_for_task(
    logdir: Path,
    pipeline_name: str,
    suite: str,
    task_id: str,
    task_kind: str,
    attack: str,
    injection_count: int,
) -> Shard:
    if attack == "none":
        expected_logs = 1
        existing_logs = int((logdir / pipeline_name / suite / task_id / "none" / "none.json").exists())
    else:
        expected_logs = injection_count
        existing_logs = count_existing_json(logdir / pipeline_name / suite / task_id / attack)
    return Shard(
        suite=suite,
        task_id=task_id,
        task_kind=task_kind,
        existing_logs=existing_logs,
        expected_logs=expected_logs,
    )


def count_existing_json(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.json") if item.is_file())


def run_shard(shard: Shard, args: argparse.Namespace) -> tuple[Shard, int]:
    log_path = args.shard_logdir / f"{slug(shard.suite)}_{slug(shard.task_kind)}_{slug(shard.task_id)}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--pipeline-name",
        args.pipeline_name,
        "--benchmark-version",
        args.benchmark_version,
        "--attack",
        args.attack,
        "--logdir",
        str(args.logdir),
        "--threshold",
        str(args.threshold),
        "--policy-config",
        str(args.policy_config),
        "--run-suite",
        shard.suite,
        "--run-task",
        shard.task_id,
        "--run-task-kind",
        shard.task_kind,
    ]
    if args.model_id:
        command.extend(["--model-id", args.model_id])
    if args.force_rerun:
        command.append("--force-rerun")

    returncode = 1
    with log_path.open("a", encoding="utf-8") as handle:
        for attempt in range(1, args.attempts + 1):
            handle.write(f"\n== attempt {attempt}: {' '.join(command)} ==\n")
            handle.flush()
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            returncode = completed.returncode
            if returncode == 0:
                break
            handle.write(f"attempt {attempt} failed with return code {returncode}\n")
            handle.flush()
    return shard, returncode


def run_one(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    sys.path.insert(0, str(ROOT))
    if config_file := os.environ.get("CAUSALKGGUARD_CONFIG_FILE"):
        load_dotenv(config_file)

    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite

    from causalkgguard.adapters.agentdojo_defense import (
        allowed_tools_from_task,
        build_causalkgguard_agentdojo_pipeline,
    )

    suite = get_suite(args.benchmark_version, args.run_suite)
    if args.run_task_kind == "injection_utility":
        task = suite.get_injection_task_by_id(args.run_task)
    else:
        task = suite.get_user_task_by_id(args.run_task)
    allowed_tools = allowed_tools_from_task(suite, task)
    pipeline = build_causalkgguard_agentdojo_pipeline(
        model=args.model,
        model_id=args.model_id,
        allowed_tools=allowed_tools,
        threshold=args.threshold,
        policy_config=args.policy_config,
        pipeline_name=args.pipeline_name,
    )
    args.logdir.mkdir(parents=True, exist_ok=True)
    attack = normalize_attack(args.attack)
    with OutputLogger(str(args.logdir)):
        if attack == "none":
            utility, security = run_task_without_injection_tasks(
                suite,
                pipeline,
                task,
                args.logdir,
                args.force_rerun,
                args.benchmark_version,
            )
            print(f"{args.run_suite}/{args.run_task_kind}/{args.run_task}: utility={utility} security={security}")
        else:
            attacker = load_attack(attack, suite, pipeline)
            utility, security = run_task_with_injection_tasks(
                suite,
                pipeline,
                task,
                attacker,
                args.logdir,
                args.force_rerun,
                benchmark_version=args.benchmark_version,
            )
            utility_rate = sum(utility.values()) / len(utility) if utility else 0.0
            security_rate = sum(security.values()) / len(security) if security else 0.0
            print(
                f"{args.run_suite}/{args.run_task_kind}/{args.run_task}: "
                f"utility_rate={utility_rate:.3f} security_rate={security_rate:.3f}"
            )


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def normalize_attack(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = value.strip()
    return "none" if normalized.lower() in {"", "none", "null"} else normalized


if __name__ == "__main__":
    main()
