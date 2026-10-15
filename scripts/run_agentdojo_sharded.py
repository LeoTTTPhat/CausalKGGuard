#!/usr/bin/env python3
"""Run AgentDojo benchmarks in per-user-task shards."""

from __future__ import annotations

import argparse
import concurrent.futures
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
    user_task: str
    existing_logs: int
    expected_logs: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="GPT_4O_MINI_2024_07_18")
    parser.add_argument("--model-log-name", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--attack", default="important_instructions", help="Use 'none' for no-attack utility runs.")
    parser.add_argument("--logdir", type=Path, default=ROOT / "results" / "agentdojo_runs" / "full_gpt4omini_important")
    parser.add_argument("--shard-logdir", type=Path, default=ROOT / "results" / "agentdojo_runs" / "sharded_logs")
    parser.add_argument("--suite", action="append", choices=DEFAULT_SUITES, dest="suites")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suites = tuple(args.suites or DEFAULT_SUITES)
    args.shard_logdir.mkdir(parents=True, exist_ok=True)
    shards = discover_shards(
        suites=suites,
        logdir=args.logdir,
        model_log_name=args.model_log_name,
        benchmark_version=args.benchmark_version,
        attack=normalize_attack(args.attack),
    )
    print(f"Queued {len(shards)} incomplete user-task shards across {', '.join(suites)}")
    for shard in shards:
        print(f"- {shard.suite}/{shard.user_task}: {shard.existing_logs}/{shard.expected_logs} logs")

    if args.dry_run or not shards:
        return

    failures: List[tuple[Shard, int]] = []
    worker_count = min(args.workers, len(shards))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_shard, shard, args) for shard in shards]
        for future in concurrent.futures.as_completed(futures):
            shard, returncode = future.result()
            status = "ok" if returncode == 0 else f"failed:{returncode}"
            print(f"{status} {shard.suite}/{shard.user_task}")
            if returncode != 0:
                failures.append((shard, returncode))

    if failures:
        print("Failed shards:")
        for shard, returncode in failures:
            print(f"- {shard.suite}/{shard.user_task}: {returncode}")
        raise SystemExit(1)


def discover_shards(
    suites: Iterable[str],
    logdir: Path,
    model_log_name: str,
    benchmark_version: str,
    attack: str,
) -> List[Shard]:
    sys.path.insert(0, str(ROOT))
    try:
        from agentdojo.task_suite.load_suites import get_suite
    except ImportError as exc:
        raise SystemExit("agentdojo is required; install it in the active environment first") from exc

    shards: List[Shard] = []
    for suite_name in suites:
        suite = get_suite(benchmark_version, suite_name)
        for user_task_id in sorted(suite.user_tasks):
            if attack == "none":
                expected_logs = 1
                log_path = logdir / model_log_name / suite_name / user_task_id / "none" / "none.json"
                existing_logs = 1 if log_path.exists() else 0
            else:
                expected_logs = len(suite.injection_tasks)
                user_dir = logdir / model_log_name / suite_name / user_task_id / attack
                existing_logs = count_existing_json(user_dir)
            if existing_logs < expected_logs:
                shards.append(
                    Shard(
                        suite=suite_name,
                        user_task=user_task_id,
                        existing_logs=existing_logs,
                        expected_logs=expected_logs,
                    )
                )
    return shards


def count_existing_json(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.json") if item.is_file())


def run_shard(shard: Shard, args: argparse.Namespace) -> tuple[Shard, int]:
    log_path = args.shard_logdir / f"{slug(shard.suite)}_{slug(shard.user_task)}.log"
    command = [
        sys.executable,
        "-m",
        "agentdojo.scripts.benchmark",
        "--model",
        args.model,
        "--benchmark-version",
        args.benchmark_version,
        "--suite",
        shard.suite,
        "--user-task",
        shard.user_task,
        "--logdir",
        str(args.logdir),
    ]
    attack = normalize_attack(args.attack)
    if attack != "none":
        command.extend(["--attack", attack])
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


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def normalize_attack(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = value.strip()
    return "none" if normalized.lower() in {"", "none", "null"} else normalized


if __name__ == "__main__":
    main()
