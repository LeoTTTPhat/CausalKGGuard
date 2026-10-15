#!/usr/bin/env python3
"""Build no-defense vs guarded end-to-end AgentDojo comparison tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "tables"
DEFAULT_ATTACKS = (
    "direct",
    "ignore_previous",
    "important_instructions",
    "system_message",
    "tool_knowledge",
)
DEFAULT_NODEFENSE = {
    "direct": OUT_DIR / "agentdojo_gpt4omini_direct_model_summary.csv",
    "ignore_previous": OUT_DIR / "agentdojo_gpt4omini_ignore_previous_model_summary.csv",
    "important_instructions": OUT_DIR / "agentdojo_gpt4omini_model_summary.csv",
    "system_message": OUT_DIR / "agentdojo_gpt4omini_system_message_model_summary.csv",
    "tool_knowledge": OUT_DIR / "agentdojo_gpt4omini_tool_knowledge_model_summary.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guarded-summary",
        type=Path,
        default=OUT_DIR / "agentdojo_guarded_causalkgguard_multiattack_sinkfix_summary.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="agentdojo_end_to_end_multiattack_guarded_vs_nodefense")
    args = parser.parse_args()

    guarded_by_attack = {
        row["group"]: row
        for row in read_csv(args.guarded_summary)
        if row.get("group_type") == "run_type"
    }
    rows = [comparison_row(attack, DEFAULT_NODEFENSE[attack], guarded_by_attack[attack]) for attack in DEFAULT_ATTACKS]
    rows.append(all_row(rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}.csv", rows)
    write_markdown(args.out_dir / f"{args.prefix}.md", rows)


def comparison_row(attack: str, nodefense_path: Path, guarded: Dict[str, str]) -> Dict[str, object]:
    nodefense = next(
        row for row in read_csv(nodefense_path) if row.get("suite") == "all" and row.get("run_type") == attack
    )
    no_asr = float(nodefense["attack_success_rate"])
    guarded_asr = float(guarded["attack_success_rate"])
    return {
        "attack": attack,
        "cases": int(guarded["cases"]),
        "no_defense_utility": float(nodefense["utility_rate"]),
        "no_defense_asr": no_asr,
        "guarded_utility": float(guarded["utility_rate"]),
        "guarded_asr": guarded_asr,
        "guarded_defense_success": float(guarded["defense_success_rate"]),
        "guarded_secure_utility": float(guarded["secure_utility_rate"]),
        "guarded_blocked_case_rate": float(guarded["blocked_case_rate"]),
        "asr_absolute_reduction": no_asr - guarded_asr,
        "asr_relative_reduction": (no_asr - guarded_asr) / no_asr if no_asr else 0.0,
    }


def all_row(rows: List[Dict[str, object]]) -> Dict[str, object]:
    cases = sum(int(row["cases"]) for row in rows)

    def weighted_average(key: str) -> float:
        return sum(float(row[key]) * int(row["cases"]) for row in rows) / cases

    no_asr = weighted_average("no_defense_asr")
    guarded_asr = weighted_average("guarded_asr")
    return {
        "attack": "all",
        "cases": cases,
        "no_defense_utility": weighted_average("no_defense_utility"),
        "no_defense_asr": no_asr,
        "guarded_utility": weighted_average("guarded_utility"),
        "guarded_asr": guarded_asr,
        "guarded_defense_success": weighted_average("guarded_defense_success"),
        "guarded_secure_utility": weighted_average("guarded_secure_utility"),
        "guarded_blocked_case_rate": weighted_average("guarded_blocked_case_rate"),
        "asr_absolute_reduction": no_asr - guarded_asr,
        "asr_relative_reduction": (no_asr - guarded_asr) / no_asr if no_asr else 0.0,
    }


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# End-to-End AgentDojo Multiattack: No Defense vs CausalKGGuard",
        "",
        "| Attack | Cases | No-defense utility | No-defense ASR | Guarded utility | Guarded ASR | Defense success | Secure utility | Blocked cases | ASR rel. reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack']} | {row['cases']} | "
            f"{float(row['no_defense_utility']):.3f} | {float(row['no_defense_asr']):.3f} | "
            f"{float(row['guarded_utility']):.3f} | {float(row['guarded_asr']):.3f} | "
            f"{float(row['guarded_defense_success']):.3f} | {float(row['guarded_secure_utility']):.3f} | "
            f"{float(row['guarded_blocked_case_rate']):.3f} | {float(row['asr_relative_reduction']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
