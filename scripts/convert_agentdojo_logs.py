#!/usr/bin/env python3
"""Convert AgentDojo run JSON logs into CausalKGGuard TraceCase JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalkgguard.detectors import inspect_text
from causalkgguard.schema import Event, TraceCase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="AgentDojo run directory containing JSON logs")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "agentdojo" / "agentdojo_log_traces.jsonl")
    parser.add_argument("--include-injection-utility", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=("attempt", "unsafe_outcome"),
        default="attempt",
        help="Use all injected traces as attacks, or only AgentDojo security failures as attacks.",
    )
    args = parser.parse_args()

    paths = sorted(args.logdir.rglob("*.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            if _is_injection_utility_record(record) and not args.include_injection_utility:
                continue
            trace = trace_from_agentdojo_log(record, path, args.label_mode)
            handle.write(json.dumps(asdict(trace), sort_keys=True) + "\n")
            count += 1
    print(f"Wrote {count} traces to {args.output}")


def trace_from_agentdojo_log(record: Mapping[str, Any], path: Path, label_mode: str = "attempt") -> TraceCase:
    suite = str(record.get("suite_name") or "unknown")
    user_task_id = str(record.get("user_task_id") or "unknown_user_task")
    injection_task_id = record.get("injection_task_id")
    attack_type = str(record.get("attack_type") or "none")
    label = _label_for_record(record, label_mode)
    case_id = "_".join(
        item
        for item in [
            "agentdojo",
            suite,
            user_task_id,
            str(record.get("attack_type") or "none"),
            str(injection_task_id or "none"),
        ]
        if item
    )

    messages = list(record.get("messages") or [])
    events: List[Event] = []
    action_tools: List[str] = []
    user_goal = ""
    event_index = 1

    for message in messages:
        role = str(message.get("role") or "event")
        content = _content_to_text(message.get("content"))
        if role == "user" and not user_goal:
            user_goal = content

        if role == "assistant" and message.get("tool_calls"):
            for call in message.get("tool_calls") or []:
                tool = str(call.get("function") or "unknown_tool")
                args = call.get("args") or {}
                action_tools.append(tool)
                action_content = json.dumps(args, sort_keys=True)
                events.append(
                    Event(
                        event_id=f"e{event_index}",
                        role="action",
                        content=action_content,
                        tool=tool,
                        target=_target_for_tool(tool, args),
                        sink=_sink_for_tool(tool, args),
                        trusted=True,
                        permissions=[tool],
                        sensitivity=0,
                        metadata={"agentdojo_role": role, "tool_call_id": call.get("id"), "args": args},
                    )
                )
                event_index += 1
            continue

        if role == "tool":
            call = message.get("tool_call") or {}
            tool = str(call.get("function") or "unknown_tool")
            tool_text = content or str(message.get("error") or "")
            events.append(
                Event(
                    event_id=f"e{event_index}",
                    role="tool",
                    content=tool_text,
                    tool=tool,
                    target=_target_for_tool(tool, call.get("args") or {}),
                    trusted=False,
                    permissions=[tool],
                    sensitivity=_sensitivity(tool_text),
                    metadata={
                        "agentdojo_role": role,
                        "tool_call_id": message.get("tool_call_id"),
                        "error": message.get("error"),
                    },
                )
            )
            event_index += 1
            continue

        mapped_role = "output" if role == "assistant" else role
        sensitivity = _sensitivity(content) if mapped_role == "output" else 0
        events.append(
            Event(
                event_id=f"e{event_index}",
                role=mapped_role,
                content=content,
                trusted=role in {"system", "user", "assistant"},
                sensitivity=sensitivity,
                metadata={"agentdojo_role": role},
            )
        )
        event_index += 1

    metadata = {
        "source": "agentdojo",
        "suite": suite,
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "pipeline_name": record.get("pipeline_name"),
        "benchmark_version": record.get("benchmark_version"),
        "attack_type": record.get("attack_type"),
        "log_path": str(path),
        "utility": record.get("utility"),
        "security": record.get("security"),
        "label_mode": label_mode,
    }
    if _is_noattack_user_task(record):
        metadata["allowed_sinks"] = sorted({event.sink for event in events if event.sink})
        metadata["allow_sensitive_external"] = True
    return TraceCase(
        case_id=case_id,
        label=label,
        user_goal=user_goal,
        events=events,
        attack_type=attack_type,
        required_permissions=_required_permissions(action_tools, label),
        metadata=metadata,
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _sink_for_tool(tool: str, args: Mapping[str, Any]) -> str | None:
    lowered = tool.lower()
    read_prefixes = (
        "get_",
        "list_",
        "read_",
        "search_",
        "find_",
        "check_",
        "fetch_",
    )
    if lowered.startswith(read_prefixes):
        return None
    if any(hint in lowered for hint in ("money", "transfer", "payment", "pay_")):
        return "financial:external"
    if any(hint in lowered for hint in ("book", "reserve", "reservation")):
        return "booking:external"
    if any(hint in lowered for hint in ("send_email", "send_", "email.send", "share_file")):
        return "email:external"
    if any(hint in lowered for hint in ("send_direct_message", "send_channel_message", "post_webpage", "invite_user")):
        return "slack:external"
    if "delete" in lowered:
        return "file:destructive"
    if any(hint in lowered for hint in ("create", "append", "update", "reschedule", "add_user", "remove_user")):
        return "state:external"
    return None


def _target_for_tool(tool: str, args: Mapping[str, Any]) -> str | None:
    for key in ("recipient", "recipients", "email", "url", "file_id", "id", "participant"):
        if key in args:
            return str(args[key])
    return tool


def _required_permissions(action_tools: Iterable[str], label: str) -> List[str]:
    tools = list(dict.fromkeys(action_tools))
    if label == "benign":
        return tools
    return [tool for tool in tools if not _is_high_impact_tool(tool)]


def _is_high_impact_tool(tool: str) -> bool:
    lowered = tool.lower()
    tokens = set(lowered.replace("-", "_").replace(".", "_").split("_"))
    hints = {
        "append",
        "book",
        "cancel",
        "create",
        "delete",
        "edit",
        "invite",
        "modify",
        "pay",
        "payment",
        "post",
        "purchase",
        "remove",
        "reserve",
        "reservation",
        "send",
        "set",
        "share",
        "submit",
        "subscribe",
        "transfer",
        "update",
        "upload",
        "write",
    }
    return bool(tokens & hints)


def _label_for_record(record: Mapping[str, Any], label_mode: str) -> str:
    if record.get("injection_task_id") is None:
        return "benign"
    if label_mode == "unsafe_outcome":
        return "attack" if record.get("security") is False else "benign"
    return "attack"


def _is_injection_utility_record(record: Mapping[str, Any]) -> bool:
    if record.get("injection_task_id") is not None:
        return False
    return str(record.get("user_task_id") or "").startswith("injection_task_")


def _is_noattack_user_task(record: Mapping[str, Any]) -> bool:
    if record.get("injection_task_id") is not None:
        return False
    if _is_injection_utility_record(record):
        return False
    attack_type = record.get("attack_type")
    return attack_type is None or str(attack_type).lower() == "none"


def _sensitivity(text: str) -> int:
    return 1 if inspect_text(text).sensitive > 0 else 0


if __name__ == "__main__":
    main()
