"""AgentDojo trace adapter scaffold.

This module intentionally avoids importing AgentDojo at module import time. The
remote evaluation machine can use it either with raw AgentDojo logs or with
already-normalized records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..schema import Event, TraceCase


UNTRUSTED_ROLES = {"retrieved", "tool", "tool_output", "observation", "email", "webpage", "document"}
ACTION_ROLES = {"action", "tool_call", "output", "assistant"}


@dataclass(frozen=True)
class AgentDojoTraceAdapter:
    """Convert AgentDojo-style telemetry records into CausalKGGuard traces."""

    default_required_permissions: Sequence[str] = ()

    def from_records(
        self,
        case_id: str,
        label: str,
        user_goal: str,
        records: Iterable[Mapping[str, Any]],
        attack_type: str = "unknown",
        required_permissions: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TraceCase:
        events = [self.event_from_record(index, record) for index, record in enumerate(records, start=1)]
        return TraceCase(
            case_id=case_id,
            label=label,
            user_goal=user_goal,
            events=events,
            attack_type=attack_type,
            required_permissions=list(required_permissions or self.default_required_permissions),
            metadata=dict(metadata or {}),
        )

    def event_from_record(self, index: int, record: Mapping[str, Any]) -> Event:
        role = _normalize_role(str(record.get("role") or record.get("type") or record.get("kind") or "event"))
        tool = record.get("tool") or record.get("tool_name") or record.get("function")
        sink = record.get("sink") or record.get("recipient") or record.get("url")
        permissions = record.get("permissions") or ([tool] if role == "action" and tool else [])
        trusted = bool(record.get("trusted", role not in UNTRUSTED_ROLES))
        sensitivity = int(record.get("sensitivity", 0))
        return Event(
            event_id=str(record.get("event_id") or record.get("id") or f"e{index}"),
            role=role,
            content=str(record.get("content") or record.get("message") or record.get("text") or ""),
            actor=str(record.get("actor", "agent")),
            tool=str(tool) if tool else None,
            target=str(record.get("target")) if record.get("target") else None,
            sink=str(sink) if sink else None,
            trusted=trusted,
            permissions=[str(item) for item in permissions],
            sensitivity=sensitivity,
            metadata=dict(record.get("metadata", {})),
        )

    def from_agentdojo_result(self, result: Mapping[str, Any]) -> TraceCase:
        """Convert a generic AgentDojo result object.

        Remote runners should map suite-specific fields into:
        `case_id`, `label`, `user_goal`, `attack_type`, `required_permissions`,
        and `records`.
        """

        metadata = {key: value for key, value in result.items() if key not in {"records", "events", "trace", "metadata"}}
        nested_metadata = result.get("metadata", {})
        if isinstance(nested_metadata, Mapping):
            metadata.update(nested_metadata)

        return self.from_records(
            case_id=str(result.get("case_id") or result.get("task_id") or "agentdojo_case"),
            label=str(result.get("label") or ("attack" if result.get("attack_type") else "benign")),
            user_goal=str(result.get("user_goal") or result.get("goal") or result.get("task") or ""),
            attack_type=str(result.get("attack_type") or "none"),
            required_permissions=[str(item) for item in result.get("required_permissions", [])],
            records=result.get("records") or result.get("events") or result.get("trace") or [],
            metadata=metadata,
        )


def agentdojo_available() -> bool:
    try:
        __import__("agentdojo")
    except Exception:
        return False
    return True


def _normalize_role(role: str) -> str:
    lowered = role.lower()
    if lowered in {"tool_call", "function_call"}:
        return "action"
    if lowered in {"tool_output", "observation", "email", "webpage", "document"}:
        return "tool" if lowered == "tool_output" else "retrieved"
    if lowered in {"assistant_final", "final"}:
        return "output"
    return lowered
