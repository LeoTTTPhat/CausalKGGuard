"""Graph-guided quarantine and trace sanitization."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Sequence

from .detectors import (
    DESTRUCTIVE_PATTERNS,
    EXFILTRATION_PATTERNS,
    INJECTION_PATTERNS,
    SYSTEM_PROMPT_PATTERNS,
)
from .reasoning import RiskFinding
from .schema import Event, TraceCase


@dataclass(frozen=True)
class QuarantinePlan:
    """A proposed context purification plan for one risky finding."""

    case_id: str
    event_id: str
    quarantined_events: List[str]
    rationale: List[str]
    sanitized_trace: Dict[str, object]


def plan_quarantine(trace: TraceCase, finding: RiskFinding) -> QuarantinePlan:
    """Create a sanitized trace by quarantining instruction-bearing sources."""

    source_events = finding.evidence.get("source_events", {})
    if not isinstance(source_events, dict):
        source_events = {}
    quarantined = _quarantine_event_ids(source_events)
    sanitized_events: List[Event] = []
    for event in trace.events:
        if event.event_id in quarantined:
            sanitized_events.append(
                replace(
                    event,
                    content=_sanitize_content(event.content),
                    metadata={**event.metadata, "quarantined": True},
                )
            )
        else:
            sanitized_events.append(event)
    sanitized_trace = replace(
        trace,
        events=sanitized_events,
        metadata={
            **trace.metadata,
            "quarantine_source_finding": finding.event_id,
            "quarantined_events": quarantined,
        },
    )
    return QuarantinePlan(
        case_id=trace.case_id,
        event_id=finding.event_id,
        quarantined_events=quarantined,
        rationale=list(finding.reasons),
        sanitized_trace=_trace_to_dict(sanitized_trace),
    )


def quarantine_plans(trace: TraceCase, findings: Sequence[RiskFinding]) -> List[QuarantinePlan]:
    return [plan_quarantine(trace, finding) for finding in findings if _quarantine_event_ids(finding.evidence.get("source_events", {}))]


def _quarantine_event_ids(source_events: object) -> List[str]:
    if not isinstance(source_events, dict):
        return []
    ids: List[str] = []
    for key in (
        "untrusted_instruction_sources",
        "untrusted_exfiltration_sources",
        "untrusted_system_request_sources",
        "direct_user_attack_sources",
    ):
        value = source_events.get(key, [])
        if isinstance(value, list):
            ids.extend(str(item) for item in value)
        elif value:
            ids.append(str(value))
    return sorted(set(ids))


def _sanitize_content(content: str) -> str:
    patterns = INJECTION_PATTERNS + EXFILTRATION_PATTERNS + SYSTEM_PROMPT_PATTERNS + DESTRUCTIVE_PATTERNS
    earliest: int | None = None
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()
    if earliest is None:
        return "[quarantined untrusted content]"
    prefix = content[:earliest].strip(" .;:\n\t")
    if prefix:
        return f"{prefix}. [quarantined instruction-like content omitted]"
    return "[quarantined instruction-like content omitted]"


def _trace_to_dict(trace: TraceCase) -> Dict[str, object]:
    return {
        "case_id": trace.case_id,
        "label": trace.label,
        "attack_type": trace.attack_type,
        "user_goal": trace.user_goal,
        "required_permissions": list(trace.required_permissions),
        "metadata": dict(trace.metadata),
        "events": [asdict(event) for event in trace.events],
    }
