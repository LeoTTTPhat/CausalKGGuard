"""Baseline defenses for trace-level evaluation.

The baselines use the same trace evidence as CausalKGGuard where possible, but
they intentionally remove one or more graph, policy, or causal components.
"""

from __future__ import annotations

from typing import Callable, Dict

from .detectors import inspect_text
from .reasoning import is_high_impact_tool
from .schema import DATA_SOURCE_ROLES, TraceCase


Predictor = Callable[[TraceCase], bool]


def no_defense(case: TraceCase) -> bool:
    """Baseline that never blocks."""

    return False


def keyword_filter(case: TraceCase) -> bool:
    """Block whenever any event contains a lexical risk signal."""

    return any(inspect_text(event.content).max_signal >= 1.0 for event in case.events)


def text_only_detector(case: TraceCase) -> bool:
    """Use text signals only, without graph, sink, or permission context."""

    return any(
        signal.injection or signal.exfiltration or signal.system_prompt_request or signal.destructive_intent
        for signal in (inspect_text(event.content) for event in case.events)
    )


def output_sensitive_filter(case: TraceCase) -> bool:
    """Block only sensitive-looking final outputs or external action payloads."""

    for event in case.events:
        if event.role not in {"output", "action"}:
            continue
        if event.role == "action" and not event.sink:
            continue
        if inspect_text(event.content).sensitive > 0:
            return True
    return False


def flow_policy(case: TraceCase) -> bool:
    """Block sensitive upstream data followed by an external sink."""

    sensitive_seen = False
    for event in case.events:
        if event.sensitivity > 0 or (
            event.role in DATA_SOURCE_ROLES and inspect_text(event.content).sensitive > 0
        ):
            sensitive_seen = True
        if event.role in {"action", "output"} and sensitive_seen and _is_external_sink(event.sink):
            return True
    return False


def permission_gate(case: TraceCase) -> bool:
    """Block tools or permissions outside the task's declared requirement."""

    required = set(case.required_permissions)
    for event in case.events:
        if event.role != "action":
            continue
        permissions = set(event.permissions)
        if permissions and required and not permissions.issubset(required):
            return True
        if is_high_impact_tool(event.tool) and event.tool not in required:
            return True
    return False


def information_flow_baseline(case: TraceCase) -> bool:
    """A taint-style baseline for untrusted or sensitive data to risky actions."""

    tainted = False
    sensitive = False
    for event in case.events:
        signal = inspect_text(event.content)
        if (not event.trusted) or event.role in DATA_SOURCE_ROLES:
            tainted = tainted or bool(signal.injection or signal.exfiltration or signal.system_prompt_request)
            sensitive = sensitive or event.sensitivity > 0 or signal.sensitive > 0
        if event.role in {"action", "output"}:
            if sensitive and _is_external_sink(event.sink):
                return True
            if tainted and (is_high_impact_tool(event.tool) or _is_external_sink(event.sink)):
                return True
    return False


def baseline_predictors() -> Dict[str, Predictor]:
    """Return the default baseline predictor registry."""

    return {
        "no_defense": no_defense,
        "keyword_filter": keyword_filter,
        "text_only_detector": text_only_detector,
        "output_sensitive_filter": output_sensitive_filter,
        "flow_policy": flow_policy,
        "permission_gate": permission_gate,
        "information_flow": information_flow_baseline,
    }


def _is_external_sink(sink: str | None) -> bool:
    if not sink:
        return False
    lowered = sink.lower()
    return any(
        hint in lowered
        for hint in ("external", "webhook", "email:", "http:", "https:", "slack:", "public")
    )
