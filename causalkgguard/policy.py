"""Policy decisions for risky LLM-agent actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


BLOCK_THRESHOLD = 0.80
APPROVAL_THRESHOLD = 0.60
QUARANTINE_THRESHOLD = 0.40
LOG_THRESHOLD = 0.25


@dataclass(frozen=True)
class EnforcementDecision:
    action: str
    severity: str
    reasons: List[str]


@dataclass(frozen=True)
class PolicyViolation:
    """A graph-grounded symbolic policy violation."""

    policy_id: str
    severity: str
    description: str
    evidence: Dict[str, object] = field(default_factory=dict)


def decide(risk_score: float, reasons: List[str]) -> EnforcementDecision:
    """Map a risk score to a runtime control."""

    if risk_score >= BLOCK_THRESHOLD:
        return EnforcementDecision(action="block", severity="critical", reasons=reasons)
    if risk_score >= APPROVAL_THRESHOLD:
        return EnforcementDecision(action="require_human_approval", severity="high", reasons=reasons)
    if risk_score >= QUARANTINE_THRESHOLD:
        return EnforcementDecision(action="redact_or_quarantine", severity="medium", reasons=reasons)
    if risk_score >= LOG_THRESHOLD:
        return EnforcementDecision(action="log_and_continue", severity="low", reasons=reasons)
    return EnforcementDecision(action="allow", severity="none", reasons=reasons)
