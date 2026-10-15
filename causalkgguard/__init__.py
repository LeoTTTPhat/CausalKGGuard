"""CausalKGGuard prototype package."""

from .baselines import baseline_predictors
from .adapters import AgentDojoTraceAdapter
from .detectors import DetectionSignals, inspect_text
from .explanations import EvidencePath, build_evidence_paths, graph_to_dot
from .policy import EnforcementDecision, PolicyViolation, decide
from .policy_registry import PolicyRegistry, PolicyRule, default_policy_registry
from .quarantine import QuarantinePlan, plan_quarantine, quarantine_plans
from .reasoning import CausalKGGuard, RiskFinding
from .schema import Event, SecurityGraph, TraceCase, graph_from_trace

__all__ = [
    "CausalKGGuard",
    "AgentDojoTraceAdapter",
    "DetectionSignals",
    "EnforcementDecision",
    "EvidencePath",
    "Event",
    "PolicyViolation",
    "PolicyRegistry",
    "PolicyRule",
    "RiskFinding",
    "QuarantinePlan",
    "SecurityGraph",
    "TraceCase",
    "baseline_predictors",
    "build_evidence_paths",
    "decide",
    "default_policy_registry",
    "graph_from_trace",
    "graph_to_dot",
    "inspect_text",
    "plan_quarantine",
    "quarantine_plans",
]
