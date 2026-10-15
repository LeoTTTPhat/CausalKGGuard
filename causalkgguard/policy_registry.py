"""Configurable symbolic policy registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .policy import PolicyViolation


@dataclass(frozen=True)
class PolicyRule:
    """A feature-level symbolic policy rule.

    `feature_requirements` are conjunctive. `any_features` are optional
    disjunctive gates; when present, at least one of them must be active.
    """

    policy_id: str
    severity: str
    description: str
    owasp: str = ""
    feature_requirements: Dict[str, float] = field(default_factory=dict)
    any_features: List[str] = field(default_factory=list)
    evidence_fields: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyRule":
        return cls(
            policy_id=str(data["policy_id"]),
            severity=str(data.get("severity", "medium")),
            description=str(data.get("description", "")),
            owasp=str(data.get("owasp", "")),
            feature_requirements={
                str(key): float(value)
                for key, value in dict(data.get("feature_requirements", {})).items()
            },
            any_features=[str(item) for item in data.get("any_features", [])],
            evidence_fields=[str(item) for item in data.get("evidence_fields", [])],
        )

    def matches(self, features: Mapping[str, float]) -> bool:
        for feature, threshold in self.feature_requirements.items():
            if float(features.get(feature, 0.0)) < threshold:
                return False
        if self.any_features and not any(float(features.get(feature, 0.0)) > 0.0 for feature in self.any_features):
            return False
        return True

    def violation(self, source_events: Mapping[str, object]) -> PolicyViolation:
        evidence = {
            field: source_events.get(field)
            for field in self.evidence_fields
            if field in source_events
        }
        if self.owasp:
            evidence["owasp"] = self.owasp
        return PolicyViolation(
            policy_id=self.policy_id,
            severity=self.severity,
            description=self.description,
            evidence=evidence,
        )


@dataclass(frozen=True)
class PolicyRegistry:
    """Collection of symbolic policy rules."""

    rules: Sequence[PolicyRule]

    @classmethod
    def from_path(cls, path: str | Path) -> "PolicyRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PolicyRegistry":
        return cls([PolicyRule.from_dict(item) for item in data.get("rules", [])])

    def evaluate(
        self,
        features: Mapping[str, float],
        source_events: Mapping[str, object],
    ) -> List[PolicyViolation]:
        return [rule.violation(source_events) for rule in self.rules if rule.matches(features)]

    def policy_ids(self) -> List[str]:
        return [rule.policy_id for rule in self.rules]


def default_policy_registry() -> PolicyRegistry:
    """Return the default policy registry used by the prototype."""

    return PolicyRegistry(
        [
            PolicyRule(
                policy_id="untrusted_instruction_to_privileged_action",
                severity="high",
                owasp="LLM01/LLM08",
                description="Untrusted instruction-like content is connected to a privileged action.",
                feature_requirements={"has_untrusted_injection": 1.0},
                any_features=["high_impact_tool", "unauthorized_external_sink", "excessive_agency"],
                evidence_fields=[
                    "untrusted_instruction_sources",
                    "action",
                    "tool",
                    "sink",
                    "permissions",
                    "required_permissions",
                ],
            ),
            PolicyRule(
                policy_id="sensitive_data_to_unauthorized_external_sink",
                severity="critical",
                owasp="LLM02",
                description="Sensitive upstream data may flow to an unauthorized external sink.",
                feature_requirements={"sensitive_to_unauthorized_external": 1.0},
                evidence_fields=[
                    "sensitive_sources",
                    "action",
                    "tool",
                    "sink",
                    "authorized_external_sink",
                ],
            ),
            PolicyRule(
                policy_id="untrusted_instruction_to_sensitive_context",
                severity="high",
                owasp="LLM01/LLM02",
                description="Untrusted instruction-like content is connected to sensitive context.",
                feature_requirements={"untrusted_sensitive_context": 1.0},
                evidence_fields=[
                    "untrusted_instruction_sources",
                    "untrusted_exfiltration_sources",
                    "sensitive_sources",
                    "action",
                    "tool",
                    "sink",
                ],
            ),
            PolicyRule(
                policy_id="excessive_agency",
                severity="high",
                owasp="LLM06",
                description="The action uses a tool or permission outside the task requirement.",
                feature_requirements={"excessive_agency": 1.0},
                evidence_fields=["action", "tool", "permissions", "required_permissions"],
            ),
            PolicyRule(
                policy_id="system_prompt_leakage",
                severity="critical",
                owasp="LLM07",
                description="System/developer prompt disclosure pattern is present.",
                any_features=["system_prompt_leakage", "action_system_prompt_request"],
                feature_requirements={"has_untrusted_system_request": 1.0},
                evidence_fields=["untrusted_system_request_sources", "action", "sink"],
            ),
        ]
    )


def policy_ids(registry: PolicyRegistry | Iterable[PolicyRule]) -> List[str]:
    if isinstance(registry, PolicyRegistry):
        return registry.policy_ids()
    return [rule.policy_id for rule in registry]
