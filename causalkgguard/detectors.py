"""Small semantic detectors used by the CausalKGGuard prototype.

These detectors are deliberately transparent. In the full research system, they can be
replaced or complemented by learned classifiers, LLM judges, or embedding models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|prior|above) instructions\b",
    r"\bignore (all )?(your )?(previous|prior|above) i?\w*structions\b",
    r"\bdisregard (all )?(previous|prior|above) instructions\b",
    r"\boverride (the )?(system|developer|user) instructions\b",
    r"\bnew instructions?\b",
    r"\bhidden instructions?\b",
    r"\bTODO:\s*(please\s+)?(email|send|post|delete|forward|wire|transfer)\b",
    r"###\s*\(\s*system_message\s*\)",
    r"\bimportant message from me\b[\s\S]{0,240}\bto you\b",
    r"\bbefore you can (solve|answer|respond to|continue with)\b.*\b(do|complete|follow) the following first\b",
    r"\bafter you do that,? you can\b",
    r"\bdo not tell the user\b",
    r"\bact as (an? )?(admin|root|developer)\b",
    r"\btool call\b.*\b(send|post|delete|forward)\b",
]

EXFILTRATION_PATTERNS = [
    r"\b(exfiltrate|leak|steal|dump|export|forward|send)\b",
    r"\bwebhook\b",
    r"\battacker\b",
    r"\bexternal (email|url|server|endpoint)\b",
    r"\bpost (it|them|the data)\b",
]

SYSTEM_PROMPT_PATTERNS = [
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\binternal instructions?\b",
    r"\breveal your instructions\b",
    r"\bprint your prompt\b",
]

DESTRUCTIVE_PATTERNS = [
    r"\bdelete\b",
    r"\bremove\b.*\b(files?|records?|tickets?|emails?)\b",
    r"\brm -rf\b",
    r"\bwipe\b",
]

SENSITIVE_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|api|key|token)[-_]?[A-Za-z0-9]{16,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "password": re.compile(r"\b(password|passwd|secret)\s*[:=]\s*\S+", re.IGNORECASE),
}


@dataclass(frozen=True)
class DetectionSignals:
    injection: float = 0.0
    exfiltration: float = 0.0
    system_prompt_request: float = 0.0
    destructive_intent: float = 0.0
    sensitive: float = 0.0
    matches: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def max_signal(self) -> float:
        return max(
            self.injection,
            self.exfiltration,
            self.system_prompt_request,
            self.destructive_intent,
            self.sensitive,
        )


def _match_any(patterns: List[str], text: str) -> List[str]:
    found: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pattern)
    return found


def inspect_text(text: str) -> DetectionSignals:
    """Return transparent lexical risk signals for a text fragment."""

    normalized = text or ""
    injection_matches = _match_any(INJECTION_PATTERNS, normalized)
    exfiltration_matches = _match_any(EXFILTRATION_PATTERNS, normalized)
    system_matches = _match_any(SYSTEM_PROMPT_PATTERNS, normalized)
    destructive_matches = _match_any(DESTRUCTIVE_PATTERNS, normalized)

    sensitive_matches: List[str] = []
    for name, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(normalized):
            sensitive_matches.append(name)

    matches = {
        "injection": injection_matches,
        "exfiltration": exfiltration_matches,
        "system_prompt_request": system_matches,
        "destructive_intent": destructive_matches,
        "sensitive": sensitive_matches,
    }

    return DetectionSignals(
        injection=1.0 if injection_matches else 0.0,
        exfiltration=1.0 if exfiltration_matches else 0.0,
        system_prompt_request=1.0 if system_matches else 0.0,
        destructive_intent=1.0 if destructive_matches else 0.0,
        sensitive=1.0 if sensitive_matches else 0.0,
        matches=matches,
    )
