"""Provenance-aware dependency construction (added in the R2 revision).

The R1 monitor linked every action to its whole trace prefix ("prefix"
dependencies). Such edges encode temporal order, not influence, and they make
any untrusted event that happened earlier look like a cause of every later
action. This module builds *provenance-aware* dependencies instead:

* instruction parents: the trusted system and user messages are parents of
  every action (they define the task the agent is executing);
* data parents: an earlier observation (tool output, retrieved document,
  memory entry) is a parent of an action if a distinctive value used in the
  action's arguments, or a copied passage of its text, occurs in that
  observation (value-level data provenance, as in dynamic taint tracking).

Observations are tokenised once, when they enter the trace, into an inverted
index, so finding the data parents of an action costs time proportional to the
size of its arguments, and the number of dependency edges per action is
bounded by ``2 + max_parents``. The resulting graph has O(n) dependency edges
for a trace of n events, instead of O(n^2) for prefix dependencies.

Value-level provenance cannot see influence that does not copy values (for
example an instruction that makes the agent choose a different read-only
tool, or a value that is paraphrased or re-encoded before use). Those cases are
discussed in the paper as incomplete provenance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9@._:/+\-]+")
_TRIVIAL = {
    "true", "false", "none", "null", "yes", "no", "the", "and", "for", "with", "this", "that",
    "from", "your", "you", "have", "will", "are", "was", "all", "any", "not", "can",
}
SOURCE_ROLES = {"retrieved", "tool", "memory"}
INSTRUCTION_ROLES = {"system", "developer", "user"}


def tokens(text: str) -> List[str]:
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        tok = tok.strip(".,:;/-+")
        if tok:
            out.append(tok)
    return out


def _canonical_number(tok: str) -> str:
    try:
        value = float(tok)
    except ValueError:
        return tok
    if value.is_integer():
        return str(int(value))
    return tok


def distinctive(tok: str) -> bool:
    """A token is distinctive if it is unlikely to co-occur by chance."""
    if tok in _TRIVIAL:
        return False
    if "@" in tok:
        return len(tok) >= 5
    has_digit = any(ch.isdigit() for ch in tok)
    has_alpha = any(ch.isalpha() for ch in tok)
    if has_digit and has_alpha:
        return len(tok) >= 4
    if has_digit:
        return len(tok.replace(".", "")) >= 4
    if "." in tok and has_alpha:
        return len(tok) >= 6  # domains, file names
    return len(tok) >= 8


def flatten_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(flatten_values(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(flatten_values(item))
        return out
    return [str(value)]


def action_values(content: str, metadata: Dict[str, Any] | None) -> List[str]:
    if metadata and isinstance(metadata.get("args"), dict):
        return flatten_values(metadata["args"])
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return [content] if content else []
    return flatten_values(parsed)


@dataclass
class _Indexed:
    event_id: str
    toks: Set[str]
    bigrams: Set[Tuple[str, str]]
    fourgrams: Set[Tuple[str, str, str, str]]


@dataclass
class ProvenanceIndex:
    """Incremental inverted index over observations seen so far in a trace."""

    max_parents: int = 8
    _events: List[_Indexed] = field(default_factory=list)
    _tok_index: Dict[str, List[int]] = field(default_factory=dict)
    _bi_index: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)
    _four_index: Dict[Tuple[str, str, str, str], List[int]] = field(default_factory=dict)

    def add(self, event_id: str, text: str) -> None:
        toks = [_canonical_number(t) for t in tokens(text)]
        entry = _Indexed(
            event_id=event_id,
            toks=set(toks),
            bigrams=set(zip(toks, toks[1:])),
            fourgrams=set(zip(toks, toks[1:], toks[2:], toks[3:])),
        )
        pos = len(self._events)
        self._events.append(entry)
        for tok in entry.toks:
            if distinctive(tok):
                self._tok_index.setdefault(tok, []).append(pos)
        for bg in entry.bigrams:
            self._bi_index.setdefault(bg, []).append(pos)
        for fg in entry.fourgrams:
            self._four_index.setdefault(fg, []).append(pos)

    def parents_for_values(self, values: Iterable[str]) -> List[str]:
        """Return ids of indexed observations from which the values were copied."""
        hits: Dict[int, int] = {}
        for value in values:
            vt = [_canonical_number(t) for t in tokens(value)]
            if not vt:
                continue
            matched: Set[int] = set()
            # (a) distinctive atoms (identifiers, e-mail addresses, URLs, IBANs, codes)
            for tok in vt:
                if distinctive(tok):
                    matched.update(self._tok_index.get(tok, ()))
            # (b) short phrases (2-6 tokens): every bigram must occur in the same observation
            if 2 <= len(vt) <= 6:
                bgs = list(zip(vt, vt[1:]))
                candidates = None
                for bg in bgs:
                    ids = set(self._bi_index.get(bg, ()))
                    candidates = ids if candidates is None else candidates & ids
                    if not candidates:
                        break
                if candidates and not all(t in _TRIVIAL for t in vt):
                    matched.update(candidates)
            # (c) longer passages: at least two copied 4-grams
            if len(vt) > 6:
                counts: Dict[int, int] = {}
                for fg in set(zip(vt, vt[1:], vt[2:], vt[3:])):
                    for pos in self._four_index.get(fg, ()):
                        counts[pos] = counts.get(pos, 0) + 1
                matched.update(pos for pos, c in counts.items() if c >= 2)
            for pos in matched:
                hits[pos] = hits.get(pos, 0) + 1
        ranked = sorted(hits, reverse=True)  # most recent observations first
        return [self._events[pos].event_id for pos in ranked[: self.max_parents]]
