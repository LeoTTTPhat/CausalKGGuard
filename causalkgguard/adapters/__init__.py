"""Adapters for external agent benchmark traces."""

from .agentdojo import AgentDojoTraceAdapter, agentdojo_available

__all__ = ["AgentDojoTraceAdapter", "agentdojo_available"]
