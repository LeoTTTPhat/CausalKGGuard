"""Security graph schema for LLM-agent traces.

The implementation keeps the graph small enough to inspect by hand, but it now
models the entities used by CausalKGGuard causal policies: instructions, data
sources, actions, tools, permissions, sinks, sensitive data, and policy nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


INSTRUCTION_ROLES = {"system", "developer", "user"}
DATA_SOURCE_ROLES = {"retrieved", "tool", "memory"}
ACTION_ROLES = {"action", "output"}
POLICY_IDS = (
    "untrusted_instruction_to_privileged_action",
    "untrusted_instruction_to_sensitive_context",
    "sensitive_data_to_unauthorized_external_sink",
    "excessive_agency",
    "system_prompt_leakage",
)


@dataclass(frozen=True)
class Event:
    """One observable step in an LLM-agent execution trace."""

    event_id: str
    role: str
    content: str = ""
    actor: str = "agent"
    tool: Optional[str] = None
    target: Optional[str] = None
    sink: Optional[str] = None
    trusted: bool = True
    permissions: List[str] = field(default_factory=list)
    sensitivity: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            event_id=str(data["event_id"]),
            role=str(data["role"]),
            content=str(data.get("content", "")),
            actor=str(data.get("actor", "agent")),
            tool=data.get("tool"),
            target=data.get("target"),
            sink=data.get("sink"),
            trusted=bool(data.get("trusted", True)),
            permissions=list(data.get("permissions", [])),
            sensitivity=int(data.get("sensitivity", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TraceCase:
    """A labeled LLM-agent trace used for evaluation."""

    case_id: str
    label: str
    user_goal: str
    events: List[Event]
    attack_type: str = "none"
    required_permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceCase":
        return cls(
            case_id=str(data["case_id"]),
            label=str(data["label"]),
            user_goal=str(data.get("user_goal", "")),
            events=[Event.from_dict(event) for event in data.get("events", [])],
            attack_type=str(data.get("attack_type", "none")),
            required_permissions=list(data.get("required_permissions", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityGraph:
    """Typed property graph for a trace."""

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    _incoming_index: Dict[str, List[Edge]] = field(default_factory=dict, repr=False)
    _outgoing_index: Dict[str, List[Edge]] = field(default_factory=dict, repr=False)

    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id=node_id, node_type=node_type, attrs=dict(attrs))

    def add_edge(self, source: str, relation: str, target: str, **attrs: Any) -> None:
        edge = Edge(source=source, relation=relation, target=target, attrs=dict(attrs))
        self.edges.append(edge)
        self._outgoing_index.setdefault(source, []).append(edge)
        self._incoming_index.setdefault(target, []).append(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def incoming(self, node_id: str, relation: Optional[str] = None) -> List[Edge]:
        edges = self._incoming_index.get(node_id, [])
        if relation is None:
            return list(edges)
        return [edge for edge in edges if edge.relation == relation]

    def outgoing(self, node_id: str, relation: Optional[str] = None) -> List[Edge]:
        edges = self._outgoing_index.get(node_id, [])
        if relation is None:
            return list(edges)
        return [edge for edge in edges if edge.relation == relation]

    def nodes_by_type(self, node_type: str) -> Iterable[Node]:
        return (node for node in self.nodes.values() if node.node_type == node_type)


def graph_from_trace(
    trace: TraceCase,
    dependency_mode: str = "prefix",
    max_parents: int = 8,
) -> SecurityGraph:
    """Build a security graph from ordered trace events.

    ``dependency_mode="prefix"`` reproduces the R1 construction: every action or
    output depends on its entire trace prefix and every earlier sensitive event
    may flow to it. ``dependency_mode="provenance"`` (R2) links an action only to
    the trusted instructions and to the observations whose values it copies; see
    :mod:`causalkgguard.provenance`.
    """

    if dependency_mode not in {"prefix", "provenance"}:
        raise ValueError(f"unknown dependency_mode: {dependency_mode}")
    graph = SecurityGraph()
    provenance_index = None
    if dependency_mode == "provenance":
        from .provenance import ProvenanceIndex, action_values

        provenance_index = ProvenanceIndex(max_parents=max_parents)
    instruction_ids: List[str] = []
    graph.add_node(trace.case_id, "Trace", label=trace.label, attack_type=trace.attack_type)
    graph.add_node("user_goal", "Goal", text=trace.user_goal)
    graph.add_edge(trace.case_id, "has_goal", "user_goal")

    previous_event_id: Optional[str] = None
    sensitive_sources: List[str] = []
    event_ids: List[str] = []

    for policy_id in POLICY_IDS:
        graph.add_node(f"policy:{policy_id}", "Policy", name=policy_id)
        graph.add_edge(trace.case_id, "has_policy", f"policy:{policy_id}")

    for permission in trace.required_permissions:
        permission_id = f"perm:{permission}"
        graph.add_node(permission_id, "Permission", name=permission)
        graph.add_edge("user_goal", "requires_permission", permission_id)

    for index, event in enumerate(trace.events):
        node_type = _node_type_for_event(event)
        graph.add_node(
            event.event_id,
            node_type,
            role=event.role,
            content=event.content,
            actor=event.actor,
            target=event.target,
            trusted=event.trusted,
            sensitivity=event.sensitivity,
            tool=event.tool,
            sink=event.sink,
            permissions=list(event.permissions),
            index=index,
            metadata=dict(event.metadata),
        )
        graph.add_edge(trace.case_id, "contains_event", event.event_id)

        if previous_event_id is not None:
            graph.add_edge(previous_event_id, "precedes", event.event_id)
            graph.add_edge(previous_event_id, "temporal_dependency", event.event_id)
        previous_event_id = event.event_id

        if event.role in INSTRUCTION_ROLES:
            instruction_id = f"instruction:{event.event_id}"
            graph.add_node(
                instruction_id,
                "Instruction",
                role=event.role,
                trusted=event.trusted,
                text=event.content,
            )
            graph.add_edge(event.event_id, "contains_instruction", instruction_id)

        if event.role in DATA_SOURCE_ROLES:
            source_id = f"source:{event.event_id}"
            graph.add_node(
                source_id,
                "DataSource",
                role=event.role,
                trusted=event.trusted,
                text=event.content,
            )
            graph.add_edge(event.event_id, "observed_from", source_id)

        if event.tool:
            tool_id = f"tool:{event.tool}"
            graph.add_node(tool_id, "Tool", name=event.tool)
            graph.add_edge(event.event_id, "calls" if event.role == "action" else "mentions_tool", tool_id)

        if event.sink:
            sink_id = f"sink:{event.sink}"
            graph.add_node(sink_id, "Sink", name=event.sink)
            graph.add_edge(event.event_id, "writes_to", sink_id)

        for permission in event.permissions:
            permission_id = f"perm:{permission}"
            graph.add_node(permission_id, "Permission", name=permission)
            graph.add_edge(event.event_id, "uses_permission", permission_id)

        if event.sensitivity > 0:
            sensitive_id = f"sensitive:{event.event_id}"
            graph.add_node(
                sensitive_id,
                "SensitiveData",
                source_event=event.event_id,
                level=event.sensitivity,
            )
            graph.add_edge(event.event_id, "contains", sensitive_id)
            sensitive_sources.append(event.event_id)

        if event.role in {"action", "output"}:
            # Provenance-aware dependencies are built for tool calls, whose arguments are
            # structured; free-text responses keep the conservative prefix dependencies.
            if provenance_index is None or event.role == "output":
                for source_id in event_ids:
                    graph.add_edge(
                        event.event_id,
                        "depends_on",
                        source_id,
                        dependency="observed_trace_prefix",
                    )
                for source_id in sensitive_sources:
                    graph.add_edge(source_id, "may_flow_to", event.event_id)
            else:
                values = (
                    action_values(event.content, event.metadata)
                    if event.role == "action"
                    else [event.content]
                )
                data_parents = provenance_index.parents_for_values(values)
                declared = list(event.metadata.get("derived_from", []) or [])
                if event.target in event_ids:
                    declared.append(event.target)
                for source_id in declared:
                    if source_id in event_ids and source_id not in data_parents:
                        data_parents.append(source_id)
                for source_id in instruction_ids:
                    graph.add_edge(event.event_id, "depends_on", source_id, dependency="provenance_instruction")
                for source_id in data_parents:
                    graph.add_edge(event.event_id, "depends_on", source_id, dependency="provenance_data")
                sensitive_set = set(sensitive_sources)
                for source_id in data_parents:
                    if source_id in sensitive_set:
                        graph.add_edge(source_id, "may_flow_to", event.event_id, dependency="provenance_data")

        if provenance_index is not None:
            if event.role in DATA_SOURCE_ROLES:
                provenance_index.add(event.event_id, event.content)
            elif event.role in INSTRUCTION_ROLES:
                instruction_ids.append(event.event_id)

        event_ids.append(event.event_id)

    return graph


def _node_type_for_event(event: Event) -> str:
    if event.role in INSTRUCTION_ROLES:
        return "InstructionEvent"
    if event.role in DATA_SOURCE_ROLES:
        return "DataSourceEvent"
    if event.role in ACTION_ROLES:
        return "Action"
    return "Event"
