"""Graph-grounded explanation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

from .policy import PolicyViolation
from .schema import Edge, SecurityGraph


@dataclass(frozen=True)
class EvidencePath:
    """A compact policy-relevant path through the security graph."""

    path_id: str
    policy_id: str
    summary: str
    nodes: List[Dict[str, object]] = field(default_factory=list)
    edges: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "path_id": self.path_id,
            "policy_id": self.policy_id,
            "summary": self.summary,
            "nodes": self.nodes,
            "edges": self.edges,
        }


def build_evidence_paths(
    graph: SecurityGraph,
    source_events: Mapping[str, object],
    violations: Sequence[PolicyViolation],
) -> List[EvidencePath]:
    """Build human-readable graph paths for policy violations."""

    action = _as_string(source_events.get("action"))
    if not action:
        return []

    paths: List[EvidencePath] = []
    for violation in violations:
        if violation.policy_id == "untrusted_instruction_to_privileged_action":
            for source in _as_list(source_events.get("untrusted_instruction_sources")):
                paths.append(_source_to_action_path(graph, violation.policy_id, source, action))
        elif violation.policy_id == "sensitive_data_to_unauthorized_external_sink":
            for source in _as_list(source_events.get("sensitive_sources")):
                paths.append(_sensitive_to_sink_path(graph, violation.policy_id, source, action))
        elif violation.policy_id == "excessive_agency":
            paths.append(_action_policy_path(graph, violation.policy_id, action, "uses permission outside task requirement"))
        elif violation.policy_id == "system_prompt_leakage":
            sources = _as_list(source_events.get("untrusted_system_request_sources"))
            if not sources:
                sources = _as_list(source_events.get("dependency_sources"))
            for source in sources:
                paths.append(_source_to_action_path(graph, violation.policy_id, source, action))
        else:
            paths.append(_action_policy_path(graph, violation.policy_id, action, violation.description))

    return paths


def graph_to_dot(
    graph: SecurityGraph,
    highlighted_nodes: Iterable[str] = (),
    highlighted_edges: Iterable[tuple[str, str, str]] = (),
) -> str:
    """Render a security graph as Graphviz DOT text."""

    highlighted_node_set = set(highlighted_nodes)
    highlighted_edge_set = set(highlighted_edges)
    lines = ["digraph CausalKGGuard {", "  rankdir=LR;", "  node [shape=box, fontsize=10];"]
    for node_id, node in sorted(graph.nodes.items()):
        attrs = {
            "label": f"{node_id}\\n{node.node_type}",
            "style": "filled",
            "fillcolor": _node_color(node.node_type, node_id in highlighted_node_set),
        }
        lines.append(f"  {_dot_id(node_id)} [{_dot_attrs(attrs)}];")

    for edge in graph.edges:
        attrs = {"label": edge.relation}
        if (edge.source, edge.relation, edge.target) in highlighted_edge_set:
            attrs["color"] = "red"
            attrs["penwidth"] = "2"
        lines.append(f"  {_dot_id(edge.source)} -> {_dot_id(edge.target)} [{_dot_attrs(attrs)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def highlighted_elements(paths: Sequence[Mapping[str, object]]) -> tuple[List[str], List[tuple[str, str, str]]]:
    nodes: List[str] = []
    edges: List[tuple[str, str, str]] = []
    for path in paths:
        for node in path.get("nodes", []):
            if isinstance(node, dict) and "id" in node:
                nodes.append(str(node["id"]))
        for edge in path.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", ""))
            relation = str(edge.get("relation", ""))
            target = str(edge.get("target", ""))
            if source and relation and target and not relation.endswith("(reverse)"):
                edges.append((source, relation, target))
    return sorted(set(nodes)), sorted(set(edges))


def _source_to_action_path(graph: SecurityGraph, policy_id: str, source: str, action: str) -> EvidencePath:
    nodes = [_node_ref(graph, source), _node_ref(graph, action)]
    edges = [{"source": source, "relation": "depends_on(reverse)", "target": action}]
    nodes.extend(_action_context_nodes(graph, action))
    edges.extend(_action_context_edges(graph, action))
    return EvidencePath(
        path_id=f"{policy_id}:{source}->{action}",
        policy_id=policy_id,
        summary=f"{source} is connected to risky action {action}.",
        nodes=_unique_dicts(nodes),
        edges=edges,
    )


def _sensitive_to_sink_path(graph: SecurityGraph, policy_id: str, source: str, action: str) -> EvidencePath:
    nodes = [_node_ref(graph, source), _node_ref(graph, action)]
    edges = [{"source": source, "relation": "may_flow_to", "target": action}]
    nodes.extend(_action_context_nodes(graph, action))
    edges.extend(_action_context_edges(graph, action))
    return EvidencePath(
        path_id=f"{policy_id}:{source}->{action}",
        policy_id=policy_id,
        summary=f"Sensitive source {source} may flow to action {action}.",
        nodes=_unique_dicts(nodes),
        edges=edges,
    )


def _action_policy_path(graph: SecurityGraph, policy_id: str, action: str, summary: str) -> EvidencePath:
    policy = f"policy:{policy_id}"
    nodes = [_node_ref(graph, action), _node_ref(graph, policy)]
    edges = [{"source": action, "relation": "violates", "target": policy}]
    nodes.extend(_action_context_nodes(graph, action))
    edges.extend(_action_context_edges(graph, action))
    return EvidencePath(
        path_id=f"{policy_id}:{action}",
        policy_id=policy_id,
        summary=summary,
        nodes=_unique_dicts(nodes),
        edges=edges,
    )


def _action_context_nodes(graph: SecurityGraph, action: str) -> List[Dict[str, object]]:
    refs: List[Dict[str, object]] = []
    for relation in ("calls", "writes_to", "uses_permission"):
        for edge in graph.outgoing(action, relation):
            refs.append(_node_ref(graph, edge.target))
    return refs


def _action_context_edges(graph: SecurityGraph, action: str) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    for relation in ("calls", "writes_to", "uses_permission"):
        for edge in graph.outgoing(action, relation):
            edges.append(_edge_ref(edge))
    return edges


def _node_ref(graph: SecurityGraph, node_id: str) -> Dict[str, object]:
    node = graph.get_node(node_id)
    if node is None:
        return {"id": node_id, "type": "Unknown"}
    return {"id": node_id, "type": node.node_type, "attrs": dict(node.attrs)}


def _edge_ref(edge: Edge) -> Dict[str, object]:
    return {
        "source": edge.source,
        "relation": edge.relation,
        "target": edge.target,
        "attrs": dict(edge.attrs),
    }


def _unique_dicts(items: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    result: List[Dict[str, object]] = []
    for item in items:
        key = str(item.get("id", item))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _as_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _as_string(value: object) -> str:
    return "" if value is None else str(value)


def _node_color(node_type: str, highlighted: bool) -> str:
    if highlighted:
        return "#ffd6d6"
    if node_type in {"Action", "Tool"}:
        return "#d7ebff"
    if node_type in {"DataSourceEvent", "SensitiveData"}:
        return "#fff0bf"
    if node_type in {"Policy"}:
        return "#eadcff"
    if node_type in {"Sink"}:
        return "#ffd9c8"
    return "#f2f2f2"


def _dot_id(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dot_attrs(attrs: Mapping[str, str]) -> str:
    return ", ".join(f"{key}={_dot_id(str(value))}" for key, value in attrs.items())
