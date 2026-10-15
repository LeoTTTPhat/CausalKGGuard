"""AgentDojo runtime defense integration for CausalKGGuard.

The integration is intentionally kept outside the AgentDojo package. It builds a
normal AgentDojo pipeline but replaces the tool executor with a guarded executor
that removes blocked tool calls before AgentDojo records function traces.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from causalkgguard import CausalKGGuard
from causalkgguard.detectors import inspect_text
from causalkgguard.schema import Event, TraceCase


def build_causalkgguard_agentdojo_pipeline(
    model: str,
    model_id: str | None = None,
    allowed_tools: Sequence[str] = (),
    threshold: float = 0.60,
    policy_config: str | Path | None = None,
    pipeline_name: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    tool_output_format: str | None = None,
):
    """Build an AgentDojo pipeline with CausalKGGuard before tool execution."""

    from agentdojo.agent_pipeline.agent_pipeline import (
        AgentPipeline,
        PipelineConfig,
        get_llm,
    )
    from agentdojo.agent_pipeline.agent_pipeline import MODEL_PROVIDERS, ModelsEnum
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, tool_result_to_str

    model_value = _normalize_agentdojo_model(model)
    config = PipelineConfig(
        llm=model_value,
        model_id=model_id,
        defense=None,
        tool_delimiter=tool_delimiter,
        system_message_name=system_message_name,
        system_message=system_message,
        tool_output_format=tool_output_format,  # type: ignore[arg-type]
    )
    llm = get_llm(MODEL_PROVIDERS[ModelsEnum(config.llm)], config.llm, config.model_id, config.tool_delimiter)
    if config.tool_output_format == "json":
        tool_output_formatter = partial(tool_result_to_str, dump_fn=json.dumps)
    else:
        tool_output_formatter = tool_result_to_str

    guarded_executor = CausalKGGuardToolsExecutor(
        allowed_tools=allowed_tools,
        threshold=threshold,
        policy_config=policy_config,
        tool_output_formatter=tool_output_formatter,
    )
    tools_loop = ToolsExecutionLoop([guarded_executor, llm])
    pipeline = AgentPipeline([SystemMessage(config.system_message or ""), InitQuery(), llm, tools_loop])
    pipeline.name = pipeline_name or f"{model_value}-causalkgguard"
    return pipeline


class CausalKGGuardToolsExecutor:
    """AgentDojo pipeline element that blocks risky tool calls before execution."""

    name = "causalkgguard_tool_executor"

    def __init__(
        self,
        allowed_tools: Sequence[str] = (),
        threshold: float = 0.60,
        policy_config: str | Path | None = None,
        tool_output_formatter=None,
        dependency_mode: str = "prefix",
        decision_rule: str = "score",
        lift_threshold: float = 0.20,
        decision_scope: str = "trace",
    ) -> None:
        """``decision_scope="trace"`` reproduces R1 (block if any event of the
        reconstructed trace scores >= threshold); ``"candidate"`` (R2) decides on
        the candidate call only, using the configured enforcement rule."""
        from agentdojo.agent_pipeline.tool_execution import ToolsExecutor

        self.allowed_tools = list(dict.fromkeys(str(tool) for tool in allowed_tools))
        self.guard = CausalKGGuard(
            threshold=threshold,
            finding_threshold=0.0,
            policy_path=policy_config,
            dependency_mode=dependency_mode,
            decision_rule=decision_rule,
            lift_threshold=lift_threshold,
        )
        self.decision_scope = decision_scope
        self.threshold = threshold
        self.executor = ToolsExecutor(tool_output_formatter=tool_output_formatter) if tool_output_formatter else ToolsExecutor()

    def query(
        self,
        query: str,
        runtime,
        env=None,
        messages: Sequence[Mapping[str, Any]] = (),
        extra_args: dict | None = None,
    ):
        from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
        from agentdojo.types import text_content_block_from_string

        if extra_args is None:
            extra_args = {}
        if not messages:
            return query, runtime, env, messages, extra_args
        last_message = messages[-1]
        if last_message.get("role") != "assistant":
            return query, runtime, env, messages, extra_args
        tool_calls = list(last_message.get("tool_calls") or [])
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        allowed_calls = []
        blocked_calls = []
        for tool_call in tool_calls:
            if getattr(tool_call, "function", None) == EMPTY_FUNCTION_NAME:
                allowed_calls.append(tool_call)
                continue
            decision = self._decision_for_tool_call(query, messages, tool_call)
            if decision["blocked"]:
                blocked_calls.append(decision)
            else:
                allowed_calls.append(tool_call)

        if blocked_calls:
            extra_args.setdefault("causalkgguard_blocks", []).extend(blocked_calls)

        if len(allowed_calls) == len(tool_calls):
            return self.executor.query(query, runtime, env, messages, extra_args)

        replacement = dict(last_message)
        replacement["tool_calls"] = allowed_calls
        if not allowed_calls:
            names = ", ".join(str(item["tool"]) for item in blocked_calls)
            replacement["content"] = [
                text_content_block_from_string(
                    f"CausalKGGuard blocked unsafe tool call(s): {names}. Continue without executing them."
                )
            ]
            guarded_messages = [*messages[:-1], replacement]
            return query, runtime, env, guarded_messages, extra_args

        guarded_messages = [*messages[:-1], replacement]
        return self.executor.query(query, runtime, env, guarded_messages, extra_args)

    def _decision_for_tool_call(
        self,
        query: str,
        messages: Sequence[Mapping[str, Any]],
        tool_call: Any,
    ) -> dict[str, Any]:
        trace = trace_from_agentdojo_messages(
            query=query,
            messages=messages[:-1],
            candidate_tool_call=tool_call,
            required_permissions=self.allowed_tools,
        )
        if self.decision_scope == "candidate":
            max_finding = self.guard.decide_candidate(trace)
            score = max_finding.risk_score if max_finding is not None else 0.0
            blocked = bool(max_finding is not None and max_finding.evidence.get("intercept"))
        else:
            findings = self.guard.evaluate(trace)
            max_finding = max(findings, key=lambda finding: finding.risk_score, default=None)
            score = max_finding.risk_score if max_finding is not None else 0.0
            blocked = score >= self.threshold
        return {
            "blocked": blocked,
            "counterfactual_score": float(max_finding.evidence.get("counterfactual_without_untrusted_instruction", 0.0))
            if max_finding is not None else 0.0,
            "causal_lift": float(max_finding.causal_lift) if max_finding is not None else 0.0,
            "tool": str(getattr(tool_call, "function", "unknown_tool")),
            "score": score,
            "reasons": list(max_finding.reasons if max_finding is not None else []),
            "features": dict(max_finding.evidence.get("features", {}) if max_finding is not None else {}),
            "policy_violations": list(
                max_finding.evidence.get("policy_violations", []) if max_finding is not None else []
            ),
        }


def trace_from_agentdojo_messages(
    query: str,
    messages: Sequence[Mapping[str, Any]],
    candidate_tool_call: Any,
    required_permissions: Sequence[str],
) -> TraceCase:
    events: List[Event] = []
    user_goal = query
    event_index = 1

    for message in messages:
        role = str(message.get("role") or "event")
        content = _message_content_to_text(message)
        if role == "user" and not user_goal:
            user_goal = content
        if role == "assistant":
            if content:
                events.append(
                    Event(
                        event_id=f"e{event_index}",
                        role="output",
                        content=content,
                        trusted=True,
                        sensitivity=_sensitivity(content),
                        metadata={"agentdojo_role": role},
                    )
                )
                event_index += 1
            for call in message.get("tool_calls") or []:
                events.append(_event_from_tool_call(event_index, call))
                event_index += 1
            continue
        if role == "tool":
            tool_call = message.get("tool_call")
            tool = str(getattr(tool_call, "function", "unknown_tool"))
            events.append(
                Event(
                    event_id=f"e{event_index}",
                    role="tool",
                    content=content or str(message.get("error") or ""),
                    tool=tool,
                    target=_target_for_tool(tool, getattr(tool_call, "args", {}) or {}),
                    trusted=False,
                    permissions=[tool],
                    sensitivity=_sensitivity(content),
                    metadata={"agentdojo_role": role, "error": message.get("error")},
                )
            )
            event_index += 1
            continue
        mapped_role = "output" if role == "assistant" else role
        events.append(
            Event(
                event_id=f"e{event_index}",
                role=mapped_role,
                content=content,
                trusted=role in {"system", "user", "assistant"},
                sensitivity=_sensitivity(content) if mapped_role == "output" else 0,
                metadata={"agentdojo_role": role},
            )
        )
        event_index += 1

    events.append(_event_from_tool_call(event_index, candidate_tool_call))
    return TraceCase(
        case_id="agentdojo_live_candidate",
        label="unknown",
        user_goal=user_goal,
        events=events,
        attack_type="unknown",
        required_permissions=list(required_permissions),
        metadata={
            "source": "agentdojo_live",
            "allowed_sinks": [],
            "allow_sensitive_external": True,
        },
    )


def _event_from_tool_call(index: int, tool_call: Any) -> Event:
    tool = str(getattr(tool_call, "function", "unknown_tool"))
    args = getattr(tool_call, "args", {}) or {}
    return Event(
        event_id=f"e{index}",
        role="action",
        content=json.dumps(args, sort_keys=True, default=str),
        tool=tool,
        target=_target_for_tool(tool, args),
        sink=_sink_for_tool(tool, args),
        trusted=True,
        permissions=[tool],
        sensitivity=0,
        metadata={"agentdojo_role": "assistant", "tool_call_id": getattr(tool_call, "id", None), "args": args},
    )


def _message_content_to_text(message: Mapping[str, Any]) -> str:
    from agentdojo.types import get_text_content_as_str

    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return get_text_content_as_str(content)


def _sink_for_tool(tool: str, args: Mapping[str, Any]) -> str | None:
    lowered = tool.lower()
    read_prefixes = (
        "get_",
        "list_",
        "read_",
        "search_",
        "find_",
        "check_",
        "fetch_",
    )
    if lowered.startswith(read_prefixes):
        return None
    if any(hint in lowered for hint in ("money", "transfer", "payment", "pay_")):
        return "financial:external"
    if any(hint in lowered for hint in ("book", "reserve", "reservation")):
        return "booking:external"
    if any(hint in lowered for hint in ("send_email", "send_", "email.send", "share_file")):
        return "email:external"
    if any(hint in lowered for hint in ("send_direct_message", "send_channel_message", "post_webpage", "invite_user")):
        return "slack:external"
    if "delete" in lowered:
        return "file:destructive"
    if any(hint in lowered for hint in ("create", "append", "update", "reschedule", "add_user", "remove_user")):
        return "state:external"
    return None


def _target_for_tool(tool: str, args: Mapping[str, Any]) -> str | None:
    for key in ("recipient", "recipients", "email", "url", "file_id", "id", "participant", "hotel_name"):
        if key in args:
            return str(args[key])
    return tool


def _sensitivity(text: str) -> int:
    return 1 if inspect_text(text).sensitive > 0 else 0


def allowed_tools_from_task(suite, task) -> list[str]:
    """Infer task-authorized tools from AgentDojo ground-truth calls."""

    try:
        environment = suite.load_and_inject_default_environment({})
        if hasattr(task, "init_environment"):
            environment = task.init_environment(environment)
        calls = task.ground_truth(environment)
    except Exception:
        return []
    tools = []
    for call in calls:
        tool = getattr(call, "function", None)
        if tool is None and isinstance(call, Mapping):
            tool = call.get("function")
        if tool is not None:
            tools.append(str(tool))
    return list(dict.fromkeys(tools))


def _normalize_agentdojo_model(model: str) -> str:
    from agentdojo.models import ModelsEnum

    if model in ModelsEnum.__members__:
        return str(ModelsEnum[model].value)
    return model
