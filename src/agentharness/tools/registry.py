"""The tool registry: names -> handlers, plus dispatch of a ToolCall.

A handler takes the parsed argument dict and returns a string. ``dispatch``
never raises: a handler exception (or an unknown tool) becomes a ToolResult
with ``is_error=True`` so the model can see the failure and recover, rather
than the REPL crashing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentharness.providers.base import ToolCall, ToolResult, ToolSpec

Handler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n].spec() for n in sorted(self._tools)]

    def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.id,
                content=f"Error: unknown tool {call.name!r}",
                is_error=True,
            )
        try:
            content = tool.handler(call.arguments)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; surfaced to the model
            return ToolResult(
                call_id=call.id,
                content=f"Error: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(call_id=call.id, content=content, is_error=False)


def build_default_registry(skillset) -> ToolRegistry:
    """Build the registry with the example ``greet`` tool and the skill tools.

    ``skillset`` is a live reference; the skill tools read from it at call time,
    so a ``/reload`` that replaces the SkillSet is picked up if callers rebuild
    the registry (see repl.reload).
    """
    from agentharness.tools.greet import greet_tool
    from agentharness.tools.provider_tools import list_providers_tool
    from agentharness.tools.skill_tools import make_skill_tools
    from agentharness.tools.web_tools import web_fetch_tool

    registry = ToolRegistry()
    registry.register(greet_tool())
    registry.register(list_providers_tool())
    registry.register(web_fetch_tool())
    for tool in make_skill_tools(skillset):
        registry.register(tool)
    return registry
