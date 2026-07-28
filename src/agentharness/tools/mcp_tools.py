"""Wrap tools discovered on MCP servers as ordinary harness tools.

The same factory shape as ``make_skill_tools`` and ``make_fs_tools``: hand it
the live collection, get back Tools the registry can hold. Once registered, an
MCP tool is indistinguishable from a built-in one to the agent loop — it is
dispatched, timed out, and error-wrapped by exactly the same code.

Security note: the name and description of every tool here come from a remote
server and are injected into the model's context on each turn. Namespacing
(done in ``mcp.manager``) stops a server claiming a built-in's name; the
per-server ``tools:`` allowlist limits what a chatty or hostile server can put
in front of the model in the first place.
"""

from __future__ import annotations

from typing import Any

from agentharness.mcp.manager import McpManager, McpTool
from agentharness.tools.registry import Tool


def make_mcp_tools(manager: McpManager) -> list[Tool]:
    """One Tool per discovered MCP tool, in discovery order."""
    return [_wrap(manager, tool) for tool in manager.tools]


def _wrap(manager: McpManager, tool: McpTool) -> Tool:
    def handler(args: dict[str, Any]) -> str:
        return manager.call(tool, args)

    return Tool(
        name=tool.name,
        description=f"[mcp:{tool.server}] {tool.description}",
        input_schema=tool.input_schema,
        handler=handler,
    )
