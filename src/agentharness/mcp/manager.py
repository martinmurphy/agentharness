"""Connect the configured MCP servers and present their tools as one set.

The manager is to MCP what ``SkillSet`` is to skills: a live collection the
harness holds, carrying both what loaded and what failed. A server that cannot
be reached is an entry in ``errors``, never an exception into the REPL — one
unreachable server must not stop the others or the session.

Tool identity is decided here. A server's ``read`` becomes ``mcp__fs__read``:

* namespaced, so a server cannot shadow a built-in tool like ``write_file``
  merely by naming one of its own that way;
* sanitised and length-capped, because all three provider APIs constrain
  function names and an unsanitised server name would be a 400 at call time;
* collision-checked, so the registry's duplicate guard is never what discovers
  the problem.

Result rendering is here too, since it is MCP knowledge: content blocks in,
one string out, and a server-reported failure raised so that
``ToolRegistry.dispatch`` turns it into an error result the model can read.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from mcp import types

from agentharness.mcp.config import McpServerConfig
from agentharness.mcp.runtime import McpConnectionError, McpRuntime

# Provider function-name rules are the intersection of Anthropic's, OpenAI's and
# Gemini's: letters, digits, underscore and dash, 64 characters.
_NAME_SAFE = re.compile(r"[^A-Za-z0-9_-]")
MAX_TOOL_NAME = 64
_HASH_LEN = 6

# Cap on the text handed back from one tool call, matching web_fetch's intent:
# a tool result goes into the model's context and must not be unbounded.
MAX_RESULT_CHARS = 100_000

# Enough to make startup concurrent without spawning a thread per entry in a
# config that lists a great many servers.
_MAX_PARALLEL_CONNECTS = 8


@dataclass(frozen=True)
class McpError:
    """A server that could not be used, and why. Shown at startup and by /mcp."""

    server: str
    reason: str


@dataclass(frozen=True)
class McpTool:
    """One tool discovered on a server, under the name the model will use."""

    server: str
    remote_name: str  # what the server calls it
    name: str  # what the model calls it: mcp__<server>__<tool>
    description: str
    input_schema: dict[str, Any]


@dataclass
class McpManager:
    """Owns the runtime, the connected servers, and the discovered tools."""

    servers: list[McpServerConfig]
    runtime: McpRuntime = field(default_factory=McpRuntime)
    tools: list[McpTool] = field(default_factory=list)
    errors: list[McpError] = field(default_factory=list)
    connected: list[str] = field(default_factory=list)

    def connect_all(self) -> None:
        """Connect every enabled server, collecting failures instead of raising.

        Servers connect *concurrently* — each one costs a process spawn or a
        network round trip, and the REPL cannot show a prompt until the tool
        list is known, so doing them in series would make one slow server the
        startup time for all of them. Results are then applied in config order,
        so which tool wins a name collision does not depend on who answered
        first.
        """
        enabled = [s for s in self.servers if s.enabled]
        if not enabled:
            return
        self.runtime.start()
        with ThreadPoolExecutor(
            max_workers=min(len(enabled), _MAX_PARALLEL_CONNECTS),
            thread_name_prefix="mcp-connect",
        ) as pool:
            pending = [(s, pool.submit(self.runtime.connect, s)) for s in enabled]
            for server, future in pending:
                self._apply(server, future.result)

    def reconnect(self, name: str) -> None:
        """Drop and re-establish one server, replacing its tools and errors.

        The reason this exists: a server that was down at startup, or one whose
        token has just been set, should not cost a whole session to pick up.
        """
        server = self.server(name)
        if server is None:
            raise ValueError(f"no such mcp server: {name}")
        self.runtime.disconnect(name)
        self.tools = [t for t in self.tools if t.server != name]
        self.errors = [e for e in self.errors if e.server != name]
        if name in self.connected:
            self.connected.remove(name)
        if not server.enabled:
            return
        self._apply(server, lambda: self.runtime.connect(server))

    def server(self, name: str) -> McpServerConfig | None:
        return next((s for s in self.servers if s.name == name), None)

    def _apply(
        self, server: McpServerConfig, fetch: Callable[[], list[types.Tool]]
    ) -> None:
        """Record one server's outcome: its tools, or why it has none."""
        try:
            discovered = fetch()
        except McpConnectionError as exc:
            self.errors.append(McpError(server.name, str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 - a bad config or SDK failure is still just this server's
            self.errors.append(McpError(server.name, f"{type(exc).__name__}: {exc}"))
            return

        self.connected.append(server.name)
        taken = {t.name for t in self.tools}
        for tool in discovered:
            if not server.allows(tool.name):
                continue
            qualified = qualified_name(server.name, tool.name)
            if qualified in taken:
                self.errors.append(
                    McpError(
                        server.name,
                        f"tool {tool.name!r} maps to {qualified!r}, which is already taken; skipped",
                    )
                )
                continue
            taken.add(qualified)
            self.tools.append(
                McpTool(
                    server=server.name,
                    remote_name=tool.name,
                    name=qualified,
                    description=tool.description or f"Tool {tool.name!r} on MCP server {server.name!r}.",
                    input_schema=tool.inputSchema or {"type": "object", "properties": {}},
                )
            )

    def tools_for(self, server_name: str) -> list[McpTool]:
        return [t for t in self.tools if t.server == server_name]

    def call(self, tool: McpTool, arguments: dict[str, Any]) -> str:
        """Call one tool and render its result as text.

        Raises on failure — both a transport failure and a server-reported one.
        ``ToolRegistry.dispatch`` converts the exception into an error result,
        so the model sees what went wrong and can try something else.
        """
        result = self.runtime.call_tool(tool.server, tool.remote_name, arguments)
        text = render_result(result)
        if result.isError:
            raise ValueError(text or f"{tool.name} failed with no detail")
        return text

    def shutdown(self) -> None:
        self.runtime.shutdown()
        self.connected.clear()


def qualified_name(server: str, tool: str) -> str:
    """``mcp__<server>__<tool>``, sanitised and capped at 64 characters.

    Over-long names keep their readable prefix and gain a short hash of the full
    name, so two tools that truncate to the same stem stay distinguishable.
    """
    safe_server = _NAME_SAFE.sub("_", server)
    safe_tool = _NAME_SAFE.sub("_", tool)
    name = f"mcp__{safe_server}__{safe_tool}"
    if len(name) <= MAX_TOOL_NAME:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_HASH_LEN]
    return f"{name[: MAX_TOOL_NAME - _HASH_LEN - 1]}_{digest}"


def render_result(result: types.CallToolResult) -> str:
    """Flatten MCP content blocks into the single string a handler returns.

    Non-text blocks become a short placeholder rather than being dropped: the
    model should know something came back that it cannot see, which is the same
    choice ``web_fetch`` makes for non-text bodies.
    """
    parts: list[str] = []
    for block in result.content:
        parts.append(_render_block(block))
    if not parts and result.structuredContent is not None:
        parts.append(json.dumps(result.structuredContent, indent=2, default=str))
    text = "\n".join(p for p in parts if p)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n… (result truncated)"
    return text


def _render_block(block: types.ContentBlock) -> str:
    if isinstance(block, types.TextContent):
        return block.text
    if isinstance(block, types.ImageContent):
        return f"<image: {block.mimeType}, {len(block.data)} base64 chars>"
    if isinstance(block, types.AudioContent):
        return f"<audio: {block.mimeType}, {len(block.data)} base64 chars>"
    if isinstance(block, types.ResourceLink):
        return f"<resource: {block.uri}>"
    if isinstance(block, types.EmbeddedResource):
        resource = block.resource
        text = getattr(resource, "text", None)
        if text is not None:
            return str(text)
        return f"<embedded resource: {resource.uri}>"
    return f"<unsupported content block: {type(block).__name__}>"
