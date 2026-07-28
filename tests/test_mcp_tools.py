"""MCP tools end to end: a real stdio server, through the registry and the loop.

The naming and rendering tests are pure functions and cheap; the integration
tests spawn tests/fixtures/echo_mcp_server.py so that what is verified is the
whole path a real turn takes — discovery, registration, dispatch, rendering.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from mcp import types

from agentharness import agent, repl
from agentharness.config import Config
from agentharness.mcp.config import parse_servers
from agentharness.mcp.manager import (
    MAX_TOOL_NAME,
    McpManager,
    qualified_name,
    render_result,
)
from agentharness.providers.base import Message, ProviderResponse, TextBlock, ToolCall, Usage
from agentharness.tools.mcp_tools import make_mcp_tools
from agentharness.tools.registry import ToolRegistry

FIXTURE = str(Path(__file__).parent / "fixtures" / "echo_mcp_server.py")


def _servers(**blocks):
    return parse_servers(Config(mcp_servers=dict(blocks)))


def _echo_block(**overrides):
    return {"type": "stdio", "command": sys.executable, "args": [FIXTURE], **overrides}


@pytest.fixture
def manager():
    mgr = McpManager(_servers(echo=_echo_block()))
    mgr.connect_all()
    yield mgr
    mgr.shutdown()


# ---- naming -----------------------------------------------------------------


def test_names_are_namespaced():
    assert qualified_name("fs", "read") == "mcp__fs__read"


def test_names_are_sanitised():
    # Providers accept only [A-Za-z0-9_-]; a dotted or spaced server name would
    # otherwise be rejected at call time by the model API, not here.
    assert qualified_name("my.server", "do it!") == "mcp__my_server__do_it_"


def test_long_names_are_capped_and_stay_distinct():
    a = qualified_name("server", "x" * 200)
    b = qualified_name("server", "x" * 199 + "y")
    assert len(a) == MAX_TOOL_NAME and len(b) == MAX_TOOL_NAME
    assert a != b
    assert a.startswith("mcp__server__")


# ---- result rendering --------------------------------------------------------


def _result(*content, is_error=False, structured=None):
    return types.CallToolResult(
        content=list(content), isError=is_error, structuredContent=structured
    )


def test_text_blocks_are_joined():
    out = render_result(
        _result(types.TextContent(type="text", text="a"), types.TextContent(type="text", text="b"))
    )
    assert out == "a\nb"


def test_non_text_blocks_become_a_placeholder():
    out = render_result(_result(types.ImageContent(type="image", data="AAAA", mimeType="image/png")))
    assert "image/png" in out and "<image:" in out


def test_structured_content_is_used_when_there_are_no_blocks():
    assert '"ok": true' in render_result(_result(structured={"ok": True}))


def test_oversized_results_are_truncated():
    huge = types.TextContent(type="text", text="x" * 200_000)
    out = render_result(_result(huge))
    assert len(out) < 120_000
    assert out.endswith("(result truncated)")


# ---- discovery and dispatch --------------------------------------------------


def test_discovered_tools_are_namespaced_and_described(manager):
    names = {t.name for t in manager.tools}
    assert "mcp__echo__echo" in names
    assert manager.errors == []
    assert manager.connected == ["echo"]


def test_registry_dispatch_returns_the_servers_output(manager):
    registry = ToolRegistry()
    for tool in make_mcp_tools(manager):
        registry.register(tool)

    result = registry.dispatch(
        ToolCall(id="c1", name="mcp__echo__echo", arguments={"text": "hi"})
    )
    assert result.is_error is False
    assert "echo: hi" in result.content


def test_a_failing_tool_becomes_an_error_result_not_a_crash(manager):
    registry = ToolRegistry()
    for tool in make_mcp_tools(manager):
        registry.register(tool)

    result = registry.dispatch(ToolCall(id="c1", name="mcp__echo__boom", arguments={}))
    assert result.is_error is True
    assert "always fails" in result.content


def test_the_allowlist_limits_what_is_advertised():
    mgr = McpManager(_servers(echo=_echo_block(tools=["echo"])))
    try:
        mgr.connect_all()
        assert [t.remote_name for t in mgr.tools] == ["echo"]
    finally:
        mgr.shutdown()


def test_a_disabled_server_is_not_connected():
    mgr = McpManager(_servers(echo=_echo_block(enabled=False)))
    try:
        mgr.connect_all()
        assert mgr.tools == [] and mgr.connected == [] and mgr.errors == []
    finally:
        mgr.shutdown()


def test_an_unreachable_server_is_an_error_not_an_exception():
    mgr = McpManager(_servers(nope={"type": "stdio", "command": "definitely-not-real-xyz"}))
    try:
        mgr.connect_all()
        assert [e.server for e in mgr.errors] == ["nope"]
        assert mgr.tools == []
    finally:
        mgr.shutdown()


# ---- several servers at once -------------------------------------------------


def test_servers_connect_concurrently():
    # Three servers that each take a second to start should cost about one
    # second, not three: the REPL cannot prompt until discovery is done.
    delay = 1.0
    blocks = {
        f"s{i}": _echo_block(args=[FIXTURE, "--delay", str(delay)]) for i in range(3)
    }
    mgr = McpManager(_servers(**blocks))
    try:
        started = time.monotonic()
        mgr.connect_all()
        elapsed = time.monotonic() - started
        assert mgr.connected == ["s0", "s1", "s2"]
        assert elapsed < delay * 2, f"connects look serial: {elapsed:.1f}s for 3x{delay}s"
    finally:
        mgr.shutdown()


def test_results_are_applied_in_config_order_not_completion_order():
    # The slow server is listed first, so if ordering followed completion the
    # fast one would win — and which tool survives a name clash would become a
    # race.
    mgr = McpManager(
        _servers(
            slow=_echo_block(args=[FIXTURE, "--delay", "1"]),
            fast=_echo_block(),
        )
    )
    try:
        mgr.connect_all()
        assert mgr.connected == ["slow", "fast"]
        assert mgr.tools[0].server == "slow"
    finally:
        mgr.shutdown()


def test_a_name_collision_is_reported_and_the_first_server_keeps_the_name():
    # 'a.b' and 'a_b' both sanitise to mcp__a_b__*, so their tools collide.
    mgr = McpManager(_servers(**{"a.b": _echo_block(), "a_b": _echo_block()}))
    try:
        mgr.connect_all()
        assert sorted(mgr.connected) == ["a.b", "a_b"]
        # Every advertised name is unique, so the registry never sees a clash.
        names = [t.name for t in mgr.tools]
        assert len(names) == len(set(names))
        assert all(t.server == "a.b" for t in mgr.tools)
        assert [e.server for e in mgr.errors] == ["a_b"] * 4
        assert "already taken" in mgr.errors[0].reason
    finally:
        mgr.shutdown()


def test_registering_every_tool_from_two_servers_never_duplicates():
    mgr = McpManager(_servers(one=_echo_block(), two=_echo_block()))
    try:
        mgr.connect_all()
        registry = ToolRegistry()
        for tool in make_mcp_tools(mgr):
            registry.register(tool)  # raises on duplicates
        assert len(registry.names()) == len(mgr.tools) == 8
    finally:
        mgr.shutdown()


# ---- through the harness and the agent loop ---------------------------------


def _harness_with_mcp(tmp_path, monkeypatch, provider, **block_overrides):
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: provider)
    cfg = Config(
        skills_dir=str(tmp_path),
        workspace_dir=str(ws),
        mcp_servers={"echo": _echo_block(**block_overrides)},
    )
    return repl.Harness(cfg)


class _ScriptedProvider:
    name = "fake"

    def __init__(self, script):
        self._script = list(script)

    def chat(self, *, system, messages, tools, max_tokens):
        self.tools_seen = tools
        return self._script.pop(0)


def test_an_mcp_tool_is_callable_by_the_model(tmp_path, monkeypatch):
    provider = _ScriptedProvider(
        [
            ProviderResponse(
                message=Message(
                    role="assistant",
                    blocks=[ToolCall(id="c1", name="mcp__echo__echo", arguments={"text": "hi"})],
                ),
                stop_reason="tool_use",
                usage=Usage(1, 1),
            ),
            ProviderResponse(
                message=Message(role="assistant", blocks=[TextBlock("done")]),
                stop_reason="end_turn",
                usage=Usage(1, 1),
            ),
        ]
    )
    h = _harness_with_mcp(tmp_path, monkeypatch, provider)
    try:
        state = h.states.active
        state.add(Message(role="user", blocks=[TextBlock("say hi")]))
        events = list(
            agent.run_turn(
                provider=provider,
                state=state,
                registry=h.registry,
                system="SYS",
                max_tokens=100,
                max_iterations=5,
            )
        )
        results = [e for e in events if isinstance(e, agent.ToolResultEvent)]
        assert len(results) == 1
        assert "echo: hi" in results[0].result.content
        assert results[0].result.is_error is False
        # The tool was advertised to the model, not just dispatchable.
        assert any(spec.name == "mcp__echo__echo" for spec in provider.tools_seen)
    finally:
        h.shutdown()


def test_subagents_get_the_same_mcp_tools(tmp_path, monkeypatch):
    h = _harness_with_mcp(tmp_path, monkeypatch, _ScriptedProvider([]))
    try:
        assert "mcp__echo__echo" in h._subagent_registry
        assert "mcp__echo__echo" in h.registry
    finally:
        h.shutdown()


# ---- the /mcp command --------------------------------------------------------


def test_mcp_command_reports_status_and_tool_counts(tmp_path, monkeypatch, capsys):
    h = _harness_with_mcp(tmp_path, monkeypatch, _ScriptedProvider([]))
    try:
        repl._handle_command(h, "/mcp")
        out = capsys.readouterr().out
        assert "echo" in out and "stdio" in out
        assert "connected" in out and "4 tool(s)" in out
    finally:
        h.shutdown()


def test_mcp_command_lists_one_servers_tools(tmp_path, monkeypatch, capsys):
    h = _harness_with_mcp(tmp_path, monkeypatch, _ScriptedProvider([]))
    try:
        repl._handle_command(h, "/mcp tools echo")
        out = capsys.readouterr().out
        assert "mcp__echo__echo" in out
        repl._handle_command(h, "/mcp tools nope")
        assert "no such mcp server" in capsys.readouterr().out
    finally:
        h.shutdown()


def test_mcp_command_shows_why_a_server_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _ScriptedProvider([]))
    cfg = Config(
        skills_dir=str(tmp_path),
        workspace_dir=str(tmp_path / "ws"),
        mcp_servers={"broken": {"type": "stdio", "command": "definitely-not-real-xyz"}},
    )
    h = repl.Harness(cfg)
    try:
        repl._handle_command(h, "/mcp")
        out = capsys.readouterr().out
        assert "failed" in out
        assert "! broken:" in out
    finally:
        h.shutdown()


def test_reconnect_recovers_a_server_that_was_down_at_startup(tmp_path, monkeypatch, capsys):
    # The realistic case: a token was missing at startup and has just been set.
    monkeypatch.delenv("LATE_TOKEN", raising=False)
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _ScriptedProvider([]))
    cfg = Config(
        skills_dir=str(tmp_path),
        workspace_dir=str(tmp_path / "ws"),
        mcp_servers={"echo": _echo_block(env_pass=["LATE_TOKEN"], command="missing-at-first")},
    )
    h = repl.Harness(cfg)
    try:
        assert h.mcp.errors and "mcp__echo__echo" not in h.registry
        # Repair the config the way a user would repair their environment.
        h.mcp.servers[0] = parse_servers(Config(mcp_servers={"echo": _echo_block()}))[0]
        repl._handle_command(h, "/mcp reconnect echo")
        assert "reconnected echo: 4 tool(s)" in capsys.readouterr().out
        assert h.mcp.errors == []
        # The registries were rebuilt, so the model can actually call it now.
        assert "mcp__echo__echo" in h.registry
        assert "mcp__echo__echo" in h._subagent_registry
    finally:
        h.shutdown()


def test_reconnect_reports_a_server_that_is_still_down(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _ScriptedProvider([]))
    cfg = Config(
        skills_dir=str(tmp_path),
        workspace_dir=str(tmp_path / "ws"),
        mcp_servers={"broken": {"type": "stdio", "command": "definitely-not-real-xyz"}},
    )
    h = repl.Harness(cfg)
    try:
        capsys.readouterr()
        repl._handle_command(h, "/mcp reconnect broken")
        assert "broken" in capsys.readouterr().out
        assert len(h.mcp.errors) == 1  # replaced, not accumulated
        repl._handle_command(h, "/mcp reconnect nope")
        assert "no such mcp server" in capsys.readouterr().out
    finally:
        h.shutdown()


def test_mcp_command_rejects_unknown_subcommands(tmp_path, monkeypatch, capsys):
    h = _harness_with_mcp(tmp_path, monkeypatch, _ScriptedProvider([]))
    try:
        repl._handle_command(h, "/mcp frobnicate")
        assert "usage: /mcp" in capsys.readouterr().out
    finally:
        h.shutdown()


def test_mcp_command_with_no_servers_configured(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _ScriptedProvider([]))
    h = repl.Harness(Config(skills_dir=str(tmp_path), workspace_dir=str(tmp_path / "ws")))
    repl._handle_command(h, "/mcp")
    assert "no mcp servers configured" in capsys.readouterr().out
    h.shutdown()


def test_a_harness_with_no_mcp_servers_is_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _ScriptedProvider([]))
    h = repl.Harness(Config(skills_dir=str(tmp_path), workspace_dir=str(tmp_path / "ws")))
    assert h.mcp.tools == [] and h.mcp.connected == []
    h.shutdown()  # must be safe even though no runtime ever started


# ---- error messages ----------------------------------------------------------


def _http_server(url: str, name: str = "remote"):
    return parse_servers(Config(mcp_servers={name: {"type": "http", "url": url}}))[0]


def test_a_404_is_reported_as_a_404_with_the_url():
    # The SDK reports a 404 on the message endpoint as "Session terminated",
    # which reads as an auth or session problem and sends you down the wrong
    # path. It is nearly always the wrong URL — say so.
    from agentharness.mcp.runtime import _connect_error

    server = _http_server("https://mcp.example.com/v1/sse")
    err = _connect_error(server, RuntimeError("Session terminated"))
    text = str(err)
    assert "404" in text
    assert "https://mcp.example.com/v1/sse" in text
    assert "/sse" in text  # points at the likely cause


def test_an_http_status_error_reports_the_status_and_url():
    from types import SimpleNamespace

    from agentharness.mcp.runtime import _connect_error

    server = _http_server("https://mcp.example.com/mcp")
    exc = RuntimeError("Client error '401 Unauthorized'")
    exc.response = SimpleNamespace(status_code=401)
    assert "HTTP 401 from https://mcp.example.com/mcp" in str(_connect_error(server, exc))


def test_a_connection_failure_names_the_host_it_could_not_reach():
    # errno-style messages name a syscall, never the endpoint.
    from agentharness.mcp.runtime import _connect_error

    server = _http_server("https://mcp.internal.example/mcp")
    err = _connect_error(server, OSError("[Errno 8] nodename nor servname provided"))
    assert "mcp.internal.example" in str(err)


def test_a_stdio_failure_is_left_alone():
    # stdio errors already carry the server's stderr tail; a URL would be a lie.
    from agentharness.mcp.runtime import _connect_error

    server = parse_servers(Config(mcp_servers={"fs": {"type": "stdio", "command": "x"}}))[0]
    text = str(_connect_error(server, FileNotFoundError("no such file: x")))
    assert "no such file: x" in text
    assert "http" not in text.lower()


def test_a_turn_failure_names_the_provider_and_endpoint(tmp_path, monkeypatch, capsys):
    # "APIConnectionError: Connection error." alone does not say which of the
    # configured providers failed, or what host it could not reach.
    class _Exploding:
        name = "fake"

        def chat(self, **_kwargs):
            raise RuntimeError("Connection error.")

    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _Exploding())
    cfg = Config(
        skills_dir=str(tmp_path),
        workspace_dir=str(tmp_path / "ws"),
        provider="modelscorp",
        model="Qwen/Qwen3-14B",
        providers={"modelscorp": {"type": "openai", "base_url": "https://qwen.internal/v1"}},
    )
    h = repl.Harness(cfg)
    try:
        h.run_prompt("who are you")
        out = capsys.readouterr().out
        assert "RuntimeError: Connection error." in out
        assert "modelscorp" in out
        assert "https://qwen.internal/v1" in out
        assert "Qwen/Qwen3-14B" in out
    finally:
        h.shutdown()


def test_a_turn_failure_surfaces_the_underlying_cause(tmp_path, monkeypatch, capsys):
    # An SDK's "Connection error." is the same sentence for DNS failure, a
    # refused port and an untrusted certificate. The chain says which.
    class _Exploding:
        name = "fake"

        def chat(self, **_kwargs):
            try:
                raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer")
            except OSError as inner:
                raise RuntimeError("Connection error.") from inner

    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _Exploding())
    h = repl.Harness(Config(skills_dir=str(tmp_path), workspace_dir=str(tmp_path / "ws")))
    try:
        h.run_prompt("hello")
        out = capsys.readouterr().out
        assert "Connection error." in out
        assert "CERTIFICATE_VERIFY_FAILED" in out  # the part that names the fix
    finally:
        h.shutdown()


def test_root_cause_survives_a_reraise_cycle():
    from agentharness.repl import _root_cause

    a = ValueError("a")
    b = ValueError("b")
    a.__context__ = b
    b.__context__ = a  # a cycle; must terminate
    assert _root_cause(a) is b
