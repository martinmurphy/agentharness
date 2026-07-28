"""Tests for the sync/async bridge, against a real MCP server over stdio.

These spawn a subprocess (tests/fixtures/echo_mcp_server.py) rather than mock
the SDK: the whole point of the module is that context managers, threads and
subprocesses are handled correctly, and a mock would test none of that. No test
touches the network.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from agentharness.config import Config
from agentharness.mcp.config import parse_servers
from agentharness.mcp.runtime import McpConnectionError, McpRuntime

FIXTURE = str(Path(__file__).parent / "fixtures" / "echo_mcp_server.py")


def _server(name: str = "echo", **overrides):
    block = {"type": "stdio", "command": sys.executable, "args": [FIXTURE], **overrides}
    return parse_servers(Config(mcp_servers={name: block}))[0]


@pytest.fixture
def runtime():
    rt = McpRuntime()
    yield rt
    rt.shutdown()


def _mcp_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "mcp-runtime"]


def _pid_watcher(tmp_path, monkeypatch):
    """Have the next server announce its PID, and return a liveness probe.

    Checking the process itself, rather than our own bookkeeping, is the whole
    point: a subprocess we merely forgot about is exactly the bug this guards.
    """
    pid_file = tmp_path / "server.pid"
    monkeypatch.setenv("ECHO_PID_FILE", str(pid_file))

    def alive() -> bool:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pid = int(pid_file.read_text())
        for _ in range(100):  # a terminated child is reaped shortly after
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            time.sleep(0.05)
        return True

    return alive


# ---- connect / discover / call ----------------------------------------------


def test_connect_discovers_tools(runtime):
    tools = runtime.connect(_server())
    assert {t.name for t in tools} >= {"echo", "boom", "read_env"}
    assert runtime.is_connected("echo")


def test_call_tool_returns_the_servers_result(runtime):
    runtime.connect(_server())
    result = runtime.call_tool("echo", "echo", {"text": "hi"})
    assert result.isError is False
    assert "echo: hi" in result.content[0].text


def test_a_failing_tool_comes_back_as_an_error_result(runtime):
    # The server reports failure in-band; it is not a transport error.
    runtime.connect(_server())
    result = runtime.call_tool("echo", "boom", {})
    assert result.isError is True


def test_env_pass_adds_to_the_safe_default_environment(runtime, monkeypatch):
    # PATH has to survive or a server launched as `uvx …` cannot start at all;
    # anything not on the safe list and not named in env_pass must not.
    monkeypatch.setenv("PASSED_THROUGH", "yes")
    monkeypatch.setenv("HELD_BACK", "should-not-be-visible")
    runtime.connect(_server(env_pass=["PASSED_THROUGH"]))

    def read(name: str) -> str:
        return runtime.call_tool("echo", "read_env", {"name": name}).content[0].text

    assert "yes" in read("PASSED_THROUGH")
    assert "<unset>" not in read("PATH")
    assert "<unset>" in read("HELD_BACK")


def test_a_server_with_no_env_pass_still_gets_a_usable_environment(runtime):
    runtime.connect(_server())
    assert "<unset>" not in runtime.call_tool("echo", "read_env", {"name": "PATH"}).content[0].text


def test_calling_an_unconnected_server_is_an_error(runtime):
    with pytest.raises(McpConnectionError, match="not connected"):
        runtime.call_tool("nope", "echo", {})


def test_connecting_twice_is_refused(runtime):
    runtime.connect(_server())
    with pytest.raises(McpConnectionError, match="already connected"):
        runtime.connect(_server())


# ---- failure modes -----------------------------------------------------------


def test_a_command_that_does_not_exist_fails_with_the_server_named(runtime):
    server = _server(command="definitely-not-a-real-binary-xyz")
    with pytest.raises(McpConnectionError, match="'echo'"):
        runtime.connect(server)
    assert not runtime.is_connected("echo")


def test_a_server_that_never_handshakes_times_out(runtime):
    server = _server(args=[FIXTURE, "--hang"], timeout=1)
    with pytest.raises(McpConnectionError, match="did not respond within 1s"):
        runtime.connect(server)
    # The failed attempt must not be left registered.
    assert not runtime.is_connected("echo")
    assert runtime.names() == []


def test_a_servers_stderr_is_quoted_back_but_not_printed(runtime, capfd):
    # Servers log to stderr routinely, so it must not land on the terminal and
    # scribble over the REPL — but when one dies on startup, stderr is usually
    # the only place it says why, so the tail belongs in the error.
    dying = _server(command=sys.executable, args=["-c", "import sys; sys.exit(2)"])
    noisy = _server(
        command=sys.executable,
        args=["-c", "import sys; sys.stderr.write('config error: no token\\n'); sys.exit(2)"],
    )
    with pytest.raises(McpConnectionError):
        runtime.connect(dying)
    with pytest.raises(McpConnectionError, match="config error: no token"):
        runtime.connect(noisy)
    assert "config error" not in capfd.readouterr().err


def test_a_slow_call_times_out_without_killing_the_session(runtime):
    runtime.connect(_server(timeout=1))
    with pytest.raises(McpConnectionError, match="timed out after 1s"):
        runtime.call_tool("echo", "slow", {"seconds": 5})


# ---- lifecycle ---------------------------------------------------------------


def test_disconnect_reaps_the_subprocess(runtime, tmp_path, monkeypatch):
    alive = _pid_watcher(tmp_path, monkeypatch)
    runtime.connect(_server(env_pass=["ECHO_PID_FILE"]))
    assert alive()
    runtime.disconnect("echo")
    assert not runtime.is_connected("echo")
    assert runtime.names() == []
    assert not alive(), "the stdio server outlived its connection"


def test_shutdown_reaps_the_subprocess_and_the_thread(tmp_path, monkeypatch):
    before = len(_mcp_threads())
    alive = _pid_watcher(tmp_path, monkeypatch)
    rt = McpRuntime()
    rt.connect(_server(env_pass=["ECHO_PID_FILE"]))
    assert alive()
    rt.shutdown()
    # join() in shutdown() is the guarantee; assert it actually happened.
    assert len(_mcp_threads()) == before
    assert not alive(), "the stdio server outlived the runtime"


def test_a_server_that_hangs_is_still_reaped_after_a_failed_connect(tmp_path, monkeypatch):
    # The handshake timeout has to unwind the transport, not just give up on
    # waiting for it — otherwise the child is orphaned for the whole session.
    alive = _pid_watcher(tmp_path, monkeypatch)
    rt = McpRuntime()
    try:
        with pytest.raises(McpConnectionError):
            rt.connect(_server(args=[FIXTURE, "--hang"], timeout=1, env_pass=["ECHO_PID_FILE"]))
        assert not alive(), "the hung server was left running"
    finally:
        rt.shutdown()


def test_shutdown_is_safe_when_nothing_was_connected():
    rt = McpRuntime()
    rt.shutdown()  # never started
    rt.start()
    rt.shutdown()
    rt.shutdown()  # twice


def test_runtime_can_be_restarted_after_shutdown(runtime):
    runtime.connect(_server())
    runtime.shutdown()
    assert runtime.connect(_server())  # start() is idempotent and re-arms
