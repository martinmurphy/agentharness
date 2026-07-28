"""Streamable HTTP transport and static auth, against a real local server.

The server is a subprocess bound to 127.0.0.1 on a free port
(tests/fixtures/http_mcp_server.py). Nothing here reaches the network.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentharness.config import Config
from agentharness.mcp.config import parse_servers
from agentharness.mcp.manager import McpManager
from agentharness.mcp.runtime import McpConnectionError, McpRuntime

FIXTURE = str(Path(__file__).parent / "fixtures" / "http_mcp_server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_listening(port: int, proc: subprocess.Popen, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"fixture server exited early: {proc.returncode}")
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("fixture server never started listening")


@pytest.fixture
def http_server():
    """Start the fixture server; yields (url, token). Token auth is required."""
    port = _free_port()
    token = "s3cret-token"
    proc = subprocess.Popen(
        [sys.executable, FIXTURE, str(port), token],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_listening(port, proc)
        yield f"http://127.0.0.1:{port}/mcp", token
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def runtime():
    rt = McpRuntime()
    yield rt
    rt.shutdown()


def _server(url: str, name: str = "remote", **overrides):
    block = {"type": "http", "url": url, "timeout": 20, **overrides}
    return parse_servers(Config(mcp_servers={name: block}))[0]


def test_connect_over_http_and_call_a_tool(runtime, http_server, monkeypatch):
    url, token = http_server
    monkeypatch.setenv("REMOTE_TOKEN", token)
    tools = runtime.connect(_server(url, token_env="REMOTE_TOKEN"))
    assert {t.name for t in tools} >= {"ping", "whoami"}
    assert "pong" in runtime.call_tool("remote", "ping", {}).content[0].text


def test_a_wrong_token_fails_the_server_not_the_process(runtime, monkeypatch, http_server):
    url, _token = http_server
    monkeypatch.setenv("REMOTE_TOKEN", "wrong")
    with pytest.raises(McpConnectionError, match="'remote'"):
        runtime.connect(_server(url, token_env="REMOTE_TOKEN"))


def test_a_missing_token_variable_names_the_variable(runtime, monkeypatch, http_server):
    url, _token = http_server
    monkeypatch.delenv("REMOTE_TOKEN", raising=False)
    with pytest.raises(McpConnectionError, match="reads REMOTE_TOKEN, which is not set"):
        runtime.connect(_server(url, token_env="REMOTE_TOKEN"))


def test_env_references_in_headers_reach_the_server(runtime, http_server, monkeypatch):
    url, token = http_server
    monkeypatch.setenv("REMOTE_TOKEN", token)
    monkeypatch.setenv("TENANT", "acme")
    runtime.connect(_server(url, token_env="REMOTE_TOKEN", headers={"X-Tenant": "${TENANT}"}))
    result = runtime.call_tool("remote", "whoami", {"header": "x-tenant"})
    assert "acme" in result.content[0].text


def test_one_failing_server_does_not_stop_the_others(http_server, monkeypatch):
    # The point of collecting errors rather than raising: a dead server costs
    # you that server, not the session.
    url, token = http_server
    monkeypatch.setenv("REMOTE_TOKEN", token)
    servers = parse_servers(
        Config(
            mcp_servers={
                "dead": {"type": "http", "url": "http://127.0.0.1:1/mcp", "timeout": 5},
                "remote": {
                    "type": "http",
                    "url": url,
                    "token_env": "REMOTE_TOKEN",
                    "timeout": 20,
                },
            }
        )
    )
    mgr = McpManager(servers)
    try:
        mgr.connect_all()
        assert [e.server for e in mgr.errors] == ["dead"]
        assert mgr.connected == ["remote"]
        assert {t.remote_name for t in mgr.tools} >= {"ping"}
    finally:
        mgr.shutdown()
