"""Tests for parsing and validating the mcp_servers config block."""

from __future__ import annotations

import pytest

from agentharness.config import Config, load_config
from agentharness.mcp.config import (
    DEFAULT_TIMEOUT,
    McpConfigError,
    McpServerConfig,
    parse_servers,
)


def _cfg(**servers) -> Config:
    return Config(mcp_servers=dict(servers))


def _one(**block) -> McpServerConfig:
    return parse_servers(_cfg(srv=block))[0]


# ---- loading ----------------------------------------------------------------


def test_absent_block_means_no_servers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTHARNESS_CONFIG", raising=False)
    cfg = load_config()
    assert cfg.mcp_servers == {}
    assert parse_servers(cfg) == []


def test_block_is_read_from_yaml(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "mcp_servers:\n"
        "  fs:\n"
        "    type: stdio\n"
        "    command: uvx\n"
        "    args: [mcp-server-filesystem, /workspace]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTHARNESS_CONFIG", str(path))
    servers = parse_servers(load_config())
    assert [s.name for s in servers] == ["fs"]
    assert servers[0].command == "uvx"
    assert servers[0].args == ("mcp-server-filesystem", "/workspace")


def test_order_is_preserved_and_disabled_servers_are_kept():
    servers = parse_servers(
        _cfg(
            a={"type": "stdio", "command": "x"},
            b={"type": "stdio", "command": "y", "enabled": False},
        )
    )
    assert [(s.name, s.enabled) for s in servers] == [("a", True), ("b", False)]


# ---- validation -------------------------------------------------------------


def test_type_is_required_and_must_be_known():
    with pytest.raises(McpConfigError, match="'type' is required"):
        _one(command="x")
    with pytest.raises(McpConfigError, match="unknown type 'sse'"):
        _one(type="sse", command="x")


def test_unknown_key_names_the_server_and_the_key():
    with pytest.raises(McpConfigError, match=r"'srv': unknown key 'commnad'"):
        _one(type="stdio", commnad="x")


def test_http_keys_are_rejected_on_a_stdio_server():
    # The message should say which transport it is judging against.
    with pytest.raises(McpConfigError, match="unknown key 'url' for a stdio server"):
        _one(type="stdio", command="x", url="https://example.com/mcp")


def test_stdio_requires_a_command():
    with pytest.raises(McpConfigError, match="'command' is required"):
        _one(type="stdio")
    with pytest.raises(McpConfigError, match="'command' is required"):
        _one(type="stdio", command="   ")


def test_http_requires_an_http_url():
    with pytest.raises(McpConfigError, match="'url' is required"):
        _one(type="http")
    with pytest.raises(McpConfigError, match="url must be http or https"):
        _one(type="http", url="ftp://example.com/mcp")


def test_bad_scalar_types_are_rejected():
    with pytest.raises(McpConfigError, match="'enabled' must be true or false"):
        _one(type="stdio", command="x", enabled="yes")
    with pytest.raises(McpConfigError, match="'timeout' must be a positive number"):
        _one(type="stdio", command="x", timeout=0)
    with pytest.raises(McpConfigError, match="'args' must be a list of strings"):
        _one(type="stdio", command="x", args="--flag")
    with pytest.raises(McpConfigError, match="expected a mapping"):
        parse_servers(_cfg(srv="stdio"))


def test_auth_defaults_from_what_is_configured():
    assert _one(type="http", url="https://e.com/mcp").auth == "none"
    assert _one(type="http", url="https://e.com/mcp", token_env="T").auth == "bearer"
    assert _one(type="http", url="https://e.com/mcp", auth="oauth").auth == "oauth"
    with pytest.raises(McpConfigError, match="auth 'bearer' needs 'token_env'"):
        _one(type="http", url="https://e.com/mcp", auth="bearer")


def test_defaults():
    s = _one(type="stdio", command="x")
    assert s.enabled is True
    assert s.timeout == DEFAULT_TIMEOUT
    assert s.tools is None


# ---- the tools allowlist ----------------------------------------------------


def test_absent_allowlist_admits_everything():
    assert _one(type="stdio", command="x").allows("anything") is True


def test_explicit_empty_allowlist_admits_nothing():
    # Distinct from an absent key: `tools: []` is a deliberate "advertise none".
    s = _one(type="stdio", command="x", tools=[])
    assert s.tools == ()
    assert s.allows("anything") is False


def test_allowlist_filters():
    s = _one(type="stdio", command="x", tools=["read", "write"])
    assert s.allows("read") and not s.allows("delete")


# ---- credential resolution (deferred to connect time) -----------------------


def test_token_env_becomes_a_bearer_header(monkeypatch):
    monkeypatch.setenv("SRV_TOKEN", "s3cret")
    s = _one(type="http", url="https://e.com/mcp", token_env="SRV_TOKEN")
    # The declaration holds the variable name, never the value.
    assert "s3cret" not in repr(s)
    assert s.resolve_headers() == {"Authorization": "Bearer s3cret"}


def test_env_references_in_headers_are_expanded(monkeypatch):
    monkeypatch.setenv("TENANT", "acme")
    s = _one(type="http", url="https://e.com/mcp", headers={"X-Tenant": "${TENANT}/eu"})
    assert s.resolve_headers() == {"X-Tenant": "acme/eu"}


def test_missing_variable_fails_with_the_server_named(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    s = _one(type="http", url="https://e.com/mcp", token_env="NOPE")
    with pytest.raises(ValueError, match=r"mcp server 'srv' reads NOPE, which is not set"):
        s.resolve_headers()


def test_explicit_authorization_header_wins_over_token_env(monkeypatch):
    monkeypatch.setenv("SRV_TOKEN", "s3cret")
    s = _one(
        type="http",
        url="https://e.com/mcp",
        token_env="SRV_TOKEN",
        headers={"Authorization": "Basic abc"},
    )
    assert s.resolve_headers()["Authorization"] == "Basic abc"


def test_env_pass_names_the_extras_a_stdio_server_gets(monkeypatch):
    monkeypatch.setenv("WANTED", "yes")
    monkeypatch.setenv("SECRET_UNRELATED", "no")
    s = _one(type="stdio", command="x", env_pass=["WANTED", "ABSENT"])
    assert s.extra_env() == {"WANTED": "yes"}
