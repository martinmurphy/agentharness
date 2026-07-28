"""OAuth 2.1 against a real authorisation server running on localhost.

The one thing not exercised here is a human looking at a browser: the test
substitutes ``webbrowser.open`` with an HTTP GET of the same URL. Everything
else is the real path — discovery, dynamic client registration, PKCE, the code
exchange through our own loopback listener, and the cached grant being reused
on the next connect.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import httpx
import pytest

from agentharness.config import Config
from agentharness.mcp import oauth
from agentharness.mcp.config import parse_servers
from agentharness.mcp.runtime import McpConnectionError, McpRuntime

FIXTURE = str(Path(__file__).parent / "fixtures" / "oauth_mcp_server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def oauth_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, FIXTURE, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("oauth fixture exited early")
            with socket.socket() as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def token_home(tmp_path, monkeypatch):
    """Point the token cache at a temp dir so no real credentials are touched."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config" / "agentharness" / oauth.TOKEN_FILE


@pytest.fixture
def fake_browser(monkeypatch):
    """Stand in for the human: fetch the authorisation URL instead of opening it."""
    opened: list[str] = []

    def open_url(url: str) -> bool:
        opened.append(url)
        # In a thread, because the loopback listener the redirect lands on is
        # only started once this returns to the SDK's flow.
        threading.Thread(
            target=lambda: httpx.get(url, follow_redirects=True, timeout=10),
            daemon=True,
        ).start()
        return True

    monkeypatch.setattr(oauth.webbrowser, "open", open_url)
    return opened


@pytest.fixture
def runtime():
    rt = McpRuntime()
    yield rt
    rt.shutdown()


def _server(url: str, name: str = "secured", **overrides):
    # oauth_timeout is the human-in-the-loop budget; the tests have no human,
    # so it is short here. In real use it is minutes (DEFAULT_OAUTH_TIMEOUT).
    block = {
        "type": "http",
        "url": url,
        "auth": "oauth",
        "timeout": 20,
        "oauth_timeout": 20,
        **overrides,
    }
    return parse_servers(Config(mcp_servers={name: block}))[0]


# ---- the whole flow ----------------------------------------------------------


def test_oauth_flow_connects_and_caches_the_grant(
    runtime, oauth_server, token_home, fake_browser, capsys
):
    tools = runtime.connect(_server(oauth_server))
    assert {t.name for t in tools} == {"secret"}
    assert "the vault is open" in runtime.call_tool("secured", "secret", {}).content[0].text

    # The user was shown the URL, not just handed to a browser silently.
    assert fake_browser and "authorise this harness at" in capsys.readouterr().out

    stored = json.loads(token_home.read_text())["servers"][oauth_server]
    assert stored["tokens"]["access_token"]
    assert stored["client_info"]["client_id"]  # dynamic registration happened


def test_a_cached_grant_is_reused_without_authorising_again(
    runtime, oauth_server, token_home, fake_browser
):
    runtime.connect(_server(oauth_server))
    runtime.disconnect("secured")
    assert len(fake_browser) == 1

    runtime.connect(_server(oauth_server))
    assert len(fake_browser) == 1, "the second connect re-ran the browser flow"


def test_forget_makes_the_next_connect_authorise_again(
    runtime, oauth_server, token_home, fake_browser
):
    server = _server(oauth_server)
    runtime.connect(server)
    runtime.disconnect("secured")

    oauth.forget(server)  # what /mcp login does first
    assert oauth_server not in json.loads(token_home.read_text())["servers"]

    runtime.connect(server)
    assert len(fake_browser) == 2


def test_a_fixed_callback_port_is_honoured(runtime, oauth_server, token_home, fake_browser):
    port = _free_port()
    runtime.connect(_server(oauth_server, callback_port=port))
    # The redirect_uri is URL-encoded inside the authorisation URL.
    assert urllib.parse.quote(f"http://127.0.0.1:{port}/callback", safe="") in fake_browser[0]


def test_without_a_browser_the_url_is_still_printed(
    runtime, oauth_server, token_home, monkeypatch, capsys
):
    # A machine with no browser must still be able to complete the flow by
    # hand, so opening one failing cannot be fatal.
    def explode(_url: str) -> bool:
        raise RuntimeError("no browser here")

    monkeypatch.setattr(oauth.webbrowser, "open", explode)
    server = _server(oauth_server, timeout=2, oauth_timeout=2)
    with pytest.raises(McpConnectionError):
        runtime.connect(server)  # nobody completes it, so it times out
    assert "authorise this harness at" in capsys.readouterr().out


# ---- token storage -----------------------------------------------------------


def test_the_token_file_is_private(runtime, oauth_server, token_home, fake_browser):
    runtime.connect(_server(oauth_server))
    assert stat.S_IMODE(token_home.stat().st_mode) == 0o600
    assert stat.S_IMODE(token_home.parent.stat().st_mode) == 0o700


def test_a_corrupt_token_file_is_treated_as_empty(token_home, tmp_path):
    token_home.parent.mkdir(parents=True)
    token_home.write_text("{not json at all", encoding="utf-8")
    storage = oauth.FileTokenStorage(server_url="https://e.com/mcp", path=token_home)
    assert storage._load() == {}


def test_servers_do_not_share_tokens(token_home):
    from mcp.shared.auth import OAuthToken

    a = oauth.FileTokenStorage(server_url="https://a.example/mcp", path=token_home)
    b = oauth.FileTokenStorage(server_url="https://b.example/mcp", path=token_home)
    import anyio

    anyio.run(a.set_tokens, OAuthToken(access_token="for-a", token_type="Bearer"))
    anyio.run(b.set_tokens, OAuthToken(access_token="for-b", token_type="Bearer"))
    assert anyio.run(a.get_tokens).access_token == "for-a"
    assert anyio.run(b.get_tokens).access_token == "for-b"


# ---- the container boundary --------------------------------------------------


def test_oauth_is_refused_in_the_container(monkeypatch, tmp_path):
    marker = tmp_path / ".containerenv"
    marker.touch()
    monkeypatch.setattr(oauth, "_CONTAINER_MARKERS", (str(marker),))
    server = _server("https://mcp.example/mcp")
    with pytest.raises(oauth.OAuthUnavailable, match="token_env"):
        oauth.build_auth(server)


def test_a_container_refusal_reaches_the_user_as_a_connect_error(runtime, monkeypatch, tmp_path):
    marker = tmp_path / ".containerenv"
    marker.touch()
    monkeypatch.setattr(oauth, "_CONTAINER_MARKERS", (str(marker),))
    with pytest.raises(McpConnectionError, match="token_env"):
        runtime.connect(_server("https://mcp.example/mcp", timeout=5))


def test_container_detection_looks_for_runtime_markers(monkeypatch, tmp_path):
    marker = tmp_path / "marker"
    monkeypatch.setattr(oauth, "_CONTAINER_MARKERS", (str(marker),))
    assert oauth.in_container() is False
    marker.touch()
    assert oauth.in_container() is True


def test_the_token_directory_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert oauth.token_dir() == tmp_path / "xdg" / "agentharness"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert oauth.token_dir() == Path(os.path.expanduser("~")) / ".config" / "agentharness"
