"""Tests for the tool registry, greet, and skill tools."""

from __future__ import annotations

import pytest

from agentharness.providers.base import ToolCall
from agentharness.skills.loader import load_skills
from agentharness.tools.greet import greet_tool
from agentharness.tools.model_tools import list_models_tool
from agentharness.tools.registry import ToolRegistry, build_default_registry
from agentharness.tools.subagent_tools import spawn_subagent_tool


def test_greet_styles():
    tool = greet_tool()
    assert "Martin" in tool.handler({"name": "Martin", "style": "formal"})
    assert tool.handler({"name": "Sam"}).startswith("Hey Sam")  # default casual


def test_greet_rejects_bad_style():
    tool = greet_tool()
    with pytest.raises(ValueError):
        tool.handler({"name": "X", "style": "sarcastic"})


def test_greet_requires_name():
    tool = greet_tool()
    with pytest.raises(ValueError):
        tool.handler({})


def test_dispatch_success():
    reg = ToolRegistry()
    reg.register(greet_tool())
    result = reg.dispatch(ToolCall(id="c1", name="greet", arguments={"name": "Ada"}))
    assert not result.is_error
    assert result.call_id == "c1"
    assert "Ada" in result.content


def test_dispatch_unknown_tool_is_error_not_raise():
    reg = ToolRegistry()
    result = reg.dispatch(ToolCall(id="c2", name="nope", arguments={}))
    assert result.is_error
    assert "unknown tool" in result.content


def test_dispatch_handler_exception_becomes_error():
    reg = ToolRegistry()
    reg.register(greet_tool())
    result = reg.dispatch(ToolCall(id="c3", name="greet", arguments={}))  # missing name
    assert result.is_error
    assert result.call_id == "c3"


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(greet_tool())
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(greet_tool())


def test_default_registry_includes_skill_and_provider_tools(tmp_path):
    skillset = load_skills(tmp_path)
    reg = build_default_registry(skillset)
    assert set(reg.names()) == {
        "greet",
        "list_providers",
        "web_fetch",
        "read_skill",
        "read_skill_file",
    }


def _models_tool(active="anthropic", by_name=None):
    """Build a list_models tool over fake closures.

    ``by_name`` maps provider name -> list of models (or raises if a name is
    absent, mimicking a missing-key error).
    """
    by_name = by_name or {}

    def list_for(name):
        if name not in by_name:
            raise RuntimeError(f"no key for {name}")
        return by_name[name]

    reg = ToolRegistry()
    reg.register(list_models_tool(lambda: active, list_for))
    return reg


def _dispatch(reg, args):
    return reg.dispatch(ToolCall(id="c", name="list_models", arguments=args))


def test_list_models_active_provider_default():
    reg = _models_tool(active="anthropic", by_name={"anthropic": ["model-a", "model-b"]})
    result = _dispatch(reg, {})
    assert not result.is_error
    assert "anthropic:" in result.content
    assert "model-a" in result.content and "model-b" in result.content


def test_list_models_named_provider():
    reg = _models_tool(active="anthropic", by_name={"gemini": ["g-1", "g-2"]})
    result = _dispatch(reg, {"provider": "gemini"})
    assert not result.is_error
    assert "gemini:" in result.content and "g-1" in result.content


def test_list_models_unknown_provider():
    reg = _models_tool()
    result = _dispatch(reg, {"provider": "bogus"})
    assert result.is_error
    assert "unknown provider" in result.content


def test_list_models_named_provider_missing_key_errors():
    reg = _models_tool(by_name={})  # list_for raises for any name
    result = _dispatch(reg, {"provider": "gemini"})
    assert result.is_error
    assert "no key for gemini" in result.content


def test_list_models_all_reports_available_and_unavailable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    # list_for succeeds for the two available ones; openai is skipped (no key).
    reg = _models_tool(by_name={"anthropic": ["c-1"], "gemini": ["g-1"]})
    result = _dispatch(reg, {"provider": "all"})
    assert not result.is_error
    assert "c-1" in result.content and "g-1" in result.content
    assert "openai: (no key set" in result.content   # unavailable noted, not fetched


def test_list_models_all_reports_per_provider_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    # gemini available (key set) but list_for raises -> reported, others still listed.
    reg = _models_tool(by_name={"anthropic": ["c-1"], "openai": ["o-1"]})
    result = _dispatch(reg, {"provider": "all"})
    assert not result.is_error
    assert "c-1" in result.content and "o-1" in result.content
    assert "gemini: error:" in result.content        # one failure doesn't abort "all"


def test_list_providers_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    reg = build_default_registry(load_skills(tmp_path))
    result = reg.dispatch(ToolCall(id="c", name="list_providers", arguments={}))
    assert not result.is_error
    assert "anthropic (key from ANTHROPIC_API_KEY): key set" in result.content
    assert "openai (key from OPENAI_API_KEY): no key" in result.content
    assert "gemini (key from GEMINI_API_KEY): key set" in result.content


def _skillset_with_one(tmp_path):
    d = tmp_path / "demo"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\n\nfull body here\n",
        encoding="utf-8",
    )
    (d / "references" / "R.md").write_text("ref body", encoding="utf-8")
    return load_skills(tmp_path)


def test_read_skill_tool_returns_body(tmp_path):
    reg = build_default_registry(_skillset_with_one(tmp_path))
    result = reg.dispatch(ToolCall(id="c", name="read_skill", arguments={"name": "demo"}))
    assert not result.is_error
    assert "full body here" in result.content


def test_read_skill_file_tool(tmp_path):
    reg = build_default_registry(_skillset_with_one(tmp_path))
    result = reg.dispatch(
        ToolCall(id="c", name="read_skill_file",
                 arguments={"skill": "demo", "path": "references/R.md"})
    )
    assert not result.is_error
    assert result.content == "ref body"


def test_read_skill_file_traversal_is_error(tmp_path):
    reg = build_default_registry(_skillset_with_one(tmp_path))
    result = reg.dispatch(
        ToolCall(id="c", name="read_skill_file",
                 arguments={"skill": "demo", "path": "../../etc/hostname"})
    )
    assert result.is_error


def test_spawn_subagent_tool_calls_runner():
    seen = {}

    def runner(task, provider, model):
        seen["args"] = (task, provider, model)
        return "the answer"

    reg = ToolRegistry()
    reg.register(spawn_subagent_tool(runner, ["anthropic", "gemini"]))
    result = reg.dispatch(
        ToolCall(id="c", name="spawn_subagent",
                 arguments={"task": "do x", "provider": "gemini"})
    )
    assert not result.is_error
    assert result.content == "the answer"
    assert seen["args"] == ("do x", "gemini", None)  # model omitted -> None (caller default)


def test_spawn_subagent_tool_requires_task():
    reg = ToolRegistry()
    reg.register(spawn_subagent_tool(lambda *a: "x", ["anthropic"]))
    result = reg.dispatch(ToolCall(id="c", name="spawn_subagent", arguments={}))
    assert result.is_error
    assert "task" in result.content


# ---- web_fetch --------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, status=200, reason="OK", headers=None, body=b"hello"):
        self.status = status
        self.reason = reason
        self.headers = headers or {"Content-Type": "text/plain"}
        self._body = body

    def read(self, n):
        return self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _web_reg():
    from agentharness.tools.web_tools import web_fetch_tool

    reg = ToolRegistry()
    reg.register(web_fetch_tool())
    return reg


def test_web_fetch_get(monkeypatch):
    from agentharness.tools import web_tools

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["headers"] = dict(request.header_items())
        return _FakeHTTPResponse(body=b'{"ok": true}',
                                 headers={"Content-Type": "application/json"})

    monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
    result = _web_reg().dispatch(
        ToolCall(id="c", name="web_fetch",
                 arguments={"url": "https://example.com/api", "accept": "application/json"})
    )
    assert not result.is_error
    assert "HTTP 200 OK" in result.content
    assert '{"ok": true}' in result.content
    assert captured["method"] == "GET"
    # header keys are title-cased by urllib
    assert captured["headers"].get("Accept") == "application/json"


def test_web_fetch_post_sets_body_and_content_type(monkeypatch):
    from agentharness.tools import web_tools

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["method"] = request.method
        captured["data"] = request.data
        captured["headers"] = dict(request.header_items())
        return _FakeHTTPResponse(status=201, reason="Created", body=b"done")

    monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
    result = _web_reg().dispatch(
        ToolCall(id="c", name="web_fetch", arguments={
            "url": "https://example.com/api",
            "method": "POST",
            "body": "field=value",
            "content_type": "application/x-www-form-urlencoded",
        })
    )
    assert not result.is_error
    assert "HTTP 201 Created" in result.content
    assert captured["method"] == "POST"
    assert captured["data"] == b"field=value"
    assert captured["headers"].get("Content-type") == "application/x-www-form-urlencoded"


def test_web_fetch_post_defaults_content_type_to_json(monkeypatch):
    from agentharness.tools import web_tools

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        return _FakeHTTPResponse()

    monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
    _web_reg().dispatch(
        ToolCall(id="c", name="web_fetch",
                 arguments={"url": "https://x.test", "method": "POST", "body": "{}"})
    )
    assert captured["headers"].get("Content-type") == "application/json"


def test_web_fetch_rejects_non_http_scheme():
    result = _web_reg().dispatch(
        ToolCall(id="c", name="web_fetch", arguments={"url": "file:///etc/passwd"})
    )
    assert result.is_error
    assert "scheme" in result.content


def test_web_fetch_requires_url():
    result = _web_reg().dispatch(ToolCall(id="c", name="web_fetch", arguments={}))
    assert result.is_error
    assert "url" in result.content


def test_web_fetch_returns_error_body(monkeypatch):
    import urllib.error

    from agentharness.tools import web_tools

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found",
            hdrs={"Content-Type": "text/plain"}, fp=None,
        )

    monkeypatch.setattr(web_tools.urllib.request, "urlopen", fake_urlopen)
    result = _web_reg().dispatch(
        ToolCall(id="c", name="web_fetch", arguments={"url": "https://x.test/missing"})
    )
    # a 4xx is not a tool error — it's a real HTTP response returned to the model
    assert not result.is_error
    assert "HTTP 404 Not Found" in result.content


def test_web_fetch_truncates_large_body(monkeypatch):
    from agentharness.tools import web_tools

    big = b"a" * (web_tools._MAX_BYTES + 500)

    monkeypatch.setattr(
        web_tools.urllib.request, "urlopen",
        lambda request, timeout=None: _FakeHTTPResponse(body=big),
    )
    result = _web_reg().dispatch(
        ToolCall(id="c", name="web_fetch", arguments={"url": "https://x.test/big"})
    )
    assert "truncated" in result.content
