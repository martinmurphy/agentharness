"""Tests for the tool registry, greet, and skill tools."""

from __future__ import annotations

import textwrap

import pytest

from agentharness.providers.base import ToolCall
from agentharness.skills.loader import load_skills
from agentharness.skills.runner import ScriptPolicy
from agentharness.tools.greet import greet_tool
from agentharness.tools.model_tools import list_models_tool
from agentharness.tools.registry import ToolRegistry, build_default_registry
from agentharness.tools.script_tools import make_script_tools
from agentharness.tools.subagent_tools import spawn_subagent_tool
from agentharness.workspace import Workspace


def _ws(tmp_path, *, writable=True):
    """A writable workspace under tmp_path, created so the tools can use it."""
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return Workspace(root=root, writable=writable)


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


_BASE_TOOLS = {
    "greet",
    "list_providers",
    "web_fetch",
    "web_search",
    "read_skill",
    "read_skill_file",
}
_FS_READ_TOOLS = {"list_dir", "read_file"}
_FS_WRITE_TOOLS = {"write_file", "make_dir"}


def test_default_registry_includes_skill_and_provider_tools(tmp_path):
    skillset = load_skills(tmp_path)
    reg = build_default_registry(skillset, _ws(tmp_path))
    assert set(reg.names()) == _BASE_TOOLS | _FS_READ_TOOLS | _FS_WRITE_TOOLS


def test_read_only_workspace_omits_write_tools(tmp_path):
    reg = build_default_registry(load_skills(tmp_path), _ws(tmp_path, writable=False))
    assert set(reg.names()) == _BASE_TOOLS | _FS_READ_TOOLS
    # Absent entirely, not present-but-refusing: the model is never offered them.
    assert not any(s.name in _FS_WRITE_TOOLS for s in reg.specs())


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
    reg = build_default_registry(load_skills(tmp_path), _ws(tmp_path))
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
    reg = build_default_registry(_skillset_with_one(tmp_path), _ws(tmp_path))
    result = reg.dispatch(ToolCall(id="c", name="read_skill", arguments={"name": "demo"}))
    assert not result.is_error
    assert "full body here" in result.content


def test_read_skill_file_tool(tmp_path):
    reg = build_default_registry(_skillset_with_one(tmp_path), _ws(tmp_path))
    result = reg.dispatch(
        ToolCall(id="c", name="read_skill_file",
                 arguments={"skill": "demo", "path": "references/R.md"})
    )
    assert not result.is_error
    assert result.content == "ref body"


def test_read_skill_file_traversal_is_error(tmp_path):
    reg = build_default_registry(_skillset_with_one(tmp_path), _ws(tmp_path))
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


def test_spawn_subagent_tool_advertises_batching():
    """The affordance has to be in the description or the model will not use it.

    Parallel dispatch only pays off when one assistant message carries several
    calls, and a model asked to consult four backends will otherwise spawn them
    one at a time — four round-trips, nothing overlapping. Nothing in the API
    can ask for batching; the description is the only lever. Pinned so a reword
    cannot quietly drop it.
    """
    tool = spawn_subagent_tool(lambda *a: "x", ["anthropic"])
    assert "one message" in tool.description
    assert "concurrently" in tool.description


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


# ---- web_search -------------------------------------------------------------


_DDG_HTML = """
<html><body>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc">Result One</a>
  <a class="result__snippet" href="x">Snippet <b>one</b> text</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftwo">Result Two</a>
  <a class="result__snippet">Snippet two</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fthree">Result Three</a>
  <a class="result__snippet">Snippet three</a>
</div>
</body></html>
"""


class _FakeSearchResponse:
    def __init__(self, html):
        self._body = html.encode("utf-8")

    def read(self, n):
        return self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_search(monkeypatch, html=None, exc=None):
    from agentharness.tools import search_tools

    def fake_urlopen(request, timeout=None):
        if exc is not None:
            raise exc
        return _FakeSearchResponse(html)

    monkeypatch.setattr(search_tools.urllib.request, "urlopen", fake_urlopen)


def _search_reg():
    from agentharness.tools.search_tools import web_search_tool

    reg = ToolRegistry()
    reg.register(web_search_tool())
    return reg


def test_web_search_parses_results(monkeypatch):
    _patch_search(monkeypatch, html=_DDG_HTML)
    result = _search_reg().dispatch(
        ToolCall(id="c", name="web_search", arguments={"query": "example"})
    )
    assert not result.is_error
    # uddg redirect decoded to the real URL
    assert "https://example.com/one" in result.content
    assert "Result One" in result.content
    assert "Snippet one text" in result.content  # nested <b> flattened
    assert "https://example.org/two" in result.content


def test_web_search_respects_count(monkeypatch):
    _patch_search(monkeypatch, html=_DDG_HTML)
    result = _search_reg().dispatch(
        ToolCall(id="c", name="web_search", arguments={"query": "example", "count": 2})
    )
    assert "1." in result.content and "2." in result.content
    assert "3." not in result.content  # capped at 2


def test_web_search_no_results(monkeypatch):
    _patch_search(monkeypatch, html="<html><body>nothing here</body></html>")
    result = _search_reg().dispatch(
        ToolCall(id="c", name="web_search", arguments={"query": "zxcvzxcv"})
    )
    assert not result.is_error  # empty is not an error
    assert "No results found" in result.content


def test_web_search_requires_query():
    result = _search_reg().dispatch(ToolCall(id="c", name="web_search", arguments={}))
    assert result.is_error
    assert "query" in result.content


def test_web_search_request_failure_is_error(monkeypatch):
    import urllib.error

    _patch_search(monkeypatch, exc=urllib.error.URLError("boom"))
    result = _search_reg().dispatch(
        ToolCall(id="c", name="web_search", arguments={"query": "example"})
    )
    assert result.is_error
    assert "search request failed" in result.content


# ---- filesystem tools (dispatch level) ------------------------------------


def _fs_reg(tmp_path, *, writable=True):
    return build_default_registry(load_skills(tmp_path), _ws(tmp_path, writable=writable))


def test_list_dir_and_read_file_tools(tmp_path):
    reg = _fs_reg(tmp_path)
    (tmp_path / "workspace" / "notes.md").write_text("hello", encoding="utf-8")

    listing = reg.dispatch(ToolCall(id="c", name="list_dir", arguments={}))
    assert not listing.is_error
    assert "notes.md" in listing.content

    read = reg.dispatch(ToolCall(id="c", name="read_file", arguments={"path": "notes.md"}))
    assert not read.is_error
    assert read.content == "hello"


def test_read_file_traversal_is_error(tmp_path):
    reg = _fs_reg(tmp_path)
    result = reg.dispatch(
        ToolCall(id="c", name="read_file", arguments={"path": "../../etc/hostname"})
    )
    assert result.is_error
    assert "escapes the workspace" in result.content


def test_read_file_requires_path(tmp_path):
    result = _fs_reg(tmp_path).dispatch(ToolCall(id="c", name="read_file", arguments={}))
    assert result.is_error
    assert "path" in result.content


def test_write_file_then_read_back(tmp_path):
    reg = _fs_reg(tmp_path)
    written = reg.dispatch(
        ToolCall(id="c", name="write_file", arguments={"path": "out.md", "content": "written"})
    )
    assert not written.is_error
    assert (tmp_path / "workspace" / "out.md").read_text(encoding="utf-8") == "written"

    read = reg.dispatch(ToolCall(id="c", name="read_file", arguments={"path": "out.md"}))
    assert read.content == "written"


def test_write_file_requires_content(tmp_path):
    result = _fs_reg(tmp_path).dispatch(
        ToolCall(id="c", name="write_file", arguments={"path": "out.md"})
    )
    assert result.is_error
    assert "content" in result.content


def test_make_dir_tool_then_write_into_it(tmp_path):
    reg = _fs_reg(tmp_path)
    made = reg.dispatch(ToolCall(id="c", name="make_dir", arguments={"path": "reports/2026"}))
    assert not made.is_error

    written = reg.dispatch(
        ToolCall(
            id="c",
            name="write_file",
            arguments={"path": "reports/2026/q3.md", "content": "ok"},
        )
    )
    assert not written.is_error
    assert (tmp_path / "workspace" / "reports" / "2026" / "q3.md").is_file()


def test_write_tools_unavailable_on_read_only_workspace(tmp_path):
    result = _fs_reg(tmp_path, writable=False).dispatch(
        ToolCall(id="c", name="write_file", arguments={"path": "out.md", "content": "x"})
    )
    assert result.is_error
    assert "unknown tool" in result.content


# ---- provider aliases through the tools ------------------------------------


def _alias_cfg():
    from agentharness.config import Config

    return Config(
        providers={
            "lmstudio": {"type": "openai", "base_url": "http://localhost:1234/v1"},
            "together": {"type": "openai", "api_key_env": "TOGETHER_API_KEY"},
        }
    )


def test_list_providers_tool_reports_aliases(tmp_path, monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    reg = build_default_registry(load_skills(tmp_path), _ws(tmp_path), _alias_cfg())
    result = reg.dispatch(ToolCall(id="c", name="list_providers", arguments={}))
    assert not result.is_error
    assert "lmstudio (local endpoint, no key needed): ready" in result.content
    assert "together (key from TOGETHER_API_KEY): no key" in result.content
    assert "anthropic (key from ANTHROPIC_API_KEY)" in result.content  # built-ins remain


def test_list_providers_tool_without_config_is_builtins_only(tmp_path):
    reg = build_default_registry(load_skills(tmp_path), _ws(tmp_path))
    result = reg.dispatch(ToolCall(id="c", name="list_providers", arguments={}))
    assert "lmstudio" not in result.content


def test_list_models_accepts_an_alias_name():
    reg = ToolRegistry()
    reg.register(
        list_models_tool(lambda: "lmstudio", lambda name: ["qwen3.5-9b-mlx"], _alias_cfg())
    )
    result = reg.dispatch(
        ToolCall(id="c", name="list_models", arguments={"provider": "lmstudio"})
    )
    assert not result.is_error
    assert "qwen3.5-9b-mlx" in result.content
    # and the enum offered to the model lists it
    spec = reg.get("list_models").input_schema
    assert "lmstudio" in spec["properties"]["provider"]["enum"]


# ---- script_tools -----------------------------------------------------------


def _script_skillset(tmp_path, source='print("hi")\n', name="demo"):
    directory = tmp_path / "skills" / name
    (directory / "scripts").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A demo skill used by tests.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (directory / "scripts" / "run.py").write_text(
        textwrap.dedent(source).lstrip(), encoding="utf-8"
    )
    return load_skills(tmp_path / "skills")


def _script_registry(tmp_path, source='print("hi")\n', **policy_kwargs):
    reg = ToolRegistry()
    policy = ScriptPolicy(enabled=True, **policy_kwargs)
    for tool in make_script_tools(_script_skillset(tmp_path, source), _ws(tmp_path), policy):
        reg.register(tool)
    return reg


def test_disabled_policy_registers_no_tool(tmp_path):
    tools = make_script_tools(_script_skillset(tmp_path), _ws(tmp_path), ScriptPolicy())
    assert tools == []


def test_run_skill_script_renders_a_result(tmp_path):
    reg = _script_registry(tmp_path, """
        import sys
        print("counted 3 things")
        print("a warning", file=sys.stderr)
    """)
    result = reg.dispatch(ToolCall(
        id="c1",
        name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert not result.is_error
    assert "exit status: 0" in result.content
    assert "counted 3 things" in result.content
    assert "a warning" in result.content


def test_empty_streams_render_as_empty(tmp_path):
    reg = _script_registry(tmp_path, "pass\n")
    result = reg.dispatch(ToolCall(
        id="c2", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert "(empty)" in result.content


def test_non_zero_exit_is_not_a_tool_error(tmp_path):
    reg = _script_registry(tmp_path, "import sys; sys.exit(4)\n")
    result = reg.dispatch(ToolCall(
        id="c3", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert not result.is_error          # the model should read it, not give up
    assert "exit status: 4" in result.content


def test_non_string_args_are_rejected_with_advice(tmp_path):
    reg = _script_registry(tmp_path)
    result = reg.dispatch(ToolCall(
        id="c4", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py", "args": [3]},
    ))
    assert result.is_error
    assert "list of strings" in result.content


def test_bare_string_args_are_rejected_with_advice(tmp_path):
    reg = _script_registry(tmp_path)
    result = reg.dispatch(ToolCall(
        id="c4b", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py", "args": "abc"},
    ))
    assert result.is_error
    assert "list of strings" in result.content


def test_unknown_skill_names_the_available_ones(tmp_path):
    reg = _script_registry(tmp_path)
    result = reg.dispatch(ToolCall(
        id="c5", name="run_skill_script",
        arguments={"skill": "nope", "path": "scripts/run.py"},
    ))
    assert result.is_error
    assert "demo" in result.content


def test_timeout_is_an_error_carrying_partial_output(tmp_path):
    reg = _script_registry(tmp_path, """
        import time
        print("started", flush=True)
        time.sleep(60)
    """, default_timeout=1, max_timeout=1)
    result = reg.dispatch(ToolCall(
        id="c6", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert result.is_error
    assert "timeout" in result.content
    assert "started" in result.content


def test_timeout_renders_truncation_notice_when_output_was_truncated(tmp_path):
    """When a script floods output past max_output_bytes and then times out,
    the rendered error should include the truncation notice so the model
    knows the output was incomplete.
    """
    reg = _script_registry(tmp_path, """
        import sys, time
        # Write 200 bytes to exceed max_output_bytes of 100
        sys.stdout.write("x" * 200)
        sys.stdout.flush()
        # Then sleep long enough to timeout
        time.sleep(60)
    """, default_timeout=1, max_timeout=1, max_output_bytes=100)
    result = reg.dispatch(ToolCall(
        id="c7", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert result.is_error
    assert "timeout" in result.content
    # The key assertion: the truncation notice must appear so the model knows
    # output was cut off, not complete.
    assert "truncated" in result.content
    assert "more bytes" in result.content


def test_no_route_lets_the_model_run_its_own_code(tmp_path):
    """The headline invariant of the whole feature, exercised end to end
    through the real ToolRegistry rather than resolve_script directly: no
    call a model can actually make through ``run_skill_script`` may execute
    a file the model itself wrote into the workspace.

    Plants one file in the workspace and enumerates every route a model has
    to reach it:

    - ``..`` out of a normal skill's scripts/ directory
    - an absolute path
    - a symlink placed inside a normal skill's scripts/ directory
    - a skill whose scripts/ directory is *itself* a symlink into the
      workspace (the bug this branch's item 1 fixes: resolve_script's
      is_relative_to check resolves symlinks on both sides, so an escaped
      `scripts` symlink used to make every file under it pass)

    Every route must come back as a tool error, and the marker text the
    planted script would print if it ran must never appear in any result.
    """
    marker = "MODEL AUTHORED CODE RAN"
    ws = _ws(tmp_path)
    evil = ws.root / "evil.py"
    evil.write_text(f"print({marker!r})\n", encoding="utf-8")

    skills_root = tmp_path / "skills"
    normal = skills_root / "normal"
    (normal / "scripts").mkdir(parents=True)
    (normal / "SKILL.md").write_text(
        "---\nname: normal\ndescription: A normal skill.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (normal / "scripts" / "run.py").write_text('print("hi")\n', encoding="utf-8")
    # Route 3: a symlink placed inside scripts/, pointing at the workspace file.
    (normal / "scripts" / "link.py").symlink_to(evil)

    # Route 4: a skill whose scripts/ directory is itself a symlink into the
    # workspace.
    linked = skills_root / "linked"
    linked.mkdir(parents=True)
    (linked / "SKILL.md").write_text(
        "---\nname: linked\ndescription: A skill whose scripts/ is a symlink.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (linked / "scripts").symlink_to(ws.root)

    skillset = load_skills(skills_root)
    policy = ScriptPolicy(enabled=True)
    reg = build_default_registry(skillset, ws, policy=policy)
    assert "run_skill_script" in reg

    import os

    # rel_path is joined onto skill.path (resolve_script does skill.path /
    # rel_path), so a ".." escape written from scripts/ needs that prefix to
    # land where it looks like it should.
    dotdot_from_scripts = os.path.relpath(evil, start=(normal / "scripts"))
    routes = {
        "dotdot": ("normal", f"scripts/{dotdot_from_scripts}"),
        "absolute": ("normal", str(evil)),
        "symlink_in_scripts": ("normal", "scripts/link.py"),
        "scripts_dir_is_symlink": ("linked", "scripts/evil.py"),
    }
    for label, (skill_name, path) in routes.items():
        result = reg.dispatch(ToolCall(
            id=f"route-{label}",
            name="run_skill_script",
            arguments={"skill": skill_name, "path": path},
        ))
        assert result.is_error, f"route {label!r} was not refused: {result.content!r}"
        assert marker not in result.content, f"route {label!r} let the model's code run"
