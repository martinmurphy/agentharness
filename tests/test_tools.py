"""Tests for the tool registry, greet, and skill tools."""

from __future__ import annotations

import pytest

from agentharness.providers.base import ToolCall
from agentharness.skills.loader import load_skills
from agentharness.tools.greet import greet_tool
from agentharness.tools.model_tools import list_models_tool
from agentharness.tools.registry import ToolRegistry, build_default_registry


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
