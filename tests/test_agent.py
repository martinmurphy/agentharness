"""Agent loop, state manager, and REPL command tests using a fake provider."""

from __future__ import annotations

import pytest

from agentharness import agent
from agentharness.config import Config
from agentharness.providers.base import (
    Message,
    ProviderResponse,
    TextBlock,
    ToolCall,
    Usage,
)
from agentharness.skills.loader import load_skills
from agentharness.state import StateManager
from agentharness.tools.registry import build_default_registry


class FakeProvider:
    """Replays a scripted list of ProviderResponses, one per chat() call."""

    name = "fake"

    def __init__(self, script: list[ProviderResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    def chat(self, *, system, messages, tools, max_tokens):
        self.calls.append(
            {"system": system, "messages": list(messages), "max_tokens": max_tokens}
        )
        if not self._script:
            raise AssertionError("FakeProvider ran out of scripted responses")
        return self._script.pop(0)


def _text_response(text, usage=None):
    return ProviderResponse(
        message=Message(role="assistant", blocks=[TextBlock(text)]),
        stop_reason="end_turn",
        usage=usage or Usage(input_tokens=3, output_tokens=2),
    )


def _tool_response(call_id, name, args, usage=None):
    return ProviderResponse(
        message=Message(role="assistant", blocks=[ToolCall(id=call_id, name=name, arguments=args)]),
        stop_reason="tool_use",
        usage=usage or Usage(input_tokens=5, output_tokens=4),
    )


def _manager(provider: FakeProvider) -> StateManager:
    return StateManager(
        lambda name, model: provider,
        default_provider="fake",
        default_model="fake-1",
        default_system="SYS",
    )


def _drive(provider, state, registry, max_iterations=10):
    return list(
        agent.run_turn(
            provider=provider,
            state=state,
            registry=registry,
            system="SYS",
            max_tokens=100,
            max_iterations=max_iterations,
        )
    )


def test_plain_text_turn(tmp_path):
    provider = FakeProvider([_text_response("hello there")])
    mgr = _manager(provider)
    registry = build_default_registry(load_skills(tmp_path))
    state = mgr.active
    state.add(Message(role="user", blocks=[TextBlock("hi")]))

    events = _drive(provider, state, registry)
    texts = [e.text for e in events if isinstance(e, agent.TextEvent)]
    assert texts == ["hello there"]
    assert any(isinstance(e, agent.DoneEvent) for e in events)
    assert state.usage.total_tokens == 5  # 3 + 2


def test_tool_call_cycle(tmp_path):
    # First response asks for greet; second (after tool result) ends the turn.
    provider = FakeProvider(
        [
            _tool_response("c1", "greet", {"name": "Ada", "style": "formal"}),
            _text_response("done greeting"),
        ]
    )
    mgr = _manager(provider)
    registry = build_default_registry(load_skills(tmp_path))
    state = mgr.active
    state.add(Message(role="user", blocks=[TextBlock("greet Ada")]))

    events = _drive(provider, state, registry)

    assert any(isinstance(e, agent.ToolCallEvent) for e in events)
    tool_results = [e for e in events if isinstance(e, agent.ToolResultEvent)]
    assert len(tool_results) == 1
    assert not tool_results[0].result.is_error
    assert "Ada" in tool_results[0].result.content

    # Second chat() saw the tool result appended as a role="tool" message.
    second_call_messages = provider.calls[1]["messages"]
    assert second_call_messages[-1].role == "tool"
    # usage accumulated across both model calls
    assert state.usage.total_tokens == (5 + 4) + (3 + 2)


def test_max_iterations_guard(tmp_path):
    # Model keeps calling a tool forever; loop must abort.
    provider = FakeProvider([_tool_response("c", "greet", {"name": "X"}) for _ in range(10)])
    mgr = _manager(provider)
    registry = build_default_registry(load_skills(tmp_path))
    state = mgr.active
    state.add(Message(role="user", blocks=[TextBlock("go")]))

    with pytest.raises(agent.MaxIterationsExceeded):
        _drive(provider, state, registry, max_iterations=3)


# ---- state manager ----------------------------------------------------------


def test_states_are_isolated():
    provider = FakeProvider([])
    mgr = _manager(provider)
    mgr.new("research")
    mgr.active.add(Message(role="user", blocks=[TextBlock("my name is Martin")]))
    assert len(mgr.active.messages) == 1

    mgr.switch("default")
    assert len(mgr.active.messages) == 0  # default never saw the name


def test_default_state_exists():
    mgr = _manager(FakeProvider([]))
    assert mgr.active.name == "default"
    assert mgr.names() == ["default"]


def test_new_duplicate_rejected():
    mgr = _manager(FakeProvider([]))
    with pytest.raises(ValueError, match="already exists"):
        mgr.new("default")


def test_new_with_overrides():
    mgr = _manager(FakeProvider([]))
    st = mgr.new("alt", provider="openai", model="gpt-4o")
    assert st.provider_name == "openai"
    assert st.model == "gpt-4o"


def test_cannot_delete_last_state():
    mgr = _manager(FakeProvider([]))
    with pytest.raises(ValueError, match="last remaining"):
        mgr.delete("default")


def test_delete_switches_active():
    mgr = _manager(FakeProvider([]))
    mgr.new("other")  # active is now "other"
    mgr.delete("other")
    assert mgr.active.name == "default"


def test_reset_clears_history_and_usage():
    mgr = _manager(FakeProvider([]))
    st = mgr.active
    st.add(Message(role="user", blocks=[TextBlock("x")]))
    st.usage = Usage(input_tokens=10, output_tokens=5)
    st.reset()
    assert st.messages == []
    assert st.usage.total_tokens == 0


# ---- REPL command handling --------------------------------------------------


def _harness(tmp_path, monkeypatch):
    from agentharness import repl

    provider = FakeProvider([])
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: provider)
    cfg = Config(skills_dir=str(tmp_path))
    return repl.Harness(cfg), repl


def test_repl_new_and_switch(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    assert repl._handle_command(h, "/new research")
    assert h.states.active.name == "research"
    assert repl._handle_command(h, "/switch default")
    assert h.states.active.name == "default"


def test_repl_quit_returns_false(tmp_path, monkeypatch):
    h, repl = _harness(tmp_path, monkeypatch)
    assert repl._handle_command(h, "/quit") is False


def test_repl_reload(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    # add a skill on disk, then reload
    d = tmp_path / "demo"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\nbody\n", encoding="utf-8"
    )
    assert not h.skillset.skills
    repl._handle_command(h, "/reload")
    assert h.skillset.by_name("demo") is not None
    assert "read_skill" in h.registry


def test_repl_new_with_flags(tmp_path, monkeypatch):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/new alt --provider openai --model gpt-4o")
    st = h.states.active
    assert st.name == "alt"
    assert st.provider_name == "openai"
    assert st.model == "gpt-4o"


def test_repl_models_lists_and_marks_current(tmp_path, monkeypatch, capsys):
    from agentharness import repl

    class ListingProvider(FakeProvider):
        def list_models(self):
            return ["fake-1", "fake-2"]

    prov = ListingProvider([])
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: prov)
    h = repl.Harness(Config(skills_dir=str(tmp_path), model="fake-1"))
    repl._handle_command(h, "/models")
    out = capsys.readouterr().out
    assert "fake-1" in out and "fake-2" in out
    assert "*" in out  # current model marked


def test_repl_models_provider_without_support(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)  # FakeProvider has no list_models
    repl._handle_command(h, "/models")
    assert "does not support" in capsys.readouterr().out


def test_repl_models_surfaces_errors(tmp_path, monkeypatch, capsys):
    from agentharness import repl

    class FailingProvider(FakeProvider):
        def list_models(self):
            raise RuntimeError("boom")

    prov = FailingProvider([])
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: prov)
    h = repl.Harness(Config(skills_dir=str(tmp_path)))
    repl._handle_command(h, "/models")
    assert "boom" in capsys.readouterr().out
