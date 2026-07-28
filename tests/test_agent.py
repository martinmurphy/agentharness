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
from agentharness.workspace import Workspace


def _ws(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return Workspace(root=root, writable=True)


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
    registry = build_default_registry(load_skills(tmp_path), _ws(tmp_path))
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
    registry = build_default_registry(load_skills(tmp_path), _ws(tmp_path))
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
    registry = build_default_registry(load_skills(tmp_path), _ws(tmp_path))
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
    cfg = Config(skills_dir=str(tmp_path), workspace_dir=str(_ws(tmp_path).root))
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


def test_repl_providers_lists_and_marks_active(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    h, repl = _harness(tmp_path, monkeypatch)  # default provider is anthropic
    repl._handle_command(h, "/providers")
    out = capsys.readouterr().out
    for name in ("anthropic", "openai", "gemini"):
        assert name in out
    assert "(active)" in out          # anthropic is the default/active provider
    assert "no key" in out            # openai has no key set


def test_repl_help_lists_providers(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/help")
    assert "/providers" in capsys.readouterr().out


def test_repl_registers_list_models_tool(tmp_path, monkeypatch):
    from agentharness import repl
    from agentharness.providers.base import ToolCall as TC

    class ListingProvider(FakeProvider):
        def list_models(self):
            return ["fake-1", "fake-2"]

    prov = ListingProvider([])
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: prov)
    h = repl.Harness(Config(skills_dir=str(tmp_path)))

    assert "list_models" in h.registry  # registered by the harness, not build_default_registry
    # Dispatching it reaches the active state's provider (built lazily).
    result = h.registry.dispatch(TC(id="c", name="list_models", arguments={}))
    assert not result.is_error
    assert "fake-1" in result.content and "fake-2" in result.content


def test_list_models_tool_unsupported_provider(tmp_path, monkeypatch):
    from agentharness import repl
    from agentharness.providers.base import ToolCall as TC

    prov = FakeProvider([])  # no list_models method
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: prov)
    h = repl.Harness(Config(skills_dir=str(tmp_path)))
    result = h.registry.dispatch(TC(id="c", name="list_models", arguments={}))
    assert result.is_error
    assert "cannot list models" in result.content


def test_list_models_named_provider_builds_via_factory(tmp_path, monkeypatch):
    from agentharness import repl
    from agentharness.providers.base import ToolCall as TC

    class ListingProvider(FakeProvider):
        def list_models(self):
            return ["built-1"]

    built = {}

    def fake_build(name, model, config):
        built["name"] = name
        return ListingProvider([])

    monkeypatch.setattr(repl, "build_provider", fake_build)
    h = repl.Harness(Config(skills_dir=str(tmp_path)))  # active = anthropic
    result = h.registry.dispatch(TC(id="c", name="list_models", arguments={"provider": "gemini"}))
    assert not result.is_error
    assert "built-1" in result.content
    assert built["name"] == "gemini"  # a fresh provider was built for the non-active name


def test_list_models_all_via_harness(tmp_path, monkeypatch):
    from agentharness import repl
    from agentharness.providers.base import ToolCall as TC

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    class ListingProvider(FakeProvider):
        def list_models(self):
            return ["m"]

    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: ListingProvider([]))
    h = repl.Harness(Config(skills_dir=str(tmp_path)))
    result = h.registry.dispatch(TC(id="c", name="list_models", arguments={"provider": "all"}))
    assert not result.is_error
    for name in ("anthropic", "openai", "gemini"):
        assert f"{name}:" in result.content


# ---- subagent delegation ----------------------------------------------------


def _spawn_harness(tmp_path, monkeypatch, script, **cfg):
    """Harness whose build_provider yields a scripted FakeProvider.

    Only the subagent's provider is built when calling _spawn_subagent directly,
    so the single script drives the subagent's turn loop.
    """
    from agentharness import repl

    prov = FakeProvider(script)
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: prov)
    base = {"skills_dir": str(tmp_path), "workspace_dir": str(_ws(tmp_path).root)}
    return repl.Harness(Config(**base, **cfg)), prov


def test_spawn_subagent_returns_answer(tmp_path, monkeypatch, capsys):
    h, _ = _spawn_harness(tmp_path, monkeypatch, [_text_response("42")])
    answer = h._spawn_subagent("what is 6*7?", None, None)
    assert answer == "42"
    out = capsys.readouterr().out
    assert "created" in out and "task:" in out and "returning result" in out
    assert h.states.active.name == "default"  # caller's active state restored
    assert any(n.startswith("subagent-") for n in h.states.names())  # state persisted


def test_spawn_subagent_defaults_to_caller_provider_and_model(tmp_path, monkeypatch):
    h, _ = _spawn_harness(
        tmp_path, monkeypatch, [_text_response("ok")],
        provider="anthropic", model="claude-opus-4-8",
    )
    h._spawn_subagent("task", None, None)
    sub = next(s for s in h.states if s.name.startswith("subagent-"))
    assert sub.provider_name == "anthropic"
    assert sub.model == "claude-opus-4-8"


def test_spawn_subagent_explicit_provider_and_model(tmp_path, monkeypatch):
    h, _ = _spawn_harness(tmp_path, monkeypatch, [_text_response("ok")])
    h._spawn_subagent("task", "gemini", "gemini-3.5-flash")
    sub = next(s for s in h.states if s.name.startswith("subagent-"))
    assert sub.provider_name == "gemini"
    assert sub.model == "gemini-3.5-flash"


def test_spawn_subagent_runs_tool_loop_and_logs(tmp_path, monkeypatch, capsys):
    h, _ = _spawn_harness(
        tmp_path, monkeypatch,
        [
            _tool_response("c1", "greet", {"name": "Ada", "style": "formal"}),
            _text_response("Greeted Ada."),
        ],
    )
    answer = h._spawn_subagent("greet Ada formally", None, None)
    assert answer == "Greeted Ada."
    out = capsys.readouterr().out
    assert "greet(" in out and "result:" in out  # subagent tool activity is logged


def test_subagent_registry_excludes_spawn_tool(tmp_path, monkeypatch):
    h, _ = _harness(tmp_path, monkeypatch)
    assert "spawn_subagent" in h.registry            # caller can delegate
    assert "spawn_subagent" not in h._subagent_registry  # but subagents cannot (no recursion)
    # subagents still have the ordinary tools/skills tools
    assert "greet" in h._subagent_registry
    assert "read_skill" in h._subagent_registry
    assert "list_models" in h._subagent_registry


def test_spawn_subagent_max_iterations_restores_active(tmp_path, monkeypatch):
    script = [_tool_response("c", "greet", {"name": "X"}) for _ in range(20)]
    h, _ = _spawn_harness(tmp_path, monkeypatch, script, max_tool_iterations=3)
    answer = h._spawn_subagent("loop forever", None, None)
    assert "did not converge" in answer
    assert h.states.active.name == "default"  # active restored even on failure


# ---- usage ------------------------------------------------------------------


def test_repl_usage_reports_active_state_only(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    h.states.active.usage = Usage(input_tokens=10, output_tokens=3)
    repl._handle_command(h, "/new other")
    h.states.active.usage = Usage(input_tokens=100, output_tokens=7)
    repl._handle_command(h, "/usage")
    out = capsys.readouterr().out
    assert "input=100 output=7 total=107" in out
    assert "default" not in out


def test_repl_usage_all_totals_every_state(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    h.states.active.usage = Usage(input_tokens=10, output_tokens=3)
    repl._handle_command(h, "/new other")
    h.states.active.usage = Usage(input_tokens=1000, output_tokens=7)
    capsys.readouterr()  # drop the /new confirmation
    repl._handle_command(h, "/usage --all")
    out = capsys.readouterr().out
    assert "default" in out and "other" in out
    assert "1,000" in out          # thousands separators
    assert "* other" in out        # active state marked
    assert "total" in out
    assert "1,010" in out and "1,020" in out  # summed input and grand total


def test_repl_usage_rejects_unknown_argument(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/usage everything")
    assert "usage: /usage [--all]" in capsys.readouterr().out


# ---- workspace ------------------------------------------------------------


def test_repl_workspace_command(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    (h.workspace.root / "notes.md").write_text("hello", encoding="utf-8")
    (h.workspace.root / "reports").mkdir()
    repl._handle_command(h, "/workspace")
    out = capsys.readouterr().out
    assert str(h.workspace.root) in out
    assert "read-write" in out
    assert "notes.md" in out
    assert "reports/" in out


def test_repl_workspace_command_reports_missing_root(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    h.workspace.root.rmdir()
    repl._handle_command(h, "/workspace")
    assert "does not exist" in capsys.readouterr().out


def test_repl_help_lists_workspace(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/help")
    assert "/workspace" in capsys.readouterr().out


def test_harness_registers_fs_tools(tmp_path, monkeypatch):
    h, _ = _harness(tmp_path, monkeypatch)
    for name in ("list_dir", "read_file", "write_file", "make_dir"):
        assert name in h.registry
        assert name in h._subagent_registry  # subagents share the workspace


def test_read_only_workspace_hides_write_tools_in_harness(tmp_path, monkeypatch):
    from agentharness import repl

    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: FakeProvider([]))
    cfg = Config(
        skills_dir=str(tmp_path),
        workspace_dir=str(_ws(tmp_path).root),
        workspace_writable=False,
    )
    h = repl.Harness(cfg)
    assert "read_file" in h.registry
    assert "write_file" not in h.registry
    assert "make_dir" not in h.registry


def test_effective_system_mentions_the_workspace(tmp_path, monkeypatch):
    h, _ = _harness(tmp_path, monkeypatch)
    system = h.effective_system()
    assert str(h.workspace.root) in system
    assert "write_file" in system


def test_effective_system_omits_missing_workspace(tmp_path, monkeypatch):
    h, _ = _harness(tmp_path, monkeypatch)
    h.workspace.root.rmdir()
    assert "workspace directory is available" not in h.effective_system()


# ---- provider aliases -------------------------------------------------------


def _alias_harness(tmp_path, monkeypatch):
    from agentharness import repl

    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: FakeProvider([]))
    cfg = Config(
        skills_dir=str(tmp_path),
        workspace_dir=str(_ws(tmp_path).root),
        providers={
            "lmstudio": {"type": "openai", "base_url": "http://localhost:1234/v1"},
            "together": {"type": "openai", "api_key_env": "TOGETHER_API_KEY"},
        },
    )
    return repl.Harness(cfg), repl


def test_repl_providers_lists_aliases(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    h, repl = _alias_harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/providers")
    out = capsys.readouterr().out
    assert "lmstudio" in out and "no key needed" in out
    assert "together" in out and "TOGETHER_API_KEY" in out


def test_spawn_subagent_enum_includes_aliases(tmp_path, monkeypatch):
    h, _ = _alias_harness(tmp_path, monkeypatch)
    enum = h.registry.get("spawn_subagent").input_schema["properties"]["provider"]["enum"]
    assert "lmstudio" in enum and "anthropic" in enum


def test_state_can_target_an_alias(tmp_path, monkeypatch):
    h, repl = _alias_harness(tmp_path, monkeypatch)
    assert repl._handle_command(h, "/new local --provider lmstudio --model qwen3.5-9b-mlx")
    assert h.states.active.provider_name == "lmstudio"
    assert h.states.active.model == "qwen3.5-9b-mlx"


# ---- per-turn usage ---------------------------------------------------------


def test_turn_usage_line_after_a_single_call(tmp_path, monkeypatch, capsys):
    h, _ = _spawn_harness(tmp_path, monkeypatch, [_text_response("hi")])
    h.run_prompt("hello")
    out = capsys.readouterr().out
    assert "[usage] turn 3 in / 2 out = 5 (1 call)" in out
    assert "session 5" in out


def test_turn_usage_sums_tool_round_trips(tmp_path, monkeypatch, capsys):
    """One turn, two model calls: the line reports the turn, not the last call."""
    script = [_tool_response("c", "greet", {"name": "Ada"}), _text_response("done")]
    h, _ = _spawn_harness(tmp_path, monkeypatch, script)
    h.run_prompt("greet Ada")
    out = capsys.readouterr().out
    assert "[usage] turn 8 in / 6 out = 14 (2 calls)" in out  # (5+3) in, (4+2) out


def test_session_total_accumulates_across_turns(tmp_path, monkeypatch, capsys):
    h, _ = _spawn_harness(
        tmp_path, monkeypatch, [_text_response("one"), _text_response("two")]
    )
    h.run_prompt("first")
    h.run_prompt("second")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "[usage]" in ln]
    assert len(lines) == 2
    assert "session 5" in lines[0]
    assert "session 10" in lines[1]  # turn stays 5, session doubles


def test_turn_usage_suppressed_by_config(tmp_path, monkeypatch, capsys):
    h, _ = _spawn_harness(
        tmp_path, monkeypatch, [_text_response("hi")], show_usage=False
    )
    h.run_prompt("hello")
    assert "[usage]" not in capsys.readouterr().out


def test_turn_usage_skipped_when_provider_reports_nothing(tmp_path, monkeypatch, capsys):
    """Some OpenAI-compatible servers omit usage entirely; don't print zeros."""
    h, _ = _spawn_harness(
        tmp_path, monkeypatch, [_text_response("hi", usage=Usage(0, 0))]
    )
    h.run_prompt("hello")
    assert "[usage]" not in capsys.readouterr().out


def test_turn_usage_reported_even_when_the_turn_fails(tmp_path, monkeypatch, capsys):
    """A turn that aborts part-way still spent what it spent."""
    # One tool call, then the script runs out -> FakeProvider raises.
    h, _ = _spawn_harness(tmp_path, monkeypatch, [_tool_response("c", "greet", {"name": "A"})])
    h.run_prompt("go")
    out = capsys.readouterr().out
    assert "[error]" in out
    assert "[usage] turn 5 in / 4 out = 9 (1 call)" in out


def test_turn_usage_reported_when_max_iterations_hit(tmp_path, monkeypatch, capsys):
    script = [_tool_response("c", "greet", {"name": "X"}) for _ in range(10)]
    h, _ = _spawn_harness(tmp_path, monkeypatch, script, max_tool_iterations=2)
    h.run_prompt("loop")
    out = capsys.readouterr().out
    assert "[stopped]" in out
    assert "(2 calls)" in out
