"""The interactive REPL and its slash commands.

Anything not starting with ``/`` is a prompt to the active state. Commands
manage states, inspect skills/tools, and reload the skills directory.
"""

from __future__ import annotations

import argparse
import shlex
import sys

from agentharness import agent
from agentharness.config import Config, load_config
from agentharness.providers.base import Message, Provider, TextBlock
from agentharness.providers.factory import build_provider, provider_status
from agentharness.skills.loader import SkillSet, load_skills
from agentharness.state import StateManager
from agentharness.tools.model_tools import list_models_tool
from agentharness.tools.registry import ToolRegistry, build_default_registry

# readline is imported for its side effect: line editing + history on input().
try:  # pragma: no cover - platform dependent
    import readline  # noqa: F401
except ImportError:  # pragma: no cover
    pass


class _Ansi:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def cyan(self, t: str) -> str:
        return self._wrap("36", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)


def _truncate(text: str, limit: int = 300) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Harness:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.skillset: SkillSet = load_skills(config.skills_dir)
        self.ansi = _Ansi(sys.stdout.isatty())
        self.states = StateManager(
            self._build_provider,
            default_provider=config.provider,
            default_model=config.model,
            default_system=config.system_prompt,
        )
        self.registry: ToolRegistry = self._build_registry()

    def _build_provider(self, provider_name: str, model: str) -> Provider:
        return build_provider(provider_name, model, self.config)

    def _build_registry(self) -> ToolRegistry:
        """Default (state-free) tools plus the state-dependent list_models tool."""
        registry = build_default_registry(self.skillset)
        registry.register(list_models_tool(self._active_model_ids))
        return registry

    def _active_model_ids(self) -> list[str]:
        """Model IDs on the active state's provider (for the list_models tool)."""
        provider = self.states.active.provider
        lister = getattr(provider, "list_models", None)
        if lister is None:
            raise ValueError(
                f"provider {self.states.active.provider_name!r} cannot list models"
            )
        return lister()

    def effective_system(self) -> str:
        catalog = self.skillset.catalog_prompt()
        base = self.states.active.system
        return f"{base}\n\n{catalog}" if catalog else base

    # ---- turn execution -----------------------------------------------------

    def run_prompt(self, text: str) -> None:
        state = self.states.active
        state.add(Message(role="user", blocks=[TextBlock(text)]))
        try:
            for event in agent.run_turn(
                provider=state.provider,
                state=state,
                registry=self.registry,
                system=self.effective_system(),
                max_tokens=self.config.max_tokens,
                max_iterations=self.config.max_tool_iterations,
            ):
                self._render(event)
        except agent.MaxIterationsExceeded as exc:
            print(self.ansi.red(f"[stopped] {exc}"))
        except Exception as exc:  # noqa: BLE001 - surface provider/network errors, keep REPL alive
            print(self.ansi.red(f"[error] {type(exc).__name__}: {exc}"))

    def _render(self, event: agent.Event) -> None:
        a = self.ansi
        if isinstance(event, agent.ThinkingEvent):
            if self.config.show_thinking:
                print(a.dim(f"[thinking] {event.text}"))
        elif isinstance(event, agent.TextEvent):
            print(event.text)
        elif isinstance(event, agent.ToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.call.arguments.items())
            print(a.cyan(f"→ {event.call.name}({args})"))
        elif isinstance(event, agent.ToolResultEvent):
            r = event.result
            label = a.red("← error") if r.is_error else a.green("← result")
            print(f"{label}: {_truncate(r.content)}")
        elif isinstance(event, agent.DoneEvent):
            if event.stop_reason == "refusal":
                print(a.red("[refusal] the model declined this request"))
            elif event.stop_reason == "max_tokens":
                print(a.red("[truncated] hit max_tokens"))
        # UsageEvent is accumulated on the state; shown via /usage.

    # ---- commands -----------------------------------------------------------

    def reload_skills(self) -> None:
        self.skillset = load_skills(self.config.skills_dir)
        self.registry = self._build_registry()


HELP = """\
Commands:
  /help                                    this message
  /states                                  list conversation states
  /new <name> [--provider P] [--model M]   create and switch to a new state
  /switch <name>                           switch to an existing state
  /delete <name>                           delete a state
  /reset                                   clear the active state's history
  /skills                                  list loaded skills (and load failures)
  /skill <name>                            print a skill's SKILL.md body
  /tools                                   list registered tools
  /providers                               list supported providers and whether keys are set
  /models                                  list models the active provider can reach
  /reload                                  re-scan the skills directory
  /usage                                   token usage for the active state
  /quit                                    exit
Anything else is sent as a prompt to the active state."""

_NEW_PARSER = argparse.ArgumentParser(prog="/new", add_help=False)
_NEW_PARSER.add_argument("name")
_NEW_PARSER.add_argument("--provider")
_NEW_PARSER.add_argument("--model")


def _list_providers(h: Harness) -> None:
    """Print supported providers, their key env var, and whether it is set.

    Marks the active state's provider. Reads only the environment (no key is
    handled, no network call), so it works without any credentials configured.
    """
    a = h.ansi
    active = h.states.active.provider_name
    for p in provider_status():
        avail = a.green("key set") if p.available else a.red("no key")
        mark = a.green(" *") if p.name == active else "  "
        suffix = "  (active)" if p.name == active else ""
        print(f"{mark} {p.name:10} {p.env_var:20} {avail}{suffix}")


def _list_models(h: Harness) -> None:
    """Print the models the active state's provider+credentials can reach.

    Builds the provider (lazily) and makes a network call, so it needs a valid
    key. Guarded against providers that don't implement listing and against
    SDK/network errors, so it never crashes the REPL.
    """
    a = h.ansi
    state = h.states.active
    lister = getattr(state.provider, "list_models", None)
    if lister is None:
        print(a.red(f"provider {state.provider_name!r} does not support /models"))
        return
    try:
        models = lister()
    except NotImplementedError:
        print(a.red(f"provider {state.provider_name!r} does not support /models"))
        return
    except Exception as exc:  # noqa: BLE001 - surface auth/network errors, keep REPL alive
        print(a.red(f"[error] {type(exc).__name__}: {exc}"))
        return
    if not models:
        print("  (no models returned)")
        return
    print(a.dim(f"{len(models)} model(s) reachable by {state.provider_name}:"))
    for mid in models:
        mark = a.green(" *") if mid == state.model else "  "
        print(f"{mark} {mid}")


def _handle_command(h: Harness, line: str) -> bool:
    """Handle a slash command. Returns False to signal quit."""
    a = h.ansi
    parts = shlex.split(line)
    cmd, rest = parts[0], parts[1:]

    if cmd in ("/quit", "/exit"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/states":
        for st in h.states:
            marker = "*" if st is h.states.active else " "
            print(f" {marker} {st.name}  [{st.provider_name}:{st.model}]  "
                  f"{len(st.messages)} msgs")
    elif cmd == "/new":
        try:
            args = _NEW_PARSER.parse_args(rest)
            st = h.states.new(args.name, provider=args.provider, model=args.model)
            print(a.green(f"created and switched to {st.name} [{st.provider_name}:{st.model}]"))
        except SystemExit:
            print(a.red("usage: /new <name> [--provider P] [--model M]"))
        except ValueError as exc:
            print(a.red(str(exc)))
    elif cmd == "/switch":
        if not rest:
            print(a.red("usage: /switch <name>"))
        else:
            try:
                h.states.switch(rest[0])
                print(a.green(f"switched to {rest[0]}"))
            except ValueError as exc:
                print(a.red(str(exc)))
    elif cmd == "/delete":
        if not rest:
            print(a.red("usage: /delete <name>"))
        else:
            try:
                h.states.delete(rest[0])
                print(a.green(f"deleted {rest[0]}"))
            except ValueError as exc:
                print(a.red(str(exc)))
    elif cmd == "/reset":
        h.states.active.reset()
        print(a.green("history cleared"))
    elif cmd == "/skills":
        if h.skillset.skills:
            for s in sorted(h.skillset.skills, key=lambda s: s.name):
                print(f"  {a.bold(s.name)}: {_truncate(s.description, 120)}")
                if s.allowed_tools:
                    print(a.dim(f"    allowed-tools: {s.allowed_tools}"))
        else:
            print("  (no skills loaded)")
        for err in h.skillset.errors:
            print(a.red(f"  ! {err.path.name}: {err.reason}"))
    elif cmd == "/skill":
        if not rest:
            print(a.red("usage: /skill <name>"))
        else:
            skill = h.skillset.by_name(rest[0])
            if skill is None:
                print(a.red(f"no such skill: {rest[0]}"))
            else:
                print(skill.body)
    elif cmd == "/tools":
        for spec in h.registry.specs():
            print(f"  {a.bold(spec.name)}: {_truncate(spec.description, 120)}")
    elif cmd == "/providers":
        _list_providers(h)
    elif cmd == "/models":
        _list_models(h)
    elif cmd == "/reload":
        h.reload_skills()
        n = len(h.skillset.skills)
        print(a.green(f"reloaded: {n} skill(s)"))
        if h.skillset.errors:
            print(a.red(f"  {len(h.skillset.errors)} failed to load"))
    elif cmd == "/usage":
        u = h.states.active.usage
        print(f"  input={u.input_tokens} output={u.output_tokens} total={u.total_tokens}")
    else:
        print(a.red(f"unknown command: {cmd} (try /help)"))
    return True


def run_repl(config: Config | None = None) -> int:
    config = config or load_config()
    h = Harness(config)
    a = h.ansi

    print(a.bold("agentharness"))
    print(a.dim(f"provider={config.provider} model={config.model} "
                f"skills={len(h.skillset.skills)} — /help for commands"))
    if h.skillset.errors:
        print(a.red(f"{len(h.skillset.errors)} skill(s) failed to load — see /skills"))

    while True:
        try:
            line = input(a.cyan(f"[{h.states.active.name}] › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(h, line):
                break
        else:
            h.run_prompt(line)
    return 0
