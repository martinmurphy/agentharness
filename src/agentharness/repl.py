"""The interactive REPL and its slash commands.

Anything not starting with ``/`` is a prompt to the active state. Commands
manage states, inspect skills/tools, and reload the skills directory.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from agentharness import agent
from agentharness.config import Config, load_config
from agentharness.mcp import oauth
from agentharness.mcp.config import parse_servers
from agentharness.mcp.manager import McpManager
from agentharness.providers.base import Message, Provider, TextBlock, Usage
from agentharness.providers.factory import build_provider, known_providers, provider_status
from agentharness.skills.loader import SkillSet, load_skills
from agentharness.state import StateManager
from agentharness.tools.mcp_tools import make_mcp_tools
from agentharness.tools.model_tools import list_models_tool
from agentharness.tools.registry import ToolRegistry, build_default_registry
from agentharness.tools.subagent_tools import spawn_subagent_tool
from agentharness.workspace import Workspace, list_dir

SUBAGENT_SYSTEM = (
    "You are a subagent working on a single task delegated to you by another "
    "agent. Do only what that task asks. Call a tool or load a skill only when "
    "the task cannot be answered without it, and stop calling tools the moment "
    "you can answer — the workspace, the skills, and the other tools are "
    "described to you because they are available, not because this task needs "
    "them, so do not explore them. Then give your final answer clearly and "
    "concisely as your last message. Do not ask questions back — you are running "
    "autonomously and must reach an answer."
)

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


def _root_cause(exc: BaseException) -> BaseException:
    """The innermost exception in a chain — the one that says what went wrong.

    Walks ``__cause__`` then ``__context__``, guarding against the cycles a
    re-raise inside an except block can create.
    """
    seen = {id(exc)}
    current = exc
    while True:
        nxt = current.__cause__ or current.__context__
        if nxt is None or id(nxt) in seen:
            return current
        seen.add(id(nxt))
        current = nxt


class Harness:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.skillset: SkillSet = load_skills(config.skills_dir)
        self.workspace = Workspace(
            root=Path(config.workspace_dir).expanduser(),
            writable=config.workspace_writable,
        )
        self.ansi = _Ansi(sys.stdout.isatty())
        # Connect MCP servers before building the registry: their tools have to
        # be known before the first turn advertises the tool list to the model.
        self.mcp = McpManager(parse_servers(config))
        self.mcp.connect_all()
        self.states = StateManager(
            self._build_provider,
            default_provider=config.provider,
            default_model=config.model,
            default_system=config.system_prompt,
        )
        self._subagent_counter = 0
        # Subagents get the base toolset (no spawn_subagent) so delegation is
        # one level deep and cannot recurse. The main registry adds spawn.
        self._subagent_registry: ToolRegistry = self._build_base_registry()
        self.registry: ToolRegistry = self._build_registry()

    def _build_provider(self, provider_name: str, model: str) -> Provider:
        return build_provider(provider_name, model, self.config)

    def _build_base_registry(self) -> ToolRegistry:
        """Default (state-free) tools plus the state-dependent list_models tool.

        MCP tools go in here rather than in the main registry so subagents get
        them too — they already share the workspace, and a delegated task that
        cannot reach the same servers as its caller would be a trap.
        """
        registry = build_default_registry(self.skillset, self.workspace, self.config)
        for tool in make_mcp_tools(self.mcp):
            registry.register(tool)
        registry.register(
            list_models_tool(
                lambda: self.states.active.provider_name,
                self._list_models_for,
                self.config,
            )
        )
        return registry

    def _build_registry(self) -> ToolRegistry:
        """The main registry: base tools plus the spawn_subagent tool."""
        registry = self._build_base_registry()
        providers = known_providers(self.config)
        registry.register(spawn_subagent_tool(self._spawn_subagent, providers))
        return registry

    def _list_models_for(self, provider_name: str) -> list[str]:
        """Model IDs for a named provider (for the list_models tool).

        Reuses the active provider instance when it matches; otherwise builds a
        throwaway provider via the factory. The model passed to the factory is a
        placeholder — listing hits the provider's models endpoint and never uses
        the configured model. Building/listing needs that provider's API key set.
        """
        active = self.states.active
        if provider_name == active.provider_name:
            provider = active.provider
        else:
            provider = build_provider(provider_name, "(model-list)", self.config)
        lister = getattr(provider, "list_models", None)
        if lister is None:
            raise ValueError(f"provider {provider_name!r} cannot list models")
        return lister()

    def effective_system(self) -> str:
        base = self.states.active.system
        parts = [base]
        catalog = self.skillset.catalog_prompt()
        if catalog:
            parts.append(catalog)
        if self.workspace.root.is_dir():
            verbs = (
                "list_dir, read_file, write_file, and make_dir"
                if self.workspace.writable
                else "list_dir and read_file (it is read-only)"
            )
            parts.append(
                f"A workspace directory is available at {self.workspace.root}. "
                f"Use {verbs}; paths are relative to its root."
            )
        return "\n\n".join(parts)

    # ---- subagent delegation ------------------------------------------------

    def _log(self, message: str) -> None:
        """Emit a subagent-processing log line to the REPL surface."""
        print(self.ansi.dim(message))

    def _spawn_subagent(
        self, task: str, provider: str | None, model: str | None
    ) -> str:
        """Run a task on a fresh subagent state and return its final answer.

        The subagent uses its own state and provider (defaulting to the caller's
        provider/model), runs the full tool/skill loop, and logs its processing.
        The caller's active state is restored afterward, so delegation does not
        disturb the caller's conversation.
        """
        caller = self.states.active
        caller_name = caller.name
        provider_name = provider or caller.provider_name
        model_name = model or caller.model

        self._subagent_counter += 1
        name = f"subagent-{self._subagent_counter}"
        while name in self.states.names():
            self._subagent_counter += 1
            name = f"subagent-{self._subagent_counter}"

        self._log(f"[{name}] created (provider={provider_name} model={model_name})")
        self._log(f"[{name}] task: {_truncate(task, 200)}")

        # Activating the subagent makes it the 'active' state for the duration of
        # its run, so tools like list_models resolve against the subagent's
        # provider. The caller's turn keeps its own state reference, so this does
        # not corrupt it; we restore the caller's active state in `finally`.
        sub = self.states.new(
            name, provider=provider_name, model=model_name, system=SUBAGENT_SYSTEM
        )
        try:
            sub.add(Message(role="user", blocks=[TextBlock(task)]))
            for event in agent.run_turn(
                provider=sub.provider,
                state=sub,
                registry=self._subagent_registry,
                system=self.effective_system(),
                max_tokens=self.config.max_tokens,
                max_iterations=self.config.max_tool_iterations,
            ):
                self._log_subagent_event(name, event)
            answer = sub.messages[-1].text() if sub.messages else ""
        except agent.MaxIterationsExceeded as exc:
            self._log(self.ansi.red(f"[{name}] stopped: {exc}"))
            return f"Subagent did not converge on an answer: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface subagent failure to the caller
            self._log(self.ansi.red(f"[{name}] error: {type(exc).__name__}: {exc}"))
            return f"Subagent failed: {type(exc).__name__}: {exc}"
        finally:
            self.states.switch(caller_name)

        answer = answer or "(the subagent produced no text answer)"
        self._log(f"[{name}] returning result to {caller_name}: {_truncate(answer, 200)}")
        return answer

    def _log_subagent_event(self, name: str, event: agent.Event) -> None:
        if isinstance(event, agent.ToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.call.arguments.items())
            self._log(f"[{name}] → {event.call.name}({args})")
        elif isinstance(event, agent.ToolResultEvent):
            r = event.result
            label = "← error" if r.is_error else "← result"
            self._log(f"[{name}] {label}: {_truncate(r.content, 160)}")

    # ---- turn execution -----------------------------------------------------

    def run_prompt(self, text: str) -> None:
        state = self.states.active
        state.add(Message(role="user", blocks=[TextBlock(text)]))
        # Usage arrives per model call, and one turn can make several (each tool
        # round-trip resends the whole history), so sum them for the turn total.
        turn = Usage()
        calls = 0
        try:
            for event in agent.run_turn(
                provider=state.provider,
                state=state,
                registry=self.registry,
                system=self.effective_system(),
                max_tokens=self.config.max_tokens,
                max_iterations=self.config.max_tool_iterations,
            ):
                if isinstance(event, agent.UsageEvent):
                    turn = turn + event.usage
                    calls += 1
                self._render(event)
        except agent.MaxIterationsExceeded as exc:
            print(self.ansi.red(f"[stopped] {exc}"))
        except Exception as exc:  # noqa: BLE001 - surface provider/network errors, keep REPL alive
            print(self.ansi.red(f"[error] {self._describe_error(state, exc)}"))
        finally:
            # In `finally` on purpose: a turn that aborted part-way still spent
            # everything it spent, and that is when the number matters most.
            self._render_turn_usage(turn, calls)

    def _describe_error(self, state, exc: Exception) -> str:
        """Name the endpoint a failed turn was aimed at.

        An SDK's "Connection error." says which *kind* of failure happened and
        nothing about where — and with several providers configured, where is
        most of the diagnosis. The provider name and its base_url are what
        distinguish "the internal endpoint needs the VPN" from "the local
        server isn't running".
        """
        detail = f"{type(exc).__name__}: {exc}"
        # SDKs flatten transport failures into one opaque sentence — the OpenAI
        # SDK's "Connection error." is the same string for a DNS failure, a
        # refused port, and an untrusted certificate. The chain underneath says
        # which, and that is the whole diagnosis.
        root = _root_cause(exc)
        if root is not exc:
            detail += f" — {type(root).__name__}: {_truncate(str(root), 200)}"
        base_url = self.config.provider_options(state.provider_name).get("base_url")
        where = f"{state.provider_name} → {base_url}" if base_url else state.provider_name
        return f"{detail}  [{where}, model {state.model}]"

    def _render_turn_usage(self, turn: Usage, calls: int) -> None:
        """One dim line per turn: what it cost, and the running session total.

        Skipped when the turn reported nothing — some OpenAI-compatible servers
        omit the usage block entirely, and a row of zeros is noise, not data.
        """
        if not self.config.show_usage or turn.total_tokens == 0:
            return
        # state.usage already includes this turn; the agent loop accumulates it.
        session = self.states.active.usage
        call_note = "1 call" if calls == 1 else f"{calls} calls"
        print(
            self.ansi.dim(
                f"[usage] turn {turn.input_tokens:,} in / {turn.output_tokens:,} out"
                f" = {turn.total_tokens:,} ({call_note})"
                f"  ·  session {session.total_tokens:,}"
            )
        )

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
        self._subagent_registry = self._build_base_registry()
        self.registry = self._build_registry()

    def reconnect_mcp(self, name: str) -> None:
        """Re-establish one MCP server and rebuild the registries around it.

        The registries hold wrappers bound to the tools discovered at connect,
        so they have to be rebuilt or they would keep dispatching to a session
        that no longer exists.
        """
        self.mcp.reconnect(name)
        self._subagent_registry = self._build_base_registry()
        self.registry = self._build_registry()

    def shutdown(self) -> None:
        """Release everything the process owns outside itself.

        Today that is the MCP runtime: a loop thread plus any stdio servers we
        spawned, which would otherwise outlive the REPL as orphans.
        """
        self.mcp.shutdown()


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
  /workspace                               show the workspace directory and its contents
  /providers                               list supported providers and whether keys are set
  /models                                  list models the active provider can reach
  /reload                                  re-scan the skills directory
  /mcp [tools S | reconnect S | login S]   MCP servers, their tools, status, and OAuth login
  /usage [--all]                           token usage for the active state (or every state)
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
    for p in provider_status(h.config):
        avail = (
            a.green("no key needed")
            if p.keyless
            else (a.green("key set") if p.available else a.red("no key"))
        )
        mark = a.green(" *") if p.name == active else "  "
        suffix = "  (active)" if p.name == active else ""
        print(f"{mark} {p.name:10} {p.env_var:20} {avail}{suffix}")


def _show_workspace(h: Harness) -> None:
    """Print the workspace root, its mode, and a top-level listing."""
    a = h.ansi
    ws = h.workspace
    mode = a.green("read-write") if ws.writable else a.red("read-only")
    print(f"  {a.bold(str(ws.root))}  {mode}")
    if not ws.root.is_dir():
        print(a.red("  (directory does not exist)"))
        return
    listing = list_dir(ws)
    for line in listing.splitlines():
        print(f"  {line}")


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


def _show_usage(h: Harness, *, all_states: bool) -> None:
    """Print token usage for the active state, or for every state with a total.

    Usage is per-state because each state has its own history and model, so a
    single number across all of them is only meaningful alongside the breakdown.
    """
    a = h.ansi
    if not all_states:
        u = h.states.active.usage
        print(f"  input={u.input_tokens} output={u.output_tokens} total={u.total_tokens}")
        return
    states = list(h.states)
    width = max(len(st.name) for st in states)
    total = Usage()
    for st in states:
        total = total + st.usage
        marker = "*" if st is h.states.active else " "
        u = st.usage
        print(f" {marker} {st.name:<{width}}  {u.input_tokens:>9,} in / {u.output_tokens:>9,} out"
              f" = {u.total_tokens:>9,}")
    print(a.bold(f"   {'total':<{width}}  {total.input_tokens:>9,} in / {total.output_tokens:>9,} out"
                 f" = {total.total_tokens:>9,}"))


def _describe_auth(server) -> str:
    """How a server authenticates, named in terms of where the secret lives."""
    if server.is_stdio:
        return f"env: {', '.join(server.env_pass)}" if server.env_pass else ""
    if server.auth == "oauth":
        return "oauth"
    if server.token_env:
        return f"bearer: {server.token_env}"
    return ""


def _show_mcp(h: Harness) -> None:
    """Print every configured MCP server, its transport, and its status."""
    a = h.ansi
    servers = h.mcp.servers
    if not servers:
        print(a.dim("  (no mcp servers configured — add an mcp_servers: block to the config)"))
        return

    width = max(len(s.name) for s in servers)
    for server in servers:
        if not server.enabled:
            status, detail = a.dim("disabled"), ""
        elif server.name in h.mcp.connected:
            count = len(h.mcp.tools_for(server.name))
            status, detail = a.green("connected"), f"{count} tool(s)"
        else:
            status, detail = a.red("failed"), "see below"
        auth = _describe_auth(server)
        line = f"  {server.name:<{width}}  {server.transport:<5}  {status}"
        if detail:
            line += f"  {detail}"
        if auth:
            line += a.dim(f"  ({auth})")
        print(line)

    for err in h.mcp.errors:
        # A reason can carry a server's stderr tail; indent the continuation so
        # it reads as one block rather than as stray output.
        head, *rest = err.reason.splitlines() or [""]
        print(a.red(f"  ! {err.server}: {head}"))
        for line in rest:
            print(a.red(f"      {line}"))


def _show_mcp_tools(h: Harness, name: str) -> None:
    a = h.ansi
    if h.mcp.server(name) is None:
        print(a.red(f"no such mcp server: {name}"))
        return
    tools = h.mcp.tools_for(name)
    if not tools:
        print(a.dim(f"  ({name} contributed no tools)"))
        return
    for tool in tools:
        print(f"  {a.bold(tool.name)}: {_truncate(tool.description, 120)}")


def _handle_mcp(h: Harness, args: list[str]) -> None:
    a = h.ansi
    if not args:
        _show_mcp(h)
    elif args[0] == "tools" and len(args) == 2:
        _show_mcp_tools(h, args[1])
    elif args[0] in ("reconnect", "login") and len(args) == 2:
        name = args[1]
        server = h.mcp.server(name)
        if server is None:
            print(a.red(f"no such mcp server: {name}"))
            return
        if args[0] == "login":
            if server.auth != "oauth":
                print(a.red(f"mcp server {name!r} does not use auth: oauth"))
                return
            # Forget the grant first, so login means "authorise again" rather
            # than "reuse whatever is cached".
            oauth.forget(server)
        try:
            h.reconnect_mcp(name)
        except ValueError as exc:
            print(a.red(str(exc)))
            return
        if name in h.mcp.connected:
            print(a.green(f"reconnected {name}: {len(h.mcp.tools_for(name))} tool(s)"))
        else:
            reason = next((e.reason for e in h.mcp.errors if e.server == name), "still unavailable")
            print(a.red(reason))
    else:
        print(a.red("usage: /mcp [tools <server> | reconnect <server> | login <server>]"))


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
    elif cmd == "/workspace":
        _show_workspace(h)
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
    elif cmd == "/mcp":
        _handle_mcp(h, rest)
    elif cmd == "/usage":
        if rest and rest[0] not in ("--all", "-a"):
            print(a.red("usage: /usage [--all]"))
        else:
            _show_usage(h, all_states=bool(rest))
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
    if h.mcp.connected:
        print(a.dim(f"mcp: {len(h.mcp.connected)} server(s), {len(h.mcp.tools)} tool(s)"))
    if h.mcp.errors:
        print(a.red(f"{len(h.mcp.errors)} mcp server(s) unavailable — see /mcp"))

    # `finally`, not a normal exit path: an stdio server is our subprocess, and
    # a crash on the way out must not leave it orphaned.
    try:
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
    finally:
        h.shutdown()
    return 0
