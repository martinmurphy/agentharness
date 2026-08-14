"""The interactive REPL and its slash commands.

Anything not starting with ``/`` is a prompt to the active state. Commands
manage states, inspect skills/tools, and reload the skills directory.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

from agentharness import agent, context
from agentharness.config import Config, load_config
from agentharness.jobs import Job, JobRunner
from agentharness.mcp import oauth
from agentharness.mcp.config import parse_servers
from agentharness.mcp.manager import McpManager
from agentharness.providers.base import (
    Message,
    Provider,
    TextBlock,
    Usage,
    is_model_not_found,
)
from agentharness.providers.factory import (
    build_provider,
    known_providers,
    provider_status,
    resolve_model,
)
from agentharness.skills.loader import SkillSet, load_skills
from agentharness.state import ConversationState, StateManager
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


# How many model IDs a failure message quotes before summarising the rest. The
# names go into the caller's context, and a provider serving hundreds would cost
# more there than the round trip this saves.
_MAX_LISTED_MODELS = 50


def _subagent_failure(
    provider_name: str, model_name: str, exc: Exception, models: list[str] | None = None
) -> str:
    """What the *calling model* is told when a subagent could not run.

    This string is the only channel back into that model's context, so it is
    where a recoverable failure has to say how to recover. A raw SDK 404 states
    what broke and leaves the next move to be inferred — and the caller that
    just invented a model ID is not the one to trust with that inference. The
    original detail is kept in every case; the hint is added, never substituted.

    When ``models`` is known it is quoted outright rather than the caller being
    sent to fetch it. Telling it to call ``list_models`` costs a whole round
    trip and still depends on it taking the advice; the names were one call away
    at the point of failure, so they ride back with the error and the next
    message can spawn correctly. The fallback wording is for a provider that
    cannot list, has no key, or could not be reached.

    The "listed but not available" caveat belongs in *both* variants: Gemini
    serves a list containing names it then refuses to new accounts, so quoting
    the list without the caveat just invites the same name back.

    ``_describe_error`` is this function's opposite number on the foreground
    path: same job, aimed at the person at the terminal instead.
    """
    detail = f"{type(exc).__name__}: {exc}"
    if not is_model_not_found(exc):
        return f"Subagent failed: {detail}"

    head = (
        f"Subagent failed: model {model_name!r} is not available on provider "
        f"{provider_name!r}. "
    )
    if models:
        shown = ", ".join(models[:_MAX_LISTED_MODELS])
        extra = len(models) - _MAX_LISTED_MODELS
        if extra > 0:
            shown += f", … and {extra} more (list_models(provider={provider_name!r}) for all)"
        next_step = f"It serves: {shown}. Spawn again with one of those. "
    else:
        next_step = (
            f"Call list_models(provider={provider_name!r}) for the names it "
            f"serves, then spawn again with one of those. "
        )
    return (
        f"{head}{next_step}If one of them fails this same way it is listed but "
        f"not available to this account — choose a different one rather than "
        f"retrying it. ({detail})"
    )


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
            default_model_for=self._default_model_for,
            default_system=config.system_prompt,
        )
        # An atomic counter, not a read-modify-write: several turns can spawn
        # subagents at the same moment. Uniqueness is still settled by
        # StateManager.new, which is the only race-free arbiter of a name.
        self._subagent_counter = itertools.count(1)
        self.jobs = JobRunner(max_jobs=config.max_jobs)
        # provider name -> a Future holding its model IDs, for failure messages.
        # Cached because a batch of spawns on bad names fails several times at
        # once, usually on the same provider.
        self._model_lists: dict[str, Future[list[str] | None]] = {}
        # provider name -> model IDs observed to be rejected, so a name that has
        # already failed is never suggested. Guarded by the same lock.
        self._rejected_models: dict[str, set[str]] = {}
        self._model_list_lock = threading.Lock()
        # Serialises whole lines onto the terminal. Foreground output is the
        # only thing that reaches it — jobs render into their own buffers — but
        # a foreground turn's parallel subagents all log through here.
        self._out_lock = threading.Lock()
        # Subagents get the base toolset (no spawn_subagent) so delegation is
        # one level deep and cannot recurse. The main registry adds spawn.
        self._subagent_registry: ToolRegistry = self._build_base_registry()
        self.registry: ToolRegistry = self._build_registry()

    # ---- output sinks -------------------------------------------------------

    def _print(self, text: str) -> None:
        """The foreground sink: one line to the terminal, uninterleaved."""
        with self._out_lock:
            print(text)

    def _turn_state(self) -> ConversationState:
        """The conversation whose turn is executing here.

        Inside a turn the run context is always set, and ``agent`` carries it
        into tool workers explicitly, so this is the calling conversation even
        with several turns in flight. Outside one — a tool dispatched straight
        from the REPL, or from a test — the active state is still exactly
        right, because nothing else is running.
        """
        ctx = context.current()
        return ctx.state if ctx is not None else self.states.active

    def _turn_write(self) -> context.Write:
        """Where the turn executing here renders. The terminal, if none is."""
        ctx = context.current()
        return ctx.write if ctx is not None else self._print

    def _build_provider(self, provider_name: str, model: str) -> Provider:
        return build_provider(provider_name, model, self.config)

    def _default_model_for(self, provider_name: str) -> str:
        """The model a new state on this provider gets when none was named."""
        return resolve_model(provider_name, None, self.config)

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
                lambda: self._turn_state().provider_name,
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

        Reuses the calling conversation's provider instance when it matches;
        otherwise builds a throwaway provider via the factory. The model passed
        to the factory is a placeholder — listing hits the provider's models
        endpoint and never uses the configured model. Building/listing needs
        that provider's API key set.
        """
        caller = self._turn_state()
        if provider_name == caller.provider_name:
            provider = caller.provider
        else:
            provider = build_provider(provider_name, "(model-list)", self.config)
        lister = getattr(provider, "list_models", None)
        if lister is None:
            raise ValueError(f"provider {provider_name!r} cannot list models")
        return lister()

    def _models_for_failure(self, provider_name: str, failed: str) -> list[str] | None:
        """Names worth suggesting after ``failed`` was rejected, or None.

        Never raises and never lists twice for the same provider. Both matter
        because of where it is called from: an except block, on a pool worker,
        with three siblings failing beside it — a batch of spawns on invented
        names is exactly the shape that produces several failures on one
        provider at once, and each one asking the network the same question
        would be the cost this is meant to save.

        A Future rather than a lock held across the call: the first failure to
        arrive owns the fetch, the rest wait on its result, and no thread holds
        a lock while the network is slow. A failed listing is cached as None so
        an unreachable provider is asked once, not once per failure.

        Every name known to have been rejected is filtered out, ``failed``
        included. A listing is not a promise: Gemini serves one containing
        models it then refuses to newer accounts, so the name that just 404'd is
        routinely still in the list it came from, and quoting it back would make
        the message contradict itself. What the session has actually observed
        beats what the provider claims, so a rejection is remembered for the
        rest of the session. None when nothing survives the filter — there is
        no suggestion left to make, and the generic wording is honest.
        """
        with self._model_list_lock:
            rejected = self._rejected_models.setdefault(provider_name, set())
            rejected.add(failed)
            pending = self._model_lists.get(provider_name)
            fetch = pending is None
            if fetch:
                pending = Future()
                self._model_lists[provider_name] = pending
        if fetch:
            models: list[str] | None = None
            try:
                models = self._list_models_for(provider_name)
            except Exception:  # noqa: BLE001 - a missing list is not worth a second failure
                models = None
            finally:
                # In `finally` so a BaseException cannot leave siblings waiting
                # on a result that will never be set.
                pending.set_result(models)
        try:
            models = pending.result(timeout=_MODEL_LIST_TIMEOUT)
        except Exception:  # noqa: BLE001 - the hint degrades; the failure still reports
            return None
        if not models:
            return None
        with self._model_list_lock:
            rejected = set(self._rejected_models.get(provider_name, ()))
        return [m for m in models if m not in rejected] or None

    def effective_system(self, state: ConversationState) -> str:
        """The named state's base prompt plus the skills catalog and workspace.

        Takes the state rather than reading the active one: with turns running
        concurrently there is no single "current" conversation, and both call
        sites have theirs in hand anyway.
        """
        parts = [state.system]
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

    def _log(self, message: str, write: context.Write) -> None:
        """Emit a subagent-processing log line to that turn's output sink."""
        write(self.ansi.dim(message))

    def _spawn_subagent(
        self, task: str, provider: str | None, model: str | None
    ) -> str:
        """Run a task on a fresh subagent state and return its final answer.

        The subagent uses its own state and provider (defaulting to the caller's
        provider/model), runs the full tool/skill loop, and logs its processing.

        Both the caller and the log sink come from the run context, not from the
        harness: this can be one of four spawns running at once, under a turn
        that may be a background job. The subagent state is created but never
        activated — it is visible to ``/states`` and ``/usage --all``, and the
        conversation the user is typing into is left alone.
        """
        caller = self._turn_state()
        write = self._turn_write()
        caller_name = caller.name
        provider_name = provider or caller.provider_name
        # Inheriting the caller's model only makes sense on the caller's own
        # provider: a model name means nothing to a backend that has never heard
        # of it, so a delegation that changes provider resolves afresh.
        if model:
            model_name = model
        elif provider_name != caller.provider_name:
            model_name = self._default_model_for(provider_name)
        else:
            model_name = caller.model

        # Retry rather than pre-check: `names()` then `new()` is a race between
        # concurrent spawns, whereas `new()` itself decides a name atomically.
        while True:
            name = f"subagent-{next(self._subagent_counter)}"
            try:
                sub = self.states.new(
                    name,
                    provider=provider_name,
                    model=model_name,
                    system=SUBAGENT_SYSTEM,
                    activate=False,
                )
                break
            except ValueError:
                continue  # a user-made state already holds that name

        self._log(f"[{name}] created (provider={provider_name} model={model_name})", write)
        self._log(f"[{name}] task: {_truncate(task, 200)}", write)

        try:
            sub.add(Message(role="user", blocks=[TextBlock(task)]))
            # The subagent's own run context, so tools it calls resolve against
            # its state and its logs land in the same sink as its caller's.
            with context.running(sub, write):
                for event in agent.run_turn(
                    provider=sub.provider,
                    state=sub,
                    registry=self._subagent_registry,
                    system=self.effective_system(sub),
                    max_tokens=self.config.max_tokens,
                    max_iterations=self.config.max_tool_iterations,
                    max_concurrency=self.config.max_concurrency,
                ):
                    self._log_subagent_event(name, event, write)
            answer = sub.messages[-1].text() if sub.messages else ""
        except agent.MaxIterationsExceeded as exc:
            self._log(self.ansi.red(f"[{name}] stopped: {exc}"), write)
            return f"Subagent did not converge on an answer: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface subagent failure to the caller
            self._log(self.ansi.red(f"[{name}] error: {type(exc).__name__}: {exc}"), write)
            # Only pay for a listing when the names are the problem — a
            # connection error is not a naming problem, and asking the same
            # unreachable endpoint for its catalogue would just fail again.
            models = (
                self._models_for_failure(provider_name, model_name)
                if is_model_not_found(exc)
                else None
            )
            return _subagent_failure(provider_name, model_name, exc, models)

        answer = answer or "(the subagent produced no text answer)"
        self._log(
            f"[{name}] returning result to {caller_name}: {_truncate(answer, 200)}", write
        )
        return answer

    def _log_subagent_event(
        self, name: str, event: agent.Event, write: context.Write
    ) -> None:
        if isinstance(event, agent.ToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.call.arguments.items())
            self._log(f"[{name}] → {event.call.name}({args})", write)
        elif isinstance(event, agent.ToolResultEvent):
            r = event.result
            label = "← error" if r.is_error else "← result"
            self._log(f"[{name}] {label}: {_truncate(r.content, 160)}", write)

    # ---- turn execution -----------------------------------------------------

    def run_turn_into(
        self, state: ConversationState, text: str, write: context.Write
    ) -> None:
        """Run one turn on ``state``, rendering every line through ``write``.

        The whole body of a turn, foreground or background: the only difference
        between the two is the sink, and which thread it runs on.
        """
        state.add(Message(role="user", blocks=[TextBlock(text)]))
        # Usage arrives per model call, and one turn can make several (each tool
        # round-trip resends the whole history), so sum them for the turn total.
        turn = Usage()
        calls = 0
        try:
            with context.running(state, write):
                for event in agent.run_turn(
                    provider=state.provider,
                    state=state,
                    registry=self.registry,
                    system=self.effective_system(state),
                    max_tokens=self.config.max_tokens,
                    max_iterations=self.config.max_tool_iterations,
                    max_concurrency=self.config.max_concurrency,
                ):
                    if isinstance(event, agent.UsageEvent):
                        turn = turn + event.usage
                        calls += 1
                    self._render(event, write)
        except agent.MaxIterationsExceeded as exc:
            write(self.ansi.red(f"[stopped] {exc}"))
        except Exception as exc:  # noqa: BLE001 - surface provider/network errors, keep REPL alive
            write(self.ansi.red(f"[error] {self._describe_error(state, exc)}"))
        finally:
            # In `finally` on purpose: a turn that aborted part-way still spent
            # everything it spent, and that is when the number matters most.
            self._render_turn_usage(turn, calls, state, write)

    def run_prompt(self, text: str) -> None:
        """Run a turn on the active state, in the foreground, to the terminal."""
        self.run_turn_into(self.states.active, text, self._print)

    def start_job(self, text: str) -> Job:
        """Run a turn on the active state in the background, into a job buffer.

        Raises ValueError if that state already has a job running — one in-flight
        turn per state, because two turns appending to one history interleave.
        """
        state = self.states.active
        return self.jobs.submit(
            state.name,
            text,
            lambda write: self.run_turn_into(state, text, write),
        )

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

    def _render_turn_usage(
        self, turn: Usage, calls: int, state: ConversationState, write: context.Write
    ) -> None:
        """One dim line per turn: what it cost, and the running session total.

        The session figure is the *turn's own* state, passed in rather than read
        from ``states.active`` — a background job's line has to report the
        conversation it ran on, not whichever one the prompt is pointing at by
        the time it finishes.

        Skipped when the turn reported nothing — some OpenAI-compatible servers
        omit the usage block entirely, and a row of zeros is noise, not data.
        """
        if not self.config.show_usage or turn.total_tokens == 0:
            return
        # state.usage already includes this turn; the agent loop accumulates it.
        call_note = "1 call" if calls == 1 else f"{calls} calls"
        write(
            self.ansi.dim(
                f"[usage] turn {turn.input_tokens:,} in / {turn.output_tokens:,} out"
                f" = {turn.total_tokens:,} ({call_note})"
                f"  ·  session {state.usage.total_tokens:,}"
            )
        )

    def _render(self, event: agent.Event, write: context.Write) -> None:
        a = self.ansi
        if isinstance(event, agent.ThinkingEvent):
            if self.config.show_thinking:
                write(a.dim(f"[thinking] {event.text}"))
        elif isinstance(event, agent.TextEvent):
            write(event.text)
        elif isinstance(event, agent.ToolCallEvent):
            args = ", ".join(f"{k}={v!r}" for k, v in event.call.arguments.items())
            write(a.cyan(f"→ {event.call.name}({args})"))
        elif isinstance(event, agent.ToolResultEvent):
            r = event.result
            label = a.red("← error") if r.is_error else a.green("← result")
            write(f"{label}: {_truncate(r.content)}")
        elif isinstance(event, agent.DoneEvent):
            if event.stop_reason == "refusal":
                write(a.red("[refusal] the model declined this request"))
            elif event.stop_reason == "max_tokens":
                write(a.red("[truncated] hit max_tokens"))
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

    def running_job_note(self) -> str | None:
        """Why a rebuilt registry may not be what a running job is using.

        A turn captured its registry reference when it started and keeps a
        consistent view of it, so rebinding ``self.registry`` under a live job
        is safe — but silently leaving that job on the previous tool set would
        be a puzzle, so the commands that rebuild say so.
        """
        live = [j for j in self.jobs.jobs() if j.status == "running"]
        if not live:
            return None
        plural = "job" if len(live) == 1 else "jobs"
        return f"{len(live)} {plural} running will finish on the previous tool set"

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
  /jobs                                    list background jobs
  /job <id>                                replay a background job's output
  /quit                                    exit
Anything else is sent as a prompt to the active state. End a prompt with ` &`
to run it in the background; its output is buffered for /job."""

def split_background(line: str) -> tuple[str, bool]:
    """Split a trailing ``&`` off a prompt: ``("summarise it", True)``.

    Only a *standalone* trailing token counts, so "Ada & Grace" and "R&D" are
    prompts, not backgrounded ones. Backgrounding is always explicit — a slow
    turn is not on its own a reason to take the terminal away from someone.
    """
    if line.endswith(" &"):
        return line[:-2].rstrip(), True
    return line, False


# How long /quit waits for a running job before naming it abandoned and going.
_JOB_SHUTDOWN_GRACE = 2.0

# Cap on waiting for another thread's in-flight model listing. Bounded because
# this runs inside a failure path: a hint worth having is not worth hanging a
# turn for, and the failure still reports without it.
_MODEL_LIST_TIMEOUT = 20.0

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
        # A provider that serves one model names it, so `/new x --provider p`
        # is enough — the listing has to show what that would select.
        model = f"  {a.dim(p.default_model)}" if p.default_model else ""
        print(f"{mark} {p.name:10} {p.env_var:20} {avail}{suffix}{model}")


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


def _show_jobs(h: Harness) -> None:
    """Print every job: id, status, state, elapsed, and a truncated prompt."""
    a = h.ansi
    jobs = h.jobs.jobs()
    if not jobs:
        print(a.dim("  (no background jobs — end a prompt with ` &` to start one)"))
        return
    now = time.monotonic()
    colour = {"running": a.cyan, "done": a.green, "failed": a.red}
    width = max(len(j.state_name) for j in jobs)
    for job in jobs:
        status = colour[job.status](f"{job.status:<7}")
        print(
            f"  {job.id:>3}  {status}  {job.state_name:<{width}}  "
            f"{job.elapsed(now):>5.0f}s  {_truncate(job.prompt, 60)}"
        )


def _show_job(h: Harness, arg: str) -> None:
    """Replay one job's buffered output — the whole point of buffering it."""
    a = h.ansi
    try:
        job = h.jobs.get(int(arg))
    except ValueError:
        job = None
    if job is None:
        print(a.red(f"no such job: {arg}"))
        return
    print(a.dim(f"[job {job.id}] {job.status} on {job.state_name}: {_truncate(job.prompt, 80)}"))
    for line in job.output():
        print(line)
    if job.status == "running":
        print(a.dim("[job still running — /job again for the rest]"))
    if job.error:
        print(a.red(f"[job failed] {job.error}"))


def _refuse_if_busy(h: Harness, state_name: str, what: str) -> bool:
    """Print the busy message and return True if that state has a live job."""
    job = h.jobs.busy(state_name)
    if job is None:
        return False
    a = h.ansi
    print(a.red(f"[busy] {state_name} is running job {job.id}"))
    print(a.dim(f"       {what}"))
    return True


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
        note = h.running_job_note()
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
        if note:
            print(a.dim(f"  {note}"))
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
        elif not _refuse_if_busy(h, rest[0], "wait for it, or /jobs to check on it"):
            try:
                h.states.delete(rest[0])
                print(a.green(f"deleted {rest[0]}"))
            except ValueError as exc:
                print(a.red(str(exc)))
    elif cmd == "/reset":
        # Clearing a history out from under a turn that is mid-way through
        # appending to it produces a transcript with no user message.
        if not _refuse_if_busy(h, h.states.active.name, "wait for it, or /jobs to check on it"):
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
        note = h.running_job_note()
        h.reload_skills()
        n = len(h.skillset.skills)
        print(a.green(f"reloaded: {n} skill(s)"))
        if h.skillset.errors:
            print(a.red(f"  {len(h.skillset.errors)} failed to load"))
        if note:
            print(a.dim(f"  {note}"))
    elif cmd == "/mcp":
        _handle_mcp(h, rest)
    elif cmd == "/usage":
        if rest and rest[0] not in ("--all", "-a"):
            print(a.red("usage: /usage [--all]"))
        else:
            _show_usage(h, all_states=bool(rest))
    elif cmd == "/jobs":
        _show_jobs(h)
    elif cmd == "/job":
        if len(rest) != 1:
            print(a.red("usage: /job <id>"))
        else:
            _show_job(h, rest[0])
    else:
        print(a.red(f"unknown command: {cmd} (try /help)"))
    return True


def run_repl(config: Config | None = None) -> int:
    config = config or load_config()
    h = Harness(config)
    a = h.ansi

    print(a.bold("agentharness"))
    # The active state's model, not config.model: a provider may declare its own,
    # and a banner that named the wrong one would be worse than none.
    print(a.dim(f"provider={config.provider} model={h.states.active.model} "
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
            # Drained here, immediately before the prompt, and nowhere else: a
            # job that finished while the cursor sat on the input line cannot
            # print without scribbling over readline's buffer. So the notice
            # waits for the next keypress-and-Enter. The alternative — a
            # redisplay driven from the completion thread — is platform-
            # dependent and buys very little.
            for notice in h.jobs.drain_notices():
                print(a.dim(notice))
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
                _run_or_background(h, line)
    finally:
        _shutdown(h, a)
    return 0


def _run_or_background(h: Harness, line: str) -> None:
    """Send a prompt to the active state, in the foreground or as a job."""
    a = h.ansi
    text, background = split_background(line)
    if not text:
        return
    state_name = h.states.active.name
    if _refuse_if_busy(h, state_name, "/switch <name> or /new <name> to work elsewhere"):
        return
    if not background:
        h.run_prompt(text)
        return
    try:
        job = h.start_job(text)
    except ValueError as exc:  # lost the race against another submitter
        print(a.red(f"[busy] {exc}"))
        return
    print(a.dim(f"[job {job.id}] started on {state_name}"))


def _shutdown(h: Harness, a: _Ansi) -> None:
    """Name any abandoned job, release MCP, and make sure we actually exit.

    Naming the stragglers is not enough on its own: the job workers are
    non-daemon threads and the interpreter joins them at exit, so one wedged
    tool call would hold ``/quit`` open indefinitely. MCP teardown happens
    regardless — an abandoned job's in-flight call fails into that job, which is
    already reported lost — and then, if workers are still alive, the process
    leaves without waiting. A terminal is never held hostage by a background
    turn nobody is waiting for.
    """
    stragglers = h.jobs.shutdown(_JOB_SHUTDOWN_GRACE)
    for job in stragglers:
        print(a.red(f"[job {job.id}] abandoned (still running on {job.state_name})"))
    h.shutdown()
    if stragglers:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
