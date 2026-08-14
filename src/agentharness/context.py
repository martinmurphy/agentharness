"""The run context: which conversation a turn belongs to, and where it prints.

Before this existed, tools that needed the calling conversation read
``StateManager.active``, and ``_spawn_subagent`` made the subagent *the* active
state for the duration of its run so those tools would resolve against it. That
only works while exactly one turn is in flight. With several — parallel tool
calls, or a background job running beside the prompt — "the active state" is
ambient input shared between racing turns, and the restore-in-``finally`` is a
race.

So a turn carries its own context instead: the state whose turn is executing,
and the sink that turn's output goes to. The sink rides along because a tool
handler is ``Callable[[dict], str]`` — there is no argument path through
``ToolRegistry.dispatch`` by which ``_spawn_subagent`` could be handed a
``write``.

A ``ContextVar`` rather than a thread-local, for one reason that matters under
threads and one that matters later:

* ``ThreadPoolExecutor.submit`` starts its worker with an *empty* context, so
  the caller must snapshot and propagate deliberately —
  ``copy_context().run(fn, ...)`` — which is exactly the explicitness wanted
  here. A thread-local would silently be empty instead.
* If the harness is ever rewritten around ``async``, the same code keeps
  working; a thread-local would not.

Reading is deliberately two-tier. ``current()`` returns ``None`` when no turn is
executing — dispatching a tool straight from the REPL, or from a test — and
callers decide what that means. ``current_state()`` and ``current_write()``
raise, for code that has no sensible answer without a turn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentharness.state import ConversationState

# Where a turn's rendered output goes: print for the foreground, a job's buffer
# for a background one.
Write = Callable[[str], None]


@dataclass(frozen=True)
class RunContext:
    """The turn currently executing on this thread (or task)."""

    state: ConversationState
    write: Write


_current: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def current() -> RunContext | None:
    """The running turn's context, or None if no turn is executing here."""
    return _current.get()


def current_state() -> ConversationState:
    ctx = _current.get()
    if ctx is None:
        raise LookupError("no turn is running in this context")
    return ctx.state


def current_write() -> Write:
    ctx = _current.get()
    if ctx is None:
        raise LookupError("no turn is running in this context")
    return ctx.write


@contextmanager
def running(state: ConversationState, write: Write) -> Iterator[RunContext]:
    """Mark this context as executing ``state``'s turn, rendering to ``write``.

    Nests: a subagent's turn sets its own context and the token restores the
    caller's on the way out, so a tool the subagent calls resolves against the
    subagent and a tool the caller calls afterwards does not.
    """
    ctx = RunContext(state=state, write=write)
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
