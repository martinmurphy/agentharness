# Run tool calls in parallel and turns in the background

## Context

`docs/plan.md` set the original constraint deliberately: *"Multiple independent
states. Sequential only — no concurrent model calls."* Everything above
`mcp/runtime.py` is synchronous — `Provider.chat` blocks, `agent.run_turn` is a
plain generator, tool handlers are `Callable[[dict], str]` — and the REPL runs one
turn at a time to completion.

That constraint now costs real time in two places:

- **Within a turn.** A model that emits four `spawn_subagent` calls in one
  assistant message (four models asked the same question) pays four full
  round-trips end to end, each one a complete nested agent loop. `agent.run_turn`
  dispatches tool calls in a `for` loop (`agent.py:113-118`).
- **At the prompt.** A long turn owns the terminal. There is no way to start
  something slow and keep working.

This adds parallel tool execution inside a turn, and background jobs so the REPL
stays interactive.

Decisions taken up front:

| Decision | Choice |
|---|---|
| Mechanism | **Threads**, extending the existing `ThreadPoolExecutor` idiom — not an async/await rewrite |
| Streaming | **Out of scope** — token-by-token rendering is a separate, provider-shaped change |
| Backgrounding | **Explicit** — trailing `&` on a prompt, never automatic |
| Second prompt on a busy state | **Refused** — one in-flight turn per state |
| Background output | **Buffered** to the job, replayed by `/job <id>` — never printed mid-typing |

Threads rather than `asyncio` because the three provider SDKs are used through
their sync clients and every tool handler is sync, so `await` would mean
rewriting all of it. `McpManager.connect_all` already establishes the idiom — a
bounded `ThreadPoolExecutor` (`_MAX_PARALLEL_CONNECTS = 8`, `mcp/manager.py:49`)
— and this extends it rather than introducing a second concurrency model. The MCP
loop thread stays exactly as it is; `McpRuntime.call_tool` submits via
`run_coroutine_threadsafe`, which is already safe to call from many threads.

Refusing a second prompt on a busy state is the history choice: two turns
appending to one `state.messages` interleave into an unusable transcript.
Concurrency comes from using multiple states, which the harness already has.

## The blocking problem: ambient `states.active`

This has to be fixed before anything can run concurrently, and it is the bulk of
the risk.

`Harness._spawn_subagent` (`repl.py:228-254`) makes the subagent the *active*
state for the duration of its run and restores the caller in `finally`. Three
things read `states.active` as ambient input:

- `Harness.effective_system()` (`repl.py:173`) — `self.states.active.system`
- `list_models_tool`, built with `lambda: self.states.active.provider_name`
  (`repl.py:140`)
- `Harness._render_turn_usage` (`repl.py:330`) — `self.states.active.usage`, the
  "session" figure on the per-turn line

With two subagents in flight, "the active state" is meaningless and the `finally`
restore is a race. The switch/restore dance must go.

**Fix:** a `contextvars.ContextVar` holding the run context — the state whose
turn is currently executing, plus that turn's output sink (§4). The turn driver
sets it; tools that need the calling conversation read it. The sink rides along
because a tool handler is `Callable[[dict], str]` — there is no argument path
through `registry.dispatch` by which `_spawn_subagent` could be handed a
`write`. One propagation rule makes it work under threads: a pool worker starts
with an *empty* context (`ThreadPoolExecutor.submit` does not carry
contextvars), so dispatch snapshots the calling turn's context and runs each
call inside it via `contextvars.copy_context().run(...)`. The same code works
unchanged if the harness ever does go async.

(This does *not* deliver the fix `docs/future-work.md` sketches under
"Unverified: does the tightened subagent prompt curb exploration?" — that entry
wants the *workspace paragraph* dropped from the subagent's system prompt, and
`effective_system(state)` still appends it for every state. That entry stays.)

## Design

### 1. The run context — `context.py` (new, small)

```python
@dataclass(frozen=True)
class RunContext:
    state: ConversationState
    write: Callable[[str], None]   # this turn's output sink (§4)

_current: ContextVar[RunContext] = ContextVar("run_context")

def current_state() -> ConversationState: ...
def current_write() -> Callable[[str], None]: ...

@contextmanager
def running(state, write): ...   # sets and resets the var
```

Changes that follow:

- `Harness.effective_system(state)` takes the state as an argument. Both call
  sites (`run_prompt`, `_spawn_subagent`) already have it in hand.
- `list_models_tool` is built with `lambda: current_state().provider_name`, and
  `Harness._list_models_for` uses `current_state()` instead of `self.states.active`
  for its "reuse the live provider" shortcut.
- `StateManager.new()` gains `activate: bool = True` — today it unconditionally
  makes the new state active (`state.py:99`), which would let a background job's
  subagent flip the REPL's active state mid-typing. `_spawn_subagent` creates
  with `activate=False`, drops `self.states.switch(caller_name)` and its whole
  `finally` block, and wraps the subagent's `run_turn` in
  `context.running(sub, current_write())` so tools the subagent calls resolve
  against the subagent's state — the behaviour activation used to provide.
  Subagent states are still created in the `StateManager` (so `/states` shows
  them and `/usage --all` accounts for them) but are never activated.

### 2. Parallel tool dispatch — `agent.py`

Replace the sequential dispatch loop with a bounded pool when a message carries
more than one call:

- **One call:** dispatch inline, no pool — the common case pays nothing.
- **Several:** submit all to a `ThreadPoolExecutor(max_workers=max_concurrency)`,
  yield `ToolResultEvent`s **as they complete** (so the terminal shows liveness),
  but append `ToolResult`s to the state's tool message **in call order**. History
  stays deterministic; only rendering order reflects completion.
- Each call is submitted as `pool.submit(contextvars.copy_context().run,
  registry.dispatch, call)` — a pool worker's context is empty, so the run
  context must be carried in explicitly or `current_state()` raises
  `LookupError` inside every handler.
- `run_turn` gains a `max_concurrency: int` parameter, passed by both callers.

**The pool is per-turn, not shared, and that is deliberate.** A `spawn_subagent`
handler blocks its worker thread while the subagent runs its own turn, which needs
its own workers. A single process-wide pool would deadlock the moment outer tasks
filled it. Nesting is bounded: subagents get `_subagent_registry`, which has no
`spawn_subagent` (`repl.py:118`), so depth is at most two. One turn's ceiling is
`max_concurrency + max_concurrency²` threads; with `max_jobs` turns in flight
the process worst case is `max_jobs × (max_concurrency + max_concurrency²)` —
288 at the defaults, acceptable for threads that spend their lives blocked on
I/O.

`ToolRegistry.dispatch` is already safe to call concurrently — it reads a dict and
calls a handler, and it never raises (`registry.py:63-79`).

### 3. The job runner — `jobs.py` (new)

```python
@dataclass
class Job:
    id: int
    state_name: str
    prompt: str
    status: Literal["running", "done", "failed"]
    started_at: float
    finished_at: float | None
    lines: list[str]          # rendered output, captured rather than printed
    error: str | None

class JobRunner:
    def submit(self, state_name, prompt, fn) -> Job
    def busy(self, state_name) -> Job | None
    def jobs(self) -> list[Job]
    def get(self, job_id) -> Job | None
    def drain_notices(self) -> list[str]        # completions since the last call
    def shutdown(self, timeout) -> list[Job]    # returns anything still running
```

Owns its own `ThreadPoolExecutor(max_workers=config.max_jobs)`, a lock over the
job table, and a completion-notice queue. One in-flight job per state, enforced by
`busy()`.

### 4. Rendering through a sink — `repl.py`

`Harness._render(event)` and `_log(message)` currently `print` directly
(`repl.py:192`, `repl.py:340`). Both take a `write: Callable[[str], None]`:

- foreground → a `print` wrapped in a shared output lock
- background → `job.lines.append`

`run_prompt` splits into `Harness.run_turn_into(state, text, write)` — the shared
body, wrapped in `context.running(state, write)` — plus thin foreground and
background callers. `_render_turn_usage` takes the state as well as the sink —
it is the third ambient reader above; `_describe_error` already takes the state.

The subagent log sink is read from the run context (`current_write()`) — a tool
handler is `Callable[[dict], str]`, so no argument path reaches
`_spawn_subagent` — which is how a background job's subagent chatter lands in
that job's buffer rather than on the terminal.

### 5. REPL surface

```
[default] › summarise the repo &
[job 1] started on default
[default] › /jobs
  1  running  default   summarise the repo    12s
[default] › what is 2+2
[busy] default is running job 1
       /switch <name> or /new <name> to work elsewhere
[default] ›
[job 1] done — /job 1 for output
```

- **`<prompt> &`** — background it. The `&` is stripped only as a standalone
  trailing token after `.strip()`.
- **`/jobs`** — id, status, state, elapsed, truncated prompt.
- **`/job <id>`** — replay that job's buffered output.
- **Busy rejection** on a foreground prompt, and on `/delete` and `/reset`, for a
  state with a live job.
- **Notices** are drained and printed at the top of the loop *immediately before*
  `input()`. A job finishing while the cursor sits at a prompt cannot print
  without scribbling over readline's buffer, so the notice appears on the next
  keypress-and-Enter. Accepted tradeoff — the alternative (a `readline.redisplay`
  redraw from the completion thread) is platform-dependent and not worth it.
- **`/quit` and EOF** — `run_repl`'s existing `finally` (`repl.py:719`) gains
  `jobs.shutdown(timeout)` before `h.shutdown()`; still-running jobs are named
  as abandoned. Naming them is not enough on its own: executor threads are
  non-daemon and the interpreter joins them at exit, so a wedged tool call
  would hold `/quit` forever. Policy: proceed with MCP teardown anyway (an
  abandoned job's in-flight `call_tool` future fails into that job, which is
  already reported lost), then `os._exit(0)` if worker threads remain alive
  after a short grace — the terminal is never held hostage.
- `HELP` (`repl.py:388`) gains the three new lines and the `&` note.

### 6. Shared-state hardening

- **`StateManager`** (`state.py:58`) — a `threading.Lock` over `new`/`delete`/
  `switch` and the `_states` dict. `Harness._subagent_counter` (`repl.py:219`)
  becomes an `itertools.count` under the same lock; the current read-modify-write
  with a uniqueness retry loop is a race.
- **`ConversationState.provider`** (`state.py:44`) — lazy build under a per-state
  lock, so two concurrent first-uses don't construct two SDK clients.
- **Registry rebuilds.** `/reload` and `/mcp reconnect` rebind `self.registry`
  while jobs may be running. A running turn captured its registry reference at
  start and keeps a consistent view, so this is safe — but the commands should say
  `N job(s) running will finish on the previous tool set` when jobs are live.

### 7. Config — `config.py`

Two fields, both added to `_SCALAR_FIELDS` (`config.py:78`) so the
`AGENTHARNESS_*` overrides work:

```python
max_concurrency: int = 8   # parallel tool calls within one turn
max_jobs: int = 4          # background jobs at once
```

Documented in `config.example.yaml`.

## Files

| File | Change |
|---|---|
| `src/agentharness/context.py` | **new** — the run-context ContextVar |
| `src/agentharness/jobs.py` | **new** — `Job`, `JobRunner` |
| `src/agentharness/agent.py` | parallel dispatch, `max_concurrency` parameter |
| `src/agentharness/repl.py` | output sinks, job commands, busy guard, drop the active-switch |
| `src/agentharness/state.py` | `new(activate=…)`; locking; per-state provider lock |
| `src/agentharness/config.py` | two fields |
| `config.example.yaml`, `README.md`, `docs/plan.md` | document; revise the "sequential only" constraint |

## Tests

Concurrency is proved with barriers, never with timing — a `threading.Barrier(n)`
inside fake tool handlers deadlocks (and fails on its timeout) unless the calls
truly overlap.

`tests/test_agent.py`:

- N tool calls in one message all reach the barrier; results land in the state's
  tool message in **call** order regardless of completion order.
- A single call still works with no pool.
- A handler raising under concurrency still becomes an `is_error` result and does
  not kill its siblings.
- Four `spawn_subagent` calls in one assistant message run concurrently (barrier),
  each on its own state, all four answers returned.
- **Two existing assertions change meaning, not value:**
  `test_spawn_subagent_returns_answer:392` and
  `test_spawn_subagent_max_iterations_restores_active:452-457` assert
  `states.active.name == "default"` because the active state was *restored*;
  with `activate=False` they keep passing because it never changed. Update the
  comments and rename the latter (`…_leaves_active_untouched`).

`tests/test_jobs.py` (**new**): submit/list/get; busy rejection; output captured to
the job buffer and absent from stdout (`capsys`); notices drained once; `shutdown`
reports stragglers.

`tests/test_config.py`: the two new fields, from YAML and from env.

## Verification

1. `make test` and `make lint` clean; the suite run several times, since the
   barrier tests are what would flake if the pool is misused.
2. Manual, against a real provider — the actual proof, since fakes do not exercise
   SDK thread-safety:

```
/new claude --provider anthropic --model claude-opus-5
create subagents using models claude-opus-5 claude-haiku gemini-flash and
cerebras oss, ask each of them what's the capital of ireland, and output
their results
#   expect: four subagent logs interleaved, not four in sequence

summarise the workspace &
/jobs                      # running
summarise it again         # refused: busy
/new scratch
what is 2+2                # works while job 1 runs
/job 1                     # full output replayed
```

3. `/quit` with a job still running — it must report the abandoned job, leave no
   orphaned MCP subprocess (`ps` check), and actually exit rather than hang on
   the non-daemon worker join.
