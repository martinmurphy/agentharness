# agentharness — a containerised AI harness with tools, skills, and multi-state sessions

> This is the original build plan for the harness (phases 1–4). Features added
> after the initial build are summarised under **Post-build increments** at the
> end, each with a pointer to its own design note.

## Context

`agentharness` is a learning/experimentation harness: a REPL that talks to a
model, calls tools, and loads [Agent Skills](https://agentskills.io) on demand,
all running inside a rootless Podman container built on UBI10.

The goal is a small, legible harness whose seams are in the right places — so
that adding a provider, a tool, or a skill later is a new file, not a refactor.
Three requirements drive the architecture:

- **Provider-neutral from day one.** Anthropic now; OpenAI-compatible and Gemini
  later. That forbids leaning on any SDK's built-in agent loop, because those
  loops are provider-shaped. We own the loop and normalise messages.
- **Skills mounted from the host.** Skills are discovered at runtime from a
  bind-mounted directory, not baked into the image — so editing a `SKILL.md` on
  the host and running `/reload` in the REPL is the iteration cycle.
- **Multiple independent states.** Separate conversations, each with its own
  history, system prompt, provider, and model. Sequential only — no concurrent
  model calls.

Decisions taken up front (from the design questions):

| Decision | Choice |
|---|---|
| Skill exposure | Catalog in system prompt + `read_skill` / `read_skill_file` tools (spec's 3-stage progressive disclosure) |
| State persistence | **In-memory only** — states live for the life of the REPL process |
| Skill `scripts/` | **Read as text, never executed** in v1 |
| Providers | `Provider` protocol + **Anthropic and OpenAI-compatible** implementations |

### Verified environment facts

- Podman 5.2.5.
- `registry.access.redhat.com/ubi10/ubi:10.0` and `ubi10/ubi-minimal:10.0` exist.
  **There is no `ubi10/python-*` S2I image** (`podman search --list-tags` → `Repo not found`
  for `python-312`, `python-3.12`, `python-313`). So the base is `ubi10/ubi` + `dnf install`.
- RHEL 10's default `python3` is **3.12**; newer interpreters (3.13, 3.14) ship as separate
  non-modular AppStream packages installable in parallel. Target `python3.14`, probe at build
  time (see Task 14). Confirmed present in `ubi10/ubi:10.0`: `python3.14` (3.14.5) + `python3.14-pip`.
- API keys come from the environment at `podman run` time and are never written to the image
  or config file.

### Agent Skills spec (confirmed against agentskills.io/specification)

A skill is a directory containing `SKILL.md` with YAML frontmatter:

| Field | Required | Constraint |
|---|---|---|
| `name` | yes | 1–64 chars, lowercase alphanumeric + hyphens, no leading/trailing/consecutive hyphens, **must match the parent directory name** |
| `description` | yes | 1–1024 chars, non-empty |
| `license` | no | free text |
| `compatibility` | no | ≤500 chars |
| `metadata` | no | string→string map |
| `allowed-tools` | no | space-separated string (experimental — parsed and surfaced, not enforced in v1) |

Optional sibling dirs: `scripts/`, `references/`, `assets/`. Progressive disclosure is three
stages: metadata at startup (~100 tokens each) → full `SKILL.md` on activation → bundled files
on demand.

---

## Architecture

```
agentharness/
├── Containerfile
├── Makefile
├── pyproject.toml
├── config.example.yaml
├── README.md
├── skills/                              # example skills; bind-mounted to /skills at runtime
│   └── greeting-etiquette/
│       ├── SKILL.md
│       └── references/REGIONAL.md
├── src/agentharness/
│   ├── __main__.py                      # CLI entrypoint
│   ├── config.py                        # Config dataclass; YAML + env overrides
│   ├── repl.py                          # REPL loop + slash commands
│   ├── agent.py                         # provider-neutral agentic loop
│   ├── state.py                         # ConversationState, StateManager
│   ├── skills/{model.py,loader.py}      # Skill dataclass, discovery + validation
│   ├── tools/{registry.py,greet.py,skill_tools.py}
│   └── providers/{base.py,anthropic.py,openai_compatible.py}
└── tests/
```

### The neutral message model (`providers/base.py`)

This is the load-bearing abstraction — everything else depends on it, so build it first.

```python
@dataclass(frozen=True)
class TextBlock:  text: str
@dataclass(frozen=True)
class ToolCall:   id: str; name: str; arguments: dict[str, Any]
@dataclass(frozen=True)
class ToolResult: call_id: str; content: str; is_error: bool = False

Block = TextBlock | ToolCall | ToolResult

@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    blocks: list[Block]
    provider_raw: Any = None      # see below

@dataclass(frozen=True)
class ToolSpec:  name: str; description: str; input_schema: dict   # JSON Schema

@dataclass
class ProviderResponse:
    message: Message
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]
    usage: Usage

class Provider(Protocol):
    name: str
    def chat(self, *, system: str, messages: list[Message],
             tools: list[ToolSpec], max_tokens: int) -> ProviderResponse: ...
```

**`provider_raw` is deliberate, not a leak.** When a provider builds an assistant `Message` it
stashes the original wire-format content alongside the normalised blocks. On replay, a provider
re-emits its own `provider_raw` verbatim and ignores anyone else's. This is what makes
Anthropic's adaptive-thinking blocks work: thinking blocks carry signatures and **must be passed
back unmodified** on the next turn of the same conversation, but they have no neutral
representation. Reconstructing the assistant turn from `blocks` alone would silently corrupt
multi-turn tool use. A state is bound to one provider at creation, so `provider_raw` is only
ever replayed to its author.

**Wire-format mapping** — the two initial implementations differ in ways worth stating explicitly:

| Neutral | Anthropic | OpenAI-compatible |
|---|---|---|
| `TextBlock` | `{"type":"text","text":...}` | `message.content` (string) |
| `ToolCall` | `{"type":"tool_use","id","name","input":{...}}` | `tool_calls[].{id, function.name, function.arguments}` — arguments is a **JSON string**, parse it |
| `ToolResult` | `{"type":"tool_result","tool_use_id",...,"is_error"}` | `{"role":"tool","tool_call_id","content"}` |
| `ToolSpec` | `{name, description, input_schema}` | `{"type":"function","function":{name, description, parameters}}` |
| tool-use stop | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| N tool results | **one** user message containing N `tool_result` blocks | **N separate** `role:"tool"` messages |

That last row is the trap. A neutral `Message(role="tool", blocks=[r1, r2, r3])` collapses to one
Anthropic user message but fans out to three OpenAI messages. Splitting Anthropic's results
across multiple messages trains the model out of parallel tool calls, so the packing must happen
in the provider, not the loop.

### Anthropic provider specifics

Use the official `anthropic` SDK with `client.messages.create` — **not** `client.beta.messages.tool_runner`.
The tool runner is a fine helper, but it owns the loop, and the loop is exactly what has to stay
provider-neutral here.

- Model default `claude-opus-4-8`, `max_tokens: 16000` (safe non-streaming ceiling).
- `thinking={"type": "adaptive"}` + `output_config={"effort": "high"}`.
  `budget_tokens`, `temperature`, `top_p`, and `top_k` all return **400** on this model — do not
  send them.
- Thinking display defaults to `"omitted"` (empty thinking text). When `show_thinking: true` in
  config, send `thinking={"type":"adaptive","display":"summarized"}` so the REPL can render it.
- Handle `stop_reason == "refusal"` — check `stop_reason` **before** reading `content`.

### OpenAI-compatible provider specifics

`openai` SDK, `client.chat.completions.create`, `tool_choice="auto"`. `base_url` is configurable
so the same code path serves OpenAI, Ollama, vLLM, or anything else speaking that dialect. The
system prompt becomes a leading `{"role":"system"}` message rather than a top-level field.

### Agent loop (`agent.py`)

```python
def run_turn(state, provider, registry, *, max_iterations) -> Iterator[Event]:
    for _ in range(max_iterations):
        resp = provider.chat(system=state.system, messages=state.messages,
                             tools=registry.specs(), max_tokens=state.max_tokens)
        state.messages.append(resp.message)
        state.usage += resp.usage
        yield from events_for(resp)                       # text / thinking / usage
        calls = [b for b in resp.message.blocks if isinstance(b, ToolCall)]
        if not calls:
            return
        results = [registry.dispatch(c) for c in calls]   # never raises; errors → is_error
        yield from (ToolResultEvent(r) for r in results)
        state.messages.append(Message(role="tool", blocks=results))
    raise MaxIterationsExceeded(max_iterations)
```

Generator-of-events keeps rendering out of the loop, which makes the loop testable against a
`FakeProvider` with no I/O. `registry.dispatch` catches handler exceptions and returns
`ToolResult(is_error=True, content=str(exc))` so the model can recover rather than the REPL
crashing.

### Skills (`skills/loader.py`)

Scan `skills_dir` one level deep for subdirectories containing `SKILL.md`. Split frontmatter on
`---` fences, parse with `PyYAML` (`yaml.safe_load`), validate, and build a `Skill`.

Name validation is one regex covering all four spec rules — `^[a-z0-9]+(-[a-z0-9]+)*$` — plus a
length check and an equality check against the directory name. A skill that fails validation is
**skipped with a warning, never fatal**; failures are retained in a list so `/skills` can show
them.

`catalog_prompt()` renders the discovery block appended to the system prompt.

**Path safety in `read_skill_file` is required, not optional** — the `path` argument is untrusted
model output. Resolve `(skill.path / rel).resolve()` and require
`.is_relative_to(skill.path.resolve())`; this catches `..` traversal, absolute paths, and symlink
escapes in one check. Cap reads at 256 KB and return an error result on `UnicodeDecodeError`
rather than emitting binary into the context.

### Tools

| Tool | Signature | Returns |
|---|---|---|
| `greet` | `name: str, style: "formal" \| "casual" \| "enthusiastic" = "casual"` | a greeting string |
| `read_skill` | `name: str` | full `SKILL.md` body, frontmatter stripped |
| `read_skill_file` | `skill: str, path: str` | UTF-8 text of a bundled file under the skill dir |

`greet` exists only to prove the loop end-to-end; keep it trivial.

### State (`state.py`, in-memory)

`ConversationState`: `name`, `provider_name`, `model`, `system`, `messages`, `usage`,
`created_at`, plus a lazily-built provider. `StateManager` holds `dict[str, ConversationState]`
plus an active name, with `new / switch / list / delete / reset`. Each state carries its own
provider and model, so `/new research --provider openai --model gpt-4o` gives a second
conversation against a different backend in the same process. Providers are built **lazily** on
first model call so the REPL can start (and run non-model commands) with no API key.

### REPL commands

```
/help          /states        /new <name> [--provider P] [--model M]   /switch <name>
/delete <name> /reset         /skills        /skill <name>   /tools    /reload
/usage         /quit
```

Anything not starting with `/` is a prompt to the active state. Use stdlib `readline` (import for
side effects) to get arrow keys and in-session history on `input()`.

### Config

`config.yaml` resolved from `$AGENTHARNESS_CONFIG` → `/config/config.yaml` → `./config.yaml`,
with `AGENTHARNESS_*` env overrides. Keys: `provider`, `model`, `max_tokens`, `effort`,
`show_thinking`, `max_tool_iterations`, `skills_dir`, `system_prompt`, and a `providers:` block
for per-provider settings such as `base_url`. **API keys come from the environment only.**

### Container

```dockerfile
FROM registry.access.redhat.com/ubi10/ubi:10.0
RUN dnf -y install python3.14 python3.14-pip && dnf clean all && rm -rf /var/cache/dnf
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3.14 -m pip install --no-cache-dir .
RUN mkdir -p /skills /config && chgrp -R 0 /app /skills /config && chmod -R g=u /app /skills /config
USER 1001
ENV AGENTHARNESS_SKILLS_DIR=/skills
ENTRYPOINT ["python3.14", "-m", "agentharness"]
```

`USER 1001` + group 0 with group-writable dirs is the rootless/OpenShift-friendly convention.
Skills are mounted read-only: `-v "$PWD/skills:/skills:ro,Z"`.

---

## Tasks

Ordered so each phase is verifiable on its own. Phases 1–2 need no API key and no network.

**Phase 1 — skills and tools, no model**
1. Scaffold `pyproject.toml`, package layout, `.gitignore`, `git init`.
2. `config.py` — `Config` dataclass, YAML + env resolution, `config.example.yaml`.
3. `skills/model.py` + `skills/loader.py` — discovery, frontmatter parsing, spec validation,
   `catalog_prompt()`, non-fatal failure collection.
4. `tools/registry.py` — `Tool`, `ToolRegistry`, `specs()`, `dispatch()` with exception capture.
5. `tools/greet.py` and `tools/skill_tools.py`, including the path-confinement check.
6. Example skill `skills/greeting-etiquette/` with a `references/REGIONAL.md`.
7. Tests: loader, path traversal/symlink rejection, `greet`, catalog rendering.
   *Verify:* `python -m agentharness --list-skills` prints the catalog with no API key set.

**Phase 2 — providers**
8. `providers/base.py` — neutral types and the `Provider` protocol.
9. `providers/anthropic.py` — mapping both directions, `provider_raw` passthrough, adaptive
   thinking, refusal handling.
10. `providers/openai_compatible.py` — mapping both directions, tool-result fan-out, `base_url`.
11. Tests: round-trip each direction for both providers, multi-tool-result packing vs. fan-out,
    `provider_raw` preserved across a replay. Pure mapping tests — no network.

**Phase 3 — loop, state, REPL**
12. `state.py`, then `agent.py`, then `repl.py` with the command set above.
13. Tests with a `FakeProvider`: tool call → dispatch → result → end; `max_iterations` guard;
    two states keep their histories isolated.

**Phase 4 — container**
14. `Containerfile`. Probe the interpreter first and pin the newest available (3.14; fall back
    to 3.13, then 3.12).
15. `Makefile` (`build`, `run`, `shell`, `test`, `lint`) and `README.md`.

---

## Verification

```bash
python3 -m pytest -q                            # unit tests — no network, no API key
python3 -m agentharness --list-skills           # skills load on the host
podman build -t agentharness .                  # image builds
podman run --rm agentharness --version
podman run --rm -v "$PWD/skills:/skills:ro,Z" agentharness --list-skills
ANTHROPIC_API_KEY=... podman run --rm -it -e ANTHROPIC_API_KEY \
  -v "$PWD/skills:/skills:ro,Z" agentharness    # end-to-end REPL
```

In the REPL, confirm each requirement: tool calling (`greet`), skill activation
(`read_skill` → `read_skill_file`), multi-state isolation, per-state provider, live host mount
via `/reload`, and ephemeral (in-memory) state across restarts.

---

## Notes and risks

- `allowed-tools` is parsed but not enforced (experimental in the spec; nothing to gate in v1).
- A refusal (`stop_reason: "refusal"`) is a normal outcome, not an exception — surface it and end
  the turn cleanly.
- **Deliberately out of scope for v1**, each a clean addition against the seams above: streaming
  responses, Gemini, script execution, persistence, prompt caching, concurrent states.

---

## Post-build increments

Features added after the initial four-phase build, each landed against the seams above with no
change to the agent loop:

- **Gemini provider (AI Studio)** — a third `Provider` adapter (`providers/gemini.py`) using the
  `google-genai` SDK, key from `GEMINI_API_KEY`. Maps roles (assistant→model, tool results→user
  `function_response` parts), advertises tools via `parameters_json_schema`, and detects tool-use
  by function-call presence (Gemini reports `finish_reason: STOP` even when calling). Wired into
  `providers/factory.build_provider`.
- **`/models` command + `list_models` tool** — each provider grows `list_models()`. The
  `/models` REPL command lists the active provider's models (marking the current one); the
  model-callable `list_models` tool can also target a named provider or `all` providers whose
  keys are set.
- **Provider listing (`list_providers` tool + `/providers` command)** — a `provider → env var`
  mapping in `factory.py` becomes the single source of truth `_KNOWN` derives from; a shared
  `provider_status()` feeds both a model-callable tool and a REPL command. Detailed design note:
  [`plan-provider-listing.md`](plan-provider-listing.md).
- **Older-model thinking on Anthropic** — the adapter tries adaptive thinking + effort and, on
  the 400 that older models (Haiku 4.5, Sonnet 4.5) return, drops them and retries, remembering
  it per instance. An opt-in `thinking_budget` config makes those models use fixed-budget
  extended thinking on that fallback path instead of none. (The fallback is currently silent —
  see `future-work.md`.)
- **`web_fetch` tool** — model-callable HTTP GET/POST over stdlib `urllib`; `content_type` sets
  the POST input type, `accept` the desired response type. State-free, so subagents get it too.
  SSRF caveat and allowlisting are noted in `future-work.md`.
- **`web_search` tool** — keyless DuckDuckGo search (stdlib `urllib` + `html.parser`) returning
  title/URL/snippet; pairs with `web_fetch` (search finds, fetch reads). Keyed backends are a
  future seam (`future-work.md`).
- **Workspace filesystem tools** — a second bind-mounted directory (`/workspace`, config
  `workspace_dir`) and four tools over it: `list_dir`, `read_file`, `write_file`, `make_dir`.
  A new `workspace.py` owns confinement (the same resolve + `is_relative_to` check as
  `read_skill_file`) and the IO; `tools/fs_tools.py` owns the `Tool` wrappers and the write
  gate — `workspace_writable: false` returns only the read tools, so the model is never
  offered a tool that would refuse. No delete or move, by choice. Detailed design note:
  [`plan-workspace-tools.md`](plan-workspace-tools.md).
- **`spawn_subagent` tool** — delegation: the model spawns a fresh state (defaulting to its own
  provider/model, or a different one), runs a full tool/skill loop until it answers, and gets the
  answer back. Recursion-safe (subagents get the base toolset without `spawn_subagent`); the
  caller's active state is restored after the run; processing is logged to the REPL.
