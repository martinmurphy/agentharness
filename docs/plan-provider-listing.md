# Add a provider-listing capability (tool + `/providers` command)

## Context

The harness supports three providers (`anthropic`, `openai`, `gemini`), but nothing surfaces
*which* providers exist or which are usable right now. After the recent Gemini friction (wrong
model IDs, missing keys), a way to answer "what providers can I use, and are their keys set?"
is useful both to the model mid-conversation and to the human at the REPL.

This adds that listing on **both surfaces from one shared core** (the user's choice): a
model-callable `list_providers` tool (alongside `greet` / `read_skill`) and a `/providers` REPL
command (alongside `/models` / `/tools`). Each entry shows the provider name, the environment
variable its key comes from, and whether that key is currently set — with the active state's
provider marked.

Grounding fact from exploration: `providers/factory.py` already enumerates the providers in
`_KNOWN = ("anthropic", "openai", "gemini")` but has **no mapping of provider → env var** (each
SDK reads its own key internally). This change introduces that mapping as a single source of
truth and derives `_KNOWN` from it, so the factory and both new surfaces agree.

## Design

### 1. Provider metadata + status core — `providers/factory.py`

Add a small metadata table and a status function; refactor `_KNOWN` to derive from it.

```python
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class ProviderInfo:
    name: str
    env_var: str          # the env var the SDK reads its key from
    available: bool        # env_var is set in the environment

_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_KNOWN = tuple(_ENV_VARS)   # derived; build_provider's error message stays correct

def provider_status() -> list[ProviderInfo]:
    return [
        ProviderInfo(name=n, env_var=v, available=bool(os.environ.get(v)))
        for n, v in _ENV_VARS.items()
    ]
```

`provider_status()` is the shared core both surfaces call. It does **not** compute an "active"
marker — that is state-dependent and belongs to the caller (the `/providers` command has the
active state; the tool is by definition already running on it).

### 2. Model-callable tool — `tools/provider_tools.py` (new)

Mirror `tools/greet.py`: a zero-arg tool whose handler formats `provider_status()` into text.

```python
def list_providers_tool() -> Tool:
    def _handler(args):
        lines = []
        for p in provider_status():
            state = "key set" if p.available else "no key"
            lines.append(f"{p.name} (key from {p.env_var}): {state}")
        return "\n".join(lines)
    return Tool(
        name="list_providers",
        description="List the model providers this harness supports and whether each "
                    "one's API key is currently configured.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_handler,
    )
```

Register it in `tools/registry.build_default_registry` next to `greet_tool()` — it needs no
skillset, so the function signature is unchanged. It then appears in `/tools` automatically.

### 3. REPL command — `repl.py`

- Add `/providers` to the `HELP` text (after `/models`).
- Add an `elif cmd == "/providers": _list_providers(h)` branch in `_handle_command`.
- Add a `_list_providers(h)` helper (near `_list_models`) that prints `provider_status()` with
  the active provider marked and colourised availability:

```python
def _list_providers(h):
    active = h.states.active.provider_name
    for p in provider_status():
        avail = h.ansi.green("available") if p.available else h.ansi.red("no key")
        mark = h.ansi.green(" *") if p.name == active else "  "
        print(f"{mark} {p.name:10} {p.env_var:20} {avail}"
              + ("  (active)" if p.name == active else ""))
```

## Files to modify

- `src/agentharness/providers/factory.py` — add `ProviderInfo`, `_ENV_VARS`, `provider_status()`;
  derive `_KNOWN`.
- `src/agentharness/tools/provider_tools.py` — **new**, `list_providers_tool()`.
- `src/agentharness/tools/registry.py` — register the tool in `build_default_registry`.
- `src/agentharness/repl.py` — `/providers` command, help entry, `_list_providers` helper.
- `README.md` — add `/providers` to the commands block and `list_providers` to the extension
  notes.

Reuses: `Tool`/`ToolRegistry` (`tools/registry.py`), the `greet.py` tool shape, the `_Ansi`
helper and `_list_models` command pattern (`repl.py`).

## Verification

```bash
.venv/bin/python -m pytest -q          # existing 73 + new tests, no network
.venv/bin/ruff check .
```

New tests:
- `tests/test_providers.py` — `provider_status()` reflects env: `monkeypatch.setenv/delenv` the
  three vars and assert `available` flags; `_KNOWN == ("anthropic","openai","gemini")`.
- `tests/test_tools.py` — dispatch `list_providers` (no args) via the default registry; assert
  the output names all three providers and reflects a monkeypatched key.
- `tests/test_agent.py` — REPL `/providers` via the existing `_harness` fixture + `capsys`:
  assert all three names appear and the active provider is marked; `/help` lists `/providers`.

Manual (no key needed for the listing itself — it only reads env):

```bash
printf '/providers\n/tools\n/quit\n' | .venv/bin/agentharness
# expect: anthropic/openai/gemini with availability, default (anthropic) marked active;
#         list_providers present in /tools
```

End-to-end with a key (model actually calls the tool):

```bash
ANTHROPIC_API_KEY=... make run
# then: "which providers do I have set up?"  -> model calls list_providers -> answers
```

Then rebuild the image (`podman build -t agentharness .`) and commit.

## Notes

- **`available` is a key-set heuristic.** For the OpenAI-compatible provider pointed at a local
  server (Ollama/vLLM via `base_url`), no key is needed, so `available: no key` there does not
  mean unusable. The tool description and command output say "key set / no key" (a fact), not
  "usable", to avoid overclaiming. Worth a one-line note in the help/README.
- **"Active" lives in the command, not the core.** `provider_status()` stays state-free; only
  `/providers` marks the active provider. The tool omits it (the model is already on the active
  provider by construction).
- Scope is intentionally small: this lists providers and key presence. It does **not** let the
  model switch providers — that remains a human action via `/new --provider` (states bind to one
  provider at creation, per the existing design).

---

## Status: implemented

Shipped in commit `7c15442` ("Add provider-listing: list_providers tool + /providers command").
All items above landed as planned; 78 tests pass, ruff clean, and the image rebuilds with both
surfaces working.
