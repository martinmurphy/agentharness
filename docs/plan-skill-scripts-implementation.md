# Skill Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the model run a Python script bundled under a loaded skill's `scripts/` directory, in a subprocess, behind a config gate that is off by default.

**Architecture:** A new safety-critical core module `skills/runner.py` owns path resolution, environment construction, and the subprocess call, importing nothing from `tools/`. A thin `tools/script_tools.py` wraps it as a single `run_skill_script` tool that is registered only when enabled. This is the same seam the repo already uses for `workspace.py` / `fs_tools.py` and `skills/loader.py` / `tools/skill_tools.py`.

**Tech Stack:** Python 3.14, stdlib `subprocess`, pytest, ruff. No new dependencies.

**Spec:** [`docs/plan-skill-scripts.md`](plan-skill-scripts.md) — read it first; this plan argues from it.

## Global Constraints

- **Nothing model-authored ever runs.** Only files under a loaded skill's `scripts/` directory. Never the workspace, never a string.
- **Never `shell=True`.** The command is always a list; `args` entries must be strings.
- **The child environment is built from empty, never inherited.** No provider API key may reach a script.
- **A skill's `metadata.env` is a request, not a grant.** What reaches the child is the intersection with the operator's `skill_scripts.env_allowlist`, which defaults to empty.
- **Default off.** `skill_scripts.enabled` defaults to `false`, and when false the tool is never *registered*.
- **A non-zero exit is not a harness error.** It returns normally with the status visible. `is_error` is for the harness failing to run the thing at all.
- **No new dependencies.** Stdlib only.
- **Style:** British spelling in prose (the repo uses "sanitised", "authorisation"). Commits use `Assisted-By: Claude Opus 5 <noreply@anthropic.com>`, never `Co-Authored-By`.
- **Verification for every task:** `.venv/bin/pytest -q` and `.venv/bin/ruff check .` both clean before the commit.

## File Structure

| File | Responsibility |
|---|---|
| `src/agentharness/skills/model.py` | *(modify)* `Skill.script_env`, and `ENV_NAME_RE` — the shared definition of a valid variable name |
| `src/agentharness/skills/loader.py` | *(modify)* parse and validate `metadata.env` into `script_env` |
| `src/agentharness/config.py` | *(modify)* the raw `skill_scripts` block; `_as_bool` promoted to `as_bool` |
| `src/agentharness/skills/runner.py` | **new** — `ScriptPolicy`, `parse_policy`, `resolve_script`, `child_env`, `run_script`, `ScriptResult`, `ScriptTimeout`, `ScriptConfigError` |
| `src/agentharness/tools/script_tools.py` | **new** — `make_script_tools`: schema, argument coercion, rendering |
| `src/agentharness/tools/registry.py` | *(modify)* a `policy` parameter, and the registration loop |
| `src/agentharness/repl.py` | *(modify)* parse the policy once, thread it into both registries, add the system-prompt line |
| `tests/test_script_runner.py` | **new** — the core: resolution, environment, subprocess, policy |
| `tests/test_skills.py` | *(modify)* `metadata.env` parsing |
| `tests/test_config.py` | *(modify)* the `skill_scripts` block |
| `tests/test_tools.py` | *(modify)* dispatch-level tool behaviour and the disabled gate |
| `tests/test_agent.py` | *(modify)* registration and the system-prompt line |
| `skills/word-frequency/` | **new** — an example skill whose script does something the model does badly by hand |

Tasks run in order; each depends on the ones before it.

---

### Task 1: A skill declares the variables its scripts need

**Files:**
- Modify: `src/agentharness/skills/model.py`
- Modify: `src/agentharness/skills/loader.py:106-115` (the `raw_metadata` block in `_validate_and_build`)
- Test: `tests/test_skills.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Skill.script_env: frozenset[str]`; `agentharness.skills.model.ENV_NAME_RE` (a compiled `re.Pattern`).

- [ ] **Step 1: Make the test helper able to write extra frontmatter**

In `tests/test_skills.py`, add an `{extra}` slot to the template and an `extra` parameter to the helper. Existing callers pass nothing and are unaffected.

```python
VALID_FRONTMATTER = """---
name: {name}
description: {desc}
{extra}---

# Body

Some instructions.
"""


def _write_skill(root, dirname, *, name=None, desc="A valid description of the skill.",
                 body_after_fence=True, extra=""):
    name = dirname if name is None else name
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True)
    if body_after_fence:
        text = VALID_FRONTMATTER.format(name=name, desc=desc, extra=extra)
    else:
        text = f"name: {name}\ndescription: {desc}\n"  # no frontmatter fences
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_skills.py`:

```python
ENV_META = 'metadata:\n  env: "GITHUB_TOKEN JIRA_TOKEN"\n'


def test_script_env_defaults_to_empty(tmp_path):
    _write_skill(tmp_path, "plain")
    skill = load_skills(tmp_path).by_name("plain")
    assert skill.script_env == frozenset()


def test_script_env_parsed_from_metadata(tmp_path):
    _write_skill(tmp_path, "declares", extra=ENV_META)
    skill = load_skills(tmp_path).by_name("declares")
    assert skill.script_env == frozenset({"GITHUB_TOKEN", "JIRA_TOKEN"})
    # metadata itself is left intact — script_env is a parsed view, not a move.
    assert skill.metadata["env"] == "GITHUB_TOKEN JIRA_TOKEN"


def test_invalid_script_env_name_fails_only_that_skill(tmp_path):
    _write_skill(tmp_path, "bad", extra='metadata:\n  env: "not-an-env-name"\n')
    _write_skill(tmp_path, "good")
    result = load_skills(tmp_path)
    assert [s.name for s in result.skills] == ["good"]
    assert len(result.errors) == 1
    assert "not-an-env-name" in result.errors[0].reason
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_skills.py -q -k script_env`
Expected: FAIL — `AttributeError: 'Skill' object has no attribute 'script_env'`

- [ ] **Step 4: Add the field and the shared regex**

In `src/agentharness/skills/model.py`, add `import re` and:

```python
# What a valid environment variable name looks like. Shared by the loader (a
# skill's metadata.env request) and the runner (the operator's allowlist), so
# the two halves of the intersection cannot disagree about what a name is.
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
```

and, as the last field of `Skill`:

```python
    # Environment variables the skill asks be forwarded to its scripts, parsed
    # from `metadata.env`. A request, not a grant: what actually reaches a
    # child is the intersection with the operator's allowlist. See
    # skills/runner.py.
    script_env: frozenset[str] = frozenset()
```

- [ ] **Step 5: Parse it in the loader**

In `src/agentharness/skills/loader.py`, import `ENV_NAME_RE` alongside `Skill`, add the parser above `_validate_and_build`:

```python
def _parse_script_env(raw: str | None) -> frozenset[str]:
    """Parse ``metadata.env``: the variables a skill asks its scripts be given.

    Space-separated, in the style of ``allowed-tools``. Validated here so a typo
    fails this one skill at load, with the offending name quoted, rather than
    silently dropping a variable at run time where nobody would connect the two.
    """
    if raw is None:
        return frozenset()
    names = raw.split()
    for name in names:
        if not ENV_NAME_RE.match(name):
            raise ValueError(
                f"invalid metadata.env entry {name!r}: environment variable names "
                "must match [A-Z_][A-Z0-9_]*"
            )
    return frozenset(names)
```

and pass it through — after the `metadata = {...}` line, add `script_env = _parse_script_env(metadata.get("env"))`, then add `script_env=script_env,` to the `Skill(...)` construction.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_skills.py -q && .venv/bin/ruff check .`
Expected: PASS, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/agentharness/skills/model.py src/agentharness/skills/loader.py tests/test_skills.py
git commit -m "Let a skill name the variables its scripts need

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The policy and its config block

**Files:**
- Modify: `src/agentharness/config.py:60` (after `mcp_servers`), `:78` (`_as_bool`), `:119` (the block loop)
- Create: `src/agentharness/skills/runner.py`
- Create: `tests/test_script_runner.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `Skill.script_env` (Task 1), `Config`.
- Produces: `ScriptPolicy(enabled: bool, default_timeout: int, max_timeout: int, max_output_bytes: int, env_allowlist: frozenset[str])` with `clamp_timeout(requested: Any) -> int`; `parse_policy(config: Config) -> ScriptPolicy`; `ScriptConfigError(ValueError)`; `Config.skill_scripts: dict[str, Any]`; `config.as_bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_script_runner.py`:

```python
"""Tests for the skill script policy, resolution, environment, and subprocess."""

from __future__ import annotations

import pytest

from agentharness.config import Config
from agentharness.skills.runner import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ScriptConfigError,
    ScriptPolicy,
    parse_policy,
)


def test_policy_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", raising=False)
    policy = parse_policy(Config())
    assert policy.enabled is False
    assert policy.env_allowlist == frozenset()
    assert policy.max_output_bytes == DEFAULT_MAX_OUTPUT_BYTES


def test_policy_from_block(monkeypatch):
    monkeypatch.delenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", raising=False)
    policy = parse_policy(Config(skill_scripts={
        "enabled": True,
        "default_timeout": 10,
        "max_timeout": 60,
        "max_output_bytes": 2048,
        "env_allowlist": ["GITHUB_TOKEN"],
    }))
    assert policy.enabled is True
    assert policy.env_allowlist == frozenset({"GITHUB_TOKEN"})
    assert policy.default_timeout == 10


def test_env_override_wins_over_file(monkeypatch):
    monkeypatch.setenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", "true")
    assert parse_policy(Config(skill_scripts={"enabled": False})).enabled is True
    monkeypatch.setenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", "no")
    assert parse_policy(Config(skill_scripts={"enabled": True})).enabled is False


@pytest.mark.parametrize("block, fragment", [
    ({"enabbled": True}, "enabbled"),
    ({"max_timeout": 0}, "max_timeout"),
    ({"max_timeout": "30"}, "max_timeout"),
    ({"default_timeout": 90, "max_timeout": 60}, "default_timeout"),
    ({"env_allowlist": "GITHUB_TOKEN"}, "env_allowlist"),
    ({"env_allowlist": ["lower_case"]}, "lower_case"),
])
def test_bad_block_is_rejected_at_load(monkeypatch, block, fragment):
    monkeypatch.delenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", raising=False)
    with pytest.raises(ScriptConfigError) as exc:
        parse_policy(Config(skill_scripts=block))
    assert fragment in str(exc.value)


def test_clamp_timeout():
    policy = ScriptPolicy(enabled=True, default_timeout=30, max_timeout=120)
    assert policy.clamp_timeout(None) == 30
    assert policy.clamp_timeout(5) == 5
    assert policy.clamp_timeout(600) == 120       # clamped, not rejected
    with pytest.raises(ValueError):
        policy.clamp_timeout(0)
    with pytest.raises(ValueError):
        policy.clamp_timeout("30")
    with pytest.raises(ValueError):
        policy.clamp_timeout(True)                # bool is an int; not a timeout
```

Append to `tests/test_config.py`:

```python
def test_skill_scripts_block_loaded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTHARNESS_CONFIG", raising=False)
    assert load_config().skill_scripts == {}  # absent block is an empty mapping

    path = tmp_path / "c.yaml"
    path.write_text(
        "skill_scripts:\n  enabled: true\n  env_allowlist: [GITHUB_TOKEN]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTHARNESS_CONFIG", str(path))
    cfg = load_config()
    assert cfg.skill_scripts["enabled"] is True
    assert cfg.skill_scripts["env_allowlist"] == ["GITHUB_TOKEN"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_script_runner.py tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentharness.skills.runner'`

- [ ] **Step 3: Add the config block**

In `src/agentharness/config.py`: rename `_as_bool` to `as_bool` (it now has a second caller outside this module) and update its three uses in `_SCALAR_FIELDS`. Add the field after `mcp_servers`:

```python
    # Skill script execution. Held raw for the same reason as ``mcp_servers``:
    # this module knows nothing about what the block means.
    # agentharness.skills.runner parses and validates it.
    skill_scripts: dict[str, Any] = field(default_factory=dict)
```

and add it to the block loop:

```python
    for block in ("providers", "mcp_servers", "skill_scripts"):
```

- [ ] **Step 4: Write the policy**

Create `src/agentharness/skills/runner.py` with the module docstring and this much (the rest arrives in Tasks 3-5):

```python
"""Run a skill's bundled Python scripts in a subprocess.

The safety-critical core, kept free of any Tool imports so it can be tested on
its own; ``tools/script_tools.py`` wraps it. The split mirrors ``workspace.py``
vs ``tools/fs_tools.py``.

What may run is deliberately narrow: a ``.py`` file under a loaded skill's
``scripts/`` directory, which is operator-curated and mounted read-only. Code
the model wrote is never executed — that is a different feature with a different
threat model. See docs/plan-skill-scripts.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agentharness.config import Config, as_bool
from agentharness.skills.model import ENV_NAME_RE

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_TIMEOUT = 120
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024

# Unknown keys are an error rather than ignored, as in mcp/config.py: there is
# no adapter downstream to absorb a typo, so it would be a silently missing
# setting — and here the missing setting could be a security control.
_POLICY_KEYS = frozenset(
    {"enabled", "default_timeout", "max_timeout", "max_output_bytes", "env_allowlist"}
)


class ScriptConfigError(ValueError):
    """A malformed ``skill_scripts`` block. Raised at startup, not at call time."""


@dataclass(frozen=True)
class ScriptPolicy:
    """What the operator permits. Disabled is the default in every direction."""

    enabled: bool = False
    default_timeout: int = DEFAULT_TIMEOUT
    max_timeout: int = DEFAULT_MAX_TIMEOUT
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    env_allowlist: frozenset[str] = frozenset()

    def clamp_timeout(self, requested: Any) -> int:
        """Seconds to allow, given what the model asked for (possibly nothing).

        An over-large request is clamped rather than refused: a model guessing
        600 wants a long run, not a failed call. A nonsensical one is refused,
        because silently substituting a default would hide the mistake.
        """
        if requested is None:
            return self.default_timeout
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise ValueError("'timeout' must be a positive whole number of seconds")
        return min(requested, self.max_timeout)


def _positive_int(block: dict[str, Any], key: str, default: int) -> int:
    value = block.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScriptConfigError(f"skill_scripts.{key} must be a positive integer")
    return value


def parse_policy(config: Config) -> ScriptPolicy:
    """Validate the ``skill_scripts`` block into a policy.

    Called once at startup so a typo is a clear error naming the key, rather
    than a surprise inside a tool call ten minutes later.
    """
    block = config.skill_scripts or {}
    if not isinstance(block, dict):
        raise ScriptConfigError("skill_scripts must be a mapping")
    unknown = sorted(set(block) - _POLICY_KEYS)
    if unknown:
        raise ScriptConfigError(
            f"unknown skill_scripts key(s): {', '.join(unknown)}; "
            f"expected any of {', '.join(sorted(_POLICY_KEYS))}"
        )

    enabled = bool(block.get("enabled", False))
    override = os.environ.get("AGENTHARNESS_SKILL_SCRIPTS_ENABLED")
    if override is not None:
        enabled = as_bool(override)

    default_timeout = _positive_int(block, "default_timeout", DEFAULT_TIMEOUT)
    max_timeout = _positive_int(block, "max_timeout", DEFAULT_MAX_TIMEOUT)
    if default_timeout > max_timeout:
        raise ScriptConfigError(
            f"skill_scripts.default_timeout ({default_timeout}) exceeds "
            f"max_timeout ({max_timeout})"
        )

    raw_allowlist = block.get("env_allowlist", [])
    if not isinstance(raw_allowlist, list) or not all(
        isinstance(v, str) for v in raw_allowlist
    ):
        raise ScriptConfigError("skill_scripts.env_allowlist must be a list of strings")
    for name in raw_allowlist:
        if not ENV_NAME_RE.match(name):
            raise ScriptConfigError(
                f"invalid skill_scripts.env_allowlist entry {name!r}: environment "
                "variable names must match [A-Z_][A-Z0-9_]*"
            )

    return ScriptPolicy(
        enabled=enabled,
        default_timeout=default_timeout,
        max_timeout=max_timeout,
        max_output_bytes=_positive_int(block, "max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES),
        env_allowlist=frozenset(raw_allowlist),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_script_runner.py tests/test_config.py -q && .venv/bin/ruff check .`
Expected: PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/agentharness/config.py src/agentharness/skills/runner.py \
        tests/test_script_runner.py tests/test_config.py
git commit -m "Add the skill_scripts policy, off by default

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: What may run

**Files:**
- Modify: `src/agentharness/skills/runner.py`
- Test: `tests/test_script_runner.py`

**Interfaces:**
- Consumes: `Skill` (Task 1).
- Produces: `resolve_script(skill: Skill, rel_path: str) -> Path`, raising `ValueError`.

- [ ] **Step 1: Add the shared test fixture**

The rest of the tasks all need a skill directory with scripts in it. Add to `tests/test_script_runner.py`, below the imports (extend the import line to include `resolve_script`):

```python
import textwrap

from agentharness.skills.loader import load_skills
from agentharness.workspace import Workspace


def _skill(tmp_path, *, name="demo", env="", scripts=None):
    """Build a skill directory and load it through the real loader.

    Going through load_skills rather than constructing a Skill directly is
    deliberate: script_env then comes from the same parsing path production
    uses, so a test cannot pass against a shape the loader never produces.
    """
    root = tmp_path / "skills"
    directory = root / name
    (directory / "scripts").mkdir(parents=True)
    meta = f'metadata:\n  env: "{env}"\n' if env else ""
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A demo skill used by tests.\n{meta}---\n\nBody.\n",
        encoding="utf-8",
    )
    for rel, source in (scripts or {}).items():
        path = directory / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    skill = load_skills(root).by_name(name)
    assert skill is not None
    return skill


def _ws(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return Workspace(root=root, writable=True)


def _enabled(**kwargs):
    return ScriptPolicy(enabled=True, **kwargs)
```

- [ ] **Step 2: Write the failing tests**

```python
def test_resolves_a_script_under_scripts(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/ok.py": "print('hi')\n"})
    assert resolve_script(skill, "scripts/ok.py").name == "ok.py"


@pytest.mark.parametrize("rel_path, fragment", [
    ("scripts/../../escape.py", "scripts/"),
    ("/etc/passwd", "scripts/"),
    ("references/REGIONAL.md", "scripts/"),
    ("notes.py", "scripts/"),
    ("scripts/notes.txt", "only .py"),
    ("scripts/missing.py", "no such script"),
])
def test_rejects(tmp_path, rel_path, fragment):
    skill = _skill(tmp_path, scripts={
        "scripts/ok.py": "print('hi')\n",
        "scripts/notes.txt": "not a script\n",
        "references/REGIONAL.md": "docs\n",
        "notes.py": "print('top level')\n",
    })
    (tmp_path / "skills" / "escape.py").write_text("print('nope')\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        resolve_script(skill, rel_path)
    assert fragment in str(exc.value)


def test_symlink_out_of_scripts_is_rejected(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/ok.py": "print('hi')\n"})
    outside = tmp_path / "outside.py"
    outside.write_text("print('nope')\n", encoding="utf-8")
    link = skill.path / "scripts" / "link.py"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        resolve_script(skill, "scripts/link.py")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_script_runner.py -q -k "resolve or rejects or symlink"`
Expected: FAIL — `ImportError: cannot import name 'resolve_script'`

- [ ] **Step 4: Implement it**

Add `from pathlib import Path` and `from agentharness.skills.model import ENV_NAME_RE, Skill` to the imports, then:

```python
def resolve_script(skill: Skill, rel_path: str) -> Path:
    """Resolve a runnable script inside a skill, refusing anything else.

    Resolving against ``scripts/`` rather than the skill root is strictly
    tighter than ``read_skill_file`` and subsumes its escape check: ``..``,
    absolute paths, and symlink escapes all fail the one ``is_relative_to``.
    The rule it leaves is a sentence a skill author can hold in their head —
    references/ is read, scripts/ is run — so a skill cannot be talked into
    executing its own documentation.
    """
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("'path' is required and must be a non-empty string")
    base = (skill.path / "scripts").resolve()
    target = (skill.path / rel_path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(
            f"path {rel_path!r} is not under the skill's scripts/ directory"
        )
    if target.suffix != ".py":
        raise ValueError(f"only .py files can be run: {rel_path}")
    if not target.is_file():
        raise ValueError(f"no such script: {rel_path}")
    return target
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_script_runner.py -q && .venv/bin/ruff check .`
Expected: PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/agentharness/skills/runner.py tests/test_script_runner.py
git commit -m "Confine what may run to a skill's scripts/ directory

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The child environment

**Files:**
- Modify: `src/agentharness/skills/runner.py`
- Test: `tests/test_script_runner.py`

**Interfaces:**
- Consumes: `Skill.script_env`, `ScriptPolicy.env_allowlist`, `Workspace`.
- Produces: `child_env(skill: Skill, ws: Workspace, policy: ScriptPolicy) -> dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_child_env_is_built_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("SOME_OTHER_VAR", "leak")
    skill = _skill(tmp_path)
    ws = _ws(tmp_path)
    env = child_env(skill, ws, _enabled())
    assert "ANTHROPIC_API_KEY" not in env
    assert "SOME_OTHER_VAR" not in env
    assert env["AGENTHARNESS_WORKSPACE_DIR"] == str(ws.root)
    assert env["AGENTHARNESS_SKILL_DIR"] == str(skill.path)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_env_forwarded_only_when_declared_and_permitted(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-1")
    monkeypatch.setenv("JIRA_TOKEN", "jira-1")
    monkeypatch.setenv("SLACK_TOKEN", "slack-1")
    skill = _skill(tmp_path, env="GITHUB_TOKEN JIRA_TOKEN")
    policy = _enabled(env_allowlist=frozenset({"GITHUB_TOKEN", "SLACK_TOKEN"}))
    env = child_env(skill, _ws(tmp_path), policy)
    assert env["GITHUB_TOKEN"] == "gh-1"   # declared and permitted
    assert "JIRA_TOKEN" not in env         # declared, not permitted
    assert "SLACK_TOKEN" not in env        # permitted, not declared


def test_declared_and_permitted_but_unset_is_silently_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    skill = _skill(tmp_path, env="GITHUB_TOKEN")
    env = child_env(skill, _ws(tmp_path), _enabled(env_allowlist=frozenset({"GITHUB_TOKEN"})))
    assert "GITHUB_TOKEN" not in env       # the script's own check reports it better
```

Extend the `runner` import to include `child_env`.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_script_runner.py -q -k env`
Expected: FAIL — `ImportError: cannot import name 'child_env'`

- [ ] **Step 3: Implement it**

Add `from agentharness.workspace import Workspace` to the imports, then:

```python
def child_env(skill: Skill, ws: Workspace, policy: ScriptPolicy) -> dict[str, str]:
    """The environment a script runs with: built from empty, never inherited.

    A script's stdout goes straight back into the model's context, so an
    inherited ANTHROPIC_API_KEY would be one ``print(os.environ)`` away from the
    transcript. What a skill declares is a *request*; what it gets is the
    intersection with the operator's allowlist, because skills are bind-mounted
    and may come from anywhere.

    PYTHONDONTWRITEBYTECODE earns its place: /skills is mounted read-only, and a
    __pycache__ write that silently fails on every run is noise nobody will chase.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(ws.root)),
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AGENTHARNESS_WORKSPACE_DIR": str(ws.root),
        "AGENTHARNESS_SKILL_DIR": str(skill.path),
    }
    for name in sorted(skill.script_env & policy.env_allowlist):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_script_runner.py -q && .venv/bin/ruff check .`
Expected: PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/agentharness/skills/runner.py tests/test_script_runner.py
git commit -m "Build a script's environment from empty, not from ours

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Running it

**Files:**
- Modify: `src/agentharness/skills/runner.py`
- Test: `tests/test_script_runner.py`

**Interfaces:**
- Consumes: `resolve_script`, `child_env`, `ScriptPolicy.clamp_timeout`.
- Produces: `ScriptResult(exit_status: int, stdout: str, stderr: str, stdout_dropped: int, stderr_dropped: int)`; `ScriptTimeout(ValueError)` with attributes `timeout: int`, `stdout: str`, `stderr: str`; `run_script(skill, rel_path, ws, policy, *, args=(), stdin=None, timeout=None) -> ScriptResult`.

- [ ] **Step 1: Write the failing tests**

```python
import json
import os
import time

from agentharness.skills.runner import ScriptResult, ScriptTimeout, run_script


def test_runs_and_captures_stdout(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/hello.py": """
        import sys
        print("hello", *sys.argv[1:])
    """})
    result = run_script(skill, "scripts/hello.py", _ws(tmp_path), _enabled(), args=["world"])
    assert isinstance(result, ScriptResult)
    assert result.exit_status == 0
    assert result.stdout.strip() == "hello world"
    assert result.stderr == ""


def test_non_zero_exit_returns_normally(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/fail.py": """
        import sys
        print("could not parse line 4", file=sys.stderr)
        sys.exit(3)
    """})
    result = run_script(skill, "scripts/fail.py", _ws(tmp_path), _enabled())
    assert result.exit_status == 3
    assert "could not parse" in result.stderr


def test_stdin_is_passed_through(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/echo.py": """
        import sys
        sys.stdout.write(sys.stdin.read().upper())
    """})
    result = run_script(skill, "scripts/echo.py", _ws(tmp_path), _enabled(), stdin="quiet")
    assert result.stdout == "QUIET"


def test_omitted_stdin_gives_eof_not_a_hang(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/echo.py": """
        import sys
        print(repr(sys.stdin.read()))
    """})
    result = run_script(skill, "scripts/echo.py", _ws(tmp_path), _enabled(), timeout=5)
    assert result.stdout.strip() == "''"


def test_cwd_is_the_workspace_root(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/write.py": """
        from pathlib import Path
        Path("made-here.txt").write_text("ok", encoding="utf-8")
        print("done")
    """})
    ws = _ws(tmp_path)
    run_script(skill, "scripts/write.py", ws, _enabled())
    assert (ws.root / "made-here.txt").read_text(encoding="utf-8") == "ok"


def test_output_is_capped(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/loud.py": 'print("x" * 5000)\n'})
    result = run_script(skill, "scripts/loud.py", _ws(tmp_path), _enabled(max_output_bytes=1000))
    assert len(result.stdout.encode("utf-8")) <= 1000
    assert result.stdout_dropped > 0


def test_arguments_are_never_shell_interpreted(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/args.py": """
        import sys
        print(sys.argv[1])
    """})
    result = run_script(
        skill, "scripts/args.py", _ws(tmp_path), _enabled(), args=["; echo pwned"]
    )
    assert result.stdout.strip() == "; echo pwned"  # one odd string, not a command


def test_environment_reaches_the_child(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    skill = _skill(tmp_path, scripts={"scripts/dump.py": """
        import json, os
        print(json.dumps(dict(os.environ)))
    """})
    result = run_script(skill, "scripts/dump.py", _ws(tmp_path), _enabled())
    env = json.loads(result.stdout)
    assert "ANTHROPIC_API_KEY" not in env
    assert env["AGENTHARNESS_SKILL_DIR"] == str(skill.path)


def test_disabled_policy_refuses_to_run(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/hello.py": 'print("hi")\n'})
    with pytest.raises(ValueError) as exc:
        run_script(skill, "scripts/hello.py", _ws(tmp_path), ScriptPolicy())
    assert "disabled" in str(exc.value)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_timeout_kills_the_whole_process_group(tmp_path):
    """A script that forked must not leave its child running after the timeout.

    subprocess's own timeout kills the direct child only, so this asserts on the
    grandchild's PID rather than merely on our having stopped waiting.
    """
    skill = _skill(tmp_path, scripts={"scripts/forker.py": """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        time.sleep(60)
    """})
    with pytest.raises(ScriptTimeout) as exc:
        run_script(skill, "scripts/forker.py", _ws(tmp_path), _enabled(), timeout=1)

    assert exc.value.timeout == 1
    grandchild = int(exc.value.stdout.strip())   # partial output survived the kill
    deadline = time.time() + 5                   # reparenting and reaping is not instant
    while time.time() < deadline and _alive(grandchild):
        time.sleep(0.05)
    assert not _alive(grandchild)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_script_runner.py -q -k "runs or exit or stdin or cwd or capped or shell or reaches or disabled or timeout_kills"`
Expected: FAIL — `ImportError: cannot import name 'run_script'`

- [ ] **Step 3: Implement it**

Add `import signal`, `import subprocess`, `import sys`, and `from collections.abc import Sequence` to the imports, then:

```python
@dataclass(frozen=True)
class ScriptResult:
    """One completed run. ``dropped`` counts bytes lost to the output cap."""

    exit_status: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


class ScriptTimeout(ValueError):
    """A script that outlived its timeout and was killed, with what it had said.

    The partial output travels with the error because that is usually where a
    hang explains itself.
    """

    def __init__(self, timeout: int, stdout: str, stderr: str) -> None:
        super().__init__(f"script exceeded its {timeout}s timeout and was killed")
        self.timeout = timeout
        self.stdout = stdout
        self.stderr = stderr


def _truncate(text: str, limit: int) -> tuple[str, int]:
    """Cut ``text`` to ``limit`` bytes of UTF-8, on a character boundary."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, 0
    kept = raw[:limit].decode("utf-8", errors="ignore")
    return kept, len(raw) - len(kept.encode("utf-8"))


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_script(
    skill: Skill,
    rel_path: str,
    ws: Workspace,
    policy: ScriptPolicy,
    *,
    args: Sequence[str] = (),
    stdin: str | None = None,
    timeout: int | None = None,
) -> ScriptResult:
    """Run a script bundled with ``skill`` and return what it said.

    A non-zero exit is a *result*, not an exception: a script that exits 2 and
    explains itself on stderr has told the model something it can act on.
    Exceptions are for the harness failing to run the thing at all.

    Note the honest limit: a subprocess is not confined by
    ``workspace.resolve_in``. It can reach any path the container user can. The
    container is the confinement, which is why only vetted scripts may run.
    """
    if not policy.enabled:
        raise ValueError(
            "skill scripts are disabled; set skill_scripts.enabled: true in config "
            "(or AGENTHARNESS_SKILL_SCRIPTS_ENABLED=true) to allow them"
        )
    target = resolve_script(skill, rel_path)
    if not ws.root.is_dir():
        raise ValueError(f"workspace directory does not exist: {ws.root}")
    seconds = policy.clamp_timeout(timeout)

    # A list, never shell=True: there is no shell, so there is nothing to inject
    # into. sys.executable rather than a PATH lookup, so a `python` earlier on
    # PATH cannot substitute itself. start_new_session gives the child a process
    # group of its own — subprocess's timeout kills the direct child only, and a
    # script that forked would otherwise leave orphans holding the pipes open.
    proc = subprocess.Popen(  # noqa: S603 - argv list, no shell; path resolved above
        [sys.executable, str(target), *args],
        cwd=str(ws.root),
        env=child_env(skill, ws, policy),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",  # a script writing non-UTF-8 has a bug; we don't raise
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(stdin, timeout=seconds)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # a grandchild escaped the group
            out, err = "", ""
        kept_out, _ = _truncate(out or "", policy.max_output_bytes)
        kept_err, _ = _truncate(err or "", policy.max_output_bytes)
        raise ScriptTimeout(seconds, kept_out, kept_err) from None

    kept_out, dropped_out = _truncate(out, policy.max_output_bytes)
    kept_err, dropped_err = _truncate(err, policy.max_output_bytes)
    return ScriptResult(
        exit_status=proc.returncode,
        stdout=kept_out,
        stderr=kept_err,
        stdout_dropped=dropped_out,
        stderr_dropped=dropped_err,
    )
```

- [ ] **Step 4: Run the whole file to verify it passes**

Run: `.venv/bin/pytest tests/test_script_runner.py -q && .venv/bin/ruff check .`
Expected: PASS, ruff clean. If ruff objects to the `subprocess` call, keep the `noqa` comment that names *why* it is safe rather than removing the check.

- [ ] **Step 5: Commit**

```bash
git add src/agentharness/skills/runner.py tests/test_script_runner.py
git commit -m "Run a skill script in a subprocess it cannot outlive

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The tool

**Files:**
- Create: `src/agentharness/tools/script_tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `run_script`, `ScriptResult`, `ScriptTimeout`, `ScriptPolicy`, `SkillSet`, `Workspace`.
- Produces: `make_script_tools(skillset: SkillSet, ws: Workspace, policy: ScriptPolicy) -> list[Tool]`, registering a tool named `run_skill_script`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools.py` (add the imports it needs at the top: `from agentharness.skills.runner import ScriptPolicy`, `from agentharness.tools.script_tools import make_script_tools`, and `import textwrap`):

```python
def _script_skillset(tmp_path, source='print("hi")\n', name="demo"):
    directory = tmp_path / "skills" / name
    (directory / "scripts").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A demo skill used by tests.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (directory / "scripts" / "run.py").write_text(
        textwrap.dedent(source).lstrip(), encoding="utf-8"
    )
    return load_skills(tmp_path / "skills")


def _script_registry(tmp_path, source='print("hi")\n', **policy_kwargs):
    reg = ToolRegistry()
    policy = ScriptPolicy(enabled=True, **policy_kwargs)
    for tool in make_script_tools(_script_skillset(tmp_path, source), _ws(tmp_path), policy):
        reg.register(tool)
    return reg


def test_disabled_policy_registers_no_tool(tmp_path):
    tools = make_script_tools(_script_skillset(tmp_path), _ws(tmp_path), ScriptPolicy())
    assert tools == []


def test_run_skill_script_renders_a_result(tmp_path):
    reg = _script_registry(tmp_path, """
        import sys
        print("counted 3 things")
        print("a warning", file=sys.stderr)
    """)
    result = reg.dispatch(ToolCall(
        id="c1",
        name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert not result.is_error
    assert "exit status: 0" in result.content
    assert "counted 3 things" in result.content
    assert "a warning" in result.content


def test_empty_streams_render_as_empty(tmp_path):
    reg = _script_registry(tmp_path, "pass\n")
    result = reg.dispatch(ToolCall(
        id="c2", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert "(empty)" in result.content


def test_non_zero_exit_is_not_a_tool_error(tmp_path):
    reg = _script_registry(tmp_path, "import sys; sys.exit(4)\n")
    result = reg.dispatch(ToolCall(
        id="c3", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert not result.is_error          # the model should read it, not give up
    assert "exit status: 4" in result.content


def test_non_string_args_are_rejected_with_advice(tmp_path):
    reg = _script_registry(tmp_path)
    result = reg.dispatch(ToolCall(
        id="c4", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py", "args": [3]},
    ))
    assert result.is_error
    assert "list of strings" in result.content


def test_unknown_skill_names_the_available_ones(tmp_path):
    reg = _script_registry(tmp_path)
    result = reg.dispatch(ToolCall(
        id="c5", name="run_skill_script",
        arguments={"skill": "nope", "path": "scripts/run.py"},
    ))
    assert result.is_error
    assert "demo" in result.content


def test_timeout_is_an_error_carrying_partial_output(tmp_path):
    reg = _script_registry(tmp_path, """
        import time
        print("started", flush=True)
        time.sleep(60)
    """, default_timeout=1, max_timeout=1)
    result = reg.dispatch(ToolCall(
        id="c6", name="run_skill_script",
        arguments={"skill": "demo", "path": "scripts/run.py"},
    ))
    assert result.is_error
    assert "timeout" in result.content
    assert "started" in result.content
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_tools.py -q -k "script"`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentharness.tools.script_tools'`

- [ ] **Step 3: Implement it**

Create `src/agentharness/tools/script_tools.py`:

```python
"""The ``run_skill_script`` tool: execution of a skill's bundled Python.

The thin half of the seam. Everything safety-critical — what may run, what
environment it gets, how it is killed — lives in ``skills/runner.py``; this
module owns the schema, the argument coercion, and the rendering the model
reads. A disabled policy returns no tools at all, so the model is never offered
a capability it cannot have, exactly as a read-only workspace withholds
``write_file``.
"""

from __future__ import annotations

from typing import Any

from agentharness.skills.loader import SkillSet
from agentharness.skills.runner import (
    ScriptPolicy,
    ScriptResult,
    ScriptTimeout,
    run_script,
)
from agentharness.tools.registry import Tool
from agentharness.workspace import Workspace


def _stream(text: str, dropped: int) -> str:
    body = text.rstrip("\n") or "(empty)"
    if dropped:
        body += f"\n... truncated: {dropped} more bytes"
    return body


def _render(result: ScriptResult) -> str:
    return "\n".join(
        [
            f"exit status: {result.exit_status}",
            "--- stdout ---",
            _stream(result.stdout, result.stdout_dropped),
            "--- stderr ---",
            _stream(result.stderr, result.stderr_dropped),
        ]
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(
            "'args' must be a list of strings — quote numbers, e.g. [\"3\"] not [3]"
        )
    return value


def make_script_tools(
    skillset: SkillSet, ws: Workspace, policy: ScriptPolicy
) -> list[Tool]:
    if not policy.enabled:
        return []

    def _run(args: dict[str, Any]) -> str:
        name = args.get("skill")
        path = args.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError("'skill' is required")
        if not isinstance(path, str) or not path:
            raise ValueError("'path' is required")
        skill = skillset.by_name(name)
        if skill is None:
            available = ", ".join(s.name for s in skillset.skills) or "(none)"
            raise ValueError(f"unknown skill {name!r}; available: {available}")
        stdin = args.get("stdin")
        if stdin is not None and not isinstance(stdin, str):
            raise ValueError("'stdin' must be a string")
        try:
            result = run_script(
                skill,
                path,
                ws,
                policy,
                args=_string_list(args.get("args")),
                stdin=stdin,
                timeout=args.get("timeout"),
            )
        except ScriptTimeout as exc:
            # An error, but the partial output goes with it: that is usually
            # where the hang explains itself.
            raise ValueError(
                f"{exc}\n--- stdout so far ---\n{_stream(exc.stdout, 0)}\n"
                f"--- stderr so far ---\n{_stream(exc.stderr, 0)}"
            ) from None
        return _render(result)

    return [
        Tool(
            name="run_skill_script",
            description=(
                "Run a Python script bundled with a skill, under its scripts/ "
                "directory, and get back its exit status, stdout, and stderr. Use "
                "this rather than reimplementing what a skill's script already "
                "does; the skill's instructions say which script to use and what "
                "arguments it takes. The script runs with the workspace as its "
                "working directory, so relative paths are workspace paths."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "The skill's name."},
                    "path": {
                        "type": "string",
                        "description": (
                            "The script, relative to the skill directory, e.g. "
                            "'scripts/analyse.py'."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command-line arguments, each a string.",
                        "default": [],
                    },
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Text to send to the script's standard input. Omit to "
                            "give it no input."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Seconds to allow (default {policy.default_timeout}, "
                            f"capped at {policy.max_timeout})."
                        ),
                    },
                },
                "required": ["skill", "path"],
                "additionalProperties": False,
            },
            handler=_run,
        )
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tools.py -q && .venv/bin/ruff check .`
Expected: PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/agentharness/tools/script_tools.py tests/test_tools.py
git commit -m "Offer run_skill_script, but only where it is permitted

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Wiring

**Files:**
- Modify: `src/agentharness/tools/registry.py:80-115` (`build_default_registry`)
- Modify: `src/agentharness/repl.py:168-172` (Harness `__init__`), `:244` (`_build_base_registry`), `:334-355` (`effective_system`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `make_script_tools`, `parse_policy`, `ScriptPolicy`.
- Produces: `build_default_registry(skillset, workspace, config=None, policy=None)`; `Harness.script_policy: ScriptPolicy`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py` (the module already imports `Config` and `load_skills`; add `from agentharness.skills.runner import ScriptPolicy`):

```python
def test_script_tool_absent_by_default(tmp_path):
    registry = build_default_registry(load_skills(tmp_path), _ws(tmp_path))
    assert "run_skill_script" not in registry


def test_script_tool_registered_when_policy_enabled(tmp_path):
    registry = build_default_registry(
        load_skills(tmp_path), _ws(tmp_path), None, ScriptPolicy(enabled=True)
    )
    assert "run_skill_script" in registry


def _skill_with_script(tmp_path):
    directory = tmp_path / "demo"
    (directory / "scripts").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill used by tests.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (directory / "scripts" / "run.py").write_text('print("hi")\n', encoding="utf-8")


def test_harness_registers_the_script_tool_when_enabled(tmp_path, monkeypatch):
    from agentharness import repl

    _skill_with_script(tmp_path)
    monkeypatch.setenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", "true")
    provider = FakeProvider([])
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: provider)
    h = repl.Harness(Config(skills_dir=str(tmp_path), workspace_dir=str(_ws(tmp_path).root)))
    assert "run_skill_script" in h.registry
    assert "run_skill_script" in h._subagent_registry   # subagents share the workspace
    assert "run_skill_script" in h.effective_system(h.states.active)


def test_harness_omits_the_script_tool_and_its_prompt_line(tmp_path, monkeypatch):
    from agentharness import repl

    _skill_with_script(tmp_path)
    monkeypatch.delenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", raising=False)
    provider = FakeProvider([])
    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: provider)
    h = repl.Harness(Config(skills_dir=str(tmp_path), workspace_dir=str(_ws(tmp_path).root)))
    assert "run_skill_script" not in h.registry
    assert "run_skill_script" not in h.effective_system(h.states.active)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_agent.py -q -k script`
Expected: FAIL — `TypeError: build_default_registry() takes 3 positional arguments but 4 were given`

- [ ] **Step 3: Thread the policy through the registry**

In `src/agentharness/tools/registry.py`, add `ScriptPolicy` to the `TYPE_CHECKING` imports, extend the signature and the docstring, and register the tool. The policy defaults to a disabled one so every existing caller — and every test that builds a registry with two arguments — keeps today's behaviour:

```python
def build_default_registry(
    skillset: SkillSet,
    workspace: Workspace,
    config: Config | None = None,
    policy: ScriptPolicy | None = None,
) -> ToolRegistry:
```

Inside, alongside the other lazy imports, add:

```python
    from agentharness.skills.runner import ScriptPolicy as _ScriptPolicy
    from agentharness.tools.script_tools import make_script_tools
```

and after the `make_fs_tools` loop:

```python
    # A disabled default matters: a caller that passes no policy gets a harness
    # that executes nothing, which is what every existing caller expects.
    for tool in make_script_tools(skillset, workspace, policy or _ScriptPolicy()):
        registry.register(tool)
```

Add a line to the docstring: `` `policy` gates `run_skill_script`; without one, scripts are off. ``

- [ ] **Step 4: Parse the policy once in the Harness**

In `src/agentharness/repl.py`, import `parse_policy` from `agentharness.skills.runner`, and in `Harness.__init__` immediately after the `self.workspace = Workspace(...)` block:

```python
        # Parsed once, here, so a malformed skill_scripts block fails at startup
        # naming the key rather than inside a tool call ten minutes later.
        self.script_policy = parse_policy(config)
```

In `_build_base_registry`, pass it:

```python
        registry = build_default_registry(
            self.skillset, self.workspace, self.config, self.script_policy
        )
```

- [ ] **Step 5: Tell the model the scripts are there**

In `effective_system`, after the workspace block:

```python
        if self.script_policy.enabled and self.skillset.skills:
            parts.append(
                "A skill may bundle Python scripts under its scripts/ directory. "
                "Run one with run_skill_script rather than reimplementing what it "
                "does; the skill's instructions say which script to use and what "
                "arguments it takes."
            )
```

- [ ] **Step 6: Run the full suite to verify**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check .`
Expected: PASS (every existing test included — the two-argument `build_default_registry` calls must still work), ruff clean.

- [ ] **Step 7: Check it by hand**

```bash
printf '/tools\n/quit\n' | .venv/bin/agentharness | grep -c run_skill_script
#   expect: 0
printf '/tools\n/quit\n' | AGENTHARNESS_SKILL_SCRIPTS_ENABLED=true .venv/bin/agentharness \
  | grep -c run_skill_script
#   expect: 1
```

- [ ] **Step 8: Commit**

```bash
git add src/agentharness/tools/registry.py src/agentharness/repl.py tests/test_agent.py
git commit -m "Wire the script policy through the registries

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: An example skill, and the docs that currently say the opposite

**Files:**
- Create: `skills/word-frequency/SKILL.md`, `skills/word-frequency/scripts/wordfreq.py`
- Modify: `config.example.yaml:28` (after `workspace_writable`)
- Modify: `README.md:483` (before `## Extending`), `:523-524` (the *Add a skill* bullet), `:554-560` (*Scope (v1)*)
- Modify: `docs/plan.md:35` (the decisions table row), `:222` (the tool table)
- Modify: `docs/plan-skill-scripts.md` (status footer)

**Interfaces:**
- Consumes: everything above. Produces: no code interface.

- [ ] **Step 1: Write the example skill's script**

`skills/word-frequency/scripts/wordfreq.py` — stdlib only, and deliberately the kind of counting a model does unreliably by hand:

```python
"""Count word frequencies in a workspace text file.

Run via the harness's run_skill_script tool. The working directory is the
workspace root, so PATH arguments are workspace-relative.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="text file to read, relative to the workspace")
    parser.add_argument("-n", "--top", type=int, default=20, help="how many to show")
    parser.add_argument("--min-length", type=int, default=1, help="ignore shorter words")
    args = parser.parse_args()

    try:
        text = Path(args.path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {args.path}: {exc}", file=sys.stderr)
        return 1

    words = [w for w in re.findall(r"[\w'-]+", text.lower()) if len(w) >= args.min_length]
    if not words:
        print("no words found", file=sys.stderr)
        return 1

    counts = Counter(words)
    print(f"{len(words)} words, {len(counts)} distinct")
    for word, n in counts.most_common(args.top):
        print(f"{n:>7}  {word}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the example skill's instructions**

`skills/word-frequency/SKILL.md`:

```markdown
---
name: word-frequency
description: Count word frequencies in a text file in the workspace. Use when asked which words are most common in a document, for a word count, or for a vocabulary or repetition check on a file.
license: MIT
metadata:
  author: agentharness
  version: "1.0"
---

# Word frequency

Counting words by eye is exactly the kind of task a language model does
confidently and wrongly. Do not do it by hand, and do not write a new script to
do it: run the bundled one.

## Usage

Call `run_skill_script` with `skill: "word-frequency"` and
`path: "scripts/wordfreq.py"`. Paths in `args` are relative to the workspace
root, which is the script's working directory.

    args: ["notes.md"]                        # top 20 words
    args: ["notes.md", "--top", "5"]          # top 5
    args: ["notes.md", "--min-length", "4"]   # skip short function words

It prints a total, a distinct count, and one line per word, most frequent
first. A missing or unreadable file exits 1 with the reason on stderr — report
that rather than guessing at the contents.

## Reporting the result

Quote the counts as given. If the user wants prose rather than a table,
summarise the top few and say how many distinct words there were.
```

- [ ] **Step 3: Verify the example end to end**

```bash
printf 'the quick brown fox jumps over the lazy dog the fox\n' > workspace/sample.txt
AGENTHARNESS_SKILL_SCRIPTS_ENABLED=true .venv/bin/python -c "
from agentharness.config import Config
from agentharness.skills.loader import load_skills
from agentharness.skills.runner import parse_policy, run_script
from agentharness.workspace import Workspace
skill = load_skills('./skills').by_name('word-frequency')
policy = parse_policy(Config())
print(run_script(skill, 'scripts/wordfreq.py', Workspace(root=__import__('pathlib').Path('./workspace')), policy, args=['sample.txt', '--top', '3']).stdout)
"
```
Expected: `11 words, 8 distinct`, then `the` at 3 and `fox` at 2.

- [ ] **Step 4: Document the config block**

In `config.example.yaml`, after the `workspace_writable` entry:

```yaml
# Skill scripts: whether the model may RUN a .py file bundled under a loaded
# skill's scripts/ directory (never code it wrote itself — the workspace is not
# executable). Off by default: with `enabled: false` the run_skill_script tool
# is not registered at all, so the model is never offered it.
skill_scripts:
  enabled: false             # AGENTHARNESS_SKILL_SCRIPTS_ENABLED=true also flips this
  default_timeout: 30        # seconds, when the model names none
  max_timeout: 120           # a model-supplied timeout is clamped to this
  max_output_bytes: 65536    # per stream; the rest is truncated with a marker
  # A script's environment is built from empty — no API key reaches it. A skill
  # asks for variables in its frontmatter (`metadata.env`); it gets only those
  # also named here. Both halves are required, so a downloaded skill cannot help
  # itself to a token by asking.
  env_allowlist: []
```

- [ ] **Step 5: Add the README section**

In `README.md`, immediately before `## Extending` (line 483), add:

````markdown
### Skill scripts

A skill can bundle Python under `scripts/`, and — when you turn it on — the
model can run it:

```
skill_scripts:
  enabled: true
```

Nothing runs by default. With `enabled: false`, `run_skill_script` is not
registered, so the model never sees it. The bundled `word-frequency` skill is
the worked example: counting words is a thing models do confidently and wrongly,
and the script is the version that is right every time.

**Only a skill's own scripts run.** The tool takes a skill name and a path under
that skill's `scripts/` directory, resolved with the same
escape-proof check `read_skill_file` uses, plus a `.py` requirement. Code the
model wrote into the workspace is *not* executable — the workspace is data. The
rule is one sentence: `references/` is read, `scripts/` is run.

**A script's environment is built, not inherited.** It gets `PATH`, `HOME`, a
UTF-8 locale, and `AGENTHARNESS_WORKSPACE_DIR` / `AGENTHARNESS_SKILL_DIR`.
No `ANTHROPIC_API_KEY`, no anything else — a script's stdout goes back into the
model's context, so an inherited key would be one `print` away from the
transcript. A script that genuinely needs a token requires *two* declarations:
the skill asks in its frontmatter, and you permit it in config.

```yaml
# skills/deploy-notes/SKILL.md
metadata:
  env: "GITHUB_TOKEN"
```
```yaml
# config.yaml
skill_scripts:
  env_allowlist: [GITHUB_TOKEN]
```

Only the intersection is forwarded, so a skill you downloaded cannot help itself
to a token by asking for one.

Two limits worth knowing before you enable this:

- **A subprocess is not confined to the workspace.** The filesystem tools are;
  a script is not, and can reach anything the container user can. The container
  is the boundary, which is why only vetted scripts under a read-only `/skills`
  mount may run.
- **Scripts get the harness's own interpreter** — the stdlib plus what
  agentharness depends on, and nothing else. A skill needing `numpy` means
  adding it to the `Containerfile`; there is no per-skill environment and no
  install at run time.

A run is capped by `max_timeout` and killed by process *group* if it overruns, so
a script that forked cannot leave anything behind. Output is capped per stream
and truncated with a marker.
````

- [ ] **Step 6: Fix the three places that state the opposite**

`README.md`, the *Add a skill* bullet (line ~523) — replace the last sentence:

> Optional `references/` and `assets/` files are read as text via `read_skill_file`; `scripts/` are read the same way and, with `skill_scripts.enabled`, run with `run_skill_script` (see *Skill scripts*). Run `/reload`.

`README.md`, *Scope (v1)* (line ~554) — replace the clause about scripts:

> Conversation state is in-memory only (histories do not survive a restart — files the model wrote to the workspace do); a skill's `scripts/` run only where the operator enables them, and only from the read-only skills mount — the workspace stays data, never code; no streaming yet.

Also add `run_skill_script` to the tool list in *What it does* (line ~20), after the `read_skill` / `read_skill_file` entry: `` `run_skill_script` (run a skill's bundled Python, off by default) ``.

`docs/plan.md` line 35 — replace the decisions-table row:

> | Skill `scripts/` | Read as text; **run** via `run_skill_script` behind `skill_scripts.enabled`, default off (v2) |

`docs/plan.md` line 222 — add a row to the tool table:

> | `run_skill_script` | `skill: str, path: str, args: list[str], stdin: str, timeout: int` | exit status, stdout, and stderr of a `.py` under the skill's `scripts/` |

- [ ] **Step 7: Run everything**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/agentharness --list-skills          # word-frequency present and valid
podman build -t agentharness . && make list-skills
```
Expected: suite green, ruff clean, both listings include `word-frequency`.

- [ ] **Step 8: Mark the spec implemented and commit**

Append to `docs/plan-skill-scripts.md`, matching how the other plan docs close:

```markdown
---

## Status: implemented

The suite passes, ruff is clean, and the bundled `word-frequency` skill runs
end to end in the container with `skill_scripts.enabled: true`.
```

```bash
git add skills/word-frequency config.example.yaml README.md docs/plan.md \
        docs/plan-skill-scripts.md
git commit -m "Ship an example skill whose script does the counting

Assisted-By: Claude Opus 5 <noreply@anthropic.com>"
```
