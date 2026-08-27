"""Tests for the skill script policy, resolution, environment, and subprocess."""

from __future__ import annotations

import textwrap

import pytest

from agentharness.config import Config
from agentharness.skills.loader import load_skills
from agentharness.skills.runner import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ScriptConfigError,
    ScriptPolicy,
    child_env,
    parse_policy,
    resolve_script,
)
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
