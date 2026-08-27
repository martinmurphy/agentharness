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
