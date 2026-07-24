"""Tests for skill discovery, validation, and path-confined file reads."""

from __future__ import annotations

import os

import pytest

from agentharness.skills.loader import load_skills, read_skill_file

VALID_FRONTMATTER = """---
name: {name}
description: {desc}
---

# Body

Some instructions.
"""


def _write_skill(root, dirname, *, name=None, desc="A valid description of the skill.",
                 body_after_fence=True):
    name = dirname if name is None else name
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True)
    if body_after_fence:
        text = VALID_FRONTMATTER.format(name=name, desc=desc)
    else:
        text = f"name: {name}\ndescription: {desc}\n"  # no frontmatter fences
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir


def test_loads_valid_skill(tmp_path):
    _write_skill(tmp_path, "greeting-etiquette")
    result = load_skills(tmp_path)
    assert len(result.skills) == 1
    assert not result.errors
    skill = result.skills[0]
    assert skill.name == "greeting-etiquette"
    assert skill.description.startswith("A valid")
    assert "Some instructions." in skill.body
    assert skill.body.startswith("# Body")  # frontmatter stripped


def test_missing_skill_md_ignored(tmp_path):
    (tmp_path / "not-a-skill").mkdir()
    result = load_skills(tmp_path)
    assert not result.skills
    assert not result.errors


def test_nonexistent_dir_is_empty(tmp_path):
    result = load_skills(tmp_path / "does-not-exist")
    assert not result.skills and not result.errors


@pytest.mark.parametrize(
    "name",
    [
        "PDF-Processing",   # uppercase
        "-pdf",             # leading hyphen
        "pdf-",             # trailing hyphen
        "pdf--processing",  # consecutive hyphens
        "pdf_processing",   # underscore
    ],
)
def test_invalid_names_rejected(tmp_path, name):
    # Directory name must match `name`, so name the dir after the (invalid) name.
    _write_skill(tmp_path, name, name=name)
    result = load_skills(tmp_path)
    assert not result.skills
    assert len(result.errors) == 1


def test_name_must_match_directory(tmp_path):
    _write_skill(tmp_path, "dir-name", name="other-name")
    result = load_skills(tmp_path)
    assert not result.skills
    assert len(result.errors) == 1
    assert "directory name" in result.errors[0].reason


def test_missing_description_rejected(tmp_path):
    skill_dir = tmp_path / "no-desc"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\nbody\n", encoding="utf-8")
    result = load_skills(tmp_path)
    assert not result.skills
    assert len(result.errors) == 1
    assert "description" in result.errors[0].reason


def test_missing_frontmatter_rejected(tmp_path):
    _write_skill(tmp_path, "bad", body_after_fence=False)
    result = load_skills(tmp_path)
    assert not result.skills
    assert len(result.errors) == 1


def test_one_bad_skill_does_not_hide_good_ones(tmp_path):
    _write_skill(tmp_path, "good-one")
    _write_skill(tmp_path, "Bad_Name", name="Bad_Name")
    result = load_skills(tmp_path)
    assert [s.name for s in result.skills] == ["good-one"]
    assert len(result.errors) == 1


def test_catalog_prompt(tmp_path):
    _write_skill(tmp_path, "beta-skill", desc="Second skill.")
    _write_skill(tmp_path, "alpha-skill", desc="First skill.")
    result = load_skills(tmp_path)
    catalog = result.catalog_prompt()
    assert "## Available skills" in catalog
    # sorted by name: alpha before beta
    assert catalog.index("alpha-skill") < catalog.index("beta-skill")
    assert "First skill." in catalog


def test_empty_catalog_is_blank(tmp_path):
    result = load_skills(tmp_path)
    assert result.catalog_prompt() == ""


# ---- read_skill_file path confinement --------------------------------------


def test_read_bundled_file(tmp_path):
    skill_dir = _write_skill(tmp_path, "with-ref")
    ref = skill_dir / "references"
    ref.mkdir()
    (ref / "GUIDE.md").write_text("reference content", encoding="utf-8")
    result = load_skills(tmp_path)
    skill = result.by_name("with-ref")
    assert read_skill_file(skill, "references/GUIDE.md") == "reference content"


def test_traversal_rejected(tmp_path):
    _write_skill(tmp_path, "with-ref")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    result = load_skills(tmp_path)
    skill = result.by_name("with-ref")
    with pytest.raises(ValueError, match="escapes"):
        read_skill_file(skill, "../secret.txt")


def test_absolute_path_rejected(tmp_path):
    _write_skill(tmp_path, "with-ref")
    result = load_skills(tmp_path)
    skill = result.by_name("with-ref")
    with pytest.raises(ValueError, match="escapes"):
        read_skill_file(skill, "/etc/hostname")


def test_symlink_escape_rejected(tmp_path):
    skill_dir = _write_skill(tmp_path, "with-ref")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = skill_dir / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    result = load_skills(tmp_path)
    skill = result.by_name("with-ref")
    with pytest.raises(ValueError, match="escapes"):
        read_skill_file(skill, "link.txt")


def test_missing_file_rejected(tmp_path):
    _write_skill(tmp_path, "with-ref")
    result = load_skills(tmp_path)
    skill = result.by_name("with-ref")
    with pytest.raises(ValueError, match="no such file"):
        read_skill_file(skill, "references/nope.md")
