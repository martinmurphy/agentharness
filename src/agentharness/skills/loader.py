"""Discover and validate Agent Skills from a directory.

A skill is a subdirectory containing a ``SKILL.md`` file with YAML frontmatter.
Discovery walks one level deep. Skills that fail validation are collected as
``SkillLoadError`` entries rather than aborting the whole scan, so one bad skill
never hides the good ones.

See https://agentskills.io/specification for the format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentharness.skills.model import Skill

# name: 1-64 chars, lowercase alphanumeric groups joined by single hyphens,
# no leading/trailing/consecutive hyphens. This one regex covers all four rules.
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NAME_MAX = 64
_DESC_MAX = 1024
_COMPAT_MAX = 500

# A file this large under a skill is almost certainly not meant for the model.
MAX_FILE_BYTES = 256 * 1024

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass(frozen=True)
class SkillLoadError:
    """A directory that looked like a skill but failed to load or validate."""

    path: Path
    reason: str


@dataclass(frozen=True)
class SkillSet:
    """The result of scanning a skills directory."""

    skills: tuple[Skill, ...]
    errors: tuple[SkillLoadError, ...]
    root: Path

    def by_name(self, name: str) -> Skill | None:
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def catalog_prompt(self) -> str:
        """Render the discovery block appended to the system prompt.

        Returns an empty string when no skills are available so callers can
        conditionally append it.
        """
        if not self.skills:
            return ""
        lines = [
            "## Available skills",
            (
                "Use the `read_skill` tool to load a skill's full instructions when a "
                "task matches its description. Use `read_skill_file` to read a skill's "
                "bundled reference files."
            ),
            "",
        ]
        lines.extend(s.catalog_line() for s in sorted(self.skills, key=lambda s: s.name))
        return "\n".join(lines)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("SKILL.md is missing YAML frontmatter delimited by '---'")
    raw, body = match.group(1), match.group(2)
    meta = yaml.safe_load(raw)
    if not isinstance(meta, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    return meta, body.strip()


def _validate_and_build(directory: Path, meta: dict, body: str) -> Skill:
    name = meta.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("frontmatter is missing a non-empty 'name'")
    if len(name) > _NAME_MAX or not _NAME_RE.match(name):
        raise ValueError(
            f"invalid name {name!r}: must be 1-{_NAME_MAX} lowercase alphanumeric "
            "characters and single hyphens, with no leading/trailing/consecutive hyphens"
        )
    if name != directory.name:
        raise ValueError(
            f"name {name!r} must match the parent directory name {directory.name!r}"
        )

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("frontmatter is missing a non-empty 'description'")
    if len(description) > _DESC_MAX:
        raise ValueError(f"description exceeds {_DESC_MAX} characters")

    compatibility = meta.get("compatibility")
    if compatibility is not None and (
        not isinstance(compatibility, str) or len(compatibility) > _COMPAT_MAX
    ):
        raise ValueError(f"compatibility must be a string of at most {_COMPAT_MAX} characters")

    raw_metadata = meta.get("metadata") or {}
    if not isinstance(raw_metadata, dict):
        raise ValueError("metadata must be a mapping")
    metadata = {str(k): str(v) for k, v in raw_metadata.items()}

    allowed_tools = meta.get("allowed-tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        raise ValueError("allowed-tools must be a space-separated string")

    license_ = meta.get("license")

    return Skill(
        name=name,
        description=description.strip(),
        body=body,
        path=directory,
        license=str(license_) if license_ is not None else None,
        compatibility=compatibility,
        allowed_tools=allowed_tools,
        metadata=metadata,
    )


def load_skills(skills_dir: str | Path) -> SkillSet:
    """Scan ``skills_dir`` one level deep for skill directories."""
    root = Path(skills_dir).expanduser()
    skills: list[Skill] = []
    errors: list[SkillLoadError] = []

    if not root.is_dir():
        return SkillSet(skills=(), errors=(), root=root)

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
            meta, body = _split_frontmatter(text)
            skills.append(_validate_and_build(entry, meta, body))
        except (ValueError, yaml.YAMLError, UnicodeDecodeError) as exc:
            errors.append(SkillLoadError(path=entry, reason=str(exc)))

    return SkillSet(skills=tuple(skills), errors=tuple(errors), root=root)


def read_skill_file(skill: Skill, rel_path: str) -> str:
    """Read a bundled file under a skill directory, confined to that directory.

    ``rel_path`` is untrusted (it comes from the model). We resolve it and require
    the result to stay within the skill directory, which rejects ``..`` traversal,
    absolute paths, and symlink escapes in a single check.
    """
    base = skill.path.resolve()
    target = (skill.path / rel_path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"path {rel_path!r} escapes the skill directory")
    if not target.is_file():
        raise ValueError(f"no such file: {rel_path}")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file too large (> {MAX_FILE_BYTES} bytes): {rel_path}")
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {rel_path}") from exc
