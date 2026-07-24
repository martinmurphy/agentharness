"""Tools that implement Agent Skills' progressive disclosure.

``read_skill``      -> the full SKILL.md body (stage 2 of disclosure)
``read_skill_file`` -> a bundled reference/asset/script file, as text (stage 3)

Both read from a live SkillSet reference, so they always reflect whatever set
the harness currently holds.
"""

from __future__ import annotations

from typing import Any

from agentharness.skills.loader import SkillSet, read_skill_file
from agentharness.tools.registry import Tool


def make_skill_tools(skillset: SkillSet) -> list[Tool]:
    def _read_skill(args: dict[str, Any]) -> str:
        name = args.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("'name' is required")
        skill = skillset.by_name(name)
        if skill is None:
            available = ", ".join(s.name for s in skillset.skills) or "(none)"
            raise ValueError(f"unknown skill {name!r}; available: {available}")
        return skill.body

    def _read_skill_file(args: dict[str, Any]) -> str:
        skill_name = args.get("skill")
        path = args.get("path")
        if not isinstance(skill_name, str) or not skill_name:
            raise ValueError("'skill' is required")
        if not isinstance(path, str) or not path:
            raise ValueError("'path' is required")
        skill = skillset.by_name(skill_name)
        if skill is None:
            raise ValueError(f"unknown skill {skill_name!r}")
        return read_skill_file(skill, path)

    read_skill = Tool(
        name="read_skill",
        description=(
            "Load the full instructions (SKILL.md body) for a named Agent Skill. "
            "Call this when a task matches a skill's description from the catalog."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The skill's name."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=_read_skill,
    )

    read_skill_file_tool = Tool(
        name="read_skill_file",
        description=(
            "Read a bundled file inside a skill's directory (e.g. a file under "
            "references/, assets/, or scripts/), returned as text. Paths are "
            "relative to the skill directory."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "The skill's name."},
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the skill directory.",
                },
            },
            "required": ["skill", "path"],
            "additionalProperties": False,
        },
        handler=_read_skill_file,
    )

    return [read_skill, read_skill_file_tool]
