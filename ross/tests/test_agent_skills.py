import re
from importlib import resources
from pathlib import Path

import pytest

from ross.agent_skills import install_skill, uninstall_skill
from ross.agent_skills.install import (
    AGENTS,
    SKILL_NAME,
    VERSION_PLACEHOLDER,
    main,
    resolve_skills_dirs,
    ross_version,
)


@pytest.fixture
def skill_source():
    return Path(resources.files("ross") / "agent_skills" / SKILL_NAME)


def test_skill_md_frontmatter(skill_source):
    text = (skill_source / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert frontmatter is not None
    assert f"name: {SKILL_NAME}" in frontmatter.group(1)
    assert "description:" in frontmatter.group(1)
    assert VERSION_PLACEHOLDER in text


def test_skill_is_self_contained(skill_source):
    recipe_files = {p.name for p in skill_source.glob("*.md")} - {"SKILL.md"}
    indexed = set()
    for md_file in skill_source.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+\.md)\)", md_file.read_text()):
            assert "/" not in link, f"{md_file.name} links outside the skill: {link}"
            assert (skill_source / link).exists(), f"{md_file.name}: broken link {link}"
            if md_file.name == "SKILL.md":
                indexed.add(link)
    assert indexed == recipe_files


def test_install_stamps_version(tmp_path):
    target = install_skill(tmp_path)
    assert target == tmp_path / SKILL_NAME
    text = (target / "SKILL.md").read_text(encoding="utf-8")
    assert VERSION_PLACEHOLDER not in text
    assert f"ROSS {ross_version()}" in text
    assert (target / "modal_analysis.md").exists()


def test_reinstall_removes_stale_files(tmp_path):
    stale = install_skill(tmp_path) / "old_recipe.md"
    stale.write_text("stale")
    install_skill(tmp_path)
    assert not stale.exists()


def test_uninstall(tmp_path):
    install_skill(tmp_path)
    assert uninstall_skill(tmp_path) is True
    assert not (tmp_path / SKILL_NAME).exists()
    assert uninstall_skill(tmp_path) is False


def test_resolve_skills_dirs():
    assert resolve_skills_dirs([], False, "some/dir") == [Path("some/dir")]
    assert resolve_skills_dirs(["claude"], False, None) == [Path("~/.claude/skills")]
    assert resolve_skills_dirs(["copilot"], True, None) == [Path(".github/skills")]
    assert resolve_skills_dirs([], True, None) == [Path(".claude/skills")]


def test_main_with_dest(tmp_path, capsys):
    assert main(["--dest", str(tmp_path)]) == 0
    assert (tmp_path / SKILL_NAME / "SKILL.md").exists()
    assert str(tmp_path / SKILL_NAME) in capsys.readouterr().out
    assert main(["--dest", str(tmp_path), "--uninstall"]) == 0
    assert not (tmp_path / SKILL_NAME).exists()


def test_agents_registry():
    for spec in AGENTS.values():
        assert set(spec) == {"detect", "user", "project"}
