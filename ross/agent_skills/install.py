"""Install the ROSS cookbook as an agent skill for AI coding agents.

The skill follows the Agent Skills open standard (https://agentskills.io):
a folder with a SKILL.md entry point plus self-contained recipe files, read
on demand by skills-compatible agents such as Claude Code, GitHub Copilot,
Cursor and Codex.

Run ``ross-install-skill`` after installing or upgrading ROSS to copy the
skill into the personal skills directory of each agent found on this
machine, or ``ross-install-skill --project`` to install it into the current
project so it is shared with anyone working on the repository.
"""

import argparse
import shutil
import sys
from importlib import metadata, resources
from pathlib import Path

SKILL_NAME = "ross-cookbook"

VERSION_PLACEHOLDER = "> Skill version: development (repo checkout)"

AGENTS = {
    "claude": {
        "detect": Path("~/.claude"),
        "user": Path("~/.claude/skills"),
        "project": Path(".claude/skills"),
    },
    "copilot": {
        "detect": Path("~/.copilot"),
        "user": Path("~/.copilot/skills"),
        "project": Path(".github/skills"),
    },
    "cursor": {
        "detect": Path("~/.cursor"),
        "user": Path("~/.cursor/skills"),
        "project": Path(".cursor/skills"),
    },
    "codex": {
        "detect": Path("~/.codex"),
        "user": Path("~/.codex/skills"),
        "project": Path(".codex/skills"),
    },
}


def ross_version():
    """Return the installed ross-rotordynamics version.

    Falls back to importing ross when package metadata is unavailable
    (e.g. running from a plain repository checkout).
    """
    try:
        return metadata.version("ross-rotordynamics")
    except metadata.PackageNotFoundError:
        import ross

        return ross.__version__


def install_skill(skills_dir):
    """Copy the skill folder into ``skills_dir`` and stamp the ROSS version.

    Parameters
    ----------
    skills_dir : str or pathlib.Path
        Directory that holds skills (e.g. ``~/.claude/skills``). The skill
        is written to ``<skills_dir>/ross-cookbook``, replacing any
        previous installation.

    Returns
    -------
    target : pathlib.Path
        Path of the installed skill folder.
    """
    skills_dir = Path(skills_dir).expanduser()
    target = skills_dir / SKILL_NAME
    source = resources.files("ross") / "agent_skills" / SKILL_NAME

    if target.exists():
        shutil.rmtree(target)

    with resources.as_file(source) as source_path:
        shutil.copytree(source_path, target)

    version = ross_version()
    skill_md = target / "SKILL.md"
    stamp = (
        f"> Skill version: ROSS {version} — if "
        f'`python -c "import ross; print(ross.__version__)"` reports a '
        f"different version, re-run `ross-install-skill` to refresh this skill."
    )
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8").replace(VERSION_PLACEHOLDER, stamp),
        encoding="utf-8",
    )

    return target


def uninstall_skill(skills_dir):
    """Remove the skill folder from ``skills_dir``.

    Parameters
    ----------
    skills_dir : str or pathlib.Path
        Directory that holds skills (e.g. ``~/.claude/skills``).

    Returns
    -------
    removed : bool
        True if a skill folder was found and removed.
    """
    target = Path(skills_dir).expanduser() / SKILL_NAME
    if target.exists():
        shutil.rmtree(target)
        return True
    return False


def detected_agents():
    """Return the names of agents whose config directory exists in HOME."""
    return [
        name for name, spec in AGENTS.items() if spec["detect"].expanduser().exists()
    ]


def resolve_skills_dirs(agent_names, project, dest):
    """Map the CLI targeting options to a list of skills directories."""
    if dest is not None:
        return [Path(dest)]

    if not agent_names:
        if project:
            return [AGENTS["claude"]["project"]]
        agent_names = detected_agents()

    scope = "project" if project else "user"
    return [AGENTS[name][scope] for name in agent_names]


def main(argv=None):
    """Run the ross-install-skill command line interface."""
    parser = argparse.ArgumentParser(
        prog="ross-install-skill",
        description=(
            "Install the ROSS cookbook as an agent skill for AI coding agents "
            "(Claude Code, GitHub Copilot, Cursor, Codex). With no options, "
            "installs to the personal skills directory of every agent "
            "detected on this machine."
        ),
    )
    parser.add_argument(
        "--agent",
        action="append",
        choices=[*AGENTS, "all"],
        help=(
            "install for a specific agent instead of auto-detecting "
            "(repeat for several agents; 'all' selects every known agent)"
        ),
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help=(
            "install into the current project instead of the home directory "
            "(default .claude/skills/, which Claude Code, Copilot and Cursor "
            "all read; combine with --agent for another agent's project path)"
        ),
    )
    parser.add_argument(
        "--dest",
        help="install into this skills directory, ignoring --agent/--project",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="remove the skill from the selected locations instead",
    )
    args = parser.parse_args(argv)

    agent_names = args.agent or []
    if "all" in agent_names:
        agent_names = list(AGENTS)

    skills_dirs = resolve_skills_dirs(agent_names, args.project, args.dest)
    if not skills_dirs:
        print(
            "No AI coding agents detected (looked for "
            + ", ".join(str(spec["detect"]) for spec in AGENTS.values())
            + ").\nUse --agent, --project or --dest to choose a location."
        )
        return 1

    for skills_dir in skills_dirs:
        if args.uninstall:
            if uninstall_skill(skills_dir):
                print(f"Removed {Path(skills_dir).expanduser() / SKILL_NAME}")
            else:
                print(f"Nothing to remove in {Path(skills_dir).expanduser()}")
        else:
            target = install_skill(skills_dir)
            print(f"Installed {SKILL_NAME} (ROSS {ross_version()}) at {target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
