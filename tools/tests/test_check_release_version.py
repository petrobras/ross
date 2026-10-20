import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from check_release_version import check, main  # noqa: E402

WORKFLOW = TOOLS.parent / ".github/workflows/publish-to-pypi.yml"


def make_dist(path, version):
    path.mkdir(exist_ok=True)
    (path / f"ross_rotordynamics-{version}-py3-none-any.whl").touch()
    (path / f"ross_rotordynamics-{version}.tar.gz").touch()
    return path


@pytest.mark.parametrize(
    "tag, version",
    [
        ("v3.1.0", "3.1.0"),
        ("v3.1.0rc1", "3.1.0rc1"),
        ("v3.0.1", "3.0.1"),
        ("v3.1.0.post1", "3.1.0.post1"),
    ],
)
def test_matching_tag_passes(tmp_path, tag, version):
    assert check(tag, make_dist(tmp_path / "dist", version)) == []


@pytest.mark.parametrize(
    "tag, version",
    [
        ("v3.1.0", "3.1.0.dev0"),
        ("v3.1.0", "3.1.0rc1"),
        ("v3.1.0rc1", "3.1.0"),
        ("v3.1.1", "3.1.0"),
    ],
)
def test_version_mismatch_fails(tmp_path, tag, version):
    errors = check(tag, make_dist(tmp_path / "dist", version))
    assert len(errors) == 2
    assert all(version in error and tag in error for error in errors)


@pytest.mark.parametrize(
    "tag, version, canonical",
    [
        ("v3.1.0-rc1", "3.1.0rc1", "v3.1.0rc1"),
        ("v3.1.0.rc1", "3.1.0rc1", "v3.1.0rc1"),
        ("v3.1", "3.1.0", "v3.1.0"),
    ],
)
def test_equal_but_non_canonical_tag_fails(tmp_path, tag, version, canonical):
    errors = check(tag, make_dist(tmp_path / "dist", version))
    assert len(errors) == 1
    assert canonical in errors[0]


@pytest.mark.parametrize("tag", ["3.1.0", "v3.x", "vnext"])
def test_malformed_tag_fails(tmp_path, tag):
    errors = check(tag, make_dist(tmp_path / "dist", "3.1.0"))
    assert len(errors) == 1
    assert tag in errors[0]


def test_empty_dist_fails(tmp_path):
    (tmp_path / "dist").mkdir()
    errors = check("v3.1.0", tmp_path / "dist")
    assert len(errors) == 1
    assert "no wheel or sdist" in errors[0]


def test_mixed_versions_fail(tmp_path):
    dist = make_dist(tmp_path / "dist", "3.1.0")
    (dist / "ross_rotordynamics-3.1.0.dev0-py3-none-any.whl").touch()
    errors = check("v3.1.0", dist)
    assert len(errors) == 1
    assert "3.1.0.dev0" in errors[0]


def test_command_line_exit_status(tmp_path, capsys):
    dist = make_dist(tmp_path / "dist", "3.1.0.dev0")
    assert main(["v3.1.0.dev0", str(dist)]) == 0
    assert main(["v3.1.0", str(dist)]) == 1
    assert "3.1.0.dev0" in capsys.readouterr().err


def test_script_runs_standalone(tmp_path):
    dist = make_dist(tmp_path / "dist", "3.1.0.dev0")
    result = subprocess.run(
        [sys.executable, TOOLS / "check_release_version.py", "v3.1.0", dist],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "3.1.0.dev0" in result.stderr


def test_release_workflow_checks_the_tag_between_build_and_upload():
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()

    def line_of(text):
        return next(i for i, line in enumerate(lines) if text in line)

    build = line_of("python -m build")
    guard = line_of("check_release_version.py")
    upload = line_of("gh-action-pypi-publish")
    assert build < guard < upload
    assert lines[guard].strip().startswith("run:")
    assert "$GITHUB_REF_NAME" in lines[guard]
    assert "dist/" in lines[guard]
