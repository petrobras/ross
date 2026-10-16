import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pr_facts  # noqa: E402

ELEMENT_BASE = """
from abc import ABC, abstractmethod


class Element(ABC):
    @abstractmethod
    def M(self):
        pass
"""

ROTOR_BASE = '''
__all__ = ["Rotor", "rotor_example"]


class Rotor:
    """Rotor."""

    def __init__(self, shaft_elements, tag=None):
        self.shaft_elements = shaft_elements
        self.tag = tag

    def run_modal(self, speed):
        """Run modal analysis."""
        return speed


def rotor_example():
    """Return a rotor."""
    return Rotor([])
'''

RESULTS_BASE = '''
class Results:
    """Results."""
'''


def git(path, *args):
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def write(path, files):
    for name, content in files.items():
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ross@example.com")
    git(tmp_path, "config", "user.name", "ROSS")
    write(
        tmp_path,
        {
            "requirements.txt": "numpy>=2.2\nscipy\n",
            "AGENTS.md": "run_modal\n",
            "ross/agent_skills/ross/SKILL.md": "run_modal\n",
            "ross/ross_2to3/renames.py": "RENAMES = {}\n",
            "ross/element.py": ELEMENT_BASE,
            "ross/rotor_assembly.py": ROTOR_BASE,
            "ross/results.py": RESULTS_BASE,
            "ross/tests/test_rotor.py": """
                from numpy.testing import assert_allclose


                def test_modal():
                    assert_allclose(1.0, 1.0, rtol=1e-6)
                    assert 2 == 2
            """,
        },
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "checkout", "-q", "-b", "feature")
    return tmp_path


def commit(path, files):
    write(path, files)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "change")


def messages(facts, group):
    return [f["message"] for f in facts["flags"] if f["group"] == group]


def test_loosened_tolerance_is_flagged():
    base = pr_facts.describe_tests(
        "def test_a():\n    assert_allclose(x, 1.0, rtol=1e-6)\n", "t.py"
    )
    head = pr_facts.describe_tests(
        "def test_a():\n    assert_allclose(x, 1.0, rtol=1e-3)\n", "t.py"
    )
    findings = pr_facts.compare_tests(base, head)
    assert [f["kind"] for f in findings] == ["tolerance changed"]
    assert "1e-06 -> 0.001 (loosened)" in findings[0]["detail"]


def test_fewer_decimals_is_a_loosened_tolerance():
    base = pr_facts.describe_tests(
        "def test_a():\n    assert_almost_equal(x, 1.0, 6)\n", "t.py"
    )
    head = pr_facts.describe_tests(
        "def test_a():\n    assert_almost_equal(x, 1.0, decimal=2)\n", "t.py"
    )
    detail = pr_facts.compare_tests(base, head)[0]["detail"]
    assert "decimal: 6 -> 2 (loosened)" in detail


def test_renamed_attribute_with_same_tolerance_is_not_flagged():
    base = pr_facts.describe_tests(
        "def test_a():\n    assert_allclose(r.K_coupling, m, rtol=1e-6, atol=1e-5)\n",
        "t.py",
    )
    head = pr_facts.describe_tests(
        "def test_a():\n"
        "    assert_allclose(r.coupling_matrix, m, rtol=1e-6, atol=1e-5)\n",
        "t.py",
    )
    findings = pr_facts.compare_tests(base, head)
    assert [f["kind"] for f in findings] == ["changed"]


def test_weakened_test_is_flagged():
    base = pr_facts.describe_tests(
        "def test_a():\n    assert f(1) == 2.5\n    assert g() == 3\n", "t.py"
    )
    head = pr_facts.describe_tests(
        "@pytest.mark.xfail\ndef test_a():\n    assert f(1) == 2.7\n", "t.py"
    )
    kinds = {f["kind"] for f in pr_facts.compare_tests(base, head)}
    assert kinds == {
        "fewer assertions",
        "skip or xfail added",
        "expected values changed",
    }


def test_new_expected_values_alone_are_not_flagged():
    base = pr_facts.describe_tests("def test_a():\n    assert f(1) == 2\n", "t.py")
    head = pr_facts.describe_tests(
        "def test_a():\n    assert f(1) == 2\n    assert f(3) == 4\n", "t.py"
    )
    findings = pr_facts.compare_tests(base, head)
    assert [f["kind"] for f in findings] == ["changed"]


def test_moved_test_is_matched_by_body():
    body = "    rotor = build()\n    assert rotor.nodes == [0, 1, 2]\n"
    base = pr_facts.describe_tests(f"def test_old():\n{body}", "a.py")
    head = pr_facts.describe_tests(f"def test_new():\n{body}", "b.py")
    findings = pr_facts.compare_tests(base, head)
    assert [f["kind"] for f in findings] == ["moved"]
    assert findings[0]["new"] == "b.py::test_new"


def test_deleted_test_without_a_match_is_flagged():
    base = pr_facts.describe_tests("def test_a():\n    assert f(1) == 2\n", "t.py")
    findings = pr_facts.compare_tests(base, {})
    assert [f["kind"] for f in findings] == ["removed"]


def test_parameter_changes_report_defaults_and_additions():
    notes = pr_facts.parameter_changes(
        [["a", None], ["damping_ratio", "0.07"]],
        [["a", None], ["damping_ratio", "0.0"], ["b", None]],
    )
    assert notes == ["default of damping_ratio: 0.07 -> 0.0", "added b (required)"]


def test_moved_class_members_are_not_reported_as_removed():
    base_api = pr_facts.public_api(
        "class A:\n    def __init__(self):\n        self.x = 1\n"
        "    def run(self, s=1):\n        pass\n",
        "pkg.old",
    )
    head_api = pr_facts.public_api(
        "class A:\n    def __init__(self):\n        self.x = 1\n"
        "    def run(self, s=2):\n        pass\n",
        "pkg.new",
    )
    api = pr_facts.compare_api(base_api, head_api)
    assert api["moved"] == {"pkg.old.A": "pkg.new.A"}
    assert api["removed"] == []
    assert api["added"] == []
    assert api["changed"] == [
        {"name": "pkg.new.A.run", "changes": ["default of s: 1 -> 2"]}
    ]


def test_architecture_flags(repo):
    commit(
        repo,
        {
            "ross/motor.py": '''
                from ross.element import Element


                class MotorElement(Element):
                    """Motor."""

                    def M(self):
                        """Mass."""

                    def run_start(self, speed: float):
                        """Start."""


                class MotorResults:
                    """Results without the base class."""
            ''',
        },
    )
    flags = messages(pr_facts.collect("main", "HEAD", repo), "architecture")
    assert any("run_start: analyses are run_* methods" in f for f in flags)
    assert any("does not define K, C, G, dof_mapping" in f for f in flags)
    assert any("MotorResults does not subclass Results" in f for f in flags)
    assert any("run_start: type annotations" in f for f in flags)


def test_new_rotor_analysis_must_be_documented(repo):
    commit(
        repo,
        {
            "ross/rotor_assembly.py": ROTOR_BASE.replace(
                '"Rotor",', '"Rotor", "CoAxialRotor",'
            )
            + '''

class CoAxialRotor(Rotor):
    """Two shafts."""

    def run_spin(self):
        """Spin."""
''',
        },
    )
    flags = messages(pr_facts.collect("main", "HEAD", repo), "architecture")
    assert any(
        "run_spin: add it to the analysis table in AGENTS.md" in f for f in flags
    )
    assert any("run_spin: add it to ross/agent_skills" in f for f in flags)
    assert not any("analyses are run_* methods" in f for f in flags)


def test_api_changes_and_inheritance(repo):
    commit(
        repo,
        {
            "ross/rotor_assembly.py": '''
                __all__ = ["Rotor", "CoAxialRotor"]


                class Base:
                    def __init__(self, shaft_elements, tag=None):
                        self.shaft_elements = shaft_elements
                        self.tag = tag


                class Rotor(Base):
                    """Rotor."""

                    def run_modal(self, speed=0):
                        """Run modal analysis."""
            ''',
        },
    )
    facts = pr_facts.collect("main", "HEAD", repo)
    api = facts["api"]
    assert "ross.rotor_assembly.rotor_example" in api["removed"]
    assert "ross.rotor_assembly.Rotor.tag" in api["inherited"]
    assert {
        "name": "ross.rotor_assembly.Rotor.run_modal",
        "changes": ["default of speed: None -> 0"],
    } in api["changed"]


def test_scope_and_writing_signals(repo):
    commit(
        repo,
        {
            "requirements.txt": "numpy>=2.2\nscipy\ntsdownsample\n",
            "ross/motors/__init__.py": "",
            "ross/motors/drive.py": '''
                def simulate():
                    pass


                def describe():
                    """A comprehensive description \u2728."""
            ''',
        },
    )
    facts = pr_facts.collect("main", "HEAD", repo)
    assert facts["scope"]["new_subpackages"] == ["ross/motors"]
    assert facts["scope"]["new_dependencies"] == ["tsdownsample"]
    assert "ross.motors.drive.simulate" in facts["writing"]["missing_docstrings"]
    assert any("comprehensive" in item for item in facts["writing"]["filler"])
    assert messages(facts, "writing")


def test_test_configuration_and_data_changes(repo):
    commit(
        repo,
        {
            "ross/tests/data/reference.csv": "1,2\n",
        },
    )
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "feature")
    git(repo, "checkout", "-q", "-b", "second")
    commit(
        repo,
        {
            "ross/tests/data/reference.csv": "1,3\n",
            "ross/conftest.py": "collect_ignore = ['tests']\n",
            "ross/rotor_assembly.py": ROTOR_BASE
            + "\n\ndef helper():\n    try:\n        pass\n    except Exception:\n"
            "        pass\n",
        },
    )
    flags = messages(pr_facts.collect("main", "HEAD", repo), "tests")
    assert any(
        "Reference data changed: ross/tests/data/reference.csv" in f for f in flags
    )
    assert any("Test configuration changed: ross/conftest.py" in f for f in flags)
    assert any("silent except" in f for f in flags)


def test_working_tree_includes_untracked_files(repo):
    write(repo, {"ross/new_module.py": "def run_alone():\n    pass\n"})
    facts = pr_facts.collect("main", None, repo)
    assert facts["head"] == "working tree"
    assert "ross.new_module.run_alone" in facts["api"]["added"]
    assert "## Flags" in pr_facts.render(facts)


def test_size_flags(repo):
    commit(
        repo,
        {
            "ross/interface/app.py": "x = 1\n",
            "ross/extra.py": "y = 2\n",
        },
    )
    flags = messages(pr_facts.collect("main", "HEAD", repo), "size")
    assert flags == ["Mixes core-library and interface changes; split them."]


def test_edited_fixture_is_flagged(repo):
    commit(
        repo,
        {
            "ross/tests/test_rotor.py": """
                import pytest
                from numpy.testing import assert_allclose


                @pytest.fixture
                def rotor():
                    return build(damping=0.07)


                def test_modal():
                    assert_allclose(1.0, 1.0, rtol=1e-6)
                    assert 2 == 2
            """,
        },
    )
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "feature")
    git(repo, "checkout", "-q", "-b", "second")
    commit(
        repo,
        {
            "ross/tests/test_rotor.py": """
                import pytest
                from numpy.testing import assert_allclose


                @pytest.fixture
                def rotor():
                    return build(damping=0.07, mesh_damping_ratio=0.07)


                def test_modal():
                    assert_allclose(1.0, 1.0, rtol=1e-6)
                    assert 2 == 2
            """,
        },
    )
    flags = messages(pr_facts.collect("main", "HEAD", repo), "tests")
    assert flags == [
        "ross/tests/test_rotor.py::rotor: existing fixture, helper or module-level "
        "value changed; the tests that use it may now check something else."
    ]


def test_constitution_changes_note_whitespace_only(repo):
    commit(
        repo,
        {
            "ross/element.py": ELEMENT_BASE + "\n\n",
            "AGENTS.md": "run_modal\nnew rule\n",
        },
    )
    flags = messages(pr_facts.collect("main", "HEAD", repo), "constitution")
    assert flags == [
        "AGENTS.md changed.",
        "ross/element.py changed (whitespace only).",
    ]


def test_stale_notebook_cells():
    def notebook(source, outputs):
        cell = {"cell_type": "code", "source": source, "outputs": outputs}
        return json.dumps({"cells": [{"cell_type": "markdown", "source": "x"}, cell]})

    output = [{"output_type": "stream", "text": "1.0"}]
    before = notebook("run(damping=0)", output)
    assert pr_facts.stale_notebook_cells(
        before, notebook("run(damping=0.07)", output)
    ) == [1]
    rerun = [{"output_type": "stream", "text": "0.9"}]
    assert (
        pr_facts.stale_notebook_cells(before, notebook("run(damping=0.07)", rerun))
        == []
    )


def test_repeated_text_is_reported(repo):
    sentence = "A value of 0.07 is suggested for spur gears in most cases."
    commit(
        repo,
        {
            "ross/gears.py": "".join(
                f'def f{i}():\n    """Do {i}.\n\n    {sentence}\n    """\n\n\n'
                for i in range(3)
            ),
        },
    )
    facts = pr_facts.collect("main", "HEAD", repo)
    assert facts["writing"]["repeated"] == [f"{sentence.lower()!r} (x3)"]


def test_output_is_utf8_on_any_console(repo, monkeypatch):
    commit(repo, {"ross/extra.py": 'def f():\n    """Sparkle \u2728."""\n'})
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.chdir(repo)
    pr_facts.main(["--base", "main", "--head", "HEAD"])
    console.flush()
    assert "Emoji added: ross/extra.py" in console.buffer.getvalue().decode("utf-8")
