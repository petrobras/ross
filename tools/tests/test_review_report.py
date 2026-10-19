import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / ".agents/skills/ross-pr-review/scripts"
sys.path.insert(0, str(SCRIPTS))

from report import STYLE, Report, markdown, verdict_class  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "ross@example.com"),
        ("config", "user.name", "ROSS"),
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    (tmp_path / "element.py").write_text(
        "def K(self):\n    return a < b\n\n\ndef M(self):\n    pass\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_report_extracts_code_from_the_head(repo):
    report = Report("#1 <Example>", head="HEAD", base="HEAD")
    report.code("element.py", 1, 2, "Stiffness")
    page = report.html()
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert "&lt;Example&gt;" in page
    assert "return a &lt; b" in page
    assert "def M" not in page
    assert f"element.py:1-2 @ {sha}" in page


def test_report_sections_and_offline_figures(repo):
    report = Report("#1 Example", head="HEAD", base="HEAD")
    report.verdicts([("Scope", "core", "rotor models are core")])
    report.table(["speed", "error"], [[100, 0.01]], "Against the reference")
    for _ in range(2):
        report.figure(go.Figure(go.Scatter(x=[0, 1], y=[1, 2])), "Response")
    path = report.save(repo / "report.html")
    page = path.read_text(encoding="utf-8")
    assert 'class="verdict-ok">core' in page
    assert "<td>0.01</td>" in page
    assert page.count("Plotly.newPlot") == 2
    assert page.count("plotly.js v") == 1


def test_report_excerpts_a_symbol(repo):
    (repo / "rotor.py").write_text(
        "class Rotor:\n    @property\n    def C(self):\n        return 1\n\n"
        "    def K(self):\n        return 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "rotor"], check=True)
    report = Report("#1", head="HEAD", base="HEAD")
    report.code("rotor.py", symbol="Rotor.C")
    page = report.html()
    assert "@property" in page
    assert "return 1" in page
    assert "return 2" not in page
    assert "rotor.py:2-4" in page


def test_markdown_renders_a_review_card():
    page = markdown(
        "## Review: #1 Title\n\n"
        "| Check | Verdict |\n|---|---|\n| Scope | Core |\n\n"
        "### Findings\n\n"
        "1. **[tests] `a.py:3`**: the tolerance was loosened.\n"
        "   Reproduce: `pytest a.py` fails.\n"
        "2. Second finding with a [link](https://example.com).\n\n"
        "Plain <paragraph>.\n"
    )
    assert "<h3>Review: #1 Title</h3>" in page
    assert "<th>Check</th>" in page and "<td>Core</td>" in page
    assert page.count("<li>") == 2
    assert "<strong>[tests] <code>a.py:3</code></strong>" in page
    assert "loosened. Reproduce: <code>pytest a.py</code> fails." in page
    assert '<a href="https://example.com">link</a>' in page
    assert "<p>Plain &lt;paragraph&gt;.</p>" in page


def test_verdicts_are_colored_by_keyword():
    assert verdict_class("Core") == "verdict-ok"
    assert verdict_class("Missing or inaccurate") == "verdict-warn"
    assert verdict_class("Maintainer decides") == "verdict-warn"
    assert verdict_class("Belongs elsewhere") == "verdict-bad"


def test_table_cells_keep_escaped_and_code_pipes():
    page = markdown(
        "| Check | Evidence |\n|---|---|\n"
        '| Architecture | `git grep -i "a\\|b"` and `x | y` |\n| Writing | |'
    )
    assert page.count("<td>") == 4
    assert "<code>git grep -i &quot;a|b&quot;</code> and <code>x | y</code>" in page
    assert "<td>Writing</td><td></td>" in page


def test_dark_theme_sets_the_background_with_the_ink():
    light, dark = STYLE.split("@media (prefers-color-scheme:dark)")
    assert "--bg:#fff" in light and "--bg:#0b1826" in dark
    assert "background:var(--bg)" in STYLE
    assert "background:#" not in STYLE.split("body{", 1)[1]
