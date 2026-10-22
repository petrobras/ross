"""Build a self-contained HTML review report for a ROSS pull request.

The report is assembled by a small builder script that the reviewer writes
and keeps next to the review, so it can be re-run when the pull request
changes. Code excerpts are read from the pull request's head with git, never
retyped, and figures are Plotly figures produced by the builder itself.

Run the builder from the root of a ROSS checkout: git commands run in the
current directory.

Example
-------
>>> report = Report("#1 Example", head="HEAD", base="HEAD")  # doctest: +SKIP
>>> report.section("Summary", "<p>One paragraph.</p>")  # doctest: +SKIP
>>> report.save("report.html")  # doctest: +SKIP
"""

import ast
import html
import re
import subprocess
from pathlib import Path

STYLE = """
:root{color-scheme:light dark;--bg:#fff;--ink:#12263b;--muted:#5b6b7e;
--line:#dfe8f3;--soft:#f5f8fc;--accent:#1f77b4;--ok:#1d6f1d;--warn:#a85207;
--bad:#96191a}
@media (prefers-color-scheme:dark){:root{--bg:#0b1826;--ink:#dfe8f3;
--muted:#a8b6c4;--line:#24405a;--soft:#0f2233;--accent:#6bb2e0;--ok:#7fd07f;
--warn:#f0a35b;--bad:#f08b8c}}
body{margin:0;color:var(--ink);background:var(--bg);
font:16px/1.6 "IBM Plex Sans",-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:1.8rem;margin:0 0 4px}h2{font-size:1.3rem;margin:36px 0 8px}
.meta{color:var(--muted);font:13px/1.5 "IBM Plex Mono",ui-monospace,monospace}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;
vertical-align:top}th{background:var(--soft)}
.verdict-ok{color:var(--ok)}.verdict-warn{color:var(--warn)}
.verdict-bad{color:var(--bad)}
figure{margin:12px 0}figcaption,.caption{color:var(--muted);font-size:.88rem}
pre{background:var(--soft);border:1px solid var(--line);padding:10px 12px;
overflow-x:auto;font:13px/1.5 "IBM Plex Mono",ui-monospace,monospace}
.lineno{color:var(--muted);user-select:none}
"""

VERDICT_WORDS = (
    ("verdict-bad", ("elsewhere", "fail", "block")),
    ("verdict-warn", ("issue", "missing", "inaccurate", "decide", "question")),
    ("verdict-ok", ("ok", "core", "complete", "pass")),
)


def verdict_class(verdict):
    """Return the CSS class for a verdict, matched by keyword."""
    text = verdict.lower()
    for css, words in VERDICT_WORDS:
        if any(word in text for word in words):
            return css
    return ""


def inline(text):
    """Escape text and apply inline Markdown: code, bold, italics and links."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<em>\1</em>", text)
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)


def table_cells(line):
    """Split a pipe-table row into cells.

    A pipe escaped as ``\\|`` (the escape is removed, as on GitHub) or inside a
    code span belongs to the cell.
    """
    cells, cell, in_code, i = [], "", False, 0
    row = line.strip()
    row = row[1:] if row.startswith("|") else row
    while i < len(row):
        char = row[i]
        if char == "\\" and row[i + 1 : i + 2] == "|":
            cell += "|"
            i += 2
            continue
        if char == "`":
            in_code = not in_code
        if char == "|" and not in_code:
            cells.append(cell.strip())
            cell = ""
        else:
            cell += char
        i += 1
    if cell.strip():
        cells.append(cell.strip())
    return cells


def markdown(text):
    """Convert the Markdown of a review card to HTML.

    Handles headings, paragraphs, bulleted and numbered lists (with indented
    continuation lines), pipe tables and inline code, bold, italics and links.
    """
    blocks, lines = [], text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        heading = re.match(r"(#{1,4})\s+(.*)", line)
        if heading:
            level = min(len(heading.group(1)) + 1, 4)
            blocks.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = table_cells(lines[i])
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
                for row in rows[1:]
            )
            blocks.append(f"<table><tr>{head}</tr>{body}</table>")
        elif re.match(r"\s*([-*]|\d+\.)\s+", line):
            tag = "ol" if re.match(r"\s*\d+\.", line) else "ul"
            items = []
            while i < len(lines) and lines[i].strip():
                item = re.match(r"\s*([-*]|\d+\.)\s+(.*)", lines[i])
                if item and not lines[i].startswith("   "):
                    items.append(item.group(2))
                elif items:
                    items[-1] += " " + lines[i].strip()
                i += 1
            blocks.append(
                f"<{tag}>"
                + "".join(f"<li>{inline(t)}</li>" for t in items)
                + f"</{tag}>"
            )
        else:
            paragraph = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(r"(#{1,4}\s|\s*\||\s*([-*]|\d+\.)\s)", lines[i])
            ):
                paragraph.append(lines[i].strip())
                i += 1
            blocks.append(f"<p>{inline(' '.join(paragraph))}</p>")
    return "\n".join(blocks)


def symbol_lines(source, symbol):
    """Return the first and last line of a dotted class or function name."""
    nodes = ast.parse(source).body
    found = None
    for part in symbol.split("."):
        found = next(
            (
                n
                for n in nodes
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == part
            ),
            None,
        )
        if found is None:
            raise ValueError(f"{symbol} not found")
        nodes = found.body
    start = min([found.lineno] + [d.lineno for d in found.decorator_list])
    return start, found.end_lineno


def git(*args):
    """Run a git command in the current directory and return its output."""
    result = subprocess.run(["git", *args], capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace")


class Report:
    """Collect the sections of a review report and write them as HTML.

    Parameters
    ----------
    title : str
        Report title, e.g. ``"#1296 Motor element"``.
    head : str
        Git revision of the pull request's head; code excerpts are read from
        it.
    base : str
        Git revision the pull request is compared with.
    """

    def __init__(self, title, head, base):
        self.title = title
        self.head = head
        self.head_sha = git("rev-parse", "--short", head).strip()
        self.base_sha = git("rev-parse", "--short", base).strip()
        self.parts = []
        self.plotly_included = False

    def section(self, heading, body_html):
        """Add a section with a heading and an HTML body written by the reviewer."""
        self.parts.append(f"<h2>{html.escape(heading)}</h2>\n{body_html}")

    def verdicts(self, rows):
        """Add the verdict table.

        Parameters
        ----------
        rows : list of tuple
            ``(check, verdict, evidence)`` triples, e.g.
            ``("Scope", "core", "electrical machines are core")``.
        """
        cells = []
        for check, verdict, evidence in rows:
            css = verdict_class(verdict)
            cells.append(
                f"<tr><td>{html.escape(check)}</td>"
                f'<td class="{css}">{html.escape(verdict)}</td>'
                f"<td>{html.escape(evidence)}</td></tr>"
            )
        self.parts.append(
            "<h2>Verdicts</h2><table><tr><th>Check</th><th>Verdict</th>"
            f"<th>Evidence</th></tr>{''.join(cells)}</table>"
        )

    def table(self, header, rows, caption=""):
        """Add a table of values computed by the builder."""
        head = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        note = f'<p class="caption">{html.escape(caption)}</p>' if caption else ""
        self.parts.append(f"<table><tr>{head}</tr>{body}</table>{note}")

    def heading(self, text):
        """Add a section heading with no body."""
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def markdown(self, text):
        """Add Markdown written by the reviewer, e.g. the review card."""
        self.parts.append(markdown(text))

    def code(self, path, start=None, end=None, caption="", symbol=None):
        """Add an excerpt of ``path`` read from the head.

        Parameters
        ----------
        path : str
            File path in the repository.
        start, end : int, optional
            First and last line (1-based, inclusive).
        caption : str, optional
            Text shown under the excerpt.
        symbol : str, optional
            Dotted class or function name (e.g. ``"MultiRotor.C_mesh"``) to
            excerpt instead of a line range.
        """
        source = git("show", f"{self.head}:{path}")
        if symbol is not None:
            start, end = symbol_lines(source, symbol)
        lines = source.splitlines()[start - 1 : end]
        width = len(str(end))
        numbered = "\n".join(
            f'<span class="lineno">{n:>{width}}</span>  {html.escape(line)}'
            for n, line in enumerate(lines, start)
        )
        label = f"{path}:{start}-{end} @ {self.head_sha}"
        text = f"{html.escape(caption)} · " if caption else ""
        self.parts.append(
            f"<figure><pre>{numbered}</pre><figcaption>{text}<code>{html.escape(label)}"
            "</code></figcaption></figure>"
        )

    def figure(self, fig, caption=""):
        """Add a Plotly figure; plotly.js is embedded once so the file works offline."""
        div = fig.to_html(
            full_html=False,
            include_plotlyjs=not self.plotly_included,
            default_width="100%",
        )
        self.plotly_included = True
        note = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
        self.parts.append(f"<figure>{div}{note}</figure>")

    def html(self):
        """Return the complete HTML document."""
        meta = (
            f"head {self.head_sha} ({html.escape(self.head)}) · base {self.base_sha} · "
            "code excerpts and figures generated from the head"
        )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(self.title)}</title><style>{STYLE}</style></head>"
            f'<body><main><h1>{html.escape(self.title)}</h1><p class="meta">{meta}</p>'
            + "\n".join(self.parts)
            + "</main></body></html>"
        )

    def save(self, path):
        """Write the report to ``path`` and return the path."""
        path = Path(path)
        path.write_text(self.html(), encoding="utf-8")
        return path
