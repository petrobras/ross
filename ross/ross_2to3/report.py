"""Findings collected while converting files from ROSS 2 to ROSS 3."""

import re
from collections import namedtuple

Finding = namedtuple("Finding", "path location level message")

CHANGED = "changed"
CHECK = "check"
MANUAL = "manual"
VERIFIED = "verified"
ERROR = "error"
SKIPPED = "skipped"

LEVELS = (CHANGED, CHECK, MANUAL, VERIFIED, ERROR, SKIPPED)

LEVEL_LABELS = {
    CHANGED: "converted automatically",
    CHECK: "converted, please verify",
    MANUAL: "needs manual conversion",
    VERIFIED: "verified",
    ERROR: "error",
    SKIPPED: "skipped",
}


def _location_key(location):
    return tuple(int(number) for number in re.findall(r"\d+", location))


class Report:
    """Collect the findings of a conversion run and render them as text.

    Examples
    --------
    >>> report = Report()
    >>> report.add("rotor.toml", "BearingElement_b0", CHANGED, "frequency -> speed")
    >>> report.count(CHANGED)
    1
    >>> print(report.render())  # doctest: +ELLIPSIS
    rotor.toml
      [converted automatically] BearingElement_b0: frequency -> speed
    ...
    """

    def __init__(self):
        self.findings = []

    def add(self, path, location, level, message):
        """Record one finding.

        Parameters
        ----------
        path : str or pathlib.Path
            File the finding refers to.
        location : str
            Line number, notebook cell or file section.
        level : str
            One of ``LEVELS``.
        message : str
            What was done or what is left to do.
        """
        if level not in LEVELS:
            raise ValueError(f"Unknown report level {level!r}")
        self.findings.append(Finding(str(path), str(location), level, message))

    def count(self, level):
        """Return how many findings have the given level."""
        return sum(1 for f in self.findings if f.level == level)

    def for_path(self, path):
        """Return the findings of one file, ordered by location."""
        findings = [f for f in self.findings if f.path == str(path)]
        return sorted(findings, key=lambda f: _location_key(f.location))

    def paths(self):
        """Return the files that have findings, in first-seen order."""
        return list(dict.fromkeys(f.path for f in self.findings))

    def render(self):
        """Render every finding grouped by file, followed by a summary."""
        lines = []
        for path in self.paths():
            lines.append(path)
            for f in self.for_path(path):
                location = f"{f.location}: " if f.location else ""
                lines.append(f"  [{LEVEL_LABELS[f.level]}] {location}{f.message}")
            lines.append("")
        lines.append(self.summary())
        return "\n".join(lines)

    def summary(self):
        """Render the one-line count of findings per level."""
        parts = [
            f"{self.count(level)} {LEVEL_LABELS[level]}"
            for level in LEVELS
            if self.count(level)
        ]
        return "Summary: " + (", ".join(parts) if parts else "nothing to convert")
