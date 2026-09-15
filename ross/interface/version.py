# -*- coding: utf-8 -*-
"""The ROSS version the interface carries, read once and shown in three places.

`ross-interface --version`, the header of `--selftest` and the About dialog of
the page all print this value, so that they cannot disagree: the selftest header
is what a release asset is audited by, and the dialog is what a user copies into
a bug report.

Read from the distribution metadata and not via `import ross`, for the reason
`check.py` gives: when `import ross` is the failure, the header still has to
name the version.
"""

from importlib import metadata


def ross_version():
    """The installed ross-rotordynamics version, or "not installed"."""
    try:
        return metadata.version("ross-rotordynamics")
    except metadata.PackageNotFoundError:
        return "not installed"
