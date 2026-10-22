"""Check that a release tag matches the version of the built distributions.

The Release workflow runs this between building and uploading to PyPI, so a tag
pushed before ``__version__`` is bumped (``v3.1.0`` on a ``3.1.0.dev0`` commit)
fails instead of publishing the wrong version. Versions are compared as PEP 440
versions, and the tag must also be the canonical spelling of the version
(``v3.1.0rc1``, not ``v3.1.0-rc1``), because the tag name becomes the Read the
Docs version and the GitHub release name.

Usage::

    python tools/check_release_version.py v3.1.0 dist/
"""

import argparse
import sys
from pathlib import Path

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version


def distribution_versions(dist_dir):
    """Return the version of each wheel and sdist in a directory.

    Parameters
    ----------
    dist_dir : str or pathlib.Path
        Directory written by ``python -m build``.

    Returns
    -------
    versions : dict
        Version (``packaging.version.Version``) keyed by file name.
    """
    versions = {}
    for path in sorted(Path(dist_dir).iterdir()):
        try:
            if path.name.endswith(".whl"):
                versions[path.name] = parse_wheel_filename(path.name)[1]
            elif path.name.endswith(".tar.gz"):
                versions[path.name] = parse_sdist_filename(path.name)[1]
        except (InvalidWheelFilename, InvalidSdistFilename) as error:
            raise ValueError(f"cannot read the version of {path.name}: {error}")
    return versions


def check(tag, dist_dir):
    """Compare a release tag with the version of the built distributions.

    Parameters
    ----------
    tag : str
        Git tag name, such as ``v3.1.0``.
    dist_dir : str or pathlib.Path
        Directory written by ``python -m build``.

    Returns
    -------
    errors : list of str
        One message per problem found; empty when the tag matches.
    """
    if not tag.startswith("v"):
        return [f"tag {tag} does not start with 'v'"]
    try:
        tag_version = Version(tag[1:])
    except InvalidVersion:
        return [f"tag {tag} is not 'v' followed by a PEP 440 version"]

    versions = distribution_versions(dist_dir)
    if not versions:
        return [f"no wheel or sdist found in {dist_dir}"]

    errors = [
        f"tag {tag} does not match {name}, which is version {version}; "
        f'the tagged commit must set __version__ = "{tag_version}" in '
        f"ross/__init__.py"
        for name, version in versions.items()
        if version != tag_version
    ]
    built_version = str(next(iter(versions.values())))
    if not errors and tag[1:] != built_version:
        errors.append(
            f"tag {tag} is not the canonical spelling of version {built_version}; "
            f"tag v{built_version} instead"
        )
    return errors


def main(argv=None):
    """Run the check from the command line and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag", help="release tag, such as v3.1.0")
    parser.add_argument("dist_dir", help="directory with the built distributions")
    args = parser.parse_args(argv)

    errors = check(args.tag, args.dist_dir)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"{args.tag} matches the built distributions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
