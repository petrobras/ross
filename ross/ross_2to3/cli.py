"""Command line interface of ``ross_2to3``.

Run ``ross_2to3 PATH [PATH ...]`` to preview the conversion of ROSS 2 rotor
files (``.toml`` / ``.json``), scripts (``.py``) and notebooks (``.ipynb``)
as a diff plus a report, ``ross_2to3 -w PATH`` to rewrite the files in place
(a ``.bak`` copy is kept) or ``ross_2to3 -o DIR PATH`` to write the converted
files into another directory.
"""

import argparse
import difflib
import shutil
import sys
import tempfile
from pathlib import Path

from ross.ross_2to3.models import MODEL_SUFFIXES, check_model_file, convert_model_text
from ross.ross_2to3.report import ERROR, Report, SKIPPED
from ross.ross_2to3.scripts import SCRIPT_SUFFIXES, convert_notebook, convert_source

SUFFIXES = SCRIPT_SUFFIXES + MODEL_SUFFIXES


def ross_version():
    """Return the version of the installed ROSS."""
    import ross

    return ross.__version__


def iter_files(paths):
    """Yield (file, root) pairs for the given files and directories.

    ``root`` is the directory the file was found under (its own parent for
    files given explicitly), so converted files can keep their relative
    layout under ``--output-dir``.
    """
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                relative_parts = candidate.relative_to(path).parts
                if any(part.startswith(".") for part in relative_parts):
                    continue
                if candidate.is_file() and candidate.suffix.lower() in SUFFIXES:
                    yield candidate, path
        else:
            yield path, path.parent


def convert_text(path, text, report, version):
    """Convert the content of one file according to its suffix.

    Returns
    -------
    str or None
        The converted text, or None when the file is not something
        ``ross_2to3`` handles (e.g. a JSON file that is not a ROSS model).
    """
    suffix = path.suffix.lower()
    if suffix == ".py":
        return convert_source(text, path, report)
    if suffix == ".ipynb":
        return convert_notebook(text, path, report)
    return convert_model_text(text, suffix, report, path=path, version=version)


def _check(path, converted, report):
    if path.suffix.lower() not in MODEL_SUFFIXES:
        return
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / path.name
        target.write_text(converted, encoding="utf-8")
        check_model_file(target, report, label=path)


def main(argv=None):
    """Run the ross_2to3 command line interface."""
    parser = argparse.ArgumentParser(
        prog="ross_2to3",
        description=(
            "Convert ROSS 2 rotor files (.toml/.json), Python scripts (.py) and "
            "notebooks (.ipynb) to the ROSS 3 API. Without -w or -o the changes "
            "are only previewed."
        ),
    )
    parser.add_argument("paths", nargs="+", help="files or directories to convert")
    parser.add_argument(
        "-w", "--write", action="store_true", help="rewrite the files in place"
    )
    parser.add_argument(
        "-n",
        "--no-backup",
        action="store_true",
        help="with -w, do not keep a .bak copy of the original",
    )
    parser.add_argument(
        "-o", "--output-dir", help="write the converted files into this directory"
    )
    parser.add_argument(
        "--no-diff", action="store_true", help="do not print the diff of the preview"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="do not load the converted rotor files with the installed ROSS",
    )
    parser.add_argument("--report", help="also write the report to this file")
    args = parser.parse_args(argv)

    report = Report()
    version = ross_version()
    output_dir = Path(args.output_dir) if args.output_dir else None

    for path, root in iter_files(args.paths):
        if not path.is_file():
            report.add(path, "", ERROR, "file not found")
            continue
        try:
            text = path.read_text(encoding="utf-8")
            converted = convert_text(path, text, report, version)
        except Exception as exc:
            report.add(path, "", ERROR, f"could not convert: {exc!r}")
            continue
        if converted is None:
            report.add(path, "", SKIPPED, "not a ROSS rotor or element file")
            continue
        if converted == text:
            continue
        if not args.no_check:
            _check(path, converted, report)
        if output_dir is not None:
            target = output_dir / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(converted, encoding="utf-8")
            print(f"wrote {target}")
        elif args.write:
            if not args.no_backup:
                shutil.copy2(path, path.with_name(path.name + ".bak"))
            path.write_text(converted, encoding="utf-8")
            print(f"rewrote {path}")
        elif not args.no_diff:
            diff = difflib.unified_diff(
                text.splitlines(keepends=True),
                converted.splitlines(keepends=True),
                fromfile=str(path),
                tofile=f"{path} (ROSS {version})",
            )
            sys.stdout.writelines(diff)
            print()

    text = report.render()
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 1 if report.count(ERROR) else 0


if __name__ == "__main__":
    sys.exit(main())
