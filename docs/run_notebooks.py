"""Run and save documentation notebooks, refreshing their stored outputs.

Usage::

    python run_notebooks.py                        # every notebook under the current directory
    python run_notebooks.py PATH...                # only the given notebooks or directories
    python run_notebooks.py --timeout 3600 PATH    # allow slow cells (seconds, default 600)

Read the Docs never executes notebooks (see ``conf.py``), so the outputs stored
here are what the site shows. Notebooks that plot must set
``pio.renderers.default = "notebook_connected"`` so Plotly figures are stored as
HTML that loads plotly.js from its CDN (``"notebook"`` stores a 5 MB copy of it);
``fig.show()`` without it stores only the Plotly JSON mimetype, which the site
skips.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError


def notebooks_under(paths):
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("*.ipynb"))
        for notebook in candidates:
            if ".ipynb_checkpoints" in str(notebook) or "_build" in str(notebook):
                continue
            yield notebook


WARNING_LOCATION = re.compile(
    r"^(?:[A-Za-z]:)?[^\s:]*[/\\][^\s:]+\.py:\d+: (\w+: .*)\n(?:[ \t]+\S.*\n)?",
    re.MULTILINE,
)


ABSOLUTE_PATH = re.compile(
    r"(?<![\w.)\]])(?:[A-Za-z]:)?(?:[/\\][^\s/\\:]+)+[/\\]([^\s/\\:]+)"
)
WARN_SOURCE_LINE = re.compile(r"^[ \t]+_?warnings\.warn\(.*\n", re.MULTILINE)


def strip_warning_paths(nb):
    """Remove the file path, line number and source line from stored warnings.

    Python prints ``/path/to/file.py:123: UserWarning: message`` followed by the
    offending source line. The path is specific to the machine that ran the
    notebook, so only ``UserWarning: message`` is kept. Absolute paths inside
    a message are reduced to their file name.
    """
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                text = WARNING_LOCATION.sub(r"\1\n", output["text"])
                text = WARN_SOURCE_LINE.sub("", text)
                output["text"] = ABSOLUTE_PATH.sub(r"\1", text)


def run_notebook(notebook_filename, timeout=600):
    print(f"Executing {notebook_filename} with {sys.executable}", flush=True)
    # The python3 kernelspec starts "python" from PATH; make it this interpreter.
    interpreter_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = interpreter_dir + os.pathsep + os.environ.get("PATH", "")
    nb = nbformat.read(notebook_filename, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": notebook_filename.parent}},
        store_widget_state=True,
    )
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError):
        print(
            f'Error executing the notebook "{notebook_filename}".\n\n'
            f'See notebook "{notebook_filename}" for the traceback.'
        )
        raise
    finally:
        strip_warning_paths(nb)
        nbformat.write(nb, notebook_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="notebooks or folders")
    parser.add_argument(
        "--timeout", type=int, default=600, help="seconds allowed per cell"
    )
    args = parser.parse_args()
    for notebook_filename in notebooks_under(args.paths or [Path.cwd()]):
        run_notebook(notebook_filename, timeout=args.timeout)
