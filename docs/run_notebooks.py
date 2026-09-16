"""Run and save documentation notebooks, refreshing their stored outputs.

Usage::

    python run_notebooks.py                        # every notebook under the current directory
    python run_notebooks.py PATH...                # only the given notebooks or directories
    python run_notebooks.py --timeout 3600 PATH    # allow slow cells (seconds, default 600)

Read the Docs never executes notebooks (see ``conf.py``), so the outputs stored
here are what the site shows. Notebooks that plot must set
``pio.renderers.default = "notebook"`` so Plotly figures are stored as HTML;
``fig.show()`` without it stores only the Plotly JSON mimetype, which the site
skips.
"""

import argparse
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


def run_notebook(notebook_filename, timeout=600):
    print(f"Executing {notebook_filename}", flush=True)
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
