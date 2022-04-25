"""Script to run and save all documentation notebooks.

This is useful to update the notebooks before each release.
"""

import nbformat
from pathlib import Path
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

path = Path.cwd()

for notebook_filename in path.rglob("*.ipynb"):
    if ".ipynb_checkpoints" not in str(notebook_filename) and "_build" not in str(
        notebook_filename
    ):
        try:
            print(f"Executing {notebook_filename}")
            nb = nbformat.read(notebook_filename, as_version=4)
            client = NotebookClient(
                nb,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": notebook_filename.parent}},
                store_widget_state=True,
            )
            client.execute()
            nbformat.write(nb, notebook_filename)
        except (CellExecutionError, CellTimeoutError):
            msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
            msg += 'See notebook "%s" for the traceback.' % notebook_filename
            print(msg)
            raise
        finally:
            nbformat.write(nb, notebook_filename)
