#! /usr/bin/env python
"""
Convert empty IPython notebook to a sphinx doc page.
"""
import sys
from subprocess import check_call as sh
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove


def convert_nb(nbname):

    # Execute the notebook
    sh(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nbname])

    # Convert to .rst for Sphinx
    sh(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "rst",
            nbname,
            "--TagRemovePreprocessor.remove_cell_tags={'hide'}",
            "--TagRemovePreprocessor.remove_input_tags={'hide-input'}",
            "--TagRemovePreprocessor.remove_all_outputs_tags={'hide-output'}",
        ]
    )

    # Clear notebook output
    sh(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--inplace",
            "--ClearOutputPreprocessor.enabled=True",
            nbname,
        ]
    )

    # Touch the .rst file so it has a later modify time than the source
    sh(["touch", nbname + ".rst"])


def replace(file_path, patterns):
    """Replace pattern substitution tuples."""
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, "w") as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                for pattern in patterns:
                    p, s = pattern
                    line = line.replace(p, s)
                new_file.write(line)
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)


if __name__ == "__main__":
    for nbname in sys.argv[1:]:
        convert_nb(nbname)
        replace(
            (nbname + ".rst"),
            [
                (".. parsed-literal::", ".. code-block:: text"),
                (".. code:: text", ".. code-block:: text"),
            ],
        )
