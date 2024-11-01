import numpy as np
import pytest


# pytest hook to modify options for doctests
def pytest_configure(config):
    numpy_version = int(np.__version__.split(".")[0])
    if numpy_version >= 2:
        np.set_printoptions(legacy="1.25")
