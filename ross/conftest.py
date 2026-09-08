import numpy as np
import pytest

from ross.plotly_theme import _apply_plotly_compat_shim

_apply_plotly_compat_shim()


# pytest hook to modify options for doctests
def pytest_configure(config):
    _apply_plotly_compat_shim()
    numpy_version = int(np.__version__.split(".")[0])
    if numpy_version >= 2:
        np.set_printoptions(legacy="1.25")
