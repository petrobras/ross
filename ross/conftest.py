import numpy as np
import pytest

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def pytest_configure(config):
    numpy_version = int(np.__version__.split(".")[0])
    if numpy_version >= 2:
        np.set_printoptions(legacy="1.25")

    # The fluid-film bearing tests spend their time in many small BLAS calls,
    # where numpy's and scipy's OpenBLAS pools (one per core each) bring no
    # speedup single-process but make pytest-xdist workers fight for every
    # core: 8 workers on 16 cores ran the bearing tests in 12 minutes against
    # 38 seconds with the pools capped. Capping costs nothing single-process.
    if threadpool_limits is not None:
        config._ross_threadpool_limits = threadpool_limits(limits=1)
