# -*- coding: utf-8 -*-
"""Analysis registry: one class per analysis, instead of an if/elif.

Importing this package registers the twelve runners. `get_runner(name)` returns
the runner or raises ValueError listing the known names -- the message used to
be "Analysis not implemented yet.", which did not say which ones existed.

The order of the imports below does not matter; what matters is that they are
all here, or the corresponding analysis disappears from the interface without
warning. `tests/test_runners.py` requires the registry to cover exactly the analyses
the screen offers.
"""

from .base import REGISTRY, Runner, get_runner, register  # noqa: F401

from . import campbell  # noqa: F401
from . import clearance  # noqa: F401
from . import freq_response  # noqa: F401
from . import harmonic_balance  # noqa: F401
from . import modal  # noqa: F401
from . import static  # noqa: F401
from . import transient  # noqa: F401
from . import ucs  # noqa: F401
from . import unbalance  # noqa: F401

__all__ = ["REGISTRY", "Runner", "get_runner", "register"]
