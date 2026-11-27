# -*- coding: utf-8 -*-
"""Digital Twin routes.

A route reads the request, calls `domain` or `services`, and returns JSON.
The session token, the error handlers and the page itself belong to the
ROSS interface, which registers these blueprints through the extension.
"""

from .analysis import analysis
from .balance import balance
from .digital_twin import digital_twin

BLUEPRINTS = (balance, digital_twin, analysis)

__all__ = ["BLUEPRINTS"]
