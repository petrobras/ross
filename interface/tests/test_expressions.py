# -*- coding: utf-8 -*-
"""The expression evaluator: never `eval()`, and the error reaches the user.

The force fields accept an expression (`100*sin(2*t)`), and the first version
used `eval()` with an empty builtins dictionary -- which protects nothing: a
class literal reaches `__subclasses__` and from there the file system.

`services/expressions.py` walks the `ast` tree and only executes the nodes it
recognises. What is not on the list does not run, instead of running by
omission.

And an expression error **reaches the user**: before, it was swallowed and the
analysis carried on with null excitation, giving back a flat chart that looks
like a result."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from app import app
from api.security import SESSION_TOKEN
from services.expressions import safe_expression_eval, safe_math_eval
from waiting import answer_for

AUTH = {"X-ROSS-Token": SESSION_TOKEN}

SIMPLE_ROTOR = {
    "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
    "shafts": [
        {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"},
        {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "1"},
    ],
    "bearings": [
        {"element_type": "BASIC", "kxx": "1e6", "cxx": "2e2", "n": "0"},
        {"element_type": "BASIC", "kxx": "1e6", "cxx": "2e2", "n": "2"},
    ],
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2*pi*50", 2 * np.pi * 50),
        ("sqrt(2)/2", np.sqrt(2) / 2),
        ("1e6", 1e6),
        ("-9.81", -9.81),
    ],
)
def test_safe_math_eval_accepts_valid_expressions(expression, expected):
    assert safe_math_eval(expression) == pytest.approx(expected)


def test_safe_expression_eval_is_vectorised():
    t = np.linspace(0, 1, 5)
    result = safe_expression_eval("1000 * np.cos(speed * t)", {"t": t, "speed": 100.0})
    np.testing.assert_allclose(result, 1000 * np.cos(100.0 * t))


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('calc')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('x', 'w')",
        "np.load('x')",
        "2**10**10",
        "[x for x in range(9)]",
    ],
)
def test_safe_expression_eval_rejects_escapes(expression):
    with pytest.raises(ValueError):
        safe_expression_eval(expression, {"t": np.zeros(3)})


def test_invalid_force_expression_returns_400(client):
    """Before, the error was swallowed and the analysis ran with null excitation."""
    # The project goes inside the envelope since Phase 2 (slice 1).
    payload = {
        "project": SIMPLE_ROTOR,
        "conversion_type": "",
    }
    payload.update(
        {
            "analysis_type": "time_response",
            "params": {
                "speed": "100",
                "t_max": "0.05",
                "steps": "50",
                "plot_type": "1D",
                "forces": [{"node": 0, "dof": 0, "func": "1000 * cs(speed*t)"}],
                "probes": [{"node": 0, "angle": 0}],
            },
        }
    )
    response = answer_for(
        client, client.post("/run_analysis", json=payload, headers=AUTH), AUTH
    )
    assert response.status_code == 400
    assert "Invalid force" in response.json["message"]
