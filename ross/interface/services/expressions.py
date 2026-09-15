# -*- coding: utf-8 -*-
"""An allow-list expression evaluator, in place of eval().

The time response's F(t) field accepts an expression written by the user. Until
Phase 0 that went straight into eval(), which -- together with the open CORS --
gave code execution from any browser tab (BE-01).

Here the expression is compiled to an AST and walked node by node; only
numbers, the supplied variables, the arithmetic operators and the functions in
SAFE_FUNCTIONS get through. Any other node raises ValueError naming what was
refused. It left app.py in Phase 2 because the analysis runners need it and
cannot import the app.
"""

import ast
import operator

import numpy as np

SAFE_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "min": np.minimum,
    "max": np.maximum,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "heaviside": np.heaviside,
    "where": np.where,
    "clip": np.clip,
    "mod": np.mod,
    "zeros_like": np.zeros_like,
    "ones_like": np.ones_like,
    "deg2rad": np.deg2rad,
    "rad2deg": np.rad2deg,
}
SAFE_CONSTANTS = {"pi": np.pi, "e": np.e}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 1000


def _eval_node(node, variables):
    """Evaluate one AST node against the allow-list."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float, complex)
        ):
            raise ValueError(f"constant not allowed: {node.value!r}")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        names = sorted(list(variables) + list(SAFE_CONSTANTS))
        raise ValueError(
            f"unknown name '{node.id}'. Variables: {', '.join(names)}. "
            f"Functions: {', '.join(sorted(SAFE_FUNCTIONS))}"
        )

    if isinstance(node, ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr in SAFE_FUNCTIONS
        ):
            return SAFE_FUNCTIONS[node.attr]
        raise ValueError(f"attribute not allowed: '.{node.attr}'")

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        func = _eval_node(node.func, variables)
        if not any(func is allowed for allowed in SAFE_FUNCTIONS.values()):
            raise ValueError("function call not allowed")
        return func(*[_eval_node(a, variables) for a in node.args])

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Pow):
            # The exponent is evaluated before the check: '2**10**10' associates to the right,
            # so the bound has to apply to the final value, not to the literal.
            magnitude = np.max(np.abs(np.asarray(right, dtype=float)))
            if not np.isfinite(magnitude) or magnitude > _MAX_EXPONENT:
                raise ValueError(f"exponent above the limit of {_MAX_EXPONENT}")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand, variables))

    raise ValueError(f"expression not allowed: {type(node).__name__}")


def safe_expression_eval(expr, variables=None):
    """Evaluate a math expression without eval(). Returns a float or a numpy array."""
    variables = variables or {}
    text = str(expr).strip()
    if text == "":
        raise ValueError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid syntax: {exc.msg}")
    return _eval_node(tree, variables)


def safe_math_eval(expr):
    """Evaluate a scalar math expression coming from a numeric input field."""
    try:
        return float(safe_expression_eval(expr))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not evaluate the expression '{expr}': {exc}")
