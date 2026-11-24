"""Restricted unpickling for surrogate files.

Surrogate models are stored as pickles inside a zip file. Loading them with
``pickle.loads`` would execute arbitrary code from an untrusted file, so this
module only reconstructs objects from an allowlist of scientific modules and
never from ``builtins.eval`` or ``os.system``.
"""

import io
import pickle
import zlib

_ALLOWED_MODULE_PREFIXES = (
    "numpy",
    "pandas",
    "ross",
    "pint",
    "scipy",
    "sklearn",
    "joblib",
    "collections",
    "copyreg",
    "encodings",
    "codecs",
)

_ALLOWED_BUILTINS = frozenset(
    {
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "set",
        "str",
        "tuple",
        "bytes",
        "bytearray",
        "bool",
        "NoneType",
        "slice",
        "range",
        "object",
        "type",
        "staticmethod",
        "classmethod",
    }
)

# Models saved by the standalone ``rossml`` package pickled their classes under
# these module paths.
_MODULE_ALIASES = {
    "rossml.surrogates.surrogate_core": "ross.surrogates.surrogate_core",
    "rossml.cylindrical_surrogate": "ross.surrogates.surrogate_core",
    "rossml.surrogates.base": "ross.surrogates.surrogate_core",
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that rejects globals outside the allowlist."""

    def find_class(self, module, name):
        module = _MODULE_ALIASES.get(module, module)
        if module == "smt" or module.startswith("smt."):
            raise pickle.UnpicklingError(
                "This surrogate was saved with the Kriging model of SMT, which ROSS "
                "no longer uses. Train the surrogate again."
            )
        if module in ("builtins", "__builtin__"):
            if name in _ALLOWED_BUILTINS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"forbidden global {module}.{name}")
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in _ALLOWED_MODULE_PREFIXES
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"forbidden global {module}.{name}")


def loads(data):
    """Unpickle bytes that may be zlib-compressed.

    Parameters
    ----------
    data : bytes
        Pickled payload.

    Returns
    -------
    object
        The unpickled object.
    """
    try:
        payload = zlib.decompress(data)
    except zlib.error:
        payload = data
    return RestrictedUnpickler(io.BytesIO(payload)).load()
