# -*- coding: utf-8 -*-
"""Make the Digital Twin rotor models callable by the ROSS installed here.

Some models (`lmest_2discos`, for one) adjust the assembled matrices and then
replace `rotor.M`, `rotor.K`, ... with lambdas returning the adjusted matrix,
written for an older signature: `lambda frequency=None, synchronous=False`.
ROSS 3 calls `M(frequency, speed, synchronous=...)`, so every analysis that
assembles the matrices fails with a `TypeError` inside ROSS.

The adjusted matrices are kept by the model itself as `mat_rotor_M`, ... The
replacement below returns the same matrix for any arguments, which is what the
model meant: its matrices do not depend on frequency or speed. Rotors that do
not override their matrices are left untouched.
"""

OVERRIDES = {
    "M": "mat_rotor_M",
    "K": "mat_rotor_K",
    "C": "mat_rotor_C",
    "G": "mat_rotor_G",
    "Ksdt": "mat_rotor_Ksdt",
}


def accept_any_signature(rotor):
    """Rewrap constant matrix overrides so they accept the ROSS 3 arguments."""
    attributes = vars(rotor)
    for method, stored in OVERRIDES.items():
        if method in attributes and stored in attributes:
            setattr(rotor, method, _constant(attributes[stored]))
    return rotor


def _constant(matrix):
    def matrix_for(*args, **kwargs):
        return matrix

    return matrix_for
