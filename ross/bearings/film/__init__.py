"""Geometry-agnostic parts of the fluid-film bearing solvers.

The journal engine (``ross.bearings.fluid_film``) takes from this subpackage
the lubricant resolution and the two-point exponential viscosity law
(``lubricant``), the hot-oil carry-over mixing (``mixing``) and the turbulence
constants (``turbulence``). The subpackage does not import
``ross.bearings.fluid_film``.

==============  ==============================================================
Module          Responsibility
==============  ==============================================================
``condense``    Condensation of pad coordinates at any whirl frequency, with
                the closed-form ``w = 0`` and ``w -> infinity`` limits.
``lubricant``   Viscosity-temperature laws (Walther / ASTM D341, two-point
                exponential); density, specific heat, conductivity; lubricant
                name / dictionary resolution.
``mixing``      Hot-oil carry-over mixing at the pad inlets (upstream pad,
                hot-oil fractions and flows, inlet temperatures).
``loops``       ``ConvergenceRecord``, ``ConvergenceWarning`` and
                ``blas_single_thread``.
``turbulence``  Reichardt constants and turbulent Prandtl number.
==============  ==============================================================

Everything is SI with temperatures in K.
"""

from ross.bearings.film.condense import (
    condense,
    condense_limit,
    condense_static,
    dynamic_matrix,
)
from ross.bearings.film.loops import (
    ConvergenceRecord,
    ConvergenceWarning,
    blas_single_thread,
)
from ross.bearings.film.lubricant import (
    Lubricant,
    exponential_viscosity,
    lubricant_data,
)
from ross.bearings.film.mixing import (
    carryover_flows,
    carryover_weights,
    temp_mixing_carryover,
    upstream_pads,
)

__all__ = [
    "ConvergenceRecord",
    "ConvergenceWarning",
    "Lubricant",
    "blas_single_thread",
    "carryover_flows",
    "carryover_weights",
    "condense",
    "condense_limit",
    "condense_static",
    "dynamic_matrix",
    "exponential_viscosity",
    "lubricant_data",
    "temp_mixing_carryover",
    "upstream_pads",
]
