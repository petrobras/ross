"""Turbulence constants of the fluid-film bearings.

The Reichardt eddy viscosity ``eps/nu = kappa (y+ - delta tanh(y+/delta))``
and the turbulent Prandtl number of the eddy conductivity. The journal engine
passes ``REICHARDT_KAPPA`` and ``REICHARDT_DELTA`` to its kernels
(``FluidFilmBearing._engine_inputs``) and uses ``TURBULENT_PRANDTL`` as
``ross.bearings.fluid_film.thermal.PR_TURB``.
"""

__all__ = ["REICHARDT_KAPPA", "REICHARDT_DELTA", "TURBULENT_PRANDTL"]

#: Reichardt constant ``kappa``.
REICHARDT_KAPPA = 0.4
#: Reichardt constant ``delta``.
REICHARDT_DELTA = 8.8
#: Turbulent Prandtl number of the eddy conductivity.
TURBULENT_PRANDTL = 0.769
