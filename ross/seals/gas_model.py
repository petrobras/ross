"""Gas-property models shared by the compressible-flow seal solvers.

This module centralizes the thermodynamic relations used by the seal classes
(:class:`~ross.seals.labyrinth_seal.LabyrinthSeal` and, for the shared property
setup, :class:`~ross.seals.holepattern_seal.HolePatternSeal`).

Two backends implement a common interface:

``IdealGas``
    Perfect-gas relations (``Z = 1``, constant ``gamma``). Each method returns the
    exact expression used by the legacy solver, so the ideal path is reproduced
    bit-for-bit.

``RealGas``
    Equation-of-state backed relations (real ``Z``, real isentrope). A one-dimensional
    interpolation table is built once at construction along the inlet isentrope
    ``s = s_inlet`` (pressure -> temperature, density, sound speed, enthalpy and the
    isothermal density derivative). The solver inner loops then perform cheap table
    lookups instead of expensive per-station equation-of-state calls.

The geometric / empirical factors (``alphav``, discharge clearance, ``vnu``, cavity
area) are passed in as plain arguments; the models own thermodynamics only.
"""

import numpy as np
import ccp

__all__ = ["extract_gas_properties", "IdealGas", "RealGas"]

R_UNIVERSAL = 8314.0  # Universal gas constant (J/(kmol K)).


def extract_gas_properties(gas_composition, inlet_pressure, inlet_temperature):
    """Build the inlet thermodynamic state and the common gas constants.

    Creates the inlet ``ccp.State`` once and derives the molecular mass, the ratio
    of specific heats and the specific gas constant. Used by the seal constructors
    to avoid duplicating the boundary-property setup.

    Parameters
    ----------
    gas_composition : dict
        Gas composition as ``{component: molar_fraction}``.
    inlet_pressure : float
        Inlet pressure (Pa).
    inlet_temperature : float
        Inlet temperature (deg K).

    Returns
    -------
    state : ccp.State
        The inlet state, available for further property queries by the caller.
    molar : float
        Molecular mass (g/mol).
    gamma : float
        Ratio of specific heats ``cp / cv`` at the inlet.
    R : float
        Specific gas constant ``R_universal / molar`` (J/(kg K)).

    Examples
    --------
    >>> state, molar, gamma, R = extract_gas_properties(
    ...     {"Nitrogen": 0.79, "Oxygen": 0.21}, 308000, 283.15
    ... )
    >>> bool(280 < R < 300)
    True
    """
    state = ccp.State(p=inlet_pressure, T=inlet_temperature, fluid=gas_composition)
    molar = state.molar_mass("g/mol").m
    gamma = (state.cp() / state.cv()).m
    R = R_UNIVERSAL / molar
    return state, molar, gamma, R


class IdealGas:
    """Perfect-gas backend reproducing the legacy solver relations exactly.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg K)).
    gamma : float
        Ratio of specific heats.
    """

    def __init__(self, R, gamma):
        self.R = R
        self.gamma = gamma
        self.gam1 = 1 / gamma
        self.gam2 = (gamma - 1) / gamma
        self.gam3 = 2 / self.gam2
        self.gam4 = R * self.gam3
        self.gam5 = 1 / self.gam2
        self.gam7 = 2 / (gamma + 1)

    def inlet_density(self, p, T):
        """Return the inlet density ``p / (R T)`` (touch point 1)."""
        return p / (self.R * T)

    def density_isentropic(self, p_prev, pr, rho_prev):
        """Return the isentropic density update ``rho_prev * pr**(1/gamma)`` (touch point 3)."""
        return rho_prev * (pr**self.gam1)

    def temperature_isentropic(self, p_prev, pr, T_prev):
        """Return the isentropic temperature update ``T_prev * pr**((gamma-1)/gamma)`` (touch point 4)."""
        return T_prev * (pr**self.gam2)

    def throttle_mass_flux(self, alphav, Cr, p_prev, pr, rho_prev, T_prev, w_prev, vnu):
        """Return the throttle mass-flux term of the regula-falsi residual (touch point 5)."""
        return (
            alphav
            * Cr
            * rho_prev
            * (pr**self.gam1)
            * (((vnu * w_prev) * w_prev) + (self.gam4 * T_prev * (1 - (pr**self.gam2))))
            ** 0.5
        )

    def throat_velocity(self, mdot, alphav, Cr, p_prev, pr, T_prev):
        """Return the tooth (throat) velocity (touch point 2)."""
        return (mdot * self.R * T_prev) / (alphav * p_prev * (pr**self.gam1) * Cr)

    def critical_pr(self, p_prev, w_prev, vnu, T_prev):
        """Return the (momentum-augmented) critical pressure ratio (touch point 6)."""
        return (self.gam7 + (vnu * w_prev * w_prev / (self.gam4 * T_prev))) ** self.gam5

    def sound_speed(self, p, T):
        """Return the speed of sound ``sqrt(gamma R T)`` (touch point 7)."""
        return (self.gamma * self.R * T) ** 0.5

    def cg0(self, area, p, T):
        """Return the cavity-storage coefficient ``area / (R T)`` (touch point 8).

        This is the *isothermal* density derivative ``area * (d rho / d p)_T``: the
        cavity-storage perturbation is linearized isothermally in the bulk-flow
        theory, distinct from the isentropic through-flow relations above. Do not
        replace it with the isentropic derivative ``1 / (gamma R T)``.
        """
        return area / (self.R * T)


class RealGas(IdealGas):
    """Equation-of-state backed backend using a tabulated inlet isentrope.

    A pressure-indexed table is built once at construction by walking the inlet
    isentrope ``s = s_inlet`` with a single ``ccp.State`` object updated in place.
    All solver queries become ``numpy`` interpolations on the table, so the object
    holds only arrays and scalars and is safe to pickle across ``multiprocessing``.

    The ideal-gas formulas inherited from :class:`IdealGas` remain available as a
    fallback for quantities not driven by the table.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg K)).
    gamma : float
        Ratio of specific heats at the inlet.
    gas_composition : dict
        Gas composition as ``{component: molar_fraction}``.
    inlet_pressure : float
        Inlet pressure (Pa).
    inlet_temperature : float
        Inlet temperature (deg K).
    outlet_pressure : float
        Outlet pressure (Pa). Sets the lower bound of the table range.
    n_points : int, optional
        Number of table points along the isentrope. Default is 120.
    pressure_margin : float, optional
        Fraction of ``outlet_pressure`` used as the table lower bound, so the table
        extends below the outlet to cover choked-throat excursions. Default is 0.5.
    """

    def __init__(
        self,
        R,
        gamma,
        gas_composition,
        inlet_pressure,
        inlet_temperature,
        outlet_pressure,
        n_points=120,
        pressure_margin=0.5,
    ):
        super().__init__(R, gamma)

        p_lo = pressure_margin * outlet_pressure
        p_hi = inlet_pressure
        p_grid = np.logspace(np.log10(p_lo), np.log10(p_hi), n_points)

        state = ccp.State(p=inlet_pressure, T=inlet_temperature, fluid=gas_composition)
        s_inlet = state.s().m

        T_grid = np.zeros(n_points)
        rho_grid = np.zeros(n_points)
        a_grid = np.zeros(n_points)
        h_grid = np.zeros(n_points)
        drhodp_grid = np.zeros(n_points)

        for k, p in enumerate(p_grid):
            try:
                state.update(p=float(p), s=s_inlet)
                T_k = state.T().m
                rho_k = state.rho().m
                a_k = state.speed_sound().m
                h_k = state.h().m
            except (ValueError, RuntimeError) as err:
                raise RuntimeError(
                    f"Could not build the real-gas isentrope table at p={p:.1f} Pa. "
                    f"The inlet isentrope may cross a phase boundary within the seal "
                    f"pressure range, which the tabulated real-gas model does not "
                    f"support. ({err})"
                )
            if not (np.isfinite(rho_k) and rho_k > 0 and np.isfinite(a_k)):
                raise RuntimeError(
                    f"Invalid real-gas state at p={p:.1f} Pa (two-phase or "
                    f"non-converged); the isentrope leaves the single-phase region."
                )
            T_grid[k] = T_k
            rho_grid[k] = rho_k
            a_grid[k] = a_k
            h_grid[k] = h_k
            drhodp_grid[k] = self._isothermal_drhodp(state, float(p), T_k)

        self.p_grid = p_grid
        self.T_grid = T_grid
        self.rho_grid = rho_grid
        self.a_grid = a_grid
        self.h_grid = h_grid
        self.drhodp_grid = drhodp_grid
        self.p_min = p_grid[0]

    @staticmethod
    def _isothermal_drhodp(state, p, T):
        """Return ``(d rho / d p)_T`` at ``(p, T)`` by central finite difference."""
        dp = max(p * 1e-4, 1.0)
        try:
            state.update(p=p + dp, T=T)
            rho_hi = state.rho().m
            state.update(p=p - dp, T=T)
            rho_lo = state.rho().m
            return (rho_hi - rho_lo) / (2 * dp)
        except (ValueError, RuntimeError):
            state.update(p=p, T=T)
            rho_0 = state.rho().m
            state.update(p=p + dp, T=T)
            rho_hi = state.rho().m
            return (rho_hi - rho_0) / dp

    def _rho(self, p):
        return np.interp(p, self.p_grid, self.rho_grid)

    def _h(self, p):
        return np.interp(p, self.p_grid, self.h_grid)

    def _a(self, p):
        return np.interp(p, self.p_grid, self.a_grid)

    def inlet_density(self, p, T):
        return self._rho(p)

    def density_isentropic(self, p_prev, pr, rho_prev):
        return self._rho(pr * p_prev)

    def temperature_isentropic(self, p_prev, pr, T_prev):
        return np.interp(pr * p_prev, self.p_grid, self.T_grid)

    def throttle_mass_flux(self, alphav, Cr, p_prev, pr, rho_prev, T_prev, w_prev, vnu):
        p_t = pr * p_prev
        head = vnu * w_prev * w_prev + 2 * (self._h(p_prev) - self._h(p_t))
        return alphav * Cr * self._rho(p_t) * max(head, 0.0) ** 0.5

    def throat_velocity(self, mdot, alphav, Cr, p_prev, pr, T_prev):
        return mdot / (alphav * Cr * self._rho(pr * p_prev))

    def critical_pr(self, p_prev, w_prev, vnu, T_prev):
        """Return the real-gas critical pressure ratio.

        Locates, by bisection on the table, the throat pressure ``p_t`` at which the
        isentropic throat velocity ``sqrt(vnu w_prev**2 + 2 (h(p_prev) - h(p_t)))``
        equals the local sound speed ``a(p_t)``, then returns ``p_t / p_prev``.

        The critical ratio is nearly pressure-independent, so when the upstream
        pressure is not yet known (e.g. the pre-loop lower-bracket estimate, where
        ``p_prev`` is still zero) it is evaluated at the inlet reference instead.
        """
        if not (self.p_min < p_prev <= self.p_grid[-1]):
            p_prev = self.p_grid[-1]
        h_prev = self._h(p_prev)

        def excess(p_t):
            head = vnu * w_prev * w_prev + 2 * (h_prev - self._h(p_t))
            return max(head, 0.0) ** 0.5 - self._a(p_t)

        p_high = p_prev
        p_low = max(self.p_min, 1.0)
        if excess(p_low) <= 0:
            return p_low / p_prev
        if excess(p_high) >= 0:
            return p_high / p_prev
        for _ in range(60):
            p_mid = 0.5 * (p_low + p_high)
            if excess(p_mid) >= 0:
                p_low = p_mid
            else:
                p_high = p_mid
        return (0.5 * (p_low + p_high)) / p_prev

    def sound_speed(self, p, T):
        return self._a(p)

    def cg0(self, area, p, T):
        return area * np.interp(p, self.p_grid, self.drhodp_grid)
