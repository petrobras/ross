"""Gas-property models shared by the compressible-flow seal solvers.

This module holds the thermodynamic relations used by the seal classes
(:class:`~ross.seals.labyrinth_seal.LabyrinthSeal` and
:class:`~ross.seals.holepattern_seal.HolePatternSeal`), in two families with
different interfaces.

Labyrinth relations, ``IdealGas`` and ``RealGas``
    The quantities the labyrinth formulas use at each cavity and throttle (throttle
    mass flux, critical pressure ratio, sound speed, cavity storage). The flow
    through a labyrinth throttle is isentropic, so ``RealGas`` tabulates the inlet
    isentrope ``s = s_inlet`` as a function of pressure only.

    ``IdealGas``
        Perfect-gas relations (``Z = 1``, constant ``gamma``). Each method returns
        the exact expression used by the legacy solver, so the ideal path is
        reproduced bit-for-bit.

    ``RealGas``
        Equation-of-state backed relations (real ``Z``, real isentrope). A
        one-dimensional interpolation table is built once at construction along the
        inlet isentrope (pressure -> temperature, density, sound speed, enthalpy and
        the isothermal density derivative).

State functions of ``(p, T)``, ``StateGas`` and its subclasses
    The hole-pattern bulk flow is not isentropic: wall friction heats the gas and
    the rotor does work on it. Its solver needs density, enthalpy, entropy and
    viscosity as functions of pressure and temperature, with their partial
    derivatives (``GasPoint``). The labyrinth relations have neither the second
    state variable nor the viscosity, so this is a separate interface.

    ``SutherlandGas``
        Perfect gas with constant ``gamma`` and the Sutherland viscosity law, in
        closed form.

    ``TabulatedGas``
        Bicubic splines of the equation of state of ``ccp`` on a ``(p, T)`` grid
        whose cold edge follows the inlet isentrope. A state outside the table
        raises ``GasRangeError``; the table is never extrapolated.

The geometric / empirical factors (``alphav``, discharge clearance, ``vnu``, cavity
area) are passed in as plain arguments; the models own thermodynamics only.
"""

import math
from bisect import bisect_right

import numpy as np
import ccp
from CoolProp import CoolProp as coolprop
from scipy.interpolate import CubicSpline, RectBivariateSpline

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
    molar_mass : float
        Molecular mass (g/mol).
    gamma : float
        Ratio of specific heats ``cp / cv`` at the inlet.
    R : float
        Specific gas constant ``R_universal / molar_mass`` (J/(kg K)).

    Examples
    --------
    >>> state, molar_mass, gamma, R = extract_gas_properties(
    ...     {"Nitrogen": 0.79, "Oxygen": 0.21}, 308000, 283.15
    ... )
    >>> bool(280 < R < 300)
    True
    """
    state = ccp.State(p=inlet_pressure, T=inlet_temperature, fluid=gas_composition)
    molar_mass = state.molar_mass("g/mol").m
    gamma = (state.cp() / state.cv()).m
    R = R_UNIVERSAL / molar_mass
    return state, molar_mass, gamma, R


_TABLE_MARGINS = {"p_low": 0.9, "p_high": 1.001, "t_below": 5.0, "t_above": 25.0}
_TABLE_WIDENING = 3.0
_TABLE_P_LOW_WIDENING = 0.9
_TABLE_POINTS = 41
_TABLE_PHASE = "supercritical_gas"
_PHASE_RTOL = 1e-6
_SPURIOUS_RTOL = 1e-2
_SPLIT_RTOL = 1e-2
_FLOOR_BISECTIONS = 12
_ISENTROPE_SUBSTEPS = 8


def _unstable_state_error(p, T):
    """Return the error for a state that is not thermodynamically stable."""
    return ValueError(
        f"the gas state p = {p:.6g} Pa, T = {T:.6g} K is not thermodynamically "
        "stable (cp, (d rho / d p)_T or (d rho / d p)_s is not positive); near the "
        "critical point the property table of the hole-pattern seal cannot "
        "represent the gas, and the single-phase model does not support it"
    )


class GasRangeError(ValueError):
    """Signal a state outside the domain of a gas object.

    Raised for a state outside a property table, or at the pole of a Sutherland
    viscosity law. The hole-pattern solver uses the type to tell a flow that left
    the property range apart from a flow that chokes.

    Parameters
    ----------
    message : str
        Description of the state and of the domain.
    pressure : float, optional
        Pressure of the requested state (Pa).
    temperature : float, optional
        Temperature of the requested state (K).
    side : str, optional
        ``"below"`` when the state lies at a lower pressure than the domain, or is
        colder than it at its pressure; ``"above"`` for a higher pressure or a
        hotter state; ``None`` when not known.

    Examples
    --------
    >>> import pickle
    >>> error = GasRangeError("outside", 1e5, 300.0, "above")
    >>> pickle.loads(pickle.dumps(error)).side
    'above'
    """

    def __init__(self, message, pressure=None, temperature=None, side=None):
        super().__init__(message)
        self.pressure = pressure
        self.temperature = temperature
        self.side = side

    def __reduce__(self):
        return (type(self), (str(self), self.pressure, self.temperature, self.side))


class GasPoint:
    """Hold the state functions of a gas at one pressure and temperature.

    All values are SI. The derived quantities are read-only properties; their
    relations are given in the hole-pattern theory page.

    Parameters
    ----------
    p : float
        Pressure (Pa).
    T : float
        Temperature (K).
    rho : float
        Density (kg/m^3).
    rho_p : float
        ``(d rho / d p)_T`` (kg/(m^3 Pa)).
    rho_T : float
        ``(d rho / d T)_p`` (kg/(m^3 K)).
    h : float
        Specific enthalpy (J/kg).
    cp : float
        ``(d h / d T)_p`` (J/(kg K)).
    h_p : float
        ``(d h / d p)_T`` (m^3/kg).
    mu : float
        Dynamic viscosity (Pa s).
    mu_p : float
        ``(d mu / d p)_T`` (s).
    mu_T : float
        ``(d mu / d T)_p`` (Pa s/K).

    Examples
    --------
    >>> point = SutherlandGas(287.0, 1.4, 1.458e-6, 110.4).point(1e5, 300.0)
    >>> round(point.sound_speed, 6) == round((1.4 * 287.0 * 300.0) ** 0.5, 6)
    True
    """

    __slots__ = (
        "p",
        "T",
        "rho",
        "rho_p",
        "rho_T",
        "h",
        "cp",
        "h_p",
        "mu",
        "mu_p",
        "mu_T",
    )

    def __init__(self, p, T, rho, rho_p, rho_T, h, cp, h_p, mu, mu_p, mu_T):
        self.p = p
        self.T = T
        self.rho = rho
        self.rho_p = rho_p
        self.rho_T = rho_T
        self.h = h
        self.cp = cp
        self.h_p = h_p
        self.mu = mu
        self.mu_p = mu_p
        self.mu_T = mu_T

    def __repr__(self):
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.__slots__)
        return f"GasPoint({fields})"

    @property
    def rho_p_s(self):
        """``(d rho / d p)_s = rho_p - T rho_T^2 / (rho^2 cp) = 1 / a^2`` (s^2/m^2).

        Raises ``ValueError`` naming the state when it is not thermodynamically
        stable (``cp``, ``rho_p`` or this derivative not positive), which a
        table can give near the critical point.
        """
        value = self.rho_p - self.T * self.rho_T**2 / (self.rho**2 * self.cp)
        if not (value > 0.0 and self.cp > 0.0 and self.rho_p > 0.0):
            raise _unstable_state_error(self.p, self.T)
        return value

    @property
    def sound_speed(self):
        """Speed of sound (m/s), from ``a^-2 = rho_p - T rho_T^2 / (rho^2 cp)``."""
        return 1.0 / math.sqrt(self.rho_p_s)

    @property
    def kappa_s(self):
        """Isentropic compressibility ``1 / (rho a^2)`` (1/Pa)."""
        return self.rho_p_s / self.rho

    @property
    def p_rho(self):
        """``(d p / d rho)_T`` (``R T`` for a perfect gas)."""
        return 1.0 / self.rho_p

    @property
    def p_t(self):
        """``(d p / d T)_rho`` (``rho R`` for a perfect gas)."""
        return -self.rho_T / self.rho_p

    @property
    def h_rho(self):
        """``(d h / d rho)_T`` (0 for a perfect gas)."""
        return self.h_p / self.rho_p

    @property
    def h_t(self):
        """``(d h / d T)_rho`` (``cp`` for a perfect gas)."""
        return self.cp - self.h_p * self.rho_T / self.rho_p

    @property
    def mu_rho(self):
        """``(d mu / d rho)_T`` (0 for the Sutherland law)."""
        return self.mu_p / self.rho_p

    @property
    def mu_t(self):
        """``(d mu / d T)_rho``."""
        return self.mu_T - self.mu_p * self.rho_T / self.rho_p


class StateGas:
    """Define the state functions of ``(p, T)`` that the hole-pattern flow uses.

    Subclasses implement :meth:`point` and :meth:`entropy`. The inversions
    :meth:`temperature_ph` and :meth:`temperature_ps` are Newton iterations on
    those two, and :meth:`widened` re-raises, so a gas without a table cannot
    extend its domain. All quantities are SI.
    """

    def point(self, p, T):
        """Return the state functions at ``(p, T)``.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        GasPoint
            Values and partial derivatives at ``(p, T)``.

        Raises
        ------
        GasRangeError
            The state lies outside the domain of the gas.
        """
        raise NotImplementedError

    def entropy(self, p, T):
        """Return the specific entropy at ``(p, T)`` (J/(kg K)).

        The reference is arbitrary; only differences are meaningful.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Specific entropy (J/(kg K)).
        """
        raise NotImplementedError

    def temperature_ph(self, p, h, guess):
        """Return the temperature at pressure ``p`` and specific enthalpy ``h``.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        h : float
            Specific enthalpy (J/kg).
        guess : float
            Starting temperature of the Newton iteration (K).

        Returns
        -------
        float
            Temperature (K).
        """
        T = guess
        for _ in range(60):
            point = self.point(p, T)
            step = (h - point.h) / point.cp
            T += step
            if abs(step) < 1e-12 * T:
                return T
        raise RuntimeError(
            f"temperature_ph did not converge at p={p:g} Pa, h={h:g} J/kg"
        )

    def temperature_ps(self, p, s, guess):
        """Return the temperature at pressure ``p`` and specific entropy ``s``.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        s : float
            Specific entropy (J/(kg K)), with the reference of :meth:`entropy`.
        guess : float
            Starting temperature of the Newton iteration (K).

        Returns
        -------
        float
            Temperature (K).
        """
        T = guess
        for _ in range(60):
            step = (s - self.entropy(p, T)) * T / self.point(p, T).cp
            T += step
            if abs(step) < 1e-12 * T:
                return T
        raise RuntimeError(
            f"temperature_ps did not converge at p={p:g} Pa, s={s:g} J/(kg K)"
        )

    def widened(self, error):
        """Return a gas whose domain contains more of the side that ``error`` names.

        Parameters
        ----------
        error : GasRangeError
            The error raised for a state outside the domain.

        Returns
        -------
        StateGas
            A gas with a larger domain. This base implementation cannot widen and
            re-raises ``error``.
        """
        raise error


class SutherlandGas(StateGas):
    """Perfect gas with constant ``gamma`` and the Sutherland viscosity law.

    ``p = rho R T``, ``h = cp T`` and ``mu = b T^1.5 / (S + T)``.

    Parameters
    ----------
    gas_constant : float
        Specific gas constant ``R`` (J/(kg K)).
    gamma : float
        Ratio of specific heats.
    sutherland_b : float
        Sutherland constant ``b`` (Pa s/K^0.5).
    sutherland_s : float
        Sutherland temperature ``S`` (K).

    Examples
    --------
    >>> air = SutherlandGas(287.0, 1.4, 1.458e-6, 110.4)
    >>> T = air.temperature_ps(2e5, air.entropy(2e5, 300.0))
    >>> bool(abs(T - 300.0) < 1e-9)
    True
    """

    def __init__(self, gas_constant, gamma, sutherland_b, sutherland_s):
        self.gas_constant = float(gas_constant)
        self.gamma = float(gamma)
        self.sutherland_b = float(sutherland_b)
        self.sutherland_s = float(sutherland_s)
        self.cp = self.gamma * self.gas_constant / (self.gamma - 1.0)

    def viscosity(self, T):
        """Return the Sutherland viscosity at temperature ``T`` (Pa s).

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Dynamic viscosity (Pa s).
        """
        return self.sutherland_b * T**1.5 / (self.sutherland_s + T)

    def point(self, p, T):
        """Return the closed-form state functions at ``(p, T)``.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        GasPoint
            Values and partial derivatives at ``(p, T)``.

        Raises
        ------
        GasRangeError
            ``S + T <= 0``: the law has no positive viscosity there.
        """
        R = self.gas_constant
        b = self.sutherland_b
        S = self.sutherland_s
        if not S + T > 0.0:
            raise GasRangeError(
                f"the Sutherland law of the perfect gas (S = {S:.6g} K) has no "
                f"positive viscosity at T = {T:.6g} K",
                p,
                T,
                "below",
            )
        rho = p / (R * T)
        return GasPoint(
            p,
            T,
            rho,
            1.0 / (R * T),
            -rho / T,
            self.cp * T,
            self.cp,
            0.0,
            b * T**1.5 / (S + T),
            0.0,
            (b / 2.0) * math.sqrt(T) * (3.0 * S + T) / (S + T) ** 2,
        )

    def entropy(self, p, T):
        """Return ``cp ln T - R ln p`` (J/(kg K)).

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Specific entropy (J/(kg K)).
        """
        return self.cp * math.log(T) - self.gas_constant * math.log(p)

    def temperature_ph(self, p, h, guess=None):
        """Return ``h / cp`` (K); ``p`` and ``guess`` are not used.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        h : float
            Specific enthalpy (J/kg).
        guess : float, optional
            Not used.

        Returns
        -------
        float
            Temperature (K).
        """
        return h / self.cp

    def temperature_ps(self, p, s, guess=None):
        """Return ``exp((s + R ln p) / cp)`` (K); ``guess`` is not used.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        s : float
            Specific entropy (J/(kg K)), with the reference of :meth:`entropy`.
        guess : float, optional
            Not used.

        Returns
        -------
        float
            Temperature (K).
        """
        return math.exp((s + self.gas_constant * math.log(p)) / self.cp)


def _is_single_phase(state, saturation, gas_composition, eos, pure, limits, p, T):
    """Return whether ``(p, T)`` is a single-phase state of the gas.

    Parameters
    ----------
    state : ccp.State
        State with the table phase imposed; it is updated to ``(p, T)``.
    saturation : ccp.State
        Free state (no imposed phase) used for the saturation flash of a pure
        fluid.
    gas_composition : dict
        Gas composition as ``{component: molar_fraction}``.
    eos : str
        Equation-of-state backend of ``ccp``.
    pure : bool
        Whether the gas is a pure fluid.
    limits : tuple of float or None
        Critical pressure (Pa), triple-point pressure (Pa) and triple-point
        temperature (K) of a pure fluid.
    p : float
        Pressure (Pa).
    T : float
        Temperature (K).

    Returns
    -------
    bool
        For a pure fluid, ``p >= p_crit``; below the triple-point pressure,
        ``T`` above the triple-point temperature; otherwise ``T`` above the
        saturation temperature at ``p``. For a mixture, the imposed root equals
        the free flash, or the free flash is a spurious two-phase split.
    """
    if pure:
        p_crit, p_triple, T_triple = limits
        if p >= p_crit:
            return True
        if p <= p_triple:
            # below the triple point the gas borders the solid, whose sublimation
            # temperature lies below T_triple; the saturation flash has no
            # vapour-liquid line there (REFPROP raises, HEOS extrapolates it)
            return T > T_triple
        try:
            coolprop.AbstractState.update(saturation, coolprop.PQ_INPUTS, p, 1.0)
            T_sat = coolprop.AbstractState.T(saturation)
        except ValueError:
            return False
        return T > T_sat
    try:
        state.update(p=p, T=T)
        rho_imposed = state.rho().m
        # phase=False: under ccp 0.5.1 a state built without phase may get "gas"
        # imposed and then return a metastable root after update.
        free = ccp.State(p=p, T=T, fluid=gas_composition, EOS=eos, phase=False)
        rho_free = free.rho().m
    except ValueError:
        return False
    deviation = abs(rho_imposed / rho_free - 1.0)
    if deviation <= _PHASE_RTOL:
        return True
    try:
        quality = free.Q()
        rho_liquid = free.saturated_liquid_keyed_output(coolprop.iDmass)
        rho_vapor = free.saturated_vapor_keyed_output(coolprop.iDmass)
    except ValueError:
        return False
    split = abs(rho_liquid - rho_vapor) <= _SPLIT_RTOL * max(
        abs(rho_liquid), abs(rho_vapor)
    )
    return 0.0 <= quality <= 1.0 and split and deviation <= _SPURIOUS_RTOL


def _temperature_on(state, p, kind, target, guess):
    """Return the temperature at ``p`` where the entropy or enthalpy is ``target``.

    Parameters
    ----------
    state : ccp.State
        State with the table phase imposed; it is updated in place.
    p : float
        Pressure (Pa).
    kind : str
        ``"s"`` for a target entropy (J/(kg K)), ``"h"`` for a target enthalpy
        (J/kg).
    target : float
        Target value.
    guess : float
        Starting temperature (K).

    Returns
    -------
    float
        Temperature (K).
    """
    T = guess
    for _ in range(50):
        state.update(p=p, T=T)
        # the AbstractState methods: ccp's cp() calls REFPROP directly when cp < 0
        cp = coolprop.AbstractState.cpmass(state)
        if kind == "s":
            value = coolprop.AbstractState.smass(state)
        else:
            value = coolprop.AbstractState.hmass(state)
        if not (math.isfinite(cp) and cp > 0.0 and math.isfinite(value)):
            raise RuntimeError(
                f"the equation of state gives no stable state at p = {p:.6g} Pa, "
                f"T = {T:.6g} K (cp = {cp:.6g} J/(kg K))"
            )
        if kind == "s":
            step = (target - value) * T / cp
        else:
            step = (target - value) / cp
        T += step
        if not (math.isfinite(T) and T > 0.0):
            raise RuntimeError(
                f"the temperature iteration at p = {p:.6g} Pa left the positive "
                "temperatures"
            )
        if abs(step) < 1e-10 * T:
            return T
    raise RuntimeError(f"the temperature iteration at p = {p:.6g} Pa did not converge")


def _isenthalp_at(state, p, h_target, T_cold):
    """Return the temperature at ``p`` where the enthalpy is ``h_target``.

    The root is bracketed above ``T_cold``, whose enthalpy is lower, and found by
    Newton steps that fall back to bisection when they leave the bracket. A plain
    Newton iteration oscillates across the peak of ``cp`` near the
    pseudo-critical line of a dense fluid.

    Parameters
    ----------
    state : ccp.State
        State of the table; it is updated in place.
    p : float
        Pressure (Pa).
    h_target : float
        Target specific enthalpy (J/kg).
    T_cold : float
        Temperature at which the enthalpy is below ``h_target`` (K).

    Returns
    -------
    float
        Temperature (K).
    """

    def evaluate(T):
        state.update(p=p, T=T)
        cp = coolprop.AbstractState.cpmass(state)
        h = coolprop.AbstractState.hmass(state)
        if not (math.isfinite(cp) and cp > 0.0 and math.isfinite(h)):
            raise RuntimeError(
                f"the equation of state gives no stable state at p = {p:.6g} Pa, "
                f"T = {T:.6g} K (cp = {cp:.6g} J/(kg K))"
            )
        return h, cp

    low = high = T_cold
    h, cp = evaluate(high)
    if h > h_target:
        raise RuntimeError(
            f"the enthalpy at p = {p:.6g} Pa, T = {T_cold:.6g} K is above the target"
        )
    if h == h_target:
        return T_cold
    increment = 25.0
    for _ in range(20):
        if h >= h_target:
            break
        low, h_low, cp_low = high, h, cp
        high += increment
        increment *= 2.0
        h, cp = evaluate(high)
    else:
        raise RuntimeError(
            f"the temperature iteration at p = {p:.6g} Pa found no bracket"
        )
    T, h, cp = low, h_low, cp_low
    for _ in range(100):
        if h < h_target:
            low = T
        else:
            high = T
        candidate = T + (h_target - h) / cp
        if not low < candidate < high:
            candidate = 0.5 * (low + high)
        if abs(candidate - T) < 1e-10 * T or high - low < 1e-10 * T:
            return candidate
        T = candidate
        h, cp = evaluate(T)
    raise RuntimeError(f"the temperature iteration at p = {p:.6g} Pa did not converge")


def _isentrope_at(state, s_in, p_from, T_from, p):
    """Return the temperature of the isentrope ``s_in`` at ``p``.

    The isentrope is followed from the known point ``(p_from, T_from)`` in
    geometric pressure steps, each Newton iteration starting from the previous
    result. A Newton iteration started far above the root can overshoot into the
    metastable region and diverge.

    Parameters
    ----------
    state : ccp.State
        State with the table phase imposed; it is updated in place.
    s_in : float
        Specific entropy of the isentrope (J/(kg K)).
    p_from : float
        Pressure of the known point (Pa).
    T_from : float
        Temperature of the known point (K).
    p : float
        Pressure at which the temperature is wanted (Pa).

    Returns
    -------
    float
        Temperature (K).
    """
    T = T_from
    for k in range(1, _ISENTROPE_SUBSTEPS + 1):
        if k == _ISENTROPE_SUBSTEPS:
            p_step = p
        else:
            p_step = p_from * (p / p_from) ** (k / _ISENTROPE_SUBSTEPS)
        T = _temperature_on(state, p_step, "s", s_in, T)
    return T


class TabulatedGas(StateGas):
    """Tabulate the state functions of a gas on a sheared ``(p, T)`` grid.

    Bicubic interpolating splines of density, enthalpy, entropy and viscosity are
    fitted on a grid whose node ``(i, j)`` lies at ``(pressure[i], lower[i] +
    offset[j])``: the cold edge of the table is the temperature ``lower(p)``,
    interpolated by a cubic spline, and the table spans ``offset[-1]`` above it at
    every pressure. A constant ``lower`` gives a rectangular grid. The partial
    derivatives are those of the splines, so each state function is consistent
    with its derivatives. A state outside the table raises
    :class:`GasRangeError`; the table is never extrapolated.

    Parameters
    ----------
    pressure : array_like
        Strictly increasing pressure nodes (Pa), at least 4.
    offset : array_like
        Strictly increasing temperature offsets above the cold edge (K), at least
        4, starting at 0.0.
    lower : array_like
        Cold-edge temperature at each pressure node (K), shape ``(n_p,)``.
    rho : array_like
        Density at the nodes (kg/m^3), shape ``(n_p, n_T)``.
    h : array_like
        Specific enthalpy at the nodes (J/kg), shape ``(n_p, n_T)``.
    s : array_like
        Specific entropy at the nodes (J/(kg K)), shape ``(n_p, n_T)``.
    mu : array_like
        Dynamic viscosity at the nodes (Pa s), shape ``(n_p, n_T)``.

    Examples
    --------
    >>> import numpy as np
    >>> air = SutherlandGas(287.0, 1.4, 1.458e-6, 110.4)
    >>> def functions(p, T):
    ...     point = air.point(p, T)
    ...     return point.rho, point.h, air.entropy(p, T), point.mu
    >>> table = TabulatedGas.from_function(
    ...     functions, np.linspace(2e5, 7.5e5, 41), np.linspace(240.0, 340.0, 41)
    ... )
    >>> bool(abs(table.point(4e5, 300.0).rho / air.point(4e5, 300.0).rho - 1) < 1e-8)
    True
    """

    def __init__(self, pressure, offset, lower, rho, h, s, mu):
        pressure = np.array(pressure, dtype=float)
        offset = np.array(offset, dtype=float)
        lower = np.array(lower, dtype=float)
        for name, axis in (("pressure", pressure), ("offset", offset)):
            if (
                axis.ndim != 1
                or axis.size < 4
                or not np.all(np.isfinite(axis))
                or not np.all(np.diff(axis) > 0.0)
            ):
                raise ValueError(
                    f"{name} must be a strictly increasing 1-D array with at least "
                    "4 finite points"
                )
        if offset[0] != 0.0:
            raise ValueError(f"offset must start at 0.0, not {float(offset[0])!r}")
        if (
            lower.shape != pressure.shape
            or not np.all(np.isfinite(lower))
            or not np.all(lower > 0.0)
        ):
            raise ValueError(
                f"lower must have shape {pressure.shape} with finite positive values"
            )
        shape = (pressure.size, offset.size)
        self._splines = {}
        for name, values in (("rho", rho), ("h", h), ("s", s), ("mu", mu)):
            values = np.array(values, dtype=float)
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, not {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} holds non-finite values")
            self._splines[name] = RectBivariateSpline(
                pressure, offset, values, kx=3, ky=3, s=0
            )
        self._lower = CubicSpline(pressure, lower)
        self.pressure = pressure
        self.offset = offset
        self.lower = lower
        self.p_min = float(pressure[0])
        self.p_max = float(pressure[-1])
        self.offset_max = float(offset[-1])
        self.composition = None
        self.eos = None
        self.margins = None
        self.inlet_pressure = None
        self.inlet_temperature = None
        self.outlet_pressure = None
        self.n_pressure = None
        self.n_temperature = None
        self.floor_raised = None

    @classmethod
    def from_function(cls, function, pressure, temperature):
        """Tabulate an analytic gas on a rectangular grid.

        Parameters
        ----------
        function : callable
            ``function(p, T) -> (rho, h, s, mu)`` in SI units.
        pressure : array_like
            Strictly increasing pressure nodes (Pa), at least 4.
        temperature : array_like
            Strictly increasing temperature nodes (K), at least 4.

        Returns
        -------
        TabulatedGas
            Table with a constant cold edge ``temperature[0]``. It cannot widen.
        """
        pressure = np.array(pressure, dtype=float)
        temperature = np.array(temperature, dtype=float)
        values = np.array(
            [[function(float(p), float(T)) for T in temperature] for p in pressure],
            dtype=float,
        ).reshape(pressure.size, temperature.size, 4)
        table = cls(
            pressure,
            temperature - temperature[0],
            np.full(pressure.size, temperature[0]),
            values[:, :, 0],
            values[:, :, 1],
            values[:, :, 2],
            values[:, :, 3],
        )
        table.floor_raised = False
        return table

    @classmethod
    def from_composition(
        cls,
        gas_composition,
        inlet_pressure,
        inlet_temperature,
        outlet_pressure,
        eos=None,
        margins=None,
        n_pressure=_TABLE_POINTS,
        n_temperature=_TABLE_POINTS,
    ):
        """Tabulate the equation of state of ``ccp`` over the range of a seal.

        The table spans the pressures from ``p_low`` to ``p_high`` times the inlet
        pressure. Its cold edge is the inlet isentrope, ``t_below`` colder, and it
        spans up to ``t_above`` above the warmer of the inlet temperature and the
        inlet isenthalp at ``p_low``. ``p_low`` is ``margins["p_low"]`` times the
        outlet pressure, raised when needed (the floor) to the lowest pressure at
        which the cold edge is a single-phase gas. The inlet, the floor and the
        cold edge at every pressure node are checked to be single-phase: for a
        pure fluid, a pressure above the critical one, a temperature above the
        saturation temperature, or, below the triple-point pressure, a
        temperature above the triple-point one; for a mixture, the density of
        the table root agrees with that of a free flash, or the free flash is a
        spurious two-phase split. The nodes of a mixture are filled with the
        ``"supercritical_gas"`` phase imposed, those of a pure fluid by free flash
        (above the critical pressure and below the critical temperature,
        REFPROP's imposed root is the vapour-like one, not the stable one).

        Parameters
        ----------
        gas_composition : dict
            Gas composition as ``{component: molar_fraction}``.
        inlet_pressure : float
            Inlet pressure (Pa).
        inlet_temperature : float
            Inlet temperature (K).
        outlet_pressure : float
            Outlet pressure (Pa).
        eos : str, optional
            Equation-of-state backend of ``ccp`` (``"REFPROP"`` or ``"HEOS"``).
            Default is ``ccp.config.EOS``.
        margins : dict, optional
            Overrides of the table margins ``p_low`` (fraction of the outlet
            pressure, default 0.9), ``p_high`` (fraction of the inlet pressure,
            default 1.001), ``t_below`` (K, default 5.0) and ``t_above`` (K,
            default 25.0).
        n_pressure : int, optional
            Number of pressure nodes. Default is 41.
        n_temperature : int, optional
            Number of temperature nodes. Default is 41.

        Returns
        -------
        TabulatedGas
            The table, with ``composition``, ``eos``, ``margins``,
            ``inlet_pressure``, ``inlet_temperature``, ``outlet_pressure``,
            ``n_pressure``, ``n_temperature`` and ``floor_raised`` set.

        Raises
        ------
        ValueError
            The inlet is not a single-phase gas, no pressure of the seal keeps the
            cold edge single-phase, the cold edge reaches a liquid or two-phase
            state, or the equation of state fails or gives a non-finite value
            (including the viscosity) at a node.

        Notes
        -----
        The floor is conservative: the friction-heated flow is warmer than the
        isentrope at the same pressure, so a seal can be refused although its flow
        would stay a gas. For a mixture, the cold edge is checked at the pressure
        nodes only; a narrow two-phase band between nodes, or between the edge and
        the isentrope near the mixture critical point, is not detected. Under
        REFPROP, a free flash can report a spurious two-phase split whose saturated
        densities differ by more than ccp's criterion (1e-2), also far from the
        critical point; such a state is refused although HEOS calls it single phase
        (the AGA-8 Ekofisk gas from 9 to 7 MPa at 324 K is refused at 8.06 MPa,
        310.6 K). For a pure fluid, a state above the critical pressure counts as
        single phase also below the critical temperature. Below the triple-point
        pressure the gas borders the solid; the test uses the triple-point
        temperature, which lies above the sublimation temperature, so it is
        conservative there (CO2 below 0.518 MPa needs more than 216.6 K). Next to
        the critical point the splines can lose thermodynamic stability, and a flow
        that reaches such a state raises a ``ValueError`` from :class:`GasPoint`;
        their interpolation error also grows there (bicubic splines across the
        steep pseudo-critical change of the properties). The viscosity
        model of HEOS gives no value at some dense, cold states of natural-gas
        mixtures; such a table is refused (REFPROP builds it).

        Examples
        --------
        >>> table = TabulatedGas.from_composition({"methane": 1.0}, 20e6, 300.0, 10e6)
        >>> table.floor_raised
        False
        >>> bool(table.lower_temperature(18e6) < 300.0 < table.upper_temperature(18e6))
        True
        """
        if eos is None:
            eos = ccp.config.EOS
        m = dict(_TABLE_MARGINS, **(margins or {}))
        p_in = float(inlet_pressure)
        T_in = float(inlet_temperature)
        p_out = float(outlet_pressure)
        inlet_message = (
            f"the inlet state p = {p_in:.6g} Pa, T = {T_in:.6g} K is not a "
            "single-phase gas for the equation of state; the hole-pattern model "
            "needs a gas over the whole seal"
        )
        try:
            # phase=False: under ccp 0.5.1 a state built without phase may get "gas"
            # imposed and then return a metastable root after update.
            inlet = ccp.State(
                p=p_in, T=T_in, fluid=gas_composition, EOS=eos, phase=False
            )
            h_in = inlet.h().m
            s_in = inlet.s().m
            pure = len(inlet.fluid) == 1
            if pure:
                limits = (
                    inlet.p_critical().m,
                    coolprop.AbstractState.keyed_output(inlet, coolprop.iP_triple),
                    coolprop.AbstractState.keyed_output(inlet, coolprop.iT_triple),
                )
                # a pure fluid is filled by free flash: above the critical
                # pressure and below the critical temperature REFPROP's imposed
                # "supercritical_gas" root is the wrong (vapour-like) one
                state = ccp.State(
                    p=p_in, T=T_in, fluid=gas_composition, EOS=eos, phase=False
                )
            else:
                limits = None
                state = ccp.State(
                    p=p_in, T=T_in, fluid=gas_composition, EOS=eos, phase=_TABLE_PHASE
                )
        except ValueError as err:
            raise ValueError(inlet_message) from err
        saturation = inlet

        def single_phase(p, T):
            return _is_single_phase(
                state, saturation, gas_composition, eos, pure, limits, p, T
            )

        if not single_phase(p_in, T_in):
            raise ValueError(inlet_message)

        def passes(p, p_from, T_from):
            try:
                T = _isentrope_at(state, s_in, p_from, T_from, p)
            except (ValueError, RuntimeError):
                return False, None
            return single_phase(p, T - m["t_below"]), T

        p_low = m["p_low"] * p_out
        floor_raised = False
        if not passes(p_low, p_in, T_in)[0]:
            if not passes(p_in, p_in, T_in)[0]:
                raise ValueError(
                    "no pressure between the outlet and the inlet pressure keeps the "
                    f"inlet isentrope, {m['t_below']:g} K colder, a single-phase gas; "
                    "the hole-pattern model needs a gas over the whole seal"
                )
            low, high, T_high = p_low, p_in, T_in
            for _ in range(_FLOOR_BISECTIONS):
                middle = 0.5 * (low + high)
                ok, T = passes(middle, high, T_high)
                if ok:
                    high, T_high = middle, T
                else:
                    low = middle
            p_low = high
            floor_raised = True

        pressure = np.linspace(p_low, m["p_high"] * p_in, int(n_pressure))
        isentrope = np.empty(pressure.size)
        p_from, isentrope_from, T = p_in, T_in, T_in
        for i in range(pressure.size - 1, -1, -1):
            p = float(pressure[i])
            if i == 0 and floor_raised:
                # the isentrope at the floor is known from the floor search
                T = T_high
            else:
                try:
                    T = _temperature_on(state, p, "s", s_in, T)
                except (ValueError, RuntimeError):
                    # continuation in substeps, as in the floor search; a single
                    # Newton iteration can diverge near the critical point
                    try:
                        T = _isentrope_at(state, s_in, p_from, isentrope_from, p)
                    except (ValueError, RuntimeError) as err:
                        raise ValueError(
                            f"the equation of state failed at p = {p:.6g} Pa on the "
                            f"inlet isentrope ({err}); the property table of the "
                            "hole-pattern seal must lie in the single-phase gas region"
                        ) from err
            isentrope[i] = T
            p_from, isentrope_from = p, T
        lower = isentrope - m["t_below"]

        try:
            T_isenthalpic = _isenthalp_at(
                state, float(p_low), h_in, lower[0] + m["t_below"]
            )
        except (ValueError, RuntimeError) as err:
            raise ValueError(
                f"the equation of state failed at p = {p_low:.6g} Pa on the inlet "
                f"isenthalp ({err}); the property table of the hole-pattern seal "
                "must lie in the single-phase gas region"
            ) from err
        t_high = max(T_in, T_isenthalpic) + m["t_above"]
        offset = np.linspace(0.0, t_high - lower[0], int(n_temperature))

        for i in range(pressure.size):
            p, T = float(pressure[i]), float(lower[i])
            if not single_phase(p, T):
                raise ValueError(
                    "the property table of the hole-pattern seal reaches a liquid or "
                    f"two-phase state at p = {p:.6g} Pa, T = {T:.6g} K; the expansion "
                    "through the seal may cross the phase boundary, which the "
                    "single-phase model does not support"
                )

        values = np.empty((pressure.size, offset.size, 4))
        for i in range(pressure.size):
            for j in range(offset.size):
                p, T = float(pressure[i]), float(lower[i] + offset[j])
                try:
                    state.update(p=p, T=T)
                    values[i, j, :3] = (
                        coolprop.AbstractState.rhomass(state),
                        coolprop.AbstractState.hmass(state),
                        coolprop.AbstractState.smass(state),
                    )
                except (ValueError, RuntimeError) as err:
                    raise ValueError(
                        f"the equation of state failed at p = {p:.6g} Pa, "
                        f"T = {T:.6g} K ({err}); the property table of the "
                        "hole-pattern seal must lie in the single-phase gas region"
                    ) from err
                if not np.all(np.isfinite(values[i, j, :3])):
                    raise ValueError(
                        f"the equation of state failed at p = {p:.6g} Pa, "
                        f"T = {T:.6g} K (non-finite density, enthalpy or entropy); "
                        "the property table of the hole-pattern seal must lie in "
                        "the single-phase gas region"
                    )
                try:
                    values[i, j, 3] = coolprop.AbstractState.viscosity(state)
                except ValueError:
                    values[i, j, 3] = math.nan
                if not math.isfinite(values[i, j, 3]):
                    raise ValueError(
                        f"the viscosity model of the {eos} backend gives no value at "
                        f"p = {p:.6g} Pa, T = {T:.6g} K (HEOS fails for some "
                        "natural-gas mixtures at dense, cold states); the property "
                        "table of the hole-pattern seal needs the viscosity over its whole "
                        "single-phase gas region"
                    )

        table = cls(
            pressure,
            offset,
            lower,
            values[:, :, 0],
            values[:, :, 1],
            values[:, :, 2],
            values[:, :, 3],
        )
        table.composition = dict(gas_composition)
        table.eos = eos
        table.margins = m
        table.inlet_pressure = p_in
        table.inlet_temperature = T_in
        table.outlet_pressure = p_out
        table.n_pressure = int(n_pressure)
        table.n_temperature = int(n_temperature)
        table.floor_raised = floor_raised
        return table

    def _edge(self, p):
        """Return the cold-edge temperature and its slope ``d lower / d p`` at ``p``."""
        knots = self._lower.x
        i = min(max(bisect_right(knots, p) - 1, 0), knots.size - 2)
        c3, c2, c1, c0 = self._lower.c[:, i].tolist()
        d = p - float(knots[i])
        return ((c3 * d + c2) * d + c1) * d + c0, (3.0 * c3 * d + 2.0 * c2) * d + c1

    def _pressure_error(self, p, T):
        state = f"p = {p:.6g} Pa" if T is None else f"p = {p:.6g} Pa, T = {T:.6g} K"
        return GasRangeError(
            f"state {state} is outside the property table "
            f"({self.p_min:.6g} to {self.p_max:.6g} Pa); the table is not "
            "extrapolated",
            p,
            T,
            "below" if p < self.p_min else "above",
        )

    def _band_error(self, p, T, t_lo):
        return GasRangeError(
            f"state p = {p:.6g} Pa, T = {T:.6g} K is outside the property table (at "
            f"this pressure {t_lo:.6g} to {t_lo + self.offset_max:.6g} K); the table "
            "is not extrapolated",
            p,
            T,
            "below" if T < t_lo else "above",
        )

    def _check(self, p, T):
        """Return ``(T - lower(p), d lower / d p)``, or raise for a state outside."""
        if not self.p_min <= p <= self.p_max:
            raise self._pressure_error(p, T)
        t_lo, slope = self._edge(p)
        xi = T - t_lo
        if not 0.0 <= xi <= self.offset_max:
            raise self._band_error(p, T, t_lo)
        return xi, slope

    def lower_temperature(self, p):
        """Return the cold-edge temperature of the table at ``p`` (K).

        Parameters
        ----------
        p : float
            Pressure (Pa).

        Returns
        -------
        float
            Lowest temperature of the table at ``p`` (K).
        """
        if not self.p_min <= p <= self.p_max:
            raise self._pressure_error(p, None)
        return self._edge(p)[0]

    def upper_temperature(self, p):
        """Return the highest temperature of the table at ``p`` (K).

        Parameters
        ----------
        p : float
            Pressure (Pa).

        Returns
        -------
        float
            Highest temperature of the table at ``p`` (K).
        """
        return self.lower_temperature(p) + self.offset_max

    def point(self, p, T):
        """Return the interpolated state functions at ``(p, T)``.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        GasPoint
            Values and partial derivatives of the splines at ``(p, T)``.

        Raises
        ------
        GasRangeError
            The state lies outside the table.
        """
        xi, slope = self._check(p, T)
        rho = self._splines["rho"]
        h = self._splines["h"]
        mu = self._splines["mu"]
        rho_xi = float(rho.ev(p, xi, dy=1))
        h_xi = float(h.ev(p, xi, dy=1))
        mu_xi = float(mu.ev(p, xi, dy=1))
        return GasPoint(
            p,
            T,
            float(rho.ev(p, xi)),
            float(rho.ev(p, xi, dx=1)) - slope * rho_xi,
            rho_xi,
            float(h.ev(p, xi)),
            h_xi,
            float(h.ev(p, xi, dx=1)) - slope * h_xi,
            float(mu.ev(p, xi)),
            float(mu.ev(p, xi, dx=1)) - slope * mu_xi,
            mu_xi,
        )

    def entropy(self, p, T):
        """Return the interpolated specific entropy at ``(p, T)`` (J/(kg K)).

        Parameters
        ----------
        p : float
            Pressure (Pa).
        T : float
            Temperature (K).

        Returns
        -------
        float
            Specific entropy (J/(kg K)).
        """
        xi = self._check(p, T)[0]
        return float(self._splines["s"].ev(p, xi))

    def _inverted(self, p, guess, step_at, name, target):
        t_lo = self.lower_temperature(p)
        t_hi = t_lo + self.offset_max
        T = min(max(guess, t_lo), t_hi)
        for _ in range(60):
            step = step_at(T)
            if abs(step) < 1e-12 * T:
                return T + step
            clamped = min(max(T + step, t_lo), t_hi)
            if clamped == T:
                raise self._band_error(p, T + step, t_lo)
            T = clamped
        unit = "J/kg" if name == "h" else "J/(kg K)"
        raise RuntimeError(
            f"temperature_p{name} did not converge at p={p:g} Pa, "
            f"{name}={target:g} {unit}"
        )

    def temperature_ph(self, p, h, guess):
        """Return the temperature at pressure ``p`` and specific enthalpy ``h``.

        The Newton iterates are kept inside the table band at ``p``.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        h : float
            Specific enthalpy (J/kg).
        guess : float
            Starting temperature (K); clamped into the band.

        Returns
        -------
        float
            Temperature (K).

        Raises
        ------
        GasRangeError
            ``p`` is outside the table, or the root lies outside the band.
        """

        def step_at(T):
            point = self.point(p, T)
            return (h - point.h) / point.cp

        return self._inverted(p, guess, step_at, "h", h)

    def temperature_ps(self, p, s, guess):
        """Return the temperature at pressure ``p`` and specific entropy ``s``.

        The Newton iterates are kept inside the table band at ``p``.

        Parameters
        ----------
        p : float
            Pressure (Pa).
        s : float
            Specific entropy (J/(kg K)).
        guess : float
            Starting temperature (K); clamped into the band.

        Returns
        -------
        float
            Temperature (K).

        Raises
        ------
        GasRangeError
            ``p`` is outside the table, or the root lies outside the band.
        """

        def step_at(T):
            return (s - self.entropy(p, T)) * T / self.point(p, T).cp

        return self._inverted(p, guess, step_at, "s", s)

    def widened(self, error):
        """Return the table rebuilt with the side that ``error`` left widened.

        The pressure range is tested first, because the temperature band depends
        on the pressure: a state below the table lowers ``p_low`` by the factor
        0.9, a state above it triples the excess of ``p_high`` over 1, a state
        hotter than the band triples ``t_above`` and a colder one triples
        ``t_below``. Without a known state both temperature margins are tripled.

        Parameters
        ----------
        error : GasRangeError
            The error raised for a state outside the table.

        Returns
        -------
        TabulatedGas
            The rebuilt table, with the same inputs, backend and grid sizes.

        Raises
        ------
        GasRangeError
            ``error`` itself, for a table not built by :meth:`from_composition`.
        ValueError
            The state lies below a raised floor: the expansion may cross the phase
            boundary.
        """
        if self.composition is None:
            raise error
        m = dict(self.margins)
        p = error.pressure
        T = error.temperature
        if p is not None and p < self.p_min and self.floor_raised:
            raise ValueError(
                f"the hole-pattern flow asks for a state at p = {p:.6g} Pa, below the "
                f"lowest pressure ({self.p_min:.6g} Pa) at which the inlet isentrope, "
                f"{m['t_below']:g} K colder, is a single-phase gas; the expansion "
                "through the seal may cross the phase boundary, which the "
                "single-phase model does not support"
            ) from error
        if p is not None and p < self.p_min:
            m["p_low"] *= _TABLE_P_LOW_WIDENING
        elif p is not None and p > self.p_max:
            m["p_high"] = 1.0 + (m["p_high"] - 1.0) * _TABLE_WIDENING
        elif p is not None and T is not None and T > self.upper_temperature(p):
            m["t_above"] *= _TABLE_WIDENING
        elif p is not None and T is not None and T < self.lower_temperature(p):
            m["t_below"] *= _TABLE_WIDENING
        else:
            m["t_above"] *= _TABLE_WIDENING
            m["t_below"] *= _TABLE_WIDENING
        return TabulatedGas.from_composition(
            self.composition,
            self.inlet_pressure,
            self.inlet_temperature,
            self.outlet_pressure,
            eos=self.eos,
            margins=m,
            n_pressure=self.n_pressure,
            n_temperature=self.n_temperature,
        )


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

    def cg2(self, area, shaft_radius, p, T, rho):
        """Return the swirl-velocity coefficient of the cavity continuity equation (touch point 9).

        Linearizing the circumferential term ``d(rho V area) / (shaft_radius d theta)``
        with respect to the swirl velocity ``V`` gives ``rho * area / shaft_radius``.
        This backend returns the legacy perfect-gas form
        ``(p / shaft_radius) * area / (R T)`` and does not use ``rho``.

        Parameters
        ----------
        area : float
            Cavity cross-section area (m**2).
        shaft_radius : float
            Shaft radius (m).
        p : float
            Cavity pressure (Pa).
        T : float
            Cavity temperature (deg K).
        rho : float
            Cavity density of the base flow (kg/m**3).

        Returns
        -------
        float
            Coefficient of the swirl-velocity perturbation (kg/m**2).
        """
        return (p / shaft_radius) * (area / (self.R * T))


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
                # (d rho / d p)_T = rho * kappa_T, read analytically at the same
                # isentrope state, avoiding extra equation-of-state evaluations.
                kappa_t = state.isothermal_compressibility()
                kappa_t = kappa_t.m if hasattr(kappa_t, "m") else kappa_t
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
            drhodp_grid[k] = rho_k * kappa_t

        self.p_grid = p_grid
        self.T_grid = T_grid
        self.rho_grid = rho_grid
        self.a_grid = a_grid
        self.h_grid = h_grid
        self.drhodp_grid = drhodp_grid
        self.p_min = p_grid[0]

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

        The critical ratio is nearly pressure-independent, so when ``p_prev`` lies
        outside the table it is evaluated at the inlet reference instead.
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

    def cg2(self, area, shaft_radius, p, T, rho):
        """Return the swirl-velocity coefficient ``rho * area / shaft_radius``.

        Parameters and return value as in :meth:`IdealGas.cg2`. The perfect-gas
        form ``(p / shaft_radius) * cg0`` is not used: with the real ``cg0`` it
        carries the factor ``p * kappa_T``, which is 1 only for a perfect gas.
        """
        return rho * area / shaft_radius
