import math
from functools import partial
from warnings import catch_warnings, simplefilter, warn

import numpy as np
from scipy.optimize import brentq, curve_fit

from ross import SealElement
from ross.seals.gas_model import (
    R_UNIVERSAL,
    GasRangeError,
    SutherlandGas,
    TabulatedGas,
    extract_gas_properties,
)
from ross.seals.solver_tools import solve_frequencies
from ross.units import check_units

__all__ = ["HolePatternSeal"]

MOODY_FRICTION_COEFFICIENT = 1.375e-3
MOODY_ROUGHNESS_SCALE = 1.0e4
MOODY_VISCOSITY_SCALE = 5.0e5
CHOKED_MZ2_LIMIT = 0.98
COEFFICIENT_NAMES = (
    "kxx",
    "kyy",
    "kxy",
    "kyx",
    "cxx",
    "cyy",
    "cxy",
    "cyx",
    "mxx",
    "myy",
    "mxy",
    "myx",
)
PARALLEL_THRESHOLD = 4

# Relative mass-flux residual at which the isentropic entrance or exit relation
# counts as solved; the slope of the flux vanishes at Mach 1.
_FLUX_TOLERANCE = 1.0e-12
_FIRST_STEP_FLOOR = 1.0e-8
_TABLE_ATTEMPTS = 5
_ZERO_FREQUENCY_STEP = 1.0e-9
_BISECTION_RESOLUTION = 4.0e-16


def _without_warnings(solve, argument):
    """Call ``solve(argument)`` with warnings silenced (pool workers).

    A failed point carries its message in the ``"error"`` key of its result, and
    the parent process warns with it: warnings raised in a worker never reach the
    filters of the parent.
    """
    with catch_warnings():
        simplefilter("ignore")
        return solve(argument)


class _Overshoot(Exception):
    """A trial integration left the physical range (flow beyond the choke limit)."""


class ChokedAtFirstGuess(RuntimeError):
    """The seal flow is choked even for the smallest trial mass flow."""


class HolePatternBaseState:
    """Steady, centred flow along the seal.

    Arrays have ``nz + 1`` entries, one per axial station.

    Attributes
    ----------
    mass_flux : float
        Mass flux ``G = rho u`` (kg/(m^2 s)).
    leakage : float
        ``G * 2 pi R H`` (kg/s).
    z : ndarray
        Axial coordinate of the stations (m).
    p, T, rho, u, w : ndarray
        Pressure (Pa), temperature (K), density (kg/m^3), axial and
        circumferential velocity (m/s).
    mach2 : ndarray
        Axial Mach number squared ``(u / a)^2``.
    choked : bool
        True when the outlet pressure cannot be reached: the state is the
        flow at the choke limit.
    choke_location : str or None
        Where the flow chokes: ``"land"`` (the axial Mach number reaches
        the limit in the land), ``"entrance"`` (no subsonic entrance
        state), ``"exit"`` (the exit relation reaches its sonic flux), or
        None when unchoked.
    p2 : float
        Static pressure after the isentropic entrance acceleration (Pa).
    p5 : float
        Pressure behind the exit loss (Pa); the outlet pressure when
        unchoked.
    iterations : int
        Number of trial integrations of the outlet-pressure search.
    gas : StateGas
        The gas object the state was solved with.
    """

    __slots__ = (
        "mass_flux",
        "leakage",
        "z",
        "p",
        "T",
        "rho",
        "u",
        "w",
        "mach2",
        "choked",
        "choke_location",
        "p2",
        "p5",
        "iterations",
        "gas",
    )

    def __init__(
        self,
        mass_flux,
        leakage,
        z,
        p,
        T,
        rho,
        u,
        w,
        mach2,
        choked,
        p2,
        p5,
        iterations,
        gas,
        choke_location=None,
    ):
        self.mass_flux = mass_flux
        self.leakage = leakage
        self.z = z
        self.p = p
        self.T = T
        self.rho = rho
        self.u = u
        self.w = w
        self.mach2 = mach2
        self.choked = choked
        self.choke_location = choke_location
        self.p2 = p2
        self.p5 = p5
        self.iterations = iterations
        self.gas = gas


class HolePatternSolver:
    """Bulk-flow solver of a hole-pattern annular gas seal.

    Solves the base state for one rotor speed and the first-order
    perturbation for any number of whirl frequencies of that speed. The
    solver holds only plain data and a gas object, so it pickles to the
    worker processes of a multi-speed run.

    Parameters
    ----------
    shaft_radius : float
        Rotor radius (m).
    radial_clearance : float
        Radial clearance (m).
    axial_length : float
        Axial length of the seal (m).
    relative_roughness : float
        Rotor roughness divided by the radial clearance.
    cell_depth : float
        Depth of the cells (m).
    inlet_pressure, outlet_pressure : float
        Boundary pressures (Pa).
    inlet_temperature : float
        Inlet temperature (K).
    preswirl : float
        Inlet swirl ratio.
    entrance_loss_coefficient, exit_loss_coefficient : float
        Entrance and exit loss coefficients.
    excitation_ratio : float
        Default ratio of the whirl frequency to the rotor speed.
    nz : int
        Number of axial intervals.
    max_iterations : int
        Maximum number of trial integrations of the outlet-pressure search.
    tolerance : float
        Accepted outlet-pressure residual, as a fraction of the pressure drop.
    first_step_size : float
        The first trial entrance pressure is ``(1 - first_step_size)`` times
        the inlet pressure.
    gas : StateGas or None
        Gas object (see :mod:`ross.seals.gas_model`). None for a real-gas
        element built from saved coefficients, which cannot be solved.

    Attributes
    ----------
    t, mz2, mt : ndarray
        Station temperature (K), ``(u / a)^2`` and ``w / a`` of the last base
        state solved in this process.
    """

    FAILED_COEFFICIENTS = COEFFICIENT_NAMES + ("seal_leakage",)

    def __init__(
        self,
        shaft_radius,
        radial_clearance,
        axial_length,
        relative_roughness,
        cell_depth,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        preswirl,
        entrance_loss_coefficient,
        exit_loss_coefficient,
        excitation_ratio,
        nz,
        max_iterations,
        tolerance,
        first_step_size,
        gas,
    ):
        self._shaft_radius = shaft_radius
        self.radial_clearance = radial_clearance
        self.axial_length = axial_length
        self.relative_roughness = relative_roughness
        self.cell_depth = cell_depth
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure = outlet_pressure
        self.inlet_temperature = inlet_temperature
        self.preswirl = preswirl
        self.entrance_loss_coefficient = entrance_loss_coefficient
        self.exit_loss_coefficient = exit_loss_coefficient
        self.excitation_ratio = excitation_ratio
        self.nz = nz
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.first_step_size = first_step_size
        self.gas = gas

        self.dz = axial_length / float(nz)
        self.area = 2.0 * np.pi * shaft_radius * radial_clearance

        # Index maps and signs pairing the cosine and sine components of the
        # four simultaneous perturbations: the time derivative of component c
        # is sgn_t[c] * Omega * component i_t[c], the circumferential one
        # sgn_th[c] * component i_th[c].
        self.i_t = np.array([1, 0, 3, 2])
        self.i_th = np.array([2, 3, 0, 1])
        self.sgn_t = np.array([-1.0, 1.0, -1.0, 1.0])
        self.sgn_th = np.array([-1.0, -1.0, 1.0, 1.0])
        self.clearance_perturbation = np.array([0.0, 0.0, 0.0, -1.0])

        self.t = np.zeros(nz + 1)
        self.mz2 = np.zeros(nz + 1)
        self.mt = np.zeros(nz + 1)

    def solve(self, speed, frequency=None):
        """Solve the seal at one rotor speed and one whirl frequency (rad/s).

        Parameters
        ----------
        speed : float
            Rotor speed (rad/s).
        frequency : float, optional
            Whirl frequency (rad/s). Default is ``excitation_ratio * speed``.

        Returns
        -------
        result : dict
            The coefficients of ``COEFFICIENT_NAMES``, ``"seal_leakage"``,
            ``"pressure"`` and ``"choked"`` (see :meth:`solve_row`).
        """
        if frequency is None:
            frequency = self.excitation_ratio * speed
        return self.solve_row(speed, [frequency])[0]

    def solve_row(self, speed, frequencies):
        """Solve one rotor speed at several whirl frequencies (rad/s).

        The base state is solved once and the perturbation is solved for all
        frequencies in one batch. A failure warns and returns NaN for the
        frequencies concerned: a failing base state fails every frequency, a
        non-finite force fails its own frequency only.

        Parameters
        ----------
        speed : float
            Rotor speed (rad/s).
        frequencies : array_like
            Whirl frequencies (rad/s).

        Returns
        -------
        results : list of dict
            One dict per frequency with the coefficients of
            ``COEFFICIENT_NAMES`` (the mass terms are 0.0), ``"seal_leakage"``
            (kg/s), ``"pressure"`` (``nz + 4`` pressures ``[p_in, p2,
            stations, p5]``, Pa) and ``"choked"`` (bool, None on failure).
        """
        if self.gas is None:
            raise RuntimeError(
                "a real-gas HolePatternSeal built from saved coefficients has no "
                "property table; build it without coefficients to solve it"
            )
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        try:
            flow, base = self._solve_base(speed)
            sound = np.array(
                [
                    base.gas.point(float(p), float(T)).sound_speed
                    for p, T in zip(base.p, base.T, strict=True)
                ]
            )
            self.t = base.T.copy()
            self.mz2 = base.mach2.copy()
            self.mt = base.w / sound
            coefficients = flow.coefficients(base, speed, frequencies)
        except (ArithmeticError, RuntimeError, ValueError) as error:
            return [self._failed_result(speed, f, error) for f in frequencies]

        pressure = np.concatenate([[self.inlet_pressure, base.p2], base.p, [base.p5]])
        results = []
        for k, frequency in enumerate(frequencies):
            values = {name: float(coefficients[name][k]) for name in COEFFICIENT_NAMES}
            if not all(np.isfinite(value) for value in values.values()):
                results.append(
                    self._failed_result(
                        speed,
                        frequency,
                        RuntimeError("non-finite coefficients at this whirl frequency"),
                    )
                )
                continue
            values["seal_leakage"] = float(base.leakage)
            values["pressure"] = pressure.copy()
            values["choked"] = bool(base.choked)
            results.append(values)
        return results

    def solve_grid(self, speeds, frequencies, parallel_threshold=PARALLEL_THRESHOLD):
        """Solve a (speed, whirl frequency) grid.

        Returns a nested list ``rows[i][j]`` with the result dict of
        ``speeds[i]`` and ``frequencies[j]``. Speeds are solved in parallel
        when there are more than ``parallel_threshold`` of them.
        """
        return self._solve_speeds(
            partial(self.solve_row, frequencies=frequencies),
            speeds,
            parallel_threshold,
        )

    def _solve_speeds(self, solve, speeds, parallel_threshold):
        """Apply ``solve`` to every speed, in a process pool above the threshold.

        In a pool the workers do not warn; the parent warns for every failed
        point instead, so that warning filters and ``pytest.warns`` see it.
        """
        if len(speeds) <= parallel_threshold:
            return solve_frequencies(solve, speeds, parallel_threshold)
        results = solve_frequencies(
            partial(_without_warnings, solve), speeds, parallel_threshold
        )
        for result in results:
            for point in result if isinstance(result, list) else [result]:
                if point.get("error") is not None:
                    warn(point["error"])
        return results

    def base_state(self, speed):
        """Base state at the rotor ``speed`` (rad/s).

        Uses the same property-table retries as :meth:`solve_row`. Errors
        propagate.

        Returns
        -------
        base : HolePatternBaseState
        """
        if self.gas is None:
            raise RuntimeError(
                "a real-gas HolePatternSeal built from saved coefficients has no "
                "property table; build it without coefficients to solve it"
            )
        return self._solve_base(speed)[1]

    def shear(self, clearance, rho, mu, u, w, surface_speed):
        """Shear stress that the fluid exerts on the walls, resisting the flow (Pa).

        Moody's law with the hydraulic diameter ``2 H``: the rotor with the
        roughness ``relative_roughness``, the stator as a smooth wall with
        twice the viscous term, at a quarter of the Moody value.

        Parameters
        ----------
        clearance : float
            Radial clearance (m).
        rho, mu : float
            Density (kg/m^3) and viscosity (Pa s).
        u, w : float
            Axial and circumferential velocity (m/s).
        surface_speed : float
            Rotor surface speed (m/s).

        Returns
        -------
        tau_x, tau_theta, tau_rotor_theta : float
            Axial and circumferential shear of both walls, and the
            circumferential shear of the rotor alone.
        """
        alpha = MOODY_FRICTION_COEFFICIENT
        us = math.sqrt(u * u + w * w)
        w_rel = w - surface_speed
        ur = math.sqrt(u * u + w_rel * w_rel)
        viscous = MOODY_VISCOSITY_SCALE * mu / (rho * clearance)
        fs = alpha * (1.0 + (2.0 * viscous / us) ** (1.0 / 3.0))
        fr = alpha * (
            1.0
            + (MOODY_ROUGHNESS_SCALE * self.relative_roughness + viscous / ur)
            ** (1.0 / 3.0)
        )
        stator = 0.5 * rho * us * fs / 4.0
        rotor = 0.5 * rho * ur * fr
        return u * (stator + rotor), w * stator + w_rel * rotor, w_rel * rotor

    def _solve_base(self, speed):
        """Base state with property-table retries; returns ``(flow, base)``."""
        gas = self.gas
        for attempt in range(_TABLE_ATTEMPTS):
            try:
                flow = _HolePatternFlow(self, gas)
                return flow, flow.base_state(speed)
            except GasRangeError as error:
                if attempt == _TABLE_ATTEMPTS - 1:
                    raise
                gas = gas.widened(error)

    def _failed_result(self, speed, frequency, error):
        """Warn about a failed solve and return NaN coefficients.

        The result also holds the warning text under ``"error"``.
        """
        message = (
            f"Could not calculate the hole-pattern seal at speed {speed} rad/s "
            f"and whirl frequency {frequency} rad/s; the coefficients for this "
            f"point are set to NaN. Original error: {error}"
        )
        warn(message)
        failed = dict.fromkeys(self.FAILED_COEFFICIENTS, np.nan)
        failed["pressure"] = np.full(self.nz + 4, np.nan)
        failed["choked"] = None
        failed["error"] = message
        return failed


class _HolePatternFlow:
    """Base state and perturbation of one seal with one gas object.

    Created inside a solve and never pickled. Geometry and the wall shear
    come from the solver.

    Parameters
    ----------
    solver : HolePatternSolver
    gas : StateGas
    """

    def __init__(self, solver, gas):
        self.solver = solver
        self.gas = gas
        self._range_error = None
        self.h_in = gas.point(solver.inlet_pressure, solver.inlet_temperature).h
        self.s_in = gas.entropy(solver.inlet_pressure, solver.inlet_temperature)

    def base_rhs(self, mass_flux, surface_speed, p, T, w):
        """Axial derivatives ``(dp/dz, dT/dz, dw/dz)`` and ``(u / a)^2``.

        Continuity ``rho u = G``, axial momentum, circumferential momentum and
        energy with adiabatic walls and the work of the rotor.
        """
        point = self.gas.point(p, T)
        clearance = self.solver.radial_clearance
        rho = point.rho
        u = mass_flux / rho
        tau_x, tau_t, tau_rt = self.solver.shear(
            clearance, rho, point.mu, u, w, surface_speed
        )
        u2 = u * u
        mach2 = u2 * point.rho_p_s
        dw = -tau_t / (clearance * mass_flux)
        heat = -surface_speed * tau_rt / (clearance * mass_flux)
        a11 = 1.0 - u2 * point.rho_p
        a12 = -u2 * point.rho_T
        a21 = point.h_p - u2 * point.rho_p / rho
        a22 = point.cp - u2 * point.rho_T / rho
        b1 = -tau_x / clearance
        b2 = heat - w * dw
        det = a11 * a22 - a12 * a21
        dp = (b1 * a22 - a12 * b2) / det
        dT = (a11 * b2 - a21 * b1) / det
        return dp, dT, dw, mach2

    def _integrate(self, mass_flux, surface_speed, start):
        """March ``(p, T, w)`` from station 0 with classical RK4.

        Returns ``(status, profile)``: status ``"ok"``, ``"choked"`` (the
        choke limit was passed) or ``"range"`` (the gas range was left), and
        the profile of shape ``(nz + 1, 3)``.
        """
        nz = self.solver.nz
        step = self.solver.dz
        half = 0.5 * step
        limit = CHOKED_MZ2_LIMIT
        profile = np.empty((nz + 1, 3))
        p, T, w = start
        profile[0] = start

        def rhs(p, T, w):
            if not (p > 0.0 and T > 0.0):
                raise _Overshoot
            return self.base_rhs(mass_flux, surface_speed, p, T, w)

        try:
            k1 = rhs(p, T, w)
            if k1[3] > limit:
                return "choked", profile
            for iz in range(1, nz + 1):
                k2 = rhs(p + half * k1[0], T + half * k1[1], w + half * k1[2])
                k3 = rhs(p + half * k2[0], T + half * k2[1], w + half * k2[2])
                k4 = rhs(p + step * k3[0], T + step * k3[1], w + step * k3[2])
                if max(k2[3], k3[3], k4[3]) > limit:
                    return "choked", profile
                p += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
                T += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
                w += step * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
                if not (p > 0.0 and T > 0.0):
                    return "choked", profile
                k1 = rhs(p, T, w)
                if k1[3] > limit:
                    return "choked", profile
                profile[iz] = (p, T, w)
        except GasRangeError as error:
            self._range_error = error
            return "range", profile
        except _Overshoot:
            return "choked", profile
        return "ok", profile

    def _isentropic_flux_pressure(
        self, mass_flux, total_enthalpy, entropy, guess_p, guess_t, stagnation
    ):
        """Subsonic static state on an isentrope that carries ``mass_flux``.

        Solves ``rho sqrt(2 (h0 - h)) = G`` for the pressure on the isentrope
        ``s = entropy``: Newton's method first, then a safeguarded search that
        brackets the sonic point and the root. ``stagnation`` is the
        stagnation pressure of the isentrope, the highest pressure asked for
        in the search.

        Returns
        -------
        state : tuple or None
            ``(p, T, u)``, or None when the flux exceeds the sonic flux of the
            isentrope.
        """
        gas = self.gas
        temperature = [guess_t]

        def state(p):
            T = gas.temperature_ps(p, entropy, temperature[0])
            temperature[0] = T
            point = gas.point(p, T)
            return T, point.rho, total_enthalpy - point.h, point.sound_speed**2

        p = guess_p
        high = math.inf
        for _ in range(12):
            T, rho, head, sound2 = state(p)
            if head <= 0.0:
                high = min(high, p)
                break
            u = math.sqrt(2.0 * head)
            mach2 = u * u / sound2
            if mach2 >= 1.0:
                break
            residual = rho * u - mass_flux
            if residual <= 0.0:
                high = min(high, p)
            if abs(residual) <= _FLUX_TOLERANCE * mass_flux:
                return p, T, u
            newton = p - residual * u / (mach2 - 1.0)
            if abs(newton - p) < 1.0e-12 * p:
                return newton, gas.temperature_ps(newton, entropy, T), u
            if not newton > 0.0 or mach2 > 0.98:
                break
            p = newton

        if math.isinf(high):
            trial = min(p, stagnation)
            for _ in range(200):
                T, rho, head, sound2 = state(trial)
                if head <= 0.0 or (
                    2.0 * head < sound2 and rho * math.sqrt(2.0 * head) <= mass_flux
                ):
                    high = trial
                    break
                trial = min(1.02 * trial, stagnation)
            else:
                raise RuntimeError(
                    "the isentropic flux relation could not be bracketed"
                )

        def mach_excess(q):
            _, _, head, sound2 = state(q)
            return 2.0 * max(head, 0.0) / sound2 - 1.0

        sonic_low = high
        for _ in range(600):
            sonic_low *= 0.97
            if mach_excess(sonic_low) >= 0.0:
                break
        else:
            raise RuntimeError("no sonic state was found on the isentrope")
        sonic = brentq(
            mach_excess, sonic_low, high, xtol=1.0e-14 * high, rtol=8.9e-16, maxiter=200
        )
        sonic *= 1.0 + 1.0e-13

        def flux_residual(q):
            _, rho, head, _ = state(q)
            return rho * math.sqrt(2.0 * max(head, 0.0)) - mass_flux

        def solution(q):
            T, _, head, _ = state(q)
            return q, T, math.sqrt(2.0 * max(head, 0.0))

        # Two evaluations at one pressure differ by the rounding of the
        # temperature inversion, so an end whose residual is within the
        # tolerance is the root, whatever its sign.
        tolerance = _FLUX_TOLERANCE * mass_flux
        at_sonic = flux_residual(sonic)
        if at_sonic < -tolerance:
            return None
        if at_sonic <= tolerance:
            return solution(sonic)
        at_high = flux_residual(high)
        if at_high >= -tolerance:
            if at_high > 1.0e-9 * mass_flux:
                raise RuntimeError("the isentropic flux relation lost its bracket")
            return solution(high)
        try:
            root = brentq(
                flux_residual,
                sonic,
                high,
                xtol=1.0e-14 * high,
                rtol=8.9e-16,
                maxiter=200,
            )
        except GasRangeError:
            raise
        except ValueError:
            ends = sorted((abs(flux_residual(q)), q) for q in (sonic, high))
            if ends[0][0] > 1.0e-9 * mass_flux:
                raise RuntimeError(
                    "the isentropic flux relation lost its bracket"
                ) from None
            root = ends[0][1]
        return solution(root)

    def entrance(self, p2, surface_speed):
        """State of station 0 for the trial entrance pressure ``p2``.

        Isentropic acceleration from the inlet stagnation state to ``p2``,
        a total-pressure loss of ``entrance_loss_coefficient`` dynamic heads,
        and the static state that carries the same mass flux at the reduced
        total pressure. The swirl is ``preswirl * R omega * u3 / u2``.

        Returns
        -------
        entrance : tuple or None
            ``(G, (p3, T3, w3), u3)``, or None when there is no subsonic state.
        """
        solver, gas = self.solver, self.gas
        p_in, t_in = solver.inlet_pressure, solver.inlet_temperature
        T2 = gas.temperature_ps(p2, self.s_in, t_in)
        point2 = gas.point(p2, T2)
        head = self.h_in - point2.h
        if head <= 0.0:
            return None
        u2 = math.sqrt(2.0 * head)
        mass_flux = point2.rho * u2
        p30 = p_in - solver.entrance_loss_coefficient * 0.5 * point2.rho * u2 * u2
        if p30 <= 0.0:
            return None
        T30 = gas.temperature_ph(p30, self.h_in, t_in)
        s3 = gas.entropy(p30, T30)
        found = self._isentropic_flux_pressure(
            mass_flux, self.h_in, s3, p2 - (p_in - p30), T2, p30
        )
        if found is None:
            return None
        p3, T3, u3 = found
        w3 = solver.preswirl * surface_speed * u3 / u2
        return mass_flux, (p3, T3, w3), u3

    def exit_pressure(self, p4, T4, u4):
        """Pressure behind the exit for the state of the last station.

        The axial velocity is brought to rest isentropically, the total
        pressure loses ``exit_loss_coefficient`` dynamic heads, and the
        static pressure that carries the mass flux ``rho4 u4`` at that total
        pressure and the local total enthalpy ``h4 + u4^2 / 2`` is returned.

        Returns
        -------
        p5 : float or None
            None when there is no subsonic state.
        """
        exit_state = self._exit_stagnation(p4, T4, u4)
        if exit_state is None:
            return None
        rho4, h40, p40, p50, s5 = exit_state
        found = self._isentropic_flux_pressure(
            rho4 * u4, h40, s5, p4 - (p40 - p50), T4, p50
        )
        return None if found is None else found[0]

    def _exit_stagnation(self, p4, T4, u4):
        """Stagnation and loss construction of the exit relation.

        Returns
        -------
        exit_state : tuple or None
            ``(rho4, h40, p40, p50, s5)``: density of the last station, local
            total enthalpy, stagnation pressure before and after the exit
            loss, and entropy behind the loss. None when the loss leaves no
            positive pressure.
        """
        gas = self.gas
        point = gas.point(p4, T4)
        h40 = point.h + 0.5 * u4 * u4
        s4 = gas.entropy(p4, T4)
        p40, T40 = p4 + 0.5 * point.rho * u4 * u4, T4
        for _ in range(60):
            T40 = gas.temperature_ps(p40, s4, T40)
            stagnation = gas.point(p40, T40)
            step = stagnation.rho * (h40 - stagnation.h)
            p40 += step
            if abs(step) < 1.0e-13 * p40:
                break
        p50 = p40 - self.solver.exit_loss_coefficient * 0.5 * point.rho * u4 * u4
        if p50 <= 0.0:
            return None
        T50 = gas.temperature_ph(p50, h40, T40)
        s5 = gas.entropy(p50, T50)
        return point.rho, h40, p40, p50, s5

    def exit_flux_ratio(self, p4, T4, u4, guess=None):
        """Mass flux of the last station over the sonic flux of the exit relation.

        The sonic flux is that of the isentrope behind the exit loss at the
        local total enthalpy. The exit relation chokes when the ratio
        reaches 1.

        Parameters
        ----------
        p4, T4, u4 : float
            Pressure (Pa), temperature (K) and axial velocity (m/s) of the
            last station.
        guess : float, optional
            Estimate of the sonic pressure (Pa). The bracket of the sonic
            point is widened from it in small steps, so the search stays near
            the states the base flow visited. Default is a search down from
            the stagnation pressure.

        Returns
        -------
        ratio : float
        """
        exit_state = self._exit_stagnation(p4, T4, u4)
        if exit_state is None:
            raise RuntimeError("the exit loss leaves no positive pressure")
        rho4, h40, _, p50, s5 = exit_state
        gas = self.gas
        temperature = [T4]

        def state(q):
            T = gas.temperature_ps(q, s5, temperature[0])
            temperature[0] = T
            point = gas.point(q, T)
            return point, h40 - point.h

        def mach_excess(q):
            point, head = state(q)
            return 2.0 * max(head, 0.0) / point.sound_speed**2 - 1.0

        if guess is None:
            high, low = p50, p50
            for _ in range(600):
                high, low = low, 0.97 * low
                if mach_excess(low) >= 0.0:
                    break
            else:
                raise RuntimeError("no sonic state was found on the isentrope")
        else:
            low = high = min(guess, p50)
            for k in range(60):
                if mach_excess(low) >= 0.0:
                    break
                high, low = low, low * (1.0 - 1.0e-4 * 2.0**k)
            else:
                raise RuntimeError("no sonic state was found on the isentrope")
            for k in range(60):
                if high >= p50 or mach_excess(high) < 0.0:
                    break
                low, high = high, min(high * (1.0 + 1.0e-4 * 2.0**k), p50)
            else:
                raise RuntimeError("no sonic state was found on the isentrope")
        sonic = brentq(
            mach_excess, low, high, xtol=1.0e-14 * p50, rtol=8.9e-16, maxiter=200
        )
        point, head = state(sonic)
        return rho4 * u4 / (point.rho * math.sqrt(2.0 * max(head, 0.0)))

    def _trial(self, p2, surface_speed):
        """One trial of the outlet-pressure search at the entrance pressure ``p2``."""
        try:
            entrance = self.entrance(p2, surface_speed)
        except GasRangeError as error:
            self._range_error = error
            return {"status": "range", "error": error}
        if entrance is None:
            return {"status": "choked", "where": "entrance"}
        mass_flux, start, _ = entrance
        status, profile = self._integrate(mass_flux, surface_speed, start)
        if status == "range":
            return {"status": "range", "error": self._range_error}
        if status != "ok":
            return {"status": status, "where": "land"}
        p4, T4 = profile[-1, 0], profile[-1, 1]
        try:
            u4 = mass_flux / self.gas.point(p4, T4).rho
            p5 = self.exit_pressure(p4, T4, u4)
        except GasRangeError as error:
            self._range_error = error
            return {"status": "range", "error": error}
        if p5 is None:
            return {"status": "choked", "where": "exit"}
        return {
            "status": "ok",
            "p2": p2,
            "p5": p5,
            "mass_flux": mass_flux,
            "profile": profile,
            "residual": p5 - self.solver.outlet_pressure,
        }

    def base_state(self, speed):
        """Solve the steady flow at the rotor ``speed`` (rad/s).

        The entrance pressure ``p2`` is searched until the pressure behind
        the exit equals the outlet pressure (Brent's method on a bracket) or,
        when the outlet pressure cannot be reached, until the flow is at the
        choke limit (bisection to rounding).

        Returns
        -------
        base : HolePatternBaseState

        Raises
        ------
        ChokedAtFirstGuess
            The flow is choked even at the smallest trial flow.
        GasRangeError
            The flow leaves the range of the gas object.
        RuntimeError
            The search does not converge within ``max_iterations`` trials.
        """
        solver = self.solver
        p_in, p_out = solver.inlet_pressure, solver.outlet_pressure
        surface_speed = solver._shaft_radius * speed
        drop = p_in - p_out
        count = 0

        def trial(p2):
            nonlocal count
            count += 1
            if count > solver.max_iterations:
                raise RuntimeError(
                    "the outlet-pressure search of the hole-pattern seal needed more "
                    "than max_iterations trials"
                )
            return self._trial(p2, surface_speed)

        fraction = solver.first_step_size
        low = None
        low_p2 = None
        limiting = "choked"
        while True:
            candidate = trial(p_in * (1.0 - fraction))
            if candidate["status"] == "range" and candidate["error"].side != "below":
                raise self._table_limit(
                    "a trial flow of the hole-pattern base state leaves the range of the "
                    "gas properties on the hot or high-pressure side"
                )
            if candidate["status"] == "ok" and candidate["residual"] > 0.0:
                high = candidate
                break
            if candidate["status"] == "ok":
                low, low_p2 = candidate, None
            else:
                low, low_p2, limiting = (
                    None,
                    p_in * (1.0 - fraction),
                    candidate.get("where", candidate["status"]),
                )
            fraction /= 10.0
            if fraction < _FIRST_STEP_FLOOR:
                if low is not None:
                    raise RuntimeError(
                        "the outlet pressure is too close to the inlet pressure for the "
                        "hole-pattern solver"
                    )
                if limiting == "range":
                    raise self._table_limit(
                        "the smallest trial flow of the hole-pattern base state leaves the "
                        "range of the gas properties"
                    )
                raise ChokedAtFirstGuess(
                    "the hole-pattern seal flow is choked even at an entrance pressure of "
                    "(1 - 1e-8) x the inlet pressure"
                )

        if low is None and low_p2 is None:
            while low is None:
                fraction = min(fraction * 2.0, 0.999)
                candidate = trial(p_in * (1.0 - fraction))
                if candidate["status"] != "ok":
                    low_p2 = p_in * (1.0 - fraction)
                    limiting = candidate.get("where", candidate["status"])
                    break
                if candidate["residual"] > 0.0:
                    high = candidate
                    if fraction >= 0.999:
                        raise RuntimeError(
                            "no entrance pressure reaches the outlet pressure"
                        )
                else:
                    low = candidate

        choked = False
        solution = None
        while low is None:
            middle = 0.5 * (high["p2"] + low_p2)
            if (
                high["p2"] - low_p2 <= _BISECTION_RESOLUTION * p_in
                or not low_p2 < middle < high["p2"]
            ):
                choked, solution = True, high
                break
            candidate = trial(middle)
            if candidate["status"] != "ok":
                low_p2 = middle
                limiting = candidate.get("where", candidate["status"])
            elif candidate["residual"] > 0.0:
                high = candidate
            else:
                low = candidate
        if solution is None:
            seen = {low["p2"]: low, high["p2"]: high}

            def residual(p2):
                seen[p2] = trial(p2)
                if seen[p2]["status"] == "range":
                    raise self._table_limit(
                        "a trial inside the bracket of the outlet pressure leaves the "
                        "range of the gas properties"
                    )
                if seen[p2]["status"] != "ok":
                    raise RuntimeError(
                        "a trial inside the bracket of the outlet pressure failed"
                    )
                return seen[p2]["residual"]

            root = brentq(
                residual,
                low["p2"],
                high["p2"],
                xtol=1.0e-15 * p_in,
                rtol=8.9e-16,
                maxiter=200,
            )
            solution = seen[root]
            if abs(solution["residual"]) > max(solver.tolerance * drop, 1.0e-9 * drop):
                raise RuntimeError("the outlet-pressure iteration did not converge")
        if choked and limiting == "range":
            raise self._table_limit(
                "the hole-pattern base flow is limited by the range of the gas properties "
                "before it reaches the outlet pressure or the choke limit"
            )
        return self._assemble(
            solution["mass_flux"],
            solution["profile"],
            choked,
            solution["p2"],
            solution["p5"],
            count,
            limiting if choked else None,
        )

    def _table_limit(self, message):
        """A :class:`GasRangeError` that names the last state that left the gas range."""
        last = self._range_error
        if last is None:
            return GasRangeError(message)
        return GasRangeError(
            f"{message} ({last})",
            getattr(last, "pressure", None),
            getattr(last, "temperature", None),
            getattr(last, "side", None),
        )

    def _assemble(
        self, mass_flux, profile, choked, p2, p5, iterations, choke_location=None
    ):
        """Build a :class:`HolePatternBaseState` from a ``(p, T, w)`` profile."""
        solver = self.solver
        p, T, w = profile[:, 0].copy(), profile[:, 1].copy(), profile[:, 2].copy()
        points = [self.gas.point(float(a), float(b)) for a, b in zip(p, T, strict=True)]
        rho = np.array([point.rho for point in points])
        u = mass_flux / rho
        mach2 = np.array(
            [(v / point.sound_speed) ** 2 for v, point in zip(u, points, strict=True)]
        )
        return HolePatternBaseState(
            mass_flux=mass_flux,
            leakage=mass_flux * solver.area,
            z=np.arange(solver.nz + 1) * solver.dz,
            p=p,
            T=T,
            rho=rho,
            u=u,
            w=w,
            mach2=mach2,
            choked=bool(choked),
            p2=p2,
            p5=p5,
            iterations=iterations,
            gas=self.gas,
            choke_location=choke_location,
        )

    def _station_tables(self, base):
        """Linearised state functions at the stations."""
        points = [
            self.gas.point(float(a), float(b))
            for a, b in zip(base.p, base.T, strict=True)
        ]
        table = {
            "p_rho": [pt.p_rho for pt in points],
            "p_t": [pt.p_t for pt in points],
            "h_rho": [pt.h_rho for pt in points],
            "h_t": [pt.h_t for pt in points],
            "mu": [pt.mu for pt in points],
            "h": [pt.h for pt in points],
            "mu_rho": [pt.mu_rho for pt in points],
            "mu_t": [pt.mu_t for pt in points],
            "kappa": [pt.kappa_s for pt in points],
        }
        return {key: np.array(values) for key, values in table.items()}

    def _shear_derivatives(self, rho, u, w, mu, surface_speed):
        """Central differences of the wall shear at one station.

        Returns a dict of 3-vectors (axial, circumferential, rotor
        circumferential) keyed by ``"h"`` (clearance), ``"rho"``, ``"mu"``,
        ``"u"`` and ``"w"``.
        """
        shear = self.solver.shear
        args = [self.solver.radial_clearance, rho, mu, u, w]
        out = {}
        for index, name in enumerate(("h", "rho", "mu", "u", "w")):
            if name == "w":
                scale = max(abs(w), abs(u), abs(surface_speed))
            else:
                scale = abs(args[index])
            step = 1.0e-6 * scale
            plus, minus = list(args), list(args)
            plus[index] += step
            minus[index] -= step
            out[name] = (
                np.array(shear(*plus, surface_speed))
                - np.array(shear(*minus, surface_speed))
            ) / (2.0 * step)
        return out

    def _march(self, base, tables, derivs, surface_speed, omega, hc, start):
        """Integrate the first-order equations from station 0 to station nz (Heun).

        Parameters
        ----------
        base : HolePatternBaseState
        tables : dict
            Output of :meth:`_station_tables`.
        derivs : dict
            Shear derivatives per station, arrays of shape ``(nz + 1, 3)``.
        surface_speed : float
            Rotor surface speed (m/s).
        omega : ndarray, shape (nb,)
            Whirl frequency of each batch member (rad/s).
        hc : ndarray, shape (nb, 4)
            Clearance perturbation components.
        start : ndarray, shape (4, nb, 4)
            ``(rho1, T1, u1, w1)`` at station 0.

        Returns
        -------
        force : ndarray, shape (nb, 4)
            ``(fx_sin, fx_cos, fy_sin, fy_cos)`` per unit displacement (N/m).
        final : ndarray, shape (4, nb, 4)
            The perturbation at station nz.
        """
        solver = self.solver
        h0 = solver.radial_clearance
        rad = solver._shaft_radius
        depth = solver.cell_depth
        dz = solver.dz
        nz = solver.nz
        i_t, i_th = solver.i_t, solver.i_th
        sgn_t, sgn_th = solver.sgn_t, solver.sgn_th
        w_om = omega[:, None]
        pi_radius = math.pi * rad

        rho_b, u_b, w_b, p_b = base.rho, base.u, base.w, base.p
        p_rho, p_t = tables["p_rho"], tables["p_t"]
        h_rho, h_t = tables["h_rho"], tables["h_t"]
        mu_rho, mu_t = tables["mu_rho"], tables["mu_t"]
        cell = depth * tables["kappa"]
        d_h, d_rho, d_mu = derivs["h"], derivs["rho"], derivs["mu"]
        d_u, d_w = derivs["u"], derivs["w"]
        total0 = tables["h"] + 0.5 * (u_b**2 + w_b**2)

        def rotor_shear(i, state):
            r1, t1, u1, w1 = state
            return (
                d_mu[i, 2] * (mu_rho[i] * r1 + mu_t[i] * t1)
                + d_rho[i, 2] * r1
                + d_u[i, 2] * u1
                + d_w[i, 2] * w1
                + d_h[i, 2] * hc
            )

        def integrand(i, state):
            tr = rotor_shear(i, state)
            p1 = p_rho[i] * state[0] + p_t[i] * state[1]
            return pi_radius * np.stack(
                [
                    -tr[:, 0] - p1[:, 2],
                    -tr[:, 1] - p1[:, 3],
                    tr[:, 2] - p1[:, 0],
                    tr[:, 3] - p1[:, 1],
                ],
                axis=1,
            )

        def derivative(i, current, gradients):
            up, wp, rhop, pp, dp_rho, dp_t, dflux, dh_rho, dh_t, htp = gradients
            rho, u, w = rho_b[i], u_b[i], w_b[i]
            r1, t1, u1, w1 = current
            p1 = p_rho[i] * r1 + p_t[i] * t1
            ht_time = hc + cell[i] * p1
            cof = np.zeros((4, 4))
            rhs = np.empty((omega.size, 4, 4))

            cof[0, 0] = u * h0
            cof[0, 2] = rho * h0
            rhs[:, 0, :] = -(
                sgn_t * w_om * (rho * ht_time[:, i_t] + h0 * r1[:, i_t])
                + sgn_th
                * (
                    (w * h0 / rad) * r1[:, i_th]
                    + (rho * h0 / rad) * w1[:, i_th]
                    + (rho * w / rad) * hc[:, i_th]
                )
                + up * h0 * r1
                + rhop * h0 * u1
                + dflux * hc
            )

            cof[1, 0] = h0 * rho * u * h_rho[i]
            cof[1, 1] = h0 * rho * u * h_t[i]
            cof[1, 2] = h0 * rho * u * u
            cof[1, 3] = h0 * rho * u * w
            total1 = h_rho[i] * r1 + h_t[i] * t1 + u * u1 + w * w1
            rhs[:, 1, :] = -(
                h0 * rho * u * (dh_rho * r1 + dh_t * t1 + up * u1 + wp * w1)
                + rho
                * h0
                * (sgn_t * w_om * total1[:, i_t] + sgn_th * (w / rad) * total1[:, i_th])
                + htp * (h0 * u * r1 + rho * h0 * u1 + rho * u * hc)
                - h0 * sgn_t * w_om * p1[:, i_t]
                + surface_speed * rotor_shear(i, current)
            )

            cof[2, 0] = h0 * p_rho[i]
            cof[2, 1] = h0 * p_t[i]
            cof[2, 2] = h0 * rho * u
            mu1 = mu_rho[i] * r1 + mu_t[i] * t1
            friction_axial = (
                d_mu[i, 0] * mu1
                + d_rho[i, 0] * r1
                + d_u[i, 0] * u1
                + d_w[i, 0] * w1
                + d_h[i, 0] * hc
            )
            rhs[:, 2, :] = (
                -(
                    sgn_t * w_om * rho * h0 * u1[:, i_t]
                    + sgn_th * (rho * w * h0 / rad) * u1[:, i_th]
                    + h0 * u * up * r1
                    + h0 * rho * up * u1
                    + rho * u * up * hc
                )
                - h0 * dp_rho * r1
                - h0 * dp_t * t1
                - pp * hc
                - friction_axial
            )

            cof[3, 3] = h0 * rho * u
            friction_tangential = (
                d_mu[i, 1] * mu1
                + d_rho[i, 1] * r1
                + d_u[i, 1] * u1
                + d_w[i, 1] * w1
                + d_h[i, 1] * hc
            )
            rhs[:, 3, :] = (
                -(
                    sgn_t * w_om * rho * h0 * w1[:, i_t]
                    + sgn_th * (rho * w * h0 / rad) * w1[:, i_th]
                    + h0 * u * wp * r1
                    + h0 * rho * wp * u1
                    + rho * u * wp * hc
                )
                - (h0 / rad) * sgn_th * p1[:, i_th]
                - friction_tangential
            )
            return np.moveaxis(np.linalg.solve(cof, rhs), 1, 0)

        state = start.copy()
        force = np.zeros((omega.size, 4))
        force += 0.5 * integrand(0, state)
        last = np.zeros((omega.size, 4))
        for iz in range(1, nz + 1):
            o, n = iz - 1, iz
            gradients = (
                (u_b[n] - u_b[o]) / dz,
                (w_b[n] - w_b[o]) / dz,
                (rho_b[n] - rho_b[o]) / dz,
                (p_b[n] - p_b[o]) / dz,
                (p_rho[n] - p_rho[o]) / dz,
                (p_t[n] - p_t[o]) / dz,
                (rho_b[n] * u_b[n] - rho_b[o] * u_b[o]) / dz,
                (h_rho[n] - h_rho[o]) / dz,
                (h_t[n] - h_t[o]) / dz,
                (total0[n] - total0[o]) / dz,
            )
            slope_old = derivative(o, state, gradients)
            slope_new = derivative(n, state + dz * slope_old, gradients)
            state = state + dz * (slope_old + slope_new) / 2.0
            last = integrand(n, state)
            force += last
        force = (force - 0.5 * last) * dz
        return force, state

    def _boundary_relations(self, base, surface_speed):
        """Linearised quasi-steady entrance and exit relations.

        Returns
        -------
        entrance : ndarray, shape (4,)
            ``(d rho3/d u3, d T3/d u3, 1, d w3/d u3)`` of the entrance model.
        exit_row : ndarray, shape (3,)
            Row ``e`` with ``e . (rho1, T1, u1) = 0`` at station nz. Unchoked,
            the pressure behind the exit is unchanged. Choked in the land (or
            at the entrance), the axial Mach number of the last station is
            unchanged. Choked in the exit relation, the exit stays at its
            sonic flux (:meth:`exit_flux_ratio` is unchanged); this row is
            also used when the exit is so close to sonic that a perturbed
            state of the central difference has no subsonic exit, since the
            row of the pressure tends to it there.
        """
        gas = self.gas
        step = 1.0e-5 * (self.solver.inlet_pressure - base.p2)
        states = []
        for p2 in (base.p2 + step, base.p2 - step):
            found = self.entrance(p2, surface_speed)
            if found is None:
                raise RuntimeError("the entrance relation could not be linearised")
            _, (p3, T3, w3), u3 = found
            states.append((gas.point(p3, T3).rho, T3, u3, w3))
        (r_a, t_a, u_a, w_a), (r_b, t_b, u_b, w_b) = states
        du_entrance = u_a - u_b
        entrance = np.array(
            [
                (r_a - r_b) / du_entrance,
                (t_a - t_b) / du_entrance,
                1.0,
                (w_a - w_b) / du_entrance,
            ]
        )

        p4, T4, u4 = float(base.p[-1]), float(base.T[-1]), float(base.u[-1])
        point = gas.point(p4, T4)
        dp, dt, du = 1.0e-6 * p4, 1.0e-6 * T4, 1.0e-6 * u4

        def row(measure):
            values = [
                measure(p4 + dp, T4, u4),
                measure(p4 - dp, T4, u4),
                measure(p4, T4 + dt, u4),
                measure(p4, T4 - dt, u4),
                measure(p4, T4, u4 + du),
                measure(p4, T4, u4 - du),
            ]
            if any(value is None for value in values):
                return None
            m_p = (values[0] - values[1]) / (2.0 * dp)
            m_t = (values[2] - values[3]) / (2.0 * dt)
            m_u = (values[4] - values[5]) / (2.0 * du)
            return np.array([m_p * point.p_rho, m_p * point.p_t + m_t, m_u])

        if base.choked and base.choke_location != "exit":
            exit_row = row(lambda p, T, u: u / gas.point(p, T).sound_speed)
        else:
            exit_row = None if base.choked else row(self.exit_pressure)
            if exit_row is None:
                exit_row = row(
                    lambda p, T, u: self.exit_flux_ratio(p, T, u, guess=base.p5)
                )
        return entrance, exit_row

    def forces(self, base, speed, omega):
        """Reaction-force components for a unit displacement at each whirl frequency.

        One forced and four free solutions per frequency, combined so that
        the linearised exit relation holds.

        Returns
        -------
        force : ndarray, shape (n_f, 4)
            ``(fx_sin, fx_cos, fy_sin, fy_cos)`` (N/m); NaN for a frequency
            whose superposition system is singular.
        """
        solver = self.solver
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        surface_speed = solver._shaft_radius * speed
        tables = self._station_tables(base)
        stations = [
            self._shear_derivatives(
                base.rho[i], base.u[i], base.w[i], tables["mu"][i], surface_speed
            )
            for i in range(solver.nz + 1)
        ]
        derivs = {
            name: np.array([station[name] for station in stations])
            for name in ("h", "rho", "mu", "u", "w")
        }
        entrance, exit_row = self._boundary_relations(base, surface_speed)
        n_f = omega.size
        batch_omega = np.repeat(omega, 5)
        clearance = np.zeros((5 * n_f, 4))
        clearance[0::5] = solver.clearance_perturbation
        start = np.zeros((4, 5 * n_f, 4))
        for component in range(4):
            start[:, 1 + component :: 5, component] = entrance[:, None]
        force, final = self._march(
            base, tables, derivs, surface_speed, batch_omega, clearance, start
        )
        measure = (
            exit_row[0] * final[0] + exit_row[1] * final[1] + exit_row[2] * final[2]
        )
        out = np.empty((n_f, 4))
        for k in range(n_f):
            rows = slice(5 * k, 5 * k + 5)
            m, f = measure[rows], force[rows]
            try:
                amplitude = np.linalg.solve(m[1:].T, -m[0])
            except np.linalg.LinAlgError:
                out[k] = np.nan
                continue
            out[k] = f[0] + amplitude @ f[1:]
        return out

    def coefficients(self, base, speed, frequencies):
        """Stiffness and damping at the whirl ``frequencies`` (rad/s).

        ``kxx = -fx_cos``, ``kxy = fy_cos``, ``cxx = fx_sin / Omega`` and
        ``cxy = -fy_sin / Omega`` at each frequency; the mass terms are zero.
        A frequency below ``_ZERO_FREQUENCY_STEP * max(|speed|, 1)`` is
        replaced by that step, which gives the zero-frequency limit.

        Returns
        -------
        coefficients : dict
            Arrays keyed by ``COEFFICIENT_NAMES``.
        """
        omega = np.atleast_1d(np.asarray(frequencies, dtype=float))
        step = _ZERO_FREQUENCY_STEP * max(abs(speed), 1.0)
        omega = np.where(np.abs(omega) < step, step, omega)
        force = self.forces(base, speed, omega)
        k_dir, k_cross = -force[:, 1], force[:, 3]
        c_dir, c_cross = force[:, 0] / omega, -force[:, 2] / omega
        zero = np.zeros(omega.size)
        return {
            "kxx": k_dir,
            "kyy": k_dir.copy(),
            "kxy": k_cross,
            "kyx": -k_cross,
            "cxx": c_dir,
            "cyy": c_dir.copy(),
            "cxy": c_cross,
            "cyx": -c_cross,
            "mxx": zero,
            "myy": zero.copy(),
            "mxy": zero.copy(),
            "myx": zero.copy(),
        }


class HolePatternSeal(SealElement):
    """Hole-pattern (honeycomb) annular gas seal: leakage and rotordynamic coefficients from a bulk-flow model.

    The model is a single control volume of constant clearance. Its base
    state follows from continuity, axial and circumferential momentum and
    energy with adiabatic walls and the work of the rotor, marched along the
    seal for the pressure, the temperature and the swirl velocity. The flow
    enters by an isentropic acceleration with an entrance loss and an inlet
    swirl ``preswirl`` times the rotor surface speed, and leaves through an
    exit loss; the entrance pressure is searched until the pressure behind
    the exit equals ``outlet_pressure``, or the flow is at the choke limit.
    The wall shear is Moody's law with the hydraulic diameter ``2 H``: the
    rotor with the roughness ``relative_roughness``, the stator as a smooth
    wall at a quarter of the Moody value. The cells act as an isentropically
    compressed storage volume of depth ``cell_depth``; ``cell_length`` and
    ``cell_width`` are not used. The first-order equations of a whirling
    rotor are solved with quasi-steady entrance and exit relations.

    At each (speed, whirl frequency Omega) point the element stores the
    stiffness at Omega, ``kxx = kyy`` and ``kxy = -kyx``, and the damping at
    Omega, ``cxx = cyy`` and ``cxy = -cyx``; the mass terms are zero. So the
    rotor sees ``K(Omega) + i Omega C(Omega)`` exactly at every tabulated
    point, the convention of :class:`~ross.seals.labyrinth_seal.LabyrinthSeal`.
    Without ``frequency`` the 1-D table holds the coefficients at
    ``excitation_ratio * speed``; frequency-dependent analyses need
    ``frequency=``. At a zero whirl frequency the damping is its
    zero-frequency limit.

    ``gas_model="ideal"`` uses a perfect gas with constant ``gamma`` and the
    Sutherland viscosity. ``gas_model="real"`` tabulates density, enthalpy,
    entropy and viscosity over pressure and temperature with the equation of
    state of ``ccp``; the cold edge of the table follows the inlet isentrope.
    The seal must stay a single-phase gas down to the table floor: an inlet
    that is not one is refused at construction with a ``ValueError``, and a
    speed whose flow needs states below the floor fails; states outside the
    table are not extrapolated. Mixtures with many components are slow under
    the HEOS backend (seconds to minutes per table; REFPROP is much faster).

    A speed (or whirl frequency) that cannot be solved warns and gets NaN
    coefficients. A table of one or two speeds holds them. The ``"pchip"``
    interpolation of three or more points needs finite values, so such a
    table raises a ``ValueError`` that lists the failed speeds and the first
    error.

    See the theory page ``docs/theory_and_code/holepattern_seal.ipynb``.

    Parameters
    ----------
    n : int
        Node in which the seal will be located.
    shaft_diameter : float, pint.Quantity
        Diameter of the shaft (m).
    radial_clearance : float, pint.Quantity
        Seal radial clearance (m).
    axial_length : float, pint.Quantity
        Axial length of the seal (m).
    relative_roughness : float
        Roughness of the rotor surface divided by the radial clearance,
        dimensionless. It enters the rotor friction factor; the stator is
        treated as smooth.
    cell_length : float, pint.Quantity
        Typical length of a cell in the axial direction (m). Not used by the
        model: the cells act only through their depth.
    cell_width : float, pint.Quantity
        Typical length of a cell in the azimuthal direction (m). Not used by
        the model: the cells act only through their depth.
    cell_depth : float, pint.Quantity
        Depth of the cells (m). For a hole pattern, use the volume-equivalent
        depth: hole depth times hole-area ratio.
    inlet_pressure : float
        Inlet pressure (Pa).
    outlet_pressure : float
        Outlet pressure (Pa).
    inlet_temperature : float
        Inlet temperature (K).
    speed : list, pint.Quantity
        Shaft rotational speeds (rad/s). The base state is solved for each
        speed.
    frequency : array, pint.Quantity, optional
        Whirl (excitation) frequencies (rad/s). When given, the coefficients
        are computed at every (speed, frequency) pair and the element carries
        a 2-D table of shape ``(len(speed), len(frequency))`` interpolated on
        both axes; ``excitation_ratio`` is ignored.
        Default is None (1-D table over the speed axis).
    gas_composition : dict, optional
        Gas composition as a dictionary {component: molar_fraction}.
    gas_model : str, optional
        ``"ideal"``: perfect gas with ``R = 8314 / molar_mass``, constant
        ``gamma`` and the Sutherland viscosity. With ``gas_composition``,
        ``gamma`` is cp/cv of the inlet state, and ``sutherland_b`` and
        ``sutherland_s`` are fitted to the equation-of-state viscosity at
        the inlet pressure between 260 and 380 K.
        ``"real"``: density, enthalpy, entropy and viscosity from the
        equation of state of ``ccp`` (``ccp.config.EOS``), tabulated over the
        pressure and temperature range of the seal. Requires
        ``gas_composition``.
        Default is "ideal".
    molar_mass : float, pint.Quantity, optional
        Molecular mass (kg/kgmol). For air: 28.97. Required if
        gas_composition is None.
        Default is None.
    gamma : float, optional
        Ratio of specific heats cp/cv. For air: 1.4. Required if
        gas_composition is None.
        Default is None.
    sutherland_b : float, optional
        Coefficient b of the Sutherland law ``mu = b T^1.5 / (S + T)``
        (Pa s / K^0.5). Required if gas_composition is None; not used with
        ``gas_model="real"``.
        Default is None.
    sutherland_s : float, optional
        Coefficient S of the Sutherland law (K). Required if gas_composition
        is None; not used with ``gas_model="real"``.
        Default is None.
    preswirl : float, optional
        Ratio of the circumferential velocity of the gas at the inlet to the
        surface velocity of the shaft.
        Default is 0.0.
    entrance_loss_coefficient : float, optional
        Entrance loss coefficient (dynamic heads).
        Default is 0.1.
    exit_loss_coefficient : float, optional
        Exit loss coefficient (dynamic heads).
        Default is 0.5.
    excitation_ratio : float, optional
        For a 1-D table, the coefficients at each speed are those at the
        whirl frequency ``excitation_ratio * speed``. Ignored when
        ``frequency`` is given.
        Default is 1.0.
    nz : int, optional
        Number of axial intervals (at least 2).
        Default is 80.
    max_iterations : int, optional
        Maximum number of trial integrations of the outlet-pressure search.
        Default is 180.
    tolerance : float, optional
        Largest outlet-pressure residual accepted at the end of the search,
        as a fraction of the pressure drop across the seal.
        Default is 0.0001.
    first_step_size : float, optional
        The first trial entrance pressure is ``(1 - first_step_size)`` times
        the inlet pressure; the step is divided by 10 while that trial
        already passes the outlet pressure or chokes.
        Default is 0.01.
    relaxation_factor : float, optional
        Not used: the outlet-pressure search has no relaxed secant step. Kept
        so that existing scripts and saved files build.
        Default is 0.1.
    tag : str, optional
        A tag to name the element.
        Default is None.
    n_link : int, optional
        Node to which the seal will connect. If None the seal is connected
        to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the seal drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is "#77ACA2".

    Attributes
    ----------
    gas : SutherlandGas, TabulatedGas or None
        The gas object; None for a real-gas element built from saved
        coefficients.
    choked : list of bool
        Per speed: True when the outlet pressure cannot be reached and the
        leakage and coefficients are those of the choke limit; None for a
        failed point. Set only when the element is solved.
    p : list of ndarray
        Per speed: ``nz + 4`` pressures ``[p_in, p2, stations, p5]`` (Pa).
    seal_leakage : list of float
        Per speed: leakage (kg/s).

    References
    ----------
    Nelson, C. C. (1985). Rotordynamic coefficients for compressible flow in
    tapered annular seals. Journal of Tribology, 107(3), 318-325.

    Kleynhans, G. F. and Childs, D. W. (1996). The acoustic influence of cell
    depth on the rotordynamic characteristics of smooth-rotor/honeycomb-stator
    annular gas seals. ASME Turbo Expo, 96-GT-122.

    Childs, D. W. and Wade, J. (2004). Rotordynamic-coefficient and leakage
    characteristics for hole-pattern-stator annular gas seals: measurements
    versus predictions. Journal of Tribology, 126(2), 326-333.

    Examples
    --------
    >>> from ross.seals.holepattern_seal import HolePatternSeal
    >>> from ross.units import Q_
    >>> holepattern = HolePatternSeal(
    ...     n=0,
    ...     shaft_diameter=0.145,
    ...     radial_clearance=0.0003,
    ...     axial_length=0.04699,
    ...     relative_roughness=0.0001,
    ...     cell_length=0.003175,
    ...     cell_width=0.003175,
    ...     cell_depth=0.0025,
    ...     inlet_pressure=689000.0,
    ...     outlet_pressure=94300.0,
    ...     inlet_temperature=322.0,
    ...     speed=Q_([8000], "RPM"),
    ...     gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
    ...     preswirl=0.8,
    ...     entrance_loss_coefficient=0.5,
    ...     exit_loss_coefficient=1.0,
    ...     nz=18
    ... )
    >>> holepattern.choked
    [True]
    """

    _pressure_plot_label = "Hole Pattern Seal"

    @check_units
    def __init__(
        self,
        n,
        shaft_diameter,
        radial_clearance,
        axial_length,
        relative_roughness,
        cell_length,
        cell_width,
        cell_depth,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        speed,
        frequency=None,
        gas_composition=None,
        gas_model="ideal",
        molar_mass=None,
        gamma=None,
        sutherland_b=None,
        sutherland_s=None,
        preswirl=0.0,
        entrance_loss_coefficient=0.1,
        exit_loss_coefficient=0.5,
        excitation_ratio=1.0,
        nz=80,
        max_iterations=180,
        tolerance=0.0001,
        first_step_size=0.01,
        relaxation_factor=0.1,
        **kwargs,
    ):
        if gas_model not in ("ideal", "real"):
            raise ValueError(
                f"Invalid gas_model {gas_model!r}; expected 'ideal' or 'real'."
            )
        if gas_model == "real" and gas_composition is None:
            raise ValueError(
                "gas_model='real' requires gas_composition to query the equation of state."
            )
        if gas_composition is None and any(
            value is None for value in (molar_mass, gamma, sutherland_b, sutherland_s)
        ):
            raise TypeError(
                "HolePatternSeal needs gas_composition, or molar_mass, gamma, "
                "sutherland_b and sutherland_s."
            )
        if nz < 2:
            raise ValueError("nz must be at least 2.")

        self.n = n
        self.shaft_diameter = shaft_diameter
        self._shaft_radius = shaft_diameter / 2
        self.radial_clearance = radial_clearance
        self.axial_length = axial_length
        self.relative_roughness = relative_roughness
        self.cell_length = cell_length
        self.cell_width = cell_width
        self.cell_depth = cell_depth
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure = outlet_pressure
        self.inlet_temperature = inlet_temperature
        self.gas_composition = gas_composition
        self.gas_model = gas_model
        self.preswirl = preswirl
        self.entrance_loss_coefficient = entrance_loss_coefficient
        self.exit_loss_coefficient = exit_loss_coefficient
        self.excitation_ratio = excitation_ratio
        self.nz = nz
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.first_step_size = first_step_size
        self.relaxation_factor = relaxation_factor

        if gas_composition is not None:
            state, molar_mass, gamma, R = extract_gas_properties(
                gas_composition, inlet_pressure, inlet_temperature
            )
            if gas_model == "ideal":

                def sutherland_formula(T, b, S):
                    return (b * T ** (3 / 2)) / (S + T)

                x = []
                y = []
                for T in range(260, 400, 20):
                    try:
                        state.update(p=state.p(), T=T)
                        x.append(T)
                        y.append(state.viscosity().m)
                    except ValueError:
                        # Skip temperatures where the state update fails (e.g.
                        # HEOS convergence issues).
                        continue

                if len(x) < 3:
                    raise RuntimeError(
                        f"Could not collect enough viscosity data points ({len(x)} "
                        "points) to fit Sutherland coefficients. Try providing "
                        "sutherland_b, sutherland_s, molar_mass, and gamma manually."
                    )

                popt, _ = curve_fit(sutherland_formula, x, y)
                sutherland_b, sutherland_s = popt
        else:
            R = R_UNIVERSAL / molar_mass

        self.R = R
        self.molar_mass = molar_mass
        self.gamma = gamma
        self.sutherland_b = sutherland_b
        self.sutherland_s = sutherland_s

        solve = kwargs.get("kxx") is None
        if gas_model == "ideal":
            gas = SutherlandGas(R, gamma, sutherland_b, sutherland_s)
        elif solve:
            gas = TabulatedGas.from_composition(
                gas_composition, inlet_pressure, inlet_temperature, outlet_pressure
            )
        else:
            gas = None
        self.gas = gas
        self._real_gas = gas_model == "real"

        self.dz = axial_length / float(nz)
        self.z = np.zeros(nz + 4)
        self.z[0] = -self.dz
        self.z[1] = 0.0
        self.z[2:-1] = np.arange(nz + 1) * self.dz
        self.z[-1] = self.z[-2]

        self.solver = HolePatternSolver(
            shaft_radius=self._shaft_radius,
            radial_clearance=radial_clearance,
            axial_length=axial_length,
            relative_roughness=relative_roughness,
            cell_depth=cell_depth,
            inlet_pressure=inlet_pressure,
            outlet_pressure=outlet_pressure,
            inlet_temperature=inlet_temperature,
            preswirl=preswirl,
            entrance_loss_coefficient=entrance_loss_coefficient,
            exit_loss_coefficient=exit_loss_coefficient,
            excitation_ratio=excitation_ratio,
            nz=nz,
            max_iterations=max_iterations,
            tolerance=tolerance,
            first_step_size=first_step_size,
            gas=self.gas,
        )

        coefficients_dict = {}
        if solve:
            speed = np.atleast_1d(np.asarray(speed, dtype=float))
            if frequency is None:
                results = self.solver._solve_speeds(
                    self.solver.solve, speed, PARALLEL_THRESHOLD
                )
                rows = [[r] for r in results]
                coefficients_dict = {
                    c: [row[0][c] for row in rows] for c in COEFFICIENT_NAMES
                }
            else:
                frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
                rows = self.solver.solve_grid(speed, frequency, PARALLEL_THRESHOLD)
                coefficients_dict = {
                    c: [[r[c] for r in row] for row in rows] for c in COEFFICIENT_NAMES
                }

            self.p = [row[0]["pressure"] for row in rows]
            self.choked = [row[0]["choked"] for row in rows]
            coefficients_dict["seal_leakage"] = [row[0]["seal_leakage"] for row in rows]
            failures = [
                (s, point["error"])
                for s, row in zip(speed, rows, strict=True)
                for point in row
                if point.get("error") is not None
            ]
        else:
            failures = []

        try:
            super().__init__(
                self.n,
                speed=speed,
                frequency=frequency,
                **coefficients_dict,
                **kwargs,
            )
        except ValueError as error:
            if not failures:
                raise
            failed_speeds = sorted({float(s) for s, _ in failures})
            raise ValueError(
                "the coefficient table of the hole-pattern seal holds NaN at the "
                f"failed speeds {failed_speeds} rad/s, and its interpolation over "
                f"three or more points needs finite values ({error}); leave these "
                "speeds out. First failure: "
                f"{failures[0][1]}"
            ) from error
