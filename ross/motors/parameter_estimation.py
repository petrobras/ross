"""Equivalent circuit parameter estimation for three-phase induction motors.

The methods in this module estimate the per-phase parameters of the T
equivalent circuit (stator and rotor resistances and reactances, magnetizing
reactance and, when the method provides it, the core loss resistance Rc) from
nameplate or catalog data. They are used by
``MotorElement.from_nameplate()`` and ``MotorElement.from_catalog()``.

All quantities are SI and refer to the per-phase star equivalent circuit, with
the rotor referred to the stator. Voltages are phase voltages and currents are
phase (equal to line) currents of that equivalent.

References
----------
.. [1] Wengerkievicz et al. (2025). Assessment of Parameter Estimation Methods
   for Nonintrusive Loss Estimation of Induction Motors. IET Electric Power
   Applications.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from ross.motors.utils import phase_to_line
from ross.units import Q_

__all__ = [
    "EstimationResult",
    "NAMEPLATE_METHODS",
    "CATALOG_METHODS",
    "estimate_from_nameplate",
    "estimate_from_catalog",
]

X1_FRACTION = {"N": 0.40, "H": 0.30, "D": 0.50}


class _Unphysical(RuntimeError):
    """Raised when an iteration produces a non-physical circuit."""


@dataclass
class EstimationResult:
    """Result of an equivalent circuit parameter estimation.

    Parameters
    ----------
    method : str
        Name of the estimation method.
    stator_resistance, rotor_resistance : float
        Stator and rotor resistances [Ohm]. The rotor value is referred to the
        stator.
    stator_reactance, rotor_reactance : float
        Stator and rotor leakage reactances [Ohm]. The rotor value is referred
        to the stator.
    mutual_reactance : float
        Magnetizing reactance [Ohm].
    Rc : float, optional
        Core loss resistance [Ohm]. ``np.inf`` when the method has no core loss
        branch.
    converged : bool, optional
        Whether the method met its convergence criterion.
    iterations : int, optional
        Number of iterations (function evaluations for Haque 2).
    residual : float, optional
        Final residual of the method. Its meaning depends on the method.
    warnings : str, optional
        Messages about the quality of the solution.
    extra : dict, optional
        Method specific quantities.
    """

    method: str
    stator_resistance: float
    rotor_resistance: float
    stator_reactance: float
    rotor_reactance: float
    mutual_reactance: float
    Rc: float = np.inf
    converged: bool = True
    iterations: int = 0
    residual: float = np.nan
    warnings: str = ""
    extra: dict = None

    def to_dict(self):
        """Return the result as a dictionary."""
        return asdict(self)


class _Rating:
    """Rated data of a motor and the quantities derived from it.

    Parameters
    ----------
    power_rated : float
        Rated output power [W].
    voltage_rated : float
        Rated phase voltage [V].
    speed_rated : float
        Rated mechanical speed [rad/s].
    frequency_rated : float
        Rated electrical frequency [rad/s].
    n_poles : int
        Number of poles.
    efficiency : float
        Rated efficiency, between zero and one.
    power_factor : float
        Rated power factor, between zero and one.
    current_rated : float, optional
        Rated phase current [A]. Default is computed from the input power.
    starting_current_ratio : float, optional
        Ratio between starting and rated current.
    motor_category : str, optional
        Motor category, "N", "H" or "D".
    """

    def __init__(
        self,
        power_rated,
        voltage_rated,
        speed_rated,
        frequency_rated,
        n_poles,
        efficiency,
        power_factor,
        current_rated=None,
        starting_current_ratio=6.0,
        motor_category="N",
    ):
        self.power_rated = float(power_rated)
        self.voltage_rated = float(voltage_rated)
        self.speed_rated = float(speed_rated)
        self.frequency_rated = float(frequency_rated)
        self.n_poles = int(n_poles)
        self.efficiency = float(efficiency)
        self.power_factor = float(power_factor)
        self.starting_current_ratio = float(starting_current_ratio)
        self.motor_category = str(motor_category).upper()

        if self.motor_category not in X1_FRACTION:
            raise ValueError("motor_category must be 'N', 'H' or 'D'.")
        if self.starting_current_ratio <= 1.0:
            raise ValueError("starting_current_ratio must be greater than one.")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1].")
        if not 0.0 < self.power_factor <= 1.0:
            raise ValueError("power_factor must be in (0, 1].")

        self.omega_sync = 2.0 * self.frequency_rated / self.n_poles
        self.slip = 1.0 - self.speed_rated / self.omega_sync
        if not 0.0 < self.slip < 1.0:
            raise ValueError("The rated slip must be in (0, 1).")

        self.input_power = self.power_rated / self.efficiency
        if current_rated is None:
            self.current_rated = self.input_power / (
                3.0 * self.voltage_rated * self.power_factor
            )
        else:
            self.current_rated = float(current_rated)
        self.starting_current = self.starting_current_ratio * self.current_rated

    @property
    def power_hp(self):
        """Rated output power [hp]."""
        return Q_(self.power_rated, "W").to("hp").m

    @property
    def x1_fraction(self):
        """Fraction of the total leakage reactance assigned to the stator."""
        return X1_FRACTION[self.motor_category]

    @property
    def x1_x2_ratio(self):
        """Ratio between stator and rotor leakage reactances."""
        return self.x1_fraction / (1.0 - self.x1_fraction)

    def per_unit_base(self):
        """Return the base current [A] and impedance [Ohm].

        The base power is the rated output power and the base voltage is the
        rated phase voltage.
        """
        i_base = self.power_rated / (3.0 * self.voltage_rated)
        return i_base, self.voltage_rated / i_base


def _parallel(z1, z2):
    return z1 * z2 / (z1 + z2)


def _stray_loss_fraction(power):
    """Return the stray load loss as a fraction of the output power [W]."""
    if power <= 90e3:
        return 0.018
    if power <= 375e3:
        return 0.015
    if power <= 1850e3:
        return 0.012
    return 0.009


def asdp(rating, delta=0.05, acceptance_band=0.02, max_iterations=20000):
    """Estimate the parameters with the sequential nameplate adjustment (ASDP).

    The parameters are adjusted in steps of ``delta`` in per unit, with the
    rated output power as base, until the mechanical power, stator current,
    power factor and efficiency are within ``acceptance_band`` of the rated
    values. The total leakage reactance is split between stator and rotor
    according to the motor category.

    Parameters
    ----------
    rating : _Rating
        Rated data of the motor.
    delta : float, optional
        Relative step of each adjustment.
    acceptance_band : float, optional
        Admissible relative error of the rated quantities.
    max_iterations : int, optional
        Maximum number of adjustments.

    Returns
    -------
    result : EstimationResult

    References
    ----------
    R. R. Bastos, "Aula 1 - Obtenção dos Parâmetros do Circuito Equivalente",
    lecture notes, 2021.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1).")
    if not 0.0 < acceptance_band < 1.0:
        raise ValueError("acceptance_band must be in (0, 1).")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    floor = 1e-8
    slip = rating.slip
    fraction = rating.x1_fraction
    i_base, z_base = rating.per_unit_base()
    i_rated = rating.current_rated / i_base
    i_start = rating.starting_current / i_base

    x_total = 1.0 / rating.starting_current_ratio
    xm = 1.0 / (i_rated * np.sin(np.arccos(rating.power_factor)))
    parameters = np.array([slip, slip, x_total, xm, 20.0])

    def protect(values):
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("ASDP produced a NaN or infinite parameter.")
        return np.maximum(values, floor)

    def in_band(value, target):
        return (
            (1.0 - acceptance_band) * target
            <= value
            <= (1.0 + acceptance_band) * target
        )

    def relative_error(value, target):
        return abs(value - target) / max(abs(target), floor)

    def circuit(values):
        r1, r2, x_t, xm, rm = protect(values)
        x1 = fraction * x_t
        x2 = (1.0 - fraction) * x_t
        z_mag = _parallel(rm, 1j * xm)
        z_rotor = r2 / slip + 1j * x2
        z_in = r1 + 1j * x1 + _parallel(z_mag, z_rotor)
        i1 = 1.0 / z_in
        i2 = (1.0 - i1 * (r1 + 1j * x1)) / z_rotor
        p_in = np.real(np.conj(i1))
        p_mech = abs(i2) ** 2 * r2 * (1.0 - slip) / slip
        return {
            "current": abs(i1),
            "power_factor": np.cos(np.angle(z_in)),
            "efficiency": p_mech / p_in if p_in > floor else 0.0,
            "mechanical_power": p_mech,
        }

    def best_adjustment(values, indexes, error_function):
        candidates = []
        for index in indexes:
            for direction in (1.0, -1.0):
                candidate = values.copy()
                candidate[index] *= 1.0 + direction * delta
                candidate = protect(candidate)
                candidates.append((error_function(circuit(candidate)), candidate))
        return min(candidates, key=lambda item: item[0])[1]

    def mechanical_power_error(output):
        return relative_error(output["mechanical_power"], 1.0)

    def current_error(output):
        return (
            relative_error(output["current"], i_rated) ** 2
            + relative_error(output["power_factor"], rating.power_factor) ** 2
        )

    def efficiency_error(output):
        return relative_error(output["efficiency"], rating.efficiency)

    converged = False
    for iteration in range(1, max_iterations + 1):
        output = circuit(parameters)
        mechanical_power_ok = in_band(output["mechanical_power"], 1.0)
        current_ok = in_band(output["current"], i_rated)
        power_factor_ok = in_band(output["power_factor"], rating.power_factor)
        efficiency_ok = in_band(output["efficiency"], rating.efficiency)

        if mechanical_power_ok and current_ok and power_factor_ok and efficiency_ok:
            converged = True
            break
        if not mechanical_power_ok:
            parameters = best_adjustment(parameters, (1,), mechanical_power_error)
        elif not (current_ok and power_factor_ok):
            parameters = best_adjustment(parameters, (2, 3, 4), current_error)
        else:
            parameters = best_adjustment(parameters, (0,), efficiency_error)

    r1, r2, x_t, xm, rm = protect(parameters)
    output = circuit(parameters)
    residual = np.sqrt(
        np.mean(
            [
                relative_error(output["mechanical_power"], 1.0) ** 2,
                relative_error(output["current"], i_rated) ** 2,
                relative_error(output["power_factor"], rating.power_factor) ** 2,
                relative_error(output["efficiency"], rating.efficiency) ** 2,
            ]
        )
    )
    warning = ""
    if not converged:
        warning = "ASDP did not converge within the maximum number of iterations."

    return EstimationResult(
        method="ASDP",
        stator_resistance=float(r1 * z_base),
        rotor_resistance=float(r2 * z_base),
        stator_reactance=float(fraction * x_t * z_base),
        rotor_reactance=float((1.0 - fraction) * x_t * z_base),
        mutual_reactance=float(xm * z_base),
        Rc=float(rm * z_base),
        converged=converged,
        iterations=iteration,
        residual=float(residual),
        warnings=warning,
        extra={"starting_current_pu": float(i_start)},
    )


def kueck(rating, known_stator_resistance=None, tolerance=1e-8, max_iterations=500):
    """Estimate the parameters with the Kueck NEQ method.

    The rotor current is found by a relaxed fixed-point iteration. Friction and
    windage losses are 1.2 % of the rated output power and stray load losses
    are represented by a resistance in series with the rotor branch. The total
    leakage reactance follows from the starting current, neglecting the
    magnetizing branch, and is split according to the motor category. If an
    iteration becomes non-physical after a valid solution was found, that
    solution is returned with ``converged=False``.

    Parameters
    ----------
    rating : _Rating
        Rated data of the motor.
    known_stator_resistance : float, optional
        Stator resistance [Ohm]. Default is an empirical estimate from the
        rated power, voltage and number of poles.
    tolerance : float, optional
        Maximum relative variation of the rotor current, Xm and Rc between
        iterations.
    max_iterations : int, optional
        Maximum number of iterations.

    Returns
    -------
    result : EstimationResult

    References
    ----------
    Kueck, J. D. et al. Assessment of Methods for Estimating Motor Efficiency
    and Load Under Field Conditions. Oak Ridge National Laboratory,
    ORNL/TM-13165, 1996.
    """
    slip = rating.slip
    v_phase = rating.voltage_rated
    i_rated = rating.current_rated
    i_start = rating.starting_current
    ratio = rating.x1_x2_ratio
    p_out = rating.power_rated

    if known_stator_resistance is None:
        r1 = (
            0.00011
            * rating.n_poles**0.52
            * rating.power_hp ** (-1.26)
            * phase_to_line(rating.voltage_rated) ** 2
        )
    else:
        r1 = float(known_stator_resistance)
    if r1 <= 0.0:
        raise ValueError("The stator resistance must be positive.")

    p_sll = _stray_loss_fraction(p_out) * p_out
    p_fw = 0.012 * p_out
    angle = np.arccos(rating.power_factor)
    z_target = v_phase / i_rated * (np.cos(angle) + 1j * np.sin(angle))

    def step(i2, xm, rc):
        rsll = p_sll / (3.0 * i2**2)
        r2 = (p_out + p_fw) * slip / (3.0 * i2**2 * (1.0 - slip))
        if not np.isfinite(r2) or r2 <= 0.0:
            raise _Unphysical("Kueck produced a non-physical R2.")

        radicand = (v_phase / i_start) ** 2 - (r1 + r2) ** 2
        if not np.isfinite(radicand) or radicand <= 0.0:
            raise _Unphysical("Kueck produced a non-real leakage reactance.")
        x2 = np.sqrt(radicand) / (1.0 + ratio)
        x1 = ratio * x2

        z_rotor = r2 / slip + rsll + 1j * x2
        z_parallel = z_target - (r1 + 1j * x1)
        denominator = z_rotor - z_parallel
        if abs(denominator) < 1e-12:
            raise _Unphysical("The excitation branch is undetermined.")
        z_exc = z_parallel * z_rotor / denominator
        if abs(z_exc) < 1e-12:
            raise _Unphysical("The excitation impedance is zero.")
        y_exc = 1.0 / z_exc
        if not np.all(np.isfinite([y_exc.real, y_exc.imag])) or (
            y_exc.real <= 0.0 or y_exc.imag >= 0.0
        ):
            raise _Unphysical("Kueck produced a non-physical Xm or Rc.")
        xm_new = -1.0 / y_exc.imag
        rc_new = 1.0 / y_exc.real

        xm = (1.0 - relaxation) * xm + relaxation * xm_new
        rc = (1.0 - relaxation) * rc + relaxation * rc_new

        z_in = r1 + 1j * x1 + _parallel(_parallel(1j * xm, rc + 0j), z_rotor)
        if not np.all(np.isfinite([z_in.real, z_in.imag])) or abs(z_in) < 1e-12:
            raise _Unphysical("Kueck produced a non-physical input impedance.")
        i1 = v_phase / z_in
        i2_new = abs((v_phase - i1 * (r1 + 1j * x1)) / z_rotor)
        if not np.isfinite(i2_new) or i2_new <= 0.0:
            raise _Unphysical("Kueck produced a non-physical rotor current.")
        i2 = (1.0 - relaxation) * i2 + relaxation * i2_new
        return r2, x1, x2, xm, rc, rsll, i2

    relaxation = 0.20
    i2 = max(i_rated * rating.power_factor, 1e-6)
    xm = 20.0
    rc = 1000.0
    error = np.inf
    converged = False
    warning = ""
    last_valid = None

    for iteration in range(1, max_iterations + 1):
        previous = np.array([i2, xm, rc])
        try:
            state = step(i2, xm, rc)
        except _Unphysical as exc:
            if last_valid is None:
                raise RuntimeError(
                    f"Kueck did not obtain a valid solution. Reason: {exc}"
                ) from exc
            warning = f"{exc} The last valid solution was kept."
            break
        r2, x1, x2, xm, rc, rsll, i2 = state
        current = np.array([i2, xm, rc])
        error = float(np.max(np.abs((current - previous) / np.maximum(current, 1e-12))))
        last_valid = (state, error, iteration)
        if error < tolerance:
            converged = True
            break
    else:
        warning = "Kueck reached the maximum number of iterations. The last solution was kept."

    (r2, x1, x2, xm, rc, rsll, i2), error, iteration = last_valid

    return EstimationResult(
        method="Kueck",
        stator_resistance=float(r1),
        rotor_resistance=float(r2),
        stator_reactance=float(x1),
        rotor_reactance=float(x2),
        mutual_reactance=float(xm),
        Rc=float(rc),
        converged=converged,
        iterations=iteration,
        residual=float(error),
        warnings=warning,
        extra={"Rsll": float(rsll), "Pfw": float(p_fw), "Psll": float(p_sll)},
    )


def lee(
    rating,
    tolerance=1e-3,
    max_iterations=100,
    mechanical_loss_fraction=0.14,
    core_loss_fraction=0.12,
):
    """Estimate the parameters with the method of Lee et al.

    The method is a Gauss-Seidel iteration in per unit, with the rated output
    power as base. Mechanical and core losses are fractions of the total
    losses and stray load losses are a fraction of the output power that
    depends on the rated power. R1 is computed once, from the power balance.

    Parameters
    ----------
    rating : _Rating
        Rated data of the motor.
    tolerance : float, optional
        Maximum variation of the parameters in per unit between iterations.
    max_iterations : int, optional
        Maximum number of iterations.
    mechanical_loss_fraction : float, optional
        Mechanical losses as a fraction of the total losses.
    core_loss_fraction : float, optional
        Core losses as a fraction of the total losses.

    Returns
    -------
    result : EstimationResult

    Raises
    ------
    RuntimeError
        If the iteration produces a non-physical circuit or does not converge.

    References
    ----------
    K. Lee, S. Frank, P. K. Sen, L. G. Polese, M. Alahmad and C. Waters,
    "Estimation of induction motor equivalent circuit parameters from nameplate
    data," 2012 North American Power Symposium (NAPS), pp. 1-6,
    doi: 10.1109/NAPS.2012.6336384.
    """
    slip = rating.slip
    fraction = rating.x1_fraction
    i_base, z_base = rating.per_unit_base()
    i_start = rating.starting_current / i_base

    p_in = 1.0 / rating.efficiency
    p_loss = p_in - 1.0
    q_in = np.sqrt((p_in / rating.power_factor) ** 2 - p_in**2)
    i1 = p_in - 1j * q_in

    p_core = core_loss_fraction * p_loss
    p_conv = 1.0 + mechanical_loss_fraction * p_loss
    p_conv += _stray_loss_fraction(rating.power_rated)
    p_ag = p_conv / (1.0 - slip)

    r1 = (p_in - p_ag - p_core) / abs(i1) ** 2
    if r1 <= 0.0:
        raise RuntimeError("Lee produced a non-physical R1.")

    x1 = r2 = x2 = rc = xm = 0.0
    e = 1.0 + 0j
    i2 = abs(np.real(i1))
    error = np.inf

    for iteration in range(1, max_iterations + 1):
        previous = np.array([x1, r2, x2, rc, xm])

        discriminant = abs(e) ** 4 - 4.0 * p_ag**2 * x2**2
        if discriminant < 0.0:
            raise RuntimeError("Lee produced a negative discriminant for R2.")
        r2 = slip * (abs(e) ** 2 + np.sqrt(discriminant)) / (2.0 * p_ag)

        term = (1.0 / i_start) ** 2 - (r1 + r2) ** 2
        if term <= 0.0:
            raise RuntimeError("Lee could not compute X1 + X2.")
        x_total = np.sqrt(term)
        x1 = fraction * x_total
        x2 = (1.0 - fraction) * x_total

        q_m = q_in - abs(i1) ** 2 * x1 - i2**2 * x2
        if q_m <= 0.0:
            raise RuntimeError("Lee produced a non-physical magnetizing power.")
        xm = abs(e) ** 2 / q_m
        rc = abs(e) ** 2 / p_core

        error = float(np.max(np.abs(np.array([x1, r2, x2, rc, xm]) - previous)))
        if error < tolerance:
            break

        e = 1.0 - i1 * (r1 + 1j * x1)
        i2 = abs(e) / np.sqrt((r2 / slip) ** 2 + x2**2)
    else:
        raise RuntimeError("Lee did not converge.")

    return EstimationResult(
        method="Lee",
        stator_resistance=float(r1 * z_base),
        rotor_resistance=float(r2 * z_base),
        stator_reactance=float(x1 * z_base),
        rotor_reactance=float(x2 * z_base),
        mutual_reactance=float(xm * z_base),
        Rc=float(rc * z_base),
        iterations=iteration,
        residual=error,
    )


def haque1(
    rating,
    starting_torque_ratio,
    efficiency_50,
    power_factor_50,
    tolerance=1e-8,
    max_iterations=100,
):
    """Estimate the parameters with the Haque 1 method.

    R1 and the constant losses follow from the loss balances at rated load and
    half load. The total leakage reactance follows from the ratio between the
    starting and rated torques and is split according to the motor category.
    Xm follows from the reactive power balance. Constant losses are split
    equally between core and friction and windage losses.

    Parameters
    ----------
    rating : _Rating
        Rated data of the motor.
    starting_torque_ratio : float
        Ratio between starting and rated torque.
    efficiency_50 : float
        Efficiency at 50 % of the rated load.
    power_factor_50 : float
        Power factor at 50 % of the rated load.
    tolerance : float, optional
        Maximum relative variation of the parameters between iterations.
    max_iterations : int, optional
        Maximum number of iterations.

    Returns
    -------
    result : EstimationResult

    Raises
    ------
    RuntimeError
        If the iteration produces a non-physical circuit or does not converge.

    References
    ----------
    Haque, M. H. (1993). Estimation of three-phase induction motor parameters.
    Electric Power Systems Research, 26, 187-193.

    See also [1] in the module documentation.
    """
    if starting_torque_ratio <= 0.0:
        raise ValueError("starting_torque_ratio must be positive.")
    if not 0.0 < efficiency_50 <= 1.0:
        raise ValueError("efficiency_50 must be in (0, 1].")
    if not 0.0 < power_factor_50 <= 1.0:
        raise ValueError("power_factor_50 must be in (0, 1].")

    p_out = rating.power_rated
    v_phase = rating.voltage_rated
    eta = rating.efficiency
    pf = rating.power_factor
    slip = rating.slip
    fraction = rating.x1_fraction

    i1_rated = rating.input_power / (3.0 * v_phase * pf)
    i1_half = 0.5 * p_out / (3.0 * v_phase * efficiency_50 * power_factor_50)
    phi = np.arccos(pf)
    i1_phasor = i1_rated * np.exp(-1j * phi)

    e = v_phase
    p_fw = 0.0
    i2 = i1_rated * pf
    previous = None
    error = np.inf

    for iteration in range(1, max_iterations + 1):
        r2 = (p_out + p_fw) * slip / (3.0 * i2**2 * (1.0 - slip))

        matrix = np.array([[3.0 * i1_rated**2, 1.0], [3.0 * i1_half**2, 1.0]])
        rhs = np.array(
            [
                p_out * (1.0 / eta - 1.0) - 3.0 * i2**2 * r2,
                0.5 * p_out * (1.0 / efficiency_50 - 1.0) - 1.5 * i2**2 * r2,
            ]
        )
        r1, p_rot = np.linalg.solve(matrix, rhs)
        if r1 <= 0.0 or r2 <= 0.0 or p_rot <= 0.0:
            raise RuntimeError("Haque 1 produced a non-physical resistance or loss.")

        p_fw_new = 0.5 * p_rot
        rc = 3.0 * e**2 / (0.5 * p_rot)

        radicand = (1.0 / starting_torque_ratio) * (
            3.0 * v_phase**2 * r2 * (1.0 - slip) / p_out
        ) - (r1 + r2) ** 2
        if radicand <= 0.0:
            raise RuntimeError("Haque 1 produced a non-real leakage reactance.")
        x_total = np.sqrt(radicand)
        x1 = fraction * x_total
        x2 = (1.0 - fraction) * x_total

        denominator = v_phase * i1_rated * np.sin(phi) - i1_rated**2 * x1 - i2**2 * x2
        if denominator <= 0.0:
            raise RuntimeError("Haque 1 produced a non-physical Xm.")
        xm = e**2 / denominator

        e_phasor = v_phase - i1_phasor * (r1 + 1j * x1)
        z_mag = _parallel(rc + 0j, 1j * xm)
        i2_phasor = i1_phasor - e_phasor / z_mag

        parameters = np.array([r1, r2, x1, x2, xm, rc, p_fw_new])
        if previous is not None:
            error = float(
                np.max(
                    np.abs(parameters - previous) / np.maximum(np.abs(previous), 1e-12)
                )
            )
        e = abs(e_phasor)
        i2 = abs(i2_phasor)
        p_fw = p_fw_new
        if error < tolerance:
            break
        previous = parameters
    else:
        raise RuntimeError("Haque 1 did not converge.")

    return EstimationResult(
        method="Haque 1",
        stator_resistance=float(r1),
        rotor_resistance=float(r2),
        stator_reactance=float(x1),
        rotor_reactance=float(x2),
        mutual_reactance=float(xm),
        Rc=float(rc),
        iterations=iteration,
        residual=error,
        extra={"Prot": float(p_rot), "Pcore": float(0.5 * p_rot), "Pfw": float(p_fw)},
    )


def natarajan_misra(
    rating,
    efficiency_75,
    efficiency_50,
    power_factor_75,
    power_factor_50,
    starting_torque_ratio,
    maximum_torque_ratio,
    x1_x2_ratio=None,
):
    """Estimate the parameters with the method of Natarajan and Misra.

    The method is non-iterative. R1 + R2 and the constant losses follow from
    the loss balances at rated load and 75 % load. The 50 % load point only
    checks the estimated losses, which is reported in ``residual``. Constant
    losses are split equally between core and friction and windage losses.
    X2 follows from the ratio between starting and maximum torque, and X1 from
    the ratio X1/X2.

    Parameters
    ----------
    rating : _Rating
        Rated data of the motor.
    efficiency_75, efficiency_50 : float
        Efficiency at 75 % and 50 % of the rated load.
    power_factor_75, power_factor_50 : float
        Power factor at 75 % and 50 % of the rated load.
    starting_torque_ratio : float
        Ratio between starting and rated torque.
    maximum_torque_ratio : float
        Ratio between maximum and rated torque.
    x1_x2_ratio : float, optional
        Ratio between stator and rotor leakage reactances. Default follows from
        the motor category.

    Returns
    -------
    result : EstimationResult

    Raises
    ------
    RuntimeError
        If the method produces a non-physical circuit.

    References
    ----------
    Natarajan and Misra (1989). Parameter Estimation of Induction Motors Using
    a Spreadsheet Program on a Personal Computer. Electric Power Systems
    Research, 16, 157-164.

    See also [1] in the module documentation, for the corrections adopted.
    """
    p_out = rating.power_rated
    v_phase = rating.voltage_rated
    pf = rating.power_factor
    slip = rating.slip

    i_rated = rating.current_rated
    i_75 = 0.75 * p_out / (3.0 * v_phase * efficiency_75 * power_factor_75)
    i_50 = 0.50 * p_out / (3.0 * v_phase * efficiency_50 * power_factor_50)

    loss = p_out * (1.0 / rating.efficiency - 1.0)
    loss_75 = 0.75 * p_out * (1.0 / efficiency_75 - 1.0)
    loss_50 = 0.50 * p_out * (1.0 / efficiency_50 - 1.0)

    matrix = np.array([[3.0 * i_rated**2, 1.0], [3.0 * i_75**2, 1.0]])
    r_total, k = np.linalg.solve(matrix, np.array([loss, loss_75]))
    if r_total <= 0.0 or k <= 0.0:
        raise RuntimeError("Natarajan-Misra produced a non-physical Rt or K.")

    loss_50_calculated = 3.0 * i_50**2 * r_total + k
    error_50 = (loss_50_calculated - loss_50) / loss_50

    p_core = 0.5 * k
    rc = 3.0 * v_phase**2 / p_core
    i_core = v_phase / rc

    iq = i_rated * np.sqrt(1.0 - pf**2)
    iq_75 = i_75 * np.sqrt(1.0 - power_factor_75**2)
    b = 4.0 * (iq - iq_75)
    i_mag = iq - b
    if i_mag <= 0.0:
        raise RuntimeError("Natarajan-Misra produced a non-physical Im.")
    xm = v_phase / i_mag

    i2 = np.sqrt((i_rated * pf - i_core) ** 2 + b**2)
    r2 = p_out * slip / (3.0 * i2**2 * (1.0 - slip))
    r1 = r_total - r2
    if r1 <= 0.0 or r2 <= 0.0:
        raise RuntimeError("Natarajan-Misra produced a non-physical R1 or R2.")

    q = starting_torque_ratio / maximum_torque_ratio
    if not 0.0 < q <= 1.0:
        raise RuntimeError("Tst/Tm must be in (0, 1] to obtain X2.")
    x2 = r2 * q / (1.0 - np.sqrt(1.0 - q**2))

    if x1_x2_ratio is None:
        x1_x2_ratio = rating.x1_x2_ratio

    return EstimationResult(
        method="Natarajan-Misra",
        stator_resistance=float(r1),
        rotor_resistance=float(r2),
        stator_reactance=float(x1_x2_ratio * x2),
        rotor_reactance=float(x2),
        mutual_reactance=float(xm),
        Rc=float(rc),
        residual=abs(float(error_50)),
        extra={"K": float(k), "Pcore": float(p_core), "Pfw": float(0.5 * k)},
    )


def haque2(
    rating,
    starting_torque_ratio,
    maximum_torque_ratio,
    alpha_r=0.50,
    alpha_c=0.50,
    max_nfev=5000,
    residual_tolerance=1e-6,
):
    """Estimate the parameters with the Haque 2 method.

    X1, R20, X20, Xm and the constant losses are adjusted simultaneously by
    Levenberg-Marquardt to match the rated output, active and reactive powers
    and the starting and maximum torques. R1 is ``alpha_r * R20``. Rotor
    resistance and reactance vary with slip above the slip of maximum torque
    ``sm``; the returned R2 and X2 are the reference values R20 and X20.

    Parameters
    ----------
    rating : _Rating
        Rated data of the motor.
    starting_torque_ratio : float
        Ratio between starting and rated torque.
    maximum_torque_ratio : float
        Ratio between maximum and rated torque.
    alpha_r : float, optional
        Ratio R1/R20, between 0.5 and 2.5.
    alpha_c : float, optional
        Fraction of the constant losses that are core losses, between zero and
        one.
    max_nfev : int, optional
        Maximum number of function evaluations.
    residual_tolerance : float, optional
        Maximum root mean square of the normalized residuals.

    Returns
    -------
    result : EstimationResult
        ``extra["smax"]`` is the slip of maximum torque.

    References
    ----------
    M. H. Haque, "Determination of NEMA Design Induction Motor Parameters From
    Manufacturer Data," IEEE Transactions on Energy Conversion, 2008.

    See also [1] in the module documentation.
    """
    if starting_torque_ratio <= 0.0:
        raise ValueError("starting_torque_ratio must be positive.")
    if maximum_torque_ratio <= 0.0:
        raise ValueError("maximum_torque_ratio must be positive.")
    if not 0.5 <= alpha_r <= 2.5:
        raise ValueError("alpha_r must be in [0.5, 2.5].")
    if not 0.0 <= alpha_c <= 1.0:
        raise ValueError("alpha_c must be in [0, 1].")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive.")

    v1 = rating.voltage_rated
    s_r = rating.slip
    p_out = rating.power_rated
    p_in = rating.input_power
    q_in = p_in * np.tan(np.arccos(rating.power_factor))
    i1_rated = p_in / (3.0 * v1 * rating.power_factor)
    omega_s = rating.omega_sync
    omega_r = rating.speed_rated

    def rotor_parameters(s, sm, r20, x20):
        if s <= sm:
            return r20, x20
        return (
            (0.5 + 0.5 * np.sqrt(s / sm)) * r20,
            (0.4 + 0.6 * np.sqrt(sm / s)) * x20,
        )

    def shunt(xm, rc):
        return 1j * xm if np.isinf(rc) else _parallel(rc + 0j, 1j * xm)

    def core_resistance(x1, r20, x20, xm, p_rot):
        p_core = alpha_c * p_rot
        if p_core == 0.0:
            return np.inf
        r1 = alpha_r * r20
        rc = 3.0 * v1**2 / p_core
        for _ in range(100):
            z_in = r1 + 1j * x1 + _parallel(shunt(xm, rc), r20 / s_r + 1j * x20)
            e = v1 - v1 / z_in * (r1 + 1j * x1)
            rc_new = 3.0 * abs(e) ** 2 / p_core
            if abs(rc_new - rc) / max(abs(rc_new), 1.0) < 1e-10:
                return float(rc_new)
            rc = rc_new
        return float(rc)

    def slip_of_maximum_torque(x1, r20, x20, xm, rc):
        z1 = alpha_r * r20 + 1j * x1
        z_th = _parallel(z1, shunt(xm, rc))
        den = np.sqrt(np.real(z_th) ** 2 + (np.imag(z_th) + x20) ** 2)
        if den <= 0.0:
            raise FloatingPointError("Invalid sm.")
        return float(r20 / den)

    def performance(s, x1, r20, x20, xm, p_rot, rc, sm):
        r1 = alpha_r * r20
        r2, x2 = rotor_parameters(s, sm, r20, x20)
        z2 = r2 / s + 1j * x2
        i1 = v1 / (r1 + 1j * x1 + _parallel(shunt(xm, rc), z2))
        e = v1 - i1 * (r1 + 1j * x1)
        p_ag = 3.0 * abs(e / z2) ** 2 * r2 / s
        s_in = 3.0 * v1 * np.conj(i1)
        return {
            "Po": (1.0 - s) * p_ag - (1.0 - alpha_c) * p_rot,
            "Pin": float(np.real(s_in)),
            "Qin": float(np.imag(s_in)),
            "T": p_ag / omega_s,
        }

    def evaluate(x):
        x1, r20, x20, xm, p_rot = map(float, x)
        if min(x1, r20, x20, xm, p_rot) <= 0.0:
            raise FloatingPointError("Non-positive parameter.")
        rc = core_resistance(x1, r20, x20, xm, p_rot)
        sm = slip_of_maximum_torque(x1, r20, x20, xm, rc)
        xr = x1 / x20
        sm_star = ((0.8198 + xr) / (0.2395 + xr)) * sm

        rated = performance(s_r, x1, r20, x20, xm, p_rot, rc, sm)
        starting = performance(1.0, x1, r20, x20, xm, p_rot, rc, sm)
        maximum = performance(sm_star, x1, r20, x20, xm, p_rot, rc, sm)

        t_internal = (p_out + (1.0 - alpha_c) * p_rot) / omega_r
        return {
            "rc": rc,
            "sm": sm,
            "rated": rated,
            "Tst": starting["T"],
            "Tm": maximum["T"],
            "Tst_target": starting_torque_ratio * t_internal,
            "Tm_target": maximum_torque_ratio * t_internal,
        }

    def residuals(x):
        try:
            v = evaluate(x)
        except (FloatingPointError, ZeroDivisionError, ValueError, OverflowError):
            return np.full(5, 1e20)
        return np.array(
            [
                p_out - v["rated"]["Po"],
                p_in - v["rated"]["Pin"],
                q_in - v["rated"]["Qin"],
                v["Tm_target"] - v["Tm"],
                v["Tst_target"] - v["Tst"],
            ]
        )

    x1_0 = 0.05 * v1 / i1_rated
    x0 = np.array(
        [
            x1_0,
            s_r * p_out / (3.0 * i1_rated**2),
            x1_0,
            v1 / (0.2 * i1_rated),
            0.03 * p_out,
        ]
    )
    solution = least_squares(
        residuals,
        x0,
        method="lm",
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    final = evaluate(solution.x)
    f = residuals(solution.x)
    scales = np.maximum(
        np.abs([p_out, p_in, q_in, final["Tm_target"], final["Tst_target"]]), 1.0
    )
    residual = float(np.sqrt(np.mean((f / scales) ** 2)))

    converged = bool(
        solution.success
        and np.all(np.isfinite(solution.x))
        and np.all(solution.x > 0.0)
        and residual <= residual_tolerance
    )
    messages = []
    if not solution.success:
        messages.append(solution.message)
    if np.any(solution.x <= 0.0):
        messages.append("The solution contains non-physical parameters.")
    if residual > residual_tolerance:
        messages.append("The relative residual stayed above the tolerance.")

    x1, r20, x20, xm, p_rot = solution.x
    return EstimationResult(
        method="Haque 2",
        stator_resistance=float(alpha_r * r20),
        rotor_resistance=float(r20),
        stator_reactance=float(x1),
        rotor_reactance=float(x20),
        mutual_reactance=float(xm),
        Rc=float(final["rc"]),
        converged=converged,
        iterations=int(solution.nfev),
        residual=residual,
        warnings=" ".join(messages),
        extra={
            "Prot": float(p_rot),
            "Pcore": float(alpha_c * p_rot),
            "Pfw": float((1.0 - alpha_c) * p_rot),
            "smax": float(final["sm"]),
        },
    )


NAMEPLATE_METHODS = {"asdp": asdp, "kueck": kueck, "lee": lee}
CATALOG_METHODS = {
    "haque1": ("starting_torque_ratio", "efficiency_50", "power_factor_50"),
    "natarajan_misra": (
        "efficiency_75",
        "efficiency_50",
        "power_factor_75",
        "power_factor_50",
        "starting_torque_ratio",
        "maximum_torque_ratio",
    ),
    "haque2": ("starting_torque_ratio", "maximum_torque_ratio"),
}
_CATALOG_FUNCTIONS = {
    "haque1": haque1,
    "natarajan_misra": natarajan_misra,
    "haque2": haque2,
}


def estimate_from_nameplate(
    method,
    power_rated,
    voltage_rated,
    speed_rated,
    frequency_rated,
    n_poles,
    efficiency,
    power_factor,
    current_rated=None,
    starting_current_ratio=6.0,
    motor_category="N",
    method_options=None,
):
    """Run a nameplate-only estimation method.

    Parameters
    ----------
    method : str
        One of ``NAMEPLATE_METHODS``: "asdp", "kueck" or "lee".
    power_rated : float
        Rated output power [W].
    voltage_rated : float
        Rated phase voltage [V].
    speed_rated : float
        Rated mechanical speed [rad/s].
    frequency_rated : float
        Rated electrical frequency [rad/s].
    n_poles : int
        Number of poles.
    efficiency : float
        Rated efficiency, between zero and one.
    power_factor : float
        Rated power factor, between zero and one.
    current_rated : float, optional
        Rated phase current [A]. Default is computed from the input power.
    starting_current_ratio : float, optional
        Ratio between starting and rated current.
    motor_category : str, optional
        Motor category, "N", "H" or "D".
    method_options : dict, optional
        Options of the method.

    Returns
    -------
    result : EstimationResult

    Examples
    --------
    >>> result = estimate_from_nameplate(
    ...     "lee",
    ...     power_rated=18642.5,
    ...     voltage_rated=265.6,
    ...     speed_rated=180.6,
    ...     frequency_rated=377.0,
    ...     n_poles=4,
    ...     efficiency=0.83,
    ...     power_factor=0.9,
    ... )
    >>> result.method
    'Lee'
    """
    if method not in NAMEPLATE_METHODS:
        raise ValueError(f"method must be one of {sorted(NAMEPLATE_METHODS)}.")
    rating = _Rating(
        power_rated,
        voltage_rated,
        speed_rated,
        frequency_rated,
        n_poles,
        efficiency,
        power_factor,
        current_rated,
        starting_current_ratio,
        motor_category,
    )
    return NAMEPLATE_METHODS[method](rating, **(method_options or {}))


def estimate_from_catalog(
    method,
    catalog,
    power_rated,
    voltage_rated,
    speed_rated,
    frequency_rated,
    n_poles,
    efficiency,
    power_factor,
    current_rated=None,
    starting_current_ratio=6.0,
    motor_category="N",
    method_options=None,
):
    """Run a catalog-data estimation method.

    Parameters
    ----------
    method : str
        One of ``CATALOG_METHODS``: "haque1", "natarajan_misra" or "haque2".
    catalog : dict
        Catalog data. It must contain the keys required by the method, listed
        in ``CATALOG_METHODS``.
    power_rated : float
        Rated output power [W].
    voltage_rated : float
        Rated phase voltage [V].
    speed_rated : float
        Rated mechanical speed [rad/s].
    frequency_rated : float
        Rated electrical frequency [rad/s].
    n_poles : int
        Number of poles.
    efficiency : float
        Rated efficiency, between zero and one.
    power_factor : float
        Rated power factor, between zero and one.
    current_rated : float, optional
        Rated phase current [A]. Default is computed from the input power.
    starting_current_ratio : float, optional
        Ratio between starting and rated current.
    motor_category : str, optional
        Motor category, "N", "H" or "D".
    method_options : dict, optional
        Options of the method.

    Returns
    -------
    result : EstimationResult
    """
    if method not in CATALOG_METHODS:
        raise ValueError(f"method must be one of {sorted(CATALOG_METHODS)}.")
    missing = [name for name in CATALOG_METHODS[method] if catalog.get(name) is None]
    if missing:
        raise ValueError(f"The method '{method}' requires {', '.join(missing)}.")
    rating = _Rating(
        power_rated,
        voltage_rated,
        speed_rated,
        frequency_rated,
        n_poles,
        efficiency,
        power_factor,
        current_rated,
        starting_current_ratio,
        motor_category,
    )
    data = {name: catalog[name] for name in CATALOG_METHODS[method]}
    return _CATALOG_FUNCTIONS[method](rating, **data, **(method_options or {}))
