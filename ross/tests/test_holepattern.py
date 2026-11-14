import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ross.seals.holepattern_seal import HolePatternSeal
from ross.units import Q_

import ccp

# Detect REFPROP availability using ccp
# ccp is always available as a dependency, but REFPROP is optional
# Check if REFPROP version is available (not "n/a")
REFPROP_AVAILABLE = "REFPROP : n/a" not in ccp.__version__full


# Common test parameters
COMMON_PARAMS = {
    "n": 0,
    "speed": Q_([5000], "RPM"),
    "axial_length": 0.0254,
    "shaft_diameter": 0.1502,
    "radial_clearance": 0.0004,
    "relative_roughness": 0.00198,
    "cell_length": 0.001,
    "cell_width": 0.001,
    "cell_depth": 0.00229,
    "inlet_pressure": 1830000.0,
    "outlet_pressure": 823500.0,
    "inlet_temperature": 300.0,
    "preswirl": 1.0,
    "entrance_loss_coefficient": 0.1,
    "exit_loss_coefficient": 0.5,
}

MANUAL_PARAMS = {
    "sutherland_b": 1.458e-6,
    "sutherland_s": 110.4,
    "molar_mass": 29.0,
    "gamma": 1.4,
}

GAS_COMPOSITION = {
    "Nitrogen": 0.7812,
    "Oxygen": 0.2096,
    "Argon": 0.0092,
}

COEFFICIENTS = ("kxx", "kxy", "cxx", "cxy")

# Childs and Wade (2004), J. Tribol. 126(2):326-333, Tables 2 and 4. The cell
# depth is the hole depth times the hole-area ratio (volume-equivalent depth);
# the hole diameter is given as cell length and width, which the model ignores.
CHILDS_WADE = {
    "n": 0,
    "shaft_diameter": 114.721e-3,
    "axial_length": 85.7e-3,
    "relative_roughness": 0.0,
    "cell_length": 3.18e-3,
    "cell_width": 3.18e-3,
    "cell_depth": 3.30e-3 * 0.684,
    "inlet_pressure": 70e5,
    "entrance_loss_coefficient": 0.1,
    "exit_loss_coefficient": 0.5,
}
CLEARANCE_020 = 0.5 * (0.2115 + 0.2102) * 1e-3
CLEARANCE_010 = 0.5 * (0.1099 + 0.1086) * 1e-3

AIR = {
    "molar_mass": 28.97,
    "gamma": 1.4,
    "sutherland_b": 1.458e-6,
    "sutherland_s": 110.4,
}

FIG4 = {
    **CHILDS_WADE,
    **AIR,
    "radial_clearance": CLEARANCE_020,
    "preswirl": 0.598,
    "outlet_pressure": 0.49 * 70e5,
    "inlet_temperature": 292.21,
    "speed": Q_([20200], "RPM"),
}

FIG6 = {
    **CHILDS_WADE,
    **AIR,
    "radial_clearance": CLEARANCE_010,
    "preswirl": 0.303,
    "outlet_pressure": 0.47 * 70e5,
    "inlet_temperature": 285.02,
    "speed": Q_([20200], "RPM"),
}

METHANE = {
    **CHILDS_WADE,
    "radial_clearance": CLEARANCE_020,
    "inlet_pressure": 20e6,
    "outlet_pressure": 10e6,
    "inlet_temperature": 300.0,
    "preswirl": 0.2,
    "gas_composition": {"methane": 1.0},
    "speed": Q_([15200], "RPM"),
}

VDW_SEAL = {
    **CHILDS_WADE,
    **AIR,
    "radial_clearance": CLEARANCE_020,
    "relative_roughness": 1e-3,
    "inlet_pressure": 30e6,
    "outlet_pressure": 15e6,
    "inlet_temperature": 380.0,
    "preswirl": 0.3,
    "speed": Q_([15200], "RPM"),
}


def _rad(rpm):
    return Q_(rpm, "RPM").to("rad/s").m


def _unsolved(**inputs):
    """Build a seal from placeholder coefficients, so that nothing is solved."""
    return HolePatternSeal(**inputs, kxx=0.0, cxx=0.0)


def _relative(a, b):
    return float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1.0)))


def _state_functions(gas):
    """Return ``function(p, T) -> (rho, h, s, mu)`` of a closed-form gas."""

    def function(p, T):
        point = gas.point(p, T)
        return point.rho, point.h, gas.entropy(p, T), point.mu

    return function


def _van_der_waals():
    """Van der Waals gas with a density-dependent viscosity, in closed form.

    ``p = rho R T / (1 - b rho) - a rho^2``, ``h = cv T - a rho + p / rho``,
    ``s = cv ln T + R ln((1 - b rho) / rho)`` and ``mu = mu0 (T / T0)^0.7 (1 +
    c rho)``. Every state derivative that vanishes for a perfect gas (``h_p``,
    ``mu_p``, the pressure dependence of ``cp``) is non-zero.
    """
    import math

    from ross.seals.gas_model import GasPoint, StateGas

    class VanDerWaals(StateGas):
        def __init__(self):
            self.gas_constant, self.a, self.b = 430.0, 60.0, 1.2e-3
            self.cv, self.c, self.mu0, self.t0 = 1700.0, 2e-2, 1.8e-5, 300.0

        def density(self, p, T):
            R, a, b = self.gas_constant, self.a, self.b
            rho = p / (R * T)
            for _ in range(100):
                f = rho * R * T / (1 - b * rho) - a * rho**2 - p
                step = f / (R * T / (1 - b * rho) ** 2 - 2 * a * rho)
                rho -= step
                if abs(step) < 1e-14 * rho:
                    return rho
            raise RuntimeError("no van der Waals density")

        def functions(self, p, T):
            rho = self.density(p, T)
            h = self.cv * T - self.a * rho + p / rho
            s = self.cv * math.log(T) + self.gas_constant * math.log(
                (1 - self.b * rho) / rho
            )
            mu = self.mu0 * (T / self.t0) ** 0.7 * (1 + self.c * rho)
            return rho, h, s, mu

        def point(self, p, T):
            R, a, b = self.gas_constant, self.a, self.b
            rho, h, _, mu = self.functions(p, T)
            p_rho = R * T / (1 - b * rho) ** 2 - 2 * a * rho
            p_t = rho * R / (1 - b * rho)
            rho_p, rho_t = 1 / p_rho, -p_t / p_rho
            h_rho = -a + (p_rho * rho - p) / rho**2
            h_t = self.cv + p_t / rho
            mu_rho = self.mu0 * (T / self.t0) ** 0.7 * self.c
            mu_t = 0.7 * mu / T
            return GasPoint(
                p=p,
                T=T,
                rho=rho,
                rho_p=rho_p,
                rho_T=rho_t,
                h=h,
                cp=h_t + h_rho * rho_t,
                h_p=h_rho * rho_p,
                mu=mu,
                mu_p=mu_rho * rho_p,
                mu_T=mu_t + mu_rho * rho_t,
            )

        def entropy(self, p, T):
            return self.functions(p, T)[2]

    return VanDerWaals()


def _narrow_gases(limit):
    """Return ``(Narrow, NeverWide)`` gas classes that refuse T > ``limit``.

    ``Narrow`` widens to the plain perfect gas with the same constants;
    ``NeverWide`` cannot widen.
    """
    from ross.seals.gas_model import GasRangeError, SutherlandGas

    class NeverWide(SutherlandGas):
        def point(self, p, T):
            if T > limit:
                raise GasRangeError(
                    f"state p = {p:.6g} Pa, T = {T:.6g} K is outside the test gas",
                    p,
                    T,
                    "above",
                )
            return super().point(p, T)

    class Narrow(NeverWide):
        def widened(self, error):
            return SutherlandGas(
                self.gas_constant, self.gamma, self.sutherland_b, self.sutherland_s
            )

    return Narrow, NeverWide


@pytest.fixture
def heos():
    original = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"
        yield
    finally:
        ccp.config.EOS = original


@pytest.fixture(scope="module")
def methane():
    """METHANE seal, real and ideal closures, built under HEOS."""
    original = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"
        return {
            "real": HolePatternSeal(**METHANE, gas_model="real"),
            "ideal": HolePatternSeal(**METHANE),
        }
    finally:
        ccp.config.EOS = original


@pytest.fixture
def holepattern_manual():
    """HolePatternSeal with manual thermodynamic parameters."""
    seal = HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS)
    return seal


@pytest.fixture
def holepattern_gas_composition():
    """HolePatternSeal with gas_composition."""
    seal = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    return seal


def test_holepattern_manual_coefficients(holepattern_manual):
    """The coefficients are isotropic and carry no mass terms.

    This test should always work regardless of REFPROP availability.
    """
    seal = holepattern_manual
    assert_array_equal(seal.kyy, seal.kxx)
    assert_array_equal(seal.kyx, -np.asarray(seal.kxy))
    assert_array_equal(seal.cyy, seal.cxx)
    assert_array_equal(seal.cyx, -np.asarray(seal.cxy))
    assert_array_equal(seal.mxx, [0.0])
    assert_array_equal(seal.mxy, [0.0])


def test_holepattern_manual_attributes(holepattern_manual):
    """Test that manual parameters are correctly stored."""
    assert holepattern_manual.molar_mass == 29.0
    assert holepattern_manual.gamma == 1.4
    assert holepattern_manual.sutherland_b == 1.458e-6
    assert holepattern_manual.sutherland_s == 110.4


def test_holepattern_gas_composition_creation(holepattern_gas_composition):
    """Test that HolePatternSeal can be created with gas_composition.

    This should work with both REFPROP and HEOS backends.
    """
    assert holepattern_gas_composition is not None
    assert hasattr(holepattern_gas_composition, "kxx")
    assert hasattr(holepattern_gas_composition, "seal_leakage")


def test_holepattern_gas_composition_derived_properties(holepattern_gas_composition):
    """Test that thermodynamic properties are auto-derived correctly."""
    # Check that properties were derived
    assert holepattern_gas_composition.molar_mass is not None
    assert holepattern_gas_composition.gamma is not None
    assert holepattern_gas_composition.sutherland_b is not None
    assert holepattern_gas_composition.sutherland_s is not None

    # Molar mass should be close to air (28.97 g/mol)
    # Expected: ~28.96 kg/kgmol, tolerance: 2%
    assert_allclose(holepattern_gas_composition.molar_mass, 28.97, rtol=0.02)

    # Gamma should be close to air (1.4)
    # Expected: ~1.4, tolerance: 3% (HEOS and REFPROP differ slightly)
    assert_allclose(holepattern_gas_composition.gamma, 1.4, rtol=0.03)

    # Sutherland coefficients should be in reasonable range for air
    # Expected: ~1.46e-6, tolerance: 25% (varies between HEOS and REFPROP)
    assert_allclose(holepattern_gas_composition.sutherland_b, 1.46e-6, rtol=0.25)
    # Expected: ~100, tolerance: 25% (varies between HEOS and REFPROP)
    assert_allclose(holepattern_gas_composition.sutherland_s, 100, rtol=0.25)


def test_holepattern_gas_composition_coefficients_range(holepattern_gas_composition):
    """The leakage of the air seal stays in its range on both backends."""
    assert_allclose(
        holepattern_gas_composition.seal_leakage[0], 0.634, rtol=0.02
    )  # ±2%


def test_holepattern_gas_composition_vs_manual_similarity():
    """Composition and manual inputs reach the same perfect-gas closure.

    A manual seal built with the molar mass, gamma and Sutherland constants
    that the composition seal derived must give the same results: the two
    input paths build the same gas. Against textbook air (gamma 1.4) the
    damping and the leakage stay close; the stiffness at the whirl frequency
    follows the cell compressibility 1/(gamma p) and is not compared.
    """
    holepattern_gas = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    derived = {
        name: getattr(holepattern_gas, name)
        for name in ("molar_mass", "gamma", "sutherland_b", "sutherland_s")
    }
    holepattern_same = HolePatternSeal(**COMMON_PARAMS, **derived)
    # two input paths to the same closure: rounding only
    for name in COEFFICIENTS + ("seal_leakage",):
        assert_allclose(
            getattr(holepattern_same, name),
            getattr(holepattern_gas, name),
            rtol=1e-12,
            err_msg=name,
        )

    holepattern_man = HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS)
    assert_allclose(holepattern_gas.cxx[0], holepattern_man.cxx[0], rtol=0.05)
    assert_allclose(
        holepattern_gas.seal_leakage[0], holepattern_man.seal_leakage[0], rtol=0.01
    )


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_holepattern_refprop_backend():
    """Test HolePatternSeal with REFPROP backend explicitly.

    This test only runs when REFPROP is available.
    """
    # Ensure we're using REFPROP
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "REFPROP"

        holepattern = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        assert_allclose(holepattern.seal_leakage[0], 0.633606, rtol=1e-3)

        # Check derived properties
        assert_allclose(holepattern.molar_mass, 28.9586, rtol=1e-3)
        assert_allclose(holepattern.gamma, 1.4321, rtol=1e-3)

    finally:
        ccp.config.EOS = original_eos


def test_holepattern_heos_backend():
    """Test HolePatternSeal with HEOS backend explicitly.

    This test should always work.
    """
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"

        holepattern = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        assert_allclose(holepattern.seal_leakage[0], 0.633636, rtol=1e-3)

        # Check derived properties
        assert_allclose(holepattern.molar_mass, 28.9586, rtol=1e-3)
        # HEOS gamma calculation should be similar to REFPROP
        assert_allclose(holepattern.gamma, 1.4321, rtol=1e-2)

    finally:
        ccp.config.EOS = original_eos


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_holepattern_refprop_vs_heos_consistency():
    """Compare REFPROP and HEOS results for the same configuration.

    Results should be within 1% of each other.
    """
    original_eos = ccp.config.EOS

    try:
        # Test with REFPROP
        ccp.config.EOS = "REFPROP"
        holepattern_refprop = HolePatternSeal(
            **COMMON_PARAMS, gas_composition=GAS_COMPOSITION
        )

        # Test with HEOS
        ccp.config.EOS = "HEOS"
        holepattern_heos = HolePatternSeal(
            **COMMON_PARAMS, gas_composition=GAS_COMPOSITION
        )

        # Compare coefficients (should be within 1%)
        assert_allclose(
            holepattern_refprop.kxx[0],
            holepattern_heos.kxx[0],
            rtol=0.01,
            err_msg="kxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            holepattern_refprop.cxx[0],
            holepattern_heos.cxx[0],
            rtol=0.01,
            err_msg="cxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            holepattern_refprop.seal_leakage[0],
            holepattern_heos.seal_leakage[0],
            rtol=0.001,
            err_msg="seal_leakage differs by >0.1% between REFPROP and HEOS",
        )

        # Derived properties should match closely
        assert_allclose(
            holepattern_refprop.molar_mass, holepattern_heos.molar_mass, rtol=1e-4
        )
        assert_allclose(holepattern_refprop.gamma, holepattern_heos.gamma, rtol=1e-2)

    finally:
        ccp.config.EOS = original_eos


def test_holepattern_without_gas_params_raises():
    """Test that creating HolePatternSeal without gas params raises error."""
    with pytest.raises(TypeError, match="gas_composition"):
        # Should fail because neither gas_composition nor manual params provided
        HolePatternSeal(**COMMON_PARAMS)


def test_holepattern_invalid_gas_composition():
    """Test that invalid gas composition raises appropriate error."""
    invalid_gas = {"InvalidFluid123": 1.0}

    with pytest.raises((ValueError, KeyError)):
        HolePatternSeal(**COMMON_PARAMS, gas_composition=invalid_gas)


def test_holepattern_multiple_frequencies():
    """Test HolePatternSeal with multiple frequency points."""
    params = COMMON_PARAMS.copy()
    params["speed"] = Q_([3000, 5000, 7000], "RPM")

    holepattern = HolePatternSeal(**params, **MANUAL_PARAMS)

    # Should have 3 values for each coefficient
    assert len(holepattern.kxx) == 3
    assert len(holepattern.cxx) == 3
    assert len(holepattern.seal_leakage) == 3

    # All should be positive
    assert all(np.array(holepattern.kxx) > 0)
    assert all(np.array(holepattern.cxx) > 0)
    assert all(np.array(holepattern.seal_leakage) > 0)


def test_holepattern_whirl_frequency_grid():
    """A 2-D table shares the base state per speed and follows the whirl frequency."""
    speed = Q_([5000, 8000], "RPM").to("rad/s").m
    params = dict(COMMON_PARAMS, **MANUAL_PARAMS)
    params["speed"] = speed
    params["nz"] = 18

    synchronous = HolePatternSeal(**params)
    half_ratio = HolePatternSeal(excitation_ratio=0.5, **params)
    frequency = np.sort([0.5 * speed[0], speed[0], 0.5 * speed[1], speed[1]])
    grid = HolePatternSeal(frequency=frequency, **params)

    assert grid.kxx_interpolated.kind == "grid"
    kxx, cxx, mxx = (np.array(getattr(grid, c)) for c in ("kxx", "cxx", "mxx"))
    assert kxx.shape == (2, 4)

    # the grid points at excitation_ratio * speed reproduce the 1-D solves
    assert kxx[0, 0] == half_ratio.kxx[0]
    assert cxx[0, 0] == half_ratio.cxx[0]
    assert mxx[0, 0] == half_ratio.mxx[0]
    assert kxx[1, 1] == half_ratio.kxx[1]
    assert kxx[0, 2] == synchronous.kxx[0]
    assert cxx[1, 3] == synchronous.cxx[1]

    # the base state does not depend on the whirl frequency
    assert_allclose(grid.seal_leakage, synchronous.seal_leakage)
    assert_allclose(grid.p[1], synchronous.p[1])


def test_holepattern_solver_row_matches_single_solves():
    params = dict(COMMON_PARAMS, **MANUAL_PARAMS)
    params["nz"] = 18
    seal = HolePatternSeal(**params)
    speed = Q_(5000, "RPM").to("rad/s").m
    row = seal.solver.solve_row(speed, [0.5 * speed, speed])
    assert row[0]["kxx"] == seal.solver.solve(speed, 0.5 * speed)["kxx"]
    assert row[1]["cxx"] == seal.cxx[0]
    assert row[1]["mxx"] == seal.mxx[0]


def _static_stiffness_error(inputs, speed, nz, gas=None):
    """Return ``abs(K / K_ref - 1)`` for the static direct stiffness.

    ``K`` is the stiffness at zero whirl frequency. ``K_ref = -pi R
    integral(d p0 / d H) dz`` comes from central differences of the base
    state in the clearance (steps of 1e-3 H, trapezoid over the stations):
    with the clearance perturbation ``H1 = -e cos(theta)`` and ``Fx =
    -integral(p cos(theta) R dtheta dz)``, a static displacement of a seal
    without circumferential coupling is a local change of clearance. When
    ``gas`` is given, unsolved seals receive it and the base state is solved
    directly.
    """
    clearance = inputs["radial_clearance"]
    profiles = []
    for sign in (1.0, -1.0):
        changed = dict(inputs, radial_clearance=clearance * (1.0 + sign * 1e-3), nz=nz)
        if gas is None:
            profiles.append(np.asarray(HolePatternSeal(**changed).p[0][2:-1], float))
        else:
            seal = _unsolved(**changed)
            seal.solver.gas = gas
            profiles.append(np.asarray(seal.solver.base_state(speed).p, float))
    slope = (profiles[0] - profiles[1]) / (2e-3 * clearance)
    dz = inputs["axial_length"] / nz
    integral = dz * (np.sum(slope) - 0.5 * (slope[0] + slope[-1]))
    reference = -np.pi * 0.5 * inputs["shaft_diameter"] * integral

    if gas is None:
        center = HolePatternSeal(**dict(inputs, nz=nz))
    else:
        center = _unsolved(**dict(inputs, nz=nz))
        center.solver.gas = gas
    stiffness = center.solver.solve(speed, 0.0)["kxx"]
    return abs(stiffness / reference - 1.0)


@pytest.mark.parametrize(
    "gas_name, surface_speed, preswirl",
    [
        ("air", 0.0, 0.0),
        ("air", 110.0, 0.2),
        ("van_der_waals", 0.0, 0.0),
        ("van_der_waals", 110.0, 0.3),
        ("methane", 110.0, 0.2),
    ],
)
def test_static_stiffness_is_the_clearance_derivative_of_the_base_state(
    heos, gas_name, surface_speed, preswirl
):
    """Static first-order solution against the clearance derivative of the base state.

    With a 1e5 m radius the circumferential coupling vanishes and a static
    displacement is a local change of clearance, so the static stiffness
    must equal ``-pi R integral(d p0 / d H) dz`` for any gas. This exercises
    the continuity, momentum and linearised energy equations, the friction
    derivatives, the entrance and exit relations and the state derivatives
    together. A frozen energy perturbation (#1475 M3) misses by 3.7 % to 18 %.
    """
    radius = 1e5
    gas = None
    if gas_name == "air":
        inputs = dict(FIG4)
    elif gas_name == "van_der_waals":
        inputs = dict(VDW_SEAL)
        gas = _van_der_waals()
    else:
        inputs = dict(METHANE, gas_model="real")
    speed = surface_speed / radius
    inputs.update(shaft_diameter=2 * radius, preswirl=preswirl, speed=[speed])

    error = {nz: _static_stiffness_error(inputs, speed, nz, gas) for nz in (40, 80)}
    # discretisation error of the second-order march (observed 7e-5 to 1.8e-4)
    assert error[80] < 1e-3, error
    # second-order convergence: the ratio is 0.25 (observed 0.254-0.255)
    assert error[80] < 0.4 * error[40], error


def test_smooth_seal_damping_is_positive_and_static_stiffness_ignores_cell_depth():
    """A smooth annular gas seal has positive direct damping (Nelson 1985).

    Storage-only cells store no mass under a static displacement (Kleynhans
    and Childs 1996), so the static stiffness does not depend on the cell
    depth. The second part is the reproduction of petrobras/ross#1475, which
    reported ``cxx[0][0] = -1.12e5`` N s/m for a smooth seal.
    """
    speed = Q_([10200, 15200, 20200], "RPM")
    smooth = {
        **CHILDS_WADE,
        **AIR,
        "radial_clearance": CLEARANCE_020,
        "outlet_pressure": 35e5,
        "inlet_temperature": 290.0,
        "preswirl": 0.0,
        "entrance_loss_coefficient": 0.0,
        "exit_loss_coefficient": 0.0,
        "speed": speed,
    }
    for ratio in (0.5, 1.0):
        seal = HolePatternSeal(**dict(smooth, cell_depth=0.0), excitation_ratio=ratio)
        assert np.all(np.array(seal.cxx) > 0), (ratio, seal.cxx)

    static = {}
    for depth in (0.0, 1e-3, 2.257e-3, 3.3e-3):
        seal = HolePatternSeal(**dict(smooth, cell_depth=depth, speed=speed[1:2]))
        static[depth] = [seal.solver.solve(s, 0.0)["kxx"] for s in speed.to("rad/s").m]
    for depth, values in static.items():
        # rounding: observed 12 digits; the zero-frequency step adds about 3e-17
        assert_allclose(values, static[0.0], rtol=1e-10, err_msg=str(depth))

    issue = HolePatternSeal(
        n=0,
        shaft_diameter=0.1147,
        radial_clearance=0.0002,
        axial_length=0.0857,
        relative_roughness=1e-4,
        cell_length=0.0032,
        cell_width=0.0032,
        cell_depth=0.0,
        inlet_pressure=70e5,
        outlet_pressure=35e5,
        inlet_temperature=300.0,
        preswirl=0.0,
        entrance_loss_coefficient=0.1,
        exit_loss_coefficient=0.5,
        nz=80,
        speed=speed,
        frequency=speed,
        **AIR,
    )
    cxx = np.array(issue.cxx)
    assert np.all(np.diag(cxx) > 0), cxx


@pytest.mark.parametrize("exit_loss", [0.0, 0.5])
def test_choked_flow_does_not_depend_on_the_outlet_pressure(exit_loss):
    """Beyond choke the upstream flow cannot see the outlet pressure."""
    from ross.seals.holepattern_seal import CHOKED_MZ2_LIMIT

    inputs = {
        **CHILDS_WADE,
        **AIR,
        "radial_clearance": CLEARANCE_020,
        "inlet_temperature": 290.0,
        "preswirl": 0.3,
        "exit_loss_coefficient": exit_loss,
        "speed": Q_([15000], "RPM"),
    }
    seals = {
        ratio: HolePatternSeal(**inputs, outlet_pressure=ratio * 70e5)
        for ratio in (0.30, 0.10)
    }
    high, low = seals[0.30], seals[0.10]
    # the choke limit is found to 4e-16 p_in (bisection resolution); observed 10+ digits
    assert_allclose(low.seal_leakage[0], high.seal_leakage[0], rtol=1e-9)
    assert_allclose(low.p[0][-1], high.p[0][-1], rtol=1e-9)
    assert_allclose(low.kxx[0], high.kxx[0], rtol=1e-9)
    assert_allclose(low.cxx[0], high.cxx[0], rtol=1e-9)

    for ratio, seal in seals.items():
        assert seal.p[0][-1] > ratio * 70e5
        assert seal.choked == [True]
        mach2 = seal.solver.base_state(_rad(15000)).mach2[-1]
        if exit_loss == 0.0:
            # choked in the land: the accepted trial sits within 4e-16 p_in of
            # the limit, and (u/a)^2 is recomputed, so rounding can exceed it
            assert 0.95 < mach2 <= CHOKED_MZ2_LIMIT * (1 + 1e-9)
        else:
            # choked in the exit relation, below the land limit (observed 0.43)
            assert mach2 < 0.6


def test_reversed_rotation_mirrors_the_coefficients():
    """Reversed rotation and swirl: same direct terms, opposite cross-coupled terms."""
    seal = HolePatternSeal(**FIG4)
    speed = _rad(20200)
    forward = seal.solver.solve(speed, 0.7 * speed)
    backward = seal.solver.solve(-speed, 0.7 * speed)
    # mirror image of the same flow: rounding only (observed exact)
    for name in ("kxx", "cxx", "seal_leakage"):
        assert_allclose(backward[name], forward[name], rtol=1e-12, err_msg=name)
    for name in ("kxy", "cxy"):
        assert_allclose(backward[name], -forward[name], rtol=1e-12, err_msg=name)


@pytest.mark.parametrize(
    "inputs, damping, table_5_leakage",
    [
        pytest.param(
            FIG4,
            [
                (28.6, 32.80),
                (91.8, 27.31),
                (145.0, 21.45),
                (210.8, 16.84),
                (285.4, 12.17),
            ],
            0.409,
            id="fig4",
        ),
        pytest.param(
            FIG6, [(144.1, 18.64), (210.9, 11.36), (274.6, 6.94)], 0.175, id="fig6"
        ),
    ],
)
def test_childs_wade_frequency_dependence(inputs, damping, table_5_leakage):
    """Childs and Wade (2004), Figs. 4 and 6 and Table 5.

    The paper prints no coefficient tables. The damping values (Hz, kN s/m)
    are the means of groups of markers read from the figures (300 dpi page
    images, axes calibrated on the ticks). The model treats the stator as a
    smooth wall at a quarter of the Moody friction (#1475 M4), so the damping
    and the leakage are high; the bounds below hold that known offset, and
    the frequency trends are checked strictly.
    """
    read_hz = np.array([f for f, _ in damping])
    read_damping = 1e3 * np.array([c for _, c in damping])
    grid_hz = np.arange(20, 301, 20)
    frequency_hz = np.unique(np.concatenate([grid_hz, read_hz]))
    seal = HolePatternSeal(**inputs, frequency=2 * np.pi * frequency_hz)
    kxx, kxy, cxx = (np.array(getattr(seal, name))[0] for name in ("kxx", "kxy", "cxx"))

    ratio = cxx[np.searchsorted(frequency_hz, read_hz)] / read_damping
    # observed 1.11-1.70; the factor of 2 holds the stator friction (M4)
    assert np.all((0.5 < ratio) & (ratio < 2.0)), ratio

    on_grid = np.isin(frequency_hz, grid_hz)
    assert np.all(np.diff(kxx[on_grid]) > 0), kxx[on_grid]
    assert np.all(np.diff(cxx[on_grid]) < 0), cxx[on_grid]
    assert np.all(np.diff(kxy[on_grid]) < 0), kxy[on_grid]

    # effective damping crosses zero inside the band (paper: 85 to 95 Hz)
    effective = cxx - kxy / (2 * np.pi * frequency_hz)
    assert effective[0] < 0 < effective[-1], effective

    # observed 1.86 (Fig. 4) and 1.76 (Fig. 6); high because of M4
    assert 1.5 < seal.seal_leakage[0] / table_5_leakage < 2.1, seal.seal_leakage


def _simpson(values, dz):
    weights = np.ones(values.size)
    weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
    return dz / 3.0 * np.sum(weights * values)


def _conservation_residuals(seal, speed):
    """Integral balances of the base state, evaluated with Simpson's rule."""
    base = seal.solver.base_state(speed)
    clearance = seal.radial_clearance
    surface_speed = 0.5 * seal.shaft_diameter * speed
    points = [
        base.gas.point(float(p), float(T)) for p, T in zip(base.p, base.T, strict=True)
    ]
    shear = np.array(
        [
            seal.solver.shear(
                clearance, pt.rho, pt.mu, float(u), float(w), surface_speed
            )
            for pt, u, w in zip(points, base.u, base.w, strict=True)
        ]
    )
    rho = np.array([pt.rho for pt in points])
    h = np.array([pt.h for pt in points])
    g = base.mass_flux
    dz = seal.axial_length / seal.nz
    momentum = (base.p[-1] + g * base.u[-1]) - (base.p[0] + g * base.u[0])
    momentum += _simpson(shear[:, 0], dz) / clearance
    total = h + 0.5 * (base.u**2 + base.w**2)
    energy = total[-1] - total[0]
    energy += surface_speed * _simpson(shear[:, 2], dz) / (clearance * g)
    swirl = base.w[-1] - base.w[0] + _simpson(shear[:, 1], dz) / (clearance * g)
    return {
        "mass": np.max(np.abs(rho * base.u / g - 1.0)),
        "momentum": momentum / (base.p[0] - base.p[-1]),
        "total_enthalpy": energy / (points[0].cp * base.T[0]),
        "swirl": swirl / surface_speed,
    }


@pytest.mark.parametrize("case", ["common", "fig4", "methane"])
def test_base_state_conserves_momentum_swirl_and_total_enthalpy(heos, methane, case):
    """The base state satisfies the integral conservation laws of the land."""
    if case == "common":
        seal, speed = HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS), _rad(5000)
    elif case == "fig4":
        seal, speed = HolePatternSeal(**FIG4), _rad(20200)
    else:
        seal, speed = methane["real"], _rad(15200)
    assert seal.nz == 80
    residuals = _conservation_residuals(seal, speed)
    # u = G / rho by construction
    assert residuals["mass"] < 1e-12, residuals
    # Simpson quadrature of the check plus the RK4 error (observed 1e-11 to 1e-9)
    for name in ("momentum", "total_enthalpy", "swirl"):
        assert abs(residuals[name]) < 1e-7, (name, residuals)


def test_coefficients_converge_at_second_order_in_nz():
    """Self-convergence of the second-order perturbation march."""
    speed = _rad(20200)
    solved = {}
    for nz in (40, 80, 160, 320):
        row = _unsolved(**FIG4, nz=nz).solver.solve_row(speed, [0.5 * speed, speed])
        solved[nz] = row
    finest = solved[320]

    def error(nz, names):
        return max(
            _relative([r[name] for r in solved[nz]], [r[name] for r in finest])
            for name in names
        )

    e40, e80, e160 = (error(nz, COEFFICIENTS) for nz in (40, 80, 160))
    # second order: the ratios are about 0.24 and 0.2 against nz = 320
    assert e80 < 0.35 * e40, (e40, e80)
    assert e160 < 0.35 * e80, (e80, e160)
    # observed 6.6e-4
    assert e80 < 2e-3, e80
    # fourth-order RK4 base state
    assert error(80, ("seal_leakage",)) < 1e-6


def test_holepattern_gas_model_default_is_ideal(holepattern_manual):
    """The default closure is the perfect gas."""
    from ross.seals.gas_model import SutherlandGas

    assert holepattern_manual.gas_model == "ideal"
    assert holepattern_manual._real_gas is False
    assert isinstance(holepattern_manual.gas, SutherlandGas)


def test_holepattern_real_requires_gas_composition():
    """gas_model='real' without gas_composition raises a clear error."""
    with pytest.raises(ValueError, match="requires gas_composition"):
        HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS, gas_model="real")


def test_holepattern_invalid_gas_model():
    """An unknown gas_model raises a clear error."""
    with pytest.raises(ValueError, match="Invalid gas_model"):
        HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS, gas_model="perfect")


def test_holepattern_needs_two_axial_intervals():
    with pytest.raises(ValueError, match="nz must be at least 2"):
        HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS, nz=1)


def _scaled_methane(scale, gas_model):
    return HolePatternSeal(
        **dict(
            METHANE,
            inlet_pressure=scale * METHANE["inlet_pressure"],
            outlet_pressure=scale * METHANE["outlet_pressure"],
        ),
        gas_model=gas_model,
    )


def test_real_gas_tends_to_the_ideal_gas_at_low_pressure(heos):
    """The equation of state tends to a perfect gas as the pressure goes to zero.

    What remains at low pressure is the temperature dependence of cp, which
    the constant gamma of the ideal closure ignores.
    """
    deviation = {}
    for scale in (0.1, 0.01):
        real, ideal = (_scaled_methane(scale, model) for model in ("real", "ideal"))
        deviation[scale] = max(
            _relative(getattr(real, name), getattr(ideal, name))
            for name in ("seal_leakage", "kxx", "kxy", "cxx")
        )
    # observed 0.072 and 0.0034: linear in pressure
    assert deviation[0.01] < 1e-2, deviation
    assert deviation[0.01] < 0.2 * deviation[0.1], deviation


def test_real_gas_dense_methane_leaks_more_than_ideal(methane):
    """Z = 0.89 at the inlet: the denser gas carries more mass (observed +15 %)."""
    real, ideal = methane["real"], methane["ideal"]
    assert real.seal_leakage[0] > ideal.seal_leakage[0]
    for name in COEFFICIENTS:
        assert np.all(np.isfinite(getattr(real, name))), name


def test_table_reproduces_its_closed_form_gas():
    """Closed-form gases against spline tables of the same gases.

    The table derives the state derivatives from its splines; the closed
    forms write them down. (a) is a perfect gas, (b) the van der Waals gas,
    for which ``h_p``, ``mu_p`` and the pressure dependence of ``cp`` do not
    vanish.
    """
    from ross.seals.gas_model import SutherlandGas, TabulatedGas

    air = SutherlandGas(8314.0 / 28.97, 1.4, 1.458e-6, 110.4)
    pressure = np.linspace(2e5, 7.5e5, 41)
    temperature = np.linspace(240.0, 340.0, 41)
    air_table = TabulatedGas.from_function(_state_functions(air), pressure, temperature)

    names = ("rho", "h", "cp", "rho_p", "rho_T", "mu", "mu_T", "sound_speed")
    centres = [
        (p, T)
        for p in 0.5 * (pressure[1:] + pressure[:-1])
        for T in 0.5 * (temperature[1:] + temperature[:-1])
    ]
    found = [air_table.point(p, T) for p, T in centres]
    expected = [air.point(p, T) for p, T in centres]
    for name in names:
        # interpolation error of the bicubic table at the cell centres
        assert_allclose(
            [getattr(pt, name) for pt in found],
            [getattr(pt, name) for pt in expected],
            rtol=1e-5,
            err_msg=name,
        )
    # zero for a perfect gas: 1e-12 of the scales 1/rho and mu/p is rounding
    h_p_scaled = [pt.h_p * ref.rho for pt, ref in zip(found, expected, strict=True)]
    mu_p_scaled = [
        pt.mu_p * p / ref.mu
        for pt, ref, (p, _) in zip(found, expected, centres, strict=True)
    ]
    assert np.max(np.abs(h_p_scaled)) < 1e-12
    assert np.max(np.abs(mu_p_scaled)) < 1e-12

    air_seal = {
        "n": 0,
        "shaft_diameter": 0.145,
        "radial_clearance": 0.3e-3,
        "axial_length": 46.99e-3,
        "relative_roughness": 1e-4,
        "cell_length": 3e-3,
        "cell_width": 3e-3,
        "cell_depth": 2.5e-3,
        "inlet_pressure": 7e5,
        "outlet_pressure": 4e5,
        "inlet_temperature": 322.0,
        "preswirl": 0.5,
        "speed": Q_([8000], "RPM"),
        **AIR,
    }
    vdw = _van_der_waals()
    vdw_table = TabulatedGas.from_function(
        vdw.functions, np.linspace(13e6, 30.1e6, 61), np.linspace(300.0, 440.0, 61)
    )
    # interpolation error of each table (observed 4e-6 for (a), 2.4e-7 for (b))
    cases = [
        (air_seal, air, air_table, _rad(8000), 2e-5),
        (VDW_SEAL, vdw, vdw_table, _rad(15200), 1e-5),
    ]
    for inputs, closed, table, speed, rtol in cases:
        solved = {}
        for name, gas in (("closed", closed), ("table", table)):
            seal = _unsolved(**inputs)
            seal.solver.gas = gas
            solved[name] = seal.solver.solve_row(speed, [0.5 * speed, speed])
        for row, reference in zip(solved["table"], solved["closed"], strict=True):
            assert_allclose(row["seal_leakage"], reference["seal_leakage"], rtol=1e-6)
            for name in COEFFICIENTS:
                assert_allclose(row[name], reference[name], rtol=rtol, err_msg=name)


def test_real_gas_table_matches_the_equation_of_state(heos):
    """Table values and derivatives against the equation of state of ccp."""
    from ross.seals.gas_model import TabulatedGas

    fluid = {"methane": 1.0}
    table = TabulatedGas.from_composition(fluid, 20e6, 300.0, 10e6)
    assert table.floor_raised is False

    def state(p, T):
        return ccp.State(p=p, T=T, fluid=fluid, EOS="HEOS", phase=False)

    for p, T in ((18e6, 300.0), (14e6, 290.0), (10.5e6, 270.0)):
        assert table.lower_temperature(p) < T < table.upper_temperature(p)
        point = table.point(p, T)
        reference = state(p, T)
        # interpolation error of the 41 x 41 table
        assert_allclose(point.rho, reference.rho().m, rtol=1e-6)
        assert_allclose(point.mu, reference.viscosity().m, rtol=1e-6)
        assert_allclose(point.cp, reference.cp().m, rtol=1e-4)
        assert_allclose(point.sound_speed, reference.speed_sound().m, rtol=1e-4)

        dp, dT = 1e-4 * p, 1e-4 * T
        up, down = state(p + dp, T), state(p - dp, T)
        warm, cold = state(p, T + dT), state(p, T - dT)
        derivatives = {
            "rho_p": (up.rho().m - down.rho().m) / (2 * dp),
            "rho_T": (warm.rho().m - cold.rho().m) / (2 * dT),
            "h_p": (up.h().m - down.h().m) / (2 * dp),
            "mu_p": (up.viscosity().m - down.viscosity().m) / (2 * dp),
            "mu_T": (warm.viscosity().m - cold.viscosity().m) / (2 * dT),
        }
        for name, value in derivatives.items():
            # central differences with 1e-4 relative steps
            assert_allclose(getattr(point, name), value, rtol=1e-4, err_msg=name)

        # (d h / d p)_T = (1 + T rho_T / rho) / rho (observed 1e-6)
        assert_allclose(
            point.h_p, (1 + T * point.rho_T / point.rho) / point.rho, rtol=1e-4
        )
        # not a perfect gas
        assert abs(point.h_p * point.rho) > 1e-2
        assert point.mu_p > 0


def test_gas_range_contract():
    """States outside a gas object's domain raise a GasRangeError with its side."""
    from ross.seals.gas_model import GasRangeError, SutherlandGas, TabulatedGas

    air = SutherlandGas(287.0, 1.4, 1.458e-6, 110.4)
    pressure = np.linspace(1e5, 5e5, 6)
    temperature = np.linspace(250.0, 350.0, 6)
    table = TabulatedGas.from_function(_state_functions(air), pressure, temperature)

    outside = [
        (0.5e5, 300.0, "below"),
        (6e5, 300.0, "above"),
        (3e5, 240.0, "below"),
        (3e5, 360.0, "above"),
    ]
    for p, T, side in outside:
        with pytest.raises(GasRangeError, match="outside the property table") as info:
            table.point(p, T)
        error = info.value
        assert (error.pressure, error.temperature, error.side) == (p, T, side)
        copy = pickle.loads(pickle.dumps(error))
        assert type(copy) is GasRangeError
        assert str(copy) == str(error)
        assert (copy.pressure, copy.temperature, copy.side) == (p, T, side)

    with pytest.raises(GasRangeError) as info:
        table.temperature_ps(3e5, air.entropy(3e5, 200.0), 300.0)
    assert info.value.side == "below"

    with pytest.raises(GasRangeError) as info:
        SutherlandGas(287.0, 1.4, 1.458e-6, -150.0).point(1e5, 140.0)
    assert info.value.side == "below"

    with pytest.raises(ValueError):
        TabulatedGas.from_function(_state_functions(air), pressure[:3], temperature[:3])

    values = np.array(
        [[_state_functions(air)(p, T) for T in temperature] for p in pressure]
    )
    rho, h, s, mu = (values[:, :, k] for k in range(4))
    lower = np.full(pressure.size, temperature[0])
    offset = temperature - temperature[0]
    TabulatedGas(pressure, offset, lower, rho, h, s, mu)
    with pytest.raises(ValueError):
        TabulatedGas(pressure, offset + 1.0, lower, rho, h, s, mu)


def test_sutherland_gas_closed_forms():
    """Perfect gas with the Sutherland viscosity, in closed form."""
    from ross.seals.gas_model import SutherlandGas

    R, gamma, b, S = 287.0, 1.4, 1.458e-6, 110.4
    gas = SutherlandGas(R, gamma, b, S)
    cp = gamma * R / (gamma - 1)
    for p, T in ((1e5, 250.0), (5e5, 300.0), (2e6, 400.0)):
        point = gas.point(p, T)
        rho = p / (R * T)
        expected = {
            "rho": rho,
            "rho_p": 1 / (R * T),
            "rho_T": -rho / T,
            "h": cp * T,
            "cp": cp,
            "mu": b * T**1.5 / (S + T),
            "sound_speed": np.sqrt(gamma * R * T),
            "kappa_s": 1 / (gamma * p),
            "p_rho": R * T,
            "p_t": rho * R,
            "h_t": cp,
        }
        for name, value in expected.items():
            assert_allclose(getattr(point, name), value, rtol=1e-12, err_msg=name)
        assert point.h_p == 0.0
        assert point.mu_p == 0.0
        assert point.h_rho == 0.0

        dT = 1e-4 * T
        derivative = (gas.viscosity(T + dT) - gas.viscosity(T - dT)) / (2 * dT)
        # central difference, truncation error about (dT / T)^2
        assert_allclose(point.mu_T, derivative, rtol=1e-7)

        assert_allclose(gas.temperature_ph(p, cp * T), T, rtol=1e-12)
        assert_allclose(gas.temperature_ps(p, gas.entropy(p, T)), T, rtol=1e-12)


def test_flow_that_leaves_the_gas_range_is_retried_or_reported(holepattern_manual):
    """A gas that refuses a state is widened and retried, or the point fails alone."""
    seal = holepattern_manual
    speed = _rad(5000)
    expected = seal.solver.solve(speed)
    Narrow, NeverWide = _narrow_gases(seal.inlet_temperature - 1.0)
    constants = (seal.gas.gas_constant, seal.gas.gamma, seal.gas.sutherland_b)
    constants += (seal.gas.sutherland_s,)

    seal.solver.gas = Narrow(*constants)
    found = seal.solver.solve(speed)
    for name in COEFFICIENTS + ("kyy", "kyx", "cyy", "cyx", "mxx", "seal_leakage"):
        assert found[name] == expected[name], name
    assert np.array_equal(found["pressure"], expected["pressure"])

    seal.solver.gas = NeverWide(*constants)
    with pytest.warns(UserWarning, match="outside"):
        failed = seal.solver.solve(speed)
    for name in COEFFICIENTS + ("seal_leakage",):
        assert np.isnan(failed[name]), name


@pytest.mark.parametrize("outlet_pressure", [3e6, 4.5e6])
def test_real_gas_refuses_a_liquid_inlet(heos, outlet_pressure):
    """CO2 at 6 MPa and 280 K is a liquid (T_sat = 295 K, p below p_crit 7.38 MPa)."""
    with pytest.raises(ValueError, match="single-phase gas"):
        HolePatternSeal(
            **dict(CHILDS_WADE, inlet_pressure=6e6),
            radial_clearance=CLEARANCE_020,
            gas_model="real",
            gas_composition={"co2": 1.0},
            inlet_temperature=280.0,
            outlet_pressure=outlet_pressure,
            speed=Q_([15200], "RPM"),
        )


AIR_COMPOSITION = {"nitrogen": 0.7812, "oxygen": 0.2096, "argon": 0.0092}


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_real_gas_accepts_refprop_spurious_split():
    """REFPROP splits air (N2/O2/Ar) at 5 MPa, 290 K, far above its critical point.

    The free flash reports a two-phase state with a quality of 3e-7 and
    saturated densities 2.1e-3 apart, which is ccp's spurious-split criterion
    (within 1e-2). HEOS reports single phase there. The table must accept the
    state as a gas and hold the imposed root, which agrees with HEOS; the split
    density is 4.6e-4 off.
    """
    from CoolProp import CoolProp as coolprop

    from ross.seals.gas_model import TabulatedGas

    free = ccp.State(p=5e6, T=290.0, fluid=AIR_COMPOSITION, EOS="REFPROP", phase=False)
    assert 0.0 <= free.Q() <= 1.0
    rho_l = free.saturated_liquid_keyed_output(coolprop.iDmass)
    rho_v = free.saturated_vapor_keyed_output(coolprop.iDmass)
    # ccp 0.5.1 State._spurious_two_phase: saturated densities within 1e-2
    assert abs(rho_l - rho_v) <= 1e-2 * max(rho_l, rho_v)

    table = TabulatedGas.from_composition(
        AIR_COMPOSITION, 5e6, 290.0, 3e6, eos="REFPROP"
    )
    assert table.floor_raised is False
    heos = ccp.State(p=5e6, T=290.0, fluid=AIR_COMPOSITION, EOS="HEOS", phase=False)
    # imposed root against HEOS 1.4e-6 (observed); the split density is
    # 4.6e-4 off, so 1e-4 tells the stored root from the split
    rho = table.point(5e6, 290.0).rho
    assert_allclose(rho, heos.rho().m, rtol=1e-4)
    assert abs(rho / free.rho().m - 1) > 2e-4


def test_holepattern_save_load_round_trip(tmp_path, heos, methane):
    """Saved seals load with the same coefficients, without solving again."""
    speed = Q_([5000, 8000], "RPM").to("rad/s").m
    grid = HolePatternSeal(
        **dict(COMMON_PARAMS, speed=speed, nz=18),
        **AIR,
        frequency=[0.5 * speed[0], speed[1]],
    )
    file = tmp_path / "holepattern_grid.toml"
    grid.save(file)
    loaded = HolePatternSeal.load(file)
    assert isinstance(loaded, HolePatternSeal)
    assert loaded.gas_model == "ideal"
    assert_allclose(loaded.speed, grid.speed)
    assert_allclose(loaded.frequency, grid.frequency)
    for name in ("kxx", "cxx", "seal_leakage"):
        assert_allclose(np.array(getattr(loaded, name)), np.array(getattr(grid, name)))
    assert_allclose(
        loaded.K(frequency=500.0, speed=700.0), grid.K(frequency=500.0, speed=700.0)
    )

    real = methane["real"]
    file = tmp_path / "holepattern_real.toml"
    real.save(file)
    loaded = HolePatternSeal.load(file)
    assert loaded.gas_model == "real"
    assert loaded.gas is None
    for name in COEFFICIENTS + ("seal_leakage",):
        assert_allclose(getattr(loaded, name), getattr(real, name), err_msg=name)


def test_solver_pickles(holepattern_manual, methane):
    """The solver crosses a process boundary, as the pool requires."""
    for seal, speed in (
        (holepattern_manual, _rad(5000)),
        (methane["real"], _rad(15200)),
    ):
        copy = pickle.loads(pickle.dumps(seal.solver))
        assert copy.solve(speed)["kxx"] == seal.solver.solve(speed)["kxx"]


def test_zero_frequency_damping_is_the_limit(holepattern_manual):
    """C(Omega) is even and continuous: at Omega = 0 it is the limit, not zero."""
    speed = _rad(5000)
    solver = holepattern_manual.solver
    at_zero, near_zero = solver.solve(speed, 0.0), solver.solve(speed, 1e-4 * speed)
    # C(Omega) - C(0) is of order (1e-4)^2
    for name in ("cxx", "cxy"):
        assert_allclose(at_zero[name], near_zero[name], rtol=1e-6, err_msg=name)


def test_real_gas_table_floor_keeps_the_inlet_isentrope_single_phase(heos):
    """The table floor sits where the inlet isentrope, 5 K colder, meets saturation.

    The reference curves come from CoolProp's own (P, S) and (P, Q) flashes,
    not from the imposed-root Newton of the table code. A choked seal does
    not see its outlet pressure, so the seal with a raised floor and the one
    without give the same result to the interpolation error of their tables.
    """
    from CoolProp.CoolProp import PropsSI

    from ross.seals.gas_model import TabulatedGas

    table = TabulatedGas.from_composition({"methane": 1.0}, 20e6, 300.0, 1e6)
    assert table.floor_raised is True
    assert table.p_min > 0.9e6
    s_in = PropsSI("S", "P", 20e6, "T", 300.0, "HEOS::Methane")
    edge = PropsSI("T", "P", table.p_min, "S", s_in, "HEOS::Methane") - 5.0
    saturation = PropsSI("T", "P", table.p_min, "Q", 1, "HEOS::Methane")
    # bisection resolution 4.7 kPa times about 7 K/MPa is 0.03 K (observed 0.013 K)
    assert 0.0 < edge - saturation < 0.1, (table.p_min, edge, saturation)

    seals = {
        outlet: HolePatternSeal(
            **dict(METHANE, outlet_pressure=outlet), gas_model="real"
        )
        for outlet in (1e6, 4e6)
    }
    low, high = seals[1e6], seals[4e6]
    assert low.choked == [True] and high.choked == [True]
    assert low.gas.floor_raised is True
    assert high.gas.floor_raised is False
    # interpolation error of two different tables (prototype: 3.9e-9 and 6e-5)
    assert_allclose(low.seal_leakage, high.seal_leakage, rtol=1e-6)
    for name in ("kxx", "kxy", "cxx"):
        assert_allclose(
            getattr(low, name), getattr(high, name), rtol=5e-4, err_msg=name
        )


def test_real_gas_seal_that_expands_into_the_two_phase_region_is_refused(heos):
    """The CO2 isentrope from 10 MPa, 330 K passes the critical point on the liquid side.

    Below the critical pressure the expansion meets the saturation curve, so
    the floor is the critical pressure, and a seal whose flow needs lower
    pressures fails with a message.
    """
    from CoolProp.CoolProp import PropsSI

    from ross.seals.gas_model import TabulatedGas

    table = TabulatedGas.from_composition({"co2": 1.0}, 10e6, 330.0, 2e6)
    p_crit = PropsSI("Pcrit", "HEOS::CO2")
    assert table.floor_raised is True
    # bisection resolution 2.0 kPa, 2.7e-4 of p_crit
    assert p_crit <= table.p_min <= p_crit * (1 + 1e-3), (table.p_min, p_crit)

    with pytest.warns(UserWarning, match="single-phase"):
        seal = HolePatternSeal(
            **dict(CHILDS_WADE, inlet_pressure=10e6),
            radial_clearance=CLEARANCE_020,
            gas_model="real",
            gas_composition={"co2": 1.0},
            outlet_pressure=2e6,
            inlet_temperature=330.0,
            preswirl=0.2,
            speed=Q_([15200], "RPM"),
        )
    for name in COEFFICIENTS + ("seal_leakage",):
        assert np.all(np.isnan(getattr(seal, name))), name
    assert seal.choked == [None]
