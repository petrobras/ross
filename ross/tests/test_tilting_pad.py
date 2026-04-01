import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.tilting_pad import TiltingPad
from ross.units import Q_

journal_diameter  = 101.6e-3                               # m
radial_clearance  = 74.9e-6                                # m
pad_thickness     = 12.7e-3                                # m
n_pads            = 5
frequency         = Q_([3000], "RPM")
pivot_angles      = Q_([18, 90, 162, 234, 306], "deg")
pad_arc           = Q_([60] * n_pads, "deg")
pad_axial_length  = Q_([50.8e-3] * n_pads, "m")
pre_load          = [0.5] * n_pads
offset            = [0.5] * n_pads
lubricant         = "ISOVG32"
oil_supply_temp   = Q_(40, "degC")
load              = [8.8405e02, -2.6704e03]                # N, [Fx, Fy]
eccentricity      = 0.35
attitude_angle    = Q_(287.5, "deg")
thermal_type      = "full"

base_kwargs = dict(
    n=1,
    frequency=frequency,
    journal_diameter=journal_diameter,
    radial_clearance=radial_clearance,
    pad_thickness=pad_thickness,
    pivot_angle=pivot_angles,
    pad_arc=pad_arc,
    pad_axial_length=pad_axial_length,
    pre_load=pre_load,
    offset=offset,
    lubricant=lubricant,
    oil_supply_temperature=oil_supply_temp,
    nx=30,
    nz=30,
    eccentricity=eccentricity,
    attitude_angle=attitude_angle,
    load=load,
    thermal_type=thermal_type,
)

# Converged pad tilt angles [rad]
psi_pad = {
    "match":     [1.07426e-03, 7.29330e-04, 2.95380e-04, 3.49720e-04, 8.17350e-04],
    "determine": [1.07420e-03, 7.20800e-04, 2.93690e-04, 3.49690e-04, 8.16040e-04],
}

# Converged eccentricity per equilibrium method
eccentricity_result = {
    "match":     0.35,
    "determine": 0.3722,
}

# Converged attitude angle [rad] per equilibrium method
attitude_angle_result = {
    "match":     5.017821599483698,
    "determine": 5.064506361232241,
}

# 2x2 stiffness [N/m] and damping [N·s/m] coefficients
coefficients = {
    "match": dict(
        kxx= 8.9499e07, kxy=-3.3817e07,
        kyx=-3.3817e07, kyy= 1.1652e08,
        cxx= 2.8202e05, cxy=-2.8284e04,
        cyx=-2.8284e04, cyy= 3.3170e05,
    ),
    "determine": dict(
        kxx= 9.5254e07, kxy=-4.7141e07,
        kyx=-4.7141e07, kyy= 1.2676e08,
        cxx= 2.9184e05, cxy=-4.0367e04,
        cyx=-4.0367e04, cyy= 3.5168e05,
    ),
}

# Hydrodynamic forces per pad [N]
force_x = {
    "match":     np.array([-9.22162496e02,  3.11147456e-01,  5.52500475e02,
                             1.04579373e03, -1.57342191e03]),
    "determine": np.array([-8.613055e02,  2.827307e-01,  5.238299e02,
                             1.124814e03, -1.953835e03]),
}
force_y = {
    "match":     np.array([-3.007243e02, -4.2661925e02, -1.7933787e02,
                             1.4404706e03,  2.1619113e03]),
    "determine": np.array([-2.80841169e02, -3.90888628e02, -1.70035685e02,
                             1.549360826e03,  2.684245728e03]),
}


@pytest.fixture(scope="module")
def bearing_match():
    """Return a TiltingPad with prescribed eccentricity (match_eccentricity).

    Journal position is fixed; only pad tilt angles are solved.
    Shared across all tests in the module (scope='module').
    """
    return TiltingPad(**base_kwargs, equilibrium_type="match_eccentricity")


@pytest.fixture(scope="module")
def bearing_determine():
    """Return a TiltingPad solved via thermo-hydrodynamic Newton iteration.

    Both journal equilibrium position and pad tilt angles are determined
    simultaneously. Shared across all tests in the module (scope='module').
    """
    return TiltingPad(
        **base_kwargs,
        equilibrium_type="determine_eccentricity",
        model_type="thermo_hydro_dynamic",
        initial_pads_angles=[1.0742e-03, 7.2080e-04, 2.9369e-04,
                              3.4969e-04, 8.1604e-04],
        solver_options={"xtol": 1e-2, "ftol": 1e-2, "maxiter": 1000},
    )


class TestTiltingPadGeometry:
    """Verify physical geometry and operating point after construction."""

    @pytest.mark.parametrize("bearing", ["bearing_match", "bearing_determine"])
    def test_journal_radius(self, bearing, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.journal_radius, journal_diameter / 2)

    @pytest.mark.parametrize("bearing", ["bearing_match", "bearing_determine"])
    def test_radial_clearance(self, bearing, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.radial_clearance, radial_clearance)

    @pytest.mark.parametrize("bearing", ["bearing_match", "bearing_determine"])
    def test_n_pad(self, bearing, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.n_pad, n_pads)

    @pytest.mark.parametrize("bearing", ["bearing_match", "bearing_determine"])
    def test_frequency(self, bearing, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.frequency, Q_(3000, "RPM").to("rad/s").m, rtol=1e-6)

    @pytest.mark.parametrize("bearing", ["bearing_match", "bearing_determine"])
    def test_reference_temperature(self, bearing, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.reference_temperature, 40)


class TestTiltingPadEquilibrium:
    """Validate converged journal position and pad tilt angles for both strategies."""

    @pytest.mark.parametrize("bearing,case", [
        ("bearing_match",     "match"),
        ("bearing_determine", "determine"),
    ])
    def test_eccentricity(self, bearing, case, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.eccentricity, eccentricity_result[case], rtol=1e-2)

    @pytest.mark.parametrize("bearing,case", [
        ("bearing_match",     "match"),
        ("bearing_determine", "determine"),
    ])
    def test_attitude_angle(self, bearing, case, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.attitude_angle, attitude_angle_result[case], rtol=1e-2)

    @pytest.mark.parametrize("bearing,case,pad_index", [
        pytest.param("bearing_match",     "match",     i, id=f"match-pad{i}")
        for i in range(n_pads)
    ] + [
        pytest.param("bearing_determine", "determine", i, id=f"determine-pad{i}")
        for i in range(n_pads)
    ])
    def test_pad_tilt_angles(self, bearing, case, pad_index, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(
            b.psi_pad[pad_index],
            psi_pad[case][pad_index],
            rtol=0.1,
        )


class TestTiltingPadDynamicCoefficients:
    """Validate full 2x2 stiffness (K) and damping (C) matrices."""

    @pytest.mark.parametrize("bearing,case,rtol", [
        ("bearing_match",     "match",     1e-3),
        ("bearing_determine", "determine", 1e-2),
    ])
    @pytest.mark.parametrize("coeff", ["kxx", "kxy", "kyx", "kyy",
                                        "cxx", "cxy", "cyx", "cyy"])
    def test_coefficient(self, bearing, case, rtol, coeff, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(
            getattr(b, coeff),
            coefficients[case][coeff],
            rtol=rtol,
        )


class TestTiltingPadHydrodynamicForces:
    """Validate hydrodynamic forces per pad."""

    @pytest.mark.parametrize("bearing,case", [
        ("bearing_match",     "match"),
        ("bearing_determine", "determine"),
    ])
    def test_force_x(self, bearing, case, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.force_x_dim, force_x[case], rtol=1e-2)

    @pytest.mark.parametrize("bearing,case", [
        ("bearing_match",     "match"),
        ("bearing_determine", "determine"),
    ])
    def test_force_y(self, bearing, case, request):
        b = request.getfixturevalue(bearing)
        assert_allclose(b.force_y_dim, force_y[case], rtol=1e-2)