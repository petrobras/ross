import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_almost_equal

from ross.seals.gas_model import R_UNIVERSAL
from ross.seals import labyrinth_seal
from ross.seals.labyrinth_seal import LabyrinthSeal
from ross.units import Q_

import ccp

# Detect REFPROP availability using ccp
# ccp is always available as a dependency, but REFPROP is optional
# Check if REFPROP version is available (not "n/a")
REFPROP_AVAILABLE = "REFPROP : n/a" not in ccp.__version__full


# Common test parameters
COMMON_PARAMS = {
    "n": 0,
    "inlet_pressure": 308000,
    "outlet_pressure": 94300,
    "inlet_temperature": 283.15,
    "preswirl": 0.98,
    "speed": Q_([8000], "RPM"),
    "n_teeth": 16,
    "shaft_diameter": Q_(145, "mm"),
    "radial_clearance": Q_(0.3, "mm"),
    "pitch": Q_(3.175, "mm"),
    "tooth_height": Q_(3.175, "mm"),
    "tooth_width": Q_(0.1524, "mm"),
    "seal_type": "inter",
}

MANUAL_PARAMS = {
    "molar_mass": 28.96807,
    "reference_temperatures": [283.15, 282.60903080958565],
    "reference_viscosities": [1.7746561138374613e-05, 1.7687886306966975e-05],
    "gamma": 1.41,
}

GAS_COMPOSITION = {
    "Nitrogen": 0.7812,
    "Oxygen": 0.2096,
    "Argon": 0.0092,
}


@pytest.fixture
def labyrinth_manual():
    """LabyrinthSeal with manual thermodynamic parameters."""
    seal = LabyrinthSeal(**COMMON_PARAMS, **MANUAL_PARAMS)
    return seal


@pytest.fixture
def labyrinth_gas_composition():
    """LabyrinthSeal with gas_composition."""
    seal = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    return seal


def test_labyrinth_manual_coefficients(labyrinth_manual):
    """Test LabyrinthSeal coefficients with manual parameters.

    This test should always work regardless of REFPROP availability.
    """
    assert_allclose(labyrinth_manual.kxx, -50456.586597, rtol=1e-4)
    assert_allclose(labyrinth_manual.kxy, 35541.605312, rtol=1e-4)
    assert_allclose(labyrinth_manual.kyx, -35541.605312, rtol=1e-4)
    assert_allclose(labyrinth_manual.kyy, -50456.586597, rtol=1e-4)
    assert_allclose(labyrinth_manual.cxx, 23.825111, rtol=1e-4)
    assert_allclose(labyrinth_manual.cxy, 56.255682, rtol=1e-4)
    assert_allclose(labyrinth_manual.cyx, -56.255682, rtol=1e-4)
    assert_allclose(labyrinth_manual.cyy, 23.825111, rtol=1e-4)


def test_labyrinth_manual_attributes(labyrinth_manual):
    """Test that manual parameters are correctly stored."""
    assert_allclose(labyrinth_manual.molar_mass, 28.96807, rtol=1e-6)
    assert_allclose(labyrinth_manual.R, 287.005658, rtol=1e-4)
    assert labyrinth_manual.gamma == 1.41
    assert_allclose(labyrinth_manual.reference_temperatures[0], 283.15, rtol=1e-6)
    assert_allclose(
        labyrinth_manual.reference_temperatures[1], 282.60903080958565, rtol=1e-6
    )
    assert_allclose(
        labyrinth_manual.reference_viscosities[0], 1.7746561138374613e-05, rtol=1e-6
    )
    assert_allclose(
        labyrinth_manual.reference_viscosities[1], 1.7687886306966975e-05, rtol=1e-6
    )


def test_labyrinth_gas_composition_creation(labyrinth_gas_composition):
    """Test that LabyrinthSeal can be created with gas_composition.

    This should work with both REFPROP and HEOS backends.
    """
    assert labyrinth_gas_composition is not None
    assert hasattr(labyrinth_gas_composition, "kxx")
    assert hasattr(labyrinth_gas_composition, "seal_leakage")


def test_labyrinth_gas_composition_derived_properties(labyrinth_gas_composition):
    """Test that thermodynamic properties are auto-derived correctly."""
    # Check that properties were derived
    assert labyrinth_gas_composition.R is not None
    assert labyrinth_gas_composition.gamma is not None
    assert labyrinth_gas_composition.reference_temperatures is not None
    assert labyrinth_gas_composition.reference_viscosities is not None

    assert_allclose(labyrinth_gas_composition.R, 287, rtol=0.02)
    assert_allclose(labyrinth_gas_composition.gamma, 1.4, rtol=0.03)
    assert len(labyrinth_gas_composition.reference_temperatures) == 2
    assert_allclose(
        labyrinth_gas_composition.reference_temperatures[0], 283.15, rtol=1e-4
    )
    assert len(labyrinth_gas_composition.reference_viscosities) == 2
    assert_allclose(
        labyrinth_gas_composition.reference_viscosities[0], 1.77e-05, rtol=0.05
    )


def test_labyrinth_gas_composition_coefficients_range(labyrinth_gas_composition):
    """Test that coefficients are in reasonable range.

    Results may differ slightly between REFPROP and HEOS, but should be
    within a few percent of each other.
    """
    # Expected values with some tolerance for backend differences
    assert_allclose(labyrinth_gas_composition.kxx[0], -50440, rtol=0.01)  # ±1%
    assert_allclose(abs(labyrinth_gas_composition.kxy[0]), 35515, rtol=0.01)  # ±1%
    assert_allclose(labyrinth_gas_composition.cxx[0], 23.8, rtol=0.01)  # ±1%
    assert_allclose(
        labyrinth_gas_composition.seal_leakage[0], 0.023714, rtol=0.01
    )  # ±1%


def test_labyrinth_gas_composition_vs_manual_similarity():
    """Test that gas_composition and manual parameters give similar results.

    They won't be identical because:
    1. gas_composition auto-derives properties from real gas data
    2. Manual parameters use simplified air properties

    But they should be within ~5% of each other.
    """
    labyrinth_gas = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    labyrinth_man = LabyrinthSeal(**COMMON_PARAMS, **MANUAL_PARAMS)

    assert_allclose(labyrinth_gas.kxx[0], labyrinth_man.kxx[0], rtol=0.05)
    assert_allclose(labyrinth_gas.cxx[0], labyrinth_man.cxx[0], rtol=0.05)
    assert_allclose(
        labyrinth_gas.seal_leakage[0], labyrinth_man.seal_leakage[0], rtol=0.05
    )


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_labyrinth_refprop_backend():
    """Test LabyrinthSeal with REFPROP backend explicitly.

    This test only runs when REFPROP is available.
    """
    # Ensure we're using REFPROP
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "REFPROP"

        labyrinth = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # REFPROP expected values (from testing)
        assert_allclose(labyrinth.kxx[0], -50435.21, rtol=1e-3)
        assert_allclose(labyrinth.cxx[0], 23.808, rtol=1e-3)
        assert_allclose(labyrinth.seal_leakage[0], 0.023714, rtol=1e-3)

        # Check derived properties
        assert_allclose(labyrinth.R, 287.12, rtol=1e-3)
        assert_allclose(labyrinth.gamma, 1.406521, rtol=1e-3)

    finally:
        ccp.config.EOS = original_eos


def test_labyrinth_heos_backend():
    """Test LabyrinthSeal with HEOS backend explicitly.

    This test should always work.
    """
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"

        labyrinth = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # HEOS expected values (from testing)
        # Note: HEOS values differ slightly from REFPROP (<0.05%)
        assert_allclose(labyrinth.kxx[0], -50449.96, rtol=1e-3)
        assert_allclose(labyrinth.cxx[0], 23.797, rtol=1e-3)
        assert_allclose(labyrinth.seal_leakage[0], 0.023714, rtol=1e-3)

        # Check derived properties
        assert_allclose(labyrinth.R, 287.12, rtol=1e-3)
        assert_allclose(labyrinth.gamma, 1.40652, rtol=1e-3)

    finally:
        ccp.config.EOS = original_eos


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_labyrinth_refprop_vs_heos_consistency():
    """Compare REFPROP and HEOS results for the same configuration.

    Results should be within 1% of each other.
    """
    original_eos = ccp.config.EOS

    try:
        # Test with REFPROP
        ccp.config.EOS = "REFPROP"
        labyrinth_refprop = LabyrinthSeal(
            **COMMON_PARAMS, gas_composition=GAS_COMPOSITION
        )

        # Test with HEOS
        ccp.config.EOS = "HEOS"
        labyrinth_heos = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # Compare coefficients (should be within 1%)
        assert_allclose(
            labyrinth_refprop.kxx[0],
            labyrinth_heos.kxx[0],
            rtol=0.01,
            err_msg="kxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            labyrinth_refprop.cxx[0],
            labyrinth_heos.cxx[0],
            rtol=0.01,
            err_msg="cxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            labyrinth_refprop.seal_leakage[0],
            labyrinth_heos.seal_leakage[0],
            rtol=0.01,
            err_msg="seal_leakage differs by >1% between REFPROP and HEOS",
        )

        # Derived properties should match closely
        assert_allclose(labyrinth_refprop.R, labyrinth_heos.R, rtol=1e-3)
        assert_allclose(labyrinth_refprop.gamma, labyrinth_heos.gamma, rtol=1e-2)

    finally:
        ccp.config.EOS = original_eos


def test_labyrinth_invalid_gas_composition():
    """Test that invalid gas composition raises appropriate error."""
    invalid_gas = {"InvalidFluid123": 1.0}

    with pytest.raises((ValueError, KeyError)):
        LabyrinthSeal(**COMMON_PARAMS, gas_composition=invalid_gas)


@pytest.fixture
def labyrinth():
    """Original fixture for backward compatibility."""
    seal = LabyrinthSeal(
        n=0,
        inlet_pressure=308000,
        outlet_pressure=94300,
        inlet_temperature=283.15,
        preswirl=0.98,
        speed=Q_([8000], "RPM"),
        n_teeth=16,
        shaft_diameter=Q_(145, "mm"),
        radial_clearance=Q_(0.3, "mm"),
        pitch=Q_(3.175, "mm"),
        tooth_height=Q_(3.175, "mm"),
        tooth_width=Q_(0.1524, "mm"),
        seal_type="inter",
        molar_mass=28.96807,
        reference_temperatures=[283.15, 282.60903080958565],
        reference_viscosities=[1.7746561138374613e-05, 1.7687886306966975e-05],
        gamma=1.41,
    )

    return seal


def test_labyrinth_coefficients(labyrinth):
    """Original test - kept for backward compatibility."""
    assert_allclose(labyrinth.kxx, -50456.586597, rtol=1e-4)
    assert_allclose(labyrinth.kxy, 35541.605312, rtol=1e-4)
    assert_allclose(labyrinth.kyx, -35541.605312, rtol=1e-4)
    assert_allclose(labyrinth.kyy, -50456.586597, rtol=1e-4)
    assert_allclose(labyrinth.cxx, 23.825111, rtol=1e-4)
    assert_allclose(labyrinth.cxy, 56.255682, rtol=1e-4)
    assert_allclose(labyrinth.cyx, -56.255682, rtol=1e-4)
    assert_allclose(labyrinth.cyy, 23.825111, rtol=1e-4)


def test_labyrinth_always_computes_coefficients(labyrinth_manual):
    """The seal always computes rotordynamic coefficients.

    The removed ``analz="LEAKAGE"`` option used to skip ``pert()`` and then
    crash while assembling the (never created) coefficient attributes. There is
    no longer a leakage-only mode: stiffness, damping and leakage are always
    available after construction.
    """
    import inspect

    assert "analz" not in inspect.signature(LabyrinthSeal.__init__).parameters

    for coef in ("kxx", "kyy", "kxy", "kyx", "cxx", "cyy", "cxy", "cyx"):
        value = getattr(labyrinth_manual, coef)
        assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(labyrinth_manual.seal_leakage))


# Real-gas (gas_model) tests --------------------------------------------------

# Dense-gas case (Z well below 1) used for the directional real-gas check.
# Pure methane is robust under both the HEOS and REFPROP backends here.
DENSE_GAS_PARAMS = {
    "n": 0,
    "inlet_pressure": 8e6,
    "outlet_pressure": 3e6,
    "inlet_temperature": 300,
    "preswirl": 0.5,
    "speed": Q_([8000], "RPM"),
    "n_teeth": 10,
    "shaft_diameter": Q_(145, "mm"),
    "radial_clearance": Q_(0.3, "mm"),
    "pitch": Q_(3.175, "mm"),
    "tooth_height": Q_(3.175, "mm"),
    "tooth_width": Q_(0.1524, "mm"),
    "seal_type": "inter",
}
DENSE_GAS = {"Methane": 1.0}

# Methane at 350 K has its minimum of Z(p) near 16 MPa, so across these inlet
# pressures p * kappa_T = 1 - (p / Z) (dZ/dp)_T spans 1.05 (10 MPa) to 0.62
# (40 MPa). Nitrogen at the state of COMMON_PARAMS is the near-ideal case
# (p * kappa_T = 1.001); it stands in for air, whose isentrope table is slow to
# build under HEOS.
SWIRL_TERM_TEMPERATURE = 350.0
SWIRL_TERM_PRESSURES = [5e6, 10e6, 20e6, 30e6, 40e6]
SWIRL_TERM_CASES = {
    "nitrogen": (COMMON_PARAMS, {"Nitrogen": 1.0}),
    **{
        f"methane-{p / 1e6:g}MPa": (
            dict(
                DENSE_GAS_PARAMS,
                inlet_pressure=p,
                outlet_pressure=0.8 * p,
                inlet_temperature=SWIRL_TERM_TEMPERATURE,
            ),
            DENSE_GAS,
        )
        for p in SWIRL_TERM_PRESSURES
    },
}


def test_labyrinth_gas_model_default_is_ideal(labyrinth_manual):
    """The default backend is the perfect-gas model."""
    assert labyrinth_manual.gas_model == "ideal"
    assert labyrinth_manual._real_gas is False


def test_labyrinth_ideal_backend_unchanged(labyrinth_manual):
    """The ideal backend reproduces the reference coefficients tightly.

    The tolerance here is far tighter than the ``rtol=1e-4`` used by the other
    tests (so any real perturbation of the ideal-gas path is caught) while
    tolerating last-bit floating-point differences between platforms and BLAS
    backends. The reference values were captured when the solver iterations
    moved to bracketed root-finding (scipy.optimize.brentq); they agree with
    the pre-existing values to better than 5e-8.
    """
    assert_allclose(labyrinth_manual.kxx[0], -50456.58797187634, rtol=1e-9)
    assert_allclose(labyrinth_manual.kxy[0], 35541.60651449606, rtol=1e-9)
    assert_allclose(labyrinth_manual.cxx[0], 23.825112375903597, rtol=1e-9)
    assert_allclose(labyrinth_manual.cxy[0], 56.255683485760905, rtol=1e-9)
    assert_allclose(labyrinth_manual.seal_leakage[0], 0.023719116584285112, rtol=1e-9)


def test_labyrinth_real_requires_gas_composition():
    """gas_model='real' without gas_composition raises a clear error."""
    with pytest.raises(ValueError, match="requires gas_composition"):
        LabyrinthSeal(**COMMON_PARAMS, **MANUAL_PARAMS, gas_model="real")


def test_labyrinth_invalid_gas_model():
    """An unknown gas_model value raises a clear error."""
    with pytest.raises(ValueError, match="Invalid gas_model"):
        LabyrinthSeal(
            **COMMON_PARAMS, gas_composition=GAS_COMPOSITION, gas_model="bogus"
        )


def test_labyrinth_real_gas_near_ideal():
    """For near-ideal gas (Z ~ 1) the real backend matches the ideal one.

    The standard air case sits at Z ~ 0.999, so the real-gas coefficients and
    leakage must stay within a few percent of the ideal-gas results.
    """
    ideal = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    real = LabyrinthSeal(
        **COMMON_PARAMS, gas_composition=GAS_COMPOSITION, gas_model="real"
    )

    assert real.gas_model == "real"
    assert real._real_gas is True
    assert_allclose(real.kxx[0], ideal.kxx[0], rtol=0.01)
    assert_allclose(real.kxy[0], ideal.kxy[0], rtol=0.01)
    assert_allclose(real.cxx[0], ideal.cxx[0], rtol=0.01)
    assert_allclose(real.seal_leakage[0], ideal.seal_leakage[0], rtol=0.01)


def test_labyrinth_real_gas_dense_gas_leakage():
    """For a dense gas (Z < 1) the real backend predicts higher leakage.

    Leakage scales with density rho = P / (Z R T); for Z < 1 the real density
    exceeds the perfect-gas value, so the real-gas leakage must exceed the
    ideal-gas leakage. Coefficients must remain finite.
    """
    ideal = LabyrinthSeal(**DENSE_GAS_PARAMS, gas_composition=DENSE_GAS)
    real = LabyrinthSeal(
        **DENSE_GAS_PARAMS, gas_composition=DENSE_GAS, gas_model="real"
    )

    assert np.isfinite(real.kxx[0])
    assert np.isfinite(real.seal_leakage[0])
    assert real.seal_leakage[0] > ideal.seal_leakage[0]
    # Density correction at this state (Z ~ 0.88) lifts leakage by a few percent.
    assert_allclose(real.seal_leakage[0], ideal.seal_leakage[0], rtol=0.15)


@pytest.mark.parametrize("gas_model", ["ideal", "real"])
@pytest.mark.parametrize("case", list(SWIRL_TERM_CASES))
def test_labyrinth_swirl_term_is_cavity_density(case, gas_model):
    """The swirl-velocity coefficient of the cavity continuity equation is rho A / Rs.

    Linearizing the circumferential term ``d(rho V A) / (Rs d theta)`` of the
    cavity continuity equation gives ``rho A / Rs`` as the coefficient of the
    swirl-velocity perturbation, with ``rho`` the cavity density of the base
    flow, for any equation of state. One speed is solved serially, so the base
    flow and the gradient table ``cg`` stay on ``seal.solver``.
    """
    params, gas_composition = SWIRL_TERM_CASES[case]
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"
        seal = LabyrinthSeal(
            **params, gas_composition=gas_composition, gas_model=gas_model
        )
    finally:
        ccp.config.EOS = original_eos

    solver = seal.solver
    cavities = np.arange(1, solver.n_teeth)
    area = (solver.radial_clearance + solver.tooth_height) * solver.pitch
    expected = solver.rho[cavities] * area / solver.shaft_radius
    # Round-off only: the ideal closure writes rho as p / (R T), while the base
    # flow marches rho and T separately along the isentrope.
    assert_allclose(solver.cg[2, cavities], expected, rtol=1e-12)


@pytest.mark.parametrize("preswirl", [0.0, 0.5])
@pytest.mark.parametrize("inlet_pressure", SWIRL_TERM_PRESSURES)
def test_labyrinth_real_gas_low_speed_limit(inlet_pressure, preswirl):
    """At low speed the real closure matches the perfect gas of the same density.

    The isothermal compressibility enters the linearized cavity continuity
    equation only through the storage terms ``(omega +- V / Rs) A (d rho / d p)_T``.
    At a rotor speed and whirl frequency of 1 rad/s they are about 1e-5 of the
    leakage terms, while the swirl term ``(rho A / Rs) d V1 / d theta`` keeps
    its weight. The coefficients then depend on the gas only through its
    density along the isentrope and its viscosity in the wall shear, so they
    must match the ideal closure for the perfect gas with the inlet density
    (molar mass ``R_u rho T / p``) and the isentropic exponent ``rho a**2 / p``
    of the real gas, given the same viscosity. The two isentropes reach outlet
    temperatures up to 7 K apart (40 MPa), so each seal gets the viscosity law
    through the real-gas viscosities at the inlet and at the isentropic outlet
    state, placed at its own isentrope temperatures. With preswirl 0 the rotor
    drag drives the swirl; with 0.5 the rotor and stator shear of this inter
    seal balance. Written as ``(p / Rs) A (d rho / d p)_T``, the swirl term
    would carry the factor ``p kappa_T`` and move the coefficients by 4 %
    (5 MPa) to 60 % (40 MPa).
    """
    outlet_pressure = 0.95 * inlet_pressure
    params = dict(
        DENSE_GAS_PARAMS,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        inlet_temperature=SWIRL_TERM_TEMPERATURE,
        preswirl=preswirl,
        speed=[1.0],
    )
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"
        inlet = ccp.State(p=inlet_pressure, T=SWIRL_TERM_TEMPERATURE, fluid=DENSE_GAS)
        outlet = ccp.State(p=outlet_pressure, s=inlet.s(), fluid=DENSE_GAS)
        density = inlet.rho().m
        gamma = density * inlet.speed_sound().m ** 2 / inlet_pressure
        viscosities = [inlet.viscosity().m, outlet.viscosity().m]
        real = LabyrinthSeal(
            **params,
            gas_composition=DENSE_GAS,
            gas_model="real",
            reference_temperatures=[SWIRL_TERM_TEMPERATURE, outlet.T().m],
            reference_viscosities=viscosities,
        )
    finally:
        ccp.config.EOS = original_eos

    perfect_outlet_temperature = SWIRL_TERM_TEMPERATURE * (
        outlet_pressure / inlet_pressure
    ) ** ((gamma - 1) / gamma)
    perfect_gas = LabyrinthSeal(
        **params,
        molar_mass=R_UNIVERSAL * density * SWIRL_TERM_TEMPERATURE / inlet_pressure,
        gamma=gamma,
        reference_temperatures=[SWIRL_TERM_TEMPERATURE, perfect_outlet_temperature],
        reference_viscosities=viscosities,
    )

    # The leakage iteration matches the outlet pressure to 1e-5, which is 2e-4
    # of the 5 % pressure drop; the two isentropes and the two viscosity laws
    # differ at second order in the pressure drop (2e-4 in density at the
    # outlet). Without preswirl cxy is about 4e-5 of cxx, and its relative
    # difference reaches 6.4e-4.
    rtol = 2e-3
    assert_allclose(real.seal_leakage[0], perfect_gas.seal_leakage[0], rtol=rtol)
    for name in ("kxx", "kxy", "cxx", "cxy"):
        assert_allclose(
            getattr(real, name)[0],
            getattr(perfect_gas, name)[0],
            rtol=rtol,
            err_msg=name,
        )


# Published AGA-8 / ISO 12213-2 "Ekofisk" reference gas, mole percent.
EKOFISK_GAS = {
    "methane": 85.9063,
    "ethane": 8.4919,
    "propane": 2.3015,
    "isobutane": 0.3486,
    "n-butane": 0.3506,
    "isopentane": 0.0509,
    "n-pentane": 0.0480,
    "nitrogen": 1.0068,
    "carbon dioxide": 1.4954,
}

LEAKAGE_BISECTION_BASE = {
    "n": 0,
    "shaft_diameter": 0.2,
    "tooth_height": 3e-3,
    "speed": [1000.0],
    "preswirl": 0.5,
}

PERFECT_GAS_VISCOSITY = {
    "reference_temperatures": [300.0, 290.0],
    "reference_viscosities": [1.8e-5, 1.75e-5],
}


def _throttle_flux(solver, station, pressure_ratio):
    """Return the mass flux through the throttle at ``station`` at a pressure ratio."""
    i = station
    return solver.gas.throttle_mass_flux(
        solver.discharge_coefficient,
        solver.radial_clearance,
        solver.p[i - 1],
        pressure_ratio,
        solver.rho[i - 1],
        solver.t[i - 1],
        solver.w[i - 1],
        solver.carryover_factor,
    )


def _critical_ratios(solver):
    """Return the critical pressure ratio of every throttle from its upstream state."""
    return np.array(
        [
            solver.gas.critical_pr(
                solver.p[i - 1],
                solver.w[i - 1],
                solver.carryover_factor,
                solver.t[i - 1],
            )
            for i in range(1, solver.n_teeth + 1)
        ]
    )


def _assert_base_flow_converged(seal, choked):
    """Check the base flow left on the solver against the equations it solves.

    The mass flux through every throttle, recomputed with the gas model from
    the throttle state, must equal the leakage mass flux, and no throttle may
    expand below its critical pressure ratio. An unchoked seal must meet the
    outlet pressure within the solver tolerance; a choked one ends at the
    critical pressure ratio of the last throttle, within the choke tolerance,
    with a throat pressure above the outlet pressure.
    """
    solver = seal.solver
    flux = [
        _throttle_flux(solver, i, solver.pr[i - 1])
        for i in range(1, solver.n_teeth + 1)
    ]
    critical_pr = _critical_ratios(solver)
    # brentq solves each pressure ratio to xtol=1e-14.
    assert_allclose(flux, solver.mdot, rtol=1e-9)
    assert np.all(solver.pr >= critical_pr)
    if choked:
        # Choke test of the solver: within 0.5 % of the critical ratio.
        assert_allclose(solver.pr[-1], critical_pr[-1], rtol=0.005)
        assert solver.p[-1] > solver.outlet_pressure
    else:
        # Outlet test of the solver: relative pressure error below 1e-5.
        assert_allclose(solver.p[-1], solver.outlet_pressure, rtol=1e-5)


@pytest.mark.parametrize(
    "eos, gas, params",
    [
        pytest.param(
            "HEOS",
            {"methane": 1.0},
            {
                "seal_type": "rotor",
                "n_teeth": 13,
                "radial_clearance": 0.125e-3,
                "pitch": 5.3e-3,
                "tooth_width": 0.34e-3,
                "inlet_pressure": 0.328e6,
                "outlet_pressure": 0.204e6,
                "inlet_temperature": 305.0,
            },
            id="methane-rotor-13-teeth",
        ),
        pytest.param(
            "HEOS",
            {"carbon dioxide": 1.0},
            {
                "seal_type": "rotor",
                "n_teeth": 4,
                "radial_clearance": 0.114e-3,
                "pitch": 4.8e-3,
                "tooth_width": 0.345e-3,
                "inlet_pressure": 39.0e6,
                "outlet_pressure": 35.3e6,
                "inlet_temperature": 366.0,
            },
            id="co2-rotor-4-teeth",
        ),
        pytest.param(
            "HEOS",
            {"nitrogen": 1.0},
            {
                "seal_type": "rotor",
                "n_teeth": 20,
                "radial_clearance": 0.196e-3,
                "pitch": 5.53e-3,
                "tooth_width": 0.255e-3,
                "inlet_pressure": 1.2e6,
                "outlet_pressure": 0.512e6,
                "inlet_temperature": 304.0,
            },
            id="nitrogen-rotor-20-teeth",
        ),
        pytest.param(
            "HEOS",
            {"ethane": 1.0},
            {
                "seal_type": "stator",
                "n_teeth": 17,
                "radial_clearance": 0.115e-3,
                "pitch": 5.5e-3,
                "tooth_width": 0.4e-3,
                "inlet_pressure": 5.04e6,
                "outlet_pressure": 3.63e6,
                "inlet_temperature": 370.0,
            },
            id="ethane-stator-17-teeth",
        ),
        pytest.param(
            "REFPROP",
            EKOFISK_GAS,
            {
                "seal_type": "stator",
                "n_teeth": 4,
                "radial_clearance": 0.3e-3,
                "pitch": 5e-3,
                "tooth_width": 0.2e-3,
                "inlet_pressure": 20.0e6,
                "outlet_pressure": 19.0e6,
                "inlet_temperature": 340.0,
            },
            id="ekofisk-stator-4-teeth",
            marks=pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP"),
        ),
    ],
)
def test_labyrinth_real_gas_leakage_bisection_converges(eos, gas, params):
    """The real-gas leakage bisection converges where it used to cycle forever.

    Every tooth used to be bracketed, and judged choked, with the critical
    pressure ratio of the last tooth in the previous iterate, frozen at the
    first bracket restart. After an iterate at a higher flux, which marched
    most teeth before it choked, that ratio was close to 1, fluxes near the
    solution were reported as choked, and the bisection never met its tests.
    """
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = eos
        seal = LabyrinthSeal(
            **LEAKAGE_BISECTION_BASE, **params, gas_composition=gas, gas_model="real"
        )
    finally:
        ccp.config.EOS = original_eos

    _assert_base_flow_converged(seal, choked=False)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "seal_type": "rotor",
                "n_teeth": 11,
                "radial_clearance": 0.45e-3,
                "pitch": 4e-3,
                "tooth_width": 0.2e-3,
            },
            id="rotor-11-teeth",
        ),
        pytest.param(
            {
                "seal_type": "stator",
                "n_teeth": 12,
                "radial_clearance": 0.45e-3,
                "pitch": 4e-3,
                "tooth_width": 0.3e-3,
            },
            id="stator-12-teeth",
        ),
    ],
)
def test_labyrinth_choked_leakage_bisection_converges(params):
    """The perfect-gas leakage bisection converges where it used to cycle forever.

    These seals are choked. As in the real-gas case, a ratio left by an
    iterate at a higher flux reported a flux below the choked one as choked,
    and the ratio frozen at the bracket restart then kept the last throttle
    out of the choke tolerance. The ratio of specific heats is the inlet
    value ``gas_composition`` gives for natural gas at 20 MPa and 300 K
    (cp / cv = 1.98 for the AGA-8 Ekofisk gas).
    """
    seal = LabyrinthSeal(
        **LEAKAGE_BISECTION_BASE,
        **params,
        **PERFECT_GAS_VISCOSITY,
        inlet_pressure=20.0e6,
        outlet_pressure=2.0e6,
        inlet_temperature=300.0,
        molar_mass=19.0,
        gamma=2.0,
    )

    _assert_base_flow_converged(seal, choked=True)


@pytest.mark.parametrize(
    "gamma, molar_mass, params",
    [
        pytest.param(1.4, 19.0, {"n_teeth": 4}, id="gamma-1.4"),
        pytest.param(2.5, 19.0, {"n_teeth": 4}, id="gamma-2.5"),
        pytest.param(4.0, 19.0, {"n_teeth": 4}, id="gamma-4"),
        pytest.param(6.0, 19.0, {"n_teeth": 4}, id="gamma-6"),
        pytest.param(4.0, 19.0, {"n_teeth": 16}, id="gamma-4-16-teeth"),
        pytest.param(
            8.34,
            44.01,
            {
                "n_teeth": 8,
                "radial_clearance": 0.45e-3,
                "inlet_pressure": 8.0e6,
                "outlet_pressure": 1.2e6,
                "inlet_temperature": 310.0,
            },
            id="gamma-8.34-carbon-dioxide",
        ),
    ],
)
def test_labyrinth_choked_throttles_stay_above_critical_ratio(
    gamma, molar_mass, params
):
    """Every throttle is bracketed by its own critical pressure ratio.

    With carry-over, the perfect-gas critical ratio lies above the maximum of
    the throttle flux, the more so the larger the ratio of specific heats. A
    lower bracket, such as the ratio of the first tooth, which has no
    carry-over, lets a throttle expand below its critical ratio. The last
    case uses the inlet cp / cv of carbon dioxide at 8 MPa and 310 K: there
    the loop used to return an outlet match with the last throttle 11 %
    below its critical ratio.
    """
    seal = LabyrinthSeal(
        **{
            **LEAKAGE_BISECTION_BASE,
            "seal_type": "rotor",
            "radial_clearance": 0.2e-3,
            "pitch": 3.175e-3,
            "tooth_width": 0.1524e-3,
            "inlet_pressure": 10.0e6,
            "outlet_pressure": 0.5e6,
            "inlet_temperature": 300.0,
            **params,
        },
        **PERFECT_GAS_VISCOSITY,
        molar_mass=molar_mass,
        gamma=gamma,
    )

    _assert_base_flow_converged(seal, choked=True)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "seal_type": "rotor",
                "n_teeth": 9,
                "radial_clearance": 0.31e-3,
                "pitch": 4.3e-3,
                "tooth_width": 0.18e-3,
                "inlet_pressure": 14.6e6,
                "outlet_pressure": 2.97e6,
                "inlet_temperature": 433.0,
                "molar_mass": 53.0,
                "gamma": 2.52,
            },
            id="rotor-9-teeth",
        ),
        pytest.param(
            {
                "seal_type": "rotor",
                "n_teeth": 13,
                "radial_clearance": 0.28e-3,
                "pitch": 4.6e-3,
                "tooth_width": 0.35e-3,
                "inlet_pressure": 1.44e6,
                "outlet_pressure": 0.326e6,
                "inlet_temperature": 271.5,
                "molar_mass": 21.55,
                "gamma": 1.43,
            },
            id="rotor-13-teeth",
        ),
    ],
)
def test_labyrinth_near_choked_overshoot_is_rejected(params):
    """A march near the choke is a solution only above the outlet pressure.

    The last throttle of these unchoked seals comes within the choke
    tolerance of its critical ratio at a mass flux above the leakage, with
    the outlet station below the outlet pressure. The loop used to stop
    there, 11 % and 0.8 % below the outlet pressure; the bisection must go
    on to the outlet match.
    """
    seal = LabyrinthSeal(**LEAKAGE_BISECTION_BASE, **params, **PERFECT_GAS_VISCOSITY)

    _assert_base_flow_converged(seal, choked=False)


@pytest.mark.parametrize(
    "gas, params",
    [
        pytest.param(
            {"methane": 1.0},
            {
                "seal_type": "inter",
                "n_teeth": 3,
                "radial_clearance": 0.146e-3,
                "pitch": 4.5e-3,
                "tooth_width": 0.186e-3,
                "inlet_pressure": 4.14e6,
                "outlet_pressure": 0.416e6,
                "inlet_temperature": 363.0,
            },
            id="methane-inter-3-teeth",
        ),
        pytest.param(
            {"ethane": 1.0},
            {
                "seal_type": "inter",
                "n_teeth": 5,
                "radial_clearance": 0.21e-3,
                "pitch": 6e-3,
                "tooth_width": 0.19e-3,
                "inlet_pressure": 2.0e6,
                "outlet_pressure": 0.086e6,
                "inlet_temperature": 350.0,
            },
            id="ethane-inter-5-teeth",
        ),
        pytest.param(
            {"ethane": 1.0},
            {
                "seal_type": "stator",
                "n_teeth": 5,
                "radial_clearance": 0.39e-3,
                "pitch": 3.55e-3,
                "tooth_width": 0.19e-3,
                "inlet_pressure": 0.295e6,
                "outlet_pressure": 0.0304e6,
                "inlet_temperature": 342.0,
            },
            id="ethane-stator-5-teeth",
        ),
        pytest.param(
            {"ethane": 1.0},
            {
                "seal_type": "rotor",
                "n_teeth": 16,
                "radial_clearance": 0.37e-3,
                "pitch": 4.55e-3,
                "tooth_width": 0.17e-3,
                "inlet_pressure": 1.07e6,
                "outlet_pressure": 0.118e6,
                "inlet_temperature": 374.0,
            },
            id="ethane-rotor-16-teeth",
        ),
    ],
)
def test_labyrinth_real_gas_choke_onset(gas, params):
    """A choked real-gas seal converges at the choke onset of its last throttle.

    The critical ratio of the tabulated gas, its sonic point, lies slightly
    below the maximum of the interpolated throttle flux. As the mass flux
    rises, the last pressure ratio then stays 0.5 to 1 % above the critical
    ratio until the throttle chokes, and never enters the choke tolerance.
    The loop used to cycle forever on three of these seals, and stopped on
    the stator seal only because rounding put its last pressure ratio on the
    critical ratio. The choked leakage is the flux of the last throttle at
    its critical ratio, and the maximum of its flux within the interpolation
    error of the table.
    """
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"
        with pytest.warns(UserWarning, match="Flow choked in the last throttle"):
            seal = LabyrinthSeal(
                **LEAKAGE_BISECTION_BASE,
                **params,
                gas_composition=gas,
                gas_model="real",
            )
    finally:
        ccp.config.EOS = original_eos

    solver = seal.solver
    last = solver.n_teeth
    flux = [_throttle_flux(solver, i, solver.pr[i - 1]) for i in range(1, last + 1)]
    critical_pr = _critical_ratios(solver)
    # brentq solves each pressure ratio to xtol=1e-14.
    assert_allclose(flux, solver.mdot, rtol=1e-9)
    assert np.all(solver.pr >= critical_pr)
    assert solver.p[-1] > solver.outlet_pressure
    # The bisection closes on adjacent floating-point mass fluxes.
    assert_allclose(
        _throttle_flux(solver, last, critical_pr[-1]), solver.mdot, rtol=1e-12
    )
    # The sonic ratio of the 120-point table lies within 0.4 % of the flux
    # maximum, where the flux is flat to about 2e-5.
    subsonic_flux = [
        _throttle_flux(solver, last, pressure_ratio)
        for pressure_ratio in np.linspace(critical_pr[-1], 1.0, 2001)
    ]
    assert_allclose(max(subsonic_flux), solver.mdot, rtol=1e-4)


def test_labyrinth_leakage_bisection_is_bounded(monkeypatch):
    """The leakage bisection raises instead of running without end."""
    monkeypatch.setattr(labyrinth_seal, "MAX_LEAKAGE_ITERATIONS", 3)
    with pytest.raises(RuntimeError, match="did not converge in 3 bisection steps"):
        LabyrinthSeal(**COMMON_PARAMS, **MANUAL_PARAMS)


def test_labyrinth_whirl_frequency_grid():
    """The 2-D table reuses one base flow per speed and follows the whirl frequency."""
    speed = Q_([5000, 8000], "RPM").to("rad/s").m
    params = dict(COMMON_PARAMS, **MANUAL_PARAMS)
    params["speed"] = speed
    synchronous = LabyrinthSeal(**params)

    frequency = np.array([0.25 * speed[1], speed[0], speed[1], 1.5 * speed[1]])
    grid = LabyrinthSeal(frequency=frequency, **params)

    assert grid.kxx_interpolated.kind == "grid"
    kxx = np.array(grid.kxx)
    cxy = np.array(grid.cxy)
    assert kxx.shape == (2, 4)

    # the synchronous diagonal reproduces the 1-D solve exactly
    assert kxx[0, 1] == synchronous.kxx[0]
    assert kxx[1, 2] == synchronous.kxx[1]
    assert cxy[1, 2] == synchronous.cxy[1]
    assert_allclose(float(grid.kxx_interpolated(speed[1])), synchronous.kxx[1])

    # the base flow does not depend on the whirl frequency
    assert_allclose(grid.seal_leakage, synchronous.seal_leakage)
    assert_allclose(grid.p[1], synchronous.p[1])

    # the perturbation does: the direct stiffness drops strongly with whirl
    assert kxx[1, 3] < kxx[1, 2] < kxx[1, 1] < kxx[1, 0] < 0
    assert np.all(np.diff(np.array(grid.cxx)[1]) > 0)


def test_labyrinth_solver_row_matches_single_solves():
    params = dict(COMMON_PARAMS, **MANUAL_PARAMS)
    seal = LabyrinthSeal(**params)
    speed = Q_(8000, "RPM").to("rad/s").m
    row = seal.solver.solve_row(speed, [0.5 * speed, speed])
    single = seal.solver.solve(speed, 0.5 * speed)
    assert row[0]["kxx"] == single["kxx"]
    assert row[0]["cxx"] == single["cxx"]
    assert row[1]["kxx"] == seal.kxx[0]


def test_labyrinth_grid_save_load(tmp_path):
    speed = Q_([6000, 8000], "RPM").to("rad/s").m
    params = dict(COMMON_PARAMS, **MANUAL_PARAMS)
    params["speed"] = speed
    grid = LabyrinthSeal(frequency=[0.5 * speed[0], speed[1]], **params)
    file = tmp_path / "labyrinth_grid.toml"
    grid.save(file)
    loaded = LabyrinthSeal.load(file)
    assert_allclose(loaded.speed, grid.speed)
    assert_allclose(loaded.frequency, grid.frequency)
    assert_allclose(np.array(loaded.kxx), np.array(grid.kxx))
    assert_allclose(loaded.seal_leakage, grid.seal_leakage)
    assert_allclose(
        loaded.K(frequency=500.0, speed=700.0), grid.K(frequency=500.0, speed=700.0)
    )
