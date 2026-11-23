import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.tilting_pad import (
    TiltingPad,
    tilting_pad_adiabatic_example,
)
from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture
from ross.units import Q_


@pytest.fixture(scope="module")
def tilting_pad():
    return tilting_pad_adiabatic_example()


def test_geometry_translation(tilting_pad):
    assert type(tilting_pad) is TiltingPad
    assert tilting_pad.n_pads == 5
    assert_allclose(tilting_pad.pivot_angle, np.radians([18, 90, 162, 234, 306]))
    assert_allclose(tilting_pad.pad_arc, np.radians([60] * 5))
    assert_allclose(tilting_pad.preload, [0.5] * 5)
    assert tilting_pad.bearing_type == "conventional_tilting_pad"
    assert tilting_pad.thermal_type == "adiabatic"
    assert_allclose(tilting_pad.hot_oil_lambda, 0.8)
    assert_allclose(tilting_pad.pad_conductivity, 116.0)
    assert_allclose(tilting_pad.edges_convection, 1500.0)
    assert tilting_pad.total_ex_film == 20
    assert tilting_pad.total_ez_film == 10
    assert tilting_pad.total_ey_pad == 10


def test_coefficients_solved(tilting_pad):
    for name in ("kxx", "kyy", "cxx", "cyy"):
        assert float(getattr(tilting_pad, name)[0]) > 0
    out = tilting_pad._results.outputs[0]
    assert len(out["tilt_angle"][0]) == 5


def test_matches_solver_fixture():
    """TiltingPad reproduces the raw 5-pad tilting-pad solver fixture."""
    kwargs, outputs = bearing_kwargs_from_fixture("tilt_5pad_isoviscous")
    bearing = TiltingPad(
        n=kwargs["n"],
        speed=kwargs["speed"],
        journal_diameter=kwargs["journal_diameter"],
        radial_clearance=kwargs["radial_clearance"],
        pad_thickness=kwargs["pad_thickness"],
        pivot_angle=kwargs["pivot_angle"],
        pad_arc=kwargs["pad_arc"],
        pad_axial_length=kwargs["pad_axial_length"],
        preload=kwargs["preload"],
        offset=kwargs["offset"],
        lubricant=kwargs["lubricant"],
        oil_supply_temperature=kwargs["oil_supply_temperature"],
        oil_flow_v=kwargs["oil_flow_v"],
        thermal_type=kwargs["thermal_type"],
        equilibrium_type=kwargs["equilibrium_type"],
        xj=kwargs["initial_position"][0],
        yj=kwargs["initial_position"][1],
        fxs_load=kwargs["fxs_load"],
        fys_load=kwargs["fys_load"],
        hot_oil_lambda=kwargs["hot_oil_lambda"],
        journal_temperature=kwargs["journal_temperature"],
        total_ex_film=kwargs["total_ex_film"],
        total_ez_film=kwargs["total_ez_film"],
        total_ey_pad=kwargs["total_ey_pad"],
        total_ey_film=kwargs["total_ey_film"],
        weight=kwargs["weight"],
        probes=kwargs["probes"],
        pad_E=kwargs["pad_E"],
        pad_poisson=kwargs["pad_poisson"],
        pad_conductivity=kwargs["pad_conductivity"],
        pad_expansion=kwargs["pad_expansion"],
        pad_density=kwargs["pad_density"],
        journal_expansion=kwargs["journal_expansion"],
        shell_expansion=kwargs["shell_expansion"],
        pad_convection=kwargs["pad_convection"],
        edges_convection=kwargs["edges_convection"],
        environment_temperature=kwargs["environment_temperature"],
        environment_convection=kwargs["environment_convection"],
        reference_temperature=kwargs["reference_temperature"],
        excitation_ratio=kwargs["excitation_ratio"],
        starvation_number=kwargs["starvation_number"],
        relax_pressure=kwargs["relax_pressure"],
        relax_temperature=kwargs["relax_temperature"],
        relax_deformation=kwargs["relax_deformation"],
        re_laminar=kwargs["re_laminar"],
        re_turbulent=kwargs["re_turbulent"],
    )
    for name in ("kxx", "kyy", "cxx", "cyy"):
        assert_allclose(
            np.asarray(getattr(bearing, name), dtype=float),
            np.asarray(outputs[name], dtype=float),
            rtol=1e-8,
            err_msg=f"{name} differs from the solver fixture",
        )


def test_eccentricity_attitude_initial_position():
    bearing = TiltingPad(
        n=1,
        speed=Q_([3000], "RPM"),
        equilibrium_type="match_eccentricity",
        thermal_type=None,
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        preload=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        eccentricity=0.35,
        attitude_angle=Q_(287.5, "deg"),
        total_ex_film=20,
        total_ez_film=10,
        total_ey_pad=10,
        total_ey_film=10,
    )
    expected = (
        0.35 * np.cos(np.radians(287.5)),
        0.35 * np.sin(np.radians(287.5)),
    )
    assert_allclose(bearing.initial_position, expected)
    out = bearing._results.outputs[0]
    assert_allclose(out["eccentricity"][0], 0.35, rtol=1e-6)


def test_restrictions():
    kwargs = dict(
        n=1,
        speed=Q_([3000], "RPM"),
        equilibrium_type="match_load",
        thermal_type=None,
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        preload=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        fxs_load=8.8405e02,
        fys_load=-2.6704e03,
        total_ex_film=20,
        total_ez_film=10,
        total_ey_pad=10,
        total_ey_film=10,
    )

    with pytest.raises(ValueError, match="FixedGeometryBearing"):
        TiltingPad(bearing_type="fixed_geometry", **kwargs)

    no_flow = dict(kwargs)
    no_flow.pop("oil_flow_v")
    with pytest.warns(UserWarning, match="ample flooded supply"):
        TiltingPad(**no_flow)


def test_pivot_flexibility_runs():
    bearing = TiltingPad(
        n=1,
        speed=Q_([3000], "RPM"),
        equilibrium_type="match_load",
        thermal_type=None,
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        preload=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        fxs_load=8.8405e02,
        fys_load=-2.6704e03,
        deform_type="pad_pivot_mechanical",
        pivot_type="user_specified_stiffness",
        pivot_stiffness=5e8,
        total_ex_film=20,
        total_ez_film=10,
        total_ey_pad=10,
        total_ey_film=10,
    )
    out = bearing._results.outputs[0]
    assert_allclose(out["k_pivot"][0], [5e8] * 5)
    assert max(out["deform_pivot"][0]) > 0.0
    assert float(bearing.kxx[0]) > 0


def test_tilting_pad_coefficients_depend_on_whirl():
    """Pad dofs are condensed at the whirl frequency, so the table follows it."""
    kwargs, _ = bearing_kwargs_from_fixture("tilt_5pad_isoviscous")
    speed = np.atleast_1d(np.asarray(kwargs.pop("speed"), dtype=float))
    kwargs.pop("excitation_ratio", None)
    from ross.bearings.fluid_film_bearing import FluidFilmBearing

    bearing = FluidFilmBearing(
        speed=speed, frequency=np.array([0.3, 1.0]) * speed[0], **kwargs
    )
    reference = FluidFilmBearing(speed=speed, excitation_ratio=0.3, **kwargs)

    kxx = np.array(bearing.kxx)
    assert kxx.shape == (1, 2)
    assert kxx[0, 0] == reference.kxx[0]
    assert kxx[0, 0] > 1.2 * kxx[0, 1]
    assert_allclose(
        float(bearing.kxx_interpolated(0.3 * speed[0], speed[0])), reference.kxx[0]
    )


def _five_pad(**overrides):
    """Coarse isoviscous 5-pad bearing for the input checks (load on a pad)."""
    kwargs = dict(
        n=1,
        speed=Q_([3000], "RPM"),
        thermal_type=None,
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        preload=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        total_ex_film=20,
        total_ez_film=10,
        total_ey_pad=10,
        total_ey_film=10,
    )
    kwargs.update(overrides)
    return kwargs


def test_default_equilibrium_with_load_warns():
    """The default match_eccentricity holds the journal and ignores the load."""
    with pytest.warns(UserWarning, match="match_load"):
        held = TiltingPad(**_five_pad(fys_load=-500.0))
    # the journal stays at the default eccentricity 0.3 whatever the load
    assert_allclose(held._results.outputs[0]["eccentricity"][0], 0.3, rtol=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        TiltingPad(**_five_pad(fys_load=-500.0, equilibrium_type="match_load"))
        TiltingPad(**_five_pad())


@pytest.mark.parametrize(
    "overrides, argument",
    [
        # the pivot defaults (pivot_stiffness 0, Hertz diameters 0)
        (dict(deform_type="pad_pivot_mechanical"), "pivot_stiffness"),
        (
            dict(deform_type="pad_pivot_mechanical", pivot_type="ball_in_socket"),
            "pivot_diameter",
        ),
        (
            dict(
                deform_type="pad_pivot_mechanical",
                pivot_type="button",
                pivot_diameter=0.02,
            ),
            "house_diameter",
        ),
    ],
)
def test_pivot_defaults_raise(overrides, argument):
    with pytest.raises(ValueError, match=argument):
        TiltingPad(**_five_pad(equilibrium_type="match_load", **overrides))


def test_stiff_pivot_builds():
    """k_piv = 1e8 N/m: a valid neighbour of the soft pivot below."""
    bearing = TiltingPad(
        **_five_pad(
            equilibrium_type="match_load",
            fys_load=-2.8e3,
            deform_type="pad_pivot_mechanical",
            pivot_stiffness=1e8,
        )
    )
    out = bearing._results.outputs[0]
    assert_allclose(out["k_pivot"][0], [1e8] * 5)
    # the film carries the load: |F_film - W| / W within the Newton tolerance
    assert abs(out["fy_hydro"][0] - 2.8e3) / 2.8e3 < 2e-3
    # pad deflection = pad force / pivot stiffness (user-specified model)
    assert max(out["deform_pivot"][0]) > 0.0
    # so the pivot spring forces k * deflection, acting along the pivot radii
    # (18, 90, ..., 306 deg), must carry the 2.8 kN load on the housing
    pivot = np.radians([18, 90, 162, 234, 306])
    spring = 1e8 * np.asarray(out["deform_pivot"][0])
    resultant = [np.sum(spring * np.cos(pivot)), np.sum(spring * np.sin(pivot))]
    assert np.hypot(resultant[0], resultant[1] + 2.8e3) / 2.8e3 < 2e-3


def test_soft_pivot_does_not_crash(monkeypatch):
    """k_piv = 1e7 N/m used to raise ZeroDivisionError in the Newton step.

    The pivot deflection under the pad load (about 230 um on the loaded pads)
    is larger than the pad clearance (150 um), so the pivot iteration pushes
    every pad out of the film and the film stiffness becomes exactly zero. The
    Newton step now treats that as a singular system. The outer loops are
    capped at three iterations here so that the test only runs through the
    first collapse; reporting such a case as not converged is left to
    the convergence report.
    """
    # ``thd`` binds MAX_ITERATION at import time and ``driver.run_case``
    # imports it lazily: import it here and patch its copy too, so that the
    # cap does not stay in ``thd`` after the test (monkeypatch restores both).
    from ross.bearings.fluid_film import constants, hydrodynamics, thd

    singular = []
    newton_step = hydrodynamics._newton_step

    def spy(*args):
        step = newton_step(*args)
        if step[0] is None:
            singular.append(args[:4])
        return step

    monkeypatch.setattr(hydrodynamics, "_newton_step", spy)
    monkeypatch.setattr(constants, "MAX_ITERATION", 3)
    monkeypatch.setattr(thd, "MAX_ITERATION", 3)
    TiltingPad(
        **_five_pad(
            equilibrium_type="match_load",
            fys_load=-2.8e3,
            deform_type="pad_pivot_mechanical",
            pivot_stiffness=1e7,
        )
    )
    assert any(k[0] == 0.0 or k[3] == 0.0 for k in singular)


def _zero_preload_four_pad(pivots, **overrides):
    """Isoviscous 4-pad, 80 deg, zero-preload bearing (Someya calculation 24)."""
    kwargs = dict(
        n=1,
        speed=Q_([3000], "RPM"),
        thermal_type=None,
        journal_diameter=0.1,
        radial_clearance=100e-6,
        pad_thickness=0.01,
        pivot_angle=Q_(pivots, "deg"),
        pad_arc=Q_([80] * 4, "deg"),
        pad_axial_length=[0.05] * 4,
        preload=[0.0] * 4,
        offset=[0.5] * 4,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(20, "l/min"),
        total_ex_film=30,
        total_ez_film=12,
    )
    kwargs.update(overrides)
    return kwargs


def test_zero_preload_pads_solve():
    """Zero-preload pads used to stop the solve with ZeroDivisionError.

    At zero preload the end of the admissible tilt range is the tilt at
    which a pad edge touches the journal, so the tilt search of an unloaded
    pad could end on a film of round-off thickness (1e-20 m) and the
    flow-regime kernel divided by its zero mean viscosity. Such a film now
    counts as closed (``CLOSED_FILM``). Physics checks: the two pads facing
    the load at +-45 deg carry it and the two others carry nothing, so the
    stiffness is isotropic (each loaded pad acts along its pivot line).
    """
    bearing = TiltingPad(
        **_zero_preload_four_pad(
            [45, 135, 225, 315], equilibrium_type="match_load", fys_load=-1600.0
        )
    )
    assert bool(bearing.results.converged[0])
    out = bearing._results.outputs[0]
    assert np.all(np.asarray(out["h_min"][0]) > 1e-6)
    forces = np.hypot(*bearing.results.pad_forces[0].T)
    assert_allclose(forces[[0, 1]], 0.0, atol=0.0)
    assert_allclose(forces[2], forces[3], rtol=1e-6)
    assert_allclose(bearing.kxx[0], bearing.kyy[0], rtol=1e-6)
    assert_allclose(bearing.cxx[0], bearing.cyy[0], rtol=1e-6)


@pytest.mark.parametrize("first", [225, 315])
def test_pad_order_does_not_change_the_solution(first):
    """The listed order of the pads must not change a thin-film solution.

    Regression test. The 3-D film mesh is matched to the energy mesh on the
    first pad's film at the initial journal position. With that pad loaded
    (here 34 um of film over 30 radial elements, about 1.1 um apart) a
    matching tolerance coarser than the radial node spacing matches an
    energy node to the 3-D nodes of several layers: with 1e-6 m the pads
    listed from 315 deg gave 6534 and 3875 N on the two symmetric loaded
    pads instead of 4976 N on each, and listed from 225 deg the matching
    wrote past its array and aborted the process. The interface tolerance
    ``INTERFACE_TOL`` (1e-8 m) is far below that spacing, and both matching
    loops now raise ``ValueError`` instead of writing past a row.
    """
    held = dict(equilibrium_type="match_eccentricity", xj=0.0, yj=-0.807)
    pivots = [45, 135, 225, 315]
    shift = pivots.index(first)
    ordered = TiltingPad(**_zero_preload_four_pad(pivots, **held))
    rotated = TiltingPad(
        **_zero_preload_four_pad(pivots[shift:] + pivots[:shift], **held)
    )
    f_ordered = ordered.results.pad_forces[0]
    f_rotated = np.roll(rotated.results.pad_forces[0], shift, axis=0)
    scale = np.abs(f_ordered).max()
    assert_allclose(f_rotated, f_ordered, rtol=0, atol=1e-9 * scale)
    # The loaded pads (225 and 315 deg) mirror each other about -y.
    assert_allclose(f_ordered[2, 0], -f_ordered[3, 0], rtol=1e-9)
    assert_allclose(f_ordered[2, 1], f_ordered[3, 1], rtol=1e-9)
    for name in ("kxx", "kxy", "kyx", "kyy", "cxx", "cyy"):
        assert_allclose(getattr(rotated, name)[0], getattr(ordered, name)[0], rtol=1e-9)


@pytest.mark.parametrize(
    "equilibrium",
    [
        dict(equilibrium_type="match_load", fxs_load=1500.0, fys_load=-2600.0),
        dict(equilibrium_type="match_eccentricity", xj=0.15, yj=-0.25),
    ],
    ids=["match_load", "match_eccentricity"],
)
def test_tilt_angle_rebuilds_the_film_thickness(equilibrium):
    """The documented film-thickness expression rebuilds the film field.

    Pins the convention of ``operating_results["tilt_angle"]`` stated in
    ``FluidFilmBearingResults.operating_results``: for a rigid pad ``k`` at
    angle ``theta`` from its leading edge (global angle ``theta_g =
    leading_edge_angle + theta``, from ``+x`` towards ``+y``),

        h = C_p - xj cos(theta_g) - yj sin(theta_g)
            - (C_p - C_b) cos(theta - theta_p)
            - (R + C_b + t) delta_k sin(theta - theta_p)

    with ``C_p = C_b / (1 - preload)`` and ``theta_p = offset * pad_arc``.
    The pivots are offset, the preloads differ and the load is off a pad,
    so the pad loads and tilts all differ and a sign error in any term,
    a pivot measured from the pad centre or a tilt about the journal
    radius instead of the pivot radius gives an error of several um.

    The field is built from the same doubles by the same expression, so
    the agreement is to round-off; ``rtol=1e-12`` leaves room only for
    the order of the floating-point operations.
    """
    preload = [0.5, 0.3, 0.5, 0.4, 0.5]
    offset = 0.6
    bearing = TiltingPad(
        **_five_pad(preload=preload, offset=[offset] * 5, **equilibrium)
    )
    results = bearing.results
    op = bearing.operating_results
    assert op["converged"][0]
    tilt = np.asarray(op["tilt_angle"][0])
    xj, yj = op["xj"][0], op["yj"][0]
    radius = 0.5 * 101.6e-3
    cb = 74.9e-6
    thickness = 12.7e-3
    arc = np.radians(60.0)
    pad_load = np.asarray(op["pad_load"][0])
    assert np.ptp(pad_load) > 0.3 * pad_load.max()
    assert np.ptp(tilt) > 0.3 * np.abs(tilt).max()
    assert_allclose(
        results.leading_edge_angles,
        np.radians([18, 90, 162, 234, 306]) - offset * arc,
    )

    for k in range(5):
        cp = cb / (1.0 - preload[k])
        theta = results.theta_grids[0][k]
        theta_g = results.leading_edge_angles[k] + theta
        dtheta = theta - offset * arc
        h = (
            cp
            - xj * np.cos(theta_g)
            - yj * np.sin(theta_g)
            - (cp - cb) * np.cos(dtheta)
            - (radius + cb + thickness) * tilt[k] * np.sin(dtheta)
        )
        field = results.film_thickness_fields[0][k]
        assert_allclose(field, h, rtol=1e-12, atol=0)
        # A positive tilt closes the trailing edge of the pad.
        if tilt[k] > 0:
            assert field[-1].mean() < field[0].mean()
