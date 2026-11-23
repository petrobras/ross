"""Unreduced pad matrices and the exact frequency reduction of journal bearings.

References used here: copies of the condensation this change replaces,
numerical quadrature of the pad sector, the closed-form
limits of :mod:`ross.bearings.film.condense` and Schur complements assembled
by hand.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import dblquad

from ross.bearings.fluid_film import driver
from ross.bearings.fluid_film.diagnostics import LoopMonitor
from ross.bearings.fluid_film.padmatrices import (
    PadMatrices,
    SingularPadStiffness,
    pad_inertia_about_pivot,
)
from ross.bearings.fluid_film_bearing import FluidFilmBearing
from ross.bearings.tilting_pad import TiltingPad
from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture
from ross.units import Q_

# the solver fixtures pass 1 atm absolute as the ambient pressure
pytestmark = pytest.mark.filterwarnings(
    "ignore:operating_type='regular_flooded':UserWarning"
)

DATA = Path(__file__).parent / "data" / "fluid_film"
WHIRL_RATIOS = (0.1, 1.0, 3.0, 10.0)


# Copies of ``dynamic_reduction`` and ``dynamic_reduction_pivot`` as they were
# in ``ross/bearings/fluid_film/coefficients.py`` before this change, with the
# non-pivoting complex inverse they used. The only change is the optional
# ``inertia``, ``mass`` and ``coupling`` arguments that replace the
# centroidal pad inertia ``m (t^2 + l^2) / 12`` and mass ``l t L rho`` of the
# previous code (``coupling`` fills the tilt-pivot entries of the mass matrix,
# which the previous code left at 0).
def _previous_masses(total_pads, pads, pad_density, inertia, mass):
    mass_pad = np.zeros(total_pads, dtype=np.float64)
    ip = np.zeros(total_pads, dtype=np.float64)
    for k in range(total_pads):
        mass_pad[k] = (
            pads.pad_length[k] * pads.pad_thickness * pads.axial_length[k] * pad_density
        )
        ip[k] = mass_pad[k] * (pads.pad_thickness**2 + pads.pad_length[k] ** 2) / 12.0
    if inertia is not None:
        ip = np.asarray(inertia, dtype=float)
    if mass is not None:
        mass_pad = np.asarray(mass, dtype=float)
    return mass_pad, ip


def base_dynamic_reduction(
    total_pads,
    stiffness,
    damping_block,
    pads,
    pad_density,
    excit_rad,
    k_rotate,
    inertia=None,
):
    s = complex(0.0, excit_rad)
    t = s * s
    _, ip = _previous_masses(total_pads, pads, pad_density, inertia, None)
    kuu = np.array(
        [[stiffness.xx, stiffness.xy], [stiffness.yx, stiffness.yy]], dtype=complex
    )
    cuu = np.array(
        [[damping_block.xx, damping_block.xy], [damping_block.yx, damping_block.yy]],
        dtype=complex,
    )
    kudelta = np.zeros((2, total_pads), dtype=complex)
    cudelta = np.zeros((2, total_pads), dtype=complex)
    kdeltau = np.zeros((total_pads, 2), dtype=complex)
    cdeltau = np.zeros((total_pads, 2), dtype=complex)
    kdeldel = np.zeros((total_pads, total_pads), dtype=complex)
    cdeldel = np.zeros((total_pads, total_pads), dtype=complex)
    mass_ip = np.zeros((total_pads, total_pads), dtype=complex)
    for k in range(total_pads):
        kudelta[0, k] = stiffness.xdelta[k]
        kudelta[1, k] = stiffness.ydelta[k]
        cudelta[0, k] = damping_block.xdelta[k]
        cudelta[1, k] = damping_block.ydelta[k]
        kdeltau[k, 0] = stiffness.deltax[k]
        kdeltau[k, 1] = stiffness.deltay[k]
        cdeltau[k, 0] = damping_block.deltax[k]
        cdeltau[k, 1] = damping_block.deltay[k]
        kdeldel[k, k] = stiffness.deltadelta[k] + k_rotate[k]
        cdeldel[k, k] = damping_block.deltadelta[k]
        mass_ip[k, k] = ip[k]
    a1 = s * cudelta + kudelta
    a3 = t * mass_ip + s * cdeldel + kdeldel
    a3_inverse = np.zeros((total_pads, total_pads), dtype=complex)
    for k in range(total_pads):
        a3_inverse[k, k] = 1.0 / a3[k, k]
    a4 = s * cdeltau + kdeltau
    ds = s * cuu + kuu - (a1 @ a3_inverse) @ a4
    return ds.real, ds.imag / excit_rad


def base_dynamic_reduction_pivot(
    total_pads,
    pads,
    pad_density,
    stiffness,
    damping_block,
    k_pivot,
    excit_rad,
    k_rotate,
    inertia=None,
    mass=None,
    coupling=None,
):
    n = 2 * total_pads
    mass_pad, ip = _previous_masses(total_pads, pads, pad_density, inertia, mass)
    kuu = np.array(
        [[stiffness.xx, stiffness.xy], [stiffness.yx, stiffness.yy]], dtype=complex
    )
    cuu = np.array(
        [[damping_block.xx, damping_block.xy], [damping_block.yx, damping_block.yy]],
        dtype=complex,
    )
    kutp = np.zeros((2, n), dtype=complex)
    cutp = np.zeros((2, n), dtype=complex)
    ktpu = np.zeros((n, 2), dtype=complex)
    ctpu = np.zeros((n, 2), dtype=complex)
    ktptp = np.zeros((n, n), dtype=complex)
    ctptp = np.zeros((n, n), dtype=complex)
    mass_matrix = np.zeros((n, n), dtype=complex)
    for k in range(total_pads):
        theta_pi = pads.leading_angle_rad[k] + pads.x_pivot_rad[k]
        c = np.cos(theta_pi)
        sn = np.sin(theta_pi)
        for blk, u_e, e_u, e_e, extra_tt, extra_pp in (
            (stiffness, kutp, ktpu, ktptp, k_rotate[k], k_pivot[k]),
            (damping_block, cutp, ctpu, ctptp, 0.0, 0.0),
        ):
            u_e[0, k] = blk.xdelta[k]
            u_e[1, k] = blk.ydelta[k]
            u_e[0, total_pads + k] = -blk.xxi[k] * c - blk.xyi[k] * sn
            u_e[1, total_pads + k] = -blk.yxi[k] * c - blk.yyi[k] * sn
            e_u[k, 0] = blk.deltax[k]
            e_u[k, 1] = blk.deltay[k]
            e_u[total_pads + k, 0] = -blk.xxi[k] * c - blk.yxi[k] * sn
            e_u[total_pads + k, 1] = -blk.xyi[k] * c - blk.yyi[k] * sn
            e_e[k, k] = blk.deltadelta[k] + extra_tt
            e_e[k, total_pads + k] = -blk.deltax[k] * c - blk.deltay[k] * sn
            e_e[total_pads + k, k] = -blk.xdelta[k] * c - blk.ydelta[k] * sn
            e_e[total_pads + k, total_pads + k] = (
                c * (blk.xxi[k] * c + blk.xyi[k] * sn)
                + sn * (blk.yxi[k] * c + blk.yyi[k] * sn)
                + extra_pp
            )
        mass_matrix[k, k] = ip[k]
        mass_matrix[total_pads + k, total_pads + k] = mass_pad[k]
        if coupling is not None:
            mass_matrix[k, total_pads + k] = coupling[k]
            mass_matrix[total_pads + k, k] = coupling[k]
    s = complex(0.0, excit_rad)
    t = s * s
    a1 = s * cutp + kutp
    a3 = t * mass_matrix + s * ctptp + ktptp
    a4 = s * ctpu + ktpu
    ds = s * cuu + kuu - (a1 @ _base_inverse_complex(a3)) @ a4
    return ds.real, ds.imag / excit_rad


def _base_inverse_complex(a):
    n = a.shape[0]
    lu = np.array(a, dtype=complex, copy=True)
    for i in range(n - 1):
        for j in range(i + 1, n):
            xmult = lu[j, i] / lu[i, i]
            for k in range(i + 1, n):
                lu[j, k] = lu[j, k] - xmult * lu[i, k]
            lu[j, i] = xmult
    y = np.zeros((n, n), dtype=complex)
    for col in range(n):
        b = np.zeros(n, dtype=complex)
        b[col] = 1.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                b[j] = b[j] - lu[j, i] * b[i]
        x = np.zeros(n, dtype=complex)
        x[n - 1] = b[n - 1] / lu[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            ssum = b[i]
            for j in range(i + 1, n):
                ssum = ssum - lu[i, j] * x[j]
            x[i] = ssum / lu[i, i]
        y[:, col] = x
    return y


def _solve_capturing(case, **overrides):
    """Run a fixture case and keep the engine state and blocks of the reduction."""
    inputs = json.loads((DATA / f"{case}.json").read_text())["inputs"]
    inputs.update(overrides)
    captured = {}
    original = driver._pad_matrices

    def capture(g, stiffness, damping_block):
        captured.update(g=g, stiffness=stiffness, damping=damping_block)
        return original(g, stiffness, damping_block)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(driver, "_pad_matrices", capture)
        out = driver.run_case(**inputs, pad_matrices=True)
    captured["out"] = out
    captured["pads"] = out["pad_matrices"][0]
    captured["speed"] = float(inputs["frequency"][0])
    return captured


ENGINE_CASES = {
    "rigid": ("tilt_5pad_isoviscous", {}),
    "rigid_offset": ("tilt_5pad_isoviscous", {"offset": [0.6] * 5}),
    "flexible": ("tilt_5pad_deform5", {}),
    "flexible_offset": ("tilt_5pad_deform5", {"offset": [0.6] * 5}),
}


@pytest.fixture(scope="module", params=sorted(ENGINE_CASES))
def engine_case(request):
    case, overrides = ENGINE_CASES[request.param]
    captured = _solve_capturing(case, **overrides)
    captured["name"] = request.param
    return captured


def _previous_reduction(captured, w, inertia=None, mass=None, coupling=None):
    g = captured["g"]
    total_pads = g["total_pads"]
    if g["deform_type"] in driver.PIVOT_FLEX_DEFORM_TYPES:
        return base_dynamic_reduction_pivot(
            total_pads,
            g["pads"],
            g["pad_density"],
            captured["stiffness"],
            captured["damping"],
            g["k_pivot"],
            w,
            g["k_rotate"],
            inertia=inertia,
            mass=mass,
            coupling=coupling,
        )
    return base_dynamic_reduction(
        total_pads,
        captured["stiffness"],
        captured["damping"],
        g["pads"],
        g["pad_density"],
        w,
        g["k_rotate"],
        inertia=inertia,
    )


def _assert_same_matrix(new, reference, label):
    """Agreement to 1e-10 of the largest entry of the reference matrix."""
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(new - reference)) <= 1e-10 * scale, label


@pytest.mark.parametrize("ratio", WHIRL_RATIOS)
def test_condensation_reproduces_the_previous_reduction(engine_case, ratio):
    """``film.condense`` equals the previous reduction (old inertia).

    Reference: copies of the previous ``dynamic_reduction`` /
    ``dynamic_reduction_pivot`` (non-pivoting complex inverse included), fed
    with the engine's blocks of the same solve. The pad matrices are built
    with the previous centroidal inertia ``m (t^2 + l^2) / 12``, mass
    ``l t L rho`` and no tilt-pivot coupling, so both sides condense the same
    model; they differ only in the order of the arithmetic.
    """
    g = engine_case["g"]
    pads = g["pads"]
    total_pads = g["total_pads"]
    w = ratio * engine_case["speed"]
    mass_old, inertia_old = _previous_masses(
        total_pads, pads, g["pad_density"], None, None
    )
    flexible = g["deform_type"] in driver.PIVOT_FLEX_DEFORM_TYPES
    matrices = PadMatrices.from_blocks(
        engine_case["stiffness"],
        engine_case["damping"],
        pad_mass=mass_old,
        pad_inertia=inertia_old,
        pad_coupling=0.0,
        k_rotate=g["k_rotate"],
        k_pivot=g["k_pivot"] if flexible else None,
        pivot_direction=pads.leading_angle_rad + pads.x_pivot_rad,
    )
    K_new, C_new = matrices.condense(w)
    K_old, C_old = _previous_reduction(engine_case, w)
    _assert_same_matrix(K_new, K_old, f"K {engine_case['name']} {ratio}")
    _assert_same_matrix(C_new, C_old, f"C {engine_case['name']} {ratio}")


@pytest.mark.parametrize("ratio", WHIRL_RATIOS)
def test_engine_coefficients_are_the_previous_reduction_with_the_new_inertia(
    engine_case, ratio
):
    """The regenerated tilting-pad coefficients.

    Reference: the previous reduction (copy above) fed with the engine's
    blocks and with the pad mass, inertia about the pivot and tilt-pivot
    coupling of :func:`pad_inertia_about_pivot` (checked against quadrature
    below). The engine now reports exactly this reduction; the tilting-pad
    fixtures were regenerated from it.
    """
    g = engine_case["g"]
    pads = g["pads"]
    mass, inertia, coupling = pad_inertia_about_pivot(
        pads.journal_radius,
        g["cp"],
        pads.pad_thickness,
        pads.arc_length_rad,
        pads.axial_length,
        g["pad_density"],
        g["offset"],
    )
    w = ratio * engine_case["speed"]
    K_old, C_old = _previous_reduction(
        engine_case, w, inertia=inertia, mass=mass, coupling=coupling
    )
    K_new, C_new = engine_case["pads"].condense(w)
    _assert_same_matrix(K_new, K_old, f"K {engine_case['name']} {ratio}")
    _assert_same_matrix(C_new, C_old, f"C {engine_case['name']} {ratio}")
    if ratio == 1.0:
        out = engine_case["out"]
        reported_K = np.array(
            [[out["kxx"][0], out["kxy"][0]], [out["kyx"][0], out["kyy"][0]]]
        )
        reported_C = np.array(
            [[out["cxx"][0], out["cxy"][0]], [out["cyx"][0], out["cyy"][0]]]
        )
        _assert_same_matrix(reported_K, K_old, "reported K")
        _assert_same_matrix(reported_C, C_old, "reported C")


def test_the_inertia_change_is_visible_at_high_whirl(
    engine_case,
):
    """The previous reduction with the old and with the new inertia differ.

    Guard for the two tests above: the inertia change is large enough on these
    thick pads (``t / R = 0.75``) to be seen at 10X, so they compare
    different models.
    """
    g = engine_case["g"]
    pads = g["pads"]
    mass, inertia, coupling = pad_inertia_about_pivot(
        pads.journal_radius,
        g["cp"],
        pads.pad_thickness,
        pads.arc_length_rad,
        pads.axial_length,
        g["pad_density"],
        g["offset"],
    )
    w = 10.0 * engine_case["speed"]
    K_old, _ = _previous_reduction(engine_case, w)
    K_new, _ = _previous_reduction(
        engine_case, w, inertia=inertia, mass=mass, coupling=coupling
    )
    assert np.max(np.abs(K_new - K_old)) > 1e-3 * np.max(np.abs(K_old))


# Pad sectors: example 5-pad bearing (D 101.6 mm, t 12.7 mm, 60 deg) and the
# thick pads of the solver fixtures (D 400 mm, t 150 mm), several offsets.
SECTORS = [
    dict(R=50.8e-3, cp=149.8e-6, t=12.7e-3, beta=np.radians(60), L=50.8e-3, rho=7830),
    dict(R=0.2, cp=277.9e-6, t=0.1496, beta=np.radians(60), L=0.1, rho=7835.6),
    dict(R=0.15, cp=0.0, t=0.03, beta=np.radians(80), L=0.075, rho=8900),
]
OFFSETS = (0.5, 0.6, 0.65, 0.3, 1.0)


def _quadrature(f, sector):
    """Integral of ``f(r, theta) dm`` over the sector centred on theta = 0."""
    s = sector
    value, _ = dblquad(
        lambda r, th: f(r, th) * s["rho"] * s["L"] * r,
        -s["beta"] / 2,
        s["beta"] / 2,
        s["R"] + s["cp"],
        s["R"] + s["cp"] + s["t"],
        epsabs=0.0,
        epsrel=1e-13,
    )
    return value


def _pivot(sector, offset):
    theta_p = (offset - 0.5) * sector["beta"]
    r_p = sector["R"] + sector["t"]
    return theta_p, r_p * np.cos(theta_p), r_p * np.sin(theta_p)


def _inertia(sector, offset):
    s = sector
    return pad_inertia_about_pivot(
        s["R"], s["cp"], s["t"], s["beta"], s["L"], s["rho"], offset
    )


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("sector", SECTORS)
def test_pad_inertia_about_the_pivot_against_quadrature(sector, offset):
    """Mass and ``I_P = int |r - r_P|^2 dm`` of the annular sector.

    Reference: ``scipy.integrate.dblquad`` over the sector
    ``[R + C_p, R + C_p + t] x [-beta/2, beta/2]`` with the pivot at radius
    ``R + t`` and angle ``(offset - 0.5) beta``.
    """
    _, px, py = _pivot(sector, offset)
    mass, inertia, _ = _inertia(sector, offset)
    mass_q = _quadrature(lambda r, th: 1.0, sector)
    inertia_q = _quadrature(
        lambda r, th: (r * np.cos(th) - px) ** 2 + (r * np.sin(th) - py) ** 2, sector
    )
    assert_allclose(mass[0], mass_q, rtol=1e-10)
    assert_allclose(inertia[0], inertia_q, rtol=1e-10)


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("sector", SECTORS)
def test_pad_kinetic_energy_against_quadrature(sector, offset):
    """The mass block ``[[I_P, c], [c, m]]`` gives the pad's kinetic energy.

    Reference: ``1/2 int |v_dot e_P + delta_dot z x (r - r_P)|^2 dm`` by
    quadrature for random tilt and pivot velocities (``delta`` positive
    counter-clockwise about the pivot, ``v`` outwards along ``e_P``).
    """
    theta_p, px, py = _pivot(sector, offset)
    mass, inertia, coupling = _inertia(sector, offset)
    block = np.array([[inertia[0], coupling[0]], [coupling[0], mass[0]]])
    rng = np.random.default_rng(16)
    for delta_dot, v_dot in rng.normal(size=(3, 2)):

        def speed_squared(r, th, delta_dot=delta_dot, v_dot=v_dot):
            vx = v_dot * np.cos(theta_p) - delta_dot * (r * np.sin(th) - py)
            vy = v_dot * np.sin(theta_p) + delta_dot * (r * np.cos(th) - px)
            return vx**2 + vy**2

        q_dot = np.array([delta_dot, v_dot])
        energy = 0.5 * q_dot @ block @ q_dot
        assert_allclose(energy, 0.5 * _quadrature(speed_squared, sector), rtol=1e-10)


@pytest.mark.parametrize("sector", SECTORS)
def test_centred_pivot_has_no_coupling(sector):
    """A pivot on the pad centre line has no tilt-translation mass coupling."""
    mass, _, coupling = _inertia(sector, 0.5)
    assert coupling[0] == 0.0
    # and an offset pivot has, with the sign of the offset
    _, _, ahead = _inertia(sector, 0.6)
    _, _, behind = _inertia(sector, 0.4)
    assert ahead[0] > 0.0
    assert_allclose(behind[0], -ahead[0], rtol=1e-12)


def _pad_matrices_by_hand(pads, s):
    """``sum_k Z_uu - Z_ue Z_ee^-1 Z_eu`` over every pad, assembled directly."""
    total = np.zeros((2, 2), dtype=complex)
    for K, C, M in zip(pads.K, pads.C, pads.M, strict=True):
        Z = K + s * C + s**2 * M
        total += Z[:2, :2]
        if Z.shape[0] > 2:
            total -= Z[:2, 2:] @ np.linalg.inv(Z[2:, 2:]) @ Z[2:, :2]
    return total


def test_impedance_is_the_schur_complement(engine_case):
    """``impedance(s)`` at complex ``s`` and on the imaginary axis.

    Reference: the Schur complement assembled pad by pad with
    ``numpy.linalg.inv`` (every pad included, unloaded ones too); on the
    imaginary axis, ``K(w) = Re D(i w)`` and ``C(w) = Im D(i w) / w`` of
    :meth:`PadMatrices.condense`.
    """
    pads = engine_case["pads"]
    speed = engine_case["speed"]
    for s in (complex(-0.05, 1.0) * speed, complex(-0.3, 2.5) * speed, 0.7j * speed):
        expected = _pad_matrices_by_hand(pads, s)
        assert_allclose(
            pads.impedance(s), expected, rtol=0, atol=1e-10 * np.max(np.abs(expected))
        )
    for ratio in WHIRL_RATIOS:
        w = ratio * speed
        D = pads.impedance(1j * w)
        K, C = pads.condense(w)
        assert_allclose(K, D.real, rtol=1e-12, atol=0)
        assert_allclose(C, D.imag / w, rtol=1e-12, atol=0)


def test_high_frequency_limit(engine_case):
    """The condensed coefficients approach ``condense_limit`` (pad inertia on).

    Reference: the closed-form limit ``K_uu - C_ue M_ee^-1 C_eu``, ``C_uu``.
    The gap falls as ``1/w^2`` once ``w`` is well above the pad's own
    damping-to-inertia frequency ``C_ee / M_ee`` (about 5000X for the
    pivot translation of the flexible-pivot fixture).
    """
    pads = engine_case["pads"]
    K_inf, C_inf = pads.condense_limit()
    gaps = []
    for ratio in (1e5, 1e6, 1e7):
        K, C = pads.condense(ratio * engine_case["speed"])
        gaps.append(
            max(
                np.max(np.abs(K - K_inf)) / np.max(np.abs(K_inf)),
                np.max(np.abs(C - C_inf)) / np.max(np.abs(C_inf)),
            )
        )
    assert gaps[2] < 1e-6
    assert gaps[2] < 0.05 * gaps[1] < 0.05**2 * gaps[0]


def _five_pad(**overrides):
    """Isoviscous 5-pad load-between-pads bearing."""
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
        fxs_load=0.0,
        fys_load=-2.8e3,
        total_ex_film=20,
        total_ez_film=10,
        total_ey_film=10,
        total_ey_pad=10,
    )
    kwargs.update(overrides)
    return kwargs


def test_kyy_falls_with_the_whirl_frequency():
    """Tilting-pad stiffness falls with whirl from 0.25X to 2X.

    Reference: the trend of Lund (1964) for tilting-pad bearings (pad
    damping and inertia make the condensed stiffness frequency dependent).
    The previous reduction (centroidal pad inertia) gave kyy 147 / 139 / 114 /
    66 MN/m at 0.25 / 0.5 / 1 / 2X (regression only).
    """
    speed = Q_(3000, "RPM").to("rad/s").m
    ratios = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    bearing = TiltingPad(**_five_pad(frequency=ratios * speed))
    kyy = np.asarray(bearing.kyy, dtype=float)[0]
    assert np.all(np.diff(kyy) < 0.0)
    assert_allclose(kyy[[0, 1, 3, 5]], [147.2e6, 139.1e6, 114.2e6, 66.0e6], rtol=0.02)


def test_static_condensation_gives_finite_damping():
    """``excitation_ratio=0`` and ``frequency=0`` give the static limit.

    Reference: ``condense_static`` of the case's pad matrices (closed form,
    no division by ``w``) and the limit of ``C(w)`` at ``w = 1e-6 * speed``.
    Before this change the damping was NaN.
    """
    static = TiltingPad(**_five_pad(excitation_ratio=0.0))
    K0, C0 = static.results.coefficients(0, frequency=0.0)
    K_static, C_static = static.results.pad_matrices(0).condense_static()
    assert np.all(K0 == K_static)
    assert np.all(C0 == C_static)
    table = np.array(
        [static.kxx[0], static.kxy[0], static.kyx[0], static.kyy[0]], dtype=float
    )
    damping = np.array(
        [static.cxx[0], static.cxy[0], static.cyx[0], static.cyy[0]], dtype=float
    )
    assert np.all(np.isfinite(damping))
    assert np.all(table == K0.ravel())
    assert np.all(damping == C0.ravel())
    speed = static.speed_range[0]
    K_small, C_small = static.results.coefficients(0, 1e-6 * speed)
    assert_allclose(K_small, K0, rtol=0, atol=1e-6 * np.max(np.abs(K0)))
    assert_allclose(C_small, C0, rtol=0, atol=1e-6 * np.max(np.abs(C0)))

    grid = TiltingPad(**_five_pad(frequency=[0.0, speed]))
    assert np.all(np.isfinite(np.asarray(grid.cyy, dtype=float)))
    assert grid.cyy[0][0] == static.cyy[0]
    assert grid.kxx[0][0] == static.kxx[0]


def test_static_condensation_of_a_fixed_geometry_bearing():
    """For fixed geometry: the damping at ``w = 0`` is the film damping.

    Reference: a fixed-geometry film has no pad coordinates, so its
    coefficients do not depend on the whirl frequency; the synchronous case
    gives the same damping. Before this change it was NaN at ``w = 0``.
    """
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", excitation_ratio=0.0)
    static = FluidFilmBearing(**kwargs)
    synchronous = FluidFilmBearing(**dict(kwargs, excitation_ratio=1.0))
    for name in ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"):
        assert_allclose(
            getattr(static, name), getattr(synchronous, name), rtol=1e-12, err_msg=name
        )
    assert all(len(labels) == 2 for labels in static.results.pad_matrices(0).labels)


def test_two_dimensional_table_solves_each_speed_once(monkeypatch):
    """One equilibrium per speed, one results entry per pair.

    Reference: 20 separate 1-D solves with ``excitation_ratio = f / speed``
    (exactly equal: the same solved state condensed at the same frequency by
    the same scalar call) and the count of ``hydrodynamics`` calls.
    """
    from ross.bearings.fluid_film import hydrodynamics

    kwargs, _ = bearing_kwargs_from_fixture("tilt_5pad_isoviscous")
    speed = float(np.atleast_1d(kwargs.pop("speed"))[0])
    kwargs.pop("excitation_ratio")
    frequency = np.linspace(0.1, 3.0, 20) * speed

    calls = []
    original = hydrodynamics.hydrodynamics

    def counting(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    monkeypatch.setattr(hydrodynamics, "hydrodynamics", counting)
    grid = FluidFilmBearing(speed=[speed], frequency=frequency, **kwargs)
    monkeypatch.setattr(hydrodynamics, "hydrodynamics", original)
    assert len(calls) == 1

    results = grid.results
    assert results.frequency.size == 20
    assert_allclose(results.whirl_frequency, frequency)
    assert np.all(results.speed_index == 0)
    for j in range(20):
        assert results.pressure_fields[j] is results.pressure_fields[0]
        assert results.outputs[j]["h_min"][0] is results.outputs[0]["h_min"][0]
        assert results.pad_matrices(0) is results._pad_matrices[j]

    names = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")
    for j, f in enumerate(frequency):
        single = FluidFilmBearing(speed=[speed], excitation_ratio=f / speed, **kwargs)
        for name in names:
            value = np.asarray(getattr(grid, name), dtype=float)[0, j]
            reference = float(getattr(single, name)[0])
            assert_allclose(value, reference, rtol=1e-9, err_msg=name)
            assert value == reference, (name, j)
        # the table condenses at (f / speed) * speed in the engine's units, which
        # can differ from f in the last bit
        K, C = results.coefficients(0, f)
        assert_allclose(K[0, 0], np.asarray(grid.kxx)[0, j], rtol=1e-12)
        assert_allclose(C[1, 1], np.asarray(grid.cyy)[0, j], rtol=1e-12)


def test_results_condense_at_any_frequency():
    """``results.coefficients`` between the table frequencies.

    Reference: a second bearing whose table holds the intermediate frequency
    (same solved state, so the values are equal to rounding).
    """
    speed = Q_(3000, "RPM").to("rad/s").m
    bearing = TiltingPad(**_five_pad(frequency=[0.5 * speed, speed]))
    reference = TiltingPad(**_five_pad(frequency=[0.75 * speed]))
    K, C = bearing.results.coefficients(0, 0.75 * speed)
    assert_allclose(K[1, 1], reference.kyy[0][0], rtol=1e-10)
    assert_allclose(C[0, 0], reference.cxx[0][0], rtol=1e-10)
    K_sync, _ = bearing.results.coefficients(0)
    assert_allclose(K_sync[0, 0], bearing.kxx[0][1], rtol=1e-12)
    with pytest.raises(IndexError, match="speed_index 1"):
        bearing.results.coefficients(1, speed)


def test_singular_pad_block_is_reported_as_an_unloaded_pivot():
    """A pad free to move has no static condensation.

    A pad coupled to the journal whose stiffness block ``K_ee`` is singular
    (zero tilt stiffness: e.g. a pad on an unloaded Hertz pivot) raises
    ``SingularPadStiffness`` at ``w = 0`` and condenses normally at ``w > 0``
    (the inertia and damping keep ``Z_ee`` regular); the driver reports the
    static case as ``"pivot_unloaded"`` with NaN coefficients.
    """
    K = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    C = np.eye(3)
    M = np.diag([0.0, 0.0, 1e-3])
    pads = PadMatrices(K=(K,), C=(C,), M=(M,), labels=(("x", "y", "tilt_0"),))
    with pytest.raises(SingularPadStiffness):
        pads.condense_static()
    with pytest.raises(SingularPadStiffness):
        pads.condense([0.0, 1.0])
    K1, C1 = pads.condense(1.0)
    assert np.all(np.isfinite(K1)) and np.all(np.isfinite(C1))

    monitor = LoopMonitor()
    reduced = driver._reduce(pads, 0.0, monitor)
    assert np.all(np.isnan(reduced))
    assert monitor.reason == "pivot_unloaded"
    monitor = LoopMonitor()
    assert np.all(np.isfinite(driver._reduce(pads, 1.0, monitor)))
    assert monitor.converged


def test_decoupled_pads_keep_the_static_limit():
    """Fully cavitated pads (zero film blocks) do not make ``w = 0`` singular.

    Reference: the static condensation of the loaded pads alone; the two
    unloaded pads of ``tilt_5pad_isoviscous`` have zero coupling, so they add
    their (zero) journal block at every frequency.
    """
    captured = _solve_capturing("tilt_5pad_isoviscous")
    pads = captured["pads"]
    unloaded = [k for k in range(5) if not np.any(pads.K[k][:2, 2:])]
    assert unloaded == [0, 1]
    loaded = PadMatrices(
        K=pads.K[2:], C=pads.C[2:], M=pads.M[2:], labels=pads.labels[2:]
    )
    K0, C0 = pads.condense_static()
    K0_loaded, C0_loaded = loaded.condense_static()
    assert_allclose(K0, K0_loaded + sum(k[:2, :2] for k in pads.K[:2]), rtol=1e-14)
    assert_allclose(C0, C0_loaded + sum(c[:2, :2] for c in pads.C[:2]), rtol=1e-14)
