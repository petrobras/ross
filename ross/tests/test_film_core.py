"""Verification of the fluid-film core ``ross.bearings.film``.

Condensation against its closed-form low- and high-frequency limits, the
viscosity laws against the table points of ``lubricants_dict``, and the
lubricant resolver, exponential law, hot-oil mixing and turbulence constants
against the journal-engine rules they replace.
"""

import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.film import (
    ConvergenceRecord,
    ConvergenceWarning,
    Lubricant,
    blas_single_thread,
    condense,
    condense_limit,
    condense_static,
    dynamic_matrix,
    exponential_viscosity,
    lubricant_data,
)
from ross.bearings.film.lubricant import LUBRICANT_KEYS
from ross.bearings.lubricants import lubricants_dict
from ross.units import Q_


def _random_pads(rng, n_pads=3, inertia=False, physical=False):
    """Per-pad K5, C5 (and M5) on (z, alpha, beta, a_k, b_k) with K3-like structure.

    ``K3`` is non-symmetric with a positive definite symmetric part, ``C3`` is
    symmetric positive definite, both mapped by the pad matrix ``G_k``. With
    ``physical=True`` the magnitudes are those of a thrust pad (K ~ 1e9 N/m,
    C ~ 1e7 N s/m); otherwise they are of order one, so that ``w = 1e8`` is far
    beyond every transition frequency.
    """
    rp = 0.08
    K5, C5, M5 = [], [], []
    for k in range(n_pads):
        psi = 2 * np.pi * k / n_pads + 0.3
        G = np.array(
            [
                [1.0, rp * np.sin(psi), -rp * np.cos(psi), 0.0, 0.0],
                [0.0, np.cos(psi), np.sin(psi), 1.0, 0.0],
                [0.0, np.sin(psi), -np.cos(psi), 0.0, 1.0],
            ]
        )
        A = rng.normal(size=(3, 3))
        sym = A @ A.T + 3 * np.eye(3)
        B = rng.normal(size=(3, 3))
        skew = 0.3 * (B - B.T)
        B = rng.normal(size=(3, 3))
        spd = B @ B.T + 3 * np.eye(3)
        if physical:
            scale = np.diag([1.0, 30.0, 30.0])
            K3 = scale @ (sym + skew) @ scale * 1e9
            C3 = scale @ spd @ scale * 1e7
        else:
            K3, C3 = sym + skew, spd
        K5.append(G.T @ K3 @ G)
        C5.append(G.T @ C3 @ G)
        M = np.zeros((5, 5))
        if inertia:
            Bm = rng.normal(size=(2, 2))
            M[3:, 3:] = Bm @ Bm.T + np.eye(2)
            if physical:
                M[3:, 3:] *= 0.05
        M5.append(M)
    return K5, C5, M5


def test_condensation_low_frequency_limit():
    rng = np.random.default_rng(15)
    for inertia, physical in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        K5, C5, M5 = _random_pads(rng, inertia=inertia, physical=physical)
        K0, C0 = condense_static(K5, C5, 3)
        w = 1e-6
        D = dynamic_matrix(K5, C5, M5, 3, 1j * w)
        assert np.abs(D.real - K0).max() <= 1e-8 * np.abs(K0).max()
        assert np.abs(D.imag / w - C0).max() <= 1e-8 * np.abs(C0).max()
        Kw, Cw = condense(K5, C5, M5, 3, [0.0, w])
        assert_allclose(Kw[0], K0, rtol=0, atol=0)
        assert np.abs(Cw[1] - C0).max() <= 1e-8 * np.abs(C0).max()


def test_condensation_high_frequency_limits():
    rng = np.random.default_rng(16)
    for inertia in (False, True):
        K5, C5, M5 = _random_pads(rng, inertia=inertia)
        Ki, Ci = condense_limit(K5, C5, M5 if inertia else None, 3)
        Kw, Cw = condense(K5, C5, M5, 3, 1e8)
        assert np.abs(Kw - Ki).max() <= 1e-6 * np.abs(Ki).max()
        assert np.abs(Cw - Ci).max() <= 1e-6 * np.abs(Ci).max()
        # the pad coordinates held fixed (sum of K_uu) is not the limit
        Kf = sum(K[:3, :3] for K in K5)
        if inertia:
            assert_allclose(Ci, sum(C[:3, :3] for C in C5), rtol=1e-14)
        assert np.abs(Kf - Ki).max() > 1e-3 * np.abs(Ki).max()


def test_condensation_fixed_pads_and_input_checks():
    rng = np.random.default_rng(17)
    K = [rng.normal(size=(3, 3)) for _ in range(2)]
    C = [np.eye(3) for _ in range(2)]
    Kw, Cw = condense(K, C, None, 3, [0.0, 2.0])
    assert_allclose(Kw[1], K[0] + K[1])
    assert_allclose(Cw[0], 2 * np.eye(3))
    with pytest.raises(ValueError):
        condense(K, C, None, 3, [-1.0])
    with pytest.raises(ValueError):
        condense(K, C[:1], None, 3, 1.0)


@pytest.mark.parametrize("name", ["ISOVG32", "ISOVG46", "ISOVG68", "TEST"])
@pytest.mark.parametrize("model", ["walther", "exponential"])
def test_lubricant_reproduces_table_points(name, model):
    data = lubricants_dict[name]
    oil = Lubricant.create(name, model)
    assert_allclose(
        oil.viscosity(data["temperature1"]), data["liquid_viscosity1"], rtol=1e-10
    )
    assert_allclose(
        oil.viscosity(data["temperature2"]), data["liquid_viscosity2"], rtol=1e-10
    )
    T = np.linspace(300.0, 400.0, 11)
    assert np.all(np.diff(oil.viscosity(T)) < 0)


def test_lubricant_case_g_walther_and_exponential_parity():
    # case G oil: 39.0 / 5.4 mPa s at 40 / 100 degC, rho 855
    oil = Lubricant(
        Q_(39.0, "mPa*s"), Q_(40.0, "degC"), Q_(5.4, "mPa*s"), Q_(100.0, "degC"), 855.0
    )
    assert_allclose(oil.temperature_1, 313.15)
    assert_allclose(oil.viscosity([313.15, 373.15]), [0.039, 0.0054], rtol=1e-10)
    # Walther sits below the exponential law between the table points
    exp = Lubricant(0.039, 313.15, 0.0054, 373.15, 855.0, viscosity_model="exponential")
    assert oil.viscosity(343.15) < exp.viscosity(343.15)
    # exponential law equals the journal engine's two-point law
    from ross.bearings.fluid_film.state import Lubricant as JournalLubricant

    journal = JournalLubricant(0.039, 0.0054, 313.15, 373.15, 855.0, 2090.0, 0.13)
    T = np.array([300.0, 330.0, 360.0, 390.0])
    assert_allclose(exp.viscosity(T), journal.viscosity_at(T), rtol=1e-13)
    # isoviscous lubricant
    iso = Lubricant(0.0570722, 313.15, 0.0570722, 373.15, 867.0)
    assert_allclose(iso.viscosity([300.0, 400.0]), 0.0570722, rtol=1e-12)


def test_lubricant_input_checks():
    with pytest.raises(ValueError):
        Lubricant(0.03, 313.15, 0.01, 373.15, 860.0, viscosity_model="roelands")
    with pytest.raises(ValueError):
        Lubricant(-0.03, 313.15, 0.01, 373.15, 860.0)
    with pytest.raises(ValueError, match="Walther"):
        Lubricant(1e-3, 293.15, 2e-4, 373.15, 1000.0)  # below 0.3 cSt
    with pytest.raises(ValueError, match="lubricant must be one of"):
        Lubricant.from_name("NOPE")
    with pytest.raises(TypeError):
        Lubricant.create(3.0)
    with pytest.raises(ValueError):
        Lubricant.create("ISOVG46").viscosity(-1.0)


def test_lubricant_errors_are_those_of_the_journal_bearing():
    """lubricant_data raises what the journal bearing raised before the film core."""
    from ross.bearings.fluid_film_bearing import _lubricant_properties

    def message(error_type, call):
        with pytest.raises(error_type) as info:
            call()
        return str(info.value)

    expected = {
        "NOPE": (
            ValueError,
            f"lubricant must be one of {sorted(lubricants_dict)} or a dict of "
            "properties, not 'NOPE'",
        ),
        3.0: (
            TypeError,
            "lubricant must be a lubricant name (str) or a dict of properties, "
            "not float",
        ),
    }
    for bad, (error_type, text) in expected.items():
        assert message(error_type, lambda b=bad: _lubricant_properties(b)) == text
        assert message(error_type, lambda b=bad: lubricant_data(b)) == text
        assert message(error_type, lambda b=bad: Lubricant.create(b)) == text
    assert message(
        ValueError, lambda: _lubricant_properties({"liquid_viscosity1": 0.02})
    ).startswith("lubricant is missing properties: ['temperature1'")
    assert lubricant_data("ISOVG46") is lubricants_dict["ISOVG46"]


def test_exponential_law_is_the_journal_law():
    """film.Lubricant('exponential'), the journal state and its kernel share one law."""
    from ross.bearings.film.lubricant import exponential_viscosity_jit
    from ross.bearings.fluid_film import _numba_kernels
    from ross.bearings.fluid_film.state import Lubricant as JournalLubricant

    assert _numba_kernels.viscosity_temp_jit is exponential_viscosity_jit
    assert exponential_viscosity_jit.py_func is exponential_viscosity
    T = np.linspace(290.0, 410.0, 25)
    for name in ("ISOVG32", "ISOVG46", "ISOVG68", "TEST", "SEAWATER"):
        data = lubricants_dict[name]
        film = Lubricant.create(name, "exponential")
        journal = JournalLubricant(
            data["liquid_viscosity1"],
            data["liquid_viscosity2"],
            data["temperature1"],
            data["temperature2"],
            data["liquid_density"],
            data["liquid_specific_heat"],
            data["liquid_thermal_conductivity"],
        )
        # same arithmetic: bit-identical
        assert np.array_equal(film.viscosity(T), journal.viscosity_at(T))
        jit = [
            exponential_viscosity_jit(
                data["liquid_viscosity1"],
                data["liquid_viscosity2"],
                data["temperature1"],
                data["temperature2"],
                float(t),
            )
            for t in T
        ]
        assert_allclose(jit, film.viscosity(T), rtol=1e-14)


def _kinematic_cst(oil, temperature):
    """Kinematic viscosity in cSt, from the dynamic viscosity and density."""
    return np.asarray(oil.viscosity(temperature)) / oil.density_at(temperature) * 1e6


def test_from_datasheet_reproduces_the_datasheet_viscosities():
    oil = Lubricant.from_datasheet(Q_(32.0, "cSt"), Q_(5.4, "cSt"), 870.0)
    assert (oil.temperature_1, oil.temperature_2) == (313.15, 373.15)
    assert oil.viscosity_model == "walther"
    # ISO 3104 datasheet values, 40 and 100 C
    assert_allclose(_kinematic_cst(oil, 313.15), 32.0, rtol=1e-12)
    assert_allclose(_kinematic_cst(oil, 373.15), 5.4, rtol=1e-12)
    # SI input (m^2/s) and the exponential law give the same two points
    exp = Lubricant.from_datasheet(32e-6, 5.4e-6, 870.0, viscosity_model="exponential")
    assert_allclose(_kinematic_cst(exp, [313.15, 373.15]), [32.0, 5.4], rtol=1e-12)
    for bad in (
        dict(nu40=0.0, nu100=5.4e-6, rho15=870.0),
        dict(nu40=32e-6, nu100=np.nan, rho15=870.0),
        dict(nu40=32e-6, nu100=5.4e-6, rho15=-1.0),
    ):
        with pytest.raises(ValueError, match="positive and finite"):
            Lubricant.from_datasheet(**bad)


@pytest.mark.parametrize(
    "nu40, nu100, rho15", [(32e-6, 5.4e-6, 870.0), (320e-6, 25e-6, 900.0)]
)
def test_walther_kernel_equals_the_lubricant_law(nu40, nu100, rho15):
    from ross.bearings.film.lubricant import walther_viscosity_jit
    from ross.bearings.fluid_film import _numba_kernels
    from ross.bearings.fluid_film.state import Lubricant as JournalLubricant

    oil = Lubricant.from_datasheet(nu40, nu100, rho15)
    assert oil.volumetric_expansion > 0.0  # the density term is exercised
    T = np.linspace(280.0, 420.0, 57)
    args = (
        oil.walther_a,
        oil.walther_b,
        oil.density,
        oil.volumetric_expansion,
        oil.temperature_1,
    )
    jit = np.array([walther_viscosity_jit(*args, float(t)) for t in T])
    # numpy's log10 and the libm log10 of numba differ by one ulp on some
    # temperatures; the double exponential multiplies that 2.2e-16 by
    # ln(10)^2 log10(nu + 0.7) b log10(T) / (1 - 0.7 / (nu + 0.7)), about 150
    # at 280 K for the 320 cSt oil, hence 5e-14
    assert_allclose(jit, oil.viscosity(T), rtol=5e-14, atol=0.0)
    # the journal state (numpy) and its kernel dispatch (model 1)
    journal = JournalLubricant(
        oil.viscosity_1,
        oil.viscosity_2,
        oil.temperature_1,
        oil.temperature_2,
        oil.density,
        oil.specific_heat,
        oil.thermal_conductivity,
        model=1,
        walther_a=oil.walther_a,
        walther_b=oil.walther_b,
        walther_density=oil.density,
        walther_expansion=oil.volumetric_expansion,
    )
    assert np.array_equal(journal.viscosity_at(T), oil.viscosity(T))
    kernel = [
        _numba_kernels.viscosity_law_jit(
            1,
            oil.viscosity_1,
            oil.viscosity_2,
            oil.temperature_1,
            oil.temperature_2,
            *args[:4],
            float(t),
        )
        for t in T
    ]
    assert_allclose(kernel, oil.viscosity(T), rtol=5e-14, atol=0.0)


@pytest.mark.parametrize(
    "oil",
    [
        Lubricant.from_datasheet(32e-6, 5.4e-6, 870.0),
        Lubricant.from_datasheet(220e-6, 19e-6, 895.0),
    ],
    ids=["VG32", "VG220"],
)
def test_walther_chart_is_a_straight_line(oil):
    # ASTM D341 chart: log10 log10(nu + 0.7) linear in log10 T (nu in cSt)
    T = np.array([313.15, 343.15, 373.15])
    y = np.log10(np.log10(_kinematic_cst(oil, T) + 0.7))
    x = np.log10(T)
    middle = y[0] + (y[2] - y[0]) * (x[1] - x[0]) / (x[2] - x[0])
    assert_allclose(y[1], middle, rtol=0.0, atol=1e-13)


def test_cragoe_against_the_publication():
    from ross.bearings.film.lubricant import (
        cragoe_specific_heat,
        cragoe_thermal_conductivity,
    )

    # Cragoe (1929), Bureau of Standards Misc. Publ. 97, in the printed units:
    # equation (5), p. 26, c = (0.388 + 0.00045 t_F) / sqrt(d) Btu/(lb F), and
    # equation (4), p. 24, K = 0.813 [1 - 0.0003 (t_F - 32)] / d
    # Btu in/(h ft^2 F); d is the specific gravity at 60/60 F
    btu_per_lb_f = 4186.8  # J/(kg K), international-table calorie
    btu_in = 1055.05585262 * 0.0254 / (3600.0 * 0.3048**2 * 5.0 / 9.0)  # W/(m K)
    for d in (0.78, 0.87, 0.95):
        for celsius in (0.0, 40.0, 100.0, 200.0):
            t_f = 1.8 * celsius + 32.0
            c = (0.388 + 0.00045 * t_f) / np.sqrt(d) * btu_per_lb_f
            k = 0.813 * (1.0 - 0.0003 * (t_f - 32.0)) / d * btu_in
            # the SI constants 1684, 3.39 and 0.1172 are rounded conversions:
            # 5e-4 covers the rounding, well below the stated 5 % and 10 %
            assert_allclose(cragoe_specific_heat(d, celsius + 273.15), c, rtol=5e-4)
            assert_allclose(
                cragoe_thermal_conductivity(d, celsius + 273.15), k, rtol=5e-4
            )


def test_cragoe_reproduces_the_isovg32_table():
    # ross.bearings.lubricants ISOVG32 at 40 C: c_p 1948.80 and k 0.13126,
    # consistent with the same correlations at d = 0.8718
    table = lubricants_dict["ISOVG32"]
    rho = table["liquid_density"]
    nu40 = table["liquid_viscosity1"] / rho
    nu100 = table["liquid_viscosity2"] / rho
    oil = Lubricant.from_datasheet(nu40, nu100, 0.8718 * 999.0)
    assert_allclose(
        oil.specific_heat_at(313.15), table["liquid_specific_heat"], rtol=1e-3
    )
    assert_allclose(
        oil.thermal_conductivity_at(313.15),
        table["liquid_thermal_conductivity"],
        rtol=5e-3,
    )
    # the slope of c_p is Cragoe's; an explicit c_p is a constant
    assert_allclose(oil.specific_heat_slope, 3.39 / np.sqrt(0.8718))
    fixed = Lubricant.from_datasheet(nu40, nu100, 871.0, specific_heat=2000.0)
    assert fixed.specific_heat_at(373.15) == 2000.0


def test_api_density():
    from ross.bearings.film.lubricant import api_density

    rho15 = 870.0
    assert api_density(rho15, 288.15) == rho15
    assert api_density(rho15, Q_(15.0, "degC")) == rho15
    # by hand at 40 C: a15 = 0.6278 / 870, dT = 25 K
    a15 = 0.6278 / 870.0
    assert_allclose(
        api_density(rho15, 313.15),
        870.0 * np.exp(-a15 * 25.0 * (1.0 + 0.8 * a15 * 25.0)),
        rtol=1e-14,
    )
    oil = Lubricant.from_datasheet(32e-6, 5.4e-6, rho15)
    T = np.linspace(288.15, 393.15, 106)
    linear = oil.density_at(T)
    assert np.max(np.abs(linear / api_density(rho15, T) - 1.0)) < 3e-3
    assert_allclose(oil.density, api_density(rho15, 313.15), rtol=1e-15)


def test_lubricant_to_dict_round_trip():
    for oil in (
        Lubricant.from_datasheet(46e-6, 6.8e-6, 865.0),
        Lubricant.from_name("ISOVG46", "exponential"),
        Lubricant(0.03, 313.15, 0.006, 373.15, 860.0),  # no c_p, no k
    ):
        data = oil.to_dict()
        assert data["viscosity_model"] == oil.viscosity_model
        assert set(LUBRICANT_KEYS) <= set(data)
        back = Lubricant.from_dict(data, None)
        assert repr(back) == repr(oil).replace(f"name={oil.name!r}", "name=None")
        T = np.linspace(290.0, 410.0, 13)
        assert np.array_equal(back.viscosity(T), oil.viscosity(T))
    # from_dict still defaults to Walther for a table entry
    assert Lubricant.from_dict(lubricants_dict["ISOVG46"]).viscosity_model == "walther"


def test_journal_state_lubricant_keeps_the_seven_argument_call():
    from ross.bearings.fluid_film.state import Lubricant as JournalLubricant

    oil = JournalLubricant(0.039, 0.0054, 313.15, 373.15, 855.0, 2090.0, 0.13)
    assert oil.model == 0
    T = np.array([300.0, 340.0, 380.0])
    assert np.array_equal(
        oil.viscosity_at(T), exponential_viscosity(0.039, 0.0054, 313.15, 373.15, T)
    )
    with pytest.raises(ValueError, match="model must be 0"):
        JournalLubricant(0.039, 0.0054, 313.15, 373.15, 855.0, 2090.0, 0.13, model=2)
    # isoviscous limit of the Walther law: b = 0 gives the table viscosity
    iso = Lubricant(0.03, 313.15, 0.03, 373.15, 860.0)
    walther = JournalLubricant(
        0.03,
        0.03,
        313.15,
        373.15,
        860.0,
        2000.0,
        0.13,
        model=1,
        walther_a=iso.walther_a,
        walther_b=iso.walther_b,
        walther_density=860.0,
    )
    assert iso.walther_b == 0.0
    assert_allclose(walther.viscosity_at(T), 0.03, rtol=1e-12)


def test_convergence_record_warns():
    rec = ConvergenceRecord("x", 3, [1.0, 0.5], False, "max_iterations")
    with pytest.warns(ConvergenceWarning, match="x stopped after 3"):
        rec.warn()
    assert issubclass(ConvergenceWarning, RuntimeWarning)


def test_blas_single_thread_with_threadpoolctl():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    ctx = blas_single_thread()
    assert isinstance(ctx, threadpoolctl.threadpool_limits)
    with ctx:
        for info in threadpoolctl.threadpool_info():
            if info["user_api"] == "blas":
                assert info["num_threads"] == 1
    ctx.restore_original_limits()


def test_blas_single_thread_without_threadpoolctl(monkeypatch):
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)  # import raises ImportError
    ctx = blas_single_thread()
    with ctx:
        x = np.ones(3) @ np.ones(3)
    assert x == 3.0


def test_film_does_not_import_fluid_film():
    import ast
    from pathlib import Path

    import ross.bearings.film

    film_dir = Path(ross.bearings.film.__file__).parent
    for path in film_dir.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any("fluid_film" in n for n in names), path


def test_importing_the_core_compiles_nothing():
    import subprocess

    code = (
        "import ross.bearings.film, ross.bearings.film.lubricant as m;"
        "print(len(m.exponential_viscosity_jit.signatures))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0"


def test_importing_ross_does_not_load_the_core():
    import subprocess

    code = (
        "import sys, ross;"
        "print(any(m.startswith('ross.bearings.film') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_turbulence_constants_are_the_journal_values():
    """The journal engine uses the film-core constants, at the values it had."""
    from ross.bearings.film import turbulence
    from ross.bearings.fluid_film import thermal as journal_thermal
    from ross.bearings.fluid_film_bearing import FluidFilmBearing

    class _Stub:
        lubricant_properties = {}
        initial_position = (0.0, 0.0)
        probes = []
        excitation_ratio = 1.0

        def __getattr__(self, name):
            return 0

    inputs = FluidFilmBearing._engine_inputs(_Stub(), 100.0)
    assert inputs["reichardt_kappa"] == turbulence.REICHARDT_KAPPA == 0.4
    assert inputs["reichardt_delta"] == turbulence.REICHARDT_DELTA == 8.8
    assert journal_thermal.PR_TURB == turbulence.TURBULENT_PRANDTL == 0.769


def _journal_carryover_mixing(n, operating_type, q_in, q_c, t_sump, t_out):
    """Pad inlet temperatures as the journal engine computed them before the film core."""
    t_in = np.zeros(n)
    for p in range(n):
        u = n - 1 if p == 0 else p - 1
        if q_c[u] > q_in[p]:
            t_in[p] = t_out[u]
        else:
            t_in[p] = (q_c[u] * t_out[u] + (q_in[p] - q_c[u]) * t_sump) / q_in[p]
    if operating_type in ("axial_flow", "oil_ring_lubricated"):
        t_in[:] = t_sump
    return t_in


@pytest.mark.parametrize(
    "operating_type", ["regular_flooded", "axial_flow", "oil_ring_lubricated"]
)
def test_carryover_mixing_matches_the_journal_rule(operating_type):
    """film.mixing reproduces the journal hot-oil rule; the weights are its form."""
    from ross.bearings.film import mixing
    from ross.bearings.fluid_film import thd

    assert thd.temp_mixing_carryover is mixing.temp_mixing_carryover
    assert mixing.upstream_pads(3).tolist() == [2, 0, 1]
    rng = np.random.default_rng(3)
    lam, t_f = 0.8, 315.0
    for n in (1, 2, 5):
        q_in = rng.uniform(1.0, 2.0, n)
        q_tr = rng.uniform(0.5, 2.6, n)  # some carry-overs above the inflow
        q_c = lam * q_tr
        t_out = rng.uniform(330.0, 360.0, n)
        t_in = mixing.temp_mixing_carryover(
            n, operating_type, q_in, q_c, t_f, t_out, np.zeros(n)
        )
        # same arithmetic: bit-identical
        assert np.array_equal(
            t_in, _journal_carryover_mixing(n, operating_type, q_in, q_c, t_f, t_out)
        )
        w = mixing.carryover_weights(q_in, q_c)
        assert np.all((w > 0.0) & (w <= 1.0))
        assert_allclose(mixing.carryover_flows(q_in, q_c), w * q_in, rtol=1e-14)
        if operating_type == "regular_flooded":
            up = mixing.upstream_pads(n)
            assert_allclose(t_in, w * t_out[up] + (1.0 - w) * t_f, rtol=1e-14)
    # no inflow: no hot oil
    assert mixing.carryover_weights([0.0, 1.0], [1.0, 1.0]).tolist() == [0.0, 1.0]
