import os
import pickle
from pathlib import Path
from tempfile import tempdir
import control as ct

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearing_seal_element import (
    BallBearingElement,
    BearingElement,
    BearingElement,
    BearingFluidFlow,
    CylindricalBearing,
    MagneticBearingElement,
    RollerBearingElement,
)
from ross.units import Q_


@pytest.fixture
def bearing0():
    Kxx_bearing = np.array(
        [8.5e07, 1.1e08, 1.3e08, 1.6e08, 1.8e08, 2.0e08, 2.3e08, 2.5e08, 2.6e08]
    )
    Kyy_bearing = np.array(
        [9.2e07, 1.1e08, 1.4e08, 1.6e08, 1.9e08, 2.1e08, 2.3e08, 2.5e08, 2.6e08]
    )
    Cxx_bearing = np.array(
        [226837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    Cyy_bearing = np.array(
        [235837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    wb = np.array([314.2, 418.9, 523.6, 628.3, 733.0, 837.8, 942.5, 1047.2, 1151.9])
    bearing0 = BearingElement(
        4,
        kxx=Kxx_bearing,
        kyy=Kyy_bearing,
        cxx=Cxx_bearing,
        cyy=Cyy_bearing,
        frequency=wb,
    )
    return bearing0


def test_bearing_interpol_kxx(bearing0):
    assert_allclose(bearing0.kxx_interpolated(314.2), 8.5e7)
    assert_allclose(bearing0.kxx_interpolated(1151.9), 2.6e8)


def test_bearing_interpol_kyy(bearing0):
    assert_allclose(bearing0.kyy_interpolated(314.2), 9.2e7)
    assert_allclose(bearing0.kyy_interpolated(1151.9), 2.6e8)


def test_bearing_interpol_cxx(bearing0):
    assert_allclose(bearing0.cxx_interpolated(314.2), 226837, rtol=1e5)
    assert_allclose(bearing0.cxx_interpolated(1151.9), 137973, rtol=1e5)


def test_bearing_interpol_cyy(bearing0):
    assert_allclose(bearing0.kxx_interpolated(314.2), 235837, rtol=1e5)
    assert_allclose(bearing0.kxx_interpolated(1151.9), 2.6e8, rtol=1e5)


@pytest.fixture
def bearing1():
    # using lists
    kxx_bearing = [
        8.5e07,
        1.1e08,
        1.3e08,
        1.6e08,
        1.8e08,
        2.0e08,
        2.3e08,
        2.5e08,
        2.6e08,
    ]
    kyy_bearing = np.array(
        [9.2e07, 1.1e08, 1.4e08, 1.6e08, 1.9e08, 2.1e08, 2.3e08, 2.5e08, 2.6e08]
    )
    cxx_bearing = np.array(
        [226837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    cyy_bearing = np.array(
        [235837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    mxx_bearing = np.array(
        [1e-3, 1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3, 1.7e-3, 1.8e-3]
    )
    wb = [314.2, 418.9, 523.6, 628.3, 733.0, 837.8, 942.5, 1047.2, 1151.9]
    bearing1 = BearingElement(
        4,
        kxx=kxx_bearing,
        kyy=kyy_bearing,
        cxx=cxx_bearing,
        cyy=cyy_bearing,
        mxx=mxx_bearing,
        frequency=wb,
    )
    return bearing1


def test_index(bearing1):
    assert bearing1.dof_local_index()[0] == 0
    assert bearing1.dof_local_index().x_0 == 0
    assert bearing1.dof_local_index()[1] == 1
    assert bearing1.dof_local_index().y_0 == 1


def test_bearing1_interpol_kxx(bearing1):
    assert_allclose(bearing1.kxx_interpolated(314.2), 8.5e7)
    assert_allclose(bearing1.kxx_interpolated(1151.9), 2.6e8)


def test_bearing1_interpol_kyy(bearing1):
    assert_allclose(bearing1.kyy_interpolated(314.2), 9.2e7)
    assert_allclose(bearing1.kyy_interpolated(1151.9), 2.6e8)


def test_bearing1_interpol_cxx(bearing1):
    assert_allclose(bearing1.cxx_interpolated(314.2), 226837, rtol=1e5)
    assert_allclose(bearing1.cxx_interpolated(1151.9), 137973, rtol=1e5)


def test_bearing1_interpol_cyy(bearing1):
    assert_allclose(bearing1.kxx_interpolated(314.2), 235837, rtol=1e5)
    assert_allclose(bearing1.kxx_interpolated(1151.9), 2.6e8, rtol=1e5)


def test_bearing1_interpol_mxx(bearing1):
    assert_allclose(bearing1.mxx_interpolated(314.2), 1e-3, rtol=1e5)
    assert_allclose(bearing1.mxx_interpolated(1151.9), 1.8e-3, rtol=1e5)
    assert_allclose(bearing1.myy_interpolated(314.2), 1e-3, rtol=1e5)
    assert_allclose(bearing1.myy_interpolated(1151.9), 1.8e-3, rtol=1e5)


def test_bearing1_matrices(bearing1):
    # fmt: off
    K = np.array([[85000000.043218,              0., 0.],
                  [             0., 91999999.891728, 0.],
                  [             0.,              0., 0.]])
    C = np.array([[226836.917649,      0., 0.],
                  [     0., 235836.850213, 0.],
                  [     0.,            0., 0.]])
    M = np.array([[0.00099999,         0., 0.],
                  [0.        , 0.00099999, 0.],
                  [0.        ,         0., 0.]])
    # fmt: on
    assert_allclose(bearing1.K(314.2), K, rtol=1e-5)
    assert_allclose(bearing1.C(314.2), C, rtol=1e-5)
    assert_allclose(bearing1.M(314.2), M, rtol=1e-5)


def test_bearing_error_speed_not_given():
    speed = np.linspace(0, 10000, 5)
    kx = 1e8 * speed
    cx = 1e8 * speed
    with pytest.raises(Exception) as excinfo:
        BearingElement(-1, kxx=kx, cxx=cx)
    assert "Arguments (coefficients and frequency) must have the same dimension" in str(
        excinfo.value
    )


def test_bearing_error2():
    with pytest.raises(ValueError) as excinfo:
        BearingElement(
            4, kxx=[7e8, 8e8, 9e8], cxx=[0, 0, 0, 0], frequency=[10, 100, 1000, 10000]
        )
    assert "Arguments (coefficients and frequency) must have the same dimension" in str(
        excinfo.value
    )

    with pytest.raises(ValueError) as excinfo:
        BearingElement(4, kxx=[6e8, 7e8, 8e8, 9e8], cxx=[0, 0, 0, 0, 0])
    assert "Arguments (coefficients and frequency) must have the same dimension" in str(
        excinfo.value
    )


@pytest.fixture
def bearing_constant():
    bearing = BearingElement(n=4, kxx=8e7, cxx=0)
    return bearing


def test_bearing_constant(bearing_constant):
    assert_allclose(bearing_constant.kxx_interpolated(314.2), 8e7, rtol=1e5)
    assert_allclose(bearing_constant.cxx_interpolated(300.9), 0, rtol=1e5)


def test_bearing_len_2():
    bearing = BearingElement(
        n=0,
        kxx=[481, 4810],
        cxx=[3.13, 10.81],
        kyy=[481, 4810],
        kxy=[194, 2078],
        kyx=[-194, -2078],
        cyy=[3.13, 10.81],
        cxy=[0.276, 0.69],
        cyx=[-0.276, -0.69],
        frequency=[115.19, 345.575],
    )
    assert_allclose(bearing.kxx_interpolated(115.19), 481, rtol=1e5)


def test_bearing_len_3():
    bearing = BearingElement(
        n=0,
        kxx=[481, 4810, 18810],
        cxx=[3.13, 10.81, 22.99],
        kyy=[481, 4810, 18810],
        kxy=[194, 2078, 8776],
        kyx=[-194, -2078, -8776],
        cyy=[3.13, 10.81, 22.99],
        cxy=[0.276, 0.69, 1.19],
        cyx=[-0.276, -0.69, -1.19],
        frequency=[115.19, 345.575, 691.15],
    )
    assert_allclose(bearing.kxx_interpolated(115.19), 481, rtol=1e5)


def test_equality(bearing0, bearing1, bearing_constant):
    assert bearing0 == bearing0
    assert not bearing0 == bearing1
    assert not bearing0 == bearing_constant
    assert not bearing0 == 1


def test_from_table():
    bearing_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/data/bearing_seal_si.xls"
    )

    bearing = BearingElement.from_table(0, bearing_file)
    assert bearing.n == 0
    assert_allclose(bearing.frequency[2], 523.5987755985)
    assert_allclose(bearing.kxx[2], 53565700)

    # bearing with us units
    bearing_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/data/bearing_seal_us.xls"
    )
    bearing = BearingElement.from_table(0, bearing_file)
    assert bearing.n == 0
    assert_allclose(bearing.frequency[2], 523.5987755985)
    assert_allclose(bearing.kxx[2], 53565700)


def test_bearing_link_matrices():
    b0 = BearingElement(n=0, n_link=3, kxx=1, cxx=1)
    # fmt: off
    M = np.array([
        [ 1.,  0.,  0., -1., -0., -0.],
        [ 0.,  1.,  0., -0., -1., -0.],
        [ 0.,  0.,  0., -0., -0., -0.],
        [-1., -0., -0.,  1.,  0.,  0.],
        [-0., -1., -0.,  0.,  1.,  0.],
        [-0., -0., -0.,  0.,  0.,  0.]
    ])
    # fmt: on

    assert_allclose(b0.K(0), M)
    assert_allclose(b0.C(0), M)


def test_ball_bearing_element():
    n = 0
    n_balls = 8
    d_balls = 0.03
    fs = 500.0
    alpha = np.pi / 6
    tag = "ballbearing"
    ballbearing = BallBearingElement(
        n=n, n_balls=n_balls, d_balls=d_balls, fs=fs, alpha=alpha, tag=tag
    )

    K = np.array([[4.64168838e07, 0.00000000e00], [0.00000000e00, 1.00906269e08]])
    K = np.pad(K, pad_width=((0, 1), (0, 1)))

    C = np.array([[580.2110481, 0.0], [0.0, 1261.32836543]])
    C = np.pad(C, pad_width=((0, 1), (0, 1)))

    assert_allclose(ballbearing.M(0), np.zeros((3, 3)))
    assert_allclose(ballbearing.K(0), K)
    assert_allclose(ballbearing.C(0), C)
    assert_allclose(ballbearing.G(), np.zeros((3, 3)))


def test_roller_bearing_element():
    n = 0
    n_rollers = 8
    l_rollers = 0.03
    fs = 500.0
    alpha = np.pi / 6
    tag = "rollerbearing"
    rollerbearing = RollerBearingElement(
        n=n, n_rollers=n_rollers, l_rollers=l_rollers, fs=fs, alpha=alpha, tag=tag
    )

    K = np.array([[2.72821927e08, 0.00000000e00], [0.00000000e00, 5.56779444e08]])
    K = np.pad(K, pad_width=((0, 1), (0, 1)))

    C = np.array([[3410.27409251, 0.0], [0.0, 6959.74304593]])
    C = np.pad(C, pad_width=((0, 1), (0, 1)))

    assert_allclose(rollerbearing.M(0), np.zeros((3, 3)))
    assert_allclose(rollerbearing.K(0), K)
    assert_allclose(rollerbearing.C(0), C)
    assert_allclose(rollerbearing.G(), np.zeros((3, 3)))


@pytest.fixture
def magnetic_bearing():
    n = 0
    g0 = 1e-3
    i0 = 1.0
    ag = 1e-4
    nw = 200
    alpha = 0.392
    kp_pid = 1.0
    kd_pid = 1.0
    k_amp = 1.0
    k_sense = 1.0
    tag = "magneticbearing"
    magnetic_bearing = MagneticBearingElement(
        n=n,
        g0=g0,
        i0=i0,
        ag=ag,
        nw=nw,
        alpha=alpha,
        kp_pid=kp_pid,
        kd_pid=kd_pid,
        k_amp=k_amp,
        k_sense=k_sense,
        tag=tag,
    )
    return magnetic_bearing


def test_magnetic_bearing_element(magnetic_bearing):
    K_ref = np.array(
        [
            [-4.64021073e03, -7.72314366e-14],
            [-1.01494475e-13, -4.64021073e03],
        ]
    )

    C_ref = np.array(
        [
            [4.64597874e00, 6.22498335e-17],
            [3.97578344e-17, 4.64597874e00],
        ]
    )

    K = magnetic_bearing.K(0)[0:2, 0:2]
    C = magnetic_bearing.C(0)[0:2, 0:2]

    main_diag_index = np.eye(K.shape[0]).astype(bool)
    sec_diag_index = ~main_diag_index

    # M and G matrices
    assert_allclose(
        magnetic_bearing.M(0),
        np.zeros((3, 3)),
        rtol=0.0,
        atol=1e-10,
    )
    assert_allclose(
        magnetic_bearing.G(),
        np.zeros((3, 3)),
        rtol=0.0,
        atol=1e-10,
    )

    # K and C matrices
    # Main diagonal
    assert_allclose(
        K[main_diag_index],
        K_ref[main_diag_index],
        rtol=1e-8,
        atol=0.0,
    )
    assert_allclose(
        C[main_diag_index],
        C_ref[main_diag_index],
        rtol=1e-8,
        atol=0.0,
    )

    # Secondary diagonal
    assert_allclose(
        K[sec_diag_index],
        K_ref[sec_diag_index],
        rtol=0.0,
        atol=1e-10,
    )
    assert_allclose(
        C[sec_diag_index],
        C_ref[sec_diag_index],
        rtol=0.0,
        atol=1e-10,
    )


def _to_real_array(list_of_scalars):
    return np.array([np.real(float(x)) for x in list_of_scalars], dtype=float)


def test_magnetic_bearing_with_lead_controller_matches_frequency_response():
    n = 0
    g0 = 1e-3  # m
    i0 = 1.0  # A
    ag = 1e-4  # m^2
    nw = 200
    alpha = 0.0
    freq = np.array([10.0, 100.0, 1000.0])  # rad/s

    # --- Lead Controller: C(s) = K * (τ s + 1) / (a τ s + 1), com 0 < a < 1 ---
    K = 2.0
    tau = 1e-3
    a = 0.2
    s = MagneticBearingElement.s
    C_lead = K * (tau * s + 1) / (a * tau * s + 1)

    mb = MagneticBearingElement(
        n=n,
        g0=g0,
        i0=i0,
        ag=ag,
        nw=nw,
        alpha=alpha,
        frequency=freq,
        controller_transfer_function=C_lead,
    )

    C_back = mb.get_analog_controller()
    num_ref = np.array(C_lead.num).squeeze().astype(float)
    den_ref = np.array(C_lead.den).squeeze().astype(float)
    num_got = np.array(C_back.num).squeeze().astype(float)
    den_got = np.array(C_back.den).squeeze().astype(float)

    # Normalization by the first nonzero coefficient
    num_ref = num_ref / num_ref[np.flatnonzero(num_ref)[0]]
    den_ref = den_ref / den_ref[np.flatnonzero(den_ref)[0]]
    num_got = num_got / num_got[np.flatnonzero(num_got)[0]]
    den_got = den_got / den_got[np.flatnonzero(den_got)[0]]

    assert np.allclose(num_got, num_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(den_got, den_ref, rtol=1e-10, atol=1e-12)

    # Compute the frequency response of C(jw) to check kxx and cxx
    mag, phase, _ = ct.frequency_response(C_lead, freq)
    Hjw = (mag * np.exp(1j * phase)).squeeze()

    ks = mb.ks
    ki = mb.ki

    k_eq_expected = ks + ki * np.real(Hjw)
    c_eq_expected = (ki / freq) * np.imag(Hjw)

    kxx = _to_real_array(mb.kxx)
    kyy = _to_real_array(mb.kyy)
    cxx = _to_real_array(mb.cxx)
    cyy = _to_real_array(mb.cyy)

    # As alpha = 0, kxx == kyy == k_eq and cxx == cyy == c_eq
    assert np.allclose(kxx, k_eq_expected, rtol=1e-6, atol=1e-9)
    assert np.allclose(kyy, k_eq_expected, rtol=1e-6, atol=1e-9)
    assert np.allclose(cxx, c_eq_expected, rtol=1e-6, atol=1e-12)
    assert np.allclose(cyy, c_eq_expected, rtol=1e-6, atol=1e-12)

    kxy = _to_real_array(mb.kxy)
    kyx = _to_real_array(mb.kyx)
    cxy = _to_real_array(mb.cxy)
    cyx = _to_real_array(mb.cyx)

    assert np.allclose(kxy, 0.0, atol=1e-10)
    assert np.allclose(kyx, 0.0, atol=1e-10)
    assert np.allclose(cxy, 0.0, atol=1e-12)
    assert np.allclose(cyx, 0.0, atol=1e-12)


@pytest.fixture
def bearing_6dof():
    bearing_6dof = BearingElement(
        n=0, kxx=1e6, kyy=0.8e6, kzz=1e5, cxx=2e2, cyy=1.5e2, czz=0.5e2
    )

    return bearing_6dof


def test_bearing6(bearing_6dof):
    # fmt: off
    K = np.array(
        [[1000000.,      0.,      0.],
         [      0., 800000.,      0.],
         [      0.,      0., 100000.]])
    C = np.array(
        [[200.,   0.,  0.],
         [  0., 150.,  0.],
         [  0.,   0., 50.]])
    M = np.zeros((3, 3))
    G = np.zeros((3, 3))
    # fmt: on

    assert_allclose(bearing_6dof.K(0), K, rtol=1e-3)
    assert_allclose(bearing_6dof.C(0), C, rtol=1e-3)
    assert_allclose(bearing_6dof.M(0), M, rtol=1e-3)
    assert_allclose(bearing_6dof.G(), G, rtol=1e-3)


def test_bearing_6dof_equality():
    bearing_6dof_0 = BearingElement(
        n=0, kxx=1e6, kyy=0.8e6, kzz=1e5, cxx=2e2, cyy=1.5e2, czz=0.5e2
    )
    bearing_6dof_1 = BearingElement(
        n=0, kxx=1e6, kyy=0.8e6, kzz=1e5, cxx=2e2, cyy=1.5e2, czz=0.5e2
    )
    bearing_6dof_2 = BearingElement(
        n=0, kxx=2e6, kyy=0.8e6, kzz=1e5, cxx=2e2, cyy=1.5e2, czz=0.5e2
    )

    assert bearing_6dof_0 == bearing_6dof_1
    assert bearing_6dof_1 != bearing_6dof_2
    assert bearing_6dof_0 != bearing_6dof_2


def test_pickle(bearing0, bearing_constant, bearing_6dof, magnetic_bearing):
    for bearing in [bearing0, bearing_constant, bearing_6dof, magnetic_bearing]:
        bearing_pickled = pickle.loads(pickle.dumps(bearing))
        assert bearing == bearing_pickled


def test_save_load(bearing0, bearing_constant, bearing_6dof, magnetic_bearing):
    file = Path(tempdir) / "bearing0.toml"
    bearing0.save(file)
    bearing0_loaded = BearingElement.load(file)
    assert bearing0 == bearing0_loaded

    file = Path(tempdir) / "bearing_constant.toml"
    bearing_constant.save(file)
    bearing_constant_loaded = BearingElement.load(file)
    assert bearing_constant == bearing_constant_loaded

    file = Path(tempdir) / "bearing_6dof.toml"
    bearing_6dof.save(file)
    bearing_6dof_loaded = BearingElement.load(file)
    assert bearing_6dof == bearing_6dof_loaded

    file = Path(tempdir) / "magnetic_bearing.toml"
    magnetic_bearing.save(file)
    magnetic_bearing_loaded = MagneticBearingElement.load(file)
    assert magnetic_bearing == magnetic_bearing_loaded


def test_save_load_json(bearing0, bearing_constant, bearing_6dof, magnetic_bearing):
    file = Path(tempdir) / "bearing0.json"
    bearing0.save(file)
    bearing0_loaded = BearingElement.load(file)
    assert bearing0 == bearing0_loaded

    file = Path(tempdir) / "bearing_constant.json"
    bearing_constant.save(file)
    bearing_constant_loaded = BearingElement.load(file)
    assert bearing_constant == bearing_constant_loaded

    file = Path(tempdir) / "bearing_6dof.json"
    bearing_6dof.save(file)
    bearing_6dof_loaded = BearingElement.load(file)
    assert bearing_6dof == bearing_6dof_loaded

    file = Path(tempdir) / "magnetic_bearing.json"
    magnetic_bearing.save(file)
    magnetic_bearing_loaded = MagneticBearingElement.load(file)
    assert magnetic_bearing == magnetic_bearing_loaded


def test_save_load_subclasses():
    """Test save/load round-trip for bearing subclasses.

    Verifies that subclass-specific attributes are preserved and that
    loading from file skips expensive computation by using pre-computed
    coefficients passed through kwargs.
    """
    # BallBearingElement
    ball = BallBearingElement(
        n=0, n_balls=8, d_balls=0.03, fs=500.0, alpha=np.pi / 6, tag="ball"
    )
    file = Path(tempdir) / "ball_bearing.json"
    ball.save(file)
    ball_loaded = BallBearingElement.load(file)
    assert ball == ball_loaded
    assert isinstance(ball_loaded, BallBearingElement)
    assert ball_loaded.n_balls == 8

    # RollerBearingElement
    roller = RollerBearingElement(
        n=0, n_rollers=8, l_rollers=0.03, fs=500.0, alpha=np.pi / 6, tag="roller"
    )
    file = Path(tempdir) / "roller_bearing.json"
    roller.save(file)
    roller_loaded = RollerBearingElement.load(file)
    assert roller == roller_loaded
    assert isinstance(roller_loaded, RollerBearingElement)
    assert roller_loaded.n_rollers == 8

    # CylindricalBearing
    cylindrical = CylindricalBearing(
        n=0,
        speed=Q_([1500, 2000], "RPM"),
        weight=525,
        bearing_length=Q_(30, "mm"),
        journal_diameter=Q_(100, "mm"),
        radial_clearance=Q_(0.1, "mm"),
        oil_viscosity=0.1,
        tag="cylindrical",
    )
    file = Path(tempdir) / "cylindrical_bearing.json"
    cylindrical.save(file)
    cylindrical_loaded = CylindricalBearing.load(file)
    assert cylindrical == cylindrical_loaded
    assert isinstance(cylindrical_loaded, CylindricalBearing)
    assert_allclose(cylindrical_loaded.weight, 525)
    # verify derived attributes are preserved via _save_attrs
    assert_allclose(cylindrical_loaded.eccentricity, cylindrical.eccentricity)
    assert_allclose(cylindrical_loaded.attitude_angle, cylindrical.attitude_angle)
    assert_allclose(cylindrical_loaded.sommerfeld, cylindrical.sommerfeld)
    assert_allclose(
        cylindrical_loaded.modified_sommerfeld, cylindrical.modified_sommerfeld
    )

    # MagneticBearingElement
    magnetic = MagneticBearingElement(
        n=0,
        g0=1e-3,
        i0=1.0,
        ag=1e-4,
        nw=200,
        alpha=0.392,
        kp_pid=1.0,
        kd_pid=1.0,
        k_amp=1.0,
        k_sense=1.0,
        tag="magnetic",
    )
    file = Path(tempdir) / "magnetic_bearing_subclass.json"
    magnetic.save(file)
    magnetic_loaded = MagneticBearingElement.load(file)
    assert magnetic == magnetic_loaded
    assert isinstance(magnetic_loaded, MagneticBearingElement)
    assert_allclose(magnetic_loaded.g0, 1e-3)


def test_save_load_skips_computation(magnetic_bearing):
    """Test that loading from file skips computation and uses saved coefficients."""
    file = Path(tempdir) / "magnetic_skip_test.json"
    magnetic_bearing.save(file)

    # verify that loading produces matching K and C matrices
    loaded = MagneticBearingElement.load(file)
    freq = magnetic_bearing.frequency[0]
    assert_allclose(loaded.K(freq), magnetic_bearing.K(freq))
    assert_allclose(loaded.C(freq), magnetic_bearing.C(freq))


def test_bearing_fluid_flow():
    nz = 30
    ntheta = 20
    length = 0.03
    omega = [157.1]
    p_in = 0.0
    p_out = 0.0
    radius_rotor = 0.0499
    radius_stator = 0.05
    load = 525
    visc = 0.1
    rho = 860.0
    bearing = BearingFluidFlow(
        0,
        nz,
        ntheta,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        load=load,
    )

    # fmt: off
    K = np.array([[14547442.70620538, 15571505.36655864],
                  [-25596382.88167763, 12526684.40342712]])
    K = np.pad(K, pad_width=((0, 1), (0, 1)))

    C = np.array([[ 263025.76330117, -128749.90335233],
                  [ -41535.76386708,  309417.62615761]])
    C = np.pad(C, pad_width=((0, 1), (0, 1)))
    # fmt: on

    assert_allclose(bearing.K(0), K, rtol=1e-1)
    assert_allclose(bearing.C(0), C, rtol=1e-1)


def test_plot(bearing0):
    fig = bearing0.plot(coefficients="kxx")
    expected_x = np.array(
        [
            314.2,
            343.0862069,
            371.97241379,
            400.85862069,
            429.74482759,
        ]
    )
    expected_y = np.array(
        [
            8.50000000e07,
            9.39094443e07,
            1.00985975e08,
            1.06782950e08,
            1.11853726e08,
        ]
    )
    assert_allclose(fig.data[0]["x"][:5], expected_x)
    assert_allclose(fig.data[0]["y"][:5], expected_y)

    fig = bearing0.plot(coefficients="cxx")
    expected_x = np.array(
        [314.2, 343.0862069, 371.97241379, 400.85862069, 429.74482759]
    )
    expected_y = np.array(
        [
            226836.91764878,
            222164.94925285,
            217802.8600443,
            213700.3582994,
            209807.15229442,
        ]
    )
    assert_allclose(fig.data[0]["x"][:5], expected_x)
    assert_allclose(fig.data[0]["y"][:5], expected_y)


def test_cylindrical_hydrodynamic():
    cylindrical = CylindricalBearing(
        n=0,
        speed=Q_([1500, 2000], "RPM"),
        weight=525,
        bearing_length=Q_(30, "mm"),
        journal_diameter=Q_(100, "mm"),
        radial_clearance=Q_(0.1, "mm"),
        oil_viscosity=0.1,
    )
    expected_modified_sommerfeld = np.array([1.009798, 1.346397])
    expected_sommerfeld = np.array([3.571429, 4.761905])
    expected_eccentricity = np.array([0.266298, 0.212571])
    expected_attitude_angle = np.array([0.198931, 0.161713])
    expected_k = np.array([[12.80796, 16.393593], [-25.060393, 8.815303]])
    expected_k = np.pad(expected_k, pad_width=((0, 1), (0, 1)))
    expected_c = np.array([[232.89693, -81.924371], [-81.924371, 294.911619]])
    expected_c = np.pad(expected_c, pad_width=((0, 1), (0, 1)))
    assert_allclose(
        cylindrical.modified_sommerfeld, expected_modified_sommerfeld, rtol=1e-6
    )
    assert_allclose(cylindrical.sommerfeld, expected_sommerfeld, rtol=1e-6)
    assert_allclose(cylindrical.eccentricity, expected_eccentricity, rtol=1e-5)
    assert_allclose(cylindrical.attitude_angle, expected_attitude_angle, rtol=1e-5)
    assert_allclose(cylindrical.K(Q_(1500, "RPM")) / 1e6, expected_k, rtol=1e-6)
    assert_allclose(cylindrical.C(Q_(1500, "RPM")) / 1e3, expected_c, rtol=1e-6)
