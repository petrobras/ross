import os
import pickle
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearing_seal_element import (
    BallBearingElement,
    BearingElement,
    BearingElement6DoF,
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
    K = np.array([[85000000.043218,        0.      ],
                  [       0.      , 91999999.891728]])
    C = np.array([[226836.917649,      0.          ],
                  [       0.      , 235836.850213  ]])
    M = np.array([[0.00099999,      0.   ],
                  [     0.   , 0.00099999]])
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
    assert (
        "Arguments (coefficients and frequency)"
        " must have the same dimension" in str(excinfo.value)
    )


def test_bearing_error2():
    with pytest.raises(ValueError) as excinfo:
        BearingElement(
            4, kxx=[7e8, 8e8, 9e8], cxx=[0, 0, 0, 0], frequency=[10, 100, 1000, 10000]
        )
    assert (
        "Arguments (coefficients and frequency) "
        "must have the same dimension" in str(excinfo.value)
    )

    with pytest.raises(ValueError) as excinfo:
        BearingElement(4, kxx=[6e8, 7e8, 8e8, 9e8], cxx=[0, 0, 0, 0, 0])
    assert (
        "Arguments (coefficients and frequency) "
        "must have the same dimension" in str(excinfo.value)
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
    M = np.array(
        [[1, 0, -1, 0],
         [0, 1, 0, -1],
         [-1, 0, 1, 0],
         [0, -1, 0, 1]]
    )
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

    M = np.zeros((2, 2))
    K = np.array([[4.64168838e07, 0.00000000e00], [0.00000000e00, 1.00906269e08]])
    C = np.array([[580.2110481, 0.0], [0.0, 1261.32836543]])
    G = np.zeros((2, 2))

    assert_allclose(ballbearing.M(0), M)
    assert_allclose(ballbearing.K(0), K)
    assert_allclose(ballbearing.C(0), C)
    assert_allclose(ballbearing.G(), G)


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

    M = np.zeros((2, 2))
    K = np.array([[2.72821927e08, 0.00000000e00], [0.00000000e00, 5.56779444e08]])
    C = np.array([[3410.27409251, 0.0], [0.0, 6959.74304593]])
    G = np.zeros((2, 2))

    assert_allclose(rollerbearing.M(0), M)
    assert_allclose(rollerbearing.K(0), K)
    assert_allclose(rollerbearing.C(0), C)
    assert_allclose(rollerbearing.G(), G)


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
    M = np.array([[0.0, 0.0], [0.0, 0.0]])
    K = np.array([[-4640.62337718, 0.0], [0.0, -4640.62337718]])
    C = np.array([[4.64526865, 0.0], [0.0, 4.64526865]])
    G = np.array([[0.0, 0.0], [0.0, 0.0]])

    assert_allclose(magnetic_bearing.M(0), M)
    assert_allclose(magnetic_bearing.K(0), K)
    assert_allclose(magnetic_bearing.C(0), C)
    assert_allclose(magnetic_bearing.G(), G)


@pytest.fixture
def bearing_6dof():
    bearing_6dof = BearingElement6DoF(
        n=0, kxx=1e6, kyy=0.8e6, kzz=1e5, cxx=2e2, cyy=1.5e2, czz=0.5e2
    )

    return bearing_6dof


def test_bearing6(bearing_6dof):
    # fmt: off
    K = np.array(
        [[1000000., 0., 0.],
         [0., 800000., 0.],
         [0., 0., 100000.]])
    C = np.array(
        [[200., 0., 0.],
         [0., 150., 0.],
         [0., 0., 50.]])
    M = np.zeros((3, 3))
    G = np.zeros((3, 3))
    # fmt: on

    assert_allclose(bearing_6dof.K(0), K, rtol=1e-3)
    assert_allclose(bearing_6dof.C(0), C, rtol=1e-3)
    assert_allclose(bearing_6dof.M(0), M, rtol=1e-3)
    assert_allclose(bearing_6dof.G(), G, rtol=1e-3)


def test_bearing_6dof_equality():
    bearing_6dof_0 = BearingElement6DoF(
        n=0, kxx=1e6, kyy=0.8e6, kzz=1e5, cxx=2e2, cyy=1.5e2, czz=0.5e2
    )
    bearing_6dof_1 = BearingElement6DoF(
        n=0, kxx=1e6, kyy=0.8e6, kzz=1e5, cxx=2e2, cyy=1.5e2, czz=0.5e2
    )
    bearing_6dof_2 = BearingElement6DoF(
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
    bearing_6dof_loaded = BearingElement6DoF.load(file)
    assert bearing_6dof == bearing_6dof_loaded

    file = Path(tempdir) / "magnetic_bearing.toml"
    magnetic_bearing.save(file)
    magnetic_bearing_loaded = MagneticBearingElement.load(file)
    assert magnetic_bearing == magnetic_bearing_loaded


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

    C = np.array([[ 263025.76330117, -128749.90335233],
                  [ -41535.76386708,  309417.62615761]])
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
    expected_c = np.array([[232.89693, -81.924371], [-81.924371, 294.911619]])
    assert_allclose(
        cylindrical.modified_sommerfeld, expected_modified_sommerfeld, rtol=1e-6
    )
    assert_allclose(cylindrical.sommerfeld, expected_sommerfeld, rtol=1e-6)
    assert_allclose(cylindrical.eccentricity, expected_eccentricity, rtol=1e-5)
    assert_allclose(cylindrical.attitude_angle, expected_attitude_angle, rtol=1e-5)
    assert_allclose(cylindrical.K(Q_(1500, "RPM")) / 1e6, expected_k, rtol=1e-6)
    assert_allclose(cylindrical.C(Q_(1500, "RPM")) / 1e3, expected_c, rtol=1e-6)
