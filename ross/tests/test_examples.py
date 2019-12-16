import numpy as np
import pytest

import ross as rs
from ross.materials import steel


def rotor_example1(n_el=48):
    """
    This function instantiate a rotor similar to the example  5.9.1, page 206 (Dynamics of rotating machine, FRISSWELL)
    The following functions test_example1_w_equals0rpm() and test_example1_w_equals4000rpm() are the test functions of
    this example.

    P.S.:Isotropic bearings.

    :param frequency: speed of rotation.
    :param n_el: number of shaft elements.
    :return: Rotor object.
    """

    shaft_elm = []
    for i in range(n_el):
        shaft_elm.append(
            rs.ShaftElement(L=1.5 / n_el, material=steel, n=i, idl=0, odl=0.05)
        )
    disk0 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5) * 0.5, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5), material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    bearing0 = rs.BearingElement(n=0, kxx=1e6, kyy=1e6, cxx=0, cyy=0)
    bearing1 = rs.BearingElement(n=n_el, kxx=1e6, kyy=1e6, cxx=0, cyy=0)

    return rs.Rotor(shaft_elm, [disk0, disk1], [bearing0, bearing1])


def test_example1_w_equals_0rpm():
    rotor = rotor_example1(6)
    modal = rotor.run_modal(0)
    assert modal.wn == pytest.approx(
        np.array([86.66, 86.66, 274.31, 274.31, 716.78, 716.78]), 1e-3
    )


def test_example1_w_equals_4000rpm():
    rotor = rotor_example1(6)
    modal = rotor.run_modal(4000 * np.pi / 30)
    assert modal.wn == pytest.approx(
        np.array([85.39, 87.80, 251.78, 294.71, 600.18, 827.08]), 1e-3
    )


def rotor_example2(n_el=48):
    """
    This function instantiate a overhung rotor similar to the example  5.9.9, page 218 (Dynamics of rotating machine,
    FRISSWELL).
    The following functions test_example2_w_equals0rpm() and test_example2_w_equals4000rpm() are the test functions of
    this example.

    P.S.: Overhung rotor.

    :param n_el: number of shaft elements.
    :return: Rotor object.
    """

    shaft_elm = []
    for i in range(n_el):
        shaft_elm.append(
            rs.ShaftElement(L=1.5 / n_el, material=steel, n=i, idl=0, odl=0.05)
        )
    return rs.Rotor(
        shaft_elm,
        [
            rs.DiskElement.from_geometry(
                n=n_el, material=steel, width=0.07, i_d=0.05, o_d=0.35
            )
        ],
        [
            rs.BearingElement(n=0, kxx=10e6, kyy=10e6, cxx=0, cyy=0),
            rs.BearingElement(n=int((n_el / 1.5)), kxx=10e6, kyy=10e6, cxx=0, cyy=0),
        ],
    )


def test_example2_w_eq_0rpm():
    rotor = rotor_example2(6)
    modal = rotor.run_modal(0)
    assert pytest.approx(modal.wn, 1e-3) == np.array(
        [90.14, 90.14, 630.73, 630.73, 830.43, 830.43]
    )


def test_example2_w_eq_4000rpm():
    rotor = rotor_example2(6)
    modal = rotor.run_modal(4000 * np.pi / 30)
    assert pytest.approx(modal.wn, 1e-3) == np.array(
        [76.19, 103.91, 565.99, 634.23, 647.75, 1174.2]
    )


def rotor_example3(n_el=48):
    """
    This function instantiate a rotor similar to the example  5.9.2, page 208 (Dynamics of rotating machine, FRISSWELL)
    The following functions test_example3_w_equals0rpm() and test_example3_w_equals4000rpm() are the test functions of
    this example.

    P.S.: Anisotropic bearings.

    :param n_el: number of shaft elements.
    :return: Rotor object.
    """
    assert not n_el % 3, "n_el must be a multiple of 3"
    shaft_elm = []
    for i in range(n_el):
        shaft_elm.append(
            rs.ShaftElement(L=1.5 / n_el, material=steel, n=i, idl=0, odl=0.05)
        )
    disk0 = rs.DiskElement.from_geometry(
        n=(n_el / 3), material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5), material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    bearing0 = rs.BearingElement(n=0, kxx=1e6, kyy=8e5, cxx=0, cyy=0)
    bearing1 = rs.BearingElement(n=n_el, kxx=1e6, kyy=8e5, cxx=0, cyy=0)

    return rs.Rotor(shaft_elm, [disk0, disk1], [bearing0, bearing1])


def test_example3_w_equals_0rpm():
    rotor = rotor_example3(48)
    modal = rotor.run_modal(0)
    assert pytest.approx(modal.wn, 1e-3) == np.array(
        [82.65, 86.66, 254.52, 274.31, 679.49, 716.79]
    )


def test_example3_w_equals_4000rpm():
    rotor = rotor_example3(48)
    modal = rotor.run_modal(4000 * np.pi / 30)
    assert pytest.approx(modal.wn, 1e-3) == np.array(
        [82.33, 86.86, 239.64, 287.25, 583.49, 806.89]
    )


def rotor_example4(n_el=48):
    """
    This function instantiate a rotor similar to the example  5.9.5, page 212 (Dynamics of rotating machine, FRISSWELL)
    The following functions test_example4_w_equals0rpm() and test_example4_w_equals4000rpm() are the test functions of
    this example.

    P.S.: Isotropic, damped bearings.

    :param n_el: number of shaft elements.
    :return: Rotor object.
    """
    shaft_elm = []
    for i in range(n_el):
        shaft_elm.append(
            rs.ShaftElement(L=1.5 / n_el, material=steel, n=i, idl=0, odl=0.05)
        )
    disk0 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5) * 0.5, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5), material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    bearing0 = rs.BearingElement(n=0, kxx=1e6, kyy=1e6, cxx=3e3, cyy=3e3)
    bearing1 = rs.BearingElement(n=n_el, kxx=1e6, kyy=1e6, cxx=3e3, cyy=3e3)

    return rs.Rotor(shaft_elm, [disk0, disk1], [bearing0, bearing1])


def test_example4_w_equals_0rpm():
    rotor = rotor_example4(6)
    modal = rotor.run_modal(0)
    wn_hertz = modal.wn / (2 * np.pi)
    wd_hertz = modal.wd / (2 * np.pi)
    assert pytest.approx(wn_hertz, 1e-2) == np.array(
        [13.91, 13.91, 48.18, 48.18, 137.06, 137.06]
    )
    assert pytest.approx(wd_hertz, 1e-2) == np.array(
        [13.89, 13.89, 46.54, 46.54, 103.22, 103.22]
    )


def test_example4_w_equals_4000rpm():
    rotor = rotor_example4(6)
    modal = rotor.run_modal((4000 * np.pi) / 30)
    wn_hertz = modal.wn / (2 * np.pi)
    wd_hertz = modal.wd / (2 * np.pi)
    assert pytest.approx(wn_hertz, 1e-1) == np.array(
        [13.70, 14.09, 43.61, 52.18, 122.37, 149.81]
    )
    assert pytest.approx(wd_hertz, 1e-1) == np.array(
        [13.68, 14.07, 41.98, 50.65, 104.25, 105.66]
    )


def rotor_example5(n_el=48):
    """
    This function instantiate a rotor similar to the example  5.9.3, page 209 (Dynamics of rotating machine, FRISSWELL)
    The following function, test_example5_w_equals4000rpm() is the test function of
    this example.

    P.S.: Mixed Modes.

    :param n_el: number of shaft elements.
    :return: Rotor object.
    """

    shaft_elm = []
    for i in range(n_el):
        shaft_elm.append(
            rs.ShaftElement(L=1.5 / n_el, material=steel, n=i, idl=0, odl=0.05)
        )
    disk0 = rs.DiskElement.from_geometry(
        n=(n_el / 3), material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5), material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    bearing0 = rs.BearingElement(n=0, kxx=1e6, kyy=2e5, cxx=0, cyy=0)
    bearing1 = rs.BearingElement(n=n_el, kxx=1e6, kyy=2e5, cxx=0, cyy=0)

    return rs.Rotor(shaft_elm, [disk0, disk1], [bearing0, bearing1])


def test_example5_w_equals4000rpm():
    rotor = rotor_example5(48)
    modal = rotor.run_modal((4000 * np.pi) / 30)
    wn_hertz = modal.wn / (2 * np.pi)
    assert pytest.approx(wn_hertz, 1) == (
        np.array([8.545, 13.77, 22.35, 44.06, 78.76, 120.4])
    ) / (2 * np.pi)


def rotor_example6(n_el=48):
    """
    This function instantiate a rotor similar to the example  5.9.4, page 210 (Dynamics of rotating machine, FRISSWELL)
    The following functions test_example6_w_equals0rpm() and test_example6_w_equals4000rpm() are the test functions of
    this example.

    P.S.:Cross-coupled bearings.

    :param n_el: number of shaft elements.
    :return: Rotor object.
    """

    shaft_elm = []
    for i in range(n_el):
        shaft_elm.append(
            rs.ShaftElement(L=1.5 / n_el, material=steel, n=i, idl=0, odl=0.05)
        )
    disk0 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5) * 0.5, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = rs.DiskElement.from_geometry(
        n=(n_el / 1.5), material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    bearing0 = rs.BearingElement(n=0, kxx=1e6, kyy=1e6, cxx=0, kxy=5e5, cyy=0)
    bearing1 = rs.BearingElement(n=n_el, kxx=1e6, kyy=1e6, cxx=0, kxy=5e5, cyy=0)

    return rs.Rotor(shaft_elm, [disk0, disk1], [bearing0, bearing1])


def test_example6_w_equals4000rpm():
    rotor = rotor_example6(48)
    modal = rotor.run_modal((4000 * np.pi) / 30)
    wn_hertz = modal.wn / (2 * np.pi)
    assert pytest.approx(wn_hertz, 1) == (
        np.array([11.66, 14.80, 33.97, 49.19, 97.97, 126.61])
    )
