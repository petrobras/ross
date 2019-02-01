from ross import *
import numpy as np
import pytest

steel = mtr.Material(name='steel', E=211e9,G_s=81.2e9, rho=7810)


def rotor_example(w,n_el):
    shaft_elm = []
    for i in range(n_el):
        shaft_elm.append(shaft.ShaftElement(L=1.5/n_el , material=steel, n=i, i_d=0, o_d=0.05))
    disk0 = disk.DiskElement.from_geometry(n=(n_el/1.5) * 0.5, material=steel, width=0.07, i_d=0.05, o_d=0.28)
    disk1 = disk.DiskElement.from_geometry(n=(n_el/1.5), material=steel, width=0.07, i_d=0.05, o_d=0.35)

    bearing0 = bse.BearingElement(n=0, kxx=1e6, kyy=1e6, cxx=0, cyy=0)
    bearing1 = bse.BearingElement(n=n_el, kxx=1e6, kyy=1e6, cxx=0, cyy=0)

    return rotor.Rotor(shaft_elm, [disk0, disk1], [bearing0, bearing1], w=w)


def test_example1_w_eq_0():
    assert rotor_example(0, 24).wn == pytest.approx(np.array([86.66, 86.66,274.31,274.31,716.78,716.78]),1e-1)


def test_example1_w_eq_4000RPM():
    assert rotor_example(4000/60, 1200).wn == pytest.approx(np.array([85.39, 87.80, 251.78, 294.71, 600.18, 827.08]),1e-1)


def rotor_example2(w):
    return rotor.Rotor([shaft.ShaftElement(L=1, o_d=0.05, i_d=0, n=0, material=steel,
                                           rotary_inertia=True, shear_effects=True, shear_method_calc='cowper'),
                        shaft.ShaftElement(L=0.5, o_d=0.05, i_d=0, n=1, material=steel,
                                           rotary_inertia=True, shear_effects=True,shear_method_calc='cowper')],
                       [disk.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.35)],
                       [bse.BearingElement(n=0, kxx=10e6, kyy=10e6, cxx=0, cyy=0), bse.BearingElement(n=1, kxx=10e6, kyy=10e6, cxx=0, cyy=0)], w=w)


def test_example2_w_eq_0():
    assert pytest.approx(rotor_example2(0).wn) == np.array([90.14, 90.14, 630.73, 630.73, 830.43, 830.43])


def test_example2_w_eq_4000RPM():
    assert pytest.approx(rotor_example2(4000/60).wn) == np.array([76.19, 103.91, 565.99, 634.23, 647.75, 1174.2])

