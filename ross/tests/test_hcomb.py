import numpy as np
import pytest

from numpy.testing import assert_allclose

from ross.seals.hcomb_seal import HcombSeal
from ross.units import Q_


@pytest.fixture
def hcomb():

    seal = HcombSeal(
        n=0,
        frequency=Q_([5000], "RPM"),
        length=0.0254,
        radius=0.0751,
        clearance=0.0004,
        roughness=0.00198,
        cell_length=0.001,
        cell_width=0.001,
        cell_depth=0.00229,
        inlet_pressure=1830000.0,
        outlet_pressure=823500.0,
        inlet_temperature=300.0,
        b_suther=1.458e-6,
        s_suther=110.4,
        molar=29.0,
        gamma=1.4,
        preswirl=1.0,
        entr_coef=0.1,
        exit_coef=0.5
    )

    return seal

def test_hcomb_coefficients(hcomb):

    assert_allclose(hcomb.kxx, 586228.88958017, rtol=1e-4)
    assert_allclose(hcomb.kxy, 159741.17580073, rtol=1e-4)
    assert_allclose(hcomb.kyx, -159741.17580073, rtol=1e-4)
    assert_allclose(hcomb.kyy, 586228.88958017, rtol=1e-4)
    assert_allclose(hcomb.cxx, 294.42942927, rtol=1e-4)
    assert_allclose(hcomb.cxy, -27.16217997, rtol=1e-4)
    assert_allclose(hcomb.cyx, 27.16217997, rtol=1e-4)
    assert_allclose(hcomb.cyy, 294.42942927, rtol=1e-4)
    assert_allclose(hcomb.seal_leakage, 0.6313559209954082, rtol=1e-4)
