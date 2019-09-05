import numpy as np
from ross.point_mass import PointMass
from numpy.testing import assert_allclose


def test_point_mass():
    # fmt: off
    m0 = np.array([[10.0, 0],
                   [0, 10.0]])
    # fmt: on

    p = PointMass(n=0, m=10.0)

    assert_allclose(m0, p.M())
    assert_allclose(np.zeros((2, 2)), p.K())
    assert_allclose(np.zeros((2, 2)), p.C())
    assert_allclose(np.zeros((2, 2)), p.G())
