import numpy as np
from ross.point_mass import PointMass
from numpy.testing import assert_allclose


def test_point_mass():
    # fmt: off
    m0 = np.array([[10.0, 0],
                   [0, 10.0]])
    # fmt: on

    p = PointMass(n=0, m=10.0, tag="pointmass")

    assert p.tag == 'pointmass'
    assert_allclose(m0, p.M())
    assert_allclose(np.zeros((2, 2)), p.K())
    assert_allclose(np.zeros((2, 2)), p.C())
    assert_allclose(np.zeros((2, 2)), p.G())


def test_local_index():
    n = 0
    mx = 1.0
    my = 2.0
    p = PointMass(n=n, mx=mx, my=my)

    assert p.dof_local_index().x_0 == 0
    assert p.dof_local_index().y_0 == 1
