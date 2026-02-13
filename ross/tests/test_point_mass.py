import pickle
from pathlib import Path
from tempfile import tempdir

import numpy as np
from numpy.testing import assert_allclose

from ross.point_mass import PointMass


def test_point_mass():
    # fmt: off
    m0 = np.array([[10.,   0,   0],
                   [  0, 10.,   0],
                   [  0,   0, 10.]])
    # fmt: on

    p = PointMass(n=0, m=10.0, tag="pointmass")

    assert p.tag == "pointmass"
    assert_allclose(m0, p.M())
    assert_allclose(np.zeros((3, 3)), p.K())
    assert_allclose(np.zeros((3, 3)), p.C())
    assert_allclose(np.zeros((3, 3)), p.G())


def test_local_index():
    n = 0
    mx = 1.0
    my = 2.0
    mz = 3.0
    p = PointMass(n=n, mx=mx, my=my, mz=mz)

    assert p.dof_local_index().x_0 == 0
    assert p.dof_local_index().y_0 == 1
    assert p.dof_local_index().z_0 == 2


def test_pickle():
    p = PointMass(n=0, m=10.0, tag="pointmass")
    p_pickled = pickle.loads(pickle.dumps(p))
    assert p == p_pickled


def test_save_load():
    p = PointMass(n=0, m=10.0, tag="pointmass")
    file = Path(tempdir) / "point_mass.toml"
    p.save(file)
    p_loaded = p.load(file)

    assert p == p_loaded


def test_save_load_json():
    p = PointMass(n=0, m=10.0, tag="pointmass")
    file = Path(tempdir) / "point_mass.json"
    p.save(file)
    p_loaded = p.load(file)

    assert p == p_loaded
