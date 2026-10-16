"""Regression tests for polymorphic rotor persistence."""

import numpy as np

import ross as rs
from ross.multi_rotor.multi_rotor import two_shaft_rotor_example


def test_save_load_preserves_multirotor_topology(tmp_path):
    rotor = two_shaft_rotor_example()
    file = tmp_path / "multi_rotor.toml"

    rotor.save(file)
    loaded = rs.Rotor.load(file)

    assert isinstance(loaded, rs.MultiRotor)
    assert loaded.coupled_nodes == rotor.coupled_nodes
    assert np.allclose(loaded.K(0), rotor.K(0))


def test_inherited_analyses_rebuild_multirotor():
    rotor = two_shaft_rotor_example()

    ucs = rotor.run_ucs(num=2, num_modes=8)
    level1 = rotor.run_level1(n=0, stiffness_range=(1e6, 1e7), num=2)
    convergence = rotor.convergence(err_max=1e3)
    static = rotor.run_static()

    assert isinstance(ucs, rs.UCSResults)
    assert isinstance(level1, rs.Level1Results)
    assert isinstance(convergence, rs.ConvergenceResults)
    assert isinstance(static, rs.StaticResults)
