"""Regression tests for polymorphic rotor persistence."""

import numpy as np
import pytest

import ross as rs
from ross.multi_rotor.multi_rotor import two_shaft_rotor_example
from ross.rotor_assembly import coaxrotor_example


def test_save_load_preserves_multirotor_topology(tmp_path):
    rotor = two_shaft_rotor_example()
    file = tmp_path / "multi_rotor.toml"

    rotor.save(file)
    loaded = rs.Rotor.load(file)

    assert isinstance(loaded, rs.MultiRotor)
    assert loaded.coupled_nodes == rotor.coupled_nodes
    assert np.allclose(loaded.K(0), rotor.K(0))


@pytest.mark.parametrize("suffix", [".toml", ".json"])
def test_save_load_preserves_coaxial_rotor_topology(tmp_path, suffix):
    rotor = coaxrotor_example()
    file = tmp_path / f"coaxial_rotor{suffix}"

    rotor.save(file)
    loaded = rs.Rotor.load(file)

    assert isinstance(loaded, rs.CoAxialRotor)
    assert loaded.nodes_pos == rotor.nodes_pos
    assert len(loaded.shaft_elements) == len(rotor.shaft_elements)
    assert np.allclose(loaded.M(0), rotor.M(0))
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


def test_new_positional_order_remains_compatible():
    rotor = two_shaft_rotor_example()

    configured = rs.MultiRotor(
        rotor.rotors["driving"],
        rotor.rotors["driven"],
        rotor.coupled_nodes,
        rotor.mesh.stiffness,
        False,
        {"enable": False, "amplitude_ratio": 0},
        0.123,
        {"enable": False},
        0.0,
        "below",
        "comparison",
    )

    assert configured.mesh.damping_ratio == 0.123
    assert configured.position == "below"
    assert configured.tag == "comparison"
