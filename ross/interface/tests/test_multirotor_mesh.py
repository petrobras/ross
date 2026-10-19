# -*- coding: utf-8 -*-
"""The gear mesh options of the MultiRotor form, held against `rs.MultiRotor`.

ROSS #1427 gave `rs.MultiRotor` a `mesh_damping_ratio` (default 0.0; the
backlash model of ROSS 3.0 used 0.07 without saying so). The form had no field
for it, so every geared train the interface built ran with an undamped mesh.
The reference here is ROSS itself, built in this file from the same numbers,
not the interface's builder.
"""

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

rs = pytest.importorskip("ross", reason="requires ROSS installed")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ross.interface.domain.python_export import build_script  # noqa: E402
from ross.interface.domain.rotor_builder import assemble_rotor  # noqa: E402

BACKLASH = {
    "enable": True,
    "initial_value": 1e-4,
    "error_amp": 0.0,
    "smooth_operator": False,
    "sigma": 1e4,
}


def half(gear_node):
    """One rotor of the train as the screen holds it: millimetres, as typed."""
    return {
        "materials": [{"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [
            {"L": "250", "idl": "0", "odl": "50", "material": "Steel"} for _ in range(3)
        ],
        "disks": [],
        "gears": [
            {
                "element_type": "BASIC",
                "n": str(gear_node),
                "m": "5",
                "Ip": "0.04",
                "Id": "0.02",
                "n_teeth": "30",
                "pitch_diameter": "200",
            }
        ],
        "couplings": [],
        "seals": [],
        "bearings": [
            {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "0"},
            {"element_type": "BASIC", "n": "3", "kxx": "1e6", "cxx": "0"},
        ],
        "pointmasses": [],
        "probes": [],
    }


def ross_half(gear_node):
    """The same rotor written directly in ROSS, in SI."""
    steel = rs.Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
    shafts = [rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=steel)] * 3
    gear = rs.GearElement(
        n=gear_node, m=5, Ip=0.04, Id=0.02, n_teeth=30, pitch_diameter=0.2
    )
    bearings = [
        rs.BearingElement(n=0, kxx=1e6, cxx=0),
        rs.BearingElement(n=3, kxx=1e6, cxx=0),
    ]
    return rs.Rotor(shafts, [gear], bearings)


def train(**multi_params):
    params = {"coupled_nodes": "2, 1", "gear_mesh_stiffness": "1e8"}
    params.update(multi_params)
    return {
        "isMultiRotor": True,
        "driving_rotor": half(2),
        "driven_rotor": half(1),
        "multi_params": params,
    }


def ross_train(**kwargs):
    return rs.MultiRotor(
        ross_half(2),
        ross_half(1),
        coupled_nodes=(2, 1),
        gear_mesh_stiffness=1e8,
        position="above",
        **kwargs,
    )


def ross_kwargs(multi_params):
    """The arguments of `rs.MultiRotor` the form's mesh fields stand for."""
    kwargs = {"mesh_damping_ratio": float(multi_params["mesh_damping_ratio"])}
    if "backlash" in multi_params:
        kwargs["backlash"] = dict(multi_params["backlash"])
    return kwargs


def exported(project):
    """The MultiRotor the exported script builds, run up to its first figure."""
    script = build_script(project)
    modelling = script.split('print("Rotor Successfully Modeled!")')[0]
    namespace = {}
    exec(modelling, namespace)
    return namespace["rotor"]


@pytest.mark.parametrize("ratio", [0.0, 0.07, 0.3])
def test_the_mesh_damping_ratio_reaches_the_multirotor(ratio):
    built = assemble_rotor(train(mesh_damping_ratio=str(ratio))).rotor
    reference = ross_train(mesh_damping_ratio=ratio)
    assert built.mesh.damping_ratio == pytest.approx(ratio)
    for frequency in (0.0, 100.0):
        assert_allclose(built.C(frequency), reference.C(frequency), atol=1e-9)
        assert_allclose(built.K(frequency), reference.K(frequency), rtol=1e-12)


def test_a_damped_mesh_changes_the_damping_matrix():
    """Control for the test above: the field has to matter to C to be checked."""
    damped = assemble_rotor(train(mesh_damping_ratio="0.07")).rotor.C(0)
    undamped = ross_train().C(0)
    assert np.max(np.abs(damped - undamped)) > 1.0


def test_a_backlash_model_carries_the_mesh_damping_ratio():
    """With backlash, ROSS keeps the damping out of C and uses it in the
    nonlinear mesh force; the ratio still has to arrive."""
    built = assemble_rotor(train(mesh_damping_ratio="0.07", backlash=BACKLASH)).rotor
    reference = ross_train(mesh_damping_ratio=0.07, backlash=dict(BACKLASH))
    assert built.mesh.backlash and reference.mesh.backlash
    assert built.mesh.damping_ratio == pytest.approx(0.07)
    assert_allclose(built.C(0), reference.C(0), atol=1e-9)


@pytest.mark.parametrize("ratio", ["-0.1", "-1e-3"])
@pytest.mark.parametrize("backlash", [None, BACKLASH])
def test_a_negative_mesh_damping_ratio_is_refused(ratio, backlash):
    """A negative ratio makes the mesh add energy to the train; ROSS builds it
    without a word, so the builder has to say no."""
    extra = {"mesh_damping_ratio": ratio}
    if backlash is not None:
        extra["backlash"] = backlash
    with pytest.raises(ValueError, match="zero or positive, not %s" % ratio):
        assemble_rotor(train(**extra))


@pytest.mark.parametrize("blank", [{}, {"mesh_damping_ratio": ""}])
def test_a_project_saved_without_the_field_builds_an_undamped_mesh(blank):
    """A file written before the field existed, or the field left empty, gets
    ROSS's own default."""
    built = assemble_rotor(train(**blank)).rotor
    assert built.mesh.damping_ratio == 0.0
    assert_allclose(built.C(0), ross_train().C(0), atol=1e-9)


@pytest.mark.parametrize("backlash", [None, BACKLASH])
def test_the_exported_script_builds_the_same_mesh_damping(backlash):
    extra = {"mesh_damping_ratio": "0.07"}
    if backlash is not None:
        extra["backlash"] = backlash
    project = train(**extra)
    script = build_script(project)
    assert "mesh_damping_ratio=0.07" in script
    mine = exported(project)
    theirs = assemble_rotor(project).rotor
    assert mine.mesh.damping_ratio == pytest.approx(0.07)
    assert_allclose(mine.C(0), theirs.C(0), atol=1e-9)
    assert_allclose(mine.C(0), ross_train(**ross_kwargs(extra)).C(0), atol=1e-9)


def test_an_old_project_exports_without_the_field():
    project = train()
    assert "mesh_damping_ratio" not in build_script(project)
    assert exported(project).mesh.damping_ratio == 0.0
