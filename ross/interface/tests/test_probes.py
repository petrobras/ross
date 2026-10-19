# -*- coding: utf-8 -*-
"""The probes of the model: where the analyses will read the response.

ROSS's `Rotor` takes no probes -- `rs.Probe` is an argument of its response
plots -- so a probe is a category of the screen and not of the rotor. What
has to hold, and what this file checks:

* the builder checks each probe with `rs.Probe` and against the rotor, and
  names the row it refuses; the rotor it builds is the one without probes;
* the form is ROSS's `Probe(node, angle, direction, tag)`, with the node
  called `n` as in every other category;
* the 3D scene places each probe on its node, with its direction and its
  angle in radians;
* a probe moves with its node when a shaft is split, discretized or joined to
  another rotor -- which is the reason to place it on the model at all.
"""

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.domain.concatenation import concatenated_project  # noqa: E402
from ross.interface.domain.meshing import mesh_shafts  # noqa: E402
from ross.interface.domain.rotor_builder import (  # noqa: E402
    PROBE_ANGLE_UNREADABLE,
    PROBE_DIRECTION,
    PROBE_NEEDS_ANGLE,
    PROBE_NO_NODE,
    assemble_rotor,
    build_rotor_from_ui,
)
from ross.interface.domain.rotor_scene import describe_scene  # noqa: E402
from ross.interface.domain.schema import build_schema, schema_problems  # noqa: E402
from ross.interface.domain.splitting import split_shaft  # noqa: E402

STEEL = {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}


def project(probes, shafts=4):
    return {
        "materials": [dict(STEEL)],
        "shafts": [
            {"L": "250", "idl": "0", "odl": "50", "material": "Steel"}
            for _ in range(shafts)
        ],
        "disks": [],
        "gears": [],
        "couplings": [],
        "seals": [],
        "bearings": [
            {"n": "0", "kxx": "1e6", "cxx": "0"},
            {"n": str(shafts), "kxx": "1e6", "cxx": "0"},
        ],
        "pointmasses": [],
        "probes": probes,
    }


RADIAL = {"n": "2", "tag": "DE X", "direction": "radial", "angle": "45"}
AXIAL = {"n": "4", "tag": "Thrust", "direction": "axial", "angle": ""}


# --- the builder -----------------------------------------------------------------


def test_each_probe_is_an_rs_probe_in_the_list_order():
    probes = assemble_rotor(project([RADIAL, AXIAL])).placed["probes"]
    assert [type(p) for p in probes] == [ross.Probe, ross.Probe]
    assert (probes[0].node, probes[0].direction, probes[0].tag) == (2, "radial", "DE X")
    assert probes[0].angle == pytest.approx(0.25 * 3.141592653589793)
    assert (probes[1].node, probes[1].direction, probes[1].angle) == (4, "axial", None)


def test_the_angle_is_read_in_its_own_unit():
    row = dict(RADIAL, angle="0.5", angle_unit="rad")
    assert assemble_rotor(project([row])).placed["probes"][0].angle == 0.5


def test_a_name_that_is_a_number_stays_a_name():
    row = dict(RADIAL, tag="1")
    assert assemble_rotor(project([row])).placed["probes"][0].tag == "1"


def test_probes_take_no_part_in_the_rotor():
    """The same rotor with and without them: ROSS's `Rotor` has no probes."""
    with_probes = build_rotor_from_ui(project([RADIAL, AXIAL]))
    without = build_rotor_from_ui(project([]))
    assert with_probes.ndof == without.ndof
    assert (with_probes.M() == without.M()).all()
    assert (with_probes.K(0) == without.K(0)).all()


@pytest.mark.parametrize(
    "row, message",
    [
        (dict(RADIAL, n="9"), PROBE_NO_NODE % (1, 9, 0, 4)),
        (dict(RADIAL, angle=""), PROBE_NEEDS_ANGLE % 1),
        (dict(RADIAL, angle="abc"), PROBE_ANGLE_UNREADABLE % (1, "abc")),
        (dict(RADIAL, direction="diagonal"), PROBE_DIRECTION % (1, "diagonal")),
    ],
)
def test_a_probe_ross_would_misread_is_refused_by_its_row(row, message):
    with pytest.raises(ValueError) as refused:
        assemble_rotor(project([AXIAL, row]))
    assert str(refused.value) == message.replace("#1", "#2")


def test_an_axial_probe_needs_no_angle():
    assert assemble_rotor(project([AXIAL])).placed["probes"][0].angle is None


# --- the form ---------------------------------------------------------------------


def test_the_form_is_ross_probe_with_the_node_called_n():
    form = build_schema("en")["categories"]["probes"]["BASIC"]
    assert form["ross_class"] == "Probe"
    fields = {field["name"]: field for field in form["fields"]}
    assert set(fields) == {"n", "tag", "direction", "angle"}
    assert all(field["known_to_ross"] for field in fields.values())
    assert fields["direction"]["options"] == ["radial", "axial"]
    assert fields["angle"]["unit"] == "deg"
    assert form["not_in_form"] == []
    assert schema_problems() == []


# --- the scene ----------------------------------------------------------------------


def test_the_scene_places_each_probe_on_its_node():
    scene = describe_scene(assemble_rotor(project([RADIAL, AXIAL])))
    radial, axial = scene["probes"]
    assert radial == {
        "index": 0,
        "n": 2,
        "z": pytest.approx(0.5),
        "direction": "radial",
        "angle": pytest.approx(0.7853981633974483),
        "tag": "DE X",
    }
    assert (axial["n"], axial["z"], axial["angle"]) == (4, pytest.approx(1.0), None)


def test_a_rotor_with_no_probes_has_an_empty_list():
    assert describe_scene(assemble_rotor(project([])))["probes"] == []


# --- a probe follows its node ---------------------------------------------------------


def test_a_split_below_a_probe_moves_it_with_its_node():
    split = split_shaft(project([RADIAL, dict(RADIAL, n="0", tag="NDE")]), 0, "125")
    assert [p["n"] for p in split["probes"]] == ["3", "0"]


def test_a_discretized_shaft_moves_the_probes_past_it():
    meshed, _ = mesh_shafts(project([RADIAL, AXIAL]), parts="2", indexes=[0])
    assert [p["n"] for p in meshed["probes"]] == ["3", "5"]
    # and the probe still reads where it did: the same place along the shaft
    before = describe_scene(assemble_rotor(project([RADIAL, AXIAL])))["probes"]
    after = describe_scene(assemble_rotor(meshed))["probes"]
    assert [p["z"] for p in after] == pytest.approx([p["z"] for p in before])


def test_a_concatenation_carries_the_probes_of_both_rotors():
    joined = concatenated_project(project([RADIAL], shafts=2), project([AXIAL]))
    nodes = [p["n"] for p in joined["probes"]]
    tags = [p["tag"] for p in joined["probes"]]
    assert nodes == ["2", "6"]
    assert tags == ["DE X (R0)", "Thrust (R1)"]
    assert len(describe_scene(assemble_rotor(joined))["probes"]) == 2
