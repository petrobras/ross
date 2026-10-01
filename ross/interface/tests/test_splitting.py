# -*- coding: utf-8 -*-
"""Cutting a shaft in two: the geometry, the numbering, the names, the refusals.

The load-bearing test is `test_the_split_project_builds_the_rotor_ross_would`.
`domain/splitting.py` deliberately does not call `Rotor.add_nodes` -- the module
docstring says why -- and the price of that decision is that the two rules it
restates could drift from ROSS's without anything noticing. This file is what
makes that impossible: the project our module returns is built, the rotor ROSS
would have made is built the other way, and the two are compared node by node
**and diameter by diameter**.

The rest are controls on what a single comparison passes over: the tags (which
`add_nodes` destroys and we keep), the refusals, and the fields of the form that
never reach a `Rotor` at all.
"""

import copy

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.domain.rotor_builder import build_rotor_from_ui  # noqa: E402
from ross.interface.domain.splitting import split_shaft  # noqa: E402

STEEL = {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}

EMPTY = {
    "materials": [],
    "shafts": [],
    "disks": [],
    "gears": [],
    "couplings": [],
    "seals": [],
    "bearings": [],
    "pointmasses": [],
}

# Millimetres, like the form (`domain/units.py` maps `L`, `idl` and `odl` of a
# ShaftElement to mm). Tapered and hollow, and the four diameters all different
# on the first element, because equal numbers hide a wrong interpolation: a
# shaft that is 100 mm across at both ends is split correctly by any rule.
TAPERED = [
    {
        "element_type": "BASIC",
        "L": "400",
        "idl": "10",
        "odl": "100",
        "idr": "20",
        "odr": "200",
        "material": "Steel",
        "tag": "Inlet",
    },
    {
        "element_type": "BASIC",
        "L": "300",
        "idl": "20",
        "odl": "200",
        "material": "Steel",
        "tag": "Body",
    },
    {
        "element_type": "BASIC",
        "L": "300",
        "idl": "20",
        "odl": "200",
        "idr": "0",
        "odr": "100",
        "material": "Steel",
        "tag": "Outlet",
    },
]


def project(with_link=True):
    """A rotor the interface could have built, in the screen's own format."""
    built = copy.deepcopy(EMPTY)
    built["materials"] = [copy.deepcopy(STEEL)]
    built["shafts"] = copy.deepcopy(TAPERED)
    built["disks"] = [
        {"element_type": "BASIC", "n": "2", "m": "10", "Id": "0.2", "Ip": "0.1"}
    ]
    built["bearings"] = [
        {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "0"},
        {"element_type": "BASIC", "n": "3", "kxx": "1e6", "cxx": "0"},
    ]
    if with_link:
        # A point mass does not sit on a shaft node. It sits on a link node past
        # the last shaft node, reached by a bearing with `n_link`, and that link
        # node needs a support of its own -- this project's lesson 1.
        built["bearings"][1]["n_link"] = "4"
        built["bearings"].append(
            {"element_type": "BASIC", "n": "4", "kxx": "1e6", "cxx": "0"}
        )
        built["pointmasses"].append({"element_type": "BASIC", "n": "4", "m": "1.0"})
    return built


def described(rotor):
    """The rotor as numbers: where every node is, and the shape of every span.

    The diameters are in here and not only the node indices, because the whole
    reason `add_nodes` is worth reproducing is the interpolation. Nine decimals
    is a nanometre -- past any dimension a rotor has, and short of where two
    different orders of the same arithmetic start disagreeing.
    """
    return {
        "nodes": sorted(int(n) for n in rotor.nodes),
        "link_nodes": sorted(int(n) for n in rotor.link_nodes),
        "shafts": [
            (
                int(e.n_l),
                int(e.n_r),
                round(float(e.L), 9),
                round(float(e.idl), 9),
                round(float(e.odl), 9),
                round(float(e.idr), 9),
                round(float(e.odr), 9),
            )
            for e in rotor.shaft_elements
        ],
        "disks": sorted(int(e.n) for e in rotor.disk_elements),
        "bearings": sorted(
            (int(e.n), None if e.n_link is None else int(e.n_link))
            for e in rotor.bearing_elements
        ),
        "pointmasses": sorted(int(e.n) for e in rotor.point_mass_elements),
    }


def absolute(data, index, offset_mm):
    """Where the split falls along the whole rotor, in metres.

    Only the test needs this. `split_shaft` is told which element and how far
    along it, which is what the button can ask without the user adding up
    lengths -- `add_nodes` is told a position on the rotor, which is what the
    button would have to compute. The sum lives here, next to the oracle.
    """
    before = sum(float(s["L"]) for s in data["shafts"][:index])
    return (before + offset_mm) / 1000.0


# --- the one that decides the design ------------------------------------------


@pytest.mark.parametrize("with_link", [False, True])
@pytest.mark.parametrize("index,offset", [(0, 100.0), (1, 150.0), (2, 250.0)])
def test_the_split_project_builds_the_rotor_ross_would(with_link, index, offset):
    """Ours and ROSS's, compared element by element.

    Three positions and both topologies: the first element (so the split is at
    the very start of the line), a middle one, and the last (so it is at the
    end). `add_nodes` is asked for the same cut in its own terms.
    """
    data = project(with_link=with_link)

    ours = build_rotor_from_ui(split_shaft(data, index, offset))
    theirs = build_rotor_from_ui(data).add_nodes([absolute(data, index, offset)])

    assert described(ours) == described(theirs)


def test_the_comparison_can_fail():
    """Control on the test above, and it is not ceremony.

    `described` compares dictionaries of numbers; if it read nothing that a
    split changes, every assertion above would hold for any implementation at
    all. So: a rotor split at one place is not the rotor split at another.
    """
    data = project()
    here = build_rotor_from_ui(split_shaft(data, 0, 100.0))
    there = build_rotor_from_ui(split_shaft(data, 0, 300.0))

    assert described(here) != described(there)


# --- the interpolation, in the units the screen speaks ------------------------


def test_the_halves_meet_at_the_interpolated_diameter():
    """A quarter of the way along 100 -> 200 is 125, and it is written twice.

    Read off the project rather than off a rotor, because this is the number the
    form will show the user when they open either half.
    """
    split = split_shaft(project(), 0, 100.0)
    left, right = split["shafts"][0], split["shafts"][1]

    assert float(left["L"]) == 100.0
    assert float(right["L"]) == 300.0
    assert float(left["odr"]) == 125.0
    assert float(right["odl"]) == 125.0
    assert float(left["idr"]) == 12.5
    assert float(right["idl"]) == 12.5
    # The outer faces are still the element's own.
    assert float(left["odl"]) == 100.0
    assert float(right["odr"]) == 200.0


def test_a_blank_right_diameter_means_the_left_one():
    """`Body` is typed with two diameters, which is how a cylinder is typed.

    Reading a blank `odr` as zero would turn every constant section into a cone
    the moment it was split -- and ROSS reads it as `odl`, which is why `_read`
    takes a fallback.
    """
    split = split_shaft(project(), 1, 150.0)
    left, right = split["shafts"][1], split["shafts"][2]

    for row in (left, right):
        assert float(row["odl"]) == 200.0
        assert float(row["odr"]) == 200.0
        assert float(row["idl"]) == 20.0
        assert float(row["idr"]) == 20.0


def test_the_lengths_add_back_up_to_the_original():
    split = split_shaft(project(), 2, 250.0)
    assert float(split["shafts"][2]["L"]) + float(split["shafts"][3]["L"]) == 300.0


def test_a_length_does_not_come_back_as_floating_point_noise():
    """`0.30000000000000004` in a form field is something the user did not type."""
    split = split_shaft(project(), 0, 130.0)
    assert split["shafts"][0]["L"] == "130"
    assert split["shafts"][1]["L"] == "270"


def mixed_units():
    """`TAPERED`'s first element typed with a unit per field, as the form allows.

    The same geometry -- 400 mm long, 10/100 -> 20/200 mm -- with the length
    and two of the diameters in metres, so a split that read the raw numbers
    would interpolate 0.1 against 200.
    """
    data = project()
    data["shafts"][0].update(
        {
            "L": "0.4",
            "L_unit": "m",
            "odl": "0.1",
            "odl_unit": "m",
            "idr": "0.02",
            "idr_unit": "m",
        }
    )
    return data


def test_a_split_across_mixed_units_builds_the_rotor_ross_would():
    """The oracle again, on a row whose fields do not share a unit."""
    data = mixed_units()

    ours = build_rotor_from_ui(split_shaft(data, 0, 0.1))
    theirs = build_rotor_from_ui(data).add_nodes([0.1])

    assert described(ours) == described(theirs)


def test_mixed_units_are_the_same_split_as_millimetres():
    """Control on the test above: the metres rotor is the millimetres rotor."""
    in_metres = build_rotor_from_ui(split_shaft(mixed_units(), 0, 0.1))
    in_millimetres = build_rotor_from_ui(split_shaft(project(), 0, 100.0))

    assert described(in_metres) == described(in_millimetres)


def test_each_half_is_written_in_the_unit_of_the_field_it_lands_in():
    split = split_shaft(mixed_units(), 0, 0.1)
    left, right = split["shafts"][0], split["shafts"][1]

    assert (left["L"], left["L_unit"]) == ("0.1", "m")
    assert (right["L"], right["L_unit"]) == ("0.3", "m")
    assert (left["odl"], left["odl_unit"]) == ("0.1", "m")
    assert (right["odl"], right["odl_unit"]) == ("0.125", "m")
    # `odr` named no unit, so it is the form's millimetres on both halves.
    assert left["odr"] == "125" and "odr_unit" not in left
    assert right["odr"] == "200" and "odr_unit" not in right
    assert (left["idr"], left["idr_unit"]) == ("0.0125", "m")
    assert left["idl"] == "10" and "idl_unit" not in left
    assert right["idl"] == "12.5" and "idl_unit" not in right


def test_a_blank_right_diameter_takes_the_unit_of_the_left_one():
    """A blank `odr` is `odl`'s value, so it has to be `odl`'s unit as well."""
    data = project()
    data["shafts"][1].update({"odl": "0.2", "odl_unit": "m"})
    split = split_shaft(data, 1, 150.0)

    for row in split["shafts"][1:3]:
        assert (row["odl"], row["odl_unit"]) == ("0.2", "m")
        assert (row["odr"], row["odr_unit"]) == ("0.2", "m")


def test_a_diameter_unit_that_is_not_a_length_is_refused():
    data = project()
    data["shafts"][0]["odl_unit"] = "kg"
    with pytest.raises(ValueError) as raised:
        split_shaft(data, 0, 100.0)
    assert "kg" in str(raised.value)


def test_a_refusal_names_the_unit_of_the_length():
    with pytest.raises(ValueError) as raised:
        split_shaft(mixed_units(), 0, 0.4)
    assert "0.4 m long" in str(raised.value)


# --- the names, which is the half `add_nodes` throws away ---------------------


def test_both_halves_carry_the_name_and_the_second_is_told_apart():
    split = split_shaft(project(), 0, 100.0)
    assert split["shafts"][0]["tag"] == "Inlet"
    assert split["shafts"][1]["tag"] == "Inlet (2)"


def test_the_other_elements_keep_their_names():
    """The control that names the reason this module does not call `add_nodes`.

    Measured: `add_nodes` returns every shaft tagged `Shaft Element <i>`,
    including the ones it did not touch. If this project ever starts calling it
    without putting the names back, this is the test that says so.
    """
    split = split_shaft(project(), 0, 100.0)
    assert [s["tag"] for s in split["shafts"]] == [
        "Inlet",
        "Inlet (2)",
        "Body",
        "Outlet",
    ]


def test_a_suffix_that_is_already_taken_moves_on():
    data = project()
    data["shafts"][1]["tag"] = "Inlet (2)"
    split = split_shaft(data, 0, 100.0)
    assert split["shafts"][1]["tag"] == "Inlet (3)"


def test_an_unnamed_shaft_stays_unnamed():
    """Both halves blank, and not one of them named `(2)` out of nowhere."""
    data = project()
    del data["shafts"][0]["tag"]
    split = split_shaft(data, 0, 100.0)
    assert "tag" not in split["shafts"][0]
    assert "tag" not in split["shafts"][1]


# --- the numbering ------------------------------------------------------------


def test_everything_above_the_split_moves_up_by_one():
    split = split_shaft(project(), 0, 100.0)
    assert split["disks"][0]["n"] == "3"
    assert [b["n"] for b in split["bearings"]] == ["0", "4", "5"]
    assert split["bearings"][1]["n_link"] == "5"
    assert split["pointmasses"][0]["n"] == "5"


def test_a_node_below_the_split_does_not_move():
    """Control: a rule that moved every node would pass every test above this.

    Shaft #3 spans nodes 2 and 3, so cutting it puts the new node between them.
    The bearing at 0 and the disk at 2 are the ones that stay -- and node 2 is
    the interesting one, because it is the split element's own left face: the
    boundary is `>`, not `>=`, and the first draft of this test asserted the
    opposite out of pure inattention. The far bearing at 3 does move, and it
    moves for the same reason the disk does not.
    """
    split = split_shaft(project(), 2, 100.0)
    assert [b["n"] for b in split["bearings"]] == ["0", "4", "5"]
    assert split["disks"][0]["n"] == "2"


def test_an_element_that_never_pinned_its_node_gets_pinned_when_it_moves():
    """A blank `n` means "wherever the list puts me", and the list does not move
    when a *shaft* is split -- so the element would keep its number and change
    its place. The node it was standing on is what has to survive."""
    data = project(with_link=False)
    data["disks"] = [
        {"element_type": "BASIC", "m": "10", "Id": "0.2", "Ip": "0.1"},
        {"element_type": "BASIC", "m": "10", "Id": "0.2", "Ip": "0.1"},
    ]
    split = split_shaft(data, 0, 100.0)

    # Disk 0 was resolved to node 0, below the split: still nobody's business.
    assert str(split["disks"][0].get("n", "")).strip() == ""
    # Disk 1 was resolved to node 1, above it: pinned at 2.
    assert split["disks"][1]["n"] == "2"


def test_a_coupling_above_the_split_moves_whether_pinned_or_not():
    """A coupling with a blank `n` is built on its place in the list, and the
    list of couplings does not move when a shaft is split."""
    data = project(with_link=False)
    coupling = {"m_l": "1", "m_r": "1", "Ip_l": "0.01", "Ip_r": "0.01"}
    data["couplings"] = [dict(coupling, n="0"), dict(coupling), dict(coupling)]
    split = split_shaft(data, 0, 100.0)

    # The split is on node 0: the coupling pinned there stays, and the two
    # blanks, on 1 and 2 by their place in the list, move up and are pinned.
    assert [c.get("n") for c in split["couplings"]] == ["0", "2", "3"]


def test_a_shaft_that_pinned_its_node_is_renumbered_with_the_rest():
    data = project(with_link=False)
    for position, shaft in enumerate(data["shafts"]):
        shaft["n"] = str(position)
    split = split_shaft(data, 0, 100.0)
    assert [s["n"] for s in split["shafts"]] == ["0", "1", "2", "3"]


def test_the_project_that_came_in_is_not_the_one_that_goes_out():
    """Splitting adds a rotor to the screen, it does not quietly edit the old
    one -- and the caller still holds the original."""
    data = project()
    before = copy.deepcopy(data)
    split_shaft(data, 0, 100.0)
    assert data == before


# --- the refusals -------------------------------------------------------------


def test_a_split_at_the_left_face_is_refused_by_the_node_it_would_land_on():
    """ROSS answers this request by doing nothing -- and still wiping the tags.
    A button that sometimes does nothing is worse than one that says why."""
    with pytest.raises(ValueError) as raised:
        split_shaft(project(), 1, 0.0)
    assert "node 1" in str(raised.value)


def test_a_split_at_the_right_face_names_the_node_on_that_side():
    with pytest.raises(ValueError) as raised:
        split_shaft(project(), 1, 300.0)
    assert "node 2" in str(raised.value)


def test_a_split_past_the_end_of_the_element_is_refused():
    with pytest.raises(ValueError) as raised:
        split_shaft(project(), 0, 4000.0)
    assert "400" in str(raised.value)


def test_a_distance_that_is_not_a_number_is_refused_showing_what_was_typed():
    with pytest.raises(ValueError) as raised:
        split_shaft(project(), 0, "half way")
    assert "half way" in str(raised.value)


@pytest.mark.parametrize("bad", ["inf", "nan", "1_0", "0x10"])
def test_the_things_float_would_have_accepted_are_refused(bad):
    """`float('inf')` and `float('1_0')` both work in Python, and neither is a
    distance. The node syntax of this project has been here before."""
    with pytest.raises(ValueError):
        split_shaft(project(), 0, bad)


def test_a_shaft_that_does_not_exist_is_refused_saying_how_many_there_are():
    with pytest.raises(ValueError) as raised:
        split_shaft(project(), 7, 100.0)
    assert "has 3" in str(raised.value)


def test_a_shaft_with_no_length_is_refused_by_number():
    data = project()
    data["shafts"][1]["L"] = ""
    with pytest.raises(ValueError) as raised:
        split_shaft(data, 1, 100.0)
    assert "#2" in str(raised.value)
