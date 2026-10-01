# -*- coding: utf-8 -*-
"""Joining two rotors: the numbering, the materials and the refusals.

The load-bearing test here is `test_the_joined_project_builds_the_rotor_ross_
concatenates`. Everything the screen shows after a concatenation comes from the
project the domain returns, and the only claim worth making about that project
is that **building it gives the rotor ROSS would have given**. Not "the offsets
look right": the same rotor, element by element, node by node.

The rest are controls on the parts a single comparison would pass over -- the
shared joint node, the link nodes of both halves moving, the materials, and the
four refusals.
"""

import copy

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.domain.concatenation import (  # noqa: E402
    A_MULTIROTOR,
    concatenated_project,
    model_of,
    structural_nodes,
)
from ross.interface.domain.rotor_builder import build_rotor_from_ui  # noqa: E402

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


def project(shafts=2, disk_at="1", bearing_at="0", material=None, with_link=False):
    """A rotor the interface could have built, in the screen's own format.

    `with_link` reproduces the topology ROSS requires of a point mass, and it is
    written out rather than improvised because improvising it is this project's
    lesson 1: a point mass does not sit on a shaft node. It sits on a *link*
    node past the last shaft node, reached by a bearing with `n_link`, and that
    link node needs a support of its own -- without it `Rotor.__init__` raises
    `ValueError: n is not in list` while assembling.
    """
    built = copy.deepcopy(EMPTY)
    built["materials"] = [copy.deepcopy(material or STEEL)]
    name = built["materials"][0]["name"]
    for _ in range(shafts):
        # Millimetres. `domain/units.py:12` maps `L`, `idl` and `odl` of a
        # ShaftElement to `mm`, because that is the unit the form asks for --
        # this dictionary is what the screen sends, not what ROSS receives.
        #
        # The first version of this said `"L": "0.25", "odl": "0.05"`, meaning
        # a quarter-metre shaft of 50 mm. It built a shaft a quarter of a
        # *millimetre* long, and a four-element rotor one millimetre end to
        # end. Every test here passed, because all of them assert node numbers
        # and node numbers do not depend on length -- so the mistake was
        # invisible until the `add_nodes` probe, where geometry is the whole
        # point, and it cost a probe run there.
        built["shafts"].append(
            {
                "element_type": "BASIC",
                "L": "250",
                "idl": "0",
                "odl": "50",
                "material": name,
            }
        )
    if disk_at is not None:
        built["disks"].append(
            {"element_type": "BASIC", "n": disk_at, "m": "10", "Id": "0.2", "Ip": "0.1"}
        )
    if bearing_at is not None:
        built["bearings"].append(
            {"element_type": "BASIC", "n": bearing_at, "kxx": "1e6", "cxx": "0"}
        )
    if with_link:
        link = str(shafts + 1)
        built["bearings"][0]["n_link"] = link
        built["bearings"].append(
            {"element_type": "BASIC", "n": link, "kxx": "1e6", "cxx": "0"}
        )
        built["pointmasses"].append({"element_type": "BASIC", "n": link, "m": "1.0"})
    return built


def described(rotor):
    """The rotor as node numbers, which is what a concatenation changes."""
    return {
        "nodes": sorted(int(n) for n in rotor.nodes),
        "link_nodes": sorted(int(n) for n in rotor.link_nodes),
        "shafts": [(int(e.n_l), int(e.n_r)) for e in rotor.shaft_elements],
        "disks": sorted(int(e.n) for e in rotor.disk_elements),
        "bearings": sorted(
            (int(e.n), None if e.n_link is None else int(e.n_link))
            for e in rotor.bearing_elements
        ),
        "pointmasses": sorted(int(e.n) for e in rotor.point_mass_elements),
    }


# --- the fixture is a rotor, and that is checkable -----------------------------


def test_the_test_rotor_has_the_dimensions_of_a_rotor():
    """Control on `project()`, and it exists because it would have caught a bug.

    The fixture is written in the screen's units and read by `build_rotor_from_ui`
    in ROSS's, and the conversion is silent. A quarter-metre shaft written as
    `0.25` builds a quarter-*millimetre* one; every test in this file still
    passes, because they all assert node numbers and node numbers do not care
    how long a shaft is.

    So the guard is not on the number, it is on the physics: a rotor a
    millimetre long with a shaft fifty microns across is not a rotordynamics
    problem, whatever the tests say about it. The bounds are wide on purpose --
    what is being caught is three orders of magnitude, not a taste in rotors."""
    rotor = build_rotor_from_ui(project(shafts=4))

    assert 0.01 < float(rotor.L) < 100.0, (
        "the test rotor is %g m long end to end -- check the units of `L`, "
        "which the form asks for in millimetres" % float(rotor.L)
    )
    for shaft in rotor.shaft_elements:
        assert 0.001 < float(shaft.odl) < 5.0, (
            "a test shaft is %g m across -- check the units of `odl`" % float(shaft.odl)
        )


# --- the one that decides the design ------------------------------------------


@pytest.mark.parametrize("with_link", [False, True])
def test_the_joined_project_builds_the_rotor_ross_concatenates(with_link):
    """The project we return, built, is the rotor `Rotor.concatenate` returns.

    This is the whole claim of the slice in one assertion. The domain does not
    reimplement ROSS's offsets -- it asks ROSS and writes the answer into the
    project -- so what has to be proved is that the round trip through our own
    format loses none of it.
    """
    first = project(shafts=2, with_link=with_link)
    second = project(shafts=1, with_link=with_link)

    theirs = ross.Rotor.concatenate(
        build_rotor_from_ui(first), build_rotor_from_ui(second)
    )
    ours = build_rotor_from_ui(concatenated_project(first, second))

    assert described(ours) == described(theirs)


def test_the_comparison_can_tell_two_rotors_apart():
    """Control: the comparison above would notice if it were wrong.

    A comparison that passes on everything proves nothing, and `described` is
    hand-written. Two rotors that differ by one node have to come out different.
    """
    one = build_rotor_from_ui(project(shafts=2))
    other = build_rotor_from_ui(project(shafts=3))
    assert described(one) != described(other)


def with_coupling(second_coupling=False):
    """A shaft line broken by a coupling on nodes 2-3, pinned as the form must.

    The shafts after the coupling pin their nodes, because a blank `n` on a
    shaft would be resolved without looking at the coupling. With
    `second_coupling`, a coupling with a blank `n` follows a pinned one: the
    case where the builder's rule (its place in the list, node 1) and
    `effective_nodes` (the lowest free node, 0) disagree.
    """
    built = project(shafts=4, disk_at="1", bearing_at="0")
    built["shafts"][2]["n"] = "3"
    built["shafts"][3]["n"] = "4"
    coupling = {
        "element_type": "BASIC",
        "m_l": "1",
        "m_r": "1",
        "Ip_l": "0.01",
        "Ip_r": "0.01",
        "kt_x": "1e7",
        "kt_y": "1e7",
        "kr_x": "1e5",
        "kr_y": "1e5",
        "L": "100",
    }
    built["couplings"].append(dict(coupling, n="2", tag="Flex"))
    if second_coupling:
        built["couplings"].append(dict(coupling))
    built["bearings"].append(
        {"element_type": "BASIC", "n": "5", "kxx": "1e6", "cxx": "0"}
    )
    return built


@pytest.mark.parametrize("second_coupling", [False, True])
@pytest.mark.parametrize("coupled", ["first", "second", "both"])
def test_a_coupling_lands_where_ross_puts_it(coupled, second_coupling):
    """The comparison above, with a coupling in either half or in both."""
    plain = project(shafts=2)
    first = with_coupling(second_coupling) if coupled != "second" else plain
    second = with_coupling(second_coupling) if coupled != "first" else plain

    theirs = ross.Rotor.concatenate(
        build_rotor_from_ui(first), build_rotor_from_ui(second)
    )
    ours = build_rotor_from_ui(concatenated_project(first, second))

    assert described(ours) == described(theirs)


def test_a_blank_coupling_is_numbered_by_its_place_in_the_list():
    """The builder's rule, written into the joined project as an explicit `n`.

    The second rotor starts on node 2, the end of a two-shaft first rotor, so
    its couplings on 2 and 1 land on 4 and 3.
    """
    merged = concatenated_project(project(shafts=2), with_coupling(True))
    assert [c["n"] for c in merged["couplings"]] == ["4", "3"]


def test_the_screen_counts_a_coupling_as_the_builder_does():
    for data in (with_coupling(), with_coupling(True)):
        built = build_rotor_from_ui(data)
        assert structural_nodes(data) == max(int(n) for n in built.nodes)


# --- what the comparison alone would pass over --------------------------------


def test_the_joint_node_is_shared_and_not_doubled():
    """Two rotors of 2 and 1 shafts give four nodes, not five.

    `node_offset += max(rotor.nodes)` means the second rotor starts on the node
    the first ends on. The rotors are joined, not queued with a gap -- and the
    number is the only way to say which of the two it is.
    """
    joined = build_rotor_from_ui(
        concatenated_project(project(shafts=2), project(shafts=1))
    )
    assert sorted(int(n) for n in joined.nodes) == [0, 1, 2, 3]
    assert [(int(e.n_l), int(e.n_r)) for e in joined.shaft_elements] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_the_link_nodes_of_both_halves_move():
    """Not only the second rotor's: ROSS renumbers every link node.

    They go past every structural node of the joined rotor, in the order
    `link_nodes` lists them. The first rotor's link node moves too, which is the
    part that an offset applied only to the second rotor would get wrong -- and
    it is why this is read back from ROSS instead of computed here.
    """
    first, second = project(shafts=2, with_link=True), project(shafts=1, with_link=True)
    assert sorted(int(n) for n in build_rotor_from_ui(first).link_nodes) == [3]
    assert sorted(int(n) for n in build_rotor_from_ui(second).link_nodes) == [2]

    joined = build_rotor_from_ui(concatenated_project(first, second))
    assert sorted(int(n) for n in joined.nodes) == [0, 1, 2, 3]
    assert sorted(int(n) for n in joined.link_nodes) == [4, 5]


def test_every_parameter_the_user_typed_survives():
    """Only `n`, `n_link`, `material` and `tag` may differ from the source.

    The reason the design does not round-trip through `Rotor.save()` is that it
    would flatten a solver-based bearing into a coefficient table. Nothing here
    may do the same by another route.
    """
    first = project(shafts=2)
    second = project(shafts=1)
    merged = concatenated_project(first, second)

    changeable = {"n", "n_link", "material", "tag"}
    for category in ("shafts", "disks", "bearings"):
        source = first[category] + second[category]
        assert len(merged[category]) == len(source)
        for original, row in zip(source, merged[category], strict=True):
            for key, value in original.items():
                if key in changeable:
                    continue
                assert row[key] == value, "%s/%s changed" % (category, key)


# --- materials ----------------------------------------------------------------


def test_the_same_material_is_not_duplicated():
    merged = concatenated_project(project(shafts=2), project(shafts=1))
    assert [m["name"] for m in merged["materials"]] == ["Steel"]


def test_a_renamed_material_carries_no_space():
    """ROSS refuses a material name with a space, so the tag convention is out.

    `materials.py:75` raises `Spaces are not allowed in Material name`. The
    suffix ROSS puts on a *tag* is ` (R1)`, and borrowing it here -- which is
    what the first version did -- raises while building the merged rotor. A
    material name and a tag look alike and are not the same grammar."""
    softer = dict(STEEL, E="200e9")
    merged = concatenated_project(project(shafts=2), project(shafts=1, material=softer))
    for material in merged["materials"]:
        assert " " not in material["name"], material["name"]
    build_rotor_from_ui(merged)


def test_a_name_that_collides_with_other_properties_is_renamed():
    """The collision that would otherwise build half a rotor out of the wrong metal.

    `rotor_builder` answers an unknown material name with
    `list(mat_dict.values())[0]` -- the first of the list, silently. Two steels
    of different stiffness under one name is exactly that case, and nothing
    would raise.
    """
    softer = dict(STEEL, E="200e9")
    merged = concatenated_project(project(shafts=2), project(shafts=1, material=softer))

    assert [m["name"] for m in merged["materials"]] == ["Steel", "Steel_R1"]
    assert merged["shafts"][0]["material"] == "Steel"
    assert merged["shafts"][-1]["material"] == "Steel_R1"

    stiffness = {m["name"]: m["E"] for m in merged["materials"]}
    assert stiffness == {"Steel": "211e9", "Steel_R1": "200e9"}


def test_the_renamed_material_reaches_the_built_rotor():
    """Control: the rename is not cosmetic.

    Renaming the material and leaving the element pointing at the old name would
    pass the test above and still build the wrong rotor, because the lookup falls
    back to the first material instead of raising.
    """
    softer = dict(STEEL, E="200e9")
    merged = concatenated_project(project(shafts=2), project(shafts=1, material=softer))
    built = build_rotor_from_ui(merged)
    moduli = [float(e.material.E) for e in built.shaft_elements]
    assert moduli[:2] == [211e9, 211e9]
    assert moduli[-1] == 200e9


# --- the rotor model ----------------------------------------------------------


def test_a_rotor_with_no_analyses_has_no_model():
    """And the difference from "every analysis was 6 DoF" is the point.

    The frontend's `unanimousConversion` folds the two together, which is
    harmless for the export and not harmless here: a rotor nobody has analysed
    would stop being concatenable with a 4 DoF one.
    """
    assert model_of([]) is None
    assert model_of([""]) == ""
    assert model_of(["4dof", "4dof"]) == "4dof"
    assert model_of(["4dof", ""]) == "mixed"


def test_two_rotors_with_no_analyses_concatenate():
    merged = concatenated_project(project(shafts=2), project(shafts=1))
    assert len(merged["shafts"]) == 3


def test_a_rotor_with_no_analyses_concatenates_with_a_4dof_one():
    """The case the naive rule would have refused, and nobody would know why."""
    merged = concatenated_project(
        project(shafts=2),
        project(shafts=1),
        first_conversions=(),
        second_conversions=("4dof", "4dof"),
    )
    assert len(merged["shafts"]) == 3


def test_two_models_that_disagree_are_refused_by_name():
    with pytest.raises(ValueError) as error:
        concatenated_project(
            project(shafts=2),
            project(shafts=1),
            first_conversions=("4dof",),
            second_conversions=("",),
        )
    assert "4 DoF" in str(error.value) and "6 DoF" in str(error.value)


def test_a_rotor_whose_own_analyses_disagree_is_refused_alone():
    """Refused without looking at the other rotor.

    Without this, a rotor with mixed cards would slip through whenever the other
    one happened to have no analyses -- the guard would exist and still let
    through the case that motivated it.
    """
    with pytest.raises(ValueError) as error:
        concatenated_project(
            project(shafts=2),
            project(shafts=1),
            first_conversions=("4dof", "torsional"),
            second_conversions=(),
        )
    assert "more than one rotor model" in str(error.value)


def test_a_multirotor_is_refused_by_name():
    """It is a different thing, and the message says which.

    Two shafts coupled through a gear mesh, each at its own speed, is not a
    shaft line. The refusal has to say that, because the two buttons sit side by
    side in the hub.
    """
    multi = dict(copy.deepcopy(EMPTY), isMultiRotor=True)
    with pytest.raises(ValueError) as error:
        concatenated_project(multi, project(shafts=1))
    assert str(error.value) == A_MULTIROTOR


# --- what the screen asks before committing -----------------------------------


def test_the_screen_can_say_where_the_joint_falls():
    """`structural_nodes` answers without building anything.

    It is not what decides the concatenation -- that is read back from ROSS --
    so this test exists to keep the cheap answer equal to the expensive one.
    """
    for shafts in (1, 2, 5):
        described_here = structural_nodes(project(shafts=shafts))
        built = build_rotor_from_ui(project(shafts=shafts))
        assert described_here == max(int(n) for n in built.nodes)


# --- the route ----------------------------------------------------------------


def client():
    """The application and its header, the way the other route suites build it."""
    from ross.interface.api import create_app
    from ross.interface.api.security import SESSION_TOKEN

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), {"X-ROSS-Token": SESSION_TOKEN}


def test_the_route_answers_a_project_and_not_a_chart():
    """What comes back is editable, which is the point of the whole design.

    `/load_ross_file` already answers `projectData`, and this uses the same key
    on purpose: the hub does the same thing with both -- puts a rotor in the
    library. A route that answered a figure would be a dead end with a picture
    on it.
    """
    page, headers = client()
    answer = page.post(
        "/api/rotor/concatenate",
        json={"first": project(shafts=2), "second": project(shafts=1)},
        headers=headers,
    )
    assert answer.status_code == 200, answer.get_json()
    merged = answer.get_json()["projectData"]
    assert len(merged["shafts"]) == 3
    assert [row["n"] for row in merged["shafts"]] == ["0", "1", "2"]


def test_a_refusal_reaches_the_user_as_a_400_with_its_message():
    """Not a 500 about an internal error.

    The domain raises `ValueError` and `api/errors.py` turns it into a 400 that
    carries the sentence. Without it the user meets an internal-error page and
    has no way of knowing that the two rotors disagree about the model.
    """
    page, headers = client()
    answer = page.post(
        "/api/rotor/concatenate",
        json={
            "first": project(shafts=2),
            "second": project(shafts=1),
            "first_conversions": ["4dof"],
            "second_conversions": [""],
        },
        headers=headers,
    )
    assert answer.status_code == 400
    message = answer.get_json()["message"]
    assert "4 DoF" in message and "6 DoF" in message


def test_a_body_without_the_two_rotors_is_refused_by_name():
    """Control: the envelope is doing its job, and says which field is missing."""
    page, headers = client()
    answer = page.post(
        "/api/rotor/concatenate", json={"first": project(shafts=2)}, headers=headers
    )
    assert answer.status_code == 400
    assert "second" in answer.get_json()["message"]
