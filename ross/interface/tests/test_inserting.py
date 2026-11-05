# -*- coding: utf-8 -*-
"""A shaft added on a face of another (domain/inserting.py), held to ROSS.

The sketch's extrusion (frontend/features/sketch3d.js) adds an element on the
start or the end face of a shaft. The reference is the rotor ROSS builds from
the result, against the one it builds from the original: the new element
spans its length from the face, what sits on the face stays on it, and
everything past the face is where it was plus that length. Those are
positions along the rotor, which ROSS computes from the lengths alone; no
node number is taken from the code under test.
"""

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.domain.inserting import insert_shaft  # noqa: E402
from ross.interface.domain.rotor_builder import build_rotor_from_ui  # noqa: E402

STEEL = {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}
NEW = {
    "element_type": "BASIC",
    "L": "100",
    "idl": "0",
    "odl": "80",
    "material": "Steel",
}
L_NEW = 0.1


def shaft(**extra):
    return dict(
        {
            "element_type": "BASIC",
            "L": "250",
            "idl": "0",
            "odl": "50",
            "material": "Steel",
        },
        **extra,
    )


def project():
    """Three shafts (nodes 0..3, every 0.25 m); a disk on the joint at node 1;
    a probe at node 2; bearings at both ends, the right one on a support at a
    link node with a point mass on it."""
    return {
        "materials": [dict(STEEL)],
        "shafts": [shaft(tag="A"), shaft(tag="B"), shaft(tag="C")],
        "disks": [
            {
                "element_type": "BASIC",
                "n": "1",
                "m": "5",
                "Id": "0.02",
                "Ip": "0.04",
                "tag": "D",
            }
        ],
        "bearings": [
            {"element_type": "BASIC", "n": "0", "kxx": "1e7", "cxx": "0", "tag": "B0"},
            {
                "element_type": "BASIC",
                "n": "3",
                "n_link": "4",
                "kxx": "1e7",
                "cxx": "0",
                "tag": "B3",
            },
            {"element_type": "BASIC", "n": "4", "kxx": "1e8", "cxx": "0", "tag": "S"},
        ],
        "pointmasses": [{"n": "4", "m": "10", "tag": "P"}],
        "probes": [{"id": "p1", "n": "2", "angle": "0", "tag": "Pr"}],
        "couplings": [],
        "gears": [],
        "seals": [],
    }


def z_of(rotor, kind, tag):
    elements = {
        "disk": rotor.disk_elements,
        "bearing": rotor.bearing_elements,
        "pointmass": rotor.point_mass_elements,
    }[kind]
    element = next(e for e in elements if e.tag == tag)
    return rotor.nodes_pos[element.n] if element.n < len(rotor.nodes_pos) else None


def shaft_spans(rotor):
    """(z start, length, outer diameter) of each shaft, in the order of the line."""
    return sorted(
        (round(rotor.nodes_pos[e.n], 9), round(e.L, 9), round(e.odl, 9))
        for e in rotor.shaft_elements
    )


@pytest.fixture(scope="module")
def before():
    return build_rotor_from_ui(project())


def test_on_the_end_face_of_the_last_shaft_the_line_grows_past_it(before):
    after = build_rotor_from_ui(insert_shaft(project(), 2, "end", NEW))
    spans = shaft_spans(after)
    assert spans[-1] == (0.75, L_NEW, 0.08)
    assert len(after.nodes_pos) == len(before.nodes_pos) + 1
    # The bearing on that face stays on it; the shaft now overhangs it.
    assert z_of(after, "bearing", "B3") == pytest.approx(0.75)
    assert z_of(after, "disk", "D") == pytest.approx(0.25)


def test_on_the_end_face_of_a_middle_shaft_what_is_past_it_moves_along(before):
    after = build_rotor_from_ui(insert_shaft(project(), 0, "end", NEW))
    spans = shaft_spans(after)
    assert spans[1] == (0.25, L_NEW, 0.08)
    # The two shafts after it start L later.
    assert [s[0] for s in spans[2:]] == pytest.approx([0.25 + L_NEW, 0.5 + L_NEW])
    # The disk on the face sketched on stays there, on the end of A.
    assert z_of(after, "disk", "D") == pytest.approx(0.25)
    assert z_of(after, "bearing", "B0") == pytest.approx(0.0)
    assert z_of(after, "bearing", "B3") == pytest.approx(0.75 + L_NEW)


def test_on_the_start_face_of_a_shaft_its_face_and_what_is_on_it_move_with_it(before):
    after = build_rotor_from_ui(insert_shaft(project(), 1, "start", NEW))
    spans = shaft_spans(after)
    assert spans[1] == (0.25, L_NEW, 0.08)
    assert [s[0] for s in spans[2:]] == pytest.approx([0.25 + L_NEW, 0.5 + L_NEW])
    # Sketched on B's start face, the disk on that face goes with B.
    assert z_of(after, "disk", "D") == pytest.approx(0.25 + L_NEW)
    assert z_of(after, "bearing", "B0") == pytest.approx(0.0)


def test_on_the_start_face_of_the_first_shaft_the_whole_rotor_moves_along(before):
    after = build_rotor_from_ui(insert_shaft(project(), 0, "start", NEW))
    spans = shaft_spans(after)
    assert spans[0] == (0.0, L_NEW, 0.08)
    assert [s[0] for s in spans[1:]] == pytest.approx(
        [L_NEW, 0.25 + L_NEW, 0.5 + L_NEW]
    )
    assert z_of(after, "bearing", "B0") == pytest.approx(L_NEW)
    assert z_of(after, "disk", "D") == pytest.approx(0.25 + L_NEW)


def test_the_link_node_and_what_hangs_on_it_follow():
    built = insert_shaft(project(), 0, "end", NEW)
    rotor = build_rotor_from_ui(built)
    support = next(b for b in built["bearings"] if b["tag"] == "B3")
    # Still a link node beyond the line, and still the support's and the mass's.
    assert int(support["n_link"]) == len(rotor.shaft_elements) + 1
    assert (
        next(b for b in built["bearings"] if b["tag"] == "S")["n"] == support["n_link"]
    )
    assert built["pointmasses"][0]["n"] == support["n_link"]


def test_a_probe_past_the_face_follows_its_node():
    built = insert_shaft(project(), 0, "end", NEW)
    assert built["probes"][0]["n"] == "3"
    built = insert_shaft(project(), 2, "end", NEW)
    assert built["probes"][0]["n"] == "2"


def test_shafts_that_pin_their_nodes_give_the_new_one_its_node():
    data = project()
    for k, row in enumerate(data["shafts"]):
        row["n"] = str(k)
    built = insert_shaft(data, 0, "end", NEW)
    assert [s.get("n") for s in built["shafts"]] == ["0", "1", "2", "3"]
    built = insert_shaft(data, 1, "start", NEW)
    assert [s.get("n") for s in built["shafts"]] == ["0", "1", "2", "3"]
    spans = shaft_spans(build_rotor_from_ui(built))
    assert spans[1] == (0.25, L_NEW, 0.08)


def test_a_coupling_starting_on_the_face_moves_past_the_new_shaft():
    data = project()
    data["shafts"] = [
        shaft(n="0", tag="A"),
        shaft(n="2", tag="B"),
        shaft(n="3", tag="C"),
    ]
    data["couplings"] = [
        {
            "element_type": "BASIC",
            "n": "1",
            "m_l": "1",
            "m_r": "1",
            "Ip_l": "0.01",
            "Ip_r": "0.01",
            "L": "100",
        }
    ]
    data["disks"] = []
    data["bearings"] = [
        {"element_type": "BASIC", "n": "0", "kxx": "1e7", "cxx": "0", "tag": "B0"},
        {"element_type": "BASIC", "n": "4", "kxx": "1e7", "cxx": "0", "tag": "B4"},
    ]
    data["pointmasses"] = []
    data["probes"] = []
    built = insert_shaft(data, 0, "end", NEW)
    assert built["couplings"][0]["n"] == "2"
    assert [s["n"] for s in built["shafts"]] == ["0", "1", "3", "4"]
    rotor = build_rotor_from_ui(built)
    starts = sorted(
        (round(rotor.nodes_pos[e.n], 9), type(e).__name__) for e in rotor.shaft_elements
    )
    assert starts[1] == (0.25, "ShaftElement")
    assert starts[2] == (pytest.approx(0.35), "CouplingElement")
    assert z_of(rotor, "bearing", "B4") == pytest.approx(0.25 + L_NEW + 0.1 + 0.5)


def test_the_original_project_is_not_touched():
    data = project()
    insert_shaft(data, 0, "end", NEW)
    assert data == project()


def test_the_new_row_is_the_one_given():
    built = insert_shaft(project(), 2, "end", dict(NEW, n="9"))
    assert built["shafts"][3] == NEW


@pytest.mark.parametrize(
    "index, side, row, words",
    [
        (3, "end", NEW, "Shaft #4 does not exist"),
        (-1, "end", NEW, "does not exist"),
        (0, "middle", NEW, "'middle'"),
        (0, "end", "row", "row of the shaft list"),
    ],
)
def test_what_cannot_be_inserted_is_refused_by_name(index, side, row, words):
    with pytest.raises(ValueError, match=words):
        insert_shaft(project(), index, side, row)


def test_the_route_answers_the_project():
    from ross.interface.api.security import SESSION_TOKEN
    from ross.interface.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        answer = client.post(
            "/api/rotor/insert_shaft",
            json={"project": project(), "index": 2, "side": "end", "row": NEW},
            headers={"X-ROSS-Token": SESSION_TOKEN},
        )
        refused = client.post(
            "/api/rotor/insert_shaft",
            json={"project": project(), "index": 7, "side": "end", "row": NEW},
            headers={"X-ROSS-Token": SESSION_TOKEN},
        )
    assert answer.status_code == 200
    assert len(answer.get_json()["projectData"]["shafts"]) == 4
    assert refused.status_code == 400 and "Shaft #8" in refused.get_json()["message"]


def test_a_taper_on_a_start_face_meets_the_face_with_its_right_end():
    """Sketched on a start face, the element grows -z: the sketch is its
    right end (ROSS's `odr`, at the higher z), the far face its left
    (frontend/core/sketch3d.js, `shaftRow` with sense -1). Built by ROSS, the
    new element's right end has the diameter of the face it was drawn on."""
    taper = dict(NEW, odl="80", idl="0", odr="50", idr="0")
    rotor = build_rotor_from_ui(insert_shaft(project(), 1, "start", taper))
    new, b = sorted(rotor.shaft_elements, key=lambda e: e.n)[1:3]
    assert (new.odl, new.odr) == pytest.approx((0.08, 0.05))
    assert new.odr == pytest.approx(b.odl)
