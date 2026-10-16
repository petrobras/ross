# -*- coding: utf-8 -*-
"""Cutting the shaft line by the length-to-diameter rule (API RP 684, 1.5.2.1).

The load-bearing test is `test_the_meshed_project_builds_the_rotor_ross_would`:
`domain/meshing.py` does not call `Rotor.refine_mesh` -- its docstring says
why -- so the project it returns is built, the rotor ROSS refines is built the
other way, and the two are compared node by node and diameter by diameter, as
`tests/test_splitting.py` does for a single split.
"""

import copy
import warnings

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.domain.meshing import (  # noqa: E402
    mesh_by_convergence,
    mesh_shafts,
    plan,
)
from ross.interface.domain.rotor_builder import build_rotor_from_ui  # noqa: E402
from ross.interface.tests.test_splitting import described, project  # noqa: E402


def refined(data, max_ld):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_rotor_from_ui(data).refine_mesh(max_ld=max_ld)


# --- the one that decides the design ------------------------------------------


@pytest.mark.parametrize("with_link", [False, True])
@pytest.mark.parametrize("max_ld", [1.0, 0.5, 0.25])
def test_the_meshed_project_builds_the_rotor_ross_would(with_link, max_ld):
    data = project(with_link=with_link)
    meshed, _ = mesh_shafts(data, max_ld=max_ld)

    assert described(build_rotor_from_ui(meshed)) == described(refined(data, max_ld))


def test_the_comparison_can_fail():
    """Control: meshes at two ratios are two different rotors."""
    data = project()
    coarse, _ = mesh_shafts(data, max_ld=1.0)
    fine, _ = mesh_shafts(data, max_ld=0.5)

    assert described(build_rotor_from_ui(coarse)) != described(
        build_rotor_from_ui(fine)
    )


def test_units_other_than_the_forms_are_read_as_what_they_are():
    """A length typed in metres next to diameters in inches: the ratio is the
    same number whatever the units, and the pieces are written back in each
    field's own unit."""
    data = project()
    first = data["shafts"][0]
    first["L"] = "0.4"
    first["L_unit"] = "m"
    first["odl"] = str(100 / 25.4)
    first["odl_unit"] = "inch"
    meshed, _ = mesh_shafts(data, max_ld=0.5)

    assert described(build_rotor_from_ui(meshed)) == described(refined(data, 0.5))
    assert meshed["shafts"][0]["L_unit"] == "m"
    assert float(meshed["shafts"][0]["L"]) == pytest.approx(0.1)
    assert meshed["shafts"][0]["odl_unit"] == "inch"


def test_elements_on_the_same_nodes_are_cut_together_by_the_largest():
    """A sleeve over a shaft: two rows on the same pair of nodes."""
    data = project(with_link=False)
    sleeve = copy.deepcopy(data["shafts"][1])
    sleeve.update(
        {"n": "1", "idl": "200", "odl": "400", "idr": "", "odr": "", "tag": "Sleeve"}
    )
    data["shafts"][1]["n"] = "1"
    data["shafts"].insert(2, sleeve)
    meshed, report = mesh_shafts(data, max_ld=0.5)

    assert described(build_rotor_from_ui(meshed)) == described(refined(data, 0.5))
    # 300 mm under a 400 mm sleeve: two parts, not the three the 200 mm body alone
    # would take.
    assert {"rows": [1, 2], "parts": 2, "ratio": pytest.approx(0.75)} in report["cut"]


def test_a_coupling_is_not_cut():
    data = project(with_link=False)
    data["shafts"][2]["n"] = "3"
    data["couplings"] = [
        {
            "n": "2",
            "m_l": "1",
            "m_r": "1",
            "Ip_l": "0.01",
            "Ip_r": "0.01",
            "o_d": "50",
            "L": "200",
        }
    ]
    data["bearings"][1]["n"] = "4"
    meshed, report = mesh_shafts(data, max_ld=0.5)

    assert described(build_rotor_from_ui(meshed)) == described(refined(data, 0.5))
    # Inlet in 4 and Body in 3 add five nodes below it; its own span is whole.
    assert meshed["couplings"][0]["n"] == "7"
    assert meshed["couplings"][0]["L"] == "200"


def test_a_shaft_alongside_a_coupling_is_left_whole_and_said_so():
    data = project(with_link=False)
    data["shafts"][2]["n"] = "2"
    data["couplings"] = [
        {
            "n": "2",
            "m_l": "1",
            "m_r": "1",
            "Ip_l": "0.01",
            "Ip_r": "0.01",
            "o_d": "50",
            "L": "300",
        }
    ]
    meshed, report = mesh_shafts(data, max_ld=0.5)

    assert described(build_rotor_from_ui(meshed)) == described(refined(data, 0.5))
    assert report["kept"] == [{"rows": [2], "why": "coupling"}]
    assert [s.get("tag") for s in meshed["shafts"]].count("Outlet") == 1


# --- what the project looks like afterwards ------------------------------------


def test_the_pieces_are_named_after_the_element_and_keep_their_place():
    meshed, report = mesh_shafts(project(with_link=False), max_ld=1.0)
    tags = [s.get("tag") for s in meshed["shafts"]]

    # 400 / 200 -> 2 parts; 300 / 200 -> 2; 300 / 200 -> 2.
    assert tags == ["Inlet", "Inlet (2)", "Body", "Body (2)", "Outlet", "Outlet (2)"]
    assert report["before"] == 3 and report["after"] == 6
    # The disk on node 2 (between Body and Outlet) gains the node cut in
    # Inlet and the one cut in Body; the bearing at the end gains all three.
    assert meshed["disks"][0]["n"] == "4"
    assert meshed["bearings"][1]["n"] == "6"


def test_a_pinned_element_pins_its_pieces():
    data = project(with_link=False)
    for place, row in enumerate(data["shafts"]):
        row["n"] = str(place)
    meshed, _ = mesh_shafts(data, max_ld=1.0)

    assert [s["n"] for s in meshed["shafts"]] == ["0", "1", "2", "3", "4", "5"]


def test_an_element_too_short_is_reported_and_left_alone():
    data = project(with_link=False)
    data["shafts"][1]["L"] = "10"  # 10 mm under a 200 mm shaft: L/D = 0.05
    meshed, report = mesh_shafts(data, max_ld=1.0)

    assert report["short"] == [{"rows": [1], "ratio": pytest.approx(0.05)}]
    assert [s["tag"] for s in meshed["shafts"]].count("Body") == 1
    assert meshed["shafts"][2]["L"] == "10"


def test_only_the_rows_asked_for_are_cut():
    meshed, report = mesh_shafts(project(with_link=False), max_ld=1.0, indexes=[1])

    assert [s.get("tag") for s in meshed["shafts"]] == [
        "Inlet",
        "Body",
        "Body (2)",
        "Outlet",
    ]
    assert [step["rows"] for step in report["cut"]] == [[1]]


def test_an_element_of_exactly_the_ratio_stays_whole():
    """70 mm under 100 mm at L/D 0.7: floating point makes the quotient
    1.0000000000000002, which is not a reason to cut it in two."""
    data = project(with_link=False)
    data["shafts"][0].update(
        {"L": "70", "odl": "100", "odr": "100", "idl": "0", "idr": "0"}
    )
    steps, _ = plan(data, "0.7", "0.1")

    assert steps[0]["parts"] == 1


def test_nothing_to_cut_is_the_same_project():
    data = project()
    meshed, report = mesh_shafts(data, max_ld=10.0)

    assert meshed == data
    assert report["cut"] == [] and report["before"] == report["after"]


def test_the_original_project_is_not_touched():
    data = project()
    before = copy.deepcopy(data)
    mesh_shafts(data, max_ld=0.25)

    assert data == before


@pytest.mark.parametrize(
    "max_ld, min_ld",
    [
        ("abc", "0.1"),
        ("0", "0.1"),
        ("-1", "0.1"),
        ("0.5", "0.6"),
        ("0.5", "x"),
        ("nan", "0.1"),
    ],
)
def test_ratios_that_mean_nothing_are_refused_by_name(max_ld, min_ld):
    with pytest.raises(ValueError, match="L/D"):
        plan(project(), max_ld, min_ld)


@pytest.mark.parametrize("index", ["1", 1.0, True])
def test_rows_are_asked_for_by_position(index):
    with pytest.raises(ValueError, match="position"):
        plan(project(), 0.5, 0.1, [index])


# --- the route ---------------------------------------------------------------------


def test_the_route_answers_the_project_and_the_plan():
    from ross.interface.api.security import SESSION_TOKEN
    from ross.interface.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.post(
            "/api/rotor/mesh_shafts",
            json={"project": project(), "max_ld": "1"},
            headers={"X-ROSS-Token": SESSION_TOKEN},
        )
        refused = client.post(
            "/api/rotor/mesh_shafts",
            json={"project": project(), "max_ld": "much"},
            headers={"X-ROSS-Token": SESSION_TOKEN},
        )
    assert response.status_code == 200
    assert len(response.get_json()["projectData"]["shafts"]) == 6
    assert response.get_json()["report"]["after"] == 6
    assert refused.status_code == 400
    assert "much" in refused.get_json()["message"]


# --- in a number of parts --------------------------------------------------------------


@pytest.mark.parametrize("with_link", [False, True])
@pytest.mark.parametrize("parts", [1, 3])
def test_cut_in_parts_the_project_builds_the_rotor_ross_would(with_link, parts):
    data = project(with_link=with_link)
    meshed, report = mesh_shafts(data, parts=str(parts))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = build_rotor_from_ui(data).refine_mesh(parts=parts)

    assert described(build_rotor_from_ui(meshed)) == described(theirs)
    assert report["after"] == parts * report["before"]


def test_in_parts_only_the_rows_asked_for_are_cut():
    meshed, _ = mesh_shafts(project(with_link=False), parts="4", indexes=[2])

    assert [s.get("tag") for s in meshed["shafts"]] == [
        "Inlet",
        "Body",
        "Outlet",
        "Outlet (2)",
        "Outlet (3)",
        "Outlet (4)",
    ]


@pytest.mark.parametrize("parts", ["0", "2.5", "-3", "many", "inf"])
def test_a_number_of_parts_that_is_not_one_is_refused_by_name(parts):
    with pytest.raises(ValueError, match="number of parts"):
        plan(project(), parts=parts)


# --- by the convergence of the natural frequencies -----------------------------------


def coarse():
    """A metre of 40 mm shaft in two elements, a disk in the middle: far too
    coarse for its first frequencies."""
    data = {k: [] for k in ("disks", "gears", "couplings", "seals", "pointmasses")}
    data["materials"] = [
        {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}
    ]
    data["shafts"] = [
        {
            "element_type": "BASIC",
            "L": "500",
            "odl": "40",
            "idl": "0",
            "material": "Steel",
            "tag": "A",
        },
        {
            "element_type": "BASIC",
            "L": "500",
            "odl": "40",
            "idl": "0",
            "material": "Steel",
            "tag": "B",
        },
    ]
    data["disks"] = [
        {
            "element_type": "Geometry",
            "n": "1",
            "width": "50",
            "i_d": "40",
            "o_d": "300",
            "material": "Steel",
        }
    ]
    data["bearings"] = [
        {"element_type": "BASIC", "n": "0", "kxx": "1e7", "cxx": "0"},
        {"element_type": "BASIC", "n": "2", "kxx": "1e7", "cxx": "0"},
    ]
    return data


def test_the_converged_project_is_the_rotor_ross_converges_to():
    data = coarse()
    meshed, report = mesh_by_convergence(data, "6", "0.1", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs, results = build_rotor_from_ui(data).refine_mesh_by_convergence(
            n_modes=6, rtol=1e-3
        )

    assert described(build_rotor_from_ui(meshed)) == described(theirs)
    table = report["convergence"]
    assert table["chosen"] == results.chosen == 1
    assert [row["elements"] for row in table["rows"]] == [
        int(n) for n in results.el_num
    ]
    assert (
        table["rows"][table["chosen"]]["elements"]
        == report["after"]
        == len(meshed["shafts"])
    )
    # The rotor as it is first, marked by having no ratio; the frequencies in Hz.
    assert table["rows"][0]["max_ld"] is None and table["rows"][0]["elements"] == 2
    assert table["rows"][1]["wn"][0] == pytest.approx(
        results.wn[1][0] / (2 * 3.141592653589793)
    )
    assert table["rows"][table["chosen"]]["change"] <= 0.1
    assert table["rows"][-1]["change"] is None


def test_a_rotor_already_converged_is_left_as_it_is():
    data = coarse()
    meshed, report = mesh_by_convergence(data, "2", "50", "0")

    assert report["convergence"]["chosen"] == 0
    assert meshed == data and report["cut"] == []


@pytest.mark.parametrize(
    "n_modes, rtol, speed, words",
    [
        ("0", "0.1", "0", "frequencies"),
        ("x", "0.1", "0", "frequencies"),
        ("6", "0", "0", "tolerance"),
        ("6", "-1", "0", "tolerance"),
        ("6", "0.1", "fast", "rpm"),
    ],
)
def test_what_the_convergence_cannot_use_is_refused_by_name(
    n_modes, rtol, speed, words
):
    with pytest.raises(ValueError, match=words):
        mesh_by_convergence(coarse(), n_modes, rtol, speed)


def test_the_route_runs_each_way_and_refuses_one_it_does_not_know():
    from ross.interface.api.security import SESSION_TOKEN
    from ross.interface.app import app

    app.config["TESTING"] = True
    headers = {"X-ROSS-Token": SESSION_TOKEN}
    with app.test_client() as client:
        parts = client.post(
            "/api/rotor/mesh_shafts",
            json={"project": project(), "method": "parts", "parts": "2"},
            headers=headers,
        )
        converged = client.post(
            "/api/rotor/mesh_shafts",
            json={
                "project": coarse(),
                "method": "convergence",
                "n_modes": "4",
                "rtol": "0.1",
            },
            headers=headers,
        )
        unknown = client.post(
            "/api/rotor/mesh_shafts",
            json={"project": project(), "method": "magic"},
            headers=headers,
        )
    assert parts.status_code == 200 and parts.get_json()["report"]["after"] == 6
    assert (
        converged.status_code == 200
        and converged.get_json()["report"]["convergence"]["chosen"] >= 1
    )
    assert unknown.status_code == 400 and "magic" in unknown.get_json()["message"]
