import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from frontend_source import source

try:
    import ross as rs

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")

# BE-12: dressing the figure left the backend and went to the screen.
APPEARANCE = ("margin", "paper_bgcolor", "plot_bgcolor", "autosize", "legend")

PROJECT_REQUEST = {
    "name": "Compressor A",
    "uid": "rotor_123",
    "materials": [{"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}],
    "shafts": [
        {"L": "100", "odl": "50", "idl": "0", "material": "Steel"} for _ in range(3)
    ],
    "bearings": [
        {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "1e3"},
        {"element_type": "BASIC", "n": "3", "kxx": "1e6", "cxx": "1e3"},
    ],
    "disks": [],
    "gears": [],
    "seals": [],
    "couplings": [],
    "pointmasses": [],
}


def _auth():
    from api.security import SESSION_TOKEN

    return {"X-ROSS-Token": SESSION_TOKEN}


# Since slice 4 app.py is only the entry point. Each name below comes from its
# own layer -- and this explicit import is the documentation of that.
from app import app
from api.security import SESSION_TOKEN

# Every API route requires the session token (see require_session_token in app.py).
# Since Phase 2 the project goes inside an envelope ({'project': ...}), so that
# the interface can send only what describes the rotor -- with no saved charts.
AUTH = {"X-ROSS-Token": SESSION_TOKEN}


@pytest.fixture
def client():
    """Set up the Flask test environment."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# Basic Construction and Materials Test


def test_build_rotor_minimum_valid(client):
    """Tests whether the rotor is successfully created using the default material and shaft (Steel)."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [
            {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}
        ],
    }
    response = client.post("/build_rotor", json={"project": payload}, headers=AUTH)

    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert "plot_json" in response.json
    assert response.json["mass"] > 0


# Test of Mass and Inertia Elements


def test_elements_disk_gear_defaults(client):
    """Tests the injection of Disk and Gear defaults."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [
            {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}
        ],
        "disks": [{"m": "32", "Id": "0.2", "Ip": "0.3", "n": "0"}],
        "gears": [
            {
                "m": "4.67",
                "Id": "0.015",
                "Ip": "0.030",
                "n_teeth": "26",
                "pitch_diameter": "187",
                "pr_angle": "22.5",
                "helix_angle": "0",
                "n": "1",
            }
        ],
    }
    response = client.post("/build_rotor", json={"project": payload}, headers=AUTH)

    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert response.json["mass"] > 38.67


# Bearing Element Tests


def test_bearing_elements_defaults(client):
    """Tests whether the backend can instantiate complex bearings using the UI's array and string defaults."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [
            {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}
        ],
        "bearings": [
            # Bearing BASIC
            {
                "element_type": "BASIC",
                "kxx": "1e6",
                "kyy": "0.8e6",
                "cxx": "2e2",
                "cyy": "1.5e2",
                "n": "0",
            },
            # Cylindrical Bearing
            {
                "element_type": "Cylindrical",
                "speed": "[1500]",
                "weight": "525",
                "bearing_length": "30",
                "journal_diameter": "10",
                "radial_clearance": "0.1",
                "oil_viscosity": "0.1",
                "n": "1",
            },
        ],
    }
    response = client.post("/build_rotor", json={"project": payload}, headers=AUTH)

    assert response.status_code == 200
    assert response.json["status"] == "success"


# Seal Element Tests


def test_seal_elements_defaults(client):
    """Tests labels that use nested JSON dictionaries in the default (e.g., gas_composition)."""
    payload = {
        "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [
            {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"}
        ],
        "seals": [
            # Labyrinth Seal
            {
                "element_type": "Labyrinth",
                "shaft_diameter": "145",
                "radial_clearance": "0.3",
                "n_teeth": "16",
                "pitch": "3.175",
                "tooth_height": "3.175",
                "tooth_width": "0.1524",
                "seal_type": "inter",
                "inlet_pressure": "308000",
                "outlet_pressure": "94300",
                "inlet_temperature": "10",
                "frequency": "[8000]",
                "preswirl": "0.98",
                "gas_composition": '{"Nitrogen": 0.79, "Oxygen": 0.21}',
                "n": "0",
            }
        ],
    }
    response = client.post("/build_rotor", json={"project": payload}, headers=AUTH)

    assert response.status_code == 200
    assert response.json["status"] == "success"


# Error Handling Tests


def test_shaftless_rotor_failure(client):
    """Check if the server locks properly (Error 400) if the user does not submit axes."""
    payload = {"disks": [{"m": "32", "Id": "0.2", "Ip": "0.3", "n": "0"}]}
    response = client.post("/build_rotor", json={"project": payload}, headers=AUTH)

    assert response.status_code == 400
    assert response.json["status"] == "error"
    assert "Add at least one Shaft" in response.json["message"]


@needs_ross
def test_a_shared_element_is_not_altered_by_rotor_assembly():
    """The same object goes into two rotors and comes out of both as it went in."""
    steel = rs.materials.steel
    bearing = rs.BearingElement(n=0, kxx=1e6, cxx=1e3, tag="bearing_0")
    disk = rs.DiskElement(n=1, m=10.0, Id=0.1, Ip=0.2, tag="disk_0")

    def shafts():
        return [
            rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel, n=i)
            for i in range(2)
        ]

    before = (bearing.n, bearing.tag, disk.n, disk.tag)
    rs.Rotor(
        shaft_elements=shafts(),
        disk_elements=[disk],
        bearing_elements=[bearing, rs.BearingElement(n=2, kxx=1e6, cxx=1e3, tag="b1")],
    )
    rs.Rotor(
        shaft_elements=shafts(),
        disk_elements=[disk],
        bearing_elements=[bearing, rs.BearingElement(n=2, kxx=1e6, cxx=1e3, tag="b2")],
    )
    after = (bearing.n, bearing.tag, disk.n, disk.tag)

    assert before == after


@needs_ross
def test_build_rotor_reads_the_project_from_the_envelope(client):
    response = client.post(
        "/build_rotor", json={"project": PROJECT_REQUEST}, headers=_auth()
    )
    assert response.status_code == 200
    assert response.json["status"] == "success"


@pytest.mark.parametrize("key", APPEARANCE)
def test_the_backend_no_longer_dresses_the_figure(key):
    """Building the Plotly layout on the server is work that belongs to the screen."""
    with io.open(
        os.path.join(ROOT, "api", "rotor.py"), encoding="utf-8", newline=""
    ) as handle:
        js = "\n".join(
            line
            for line in handle.read().split("\n")
            if not line.strip().startswith("#")
        )
    assert "layout['%s']" % key not in js
    assert 'layout["%s"]' % key not in js


def test_the_screen_dresses_the_figure_instead():
    js = source()
    assert "ROTOR_APPEARANCE" in js
    for key in ("margin", "paper_bgcolor", "autosize", "legend"):
        assert (
            key in js[js.index("const ROTOR_APPEARANCE") : js.index("const ROTOR_MENU")]
        )


def test_the_geometry_fix_stays_with_the_rotor():
    """What needs the rotor stays on the server -- and the reason is written down.

    `fix_shapes` swaps the `xref` of the lines ROSS anchors on the paper, and for
    that it needs `rotor.L`. It is not presentation; it is geometry."""
    with io.open(
        os.path.join(ROOT, "api", "rotor.py"), encoding="utf-8", newline=""
    ) as handle:
        js = handle.read()
    assert "def fix_shapes(" in js
    assert "float(rotor.L)" in js
    assert "geometry" in js.lower()


@pytest.mark.parametrize(
    "name",
    [
        "getNum",
        "getTxt",
        "updateAnalysisParameters",
        "triggerCardUpdate",
        "unitMapFor",
        "rossClassFor",
    ],
)
def test_the_dead_code_is_gone(name):
    assert name not in source(), "%s voltou" % name
