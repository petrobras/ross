# -*- coding: utf-8 -*-
"""An analysis that reads at a probe of the model, by the probe's id.

`domain/probe_refs.py` fills a row of an analysis's probe table that names a
probe of the model from the model as it is at the run: its node, its angle,
its direction and its name. The claims:

* the row reads where the probe is now, not where it was when the row was
  made -- the reason the row names an id and not a node;
* on a MultiRotor a probe of the driven line is numbered as ROSS numbers that
  line inside the MultiRotor;
* a probe the model lost is refused, not read at node 0;
* the run puts the probe's name in the legend, and the exported script reads
  where the chart read.
"""

import ast
import json
import os

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.api.export import with_model_probes  # noqa: E402
from ross.interface.domain.probe_refs import (  # noqa: E402
    NO_SUCH_PROBE,
    model_probes,
    references,
    resolved,
)
from ross.interface.domain.python_export import build_script  # noqa: E402
from ross.interface.domain.rotor_builder import assemble_rotor  # noqa: E402
from ross.interface.domain.splitting import split_shaft  # noqa: E402
from ross.interface.services.analysis.base import probe_from_row  # noqa: E402
from ross.interface.services.analysis.pipeline import figure_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEEL = {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}


def project(probes):
    return {
        "materials": [dict(STEEL)],
        "shafts": [
            {"L": "250", "idl": "0", "odl": "50", "material": "Steel"} for _ in range(4)
        ],
        "disks": [{"n": "2", "m": "20", "Ip": "0.2", "Id": "0.1"}],
        "gears": [],
        "couplings": [],
        "seals": [],
        "bearings": [
            {"n": "0", "kxx": "1e7", "cxx": "1e3"},
            {"n": "4", "kxx": "1e7", "cxx": "1e3"},
        ],
        "pointmasses": [],
        "probes": probes,
    }


DE = {"id": "p-de", "n": "2", "tag": "DE X", "direction": "radial", "angle": "45"}
AX = {"id": "p-ax", "n": "4", "tag": "Thrust", "direction": "axial", "angle": ""}


def run(rows, built):
    return resolved({"probes": rows}, model_probes(assemble_rotor(built), built))


def test_a_row_that_names_a_probe_is_read_from_the_model():
    rows = run([{"probe": "p-de", "node": 0, "angle": 0}], project([DE, AX]))["probes"]
    assert rows == [
        {
            "probe": "p-de",
            "node": 2,
            "angle": pytest.approx(0.7853981633974483),
            "direction": "radial",
            "tag": "DE X",
        }
    ]


def test_an_axial_probe_has_no_angle():
    row = run([{"probe": "p-ax"}], project([DE, AX]))["probes"][0]
    assert (row["node"], row["angle"], row["direction"]) == (4, None, "axial")


def test_a_typed_row_is_left_as_it_was_typed():
    typed = {"node": 3, "angle": 0.5}
    assert run([typed], project([DE]))["probes"] == [typed]
    assert references({"probes": [typed]}) is False
    assert references({"probes": [typed, {"probe": "p-de"}]}) is True


def test_the_row_follows_its_probe_when_the_shaft_is_split():
    """The reason for an id: the probe on node 2 is on node 3 after a split
    below it, and the analysis reads at node 3 without being touched."""
    split = split_shaft(project([DE]), 0, "125")
    assert run([{"probe": "p-de"}], split)["probes"][0]["node"] == 3


def test_a_probe_the_model_lost_is_refused():
    with pytest.raises(ValueError) as refused:
        run([{"probe": "gone"}], project([DE]))
    assert str(refused.value) == NO_SUCH_PROBE


def test_a_driven_probe_is_numbered_as_ross_numbers_the_driven_line():
    with open(
        os.path.join(ROOT, "tests", "golden", "rotor_scenes.json"), encoding="utf-8"
    ) as handle:
        built = json.load(handle)["multirotor"]["project"]
    built["driving_rotor"]["probes"] = [dict(DE, id="drv", n="1")]
    built["driven_rotor"]["probes"] = [dict(DE, id="dvn", n="1")]
    assembled = assemble_rotor(built)
    probes = model_probes(assembled, built)
    multi = assembled.rotor
    driven = assembled.halves[1].rotor
    assert probes["drv"].node == 1
    assert probes["dvn"].node == multi.driven_nodes[list(driven.nodes).index(1)]
    assert probes["dvn"].node != 1


def test_the_probe_built_from_a_row_carries_its_direction_and_name():
    radial = probe_from_row({"node": 2, "angle": 0.5, "tag": "DE X"})
    axial = probe_from_row({"node": 4, "angle": None, "direction": "axial"})
    typed = probe_from_row({"node": 1, "angle": 0.25})
    assert (radial.node, radial.angle, radial.tag) == (2, 0.5, "DE X")
    assert (axial.direction, axial.angle) == ("axial", None)
    assert (typed.direction, typed.tag) == ("radial", None)


def test_the_chart_names_the_probe_as_the_model_does():
    params = {
        "speed_min": "0",
        "speed_max": "500",
        "speed_min_unit": "rad/s",
        "speed_max_unit": "rad/s",
        "speed_steps": "20",
        "unbalances": [{"node": 2, "mag": 0.001, "phase": 0}],
        "probes": [{"probe": "p-de", "node": 0, "angle": 0}],
    }
    figure = json.loads(figure_json("unbalance", params, "none", project([DE])))
    names = {trace.get("name") for trace in figure["data"]}
    assert "DE X" in names


def test_the_exported_script_reads_where_the_chart_read():
    built = project([DE, AX])
    analyses = [
        {
            "type": "unbalance",
            "params": {"probes": [{"probe": "p-de"}, {"probe": "p-ax"}]},
        }
    ]
    script = build_script(built, with_model_probes(built, analyses), "")
    ast.parse(script)
    assert "rs.Probe(2, 0.7853981633974483, tag='DE X')" in script
    assert "rs.Probe(4, None, direction='axial', tag='Thrust')" in script


def test_an_export_with_no_probe_named_builds_nothing():
    """The model is built only when an analysis names one of its probes: an
    export of a rotor ROSS refuses still writes the script it always did."""
    broken = project([])
    broken["bearings"] = [{"n": "99", "kxx": "1e6"}]
    analyses = [{"type": "unbalance", "params": {"probes": [{"node": 1, "angle": 0}]}}]
    assert with_model_probes(broken, analyses) == analyses
