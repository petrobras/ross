# -*- coding: utf-8 -*-
"""The frequency response read at a probe of the model.

ROSS's `FrequencyResponseResults` plots one degree of freedom as the output;
a radial probe at an angle reads x cos(angle) + y sin(angle) of its node --
the projection ROSS itself makes of a probe in the unbalance response
(`results.py`, `calculate_amplitude`: `ru cos(angle) + rv sin(angle)`). The
system is linear, so the response read by the probe is that projection of
the responses, for every input. The claims:

* `probe_output` is that projection, and at 0 and 90 degrees it is x and y;
* the chart of an output that names a probe draws it, named as the model
  names it, and an axial probe reads z;
* a model with no degree of freedom for the probe refuses it by name;
* the exported script carries the same function, giving the same numbers.
"""

import ast
import base64
import json

import numpy as np
import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.api.export import with_model_probes  # noqa: E402
from ross.interface.domain.python_export import PROBE_OUTPUT, build_script  # noqa: E402
from ross.interface.services.analysis.freq_response import (  # noqa: E402
    NO_AXIAL,
    NO_LATERAL,
    FreqResponseRunner,
    probe_output,
)
from ross.interface.services.analysis.pipeline import figure_json  # noqa: E402

STEEL = {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}
DE = {"id": "p-de", "n": "2", "tag": "DE 45", "direction": "radial", "angle": "45"}
AX = {"id": "p-ax", "n": "4", "tag": "Thrust", "direction": "axial", "angle": ""}


def project(probes=()):
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
            # Different in x and in y, so x and y respond differently and a
            # projection that read the wrong one would show.
            {"n": "0", "kxx": "1e7", "kyy": "4e7", "cxx": "1e3", "cyy": "1e3"},
            {"n": "4", "kxx": "1e7", "kyy": "4e7", "cxx": "1e3", "cyy": "1e3"},
        ],
        "pointmasses": [],
        "probes": list(probes),
    }


@pytest.fixture(scope="module")
def response():
    from ross.interface.domain.rotor_builder import build_rotor_from_ui

    rotor = build_rotor_from_ui(project())
    return rotor, rotor.run_freq_response(np.linspace(0, 1000, 12))


def test_the_probe_reads_the_projection_of_the_response(response):
    rotor, result = response
    d = rotor.number_dof
    angle = 0.3
    read = probe_output(result, 2, angle, d)
    x, y = 2 * d, 2 * d + 1
    for responses, probe in (
        (result.freq_resp, read.freq_resp),
        (result.velc_resp, read.velc_resp),
        (result.accl_resp, read.accl_resp),
    ):
        expected = (
            np.cos(angle) * responses[:, x, :] + np.sin(angle) * responses[:, y, :]
        )
        assert np.allclose(probe[:, 0, :], expected)
    assert read.freq_resp.shape == (
        result.freq_resp.shape[0],
        1,
        result.freq_resp.shape[2],
    )


def test_at_0_and_90_degrees_the_probe_reads_x_and_y(response):
    rotor, result = response
    d = rotor.number_dof
    assert np.allclose(
        probe_output(result, 2, 0.0, d).freq_resp[:, 0, :],
        result.freq_resp[:, 2 * d, :],
    )
    assert np.allclose(
        probe_output(result, 2, np.pi / 2, d).freq_resp[:, 0, :],
        result.freq_resp[:, 2 * d + 1, :],
    )
    # and x and y differ here, so the two lines above are not the same claim
    assert not np.allclose(
        result.freq_resp[:, 2 * d, :], result.freq_resp[:, 2 * d + 1, :]
    )


def test_the_script_carries_the_same_function(response):
    rotor, result = response
    namespace = {"np": np}
    exec(PROBE_OUTPUT, namespace)
    written = namespace["probe_output"](result, 2, 0.7, rotor.number_dof)
    assert np.array_equal(
        written.freq_resp, probe_output(result, 2, 0.7, rotor.number_dof).freq_resp
    )


def numbers(packed):
    """A list from plotly's JSON, which packs numpy arrays as typed bytes."""
    if isinstance(packed, dict):
        return np.frombuffer(base64.b64decode(packed["bdata"]), dtype=packed["dtype"])
    return np.asarray(packed, dtype=float)


PARAMS = {
    "speed_min": "0",
    "speed_max": "1000",
    "speed_steps": "12",
    "plot_type": "Magnitude",
    "inps": [{"node": 2, "dof": 0}],
}


def test_the_chart_reads_at_the_probe_and_names_it(response):
    rotor, result = response
    params = dict(PARAMS, outs=[{"probe": "p-de", "node": 0, "dof": 0}])
    figure = json.loads(figure_json("freq_response", params, "none", project([DE])))
    trace = figure["data"][0]
    assert trace["name"] == "In(N2 D0) | Out(DE 45)"
    d = rotor.number_dof
    expected = np.abs(probe_output(result, 2, np.pi / 4, d).freq_resp[2 * d, 0, :])
    assert np.allclose(numbers(trace["y"]), expected)


def test_an_axial_probe_reads_z(response):
    rotor, result = response
    d = rotor.number_dof
    source, out, name = FreqResponseRunner.output(
        result, {"probe": "p-ax", "node": 4, "direction": "axial", "tag": "Thrust"}, d
    )
    assert (source, out, name) == (result, 4 * d + 2, "Thrust")


def test_a_typed_output_is_read_as_it_always_was(response):
    rotor, result = response
    source, out, name = FreqResponseRunner.output(result, {"node": 3, "dof": 1}, 6)
    assert (source, out, name) == (result, 3 * 6 + 1, "N3 D1")


def test_a_model_without_the_degree_of_freedom_refuses_the_probe(response):
    _, result = response
    with pytest.raises(ValueError) as axial:
        FreqResponseRunner.output(
            result, {"probe": "a", "node": 1, "direction": "axial", "tag": "Thrust"}, 4
        )
    assert str(axial.value) == NO_AXIAL % "Thrust"
    with pytest.raises(ValueError) as radial:
        FreqResponseRunner.output(
            result,
            {"probe": "r", "node": 1, "direction": "radial", "angle": 0.1, "tag": "DE"},
            1,
        )
    assert str(radial.value) == NO_LATERAL % "DE"


def test_the_exported_script_reads_where_the_chart_read():
    built = project([DE, AX])
    analyses = [
        {
            "type": "freq_response",
            "params": dict(
                PARAMS,
                outs=[{"probe": "p-de"}, {"probe": "p-ax"}, {"node": 3, "dof": 1}],
                inps=[{"node": 2, "dof": 0}],
            ),
        }
    ]
    script = build_script(built, with_model_probes(built, analyses), "")
    ast.parse(script)
    assert script.count("def probe_output(") == 1
    assert (
        "fig_temp = probe_output(freq_0, 2, 0.7853981633974483, dofs_per_node).plot_magnitude("
        in script
    )
    assert "g_out = 4 * dofs_per_node + 2" in script
    assert "trace.name = 'In(N2 D0) | Out(DE 45)'" in script
    assert 'trace.name = f"In(N2 D0) | Out(N3 D1)"' in script
