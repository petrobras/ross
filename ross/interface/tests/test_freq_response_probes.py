# -*- coding: utf-8 -*-
"""The frequency response read at a probe of the model.

ROSS's `FrequencyResponseResults` plots one degree of freedom as the output;
a radial probe at an angle reads x cos(angle) + y sin(angle) of its node --
the projection ROSS itself makes of a probe in the unbalance response
(`results.py`, `calculate_amplitude`: `ru cos(angle) + rv sin(angle)`). The
system is linear, so the response read by the probe is that projection of
the responses, for every input. The claims:

* `probe_output`, in the service and in the exported script, is that
  projection of the response at the probe's node, checked against a direct
  solve of the dynamic stiffness on a rotor whose transfer matrix is not
  symmetric; at 0 and 90 degrees it is x and y;
* in one chart, a typed output row and a probe along the same axis draw the
  same curve;
* the chart of an output that names a probe draws it, named as the model
  names it, and an axial probe reads z;
* a model with no degree of freedom for the probe refuses it by name;
* at a 45 degree probe the chart is what `rs.Probe` reads of the forced
  response to a unit force at the input;
* the exported script carries the same function, giving the same numbers.
"""

import ast
import base64
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

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


ANGLES = (0.0, 45.0, 90.0, 135.0)
SPEED = 300.0


def coupled_project(probes=()):
    """The project above, with cross-coupled bearings.

    kxy = -kyx makes the transfer matrix non-symmetric, as the gyroscopic
    disk does at speed: the response at a probe to a force at another degree
    of freedom is then not the response there to a force at the probe.
    """
    built = project(probes)
    for bearing in built["bearings"]:
        bearing.update(kxy="5e6", kyx="-5e6")
    return built


@pytest.fixture(scope="module")
def coupled():
    from ross.interface.domain.rotor_builder import build_rotor_from_ui

    rotor = build_rotor_from_ui(coupled_project())
    frequencies = np.linspace(50, 1000, 12)
    return rotor, rotor.run_freq_response(frequencies, speed=SPEED)


def direct_solve(rotor, frequencies, inp):
    """The response of every degree of freedom to a unit force at `inp`.

    Each frequency w solves Z(w) q = e_inp, with
    Z = K - w^2 M + i w (C + speed G), by `numpy.linalg.solve`: a path
    independent of the transfer matrix `run_freq_response` stores.
    """
    force = np.zeros(rotor.ndof)
    force[inp] = 1.0
    columns = []
    for w in frequencies:
        dynamic_stiffness = (
            rotor.K(w, SPEED)
            - w**2 * rotor.M(w, SPEED)
            + 1j * w * (rotor.C(w, SPEED) + SPEED * rotor.G())
        )
        columns.append(np.linalg.solve(dynamic_stiffness, force))
    return np.array(columns).T


def radial_probes(node, angles):
    return [
        {
            "id": "p-%d" % degrees,
            "n": str(node),
            "tag": "P %d" % degrees,
            "direction": "radial",
            "angle": str(degrees),
        }
        for degrees in angles
    ]


def coupled_params(inp, outs):
    return dict(
        PARAMS,
        speed_min="50",
        speed_max="1000",
        speed=str(SPEED),
        inps=[inp],
        outs=outs,
    )


def test_the_transfer_matrix_here_is_not_symmetric(coupled):
    _, result = coupled
    h = result.freq_resp
    assert not np.allclose(h, h.transpose(1, 0, 2))


def test_the_probe_reads_the_response_at_its_node_in_its_direction(coupled):
    rotor, result = coupled
    d = rotor.number_dof
    w = result.speed_range
    namespace = {"np": np}
    exec(PROBE_OUTPUT, namespace)
    for inp in (2 * d, 2 * d + 1, 1 * d, 3 * d + 1):
        q = direct_solve(rotor, w, inp)
        for degrees in ANGLES:
            angle = np.deg2rad(degrees)
            expected = np.cos(angle) * q[2 * d] + np.sin(angle) * q[2 * d + 1]
            for function in (probe_output, namespace["probe_output"]):
                read = function(result, 2, angle, d)
                assert_allclose(read.freq_resp[0, inp], expected, rtol=1e-9)
                assert_allclose(read.velc_resp[0, inp], 1j * w * expected, rtol=1e-9)
                assert_allclose(read.accl_resp[0, inp], -(w**2) * expected, rtol=1e-9)


def test_the_chart_draws_the_response_at_the_probe(coupled):
    rotor, result = coupled
    d = rotor.number_dof
    q = direct_solve(rotor, result.speed_range, 1 * d + 1)
    probes = radial_probes(2, ANGLES)
    params = coupled_params(
        {"node": 1, "dof": 1}, [{"probe": probe["id"]} for probe in probes]
    )
    built = coupled_project(probes)
    magnitude = json.loads(figure_json("freq_response", params, "none", built))
    phase = json.loads(
        figure_json("freq_response", dict(params, plot_type="Phase"), "none", built)
    )
    for k, degrees in enumerate(ANGLES):
        angle = np.deg2rad(degrees)
        expected = np.cos(angle) * q[2 * d] + np.sin(angle) * q[2 * d + 1]
        assert magnitude["data"][k]["name"] == "In(N1 D1) | Out(P %d)" % degrees
        assert_allclose(numbers(magnitude["data"][k]["y"]), np.abs(expected), rtol=1e-9)
        drawn = numbers(phase["data"][k]["y"])
        assert_allclose(np.exp(1j * drawn), np.exp(1j * np.angle(expected)), atol=1e-9)


@pytest.mark.parametrize("inp", [{"node": 1, "dof": 0}, {"node": 3, "dof": 1}])
def test_a_typed_row_and_the_probe_on_its_axis_draw_the_same_curve(coupled, inp):
    """Out(N2 D0) is the 0 degree probe at node 2, Out(N2 D1) the 90 degree one.

    The typed rows are read by `FrequencyResponseResults` from the full
    response, the probe rows from `probe_output`; both are held to the direct
    solve of Z(w) as well, so a chart cannot mix two readings of the matrix.
    """
    rotor, result = coupled
    d = rotor.number_dof
    q = direct_solve(rotor, result.speed_range, inp["node"] * d + inp["dof"])
    probes = radial_probes(2, (0, 90))
    outs = [
        {"node": 2, "dof": 0},
        {"probe": "p-0"},
        {"node": 2, "dof": 1},
        {"probe": "p-90"},
    ]
    built = coupled_project(probes)
    params = coupled_params(inp, outs)
    magnitude = json.loads(figure_json("freq_response", params, "none", built))
    phase = json.loads(
        figure_json("freq_response", dict(params, plot_type="Phase"), "none", built)
    )
    for typed, probe, dof in ((0, 1, 0), (2, 3, 1)):
        expected = q[2 * d + dof]
        for k in (typed, probe):
            assert_allclose(
                numbers(magnitude["data"][k]["y"]), np.abs(expected), rtol=1e-9
            )
            drawn = numbers(phase["data"][k]["y"])
            assert_allclose(
                np.exp(1j * drawn), np.exp(1j * np.angle(expected)), atol=1e-9
            )


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
            np.cos(angle) * responses[x, :, :] + np.sin(angle) * responses[y, :, :]
        )
        assert np.allclose(probe[0, :, :], expected)
    assert read.freq_resp.shape == (
        1,
        result.freq_resp.shape[1],
        result.freq_resp.shape[2],
    )


def test_at_0_and_90_degrees_the_probe_reads_x_and_y(response):
    rotor, result = response
    d = rotor.number_dof
    assert np.allclose(
        probe_output(result, 2, 0.0, d).freq_resp[0, :, :],
        result.freq_resp[2 * d, :, :],
    )
    assert np.allclose(
        probe_output(result, 2, np.pi / 2, d).freq_resp[0, :, :],
        result.freq_resp[2 * d + 1, :, :],
    )
    # and x and y differ here, so the two lines above are not the same claim
    assert not np.allclose(
        result.freq_resp[2 * d, :, :], result.freq_resp[2 * d + 1, :, :]
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
    expected = np.abs(probe_output(result, 2, np.pi / 4, d).freq_resp[0, 2 * d, :])
    assert np.allclose(numbers(trace["y"]), expected)


def cross_coupled(probes=()):
    built = project(probes)
    for bearing in built["bearings"]:
        bearing.update(kxy="5e6", kyx="-8e6")
    return built


@pytest.mark.parametrize("dof", [0, 1])
def test_the_chart_matches_the_forced_response_read_by_the_probe(dof):
    """The chart at a 45 degree probe is what `rs.Probe` reads of the response.

    The reference is the forced response to a unit force at the input, read
    at the same probe by `ForcedResponseResults`, a path that does not go
    through `probe_output`. The bearings are cross-coupled with kxy != -kyx
    and differ in x and y, so the response matrix is not symmetric: reading
    its force side would give other numbers, close to a probe mirrored to
    -45 degrees. The forced response reports the phase as a lag and the
    frequency response as the angle of the complex response, so the two
    phases have opposite signs.
    """
    from ross.interface.domain.rotor_builder import build_rotor_from_ui

    rotor = build_rotor_from_ui(cross_coupled())
    d = rotor.number_dof
    speeds = np.linspace(0, 1000, 12)
    force = np.zeros((rotor.ndof, len(speeds)), dtype=complex)
    force[2 * d + dof, :] = 1.0
    forced = rotor.run_forced_response(force=force, speed_range=speeds)
    probe = [ross.Probe(2, ross.Q_(45, "deg"))]
    magnitude = forced.data_magnitude(probe=probe, amplitude_units="m").iloc[1:, 1]
    phase = forced.data_phase(probe=probe).iloc[1:, 1]
    mirrored = forced.data_magnitude(
        probe=[ross.Probe(2, ross.Q_(-45, "deg"))], amplitude_units="m"
    ).iloc[1:, 1]
    assert not np.allclose(magnitude, mirrored, rtol=1e-2, atol=0)

    params = dict(PARAMS, inps=[{"node": 2, "dof": dof}], outs=[{"probe": "p-de"}])
    built = cross_coupled([DE])
    figure = json.loads(figure_json("freq_response", params, "none", built))
    charted = numbers(figure["data"][0]["y"])[1:]
    assert np.allclose(charted, magnitude, rtol=1e-8, atol=0)

    params["plot_type"] = "Phase"
    figure = json.loads(figure_json("freq_response", params, "none", built))
    charted = numbers(figure["data"][0]["y"])[1:]
    assert np.allclose(np.exp(1j * charted), np.exp(-1j * np.asarray(phase)))


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
