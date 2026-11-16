# -*- coding: utf-8 -*-
"""BRBalance, without Streamlit.

The calculation stays in `brbalance_calc.py` of the Digital Twin app. This
module turns the form into the arrays that function expects, and turns the
arrays back into JSON, a Plotly figure and a Word report.
"""

import io
import json
import os
import re
import zipfile

import numpy as np

from ross.interface.extensions.digital_twin.domain.dt_app import ensure_on_path

BRB_VERSION = "0.1.0"
MAX_ZIP_ENTRIES = 32
MAX_JSON_BYTES = 1_000_000
MAX_SESSION_KEYS = 512

_LIMITS = {
    "n_sensors": (1, 20, 2),
    "n_tests": (1, 10, 2),
    "n_planes": (1, 5, 2),
    "n_weights": (1, 8, 2),
}


def load_calc():
    """Import the existing calculator. It does not import Streamlit."""
    ensure_on_path()
    import brbalance_calc

    return brbalance_calc


def run(state):
    """Calculate the correction and the polar figure for one form."""
    prepared = prepare(state)
    result = prepared["result"]
    figure, plot_error = _figure(prepared)
    return {
        "correction": _rows(
            "Plane", result["correction_magnitude"], result["correction_phase"]
        ),
        "applied": _rows(
            "Plane",
            result["applied_correction_magnitude"],
            result["applied_correction_phase"],
        ),
        "predicted": _rows(
            "Sensor",
            result["predicted_vibration_magnitude"],
            result["predicted_vibration_phase"],
        ),
        "figure": figure,
        "plot_error": plot_error,
    }


def report_bytes(state):
    """Build the Word report and return its bytes."""
    prepared = prepare(state)
    result = prepared["result"]
    calc = prepared["calc"]
    # generate_balance_report only saves to a path. A temporary file keeps
    # that function unchanged.
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".docx")
    os.close(fd)
    try:
        calc.generate_balance_report(
            filename=path,
            n_tests=prepared["n_tests"],
            n_planes=prepared["n_planes"],
            vib_ref=prepared["vib_ref"],
            runout_ref=prepared["runout_ref"],
            test_vibration=prepared["test_vibration"],
            test_masses=prepared["test_masses"],
            applied_correction_magnitude=result["applied_correction_magnitude"],
            applied_correction_phase=result["applied_correction_phase"],
            predicted_vibration_magnitude=result["predicted_vibration_magnitude"],
            predicted_vibration_phase=result["predicted_vibration_phase"],
            influence_matrix=result["influence_matrix"],
        )
        with open(path, "rb") as stored:
            payload = stored.read()
    finally:
        if os.path.exists(path):
            os.remove(path)
    if not payload.startswith(b"PK"):
        raise ValueError("The Word report could not be generated.")
    return payload, _download_name(state, ".docx", "BRBalance_Report.docx")


def save_brb(state):
    """Pack the form into the same .brb zip the Streamlit page writes."""
    if not isinstance(state, dict):
        raise ValueError("The balancing state must be an object.")
    encoded = json.dumps(state, ensure_ascii=False).encode("utf-8")
    if len(encoded) >= MAX_JSON_BYTES:
        raise ValueError("Invalid .brb file: JSON too large.")
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr("brb.version", BRB_VERSION)
        archive.writestr("session_state.json", encoded)
    return blob.getvalue(), _download_name(state, ".brb", "balance.brb")


def load_brb(payload):
    """Read a .brb zip and return the session object the form understands."""
    if not payload:
        raise ValueError("Send a .brb file.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile:
        raise ValueError("Invalid .brb file.")
    names = archive.namelist()
    if len(names) >= MAX_ZIP_ENTRIES:
        raise ValueError("Invalid .brb file: too many files in the zip.")
    try:
        version = archive.read("brb.version").decode("utf-8")
    except KeyError:
        version = "0.0.0"
    raw = None
    for name in names:
        if name.endswith(".json"):
            raw = archive.read(name)
            break
    if raw is None or len(raw) >= MAX_JSON_BYTES:
        raise ValueError("Invalid .brb file: JSON too large.")
    loaded = json.loads(raw)
    if not isinstance(loaded, dict) or len(loaded) >= MAX_SESSION_KEYS:
        raise ValueError("Invalid .brb file: too many session keys.")
    loaded = load_calc().convert(loaded, version)
    return _plain(loaded)


def prepare(state):
    """Turn the form object into the arrays `calculate_balance_correction` takes."""
    if not isinstance(state, dict):
        raise ValueError("The balancing state must be an object.")
    calc = load_calc()
    n_sensors = _count(state, "n_sensors")
    n_tests = _count(state, "n_tests")
    n_planes = _count(state, "n_planes")
    n_weights = _count(state, "n_weights")

    vib_ref, runout_ref = calc.build_reference_arrays(
        _floats(state, "ref_vib_amp", n_sensors, "Reference vibration"),
        _floats(state, "ref_vib_phase", n_sensors, "Reference phase"),
        _floats(state, "ref_run_amp", n_sensors, "Reference runout"),
        _floats(state, "ref_run_phase", n_sensors, "Reference runout phase"),
    )

    test_vibration = np.zeros((n_tests, n_sensors, 2))
    test_masses = np.zeros((n_planes, n_tests, n_weights, 2))
    for test in range(n_tests):
        test_vibration[test, :, 0] = _floats(
            state,
            "test%d_vib_amp" % (test + 1),
            n_sensors,
            "Test %d vibration" % (test + 1),
        )
        test_vibration[test, :, 1] = _floats(
            state,
            "test_%d_vib_phase" % (test + 1),
            n_sensors,
            "Test %d phase" % (test + 1),
        )
        for plane in range(n_planes):
            test_masses[plane, test, :, 0] = _floats(
                state,
                "test%d_plane%d_mass" % (test, plane),
                n_weights,
                "Test %d mass, plane %d" % (test + 1, plane + 1),
            )
            test_masses[plane, test, :, 1] = _floats(
                state,
                "test%d_plane%d_phase" % (test, plane),
                n_weights,
                "Test %d mass phase, plane %d" % (test + 1, plane + 1),
            )

    current_vibration = np.column_stack(
        [
            _floats(state, "curr_vib_amp", n_sensors, "Current vibration"),
            _floats(state, "curr_vib_phase", n_sensors, "Current phase"),
        ]
    )
    current_runout = np.column_stack(
        [
            _floats(state, "curr_run_amp", n_sensors, "Runout atual"),
            _floats(state, "curr_run_phase", n_sensors, "Current runout phase"),
        ]
    )
    correction_masses = np.zeros((n_planes, n_weights, 2))
    for plane in range(n_planes):
        correction_masses[plane, :, 0] = _floats(
            state,
            "corr_plane%d_mass" % plane,
            n_weights,
            "Plane %d correction" % (plane + 1),
        )
        correction_masses[plane, :, 1] = _floats(
            state,
            "corr_plane%d_phase" % plane,
            n_weights,
            "Plane %d correction phase" % (plane + 1),
        )

    result = calc.calculate_balance_correction(
        vib_ref=vib_ref,
        runout_ref=runout_ref,
        test_vibration=test_vibration,
        test_masses=test_masses,
        current_vibration=current_vibration,
        current_runout=current_runout,
        correction_masses=correction_masses,
        verbose=False,
    )
    return {
        "calc": calc,
        "result": result,
        "n_tests": n_tests,
        "n_planes": n_planes,
        "n_sensors": n_sensors,
        "vib_ref": vib_ref,
        "runout_ref": runout_ref,
        "test_vibration": test_vibration,
        "test_masses": test_masses,
        "correction_masses": correction_masses,
        "directions": _directions(state, n_planes),
    }


def _figure(prepared):
    """The polar plot, or an error string when the figure cannot be built."""
    result = prepared["result"]
    try:
        figure = prepared["calc"].plot_polar(
            prepared["correction_masses"],
            np.asarray(result["correction_magnitude"]).reshape(-1, 1),
            np.asarray(result["correction_phase"]).reshape(-1, 1),
            title="Correction mass placement per plane",
            rotation_direction=prepared["directions"],
        )
        return json.loads(figure.to_json()), None
    except Exception as error:
        return None, "%s: %s" % (type(error).__name__, error)


def _directions(state, n_planes):
    flags = state.get("invert") or []
    if not isinstance(flags, list):
        flags = []
    directions = []
    for plane in range(n_planes):
        flag = flags[plane] if plane < len(flags) else False
        directions.append("counterclockwise" if flag else "clockwise")
    return directions


def _count(state, key):
    low, high, default = _LIMITS[key]
    raw = state.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer." % key)
    if value < low or value > high:
        raise ValueError("%s must be between %d and %d." % (key, low, high))
    return value


def _floats(state, key, size, label):
    values = state.get(key) or []
    if not isinstance(values, list):
        raise ValueError("%s must be a list." % label)
    numbers = []
    for index in range(size):
        raw = values[index] if index < len(values) else 0
        if raw is None or raw == "":
            raw = 0
        try:
            numbers.append(float(raw))
        except (TypeError, ValueError):
            raise ValueError("%s, row %d: enter a number." % (label, index + 1))
    return numbers


def _rows(kind, magnitudes, phases):
    rows = []
    for index, (magnitude, phase) in enumerate(zip(magnitudes, phases, strict=True)):
        rows.append(
            {
                "name": "%s %d" % (kind, index + 1),
                "magnitude": float(magnitude),
                "phase": float(phase),
            }
        )
    return rows


def _plain(value):
    """Keep only JSON values the form can put back into an input."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _download_name(state, extension, fallback):
    raw = str(state.get("session_name") or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_. -]", "", raw).strip()
    if not cleaned:
        return fallback
    return cleaned + extension
