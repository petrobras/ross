# -*- coding: utf-8 -*-
"""One runner per analysis.

`run_analysis` was a 460-line `if/elif` for twelve analyses, with the same
block repeated twelve times. Now each analysis is a class with three methods --
`spec`, `compute` and `plot` -- and the route is 54 lines.

The split is not cosmetic. **`compute` receives only the `spec`, never the
`params`**, and the cache key is the fingerprint of the `spec`. With that a
computation has no way of depending on something left out of the key -- which
was exactly the defect this file documents in
`test_mode_shape_and_default_no_longer_share_a_cache_entry`."""

import io
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.analysis_catalog import ANALYSES
from services.analysis import REGISTRY, get_runner
from services.analysis.ucs import HALF_A_RANGE
from frontend_source import source as _frontend

# The analyses the screen offers. Until the i18n slice these were twelve
# `<option>` in index.html; now the `<select>` is filled with the catalogue
# titles, so "what the screen offers" is what the catalogue has. If someone
# adds an analysis to the catalogue without writing the runner, the test
# accuses it -- before, the analysis simply did not answer.


def _analyses_on_the_screen():
    from domain.analysis_catalog import ANALYSES

    return set(ANALYSES)


# --- A fake rotor: the `spec` only consults these three attributes -----------
# This lets the whole battery run without ROSS installed. If a `spec` starts
# reading something else off the rotor, the AttributeError shows up here and
# not in production.


class FakeRotor:
    nodes = list(range(8))
    ndof = 48
    number_dof = 6


ROTOR_REQUEST = FakeRotor()

# Plausible parameters for each analysis. Without _unit suffixes: the
# conversion depends on ROSS's pint and this battery also runs without ROSS
# installed (the conversion has coverage of its own in test_expressions). The
# speed values already go in rad/s.
PARAMS = {
    "campbell": {
        "speed_min": "0",
        "speed_max": "418.9",
        "speed_steps": "50",
        "frequencies": "6",
        "frequency_type": "wd",
        "torsional_analysis": "False",
    },
    # k_min/k_max are EXPONENTS of 10, not stiffnesses: the screen label itself
    # says "Min Stiffness (10^x N/m)". With 1e6 here, ROSS builds a logspace of
    # 10**1e6 and the computation becomes infinity.
    #
    # The bearing frequency range is filled in here on purpose. It was out of
    # the interface for as long as ROSS raised on any value for it; commit
    # 2a253e6 fixed that, and a field that came back has to be exercised by the
    # sweep that runs every analysis for real -- otherwise it is offered on the
    # screen and never once run.
    "ucs": {
        "k_min": "6",
        "k_max": "11",
        "num_modes": "16",
        "synchronous": "True",
        "bearing_freq_min": "0",
        "bearing_freq_max": "1000",
    },
    "freq_response": {
        "speed_min": "0",
        "speed_max": "1000",
        "free_free": "True",
        "modes": "[0, 1]",
        "inps": [{"node": 1, "dof": 0}],
        "outs": [{"node": 2, "dof": 1}],
    },
    "modes": {
        "speed": "104.7",
        "num_modes": "12",
        "sparse": "False",
        "synchronous": "True",
    },
    "unbalance": {
        "speed_min": "0",
        "speed_max": "1000",
        "modes": "[0]",
        "unbalances": [{"node": 1, "mag": 0.005, "phase": 0.0}],
    },
    "time_response": {
        "speed": "125.6",
        "t_max": "2",
        "steps": "2000",
        "method": "default",
        "forces": [{"node": 1, "dof": 0, "func": "10*sin(100*t)"}],
    },
    # Every field of the flexible coupling: ROSS has no default for any of them and
    # the computation blows up with None if one is missing.
    "misalignment": {
        "coupling": "flex",
        "n": "0",
        "input_torque": "100",
        "load_torque": "50",
        "mis_type": "parallel",
        "mis_distance_x": "2e-4",
        "mis_distance_y": "2e-4",
        "mis_angle": "0.01",
        "radial_stiffness": "40e3",
        "bending_stiffness": "38e3",
        "t_initial": "0",
        "t_final": "0.5",
        "t_steps": "5000",
        "speed": "125.6",
        "unbalances": [{"node": 1, "mag": 5e-4, "phase": 0}],
    },
    "rubbing": {
        "n": "1",
        "distance": "1e-4",
        "contact_stiffness": "1e6",
        "contact_damping": "1e3",
        "friction_coeff": "0.3",
        "torque": "True",
        "t_initial": "0",
        "t_final": "0.5",
        "t_steps": "5000",
        "speed": "1200",
        "unbalances": [{"node": 1, "mag": 5e-4, "phase": 0}],
    },
    "crack": {
        "n": "1",
        "depth_ratio": "0.2",
        "crack_model": "Mayes",
        "cross_divisions": "30",
        "t_initial": "0",
        "t_final": "0.5",
        "t_steps": "5000",
        "speed": "1200",
        "unbalances": [{"node": 1, "mag": 5e-4, "phase": 0}],
    },
    "static": {},
    "harmonic_balance": {
        "t_initial": "0",
        "t_final": "0.5",
        "t_steps": "1001",
        "hb_node": "1",
        "hb_magnitudes": "[2000]",
        "hb_phases": "[0]",
        "hb_harmonics": "[1]",
        "speed": "125.6",
        "gravity": "True",
        "n_harmonics": "3",
    },
    "clearance": {
        "speed": "1200",
        "unbalances": [{"node": 1, "mag": 0.05, "phase": 0.0}],
        "frequency": "[100]",
        "modes": "[0, 1]",
    },
}

NAMES = sorted(REGISTRY)


# --- the registry covers the screen ------------------------------------------


def test_the_registry_covers_every_analysis_the_screen_offers():
    from_the_screen = _analyses_on_the_screen()
    missing = from_the_screen - set(REGISTRY)
    assert not missing, "no runner: %s" % ", ".join(sorted(missing))


def test_no_runner_is_unreachable_from_the_screen():
    """A runner the screen does not offer is code nobody executes."""
    extra = set(REGISTRY) - _analyses_on_the_screen()
    assert not extra, "runner with no option on the screen: %s" % ", ".join(
        sorted(extra)
    )


def test_every_analysis_has_a_battery_of_parameters_here():
    """If a new runner arrives with no parameters here, the rest of the suite ignores it."""
    assert set(PARAMS) == set(REGISTRY)


def test_an_unknown_analysis_says_which_ones_exist():
    """Before, the message was 'Analysis not implemented yet.', which did not help."""
    with pytest.raises(ValueError) as error:
        get_runner("does_not_exist")
    message = str(error.value)
    assert "does_not_exist" in message
    assert "campbell" in message


# --- the central property: the drawing does not enter the computation --------


class SpyParams(dict):
    """A parameter dictionary that records everything looked up."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_names = set()

    def get(self, key, default=None):
        self.read_names.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read_names.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read_names.add(key)
        return super().__contains__(key)


@pytest.mark.parametrize("name", NAMES)
def test_the_spec_never_reads_a_plot_parameter(name):
    """The central check of the slice, made by spying on the reads.

    `spec` is what goes to `compute` and what forms the cache key. If it consulted
    a plot parameter, that parameter would change the computation without changing
    the key -- and the analysis would give back the result of another
    configuration, with no error on screen at all. That is exactly what Campbell
    did with `plot_type`."""
    runner = REGISTRY[name]
    spy = SpyParams(PARAMS[name])
    runner.spec(spy, ROTOR_REQUEST)

    leak = spy.read_names & set(runner.PLOT_PARAMS)
    assert not leak, "%s: o spec leu parametro de desenho %s" % (
        name,
        ", ".join(sorted(leak)),
    )


@pytest.mark.parametrize("name", NAMES)
def test_changing_a_plot_parameter_leaves_the_spec_untouched(name):
    """The same, by value: it also catches an indirect read, outside the dictionary."""
    runner = REGISTRY[name]
    base = runner.spec(dict(PARAMS[name]), ROTOR_REQUEST)
    for key in runner.PLOT_PARAMS:
        params = dict(PARAMS[name])
        params[key] = "different_value"
        assert runner.spec(params, ROTOR_REQUEST) == base, (
            "%s: plot parameter %r changed the spec" % (name, key)
        )


@pytest.mark.parametrize("name", NAMES)
def test_the_spec_is_json_serialisable(name):
    """The cache key is the fingerprint of the spec; it has to serialise.

    A numpy array inside the spec would become truncated text in the fingerprint,
    and two different arrays could share a key. That is why the spec holds scalars
    and lists, and the vectors are born in compute."""
    spec = REGISTRY[name].spec(dict(PARAMS[name]), ROTOR_REQUEST)
    json.dumps(spec, sort_keys=True)


UNBALANCE_TRIO = ("node", "unbalance_magnitude", "unbalance_phase")


@pytest.mark.parametrize("name", NAMES)
def test_the_unbalance_arguments_leave_in_the_same_shape(name):
    """The guard that would have caught the clearance defect on the day it was written.

    `run_unbalance_response` pairs node, magnitude and phase with a `zip`, and
    then hands what is left to a branch that expects scalars. Any disagreement
    between the three is therefore either a silent loss -- two magnitudes and
    one node means the second magnitude never reaches the rotor -- or a crash;
    which of the two you get depended on the version of numpy, which is why this
    survived four slices before a CI job on numpy 2.5 named it.

    None of the existing guards could see it. They ask whether the spec
    serialises, whether it ignores plot parameters, whether the twelve produce a
    figure -- all true of a spec whose three columns disagree. What was missing
    is the relation BETWEEN the three fields, and a relation has to be measured
    on purpose."""
    spec = REGISTRY[name].spec(dict(PARAMS[name]), ROTOR_REQUEST)
    present = [key for key in UNBALANCE_TRIO if key in spec]
    if not present:
        return
    assert present == list(UNBALANCE_TRIO), (
        "%s carries %s and not the other two: ROSS reads the three together"
        % (name, present)
    )
    shapes = [(key, spec[key]) for key in UNBALANCE_TRIO]
    scalars = [key for key, value in shapes if not isinstance(value, (list, tuple))]
    assert not scalars, (
        "%s sends %s as a scalar next to lists: ROSS zips the three, so the "
        "scalar makes the zip raise and the fallback receives arrays where it "
        "expects numbers" % (name, scalars)
    )
    sizes = {key: len(value) for key, value in shapes}
    assert len(set(sizes.values())) == 1, (
        "%s: the three columns have different lengths %s -- ROSS's zip would "
        "drop the extra rows without saying so" % (name, sizes)
    )


def test_some_runner_really_carries_the_unbalance_trio():
    """Control: if the three fields were renamed, the sweep above would pass empty."""
    carriers = [
        name
        for name in NAMES
        if all(
            key in REGISTRY[name].spec(dict(PARAMS[name]), ROTOR_REQUEST)
            for key in UNBALANCE_TRIO
        )
    ]
    assert sorted(carriers) == [
        "clearance",
        "crack",
        "misalignment",
        "rubbing",
        "unbalance",
    ], "the runners carrying the unbalance trio changed: %s" % sorted(carriers)


@pytest.mark.parametrize("name", NAMES)
def test_declared_plot_parameters_are_actually_used(name):
    """A declaration nobody reads becomes a lie in two refactors.

    This test was **vacuous** until Phase 3 slice 4, and for a reason that only
    shows up on a second look: it searched for the parameter name in the source of
    the runner module, and the tuple `PLOT_PARAMS = ("plot_type", ...)` was itself
    in that source. The declaration satisfied the check of the declaration. It went
    through all of Phase 2 without ever being able to fail.

    With the catalogue outside the runner it started to hold -- and the first thing
    it accused was true: Campbell's `plot_type` is read by no runner. What consumes
    it is the screen, which decides whether the card becomes the mode-shape panel.
    That field has role `interface` in the catalogue, and is checked from the other
    side, in `test_analysis_catalog.py`."""
    import inspect

    def without_comments(source):
        return "\n".join(
            line for line in source.split("\n") if not line.strip().startswith("#")
        )

    # The shared helpers (`probes`, `unbalances`, `units`) read the
    # parameters inside `base.py`, not in the runner's module.
    source = without_comments(
        inspect.getsource(sys.modules[type(REGISTRY[name]).__module__])
    )
    source += without_comments(inspect.getsource(sys.modules["services.analysis.base"]))

    catalog = {c["id"]: c["role"] for c in ANALYSES.get(name, ())}
    for key in REGISTRY[name].PLOT_PARAMS:
        if catalog.get(key) == "interface":
            continue
        assert '"%s"' % key in source or "'%s'" % key in source, (
            "%s declares %r as a plot parameter and never reads it" % (name, key)
        )


# --- the regression the slice found ------------------------------------------


def test_mode_shape_and_default_now_share_a_cache_entry():
    """Both compute the same thing, so they share the result.

    The old code cut the number of steps in mode-shape mode (`if plot_type ==
    'Mode Shape' and s_steps > 15`), but `plot_type` stayed out of the cache key:
    asking for 50 steps, opening in Mode Shape and going back to Default gave back
    a 15-point diagram with the 50-step key. While the cut existed, the two modes
    had to have separate entries.

    Slice 3 removed the cut. It existed because ROSS's `_plot_with_mode_shape` ran
    an extra `run_modal` per critical speed, ahead of time -- and now there is no
    work done ahead at all: the mode shape costs the same as the Default. Without
    the cut, both modes compute exactly the same Campbell, and sharing the cache
    entry becomes the right behaviour."""
    runner = REGISTRY["campbell"]
    base = dict(PARAMS["campbell"], speed_steps="50")

    default_spec = runner.spec(dict(base, plot_type="Default"), ROTOR_REQUEST)
    mode_shape = runner.spec(dict(base, plot_type="Mode Shape"), ROTOR_REQUEST)

    assert default_spec["steps"] == 50
    assert mode_shape["steps"] == 50
    assert default_spec == mode_shape


# --- reading the parameters --------------------------------------------------


@pytest.mark.parametrize(
    "name, field, expected",
    [
        ("campbell", "speed_steps", ("steps", 50)),
        ("campbell", "frequencies", ("frequencies", 6)),
        ("modes", "num_modes", ("num_modes", 12)),
        ("ucs", "num_modes", ("num_modes", 4)),
        ("harmonic_balance", "n_harmonics", ("n_harmonics", 1)),
    ],
)
def test_a_blank_resolution_field_falls_back_to_the_default(name, field, expected):
    """How many steps, how many modes: blank means 'the usual one'.

    A deliberate deviation. Before, `int(float(''))` raised and the analysis came
    back with "could not convert string to float: ''" -- a message that does not
    even say which field was empty."""
    key, value = expected
    spec = REGISTRY[name].spec(dict(PARAMS[name], **{field: ""}), ROTOR_REQUEST)
    assert spec[key] == value


@pytest.mark.parametrize(
    "name, field",
    [
        ("crack", "depth_ratio"),
        ("crack", "n"),
        ("rubbing", "distance"),
        ("rubbing", "contact_stiffness"),
        ("rubbing", "contact_damping"),
        ("rubbing", "friction_coeff"),
        ("rubbing", "n"),
    ],
)
def test_a_blank_physical_field_is_refused_by_name(name, field):
    """Here a default would be worse than an error.

    A blank `depth_ratio` would become zero depth, and the crack analysis would
    give back an intact rotor -- a wrong result with no warning, which is the
    antipattern this whole refactor is chasing. So the field is required, and the
    message says which one it is.

    `("clearance", "node")` used to be on this list and is now the test below.
    The field it named no longer exists: the clearance node came from a text box
    that could be left blank, and it now comes from a table cell that always
    carries a number. What survived of that requirement is the row itself -- a
    table with no rows is refused the same way, and for the same reason."""
    with pytest.raises(ValueError) as error:
        REGISTRY[name].spec(dict(PARAMS[name], **{field: ""}), ROTOR_REQUEST)
    assert field in str(error.value)


def test_clearance_refuses_an_empty_unbalance_table():
    """The excitation cannot be invented: it is what the analysis measures.

    The clearance analysis compares the vibration at the bearings against the
    radial gap, and that vibration IS the response to this unbalance. Standing
    in a default row would answer a question the user did not ask, with a number
    that looks perfectly valid. So an empty table raises, and the message says
    which analysis and what to do."""
    with pytest.raises(ValueError) as error:
        REGISTRY["clearance"].spec(
            dict(PARAMS["clearance"], unbalances=[]), ROTOR_REQUEST
        )
    assert "clearance" in str(error.value)


def test_the_other_analyses_still_stand_in_a_row_when_the_table_is_empty():
    """Control: the refusal above is clearance's, not everyone's.

    Without this, moving the fallback out of `unbalances()` altogether would
    look like a passing test instead of a behaviour change for four analyses."""
    for name in ("unbalance", "misalignment", "rubbing", "crack"):
        spec = REGISTRY[name].spec(dict(PARAMS[name], unbalances=[]), ROTOR_REQUEST)
        assert spec["node"], "%s came back with no unbalance row" % name


def test_a_physical_field_with_junk_says_so():
    with pytest.raises(ValueError) as error:
        REGISTRY["crack"].spec(dict(PARAMS["crack"], depth_ratio="abc"), ROTOR_REQUEST)
    assert "depth_ratio" in str(error.value)
    assert "abc" in str(error.value)


def test_optional_fields_left_blank_stay_out_of_the_call():
    """Not sending the parameter is how ROSS uses its own default."""
    runner = REGISTRY["misalignment"]
    params = dict(PARAMS["misalignment"])
    params["input_torque"] = ""
    del params["load_torque"]
    extras = runner.spec(params, ROTOR_REQUEST)["extras"]
    assert "input_torque" not in extras
    assert "load_torque" not in extras
    assert extras["coupling"] == "flex"


def test_a_rigid_coupling_does_not_carry_the_flexible_fields():
    runner = REGISTRY["misalignment"]
    params = dict(PARAMS["misalignment"], coupling="rigid", mis_distance="2e-4")
    extras = runner.spec(params, ROTOR_REQUEST)["extras"]
    assert extras["mis_distance"] == 2e-4
    assert "mis_distance_x" not in extras
    assert "mis_angle" not in extras


# --- what is left of the comparison with the old code ------------------------
#
# During slice 2, tests/legacy_analysis.py kept the original if/elif frozen, and
# the twelve analyses ran through both paths comparing `fig.to_json()`. Result:
# 12 of 12 identical in the base configurations and, across 90 variants, 79
# identical plus 11 differences all deliberate (discretisation defaults and
# physical fields refused by name). Green in the venv with the real ROSS, that
# file went -- it existed for one slice only.
#
# The rotor below stays: the tests that depend on the real ROSS continue.

try:
    import ross as rs

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")

# Small computations: these tests run the real ROSS.
LEAN = {
    "speed_steps": "8",
    "num_modes": "4",
    "steps": "40",
    "t_steps": "40",
    "t_max": "0.02",
    "t_final": "0.02",
    "cross_divisions": "6",
    "n_harmonics": "1",
}


def _test_rotor():
    steel = rs.materials.steel
    shafts = [
        rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel, n=i)
        for i in range(3)
    ]
    return rs.Rotor(
        shaft_elements=shafts,
        disk_elements=[rs.DiskElement(n=1, m=5.0, Id=0.02, Ip=0.04, tag="disk_0")],
        bearing_elements=[
            rs.BearingElement(n=0, kxx=1e6, cxx=1e3, tag="b0"),
            rs.BearingElement(n=3, kxx=1e6, cxx=1e3, tag="b1"),
        ],
    )


def _lean_params(name):
    params = dict(PARAMS[name])
    for key, value in LEAN.items():
        if key in params:
            params[key] = value
    return params


@needs_ross
@pytest.mark.parametrize("name", NAMES)
def test_every_runner_produces_a_figure_from_a_real_rotor(name):
    """End to end on all twelve: spec, compute and plot over a real rotor.

    It is what is left of the comparison with the old code. It no longer checks
    that the figure is the same as before -- it checks that the twelve paths
    complete without blowing up, which is what still makes sense now that the old
    code is gone."""
    rotor = _test_rotor()
    params = _lean_params(name)
    runner = REGISTRY[name]

    spec = runner.spec(dict(params), rotor)
    figure = runner.plot(runner.compute(rotor, spec), dict(params), rotor)
    assert figure.to_json()


@needs_ross
def test_the_force_expression_error_still_names_the_node_and_dof():
    """BE-05 stays covered after the change of house."""
    rotor = _test_rotor()
    runner = REGISTRY["time_response"]
    params = dict(
        _lean_params("time_response"),
        forces=[{"node": 1, "dof": 0, "func": "1000 * cs(speed*t)"}],
    )
    spec = runner.spec(params, rotor)
    with pytest.raises(ValueError) as error:
        runner.compute(rotor, spec)
    assert "Invalid force" in str(error.value)
    assert "node 1" in str(error.value)


# --- misalignment: ROSS has no default for anything --------------------------


@pytest.mark.parametrize(
    "field",
    [
        "mis_distance_x",
        "mis_distance_y",
        "radial_stiffness",
        "n",
    ],
)
def test_a_blank_parallel_misalignment_field_is_refused_by_name(field):
    """`kwargs.get(...)` with no default becomes None, and the computation blows up in ROSS.

    The message from there is `unsupported operand type(s) for ** or pow():
    'NoneType' and 'int'` -- which does not say which field was empty. Here it does."""
    with pytest.raises(ValueError) as error:
        REGISTRY["misalignment"].spec(
            dict(PARAMS["misalignment"], **{field: ""}), ROTOR_REQUEST
        )
    assert field in str(error.value)


@pytest.mark.parametrize(
    "kind, required_fields",
    [
        ("parallel", ("mis_distance_x", "mis_distance_y", "radial_stiffness")),
        ("angular", ("mis_angle", "bending_stiffness")),
        (
            "combined",
            (
                "mis_distance_x",
                "mis_distance_y",
                "mis_angle",
                "radial_stiffness",
                "bending_stiffness",
            ),
        ),
    ],
)
def test_each_misalignment_type_asks_for_its_own_fields(kind, required_fields):
    """ROSS picks the force function by type, and each one uses its own fields."""
    extras = REGISTRY["misalignment"].spec(
        dict(PARAMS["misalignment"], mis_type=kind), ROTOR_REQUEST
    )["extras"]
    for field in required_fields:
        assert field in extras


def test_an_unknown_misalignment_type_lists_the_valid_ones():
    """The ROSS message is "Check the misalignment type!", without saying which."""
    with pytest.raises(ValueError) as error:
        REGISTRY["misalignment"].spec(
            dict(PARAMS["misalignment"], mis_type="torto"), ROTOR_REQUEST
        )
    message = str(error.value)
    assert "torto" in message
    for kind in ("parallel", "angular", "combined"):
        assert kind in message


def test_a_rigid_coupling_needs_its_own_distance():
    with pytest.raises(ValueError) as error:
        REGISTRY["misalignment"].spec(
            dict(PARAMS["misalignment"], coupling="rigid"), ROTOR_REQUEST
        )
    assert "mis_distance" in str(error.value)


def test_the_time_response_refuses_an_empty_force_table():
    """With no force the response is zero at every node -- a flat chart that looks
    like a result. The same reasoning as BE-05: null excitation does not go through
    a computation.

    The probe exposed this: sending the empty table, `time_response` came out
    "NO EFFECT" on both conversions -- and zero is zero at any number of degrees of
    freedom, so what looked like incompatibility was an absence of excitation."""
    runner = REGISTRY["time_response"]
    with pytest.raises(ValueError) as error:
        runner.spec({"forces": []}, ROTOR_REQUEST)
    assert "force" in str(error.value).lower()


def test_the_frequency_response_survives_an_empty_probe_table():
    """`[]` is not an absent key: `get(key, default)` never saw the default.

    The screen sends `[]` when the user deletes every row. The loop over the
    input/output pairs did not run, the figure stayed None, and the error that
    arrived was "'NoneType' object has no attribute 'update_layout'" -- pointing at
    Plotly when the problem was an empty table. The helpers in `base.py` have
    always used `or`; this one was the only one outside the pattern."""
    import inspect

    js = inspect.getsource(sys.modules["services.analysis.freq_response"])
    assert 'params.get("inps", ' not in js
    assert 'params.get("outs", ' not in js
    assert 'params.get("inps") or' in js
    assert 'params.get("outs") or' in js


def test_no_runner_reads_a_list_with_a_default_that_an_empty_list_hides():
    """The general guard: `get(key, [...])` over a screen table is a trap.

    The key always exists -- the screen sends `[]` when the user deletes the rows --
    so the `get` default never comes in. That is how it went in `freq_response`,
    and the only way not to repeat it is to require the idiom."""
    import os as _os

    culpados = []
    folder = os.path.join(ROOT, "services", "analysis")
    for handle in sorted(_os.listdir(folder)):
        if not handle.endswith(".py"):
            continue
        with io.open(os.path.join(folder, handle), encoding="utf-8") as js:
            text = js.read()
        for number, line in enumerate(text.split("\n"), 1):
            if line.strip().startswith("#"):
                continue
            if re.search(r'params\.get\("[a-z_]+",\s*\[', line):
                culpados.append("%s:%d" % (handle, number))
    assert culpados == [], "a list default that a [] hides in %s" % culpados


def test_the_two_ends_of_the_bearing_range_compose_one_ross_argument():
    """Two fields on screen, one argument to ROSS -- the campbell's idiom.

    What this pins is the composition itself: ROSS takes a single
    `bearing_frequency_range`, and the screen asks for it the way it asks for a
    speed range, because a pair of numbers is easier to fill in and to validate
    than a list typed by hand."""
    spec = get_runner("ucs").spec(
        {
            "k_min": "6",
            "k_max": "11",
            "bearing_freq_min": "0",
            "bearing_freq_max": "1000",
        },
        None,
    )
    assert spec["bearing_frequency_range"] == (0.0, 1000.0)


def test_both_ends_empty_asks_ross_nothing():
    """Control, and the normal case: with no range ROSS derives one itself.

    The key has to be **absent**, not present as `None`. Every run of this
    analysis for months went this way, and it must keep going this way for
    whoever never touches the advanced fields."""
    for value in ("", "   ", None):
        spec = get_runner("ucs").spec(
            {
                "k_min": "6",
                "k_max": "11",
                "bearing_freq_min": value,
                "bearing_freq_max": value,
            },
            None,
        )
        assert "bearing_frequency_range" not in spec


def test_half_a_range_is_refused_by_name():
    """A question the user started and did not finish.

    Guessing the missing end would answer a question nobody asked -- and it
    would do it silently, which is the antipattern this refactoring chases. Both
    directions, because a guard written for one end passes for the other."""
    for filled, blank in (
        ("bearing_freq_min", "bearing_freq_max"),
        ("bearing_freq_max", "bearing_freq_min"),
    ):
        with pytest.raises(ValueError) as error:
            get_runner("ucs").spec(
                {"k_min": "6", "k_max": "11", filled: "500", blank: ""}, None
            )
        assert str(error.value) == HALF_A_RANGE


def test_zero_is_a_value_and_not_an_empty_field():
    """The reason the runner reads these two as text and not as numbers.

    Zero is a legitimate lower end of a frequency range, and `number()` cannot
    tell it from a blank field. Read as numbers, a range starting at zero would
    silently become no range at all -- the user asks one question and ROSS
    answers another."""
    spec = get_runner("ucs").spec(
        {
            "k_min": "6",
            "k_max": "11",
            "bearing_freq_min": "0",
            "bearing_freq_max": "0",
        },
        None,
    )
    assert spec["bearing_frequency_range"] == (0.0, 0.0)
