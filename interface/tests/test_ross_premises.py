# -*- coding: utf-8 -*-
"""What this interface assumes about ROSS itself.

Every test here breaks when **ROSS** changes, not when our code changes. Being
in a single file has one practical reason: when the library pin in
`requirements.txt` goes up, this is the first file to run and the only one that
should fail. A failure here is a message about the library, and not a defect of
ours.

They come in three kinds:

* **names ROSS may rename** -- the parameters of a seal, the way a rotor counts
  degrees of freedom;
* **a private method the interface calls** -- `_update_plot_mode_3d`, which
  draws the mode shape. Calling the private one was a conscious decision, and
  this guard is its price;
* **behaviour premises** -- that `run_campbell` already fills the modal results
  of every speed (which makes computing the mode shape ahead of time
  unnecessary), and that the `bearing_frequency_range` defect is still there.

The last one is the only test in the suite that **expects** a third-party
defect. When ROSS fixes it, this test fails -- and the guard that exists because
of it can then go."""

import inspect
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import ross as rs

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")

from domain.rotor_builder import build_rotor_from_ui

# Point masses live on *link* nodes, past the last node of the shaft, connected
# by a bearing with n_link -- and not on nodes of the shaft itself. See the
# equivalent fixture in ross/tests/test_rotor_assembly.py
# (BearingElement(0, n_link=7) + PointMass(7)).
POINT_MASS_ROTOR = {
    "materials": [{"name": "Steel", "rho": "7800", "E": "211e9", "G_s": "81.2e9"}],
    "shafts": [
        {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "0"},
        {"L": "500", "odl": "100", "idl": "0", "material": "Steel", "n": "1"},
    ],
    "bearings": [
        {"element_type": "BASIC", "kxx": "1e6", "cxx": "0", "n": "0", "n_link": "3"},
        {"element_type": "BASIC", "kxx": "1e6", "cxx": "0", "n": "3"},
        {"element_type": "BASIC", "kxx": "1e6", "cxx": "0", "n": "2", "n_link": "4"},
        {"element_type": "BASIC", "kxx": "1e6", "cxx": "0", "n": "4"},
    ],
    "pointmasses": [{"m": "2", "n": "3"}, {"m": "2", "n": "4"}],
}


@needs_ross
def test_the_private_ross_method_still_has_the_signature_we_call():
    """`_update_plot_mode_3d` is private; if it changes, we want to know the same day."""
    parameters = list(
        inspect.signature(rs.CampbellResults._update_plot_mode_3d).parameters
    )
    assert parameters == [
        "self",
        "speed",
        "natural_frequency",
        "modal_results_crit",
        "speed_units",
        "frequency_units",
        "damping_parameter",
        "animation",
    ]


@needs_ross
def test_run_campbell_still_fills_modal_results_for_every_speed():
    """The premise that makes computing the mode shape ahead of time unnecessary.

    `_update_plot_mode_3d` looks the speed up in `self.modal_results` and only
    falls back to `modal_results_crit` -- which we pass empty -- if it does not
    find it. If ROSS stops filling that in, the click would go down the fallback
    path and find nothing."""
    steel = rs.materials.steel
    shafts = [
        rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel, n=i)
        for i in range(3)
    ]
    rotor = rs.Rotor(
        shaft_elements=shafts,
        bearing_elements=[
            rs.BearingElement(n=0, kxx=1e6, cxx=1e3, tag="b0"),
            rs.BearingElement(n=3, kxx=1e6, cxx=1e3, tag="b1"),
        ],
    )

    import numpy as np

    speeds = np.linspace(0, 300, 4)
    result = rotor.run_campbell(speeds, frequencies=4)

    assert set(result.modal_results) == set(speeds)


@needs_ross
def test_ross_accepts_a_bearing_frequency_range():
    """The premise that used to be the opposite one, and that is the point.

    This test used to assert the **defect**: `run_ucs` raised on any
    `bearing_frequency_range`, because `@check_units` turned the sequence into a
    numpy array and the method then did `if bearing_frequency_range:`. Its
    docstring said what should happen when ROSS fixed it -- *"this test fails,
    and then the guard can go"* -- and that is exactly how the fix was found:
    moving the pin to commit `2a253e6` turned it red, alone, in an otherwise
    green run.

    So it was inverted rather than deleted. The interface offers the field
    again, and what holds that offer up is a premise about ROSS; a premise with
    no test is a hope. Install a ROSS without the fix and this says so, instead
    of the user discovering it on a form that stopped working.
    """
    steel = rs.materials.steel
    shafts = [
        rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel, n=i)
        for i in range(3)
    ]
    rotor = rs.Rotor(
        shaft_elements=shafts,
        bearing_elements=[
            rs.BearingElement(n=0, kxx=1e6, cxx=1e3, tag="b0"),
            rs.BearingElement(n=3, kxx=1e6, cxx=1e3, tag="b1"),
        ],
    )

    result = rotor.run_ucs(
        stiffness_range=(6, 11),
        num=5,
        num_modes=16,
        bearing_frequency_range=[0, 1000],
    )
    assert result is not None


@needs_ross
def test_ross_keeps_a_tag_that_was_already_set():
    """set_tag only names what has no name yet. The interface always names."""
    bearing = rs.BearingElement(n=0, kxx=1e6, cxx=1e3, tag="bearing_0")
    bearing.set_tag(99)
    assert bearing.tag == "bearing_0"


def test_labyrinth_seal_accepts_current_parameter_names():
    """The names renamed by PR #1360 have to exist on the class."""
    import inspect

    params = inspect.signature(rs.LabyrinthSeal.__init__).parameters
    for name in (
        "shaft_diameter",
        "molar_mass",
        "reference_temperatures",
        "reference_viscosities",
        "gas_model",
        "use_jenny_kanki",
        "print_results",
    ):
        assert name in params, f"{name} is gone from LabyrinthSeal's signature"
    for old in ("molar", "tz", "muz", "analz", "nprt", "iopt1"):
        assert old not in params, f"{old} voltou a existir -- rever os formularios"


def test_holepattern_seal_accepts_current_parameter_names():
    import inspect

    params = inspect.signature(rs.HolePatternSeal.__init__).parameters
    for name in (
        "molar_mass",
        "sutherland_b",
        "sutherland_s",
        "entrance_loss_coefficient",
        "exit_loss_coefficient",
        "excitation_ratio",
        "relaxation_factor",
    ):
        assert name in params, f"{name} is gone from HolePatternSeal's signature"


def test_number_dof_matches_ross_with_point_masses():
    """The division ndof // len(nodes) is wrong as soon as there are point masses.

    rotor.nodes holds only the nodes of the shaft, but ndof adds half a degree of
    freedom per point mass -- so the integer division does not give the DoF per
    node."""
    rotor = build_rotor_from_ui(POINT_MASS_ROTOR)

    assert rotor.number_dof == 6
    assert len(rotor.nodes) == 3
    assert len(rotor.point_mass_elements) == 2
    assert rotor.ndof == 6 * 3 + 3 * 2  # 24

    # The old calculation would give back 8 DoF per node instead of 6, and every
    # global index (g_inp, g_out, g_dof) would point at the wrong degree of freedom.
    assert rotor.ndof // len(rotor.nodes) == 8
    assert rotor.number_dof == 6
