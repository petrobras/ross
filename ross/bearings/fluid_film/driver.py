"""Top-level solver orchestration.

Converts the raw inputs, builds the meshes, loops over the speed/load cases,
and drives the hydrodynamics -> pressure -> temperature -> deformation ->
coefficient chain to produce the bearing outputs.
:func:`run_case` is the entry point.

Solver state travels through the case loop in one dict, ``g``, keyed by the
snake_case names the solver modules use.

Everything is 0-based: node, element and pad numbers index the arrays
directly, per-pad arrays are shaped ``(total_pads, dim...)``, and there is no
padding or value shift anywhere (see :mod:`ross.bearings.fluid_film.mesh`).

The package is SI throughout; ``run_case`` accepts SI and returns SI.
"""

import inspect
from types import SimpleNamespace

import numpy as np

from ross.bearings.fluid_film._numba_kernels import (
    include_press_jit,
    integrate_xz_jit,
    trapezoid_jit,
)
from ross.bearings.fluid_film.constants import (
    PI,
    PIVOT_FLEX_DEFORM_TYPES,
    TILTING_PAD_TYPES,
)
from ross.bearings.fluid_film.state import (
    EnergyMesh,
    Lubricant,
    OperatingPoint,
    PadGeometry,
    ReynoldsMesh,
    Turbulence,
)

# "No data" sentinel for temperature outputs whose thermal model did not run:
# the zero of the Fahrenheit scale expressed in kelvin. Downstream consumers
# have always received this exact placeholder for isoviscous runs; changing it
# would silently shift every "no thermal data" temperature they see.
ZERO_TEMPERATURE_SENTINEL = 255.3722222222222

# Output-gating floors. Each is the exact SI equivalent of a legacy
# unit-system epsilon (1e-6 in^3/s, 1e-6 in, 1e-6 lbf/in, 1e-6 lbf) so the
# same operating points trigger the same output branches as before; rounding
# them would flip the gating on borderline cases.
_SUPPLY_FLOW_FLOOR = 1.0e-6 * 0.0254**3
_CRUSH_FLOOR = 1.0e-6 * 0.0254
_MIDPLANE_TOL = 1.0e-6 * 0.0254
_STIFFNESS_FLOOR = 1.0e-6 * (4.4482216152605 / 0.0254)
_FORCE_FLOOR = 1.0e-6 * 4.4482216152605

# Gravity constant used by the rigid-rotor stability threshold. The threshold
# formula was calibrated with g = 386.0 in/s^2 (not the standard 386.088);
# keep the exact equivalent so threshold speeds reproduce.
_GRAVITY = 386.0 * 0.0254

# Deformation modes whose thermal growth of journal and shell is reported.
_THERMAL_GROWTH_DEFORM_TYPES = (
    "pad_mechanical_thermal",
    "pad_mechanical_thermal_shaft_shell_thermal",
    "pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical",
)

# Operating modes with an oil-starved inlet, for which the continuous-film
# onset angle is reported per pad.
_STARVED_OPERATING_TYPES = (
    "starved_condition_even",
    "starved_condition_uneven",
    "oil_ring_lubricated",
)

# ``ReynoldsMesh`` field -> the key it is stored under in the solver state.
_REYNOLDS_MESH_KEYS = {
    "dim_x": "dim_x",
    "dim_z": "dim_z",
    "dim_yf": "dim_yf",
    "dim_xz": "dim_xz",
    "dim_3d": "dim_3d",
    "bandwidth": "bandwidth_reynolds",
    "total_e_x_film": "total_e_x_film",
    "total_e_y_film": "total_e_y_film",
    "total_e_z_film": "total_e_z_film",
    "total_e_y_trackbl": "total_e_y_trackbl",
    "total_e_y_trackcore": "total_e_y_trackcore",
    "total_nodes": "total_n_reynolds",
    "total_elements": "total_e_reynolds",
    "n_index": "n_index_reynolds",
    "e_index": "e_index_reynolds",
    "node_i": "node_i_reynolds",
    "node_j": "node_j_reynolds",
    "node_k": "node_k_reynolds",
    "node_l": "node_l_reynolds",
    "match_nodes_xz": "match_nodes_xz",
    "x": "x_reynolds",
    "z": "z_reynolds",
    "x_rad": "x_reynolds_rad",
    "y_3d": "y_3d",
    "e_length": "e_length_reynolds",
    "e_width": "e_width_reynolds",
    "dx": "dx_reynolds",
    "dz": "dz_reynolds",
}


# ``PadGeometry`` field -> the key it is stored under in the solver state.
# Every field already carries its own name, so the mapping is the identity.
_PAD_GEOMETRY_KEYS = {
    f: f
    for f in (
        "journal_radius",
        "pad_thickness",
        "arc_length_rad",
        "pad_length",
        "axial_length",
        "leading_angle_rad",
        "x_pivot_rad",
        "x_pivot",
        "cp",
        "preload",
        "offset",
        "depth_track",
        "length_track",
        "length_track_rad",
        "axial_length_track",
        "length_dam",
        "axial_length_dam",
        "length_pocket",
        "axial_length_pocket",
        "length_ramp_le",
        "length_ramp_te",
        "dh_ramp_le",
        "dh_ramp_te",
    )
}


# ``EnergyMesh`` field -> the key it is stored under in the solver state.
_ENERGY_MESH_KEYS = {
    "dim_xy": "dim_xy",
    "dim_xy2": "dim_xy2",
    "bandwidth": "bandwidth_energy",
    "total_e_y_pad": "total_e_y_pad",
    "total_nodes": "total_n_energy",
    "total_elements": "total_e_energy",
    "n_index": "n_index_energy",
    "e_index": "e_index_energy",
    "node_1": "node_1_energy",
    "node_2": "node_2_energy",
    "node_3": "node_3_energy",
    "node_4": "node_4_energy",
    "match_nodes_xy": "match_nodes_xy",
    "x": "x_energy",
    "y": "y_energy",
}


# ``Lubricant`` field -> the key it is stored under in the solver state.
_LUBRICANT_KEYS = {
    "viscosity1": "viscosity1",
    "viscosity2": "viscosity2",
    "temp1": "temp1",
    "temp2": "temp2",
    "density": "lube_density",
    "cp": "lube_cp",
    "conduct": "lube_conduct",
}


# ``OperatingPoint`` field -> the key it is stored under in the solver state.
_OPERATING_POINT_KEYS = {
    f: f
    for f in (
        "bearing_type",
        "operating_type",
        "speed_surface",
        "temp_supply",
        "press_supply",
        "press_cavitate",
        "ambient_press1",
        "ambient_press2",
    )
}


def trapezoid(t, f, start, stop):
    """Integrate ``f(t)`` by the trapezoid rule over an unequal grid.

    Integrates the samples ``t[start:stop]``, using ordinary Python slice
    bounds.

    Parameters
    ----------
    t : array_like
        Independent coordinate at each sample.
    f : array_like
        Integrand sampled at each ``t``.
    start : int
        Index of the first sample.
    stop : int
        Index one past the last sample.

    Returns
    -------
    float
        The integral. Zero when the slice holds fewer than two samples.

    Examples
    --------
    >>> float(trapezoid([0.0, 1.0, 3.0], [1.0, 1.0, 1.0], 0, 3))
    3.0
    """
    if stop - start < 2:
        return 0.0
    t = np.ascontiguousarray(t, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)
    return trapezoid_jit(t, f, start, stop)


def convert(
    journal_diameter,
    pivot_angle,
    offset,
    pad_arc,
    track_arc,
    pad_axial_length,
    track_axial_length,
    taper_arc_le,
    taper_arc_te,
    pocket_arc,
    radial_clearance,
    preload,
    xj,
    yj,
    qs_supply,
    weight_e,
    weight_h,
):
    """Derive secondary geometry/state quantities from the SI inputs.

    Inputs are SI throughout (meters, radians, m^3/s, dimensionless ratios).
    The journal position is dimensionalized by the bore clearance
    ``radial_clearance``, the mesh concentration factors go from percent to
    fraction, and the pad-local angle / length quantities (``arc_length_rad``
    -> ``pad_length`` via journal radius, etc.) are built up. Inputs and
    outputs are both SI, so no unit conversion happens here.

    Parameters
    ----------
    journal_diameter : float
        Journal diameter (m).
    pivot_angle : array_like
        Pivot angular position of each pad (rad).
    offset : array_like
        Pad offset factor (dimensionless).
    pad_arc : array_like
        Pad arc length (rad).
    track_arc : array_like
        Pressure-dam track arc length (rad).
    pad_axial_length : array_like
        Pad axial length (m).
    track_axial_length : array_like
        Track axial length (m).
    taper_arc_le, taper_arc_te : array_like
        Leading/trailing-edge taper arc length (rad).
    pocket_arc : float
        Inlet-groove pocket arc length (rad).
    radial_clearance : float
        Bore assembly radial clearance (m).
    preload : array_like
        Pad preload (dimensionless).
    xj, yj : float
        Initial journal position, nondimensionalized by ``radial_clearance``.
    qs_supply : array_like
        Per-case supply flow rate (m^3/s).
    weight_e, weight_h : float
        Mesh concentration factors (percent).

    Returns
    -------
    dict
        Converted quantities (all SI): ``journal_radius`` (m),
        ``length_pocket`` (m), ``xj``/``yj`` (m, dimensionalized by clearance),
        ``arc_length_rad`` (rad), ``leading_angle_rad`` (rad),
        ``length_ramp_le``/``length_ramp_te`` (m), ``pad_length`` (m),
        ``cp`` (m), ``x_pivot_rad`` (rad), ``x_pivot`` (m),
        ``length_track_rad`` (rad), ``length_track`` (m), ``length_dam`` (m),
        ``axial_length_dam`` (m), ``qs_supply`` (m^3/s), ``weight_e`` /
        ``weight_h`` (fraction).
    """
    pivot_angle = np.asarray(pivot_angle, dtype=float)
    offset = np.asarray(offset, dtype=float)
    pad_arc = np.asarray(pad_arc, dtype=float)
    track_arc = np.asarray(track_arc, dtype=float)
    pad_axial_length = np.asarray(pad_axial_length, dtype=float)
    track_axial_length = np.asarray(track_axial_length, dtype=float)
    taper_arc_le = np.asarray(taper_arc_le, dtype=float)
    taper_arc_te = np.asarray(taper_arc_te, dtype=float)
    preload = np.asarray(preload, dtype=float)
    qs_supply = np.asarray(qs_supply, dtype=float)

    journal_radius = 0.5 * journal_diameter
    # Angles are already in rad and lengths in m, so nothing to convert.
    length_pocket = pocket_arc * journal_radius

    arc_length_rad = pad_arc
    leading_angle_rad = pivot_angle - arc_length_rad * offset
    pad_length = arc_length_rad * journal_radius
    length_ramp_le = taper_arc_le * journal_radius
    length_ramp_te = taper_arc_te * journal_radius
    cp = radial_clearance / (1.0 - preload)
    x_pivot_rad = offset * arc_length_rad
    x_pivot = offset * pad_length
    length_track_rad = track_arc
    length_track = length_track_rad * journal_radius
    length_dam = pad_length - length_track
    axial_length_dam = 0.5 * (pad_axial_length - track_axial_length)

    return {
        "journal_radius": journal_radius,
        "length_pocket": length_pocket,
        "xj": xj * radial_clearance,
        "yj": yj * radial_clearance,
        "arc_length_rad": arc_length_rad,
        "leading_angle_rad": leading_angle_rad,
        "length_ramp_le": length_ramp_le,
        "length_ramp_te": length_ramp_te,
        "pad_length": pad_length,
        "cp": cp,
        "x_pivot_rad": x_pivot_rad,
        "x_pivot": x_pivot,
        "length_track_rad": length_track_rad,
        "length_track": length_track,
        "length_dam": length_dam,
        "axial_length_dam": axial_length_dam,
        "qs_supply": qs_supply,
        "weight_e": 0.01 * weight_e,
        "weight_h": 0.01 * weight_h,
    }


def case_parameters(
    case_index,
    journal_diameter,
    speeds_rpm,
    weight,
    fxs_ext,
    fys_ext,
    qs_supply,
    excit_ratios,
):
    """Select and convert the parameters for one speed/load case.

    ``case_index`` counts from 1.

    Parameters
    ----------
    case_index : int
        Case number (1-based).
    journal_diameter : float
        Journal diameter (m).
    speeds_rpm : array_like
        Shaft speed per case (RPM).
    weight : array_like
        Gravity load per case (N).
    fxs_ext, fys_ext : array_like
        External load per case in X and Y (N).
    qs_supply : array_like
        Supply flow per case (m^3/s, already converted by :func:`convert`).
    excit_ratios : array_like
        Whirl/rotational frequency ratio per case.

    Returns
    -------
    dict
        ``speed_rpm``, ``fx_ext``, ``fy_ext``, ``q_supply``, ``excit_ratio``,
        ``speed_surface`` (journal surface speed) and ``excit_rad`` (whirl
        frequency, rad/s).
    """
    i = case_index - 1
    speed_rpm = float(speeds_rpm[i])
    fx_ext = float(fxs_ext[i])
    fy_ext = float(fys_ext[i]) - float(weight[i])
    q_supply = float(qs_supply[i])
    excit_ratio = float(excit_ratios[i])

    speed_surface = speed_rpm * PI * journal_diameter / 60.0
    excit_rad = excit_ratio * speed_rpm * PI / 30.0

    return {
        "speed_rpm": speed_rpm,
        "fx_ext": fx_ext,
        "fy_ext": fy_ext,
        "q_supply": q_supply,
        "excit_ratio": excit_ratio,
        "speed_surface": speed_surface,
        "excit_rad": excit_rad,
    }


def integrate_xz(
    pad_index,
    mesh,
    f,
):
    """Integrate a nodal field ``f`` over the Reynolds mesh of one pad.

    Uses the four-node element average times the element area. ``pad_index``
    and all node/element ids are 0-based natural, indexing the mesh module's
    0-based arrays.

    Parameters
    ----------
    pad_index : int
        Pad index (0-based).
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
        Supplies the element connectivity and the per-element sizes.
    f : array_like
        Nodal field keyed by node id.

    Returns
    -------
    float
        The surface integral of ``f`` over the pad.
    """
    return integrate_xz_jit(
        pad_index,
        mesh.total_elements,
        np.ascontiguousarray(mesh.e_index, dtype=np.int64),
        np.ascontiguousarray(mesh.node_i, dtype=np.int64),
        np.ascontiguousarray(mesh.node_j, dtype=np.int64),
        np.ascontiguousarray(mesh.node_k, dtype=np.int64),
        np.ascontiguousarray(mesh.node_l, dtype=np.int64),
        np.ascontiguousarray(mesh.e_length, dtype=np.float64),
        np.ascontiguousarray(mesh.e_width, dtype=np.float64),
        np.ascontiguousarray(f, dtype=np.float64),
    )


def sump_temp(power_loss_value, area_sump_convec, temp_environment, convec_environment):
    """Return the sump temperature from bulk energy conservation.

    Parameters
    ----------
    power_loss_value : float
        Bearing power loss.
    area_sump_convec : float
        Convection area of the bearing housing.
    temp_environment : float
        Environment temperature.
    convec_environment : float
        Environment heat-convection coefficient.

    Returns
    -------
    float
        Sump temperature.
    """
    return temp_environment + power_loss_value / (area_sump_convec * convec_environment)


def drain_temp(q_supply, lube, power_loss_value, temp_supply):
    """Return the drain temperature from bulk energy conservation.

    Parameters
    ----------
    q_supply : float
        Supply flow rate, m^3/s.
    lube : Lubricant
        Lubricant properties.
    power_loss_value : float
        Bearing power loss, W.
    temp_supply : float
        Supply temperature, K.

    Returns
    -------
    float
        Drain (sump) temperature, K.
    """
    return temp_supply + power_loss_value / (lube.density * lube.cp * q_supply)


def power_loss(
    total_pads,
    mesh,
    scale_dissip,
    dudy_n,
    vis_effect_3d,
    speed_surface,
):
    """Return bearing power loss from the journal-surface shear stress.

    All node/element ids and ``pad`` axes are 0-based (mesh-module convention).
    The journal surface row of the 3-D mesh is ``match_nodes_xz[node,
    total_e_y_film]``.

    Parameters
    ----------
    total_pads : int
        Pad count.
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
        Supplies the ``match_nodes_xz`` Reynolds-node -> 3-D-node map.
    scale_dissip : ndarray
        Dissipation scaling per node, shape ``(total_pads, dim_xz)``.
    dudy_n : ndarray
        Velocity gradients on the 3-D mesh, 1/s.
    vis_effect_3d : ndarray
        Effective viscosity on the 3-D mesh, Pa*s.
    speed_surface : float
        Journal surface speed, m/s.

    Returns
    -------
    tuple of float
        ``(power_loss, hp)`` -- power loss in watts. The two values are
        identical; the pair is kept because callers unpack it.
    """
    surface_row = mesh.total_e_y_film
    tao = np.zeros(mesh.dim_xz)
    total_power_loss = 0.0
    for pad in range(total_pads):
        for i in range(mesh.total_nodes):
            node = mesh.n_index[i]
            node_3d = mesh.match_nodes_xz[node, surface_row]
            tao[node] = (
                scale_dissip[pad, node]
                * vis_effect_3d[pad, node_3d]
                * np.sqrt(dudy_n[pad, node_3d] ** 2)
            )
        inte_tao = integrate_xz(
            pad,
            mesh,
            tao,
        )
        # SI: speed_surface [m/s] * inte_tao [N] = W directly. The
        # imperial /9336 (in*lbf/s -> BTU/s) and *3600/2545 (BTU/s -> hp)
        # are no longer needed.
        total_power_loss += speed_surface * inte_tao

    return total_power_loss, total_power_loss


def deform_shaftshell(
    deform_type,
    bearing_type,
    journal_radius,
    pad_thickness,
    journal_expand,
    shell_expand,
    temp_j,
    temp_sump,
    temp_ref,
):
    """Return the thermal expansion of the shaft and bearing shell.

    Non-zero only for ``deform_type`` 3 or 4.

    Parameters
    ----------
    deform_type : int
        Deformation model key.
    bearing_type : int
        Bearing type key (1 = fixed geometry).
    journal_radius : float
        Journal radius.
    pad_thickness : float
        Pad/shell thickness.
    journal_expand, shell_expand : float
        Journal and shell thermal expansion coefficients.
    temp_j, temp_sump, temp_ref : float
        Journal, sump and reference temperatures.

    Returns
    -------
    tuple of float
        ``(deform_journal, deform_shell)``.
    """
    if deform_type in (
        "pad_mechanical_thermal_shaft_shell_thermal",
        "pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical",
    ):
        deform_journal = journal_radius * journal_expand * (temp_j - temp_ref)
        if bearing_type == "fixed_geometry":
            deform_shell = journal_radius * shell_expand * (temp_j - temp_ref)
        elif bearing_type in TILTING_PAD_TYPES:
            deform_shell = (
                (journal_radius + pad_thickness) * shell_expand * (temp_sump - temp_ref)
            )
        else:
            deform_shell = 0.0
        return deform_journal, deform_shell
    return 0.0, 0.0


def shell_crush(shell_id, shell_od, crush):
    """Return the shell inner-surface displacement from the shrink-fit crush.

    Parameters
    ----------
    shell_id, shell_od : float
        Shell inner and outer diameters.
    crush : float
        Radial interference fit.

    Returns
    -------
    float
        Inner-surface displacement.
    """
    if shell_od == 0.0:
        # No shell defined; without an interference fit there is nothing to
        # crush. (A nonzero crush requires real shell diameters -- the
        # wrapper validates that.)
        return 0.0
    return (shell_id / shell_od) * abs(crush)


def snapshot_film_deform(dh_n):
    """Return a copy of the film-thickness deformation field.

    Parameters
    ----------
    dh_n : ndarray
        Current deformation field.

    Returns
    -------
    ndarray
        ``dh_n_old``, a copy.
    """
    return np.array(dh_n, copy=True)


def blend_film_deform(
    total_pads,
    total_e_z_film,
    total_n_reynolds,
    n_index_reynolds,
    deform_r_surface,
    deform_shell,
    deform_journal,
    deform_crush,
    dh_n,
):
    """Update the film-thickness deformation from surface/shell/journal terms.

    ``deform_r_surface`` is indexed by the circumferential station ``k``
    derived from the Reynolds node id.

    Parameters
    ----------
    total_pads, total_e_z_film : int
        Pad count and axial element count.
    total_n_reynolds, n_index_reynolds : int, array_like
        Reynolds node count and ids.
    deform_r_surface : ndarray
        Pad surface radial deformation per circumferential station.
    deform_shell, deform_journal, deform_crush : float
        Shell/journal/crush radial deformations.
    dh_n : ndarray
        Film-thickness deformation field (updated in place).

    Returns
    -------
    ndarray
        Updated ``dh_n``.
    """
    for pad in range(total_pads):
        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            k = node // (total_e_z_film + 1)
            dh_n[pad, node] = (
                -deform_r_surface[pad, k] + deform_shell - deform_journal - deform_crush
            )
    return dh_n


def film_deform_residual(
    total_pads, total_n_reynolds, n_index_reynolds, dh_n, dh_n_old
):
    """Return the largest per-pad RMS change in film-thickness deformation.

    Parameters
    ----------
    total_pads : int
        Pad count.
    total_n_reynolds, n_index_reynolds : int, array_like
        Reynolds node count and ids.
    dh_n, dh_n_old : ndarray
        Current and previous deformation fields.

    Returns
    -------
    float
        Maximum RMS deformation change over the pads.
    """
    max_h_error = 0.0
    for pad in range(total_pads):
        total = 0.0
        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            total += (dh_n[pad, node] - dh_n_old[pad, node]) ** 2
        rms = np.sqrt(total / total_n_reynolds)
        max_h_error = max(max_h_error, rms)
    return max_h_error


def relax_film_deform(
    total_pads, total_n_reynolds, n_index_reynolds, relax_d, dh_n, dh_n_old
):
    """Relax the film-thickness deformation toward the previous iterate.

    Parameters
    ----------
    total_pads : int
        Pad count.
    total_n_reynolds, n_index_reynolds : int, array_like
        Reynolds node count and ids.
    relax_d : float
        Deformation relaxation factor.
    dh_n, dh_n_old : ndarray
        Current and previous deformation fields.

    Returns
    -------
    ndarray
        Updated ``dh_n``.
    """
    for pad in range(total_pads):
        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            dh_n[pad, node] = (
                relax_d * dh_n[pad, node] + (1.0 - relax_d) * dh_n_old[pad, node]
            )
    return dh_n


def relax_pivot_deform(total_pads, deform_pivot, deform_pivot_old, relax_pivot):
    """Relax the pivot deformations toward the previous iterate.

    Parameters
    ----------
    total_pads : int
        Pad count.
    deform_pivot, deform_pivot_old : ndarray
        Current and previous pivot deformations (0-based per-pad).
    relax_pivot : float
        Pivot relaxation factor.

    Returns
    -------
    ndarray
        Updated ``deform_pivot``.
    """
    for pad in range(total_pads):
        deform_pivot[pad] = (
            relax_pivot * deform_pivot[pad]
            + (1.0 - relax_pivot) * deform_pivot_old[pad]
        )
    return deform_pivot


def pivot_deform_residual(total_pads, deform_pivot_old, deform_pivot):
    """Return the RMS change in pivot deformation across the pads.

    Parameters
    ----------
    total_pads : int
        Pad count.
    deform_pivot_old, deform_pivot : ndarray
        Previous and current pivot deformations (0-based per-pad).

    Returns
    -------
    float
        RMS pivot-deformation change.
    """
    total = 0.0
    for pad in range(total_pads):
        total += (deform_pivot[pad] - deform_pivot_old[pad]) ** 2
    return np.sqrt(total / total_pads)


def initialization(
    total_pads,
    mesh,
    energy_mesh,
    temp_j_type,
    ta_type,
    pads,
    xj,
    yj,
    operating,
    t_ambient,
    temp_j,
    lube,
    turbulence,
):
    """Initialize the iteration state for the first speed/load case.

    Seeds temperatures/viscosities at the supply condition, computes the
    initial film thickness ``h_n`` at the starting journal position, the
    radially-averaged viscosity, the local Reynolds numbers, the flow-regime
    flags and turbulence scaling, and the Couette shear-stress estimate. All
    node/element ids and ``pad`` axes are 0-based, matching
    :mod:`ross.bearings.fluid_film.mesh`.

    Parameters
    ----------
    total_pads : int
        Pad count.
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
    energy_mesh : EnergyMesh
        Film+pad cross-section (``x``-``y``) mesh.
    temp_j_type : str
        Journal-temperature model, one of
        :data:`~ross.bearings.fluid_film.constants.TEMP_J_TYPES`.
    ta_type : int
        Ambient-temperature source: 0 takes the supply temperature, 2 the
        sump temperature, anything else keeps the supplied ``t_ambient``.
    pads : PadGeometry
        Per-pad geometry.
    xj, yj : float
        Initial journal position, m.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    t_ambient : float
        Ambient temperature, K; ignored when ``ta_type`` selects a source.
    temp_j : float
        Journal temperature, K; ignored when ``temp_j_type`` seeds it from
        the supply temperature.
    lube : Lubricant
        Lubricant properties.
    turbulence : Turbulence
        Turbulence-model constants.

    Returns
    -------
    dict
        The initialized solver state (snake_case keys): ``temp_sump``,
        ``t_ambient``, ``temp_j``, ``vis_supply``, ``deform_shell``,
        ``deform_journal``, ``deform_crush``, ``deform_pivot``, ``film_onset``,
        ``q_available``, ``temp_inlet``, ``temp_full``, ``temp_adiab``,
        ``temp_3d``, ``vis_n_3d``, ``vis_n_average``, ``h_n``, ``dh_n``,
        ``flow_regime_track``, ``flow_regime_dam``, ``scale_turb_track``,
        ``scale_turb_dam``, ``shear_stress``.
    """
    temp_sump = operating.temp_supply
    if ta_type == 0:
        t_ambient = operating.temp_supply
    elif ta_type == 2:
        t_ambient = temp_sump
    if temp_j_type in ("averaged_film_temperature", "no_heat_flux_into_journal"):
        temp_j = operating.temp_supply

    deform_shell = 0.0
    deform_journal = 0.0
    deform_crush = 0.0

    film_onset = np.zeros(total_pads, dtype=int)
    q_available = np.zeros(total_pads)
    temp_inlet = np.zeros(total_pads)
    deform_pivot = np.zeros(total_pads)
    temp_full = np.zeros((total_pads, energy_mesh.dim_xy))
    temp_adiab = np.zeros((total_pads, mesh.dim_xz))
    temp_3d = np.zeros((total_pads, mesh.dim_3d))
    vis_n_3d = np.zeros((total_pads, mesh.dim_3d))
    vis_n_average = np.zeros((total_pads, mesh.dim_xz))
    h_n = np.zeros((total_pads, mesh.dim_xz))
    dh_n = np.zeros((total_pads, mesh.dim_xz))
    flow_regime_track = np.zeros(total_pads, dtype=int)
    flow_regime_dam = np.zeros(total_pads, dtype=int)
    scale_turb_track = np.zeros(total_pads)
    scale_turb_dam = np.zeros(total_pads)
    shear_stress = np.zeros((total_pads, mesh.dim_3d))

    vis_supply = lube.viscosity_at(operating.temp_supply)
    re_n = np.zeros(mesh.dim_xz)
    t_vec = np.zeros(mesh.dim_yf)
    f_vec = np.zeros(mesh.dim_yf)

    for pad in range(total_pads):
        for i in range(energy_mesh.total_nodes):
            temp_full[pad, energy_mesh.n_index[i]] = operating.temp_supply
        for i in range(mesh.total_nodes):
            temp_adiab[pad, mesh.n_index[i]] = operating.temp_supply
        temp_inlet[pad] = operating.temp_supply
        film_onset[pad] = 0
        deform_pivot[pad] = 0.0
        q_available[pad] = 1.0e8

        re_max_track = 0.0
        re_max_dam = 0.0
        limit1 = mesh.total_e_y_trackbl[pad] + mesh.total_e_y_trackcore[pad] + 1
        limit2 = mesh.total_e_y_film + 1

        for i in range(mesh.total_nodes):
            node = mesh.n_index[i]
            dh_n[pad, node] = 0.0
            for j in range(1, limit2 + 1):
                temp_3d[pad, mesh.match_nodes_xz[node, j - 1]] = operating.temp_supply
                vis_n_3d[pad, mesh.match_nodes_xz[node, j - 1]] = vis_supply

            x = mesh.x[pad, node]
            z = mesh.z[pad, node]
            xrad = mesh.x_rad[pad, node]
            base_h = (
                pads.cp[pad]
                - xj * np.cos(pads.leading_angle_rad[pad] + xrad)
                - yj * np.sin(pads.leading_angle_rad[pad] + xrad)
                - pads.preload[pad]
                * pads.cp[pad]
                * np.cos(xrad - pads.x_pivot_rad[pad])
            )

            in_pocket = (
                z > pads.axial_length_dam[pad]
                and z < pads.axial_length_dam[pad] + pads.axial_length_track[pad]
                and x < pads.length_track[pad]
            )
            in_dam = (
                x > pads.length_track[pad]
                or z < pads.axial_length_dam[pad]
                or z > pads.axial_length_dam[pad] + pads.axial_length_track[pad]
            )
            if in_pocket:
                pocket_like = True
            elif in_dam:
                pocket_like = False
            else:
                pad_edge = (
                    abs(x - pads.pad_length[pad]) < 1.0e-6
                    and z > pads.axial_length_dam[pad]
                    and z < pads.axial_length_track[pad] + pads.axial_length_dam[pad]
                ) or (
                    (abs(z) < 1.0e-6 or abs(z - pads.axial_length[pad]) < 1.0e-6)
                    and x < pads.length_track[pad]
                )
                pocket_like = pad_edge

            if pocket_like:
                h_n[pad, node] = base_h + pads.depth_track[pad]
                j_start = 1
                track_bucket = True
            else:
                h_n[pad, node] = base_h
                j_start = limit1
                track_bucket = False

            for j in range(j_start, limit2 + 1):
                t_vec[j - 1] = (
                    mesh.y_3d[pad, mesh.match_nodes_xz[node, j - 1]]
                    - pads.pad_thickness
                )
                f_vec[j - 1] = vis_n_3d[pad, mesh.match_nodes_xz[node, j - 1]]
            inte_trap = trapezoid(t_vec, f_vec, j_start - 1, limit2)

            vis_n_average[pad, node] = inte_trap / h_n[pad, node]
            re_n[node] = (
                lube.density
                * operating.speed_surface
                * h_n[pad, node]
                / vis_n_average[pad, node]
            )
            if track_bucket:
                re_max_track = max(re_max_track, re_n[node])
            else:
                re_max_dam = max(re_max_dam, re_n[node])

        flow_regime_track[pad], scale_turb_track[pad] = _flow_regime(
            re_max_track,
            turbulence,
        )
        flow_regime_dam[pad], scale_turb_dam[pad] = _flow_regime(
            re_max_dam,
            turbulence,
        )

        for i in range(mesh.total_nodes):
            node = mesh.n_index[i]
            shear_stress1 = 1.0 + 0.012 * re_n[node] ** 0.94
            for j in range(1, mesh.total_e_y_film + 2):
                shear_stress[pad, mesh.match_nodes_xz[node, j - 1]] = (
                    vis_n_average[pad, node]
                    * operating.speed_surface
                    * shear_stress1
                    / h_n[pad, node]
                )

    return {
        "temp_sump": temp_sump,
        "t_ambient": t_ambient,
        "temp_j": temp_j,
        "vis_supply": vis_supply,
        "deform_shell": deform_shell,
        "deform_journal": deform_journal,
        "deform_crush": deform_crush,
        "deform_pivot": deform_pivot,
        "film_onset": film_onset,
        "q_available": q_available,
        "temp_inlet": temp_inlet,
        "temp_full": temp_full,
        "temp_adiab": temp_adiab,
        "temp_3d": temp_3d,
        "vis_n_3d": vis_n_3d,
        "vis_n_average": vis_n_average,
        "h_n": h_n,
        "dh_n": dh_n,
        "flow_regime_track": flow_regime_track,
        "flow_regime_dam": flow_regime_dam,
        "scale_turb_track": scale_turb_track,
        "scale_turb_dam": scale_turb_dam,
        "shear_stress": shear_stress,
    }


def _flow_regime(
    re_max,
    turbulence,
):
    """Return ``(flow_regime, turbulence_scale)`` from the max Reynolds number.

    Laminar (0) below ``re_lower``, transitional (1) with a power-law blend up
    to ``re_upper``, turbulent (2) above.
    """
    if re_max < turbulence.re_lower:
        return 0, 0.0
    if turbulence.re_lower < re_max < turbulence.re_upper:
        scale = (
            1.0
            - (
                (turbulence.re_upper - re_max)
                / (turbulence.re_upper - turbulence.re_lower)
            )
            ** turbulence.scale_factor_exponent
        )
        return 1, scale
    return 2, 1.0


def _adapt(fn, ns, **extra):
    """Call ``fn`` drawing its arguments by name from ``ns`` (overridden by ``extra``).

    The solver modules share one vocabulary of parameter names, so binding
    each parameter to the matching solver-state key avoids hand-listing the
    hundreds of arguments at every call site. Parameters absent from both
    ``extra`` and ``ns`` fall back to ``fn``'s own default.
    """
    kwargs = {}
    for name in inspect.signature(fn).parameters:
        if name in extra:
            kwargs[name] = extra[name]
        elif name in ns:
            kwargs[name] = ns[name]
    return fn(**kwargs)


def run_case(**kw):
    """Run one bearing analysis. SI in, SI out.

    The entry point to the solver pipeline. Every keyword is snake_case:
    unit-bearing quantities in SI (``journal_diameter`` m,
    ``radial_clearance`` m, ``pivot_angle`` rad, ``frequency`` rad/s,
    ``ambient_pressure_1`` Pa, ``oil_supply_temperature`` K, ...) alongside
    the dimensionless and structural inputs (``total_e_x_film``,
    ``bearing_type``, ``operating_type``, ...).

    Returns a dict of 89 named outputs (equilibrium position, reduced
    stiffness/damping coefficients, power loss, flow rates, film thicknesses,
    temperature fields, per-pad summaries and center-plane profiles), with
    every unit-bearing value in SI: stiffness in N/m, damping in N*s/m,
    power in W, temperatures in K, pressures in Pa, lengths in m, angles in
    rad. Each value is a list with one entry per speed case.

    With ``field_outputs=True`` the dict gains a ``"fields"`` key: one dict
    per speed case of full-field arrays on the film mesh, shaped
    ``(total_pads, total_e_x_film + 1, total_e_z_film + 1)`` -- see
    :func:`_assemble_field_outputs`. The named-output surface above is
    unchanged.

    The internal state dict ``g`` carries SI values across every solver
    module.
    """
    from ross.bearings.fluid_film import coefficients as _coeff
    from ross.bearings.fluid_film import deform as _deform
    from ross.bearings.fluid_film import hydrodynamics as _hyd
    from ross.bearings.fluid_film import pressure as _press

    total_pads = len(np.atleast_1d(kw["pivot_angle"]))
    # ``frequency`` is in rad/s on the SI boundary; case_parameters / the
    # output writers still operate on RPM downstream, so convert once here.
    frequency_rad_s = np.atleast_1d(np.asarray(kw["frequency"], dtype=float))
    speeds_rpm = frequency_rad_s * 60.0 / (2.0 * PI)
    total_cases = len(speeds_rpm)

    def per_case(name):
        return np.broadcast_to(
            np.atleast_1d(np.asarray(kw[name], dtype=float)), (total_cases,)
        ).copy()

    te_x = int(kw["total_e_x_film"])
    te_yf = int(kw["total_e_y_film"])
    te_z = int(kw["total_e_z_film"])
    te_yp = int(kw["total_e_y_pad"])

    dim_x = te_x + 1
    dim_yf = te_yf + 1
    dim_z = te_z + 1
    dim_yp = te_yp + 1
    dim_xz = dim_x * dim_z
    dim_xy = dim_x * (te_yf + te_yp + 1)
    dim_yz = dim_yf * dim_z
    dim_xy2 = 2 * dim_x * dim_yp
    dim_3d = dim_x * dim_yf * dim_z

    conv = convert(
        kw["journal_diameter"],
        kw["pivot_angle"],
        kw["offset"],
        kw["pad_arc"],
        kw["track_arc"],
        kw["pad_axial_length"],
        kw["track_axial_length"],
        kw["taper_arc_le"],
        kw["taper_arc_te"],
        kw["pocket_arc"],
        kw["radial_clearance"],
        kw["preload"],
        kw["xj"],
        kw["yj"],
        per_case("oil_flow_v"),
        kw["weight_e"],
        kw["weight_h"],
    )

    def p1(values):
        # g is 0-based natural: per-pad arrays carry pad ``i`` at slot ``i``.
        return np.asarray(values, dtype=float)

    zeros_pad = np.zeros(total_pads)
    g = {
        "total_pads": total_pads,
        "total_cases": total_cases,
        "dim_x": dim_x,
        "dim_yf": dim_yf,
        "dim_z": dim_z,
        "dim_yp": dim_yp,
        "dim_xz": dim_xz,
        "dim_xy": dim_xy,
        "dim_yz": dim_yz,
        "dim_xy2": dim_xy2,
        "dim_3d": dim_3d,
        "total_e_x_film": te_x,
        "total_e_y_film": te_yf,
        "total_e_z_film": te_z,
        "total_e_y_pad": te_yp,
        "bearing_type": kw["bearing_type"],
        "operating_type": kw["operating_type"],
        "thermal_type": kw["thermal_type"],
        "temp_j_type": kw["temp_j_type"],
        "deform_type": kw["deform_type"],
        "equilibrium_type": kw["equilibrium_type"],
        "sump_type": kw["sump_type"],
        "ta_type": int(kw["ta_type"]),
        "journal_diameter": float(kw["journal_diameter"]),
        "cb": float(kw["radial_clearance"]),
        "pad_thickness": float(kw["pad_thickness"]),
        "preload": p1(kw["preload"]),
        "offset": p1(kw["offset"]),
        "axial_length": p1(kw["pad_axial_length"]),
        "axial_length_track": p1(kw["track_axial_length"]),
        "depth_track": p1(kw["track_depth"]),
        "dh_ramp_le": p1(kw["taper_depth_le"]),
        "dh_ramp_te": p1(kw["taper_depth_te"]),
        "convec_back": p1(kw["pad_convection"]),
        "k_rotate": p1(kw["k_rotate"]),
        "axial_length_pocket": float(kw["pocket_axial_length"]),
        "viscosity1": float(kw["viscosity1"]),
        "viscosity2": float(kw["viscosity2"]),
        "temp1": float(kw["temp1"]),
        "temp2": float(kw["temp2"]),
        "lube_density": float(kw["lube_density"]),
        "lube_cp": float(kw["lube_cp"]),
        "lube_conduct": float(kw["lube_conduct"]),
        "young": float(kw["pad_young"]),
        "poisson": float(kw["pad_poisson"]),
        "pad_conduct": float(kw["pad_conductivity"]),
        "pad_expand": float(kw["pad_expansion"]),
        "pad_density": float(kw["pad_density"]),
        "journal_expand": float(kw["journal_expansion"]),
        "shell_expand": float(kw["shell_expansion"]),
        "pivot_type": kw["pivot_type"],
        "house_diameter": float(kw["house_diameter"]),
        "pivot_diameter": float(kw["pivot_diameter"]),
        "pivot_stiff": float(kw["pivot_stiffness"]),
        "ambient_press1": float(kw["ambient_pressure_1"]),
        "ambient_press2": float(kw["ambient_pressure_2"]),
        "press_cavitate": float(kw["cavitation_pressure"]),
        "press_supply": float(kw["oil_supply_pressure"]),
        "temp_supply": float(kw["oil_supply_temperature"]),
        "temp_ref": float(kw["reference_temperature"]),
        "convec_edges": float(kw["edges_convection"]),
        "crush": float(kw["crush_fit"]),
        "shell_id": float(kw["shell_id"]),
        "shell_od": float(kw["shell_od"]),
        "starve_number": int(kw["starve_number"]),
        "hotoil_lamda": float(kw["hot_oil_lambda"]),
        "relaxp": float(kw["relax_p"]),
        "relaxt": float(kw["relax_t"]),
        "relaxd": float(kw["relax_d"]),
        "relaxpivot": float(kw["relax_pivot"]),
        "re_lower": float(kw["re_lower"]),
        "re_upper": float(kw["re_upper"]),
        "reichardt_delta": float(kw["reichardt_delta"]),
        "reichardt_kappa": float(kw["reichardt_kappa"]),
        "turb_scal_fac_exp": float(kw["turb_scal_fac_exp"]),
        # converted geometry (0-based natural, per-pad)
        "journal_radius": conv["journal_radius"],
        "length_pocket": conv["length_pocket"],
        "xj": conv["xj"],
        "yj": conv["yj"],
        "arc_length_rad": p1(conv["arc_length_rad"]),
        "leading_angle_rad": p1(conv["leading_angle_rad"]),
        "length_ramp_le": p1(conv["length_ramp_le"]),
        "length_ramp_te": p1(conv["length_ramp_te"]),
        "pad_length": p1(conv["pad_length"]),
        "cp": p1(conv["cp"]),
        "x_pivot_rad": p1(conv["x_pivot_rad"]),
        "x_pivot": p1(conv["x_pivot"]),
        "length_track_rad": p1(conv["length_track_rad"]),
        "length_track": p1(conv["length_track"]),
        "length_dam": p1(conv["length_dam"]),
        "axial_length_dam": p1(conv["axial_length_dam"]),
        "weight_e": conv["weight_e"],
        "weight_h": conv["weight_h"],
        # initial region element splits (mesh_reynolds recomputes these)
        "total_e_x_track": zeros_pad.astype(int),
        "total_e_z_track": zeros_pad.astype(int),
        "total_e_x_dam": zeros_pad.astype(int),
        "total_e_z_dam": zeros_pad.astype(int),
        # temperature probes (1-based pad number, % of pad arc from the
        # leading edge, radial distance from the pad surface in m)
        "probe_pad_number": np.atleast_1d(
            np.asarray(kw.get("probe_pad_number", []), dtype=int)
        ),
        "probe_theta": np.atleast_1d(
            np.asarray(kw.get("probe_theta", []), dtype=float)
        ),
        "probe_r_location": np.atleast_1d(
            np.asarray(kw.get("r_location", []), dtype=float)
        ),
    }

    g.update(_adapt(_mesh_reynolds(), g))
    g.update(_adapt(_mesh_energy(), g))
    g.update(_adapt(_mesh_3d(), g))

    # The x-z film mesh is complete and never changes again; freeze it into one
    # object so the solver routines take it as a single argument.
    g["mesh"] = ReynoldsMesh.from_state(g, _REYNOLDS_MESH_KEYS)
    g["pads"] = PadGeometry.from_state(g, _PAD_GEOMETRY_KEYS)
    g["energy_mesh"] = EnergyMesh.from_state(g, _ENERGY_MESH_KEYS)
    g["lube"] = Lubricant.from_state(g, _LUBRICANT_KEYS)

    g.update(
        case_parameters(
            1,
            g["journal_diameter"],
            speeds_rpm,
            per_case("weight"),
            per_case("fxs_load"),
            per_case("fys_load"),
            conv["qs_supply"],
            per_case("excit_ratios"),
        )
    )

    # Raw applied loads for this case, kept for the output summaries: the
    # solver itself only ever sees the net ``fx_ext``/``fy_ext``.
    g["weight_case"] = float(per_case("weight")[0])
    g["fxs_case"] = float(per_case("fxs_load")[0])
    g["fys_case"] = float(per_case("fys_load")[0])
    # Total and specific (projected-area) load for the case.
    g["f_total"] = float(np.hypot(g["fx_ext"], g["fy_ext"]))
    g["f_specific"] = g["f_total"] / (g["journal_diameter"] * g["axial_length"][0])

    # Everything the case needs that does not change during its iterations.
    g["operating"] = OperatingPoint.from_state(g, _OPERATING_POINT_KEYS)
    g["turbulence"] = Turbulence(
        re_lower=g["re_lower"],
        re_upper=g["re_upper"],
        reichardt_delta=g["reichardt_delta"],
        reichardt_kappa=g["reichardt_kappa"],
        scale_factor_exponent=g["turb_scal_fac_exp"],
    )

    g.update(
        _adapt(
            initialization,
            g,
            t_ambient=float(kw.get("t_ambient", g["temp_supply"])),
            temp_j=float(kw["journal_temperature"]),
        )
    )

    # Fields read by the hydrodynamic solver before anything writes them
    # (0-based natural shapes).
    for name, shape in (
        ("nodal_pressure", (total_pads, dim_xz)),
        ("dpdx_n", (total_pads, dim_xz)),
        ("dpdz_n", (total_pads, dim_xz)),
        ("dhdx_n", (total_pads, dim_xz)),
        ("pressback_n", (total_pads, dim_xz)),
        ("dudy_n", (total_pads, dim_3d)),
        ("dwdy_n", (total_pads, dim_3d)),
        ("vis_eddy_3d", (total_pads, dim_3d)),
        ("vis_effect_3d", (total_pads, dim_3d)),
        ("velocity_x_n", (total_pads, dim_3d)),
        ("velocity_y_n", (total_pads, dim_3d)),
        ("velocity_z_n", (total_pads, dim_3d)),
        ("scale_dissip", (total_pads, dim_xz)),
        ("q_x", (total_pads, dim_x)),
    ):
        g.setdefault(name, np.zeros(shape))
    g.setdefault("tilt_angle", np.zeros(total_pads))
    g.setdefault("h_min", np.zeros(total_pads))
    g.setdefault("x_hmin", np.zeros(total_pads))
    for _counter in ("elasto_index", "mixtemp_index", "jtemp_index", "vis_index"):
        g.setdefault(_counter, 1)
    g.setdefault("re_max", np.zeros(total_pads))
    g.setdefault("unloaded", np.zeros(total_pads, dtype=int))
    g.setdefault("q_carryover", np.zeros(total_pads))
    g.setdefault("q_in", np.zeros(total_pads))
    g.setdefault("q_out", np.zeros(total_pads))

    def _zero_pressure_system_wrap(dim_xz_, total_n, total_col):
        # 0-based band storage: only ``total_col = 2*bw - 1`` columns are used.
        return (
            np.zeros((dim_xz_, total_col), dtype=float),
            np.zeros(dim_xz_, dtype=float),
        )

    def _include_press_wrap(gm, gc, bw, total_bc, bc_idx, pres, total_n):
        gm = np.ascontiguousarray(gm, dtype=np.float64)
        gc = np.ascontiguousarray(gc, dtype=np.float64)
        bc_idx = np.ascontiguousarray(bc_idx, dtype=np.int64)
        pres = np.ascontiguousarray(pres, dtype=np.float64)
        # ``coefficients`` passes ``bc_idx`` as 0-based node values (length
        # ``dim_xz``, first ``total_bc`` meaningful), indexed directly by the
        # 0-based ``include_press_jit`` kernel.
        bc_idx = bc_idx[:total_bc].copy()
        pres = pres[:total_bc].copy()
        return include_press_jit(gm, gc, bw, total_bc, bc_idx, pres, total_n)

    def _integrate_xz_coeff(pad_index, mesh, f):
        """Surface integral of the nodal field ``f`` over the Reynolds mesh."""
        return integrate_xz_jit(
            int(pad_index),
            int(mesh.total_elements),
            np.ascontiguousarray(mesh.e_index, dtype=np.int64),
            np.ascontiguousarray(mesh.node_i, dtype=np.int64),
            np.ascontiguousarray(mesh.node_j, dtype=np.int64),
            np.ascontiguousarray(mesh.node_k, dtype=np.int64),
            np.ascontiguousarray(mesh.node_l, dtype=np.int64),
            np.ascontiguousarray(mesh.e_length, dtype=np.float64),
            np.ascontiguousarray(mesh.e_width, dtype=np.float64),
            np.ascontiguousarray(f, dtype=np.float64),
        )

    helpers = SimpleNamespace(
        element_press=_press.element_press,
        include_press=_include_press_wrap,
        zero_pressure_system=_zero_pressure_system_wrap,
        integrate_xz=_integrate_xz_coeff,
    )

    def fixed_brg_fn(state, equilpost_index, relaxp, press=None):
        # ``hydrodynamics`` is now 0-based, so the ``state`` reaching this
        # wrapper (from inside hydro) is already 0-based; pass the 0-based
        # integrators/helpers to ``fixed_brg``.
        return _adapt(
            _hyd.fixed_brg,
            state,
            equilpost_index=equilpost_index,
            relaxp=relaxp,
            press=press,
            integrate_xz=_integrate_xz_coeff,
            trapezoid=trapezoid,
        )

    def jacobian_fn(state):
        # ``state`` is 0-based natural (g is 0-based end-to-end).
        return _adapt(_coeff.jacobian, state, helpers=helpers)

    def _call_hyd(state0):
        # 0-based-native hydro entry. ``state0`` is the live 0-based state;
        # ``hydrodynamics`` mutates the mesh/film fields it produces in place
        # (via ``s.update(fb)`` onto the same array objects) and returns the
        # working state, so the caller observes the updates by aliasing.
        return _hyd.hydrodynamics(
            state0,
            state0["fx_ext"],
            state0["fy_ext"],
            state0["equilibrium_type"],
            state0["starve_number"],
            sump_index=state0.get("sump_index", 1),
            elasto_index=state0.get("elasto_index", 1),
            mixtemp_index=state0.get("mixtemp_index", 1),
            jtemp_index=state0.get("jtemp_index", 1),
            vis_index=state0.get("vis_index", 1),
            relaxp=state0["relaxp"],
            fixed_brg_fn=fixed_brg_fn,
            jacobian_fn=jacobian_fn,
            integrate_xz=_integrate_xz_coeff,
            press=_press.press,
        )

    def _hyd_kwargs_adapter(**kw):
        # thd calls hydrodynamics(**state0) with the full 0-based state splatted
        # as kwargs; re-bundle into the positional state dict.
        return _call_hyd(kw)

    if g["thermal_type"] is None and g["deform_type"] is None:
        g.update(_call_hyd(g))
    else:
        from ross.bearings.fluid_film import thd as _thd
        from ross.bearings.fluid_film import thermal as _thermal
        from ross.bearings.fluid_film.constants import (
            DEFORM_ERROR,
            MAX_ITERATION,
            SUMP_TEMP_ERROR,
        )

        def _filter_kw(fn, kw):
            params = inspect.signature(fn).parameters
            return {k: v for k, v in kw.items() if k in params}

        def _thermal_adiab_wrap(**kw):
            return _thermal.thermal_adiabatic(
                **_filter_kw(_thermal.thermal_adiabatic, kw)
            )

        def _thermal_full_wrap(**kw):
            return _thermal.thermal_full(**_filter_kw(_thermal.thermal_full, kw))

        g.setdefault("relax_t_max", g["relaxt"])
        g.setdefault("hd_converged", 1)
        if g["deform_type"] is not None:
            g.update(_adapt(_mesh_deform(), g))
            for name, shape in (
                ("deform_r_surface", (total_pads, dim_x)),
                ("force_pivot", (total_pads,)),
                ("pad_temp", (total_pads, dim_xy2)),
                ("nodal_force", (total_pads, dim_xy2)),
            ):
                g.setdefault(name, np.zeros(shape))
            g.setdefault("deform_pivot_old", np.zeros(total_pads))
            # Pivot-flexibility buffer: ``deform_pivots`` writes per-pad
            # k_pivot here; ``dynamic_reduction_pivot`` (deformation modes 4
            # and 5) then reads it (0-based natural, slots 0..total_pads-1).
            g.setdefault("k_pivot", np.zeros(total_pads))

        for sump_index in range(1, MAX_ITERATION + 1):
            if g["sump_type"] == "supply_temperature":
                g["temp_sump"] = g["temp_supply"]
            g["sump_index"] = sump_index

            for elasto_pivot_index in range(1, MAX_ITERATION + 1):
                for elasto_index in range(1, MAX_ITERATION + 1):
                    g["elasto_index"] = elasto_index
                    g["mixtemp_index"] = 1
                    g["jtemp_index"] = 1
                    g["vis_index"] = 1
                    # ``g`` is 0-based natural end-to-end, so pass it directly
                    # as every ``*_inputs`` argument; ``thermohydrodynamics`` and
                    # the hydro/thermal wraps are all 0-based-native and share
                    # the same array objects (in-place aliasing preserved).
                    g0 = g
                    thd_out = _thd.thermohydrodynamics(
                        total_pads=g0["total_pads"],
                        operating=g0["operating"],
                        thermal_type=g0["thermal_type"],
                        temp_j_type=g0["temp_j_type"],
                        relax_t=g0["relaxt"],
                        hd_inputs=g0,
                        hydrodynamics=_hyd_kwargs_adapter,
                        thermal_adiabatic=_thermal_adiab_wrap,
                        thermal_full=_thermal_full_wrap,
                        integrate_xz=_integrate_xz_coeff,
                        trapezoid=trapezoid,
                        thermal_inputs=g0,
                        temp_max_inputs=g0,
                        temp_journal_inputs=g0,
                        t_outlet_inputs=g0,
                        mixing_inputs=g0,
                    )
                    g.update(thd_out)

                    pl_value, hp_value = _adapt(
                        power_loss, g, speed_surface=g["speed_surface"]
                    )
                    g["powerloss"] = pl_value
                    g["hp"] = hp_value
                    g["dh_n_old"] = snapshot_film_deform(
                        g["dh_n"],
                    )

                    if g["deform_type"] is not None:
                        elasto_out = _adapt(
                            _deform.elasto,
                            g,
                            helpers=helpers,
                            integrate_xz=_integrate_xz_coeff,
                            trapezoid=trapezoid,
                        )
                        deform_r, force_p, _, _ = elasto_out
                        # ``elasto`` returns fully 0-based natural arrays.
                        g["deform_r_surface"] = deform_r
                        g["force_pivot"] = force_p
                        dj, ds = deform_shaftshell(
                            g["deform_type"],
                            g["bearing_type"],
                            g["journal_radius"],
                            g["pad_thickness"],
                            g["journal_expand"],
                            g["shell_expand"],
                            g["temp_j"],
                            g["temp_sump"],
                            g["temp_ref"],
                        )
                        g["deform_journal"] = dj
                        g["deform_shell"] = ds
                        g["deform_crush"] = shell_crush(
                            g["shell_id"], g["shell_od"], g["crush"]
                        )
                        g["dh_n"] = blend_film_deform(
                            total_pads,
                            g["total_e_z_film"],
                            g["total_n_reynolds"],
                            g["n_index_reynolds"],
                            g["deform_r_surface"],
                            ds,
                            dj,
                            g["deform_crush"],
                            g["dh_n"],
                        )

                    max_h_err = film_deform_residual(
                        total_pads,
                        g["total_n_reynolds"],
                        g["n_index_reynolds"],
                        g["dh_n"],
                        g["dh_n_old"],
                    )
                    if max_h_err / g["cb"] < DEFORM_ERROR:
                        break
                    g["dh_n"] = relax_film_deform(
                        total_pads,
                        g["total_n_reynolds"],
                        g["n_index_reynolds"],
                        g["relaxd"],
                        g["dh_n"],
                        g["dh_n_old"],
                    )
                if g["deform_type"] in PIVOT_FLEX_DEFORM_TYPES:
                    # After the inner elasto loop, snapshot the current
                    # pivot deformation, recompute it
                    # (and k_pivot) from the freshly-converged force_pivot,
                    # break the outer loop on RMS convergence, otherwise
                    # relax and iterate.
                    g["deform_pivot_old"][:] = g["deform_pivot"]
                    dp, kp = _deform.deform_pivots(
                        total_pads,
                        g["deform_type"],
                        g["pivot_type"],
                        g["poisson"],
                        g["young"],
                        g["pivot_diameter"],
                        g["house_diameter"],
                        g["axial_length"],
                        g["pivot_stiff"],
                        g["force_pivot"],
                        g["deform_pivot"],
                        g["k_pivot"],
                    )
                    g["deform_pivot"][:] = dp
                    g["k_pivot"][:] = kp
                    rms_pivot = pivot_deform_residual(
                        total_pads, g["deform_pivot_old"], g["deform_pivot"]
                    )
                    if rms_pivot / g["cb"] < 0.01:
                        break
                    g["deform_pivot"] = relax_pivot_deform(
                        total_pads,
                        g["deform_pivot"],
                        g["deform_pivot_old"],
                        g["relaxpivot"],
                    )
                else:
                    break
            temp_sump_old = g["temp_sump"]
            if g["operating_type"] == "oil_ring_lubricated":
                g["temp_sump"] = sump_temp(
                    g.get("powerloss", 0.0),
                    g.get("area_sumpconvec", 1.0),
                    g["temp_environment"],
                    g["convec_environment"],
                )
            else:
                g["temp_sump"] = drain_temp(
                    g["q_supply"],
                    g["lube"],
                    g.get("powerloss", 0.0),
                    g["temp_supply"],
                )
            if g["sump_type"] == "supply_temperature":
                break
            # Once the sump temperature settles, stop iterating the outer
            # sump loop. Without this break the solver keeps re-solving the
            # full thermo-elastic problem to MAX_ITERATION, over-converging the
            # carried-over thermal field and landing the journal equilibrium at
            # a marginally different fixed point.
            if abs(g["temp_sump"] - temp_sump_old) < SUMP_TEMP_ERROR:
                break

    power_loss_value, hp = _adapt(power_loss, g, speed_surface=g["speed_surface"])
    g["temp_sump"] = drain_temp(
        g["q_supply"],
        g["lube"],
        power_loss_value,
        g["temp_supply"],
    )

    # Deliberately reuse the Jacobian computed inside the equilibrium loop
    # rather than recomputing one here: the stiffness tuple the dynamic
    # reduction consumes is the one evaluated on the pre-convergence ``h_n``.
    # Recomputing at the post-convergence state shifts the coefficients ~0.15%
    # on the fixed-geometry cases and ~260% on pressure-dam ones. Falls back to
    # a fresh call only when equilibrium converged on the first iteration,
    # without entering the Newton branch.
    field_outputs = bool(kw.get("field_outputs", False))

    stiffness = g.pop("_last_jacobian", None)
    if stiffness is None:
        stiffness = jacobian_fn(g)
    damping_block = _adapt(_coeff.damping, g, helpers=helpers)
    if g["deform_type"] in PIVOT_FLEX_DEFORM_TYPES:
        # Pivot flexibility on: condense the
        # 2*total_pads tilt/pivot block with dynamic_reduction_pivot.
        reduced = _coeff.dynamic_reduction_pivot(
            total_pads,
            g["pads"],
            g["pad_density"],
            np.zeros(total_pads),
            stiffness,
            damping_block,
            g["k_pivot"],
            g["excit_rad"],
            g["k_rotate"],
        )
    else:
        reduced = _coeff.dynamic_reduction(
            total_pads,
            stiffness,
            damping_block,
            g["pads"],
            g["pad_density"],
            g["excit_rad"],
            np.zeros(total_pads),
            g["k_rotate"],
        )
    outputs = _assemble_outputs(g, hp, reduced)
    if field_outputs:
        outputs["fields"] = [_assemble_field_outputs(g)]
    return outputs


def _mesh_reynolds():
    from ross.bearings.fluid_film.mesh import mesh_reynolds

    return mesh_reynolds


def _mesh_energy():
    from ross.bearings.fluid_film.mesh import mesh_energy

    return mesh_energy


def _mesh_3d():
    from ross.bearings.fluid_film.mesh import mesh_3d

    return mesh_3d


def _mesh_deform():
    from ross.bearings.fluid_film.mesh import mesh_deform

    return mesh_deform


def _attitude_rad(xj, yj):
    """Quadrant-resolved attitude angle, in radians."""
    if xj == 0.0 or yj == 0.0:
        return 0.0
    a = np.arctan(abs(xj) / abs(yj))
    if xj > 0.0 and yj > 0.0:
        return PI - a
    if xj < 0.0 and yj > 0.0:
        return -PI + a
    if xj < 0.0 and yj < 0.0:
        return -a
    return a


def _max_nodal_pressure(g):
    """Maximum film pressure over all pads and its 1-based pad number.

    Scans pads in order and keeps the first strict maximum, so ties resolve
    to the lowest pad number. Returns ``(0.0, 0)`` when no node is above
    zero gauge pressure.
    """
    mesh = g["mesh"]
    nodes = mesh.n_index[: mesh.total_nodes]
    sub = g["nodal_pressure"][:, nodes]
    if sub.size == 0:
        return 0.0, 0
    flat = int(np.argmax(sub))
    press_max = float(sub.flat[flat])
    if press_max <= 0.0:
        return 0.0, 0
    return press_max, flat // sub.shape[1] + 1


def _probe_temperatures(g):
    """Temperature at each probe location, by nearest-node lookup.

    Probes are specified as (1-based pad number, circumferential position as
    a percentage of the pad arc from the leading edge, radial distance from
    the pad surface). With the full energy model the probe is resolved on the
    pad cross-section mesh, honoring its through-thickness position; with the
    adiabatic model on the film mesh at the axial midplane. Without a thermal
    model there is no temperature field and every probe reports the "no
    data" sentinel.
    """
    total_probes = g["probe_pad_number"].size
    temp_probe = np.full(total_probes, ZERO_TEMPERATURE_SENTINEL)
    thermal_type = g.get("thermal_type")
    if total_probes == 0 or thermal_type not in ("adiabatic", "full"):
        return temp_probe

    pads = g["pads"]
    for p in range(total_probes):
        pad = int(g["probe_pad_number"][p]) - 1
        probe_x = 0.01 * g["probe_theta"][p] * pads.pad_length[pad]
        if thermal_type == "full":
            em = g["energy_mesh"]
            probe_y = pads.pad_thickness - g["probe_r_location"][p]
            dist = np.hypot(
                em.x[pad, : em.total_nodes] - probe_x,
                em.y[pad, : em.total_nodes] - probe_y,
            )
            temp_probe[p] = g["temp_full"][pad, int(np.argmin(dist))]
        else:
            mesh = g["mesh"]
            probe_z = 0.5 * pads.axial_length[pad]
            dist = np.hypot(
                mesh.x[pad, : mesh.total_nodes] - probe_x,
                mesh.z[pad, : mesh.total_nodes] - probe_z,
            )
            temp_probe[p] = g["temp_adiab"][pad, int(np.argmin(dist))]
    return temp_probe


def _center_plane_profiles(g):
    """Circumferential profiles along each pad's axial center plane.

    Returns ``(theta_c, hc, pc, tc, tb, tm, dhc)``, each shaped
    ``(total_pads, total_e_x_film + 1)``: angular node position, film
    thickness, film pressure, pad surface temperature, pad back temperature,
    film average temperature and film-thickness change from deformation.

    ``tc``/``tb`` come from the pad temperature field of whichever thermal
    model ran. The adiabatic model has no through-thickness resolution, so
    its back temperature is the "no data" sentinel; without a thermal model
    both report the supply temperature.
    """
    mesh = g["mesh"]
    pads = g["pads"]
    total_pads = g["total_pads"]
    dim_x = g["dim_x"]
    te_z = g["total_e_z_film"]
    te_yp = g["total_e_y_pad"]
    te_yf = g["total_e_y_film"]
    thermal_type = g.get("thermal_type")
    t_average = g.get("t_average")

    theta_c = np.zeros((total_pads, dim_x))
    hc = np.zeros((total_pads, dim_x))
    pc = np.zeros((total_pads, dim_x))
    tm = np.zeros((total_pads, dim_x))
    dhc = np.zeros((total_pads, dim_x))
    tc = np.zeros((total_pads, dim_x))
    tb = np.zeros((total_pads, dim_x))

    for pad in range(total_pads):
        n = 0
        for i in range(mesh.total_nodes):
            node = mesh.n_index[i]
            on_center = (
                abs(mesh.z[pad, node] - 0.5 * pads.axial_length[pad]) < _MIDPLANE_TOL
            )
            if not on_center or n >= dim_x:
                continue
            theta_c[pad, n] = mesh.x_rad[pad, node]
            hc[pad, n] = g["h_n"][pad, node]
            pc[pad, n] = g["nodal_pressure"][pad, node]
            dhc[pad, n] = g["dh_n"][pad, node]
            if t_average is not None:
                tm[pad, n] = t_average[pad, node]
            n += 1

        if thermal_type == "adiabatic":
            for i in range(dim_x):
                tc[pad, i] = g["temp_adiab"][pad, te_z // 2 + i * (te_z + 1)]
            tb[pad, :] = ZERO_TEMPERATURE_SENTINEL
        elif thermal_type == "full":
            stride = te_yp + te_yf + 1
            for i in range(dim_x):
                tc[pad, i] = g["temp_full"][pad, te_yp + i * stride]
                tb[pad, i] = g["temp_full"][pad, i * stride]
        else:
            tc[pad, :] = g["temp_supply"]
            tb[pad, :] = g["temp_supply"]

    return theta_c, hc, pc, tc, tb, tm, dhc


def _film_onset_angles(g):
    """Continuous-film onset angle per pad, for starved operating modes.

    A pad whose film starts at the leading edge (flooded) or whose film
    never forms (fully cavitated) reports zero; only a genuinely starved pad
    reports the angle, measured from the pad leading edge.
    """
    total_pads = g["total_pads"]
    film_angle = np.zeros(total_pads)
    if g.get("operating_type") not in _STARVED_OPERATING_TYPES:
        return film_angle
    mesh = g["mesh"]
    te_x = g["total_e_x_film"]
    te_z = g["total_e_z_film"]
    for pad in range(total_pads):
        onset = int(g["film_onset"][pad])
        if onset == 0 or onset == te_x - 2:
            continue
        node = onset * (te_z + 1) + te_z // 2 - 1
        film_angle[pad] = mesh.x_rad[pad, node]
    return film_angle


def _rigid_rotor_stability(g, reduced):
    """Rigid-rotor stability threshold and dimensionless coefficients.

    Nondimensionalizes the reduced coefficients with the total load and
    radial clearance (``K Cb / F``; ``C omega Cb / F``) and evaluates the
    rigid-rotor stability threshold speed and whirl frequency ratio from
    them (Lund's single-mass criterion). Returns ``(threshold, whirl_ratio,
    kbxx, kbyy, cbxx, cbyy)`` with the threshold in rpm; all zero when a
    principal coefficient is below the reporting floor, the load or rotor
    weight vanishes, or the threshold is not finite and positive.
    """
    kxx, kxy, kyx, kyy, cxx, cxy, cyx, cyy, _ip = reduced
    f_total = g["f_total"]
    cb = g["cb"]
    weight = g["weight_case"]
    omega = g["speed_rpm"] * PI / 30.0

    principal_ok = (
        abs(kxx) > _STIFFNESS_FLOOR
        and abs(kyy) > _STIFFNESS_FLOOR
        and abs(cxx) > _STIFFNESS_FLOOR
        and abs(cyy) > _STIFFNESS_FLOOR
    )
    if not principal_ok or f_total < _FORCE_FLOOR:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    kbxx = kxx * cb / f_total
    kbyy = kyy * cb / f_total
    kbxy = kxy * cb / f_total
    kbyx = kyx * cb / f_total
    cbxx = cxx * omega * cb / f_total
    cbyy = cyy * omega * cb / f_total
    cbxy = cxy * omega * cb / f_total
    cbyx = cyx * omega * cb / f_total

    threshold = 0.0
    whirl = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = (kbxx * cbyy + kbyy * cbxx - kbxy * cbyx - kbyx * cbxy) / (cbxx + cbyy)
        b2 = ((b1 - kbxx) * (b1 - kbyy) - kbyx * kbxy) / (cbxx * cbyy - cbxy * cbyx)
        omega_square = np.float64(b1) / np.float64(b2)
    if np.isfinite(omega_square) and omega_square >= 0.0 and weight >= _FORCE_FLOOR:
        omega_j = np.sqrt(omega_square) * np.sqrt(f_total * _GRAVITY / (cb * weight))
        if np.isfinite(omega_j) and omega_j > 0.0 and b2 > 0.0:
            threshold = float(omega_j * 30.0 / PI)
            whirl = float(np.sqrt(b2))
    return threshold, whirl, kbxx, kbyy, cbxx, cbyy


def _assemble_field_outputs(g):
    """Build the full-field arrays on the film mesh, for plotting.

    Film nodes are numbered row-major (circumferential index times the
    axial node count plus the axial index), so the flat nodal arrays
    reshape directly into ``(total_pads, dim_x, dim_z)`` grids.

    Parameters
    ----------
    g : dict
        Solver state at output time.

    Returns
    -------
    dict
        ``"theta"`` (rad, from each pad's leading edge),
        ``"axial_position"`` (m), ``"pressure"`` (Pa),
        ``"film_thickness"`` (m) and ``"film_temperature"`` (K, the
        radially averaged film temperature; the supply temperature when no
        thermal model ran), each shaped ``(total_pads, dim_x, dim_z)``,
        plus ``"leading_edge_angle"`` (rad, shape ``(total_pads,)``) to
        place each pad on the bearing circumference.

        When the full (conducting-pad) thermal model ran, the dict also
        carries the solid pad temperature: ``"pad_temperature"`` (K,
        shape ``(total_pads, dim_x, total_e_y_pad + 1)``, radial index 0
        at the babbitt surface increasing outward to the pad back) and
        ``"pad_radial_position"`` (m, shape ``(total_e_y_pad + 1,)``, the
        radius of each radial station).
    """
    total_pads = g["total_pads"]
    dim_x, dim_z = g["dim_x"], g["dim_z"]
    mesh = g["mesh"]
    pads = g["pads"]

    def grid(a):
        flat = np.asarray(a, dtype=float)[:, : dim_x * dim_z]
        return flat.reshape(total_pads, dim_x, dim_z).copy()

    fields = {
        "theta": grid(mesh.x_rad),
        "axial_position": grid(mesh.z),
        "pressure": grid(g["nodal_pressure"]),
        "film_thickness": grid(g["h_n"]),
        "film_temperature": grid(g["t_average"]),
        "leading_edge_angle": np.asarray(pads.leading_angle_rad, dtype=float).copy(),
    }

    if g.get("thermal_type") == "full":
        te_yp = g["total_e_y_pad"]
        stride = te_yp + g["total_e_y_film"] + 1
        cross = np.asarray(g["temp_full"], dtype=float)[:, : dim_x * stride]
        cross = cross.reshape(total_pads, dim_x, stride)
        fields["pad_temperature"] = cross[:, :, te_yp::-1].copy()
        fields["pad_radial_position"] = (
            g["cb"]
            + pads.journal_radius
            + np.linspace(0.0, pads.pad_thickness, te_yp + 1)
        )

    return fields


def _assemble_outputs(g, hp, reduced):
    """Build the output dict of derived quantities.

    Keys and gating follow the historical output surface: temperature and
    pressure summaries are only reported when a thermal model ran (the "no
    data" sentinel otherwise), thermal growth only for the deformation modes
    that compute it, pivot data only with pivot flexibility on, and side
    flows only for the axial-flow operating mode. The convergence-message
    strings have always been returned empty and stay that way.
    """
    xj, yj, cb = g["xj"], g["yj"], g["cb"]
    kxx_r, kxy_r, kyx_r, kyy_r, cxx_r, cxy_r, cyx_r, cyy_r, _ip = reduced
    total_pads = g["total_pads"]
    thermal_on = g.get("thermal_type") in ("adiabatic", "full")
    zeros_pad = [0.0] * total_pads

    press_max_all, press_max_pad_all = _max_nodal_pressure(g)
    if thermal_on:
        tpad_max = float(g.get("tpad_max", 0.0))
        tpad_max_pad = float(g.get("tpad_max_pad", 0))
        press_max = press_max_all
        press_max_pad = float(press_max_pad_all)
        temp_j = float(g["temp_j"])
        temp_inlet = np.asarray(g["temp_inlet"][:total_pads]).tolist()
        temp_outlet = np.asarray(g["temp_outlet"][:total_pads]).tolist()
        temp_outlet_bulk = np.asarray(g["temp_outlet_bulk"][:total_pads]).tolist()
        convec_back = np.asarray(g["convec_back"][:total_pads]).tolist()
        if g["q_supply"] > _SUPPLY_FLOW_FLOOR:
            temp_sump = float(g["temp_sump"])
        else:
            temp_sump = ZERO_TEMPERATURE_SENTINEL
        y_max_t = tpad_max
    else:
        # Without a thermal model there is no temperature or peak-pressure
        # report. Temperatures emit the sentinel; zero-anchored quantities
        # emit plain zero.
        tpad_max = ZERO_TEMPERATURE_SENTINEL
        tpad_max_pad = 0.0
        press_max = 0.0
        press_max_pad = 0.0
        temp_j = ZERO_TEMPERATURE_SENTINEL
        temp_sump = ZERO_TEMPERATURE_SENTINEL
        temp_inlet = [ZERO_TEMPERATURE_SENTINEL] * total_pads
        temp_outlet = [ZERO_TEMPERATURE_SENTINEL] * total_pads
        temp_outlet_bulk = [ZERO_TEMPERATURE_SENTINEL] * total_pads
        convec_back = zeros_pad
        y_max_t = ZERO_TEMPERATURE_SENTINEL

    # Per-pad pivot tilt.
    tilt_angle = g.get("tilt_angle")
    if tilt_angle is not None and getattr(tilt_angle, "size", 0):
        tilt_angle_out = np.asarray(tilt_angle).tolist()
    else:
        tilt_angle_out = list(zeros_pad)

    # Film thickness at the pad inlet and exit rows (axial midplane) and the
    # per-pad minimum.
    te_z = g["total_e_z_film"]
    h_n = g["h_n"]
    h_in = h_n[:, te_z // 2].tolist()
    h_exit = h_n[:, g["total_n_reynolds"] - te_z // 2 - 1].tolist()
    h_min = np.asarray(g["h_min"][:total_pads]).tolist()

    # Flow rates. The differential flow rate (the make-up flow the bearing
    # consumes) is the summed side leakage.
    q_in = np.asarray(g["q_in"][:total_pads]).tolist()
    q_out = np.asarray(g["q_out"][:total_pads]).tolist()
    q_sides = np.asarray(g["q_sides"][:total_pads])
    q_carryover = np.asarray(g["q_carryover"][:total_pads]).tolist()
    q_diff = float(np.sum(q_sides))
    if g.get("operating_type") == "axial_flow":
        q_side_a = np.asarray(g["q_sidea"][:total_pads]).tolist()
        q_side_b = np.asarray(g["q_sideb"][:total_pads]).tolist()
    else:
        q_side_a = list(zeros_pad)
        q_side_b = list(zeros_pad)

    # Pivot deformation and stiffness, reported only when pivot flexibility
    # is part of the deformation model.
    if g["deform_type"] in PIVOT_FLEX_DEFORM_TYPES:
        deform_pivot = np.asarray(g["deform_pivot"][:total_pads]).tolist()
        k_pivot = np.asarray(g["k_pivot"][:total_pads]).tolist()
    else:
        deform_pivot = list(zeros_pad)
        k_pivot = list(zeros_pad)

    # Thermal growth of journal and shell, and the clearance change from a
    # shell shrink fit.
    if g["deform_type"] in _THERMAL_GROWTH_DEFORM_TYPES:
        deform_journal = float(g["deform_journal"])
        deform_shell = float(g["deform_shell"])
    else:
        deform_journal = 0.0
        deform_shell = 0.0
    if g["deform_type"] is not None and abs(g["crush"]) > _CRUSH_FLOOR:
        cb_crush = cb - float(g["deform_crush"])
    else:
        cb_crush = 0.0

    theta_c, hc, pc, tc, tb, tm, dhc = _center_plane_profiles(g)
    temp_probe = _probe_temperatures(g)
    total_probes = temp_probe.size
    film_angle = _film_onset_angles(g)

    # Load summary and Sommerfeld number for the case (evaluated at the
    # supply viscosity).
    f_total = g["f_total"]
    f_specific = g["f_specific"]
    if f_specific > 0.0:
        sommerfeld = (
            (g["vis_supply"] * g["speed_rpm"] / f_specific)
            * (g["journal_diameter"] / cb) ** 2
            / 240.0
        )
    else:
        sommerfeld = 0.0

    threshold, whirl, kbxx, kbyy, cbxx, cbyy = _rigid_rotor_stability(g, reduced)

    eccentricity = float(np.sqrt((xj / cb) ** 2 + (yj / cb) ** 2))
    attitude = _attitude_rad(xj, yj)
    omega = g["speed_rpm"] * PI / 30.0

    # One entry per speed case. Callers expect a list even for a single case,
    # so every scalar is wrapped in a 1-element list and every per-pad or
    # per-node array in a 1-element list of lists.
    return {
        "xj_cb": [xj / cb],
        "yj_cb": [yj / cb],
        "eccentricity": [eccentricity],
        "attitude": [attitude],
        "power_loss": [hp],
        "tpad_max": [tpad_max],
        "max_pad_temperature": [tpad_max_pad],
        "max_pressure": [press_max],
        "max_pad_pressure": [press_max_pad],
        "differential_flow_rate": [q_diff],
        "deform_journal": [deform_journal],
        "deform_shell": [deform_shell],
        "cb_crush": [cb_crush],
        "kxx": [kxx_r],
        "kxy": [kxy_r],
        "kyx": [kyx_r],
        "kyy": [kyy_r],
        "cxx": [cxx_r],
        "cxy": [cxy_r],
        "cyx": [cyx_r],
        "cyy": [cyy_r],
        "threshold_rpm": [threshold],
        "whirl_frequency_ratio": [whirl],
        # Convergence-message strings: the output surface has always
        # returned these empty; convergence is judged from the numbers.
        "non_convergence": [""],
        "sump_temperature": [""],
        "film_temperature": [""],
        "inlet_temperature": [""],
        "journal_temperature": [""],
        "pad_deformation": [""],
        "rigid_rotor": [""],
        "pad_index": [[float(p + 1) for p in range(total_pads)]],
        "tilt_angle": [tilt_angle_out],
        "h_in": [h_in],
        "h_exit": [h_exit],
        "h_min": [h_min],
        "temp_in_let": [temp_inlet],
        "temp_out_let": [temp_outlet],
        "temp_out_let_bulk": [temp_outlet_bulk],
        "convec_back": [convec_back],
        "q_carry_over": [q_carryover],
        "q_in": [q_in],
        "q_out": [q_out],
        "q_sides": [q_sides.tolist()],
        "q_side_a": [q_side_a],
        "q_side_b": [q_side_b],
        "deform_pivot": [deform_pivot],
        "k_pivot": [k_pivot],
        "dhc_re_max": [np.asarray(g["re_max"][:total_pads]).tolist()],
        "i_probe": [[float(p + 1) for p in range(total_probes)]],
        "temp_probe": [temp_probe.tolist()],
        "fx_hydro": [float(g["fx_hydro"])],
        "fy_hydro": [float(g["fy_hydro"])],
        "theta_c": [theta_c.tolist()],
        "hc": [hc.tolist()],
        "pc": [pc.tolist()],
        "tc": [tc.tolist()],
        "tb": [tb.tolist()],
        "tm": [tm.tolist()],
        "dhc": [dhc.tolist()],
        "journal_temperature_output": [temp_j],
        "sump_drain_temperature": [temp_sump],
        "film_angle": [film_angle.tolist()],
        "x_som_m": [sommerfeld],
        "x_speed": [omega],
        "x_xload": [g["fxs_case"]],
        "x_yload": [g["fys_case"]],
        "x_tot_ld": [f_total],
        "x_spec_ld": [f_specific],
        "x_oil": [float(g["q_supply"])],
        "y_kxx": [kxx_r],
        "y_kxy": [kxy_r],
        "y_kyx": [kyx_r],
        "y_kyy": [kyy_r],
        "y_cxx": [cxx_r],
        "y_cxy": [cxy_r],
        "y_cyx": [cyx_r],
        "y_cyy": [cyy_r],
        "y_kbxx": [kbxx],
        "y_kbyy": [kbyy],
        "y_cbxx": [cbxx],
        "y_cbyy": [cbyy],
        "y_ecc": [eccentricity],
        "y_phi": [attitude],
        "y_flow": [q_diff],
        "y_loss": [hp],
        "y_sump": [float(g["temp_sump"])],
        "y_jrnl_t": [float(g["temp_j"])],
        "y_max_t": [y_max_t],
        "y_max_p": [press_max_all],
    }
