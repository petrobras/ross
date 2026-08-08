"""Film and pad temperature: the energy equation.

Computes the film -- and, in the full model, pad -- temperature distribution
and feeds it back into the viscosity field the hydrodynamic solver uses.

Two thermal models are provided:

* **Adiabatic** (``thermal_type="adiabatic"``): a 2-D energy equation solved
  over the (x, z) Reynolds mesh of each pad. The dissipation is averaged
  radially across the film and the resulting temperature is taken as constant
  through the film thickness. Entry point :func:`thermal_adiabatic`.
* **Full** (``thermal_type="full"``): a generalized 2-D energy equation solved
  over the (x, y) energy mesh of each pad, spanning the film *and* the pad
  solid. It includes conduction into the pad, convection on the pad edges and
  back, and the journal surface temperature treatment. Entry point
  :func:`thermal_full`.

Index conventions
-----------------
Arrays are 0-based throughout, as everywhere in the package:

* array shapes are the natural NumPy shapes -- ``x_reynolds`` is
  ``(total_pads, dim_xz)``, with no padding on any axis;
* ``pad_index`` and every node / element / column number (including the values
  stored in ``n_index_*`` / ``e_index_*`` / ``match_nodes_*``) index the arrays
  directly, and element/node loops run ``range(total_*)``;
* ``match_nodes_*`` uses ``-1`` for unused slots.

Mutated outputs are returned as arrays or tuples; nothing is stored in module
globals.

The mesh, film-thickness, velocity and viscosity fields these routines consume
come from :mod:`ross.bearings.fluid_film.mesh` and
:mod:`ross.bearings.fluid_film.hydrodynamics` and are passed in, not recomputed.

References
----------
.. [1] Safar, Z., & Szeri, A. Z. (1974). Thermohydrodynamic lubrication in
       laminar and turbulent regimes. ASME Journal of Lubrication
       Technology, 96(1), 48-56.
"""

import numpy as np

from ross.bearings.fluid_film import banded
from ross.bearings.fluid_film._numba_kernels import (
    assemble_press_jit,
    effective_conduct_jit,
    element_temp_interior_jit,
    energy_coeffs_flooded_jit,
    expand_film_temp_flooded_jit,
    include_press_jit,
    temp_xy_assemble_all_jit,
    temp_xy_boundary_all_jit,
    trapezoid_jit,
    update_vis_jit,
)

# Viscous-dissipation unit-conversion factor. SI units are coherent (watts,
# joule/(m^3 K), ...) so no conversion is needed; the factor is kept as a
# named constant because it appears throughout the energy routines.
DISSIP_FACTOR = 1.0

# Turbulent Prandtl number used for the effective conductivity.
PR_TURB = 0.769


# ---------------------------------------------------------------------------
# Shared finite-element helpers
#
# The same assembly / banded-LU algorithms the pressure module uses, kept
# private here so this module stands alone.
# ---------------------------------------------------------------------------
def _trapezoid(t, f, start, stop):
    """Trapezoidal integral of ``f`` over ``t`` for unequal spacing.

    Integrates the samples ``t[start:stop]``, using ordinary Python slice
    bounds.

    Parameters
    ----------
    t : numpy.ndarray
        Abscissae.
    f : numpy.ndarray
        Integrand sampled at ``t``.
    start, stop : int
        Slice bounds selecting the samples to integrate.

    Returns
    -------
    float
        The trapezoidal integral, zero for fewer than two samples.
    """
    if stop - start < 2:
        return 0.0
    t = np.ascontiguousarray(t, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)
    return trapezoid_jit(t, f, start, stop)


def _assemble(
    e_matrix, e_column, local_coordinates, bandwidth, global_matrix, global_column
):
    """Assemble a 4-node element matrix/column into the condensed system.

    Mutates ``global_matrix`` and ``global_column`` in place.
    ``local_coordinates`` is a length-4 sequence of 0-based global node
    numbers, indexed directly by the shared ``assemble_press_jit`` kernel.
    """
    lc = np.ascontiguousarray(local_coordinates, dtype=np.int64)
    em = np.ascontiguousarray(e_matrix, dtype=np.float64)
    ec = np.ascontiguousarray(e_column, dtype=np.float64)
    assemble_press_jit(em, ec, lc, bandwidth, global_matrix, global_column)


def _include_prescribed(
    global_matrix, global_column, bandwidth, total_bc, bc_index, prescribed, total_n
):
    """Apply prescribed nodal values to the condensed system.

    Mutates ``global_matrix`` and ``global_column`` in place. ``bc_index``
    holds 0-based node numbers (length ``total_bc``), indexed directly by the
    shared ``include_press_jit`` kernel. ``prescribed`` holds the matching
    prescribed values.
    """
    bc_idx = np.ascontiguousarray(bc_index[:total_bc], dtype=np.int64)
    pres = np.ascontiguousarray(prescribed[:total_bc], dtype=np.float64)
    include_press_jit(
        global_matrix, global_column, bandwidth, total_bc, bc_idx, pres, total_n
    )


def adiabatic_bc(
    operating,
    pad_index,
    mesh,
    axial_bc,
    axial_length,
    temp_inlet,
):
    """Build the prescribed-temperature BC list for the adiabatic solve.

    The inlet edge (``x == 0``) is held at the pad inlet temperature; depending
    on ``operating_type`` the axial ends / upstream edge are held at the supply
    temperature.

    Parameters
    ----------
    operating : OperatingPoint
        Speed and pressure conditions of the case; ``operating_type``
        selects which edges are held at ``temp_supply``.
    pad_index : int
        0-based pad index.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    axial_bc : numpy.ndarray
        Axial BC type per node (from :func:`axial_bc_adiab`),
        length ``dim_xz``.
    axial_length : numpy.ndarray
        Axial length per pad, shape ``(total_pads,)``, m.
    temp_inlet : numpy.ndarray
        Pad inlet temperature, shape ``(total_pads,)``, K.

    Returns
    -------
    temp_bc_index : numpy.ndarray
        0-based prescribed-node indices, length ``total_bc_adiab``.
    prescribed_temp : numpy.ndarray
        Matching prescribed temperatures.
    total_bc_adiab : int
        Number of prescribed nodes.
    """
    pad = pad_index
    temp_bc_index = np.zeros(mesh.dim_xz, dtype=int)
    prescribed_temp = np.zeros(mesh.dim_xz)
    j = 0
    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        if abs(mesh.x[pad, node]) < 1.0e-6:
            temp_bc_index[j] = node
            prescribed_temp[j] = temp_inlet[pad]
            j += 1
        elif operating.operating_type == "high_ambient_pressure":
            if axial_bc[node] == 1 and (
                abs(mesh.z[pad, node]) < 1.0e-6
                or abs(mesh.z[pad, node] - axial_length[pad]) < 1.0e-6
            ):
                temp_bc_index[j] = node
                prescribed_temp[j] = operating.temp_supply
                j += 1
        elif operating.operating_type == "axial_flow":
            if abs(mesh.z[pad, node]) < 1.0e-6:
                temp_bc_index[j] = node
                prescribed_temp[j] = operating.temp_supply
                j += 1
    return temp_bc_index, prescribed_temp, j


def pde_coeff_adiab(
    pad_index,
    mesh,
    pads,
    velocity_x_n,
    velocity_z_n,
    dudy_n,
    dwdy_n,
    h_n,
    vis_effect_3d,
    lube,
):
    """Nodal PDE coefficients of the adiabatic energy equation.

    For every Reynolds node, the convective and dissipation coefficients are
    obtained by integrating the velocity and dissipation profiles radially
    across the film, distinguishing the pocket, dam and pocket-edge regions
    (relevant for pressure-dam pads). The conduction coefficients are constant
    (lubricant conductivity).

    Returns
    -------
    kx_n, kz_n, mx_n, mz_n, q_n : numpy.ndarray
        Per-node coefficient arrays of length ``dim_xz`` (0-based node
        indexing).
    """
    pad = pad_index
    kx_n = np.zeros(mesh.dim_xz)
    kz_n = np.zeros(mesh.dim_xz)
    mx_n = np.zeros(mesh.dim_xz)
    mz_n = np.zeros(mesh.dim_xz)
    q_n = np.zeros(mesh.dim_xz)

    t = np.zeros(mesh.dim_yf)
    f1 = np.zeros(mesh.dim_yf)
    f2 = np.zeros(mesh.dim_yf)
    f3 = np.zeros(mesh.dim_yf)

    limit1 = mesh.total_e_y_trackbl[pad] + mesh.total_e_y_trackcore[pad] + 1
    limit2 = mesh.total_e_y_film + 1

    def _fill(node, jlo, depth_offset):
        """Fill the T/F1/F2/F3 integrands for j in [jlo, limit2]."""
        for j in range(jlo, limit2 + 1):
            m = mesh.match_nodes_xz[node, j - 1]
            t[j - 1] = mesh.y_3d[pad, m] - pads.pad_thickness - depth_offset
            f1[j - 1] = velocity_x_n[pad, m]
            f2[j - 1] = velocity_z_n[pad, m]
            f3[j - 1] = (
                vis_effect_3d[pad, m]
                * (dudy_n[pad, m] ** 2 + dwdy_n[pad, m] ** 2)
                / DISSIP_FACTOR
            )

    def _coeffs(node, jlo):
        inte_u = _trapezoid(t, f1, jlo - 1, limit2)
        inte_w = _trapezoid(t, f2, jlo - 1, limit2)
        inte_dissip = _trapezoid(t, f3, jlo - 1, limit2)
        h = h_n[pad, node]
        mx_n[node] = lube.density * lube.cp * inte_u / h
        mz_n[node] = lube.density * lube.cp * inte_w / h
        q_n[node] = -inte_dissip / h

    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        kx_n[node] = -lube.conduct
        kz_n[node] = -lube.conduct

        x = mesh.x[pad, node]
        z = mesh.z[pad, node]
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
            _fill(node, 1, 0.0)
            _coeffs(node, 1)
        elif in_dam:
            _fill(node, limit1, pads.depth_track[pad])
            _coeffs(node, limit1)
        else:
            # Pocket edge: treat as pocket if the edge is the pad edge
            # (unshrouded pocket), otherwise treat as dam.
            if pads.pocket_edge_is_pad(x, z, pad):
                _fill(node, 1, 0.0)
                _coeffs(node, 1)
            else:
                _fill(node, limit1, pads.depth_track[pad])
                _coeffs(node, limit1)

    return kx_n, kz_n, mx_n, mz_n, q_n


def element_adiab(kx_e, kz_e, mx_e, mz_e, q_e, l_e, w_e):
    """4-node element matrix and column for the adiabatic energy equation.

    ``kx_e``/``kz_e`` are the (negative) conduction coefficients,
    ``mx_e``/``mz_e`` the convection coefficients, ``q_e`` the source,
    ``l_e``/``w_e`` the element length/width.

    Returns
    -------
    e_matrix : numpy.ndarray
        Shape ``(4, 4)`` element matrix.
    e_column : numpy.ndarray
        Length-4 element column.
    """
    e = np.zeros((4, 4))
    kxw = kx_e * w_e
    kzl = kz_e * l_e
    mxw = mx_e * w_e
    mzl = mz_e * l_e
    # fmt: off
    e[0, 0] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (-mxw / 6.0) - (-mzl / 6.0)
    e[0, 1] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (mxw / 6.0) - (-mzl / 12.0)
    e[0, 2] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (mxw / 12.0) - (mzl / 12.0)
    e[0, 3] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (-mxw / 12.0) - (mzl / 6.0)
    e[1, 0] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (-mxw / 6.0) - (-mzl / 12.0)
    e[1, 1] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (mxw / 6.0) - (-mzl / 6.0)
    e[1, 2] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (mxw / 12.0) - (mzl / 6.0)
    e[1, 3] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (-mxw / 12.0) - (mzl / 12.0)
    e[2, 0] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (-mxw / 12.0) - (-mzl / 12.0)
    e[2, 1] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (mxw / 12.0) - (-mzl / 6.0)
    e[2, 2] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (mxw / 6.0) - (mzl / 6.0)
    e[2, 3] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (-mxw / 6.0) - (mzl / 12.0)
    e[3, 0] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (-mxw / 12.0) - (-mzl / 6.0)
    e[3, 1] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (mxw / 12.0) - (-mzl / 12.0)
    e[3, 2] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (mxw / 6.0) - (mzl / 12.0)
    e[3, 3] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (-mxw / 6.0) - (mzl / 6.0)
    # fmt: on
    e_column = np.full(4, q_e * l_e * w_e / 4.0)
    return e, e_column


def temp_adiabatic(
    operating,
    pad_index,
    mesh,
    axial_bc,
    pads,
    velocity_x_n,
    velocity_z_n,
    dudy_n,
    dwdy_n,
    h_n,
    vis_effect_3d,
    lube,
    temp_inlet,
    temp_adiab,
):
    """Solve the adiabatic film temperature for one pad.

    Assembles and solves the 2-D adiabatic energy equation over the pad's
    Reynolds mesh, writing the result into the ``pad_index`` row of
    ``temp_adiab``.

    Parameters
    ----------
    operating : OperatingPoint
        Speed and pressure conditions of the case;
        ``bearing_type == "pressure_dam"`` zeroes the axial convection for
        stability and ``operating_type`` selects the prescribed-temperature
        edges (see :func:`adiabatic_bc`).
    pad_index : int
        0-based pad index being solved.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity, element sizes and the
        ``match_nodes_xz`` map to the 3-D film nodes.
    axial_bc : numpy.ndarray
        Axial BC type per node (only used for
        ``operating_type == "high_ambient_pressure"``).
    pads : PadGeometry
        Per-pad geometry.
    velocity_x_n, velocity_z_n : numpy.ndarray
        Circumferential and axial film velocity at the 3-D nodes, shape
        ``(total_pads, dim_3d)``, m/s.
    dudy_n, dwdy_n : numpy.ndarray
        Their radial derivatives, same shape, 1/s.
    h_n : numpy.ndarray
        Nodal film thickness, shape ``(total_pads, dim_xz)``, m.
    vis_effect_3d : numpy.ndarray
        Effective (laminar + eddy) viscosity at the 3-D nodes, shape
        ``(total_pads, dim_3d)``, Pa*s.
    lube : Lubricant
        Lubricant properties.
    temp_inlet : numpy.ndarray
        Pad inlet temperature, shape ``(total_pads,)``, K.
    temp_adiab : numpy.ndarray
        2-D adiabatic temperature field, shape ``(total_pads, dim_xz)``;
        updated for ``pad_index`` and returned.

    Returns
    -------
    numpy.ndarray
        ``temp_adiab`` with the ``pad_index`` row filled in.
    """
    pad = pad_index
    dim_xz = mesh.dim_xz
    total_column_adiab = 2 * mesh.bandwidth - 1

    temp_bc_index, prescribed_temp, total_bc_adiab = adiabatic_bc(
        operating,
        pad_index,
        mesh,
        axial_bc,
        pads.axial_length,
        temp_inlet,
    )

    kx_n, kz_n, mx_n, mz_n, q_n = pde_coeff_adiab(
        pad_index,
        mesh,
        pads,
        velocity_x_n,
        velocity_z_n,
        dudy_n,
        dwdy_n,
        h_n,
        vis_effect_3d,
        lube,
    )

    # Band storage: only ``total_column_adiab = 2*bw - 1`` columns are ever
    # touched (assembly writes at ``jcol = icol - irow + bw - 1``).
    global_matrix = np.zeros((dim_xz, total_column_adiab))
    global_column = np.zeros(dim_xz)

    for i in range(mesh.total_elements):
        current = mesh.e_index[i]
        ni = mesh.node_i[current]
        nj = mesh.node_j[current]
        nk = mesh.node_k[current]
        nl = mesh.node_l[current]
        local = (ni, nj, nk, nl)
        kx_e = (kx_n[ni] + kx_n[nj] + kx_n[nk] + kx_n[nl]) / 4.0
        kz_e = (kz_n[ni] + kz_n[nj] + kz_n[nk] + kz_n[nl]) / 4.0
        mx_e = (mx_n[ni] + mx_n[nj] + mx_n[nk] + mx_n[nl]) / 4.0
        mz_e = (mz_n[ni] + mz_n[nj] + mz_n[nk] + mz_n[nl]) / 4.0
        q_e = (q_n[ni] + q_n[nj] + q_n[nk] + q_n[nl]) / 4.0
        l_e = mesh.e_length[pad, current]
        w_e = mesh.e_width[pad, current]
        if operating.bearing_type == "pressure_dam":
            mz_e = 0.0
        e_matrix, e_column = element_adiab(kx_e, kz_e, mx_e, mz_e, q_e, l_e, w_e)
        _assemble(
            e_matrix, e_column, local, mesh.bandwidth, global_matrix, global_column
        )

    _include_prescribed(
        global_matrix,
        global_column,
        mesh.bandwidth,
        total_bc_adiab,
        temp_bc_index,
        prescribed_temp,
        mesh.total_nodes,
    )

    global_matrix, a_lower, index1, _d = banded.lu_factor(
        global_matrix, mesh.total_nodes, mesh.bandwidth
    )
    global_column = banded.lu_solve(
        global_matrix,
        mesh.total_nodes,
        mesh.bandwidth,
        a_lower,
        index1,
        global_column,
    )

    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        temp_adiab[pad, node] = global_column[node]

    return temp_adiab


# ---------------------------------------------------------------------------
# Adiabatic driver
# ---------------------------------------------------------------------------
def axial_bc_adiab(
    pad_index,
    mesh,
    axial_length,
    ambient_press1,
    nodal_pressure,
):
    """Axial-end BC type for high-ambient-pressure bearings.

    Along the pad centre line, nodes whose pressure is below ambient get an
    axial specified-value BC (type 1); the type is then broadcast to
    all nodes sharing the same ``x`` location.

    Returns
    -------
    numpy.ndarray
        ``axial_bc`` of length ``dim_xz`` (0-based node indexing).
    """
    pad = pad_index
    axial_bc = np.zeros(mesh.dim_xz, dtype=int)
    x = np.zeros(mesh.dim_x)
    tz = np.zeros(mesh.dim_x, dtype=int)

    k = 0
    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        if abs(mesh.z[pad, node] - 0.5 * axial_length[pad]) < 1.0e-6:
            x[k] = mesh.x[pad, node]
            if (
                nodal_pressure[pad, node] > ambient_press1
                or abs(nodal_pressure[pad, node] - ambient_press1) < 1.0e-6
            ):
                tz[k] = 0
            else:
                tz[k] = 1
            k += 1

    for i in range(1, mesh.total_e_x_film + 1 + 1):
        for j in range(mesh.total_nodes):
            node = mesh.n_index[j]
            if abs(mesh.x[pad, node] - x[i - 1]) < 1.0e-6:
                axial_bc[node] = tz[i - 1]

    return axial_bc


def temp_average_adiab(
    pad_index, total_n_reynolds, n_index_reynolds, temp_adiab, temp_adiab_old
):
    """Store the old adiabatic temperature and return its pad average.

    Copies the ``pad_index`` row of ``temp_adiab`` into ``temp_adiab_old`` and
    returns the average.

    Returns
    -------
    temp_adiab_old : numpy.ndarray
        Updated old-temperature array.
    temp_average_old : float
        Average over the pad's active nodes.
    """
    pad = pad_index
    total = 0.0
    for i in range(total_n_reynolds):
        node = n_index_reynolds[i]
        temp_adiab_old[pad, node] = temp_adiab[pad, node]
        total += temp_adiab_old[pad, node]
    return temp_adiab_old, total / total_n_reynolds


def temp_3d_adiab(
    pad_index,
    mesh,
    temp_adiab,
    temp_3d,
):
    """Expand the 2-D adiabatic temperature to the 3-D film (constant in y).

    Returns the updated ``temp_3d``.
    """
    pad = pad_index
    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        for j in range(1, mesh.total_e_y_film + 1 + 1):
            m = mesh.match_nodes_xz[node, j - 1]
            temp_3d[pad, m] = temp_adiab[pad, node]
    return temp_3d


def update_vis_adiab(
    pad_index,
    mesh,
    flow_regime_track,
    flow_regime_dam,
    pads,
    lube,
    speed_surface,
    temp_3d,
    vis_n_3d,
    vis_n_average,
    h_n,
    scale_turb_track,
    scale_turb_dam,
    turbulence,
):
    """Update viscosity, averaged viscosity and flow regime (adiabatic).

    Updates the nodal 3-D viscosity from the new temperature field, computes
    the radially averaged viscosity and the local Reynolds number (pocket / dam
    regions handled separately), and updates the track/dam flow regimes and
    turbulence scaling factors.

    Returns
    -------
    vis_n_3d : numpy.ndarray
        Updated 3-D nodal viscosity.
    vis_n_average : numpy.ndarray
        Updated radially averaged nodal viscosity.
    flow_regime_track, flow_regime_dam : numpy.ndarray
        Updated flow-regime flags (0 laminar, 1 transition, 2 turbulent).
    scale_turb_track, scale_turb_dam : numpy.ndarray
        Updated turbulence scaling factors.
    """
    pad = pad_index
    dim_yf = mesh.match_nodes_xz.shape[1]
    dim_xz = mesh.x.shape[1]
    re_n = np.zeros(dim_xz)
    t = np.zeros(dim_yf)
    f = np.zeros(dim_yf)

    limit1 = mesh.total_e_y_trackbl[pad] + mesh.total_e_y_trackcore[pad] + 1
    limit2 = mesh.total_e_y_film + 1

    re_max_track = 0.0
    re_max_dam = 0.0

    def _fill(node, jlo, depth_offset):
        for j in range(jlo, limit2 + 1):
            m = mesh.match_nodes_xz[node, j - 1]
            t[j - 1] = mesh.y_3d[pad, m] - pads.pad_thickness - depth_offset
            f[j - 1] = vis_n_3d[pad, m]

    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        # Update viscosity at each 3-D node above this 2-D node.
        for j in range(1, mesh.total_e_y_film + 1 + 1):
            m = mesh.match_nodes_xz[node, j - 1]
            vis_n_3d[pad, m] = lube.viscosity_at(temp_3d[pad, m])

        x = mesh.x[pad, node]
        z = mesh.z[pad, node]
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
            track_region = True
            _fill(node, 1, 0.0)
            inte = _trapezoid(t, f, 0, limit2)
        elif in_dam:
            track_region = False
            _fill(node, limit1, pads.depth_track[pad])
            inte = _trapezoid(t, f, limit1 - 1, limit2)
        else:
            if pads.pocket_edge_is_pad(x, z, pad):
                track_region = True
                _fill(node, 1, 0.0)
                inte = _trapezoid(t, f, 0, limit2)
            else:
                track_region = False
                _fill(node, limit1, pads.depth_track[pad])
                inte = _trapezoid(t, f, limit1 - 1, limit2)

        h = h_n[pad, node]
        vis_n_average[pad, node] = inte / h
        re_n[node] = lube.density * speed_surface * h / vis_n_average[pad, node]
        if track_region:
            re_max_track = max(re_max_track, re_n[node])
        else:
            re_max_dam = max(re_max_dam, re_n[node])

    _update_flow_regime(
        flow_regime_track,
        scale_turb_track,
        pad,
        re_max_track,
        turbulence,
    )
    _update_flow_regime(
        flow_regime_dam,
        scale_turb_dam,
        pad,
        re_max_dam,
        turbulence,
    )

    return (
        vis_n_3d,
        vis_n_average,
        flow_regime_track,
        flow_regime_dam,
        scale_turb_track,
        scale_turb_dam,
    )


def _update_flow_regime(
    flow_regime,
    scale_turb,
    pad,
    re_max,
    turbulence,
):
    """Set the flow regime flag and turbulence scaling factor for one region.

    Shared by the adiabatic and full viscosity-update routines.
    """
    if re_max < turbulence.re_lower:
        flow_regime[pad] = 0
        scale_turb[pad] = 0.0
    elif turbulence.re_lower < re_max < turbulence.re_upper:
        flow_regime[pad] = 1
        scale_turb[pad] = (
            1.0
            - (
                (turbulence.re_upper - re_max)
                / (turbulence.re_upper - turbulence.re_lower)
            )
            ** turbulence.scale_factor_exponent
        )
    else:
        flow_regime[pad] = 2
        scale_turb[pad] = 1.0


def temp_adiab_residual(
    total_pads, total_n_reynolds, n_index_reynolds, temp_adiab1, temp_adiab_old
):
    """Root-mean-square change of the adiabatic 2-D temperature.

    Returns the RMS over all pads / active nodes of ``temp_adiab1 -
    temp_adiab_old``.
    """
    total = 0.0
    n = 0
    for pad in range(total_pads):
        for i in range(total_n_reynolds):
            node = n_index_reynolds[i]
            n += 1
            total += (temp_adiab1[pad, node] - temp_adiab_old[pad, node]) ** 2
    return np.sqrt(total / n)


def thermal_adiabatic(
    total_pads,
    operating,
    pads,
    mesh,
    lube,
    vis_effect_3d,
    vis_n_3d,
    vis_n_average,
    flow_regime_track,
    flow_regime_dam,
    velocity_x_n,
    velocity_z_n,
    dudy_n,
    dwdy_n,
    h_n,
    scale_turb_track,
    scale_turb_dam,
    nodal_pressure,
    temp_inlet,
    temp_adiab,
    temp_3d,
    turbulence,
    relax_t_max,
):
    """Adiabatic film-temperature solution for all pads (one iteration).

    For every pad it (1) determines the axial BC type for high-ambient-pressure
    bearings, (2) stores the old temperature, (3) solves the adiabatic energy
    equation, (4) under-relaxes so the average pad temperature changes by at
    most 10 K per iteration (further limited by ``relax_t_max``), (5)
    expands to the 3-D film and (6) updates the viscosity / flow regime.
    Finally it returns the RMS temperature change.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    pads : PadGeometry
        Per-pad geometry.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    lube : Lubricant
        Lubricant properties.
    vis_effect_3d : numpy.ndarray
        Effective (laminar + eddy) viscosity at the 3-D nodes, shape
        ``(total_pads, dim_3d)``, Pa*s.
    vis_n_3d : numpy.ndarray
        Laminar nodal viscosity, same shape, Pa*s; updated in place.
    vis_n_average : numpy.ndarray
        Radially averaged nodal viscosity, shape ``(total_pads, dim_xz)``,
        Pa*s; updated.
    flow_regime_track, flow_regime_dam : numpy.ndarray
        Per-pad flow-regime flags (0 laminar, 1 transition, 2 turbulent) in
        the track and dam regions; updated.
    velocity_x_n, velocity_z_n : numpy.ndarray
        Circumferential and axial film velocity at the 3-D nodes, shape
        ``(total_pads, dim_3d)``, m/s.
    dudy_n, dwdy_n : numpy.ndarray
        Their radial derivatives, same shape, 1/s.
    h_n : numpy.ndarray
        Nodal film thickness, shape ``(total_pads, dim_xz)``, m.
    scale_turb_track, scale_turb_dam : numpy.ndarray
        Per-pad turbulence scaling factors; updated alongside the regimes.
    nodal_pressure : numpy.ndarray
        Nodal film pressure, shape ``(total_pads, dim_xz)``, Pa; only read
        for ``operating_type == "high_ambient_pressure"``.
    temp_inlet : numpy.ndarray
        Pad inlet temperature, shape ``(total_pads,)``, K.
    temp_adiab : numpy.ndarray
        2-D adiabatic temperature field, shape ``(total_pads, dim_xz)``, K;
        the starting value and the under-relaxed result.
    temp_3d : numpy.ndarray
        3-D film temperature, shape ``(total_pads, dim_3d)``, K; updated.
    turbulence : Turbulence
        Turbulence-model constants.
    relax_t_max : float
        Upper bound on the temperature relaxation factor.

    Returns
    -------
    temp_adiab : numpy.ndarray
        Updated (under-relaxed) 2-D adiabatic temperature.
    temp_3d : numpy.ndarray
        Updated 3-D film temperature.
    vis_n_3d, vis_n_average : numpy.ndarray
        Updated viscosity fields.
    flow_regime_track, flow_regime_dam : numpy.ndarray
        Updated flow regimes.
    scale_turb_track, scale_turb_dam : numpy.ndarray
        Updated turbulence scaling factors.
    rms_temp : float
        RMS temperature change between iterations.

    Notes
    -----
    ``rms_temp`` is computed from the *un-relaxed* new solution
    ``temp_adiab1`` versus the old one. Because ``temp_adiab1`` is a pad-local
    scratch field, the residual over all pads sees only the last pad's scratch
    row for pads other than the current one. That is deliberate -- the pinned
    regression fixtures depend on this exact residual -- and is reproduced
    here by accumulating ``temp_adiab1`` per pad.
    """
    dim_xz = mesh.x.shape[1]
    axial_bc = np.zeros(dim_xz, dtype=int)
    temp_adiab_old = temp_adiab.copy()
    temp_adiab1 = temp_adiab.copy()

    for pad_index in range(total_pads):
        pad = pad_index

        if operating.operating_type == "high_ambient_pressure":
            axial_bc = axial_bc_adiab(
                pad_index,
                mesh,
                pads.axial_length,
                operating.ambient_press1,
                nodal_pressure,
            )

        temp_adiab_old, temp_average_old = temp_average_adiab(
            pad_index,
            mesh.total_nodes,
            mesh.n_index,
            temp_adiab,
            temp_adiab_old,
        )

        temp_adiab1 = temp_adiabatic(
            operating,
            pad_index,
            mesh,
            axial_bc,
            pads,
            velocity_x_n,
            velocity_z_n,
            dudy_n,
            dwdy_n,
            h_n,
            vis_effect_3d,
            lube,
            temp_inlet,
            temp_adiab1,
        )

        total = 0.0
        for i in range(mesh.total_nodes):
            total += temp_adiab1[pad, mesh.n_index[i]]
        temp_average = total / mesh.total_nodes

        if abs(temp_average - temp_average_old) > 10.0:
            relax_t = 10.0 / abs(temp_average - temp_average_old)
        else:
            relax_t = 1.0
        relax_t = min(relax_t, relax_t_max)
        for i in range(mesh.total_nodes):
            node = mesh.n_index[i]
            temp_adiab[pad, node] = (
                relax_t * temp_adiab1[pad, node]
                + (1.0 - relax_t) * temp_adiab_old[pad, node]
            )

        temp_3d = temp_3d_adiab(
            pad_index,
            mesh,
            temp_adiab,
            temp_3d,
        )

        (
            vis_n_3d,
            vis_n_average,
            flow_regime_track,
            flow_regime_dam,
            scale_turb_track,
            scale_turb_dam,
        ) = update_vis_adiab(
            pad_index,
            mesh,
            flow_regime_track,
            flow_regime_dam,
            pads,
            lube,
            operating.speed_surface,
            temp_3d,
            vis_n_3d,
            vis_n_average,
            h_n,
            scale_turb_track,
            scale_turb_dam,
            turbulence,
        )

    rms_temp = temp_adiab_residual(
        total_pads, mesh.total_nodes, mesh.n_index, temp_adiab1, temp_adiab_old
    )

    return (
        temp_adiab,
        temp_3d,
        vis_n_3d,
        vis_n_average,
        flow_regime_track,
        flow_regime_dam,
        scale_turb_track,
        scale_turb_dam,
        rms_temp,
    )


# ===========================================================================
# Full model: 2-D energy equation on the film + pad mesh
# ===========================================================================
def temp_bc(
    temp_j_type,
    pad_index,
    energy_mesh,
    total_e_y_film,
    total_e_z_film,
    temp_j,
    temp_inlet,
    velocity_x_n,
):
    """Prescribed-temperature BC list for the full energy solve.

    The leading film edge is held at the inlet temperature where the flow
    enters the pad, and the journal surface is held at ``temp_j`` unless it is
    insulated (``temp_j_type == "insulated_shaft_surface"``).

    Returns
    -------
    temp_bc_index : numpy.ndarray
        0-based prescribed-node indices, length ``total_bc_energy``.
    prescribed_temp : numpy.ndarray
        Matching prescribed temperatures.
    total_bc_energy : int
        Number of prescribed nodes.
    """
    pad = pad_index
    temp_bc_index = np.zeros(energy_mesh.dim_xy, dtype=int)
    prescribed_temp = np.zeros(energy_mesh.dim_xy)
    layer = energy_mesh.total_e_y_pad + total_e_y_film + 1

    k = 0
    for i in range(energy_mesh.total_nodes):
        node = energy_mesh.n_index[i]
        # The layer arithmetic below counts nodes from 1, so use ``node + 1``.
        node1b = node + 1
        # Leading edge of the film.
        if (
            (energy_mesh.total_e_y_pad + 1)
            < node1b
            <= (energy_mesh.total_e_y_pad + total_e_y_film)
        ):
            m = energy_mesh.match_nodes_xy[node, total_e_z_film // 2 + 1 - 1]
            if velocity_x_n[pad, m] > 0.0:
                temp_bc_index[k] = node
                prescribed_temp[k] = temp_inlet[pad]
                k += 1
        # Journal surface.
        if node1b % layer == 0 and temp_j_type != "insulated_shaft_surface":
            temp_bc_index[k] = node
            prescribed_temp[k] = temp_j
            k += 1

    return temp_bc_index, prescribed_temp, k


def energy_coeffs_flooded(
    pad_index,
    energy_mesh,
    total_e_z_film,
    pad_thickness,
    lube,
    pad_conduct,
    axial_length,
    vis_effect_3d,
    conduct_effect,
    dudy_n,
    dwdy_n,
    velocity_x_n,
    velocity_y_n,
    z_3d,
    dim_z,
):
    """Nodal energy-equation coefficients for regular bearings.

    Assumes a constant axial temperature profile (``operating_type`` in
    {0, 3, 4, 5}). Distinguishes solid, film and
    film/pad-interface nodes; film coefficients are obtained by integrating
    the conduction/velocity/dissipation profiles axially across the film.

    Parameters
    ----------
    pad_index : int
        0-based pad index.
    energy_mesh : EnergyMesh
        Film+pad cross-section (x-y) mesh.
    total_e_z_film : int
        Axial film element count; ``total_e_z_film + 1`` samples are taken
        across the film for the axial integrals.
    pad_thickness : float
        Pad thickness, m; separates the solid, film and interface nodes.
    lube : Lubricant
        Lubricant properties.
    pad_conduct : float
        Pad thermal conductivity, W/(m*K).
    axial_length : numpy.ndarray
        Axial length per pad, shape ``(total_pads,)``, m.
    vis_effect_3d : numpy.ndarray
        Effective (laminar + eddy) viscosity at the 3-D nodes, shape
        ``(total_pads, dim_3d)``, Pa*s.
    conduct_effect : numpy.ndarray
        Effective heat conductivity at the 3-D nodes, same shape, W/(m*K).
    dudy_n, dwdy_n : numpy.ndarray
        Radial derivatives of the circumferential and axial velocity, same
        shape, 1/s.
    velocity_x_n, velocity_y_n : numpy.ndarray
        Circumferential and radial film velocity at the 3-D nodes, same
        shape, m/s.
    z_3d : numpy.ndarray
        Axial 3-D coordinate field, shape ``(total_pads, dim_3d)``; supplies
        the integration abscissae ``T``.
    dim_z : int
        Length of the axial scratch arrays used by the integrals.

    Returns
    -------
    kx_n, ky_n, mx_n, my_n, p_n, q_n : numpy.ndarray
        Per-node coefficient arrays of length ``dim_xy``.
    """
    return energy_coeffs_flooded_jit(
        pad_index,
        energy_mesh.total_nodes,
        np.ascontiguousarray(energy_mesh.n_index, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.match_nodes_xy, dtype=np.int64),
        total_e_z_film,
        float(pad_thickness),
        np.ascontiguousarray(energy_mesh.y, dtype=np.float64),
        float(lube.conduct),
        float(pad_conduct),
        float(lube.density),
        float(lube.cp),
        float(axial_length[pad_index]),
        np.ascontiguousarray(vis_effect_3d, dtype=np.float64),
        np.ascontiguousarray(conduct_effect, dtype=np.float64),
        np.ascontiguousarray(dudy_n, dtype=np.float64),
        np.ascontiguousarray(dwdy_n, dtype=np.float64),
        np.ascontiguousarray(velocity_x_n, dtype=np.float64),
        np.ascontiguousarray(velocity_y_n, dtype=np.float64),
        np.ascontiguousarray(z_3d, dtype=np.float64),
        energy_mesh.dim_xy,
        dim_z,
        DISSIP_FACTOR,
    )


def energy_coeffs_axial_flow(
    pad_index,
    energy_mesh,
    total_e_z_film,
    pad_thickness,
    lube,
    pad_conduct,
    axial_length,
    vis_effect_3d,
    conduct_effect,
    dudy_n,
    dwdy_n,
    velocity_x_n,
    velocity_y_n,
    velocity_z_n,
    temp_supply,
    z_3d,
    dim_z,
):
    """Nodal energy-equation coefficients for bearings with axial flow.

    Assumes a linear axial temperature profile
    (``operating_type == "axial_flow"``). The integrands are weighted by the axial
    coordinate; the axial convection contributes an extra source via the
    ``q1`` term involving the supply temperature.

    Returns
    -------
    kx_n, ky_n, mx_n, my_n, p_n, q_n : numpy.ndarray
        Per-node coefficient arrays of length ``dim_xy``.
    """
    pad = pad_index
    kx_n = np.zeros(energy_mesh.dim_xy)
    ky_n = np.zeros(energy_mesh.dim_xy)
    mx_n = np.zeros(energy_mesh.dim_xy)
    my_n = np.zeros(energy_mesh.dim_xy)
    p_n = np.zeros(energy_mesh.dim_xy)
    q_n = np.zeros(energy_mesh.dim_xy)

    t = np.zeros(dim_z)
    f1 = np.zeros(dim_z)
    f2 = np.zeros(dim_z)
    f3 = np.zeros(dim_z)
    f4 = np.zeros(dim_z)
    f5 = np.zeros(dim_z)

    limit = total_e_z_film + 1
    al = axial_length[pad]

    for i in range(energy_mesh.total_nodes):
        node = energy_mesh.n_index[i]
        y = energy_mesh.y[pad, node]

        if y < pad_thickness:
            kx_n[node] = -pad_conduct
            ky_n[node] = -pad_conduct
            mx_n[node] = 0.0
            my_n[node] = 0.0
            p_n[node] = 0.0
            q_n[node] = 0.0
        elif y > pad_thickness:
            kx_n[node] = -lube.conduct
            for j in range(1, limit + 1):
                m = energy_mesh.match_nodes_xy[node, j - 1]
                zz = z_3d[pad, m]
                t[j - 1] = zz
                f1[j - 1] = zz * conduct_effect[pad, m]
                f2[j - 1] = zz * velocity_x_n[pad, m]
                f3[j - 1] = zz * velocity_y_n[pad, m]
                f4[j - 1] = velocity_z_n[pad, m]
                f5[j - 1] = (
                    vis_effect_3d[pad, m]
                    * (dudy_n[pad, m] ** 2 + dwdy_n[pad, m] ** 2)
                    / DISSIP_FACTOR
                )
            ky_n[node] = -2.0 * _trapezoid(t, f1, 0, limit) / al**2
            mx_n[node] = (
                2.0 * lube.density * lube.cp * _trapezoid(t, f2, 0, limit) / al**2
            )
            my_n[node] = (
                2.0 * lube.density * lube.cp * _trapezoid(t, f3, 0, limit) / al**2
            )
            inte4 = _trapezoid(t, f4, 0, limit)
            p_n[node] = 2.0 * lube.density * lube.cp * abs(inte4) / al**2
            q1 = 2.0 * lube.density * lube.cp * temp_supply * abs(inte4) / al**2
            q_n[node] = -q1 - _trapezoid(t, f5, 0, limit) / al

        if abs(y - pad_thickness) < 1.0e-6:
            kx_n[node] = -(pad_conduct * lube.conduct) / (pad_conduct + lube.conduct)
            for j in range(1, limit + 1):
                m = energy_mesh.match_nodes_xy[node, j - 1]
                zz = z_3d[pad, m]
                ce = conduct_effect[pad, m]
                t[j - 1] = zz
                f1[j - 1] = zz * ((pad_conduct * ce) / (pad_conduct + ce))
                f2[j - 1] = zz * velocity_x_n[pad, m]
                f3[j - 1] = zz * velocity_y_n[pad, m]
                f4[j - 1] = velocity_z_n[pad, m]
                f5[j - 1] = (
                    vis_effect_3d[pad, m]
                    * (dudy_n[pad, m] ** 2 + dwdy_n[pad, m] ** 2)
                    / DISSIP_FACTOR
                )
            ky_n[node] = -2.0 * _trapezoid(t, f1, 0, limit) / al**2
            mx_n[node] = (
                2.0 * lube.density * lube.cp * _trapezoid(t, f2, 0, limit) / al**2
            )
            my_n[node] = (
                2.0 * lube.density * lube.cp * _trapezoid(t, f3, 0, limit) / al**2
            )
            inte4 = _trapezoid(t, f4, 0, limit)
            p_n[node] = 2.0 * lube.density * lube.cp * abs(inte4) / al**2
            q1 = 2.0 * lube.density * lube.cp * temp_supply * abs(inte4) / al**2
            q_n[node] = -q1 - _trapezoid(t, f5, 0, limit) / al

    return kx_n, ky_n, mx_n, my_n, p_n, q_n


def energy_coeffs_high_ambient(
    pad_index,
    energy_mesh,
    total_e_z_film,
    tz_type,
    pad_thickness,
    lube,
    pad_conduct,
    axial_length,
    vis_effect_3d,
    conduct_effect,
    dudy_n,
    dwdy_n,
    velocity_x_n,
    velocity_y_n,
    velocity_z_n,
    temp_supply,
    z_3d,
    dim_z,
):
    """Nodal energy-equation coefficients for high-ambient-pressure bearings.

    Used for ``operating_type == "high_ambient_pressure"``. Per node the
    axial profile is either constant (``tz_type == 0``, same form as
    :func:`energy_coeffs_flooded`) or parabolic (``tz_type != 0``).

    Returns
    -------
    kx_n, ky_n, mx_n, my_n, p_n, q_n : numpy.ndarray
        Per-node coefficient arrays of length ``dim_xy``.
    """
    pad = pad_index
    kx_n = np.zeros(energy_mesh.dim_xy)
    ky_n = np.zeros(energy_mesh.dim_xy)
    mx_n = np.zeros(energy_mesh.dim_xy)
    my_n = np.zeros(energy_mesh.dim_xy)
    p_n = np.zeros(energy_mesh.dim_xy)
    q_n = np.zeros(energy_mesh.dim_xy)

    t = np.zeros(dim_z)
    f1 = np.zeros(dim_z)
    f2 = np.zeros(dim_z)
    f3 = np.zeros(dim_z)
    f4 = np.zeros(dim_z)
    f5 = np.zeros(dim_z)

    limit = total_e_z_film + 1
    al = axial_length[pad]

    def _const(node, interface):
        """Constant-profile coefficients (interface uses harmonic kx)."""
        for j in range(1, limit + 1):
            m = energy_mesh.match_nodes_xy[node, j - 1]
            t[j - 1] = z_3d[pad, m]
            ce = conduct_effect[pad, m]
            if interface:
                f1[j - 1] = (pad_conduct * ce) / (pad_conduct + ce)
            else:
                f1[j - 1] = ce
            f2[j - 1] = velocity_x_n[pad, m]
            f3[j - 1] = velocity_y_n[pad, m]
            f4[j - 1] = (
                vis_effect_3d[pad, m]
                * (dudy_n[pad, m] ** 2 + dwdy_n[pad, m] ** 2)
                / DISSIP_FACTOR
            )
        ky_n[node] = -_trapezoid(t, f1, 0, limit) / al
        mx_n[node] = lube.density * lube.cp * _trapezoid(t, f2, 0, limit) / al
        my_n[node] = lube.density * lube.cp * _trapezoid(t, f3, 0, limit) / al
        p_n[node] = 0.0
        q_n[node] = -_trapezoid(t, f4, 0, limit) / al

    def _parab(node, interface):
        """Parabolic-profile coefficients."""
        if interface:
            kcond = (pad_conduct * lube.conduct) / (pad_conduct + lube.conduct)
        else:
            kcond = lube.conduct
        for j in range(1, limit + 1):
            m = energy_mesh.match_nodes_xy[node, j - 1]
            zz = z_3d[pad, m]
            t[j - 1] = zz
            shape = (zz / al) ** 2 - (zz / al)
            ce = conduct_effect[pad, m]
            if interface:
                f1[j - 1] = ((pad_conduct * ce) / (pad_conduct + ce)) * shape
            else:
                f1[j - 1] = shape * ce
            f2[j - 1] = shape * velocity_x_n[pad, m]
            f3[j - 1] = shape * velocity_y_n[pad, m]
            f4[j - 1] = ((2.0 * zz / al**2) - 1.0 / al) * velocity_z_n[pad, m]
            f5[j - 1] = (
                vis_effect_3d[pad, m]
                * (dudy_n[pad, m] ** 2 + dwdy_n[pad, m] ** 2)
                / DISSIP_FACTOR
            )
        ky_n[node] = 6.0 * _trapezoid(t, f1, 0, limit) / al
        mx_n[node] = -6.0 * lube.density * lube.cp * _trapezoid(t, f2, 0, limit) / al
        my_n[node] = -6.0 * lube.density * lube.cp * _trapezoid(t, f3, 0, limit) / al
        inte4 = _trapezoid(t, f4, 0, limit)
        p_n[node] = -6.0 * lube.density * lube.cp * inte4 / al + 12.0 * kcond / al**2
        q1 = (
            6.0 * temp_supply * lube.density * lube.cp * inte4 / al
            - 12.0 * kcond * temp_supply / al**2
        )
        q_n[node] = q1 - (3.0 * _trapezoid(t, f5, 0, limit)) / (2.0 * al)

    for i in range(energy_mesh.total_nodes):
        node = energy_mesh.n_index[i]
        y = energy_mesh.y[pad, node]

        if y < pad_thickness:
            kx_n[node] = -pad_conduct
            ky_n[node] = -pad_conduct
            mx_n[node] = 0.0
            my_n[node] = 0.0
            p_n[node] = 0.0
            q_n[node] = 0.0
        elif y > pad_thickness:
            kx_n[node] = -lube.conduct
            if tz_type[pad, node] == 0:
                _const(node, interface=False)
            else:
                _parab(node, interface=False)

        if abs(y - pad_thickness) < 1.0e-6:
            kx_n[node] = -(pad_conduct * lube.conduct) / (pad_conduct + lube.conduct)
            if tz_type[pad, node] == 0:
                _const(node, interface=True)
            else:
                _parab(node, interface=True)

    return kx_n, ky_n, mx_n, my_n, p_n, q_n


def integrand_e_temp(
    pad_index,
    current_element,
    energy_mesh,
    kx_e,
    ky_e,
    mx_e,
    my_e,
    p_e,
    q_e,
    r,
    s,
):
    """Integrand of the energy element matrix/column at a Gauss point.

    Builds the isoparametric shape functions, Jacobian, B matrix and the
    conduction (``B^T k B``), convection (``N^T V B``) and reaction (``P N^T
    N``) contributions at the natural-coordinate point ``(r, s)``.

    Returns
    -------
    integrand_e : numpy.ndarray
        Shape ``(4, 4)``.
    integrand_f : numpy.ndarray
        Length 4.
    """
    pad = pad_index
    ce = current_element
    n1 = energy_mesh.node_1[ce]
    n2 = energy_mesh.node_2[ce]
    n3 = energy_mesh.node_3[ce]
    n4 = energy_mesh.node_4[ce]

    gc = np.array(
        [
            [
                energy_mesh.x[pad, n1],
                energy_mesh.x[pad, n2],
                energy_mesh.x[pad, n3],
                energy_mesh.x[pad, n4],
            ],
            [
                energy_mesh.y[pad, n1],
                energy_mesh.y[pad, n2],
                energy_mesh.y[pad, n3],
                energy_mesh.y[pad, n4],
            ],
        ]
    )
    v = np.array([mx_e, my_e])

    n = np.array(
        [
            (1 - r) * (1 - s) / 4.0,
            (1 + r) * (1 - s) / 4.0,
            (1 + r) * (1 + s) / 4.0,
            (1 - r) * (1 + s) / 4.0,
        ]
    )
    f = np.array(
        [
            [-(1 - s) / 4.0, (1 - s) / 4.0, (1 + s) / 4.0, -(1 + s) / 4.0],
            [-(1 - r) / 4.0, -(1 + r) / 4.0, (1 + r) / 4.0, (1 - r) / 4.0],
        ]
    )

    jac = f @ gc.T
    det_j = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
    j_inv = np.array(
        [
            [jac[1, 1] / det_j, -jac[0, 1] / det_j],
            [-jac[1, 0] / det_j, jac[0, 0] / det_j],
        ]
    )

    b = j_inv @ f
    b_t = b.T

    kb = np.empty((2, 4))
    kb[0, :] = kx_e * b[0, :]
    kb[1, :] = ky_e * b[1, :]
    b_tkb = b_t @ kb

    n_tv = np.outer(n, v)
    n_tvb = n_tv @ b
    pn_tn = p_e * np.outer(n, n)

    integrand_e = (b_tkb - n_tvb - pn_tn) * det_j
    integrand_f = n * q_e * det_j
    return integrand_e, integrand_f


def integrand_line1(
    pad_index,
    current_element,
    energy_mesh,
    t_ambient,
    h,
    r,
    s,
):
    """Line (convection) integrand on an element edge at a Gauss point.

    ``h`` is the (negated) convection coefficient and ``t_ambient`` the ambient
    temperature.

    Returns
    -------
    integrand_el1 : numpy.ndarray
        Shape ``(4, 4)``.
    integrand_fl1 : numpy.ndarray
        Length 4.
    """
    pad = pad_index
    ce = current_element
    n1 = energy_mesh.node_1[ce]
    n2 = energy_mesh.node_2[ce]
    n3 = energy_mesh.node_3[ce]
    n4 = energy_mesh.node_4[ce]

    gc = np.array(
        [
            [
                energy_mesh.x[pad, n1],
                energy_mesh.x[pad, n2],
                energy_mesh.x[pad, n3],
                energy_mesh.x[pad, n4],
            ],
            [
                energy_mesh.y[pad, n1],
                energy_mesh.y[pad, n2],
                energy_mesh.y[pad, n3],
                energy_mesh.y[pad, n4],
            ],
        ]
    )

    n = np.array(
        [
            (1 - r) * (1 - s) / 4.0,
            (1 + r) * (1 - s) / 4.0,
            (1 + r) * (1 + s) / 4.0,
            (1 - r) * (1 + s) / 4.0,
        ]
    )
    f = np.array(
        [
            [-(1 - s) / 4.0, (1 - s) / 4.0, (1 + s) / 4.0, -(1 + s) / 4.0],
            [-(1 - r) / 4.0, -(1 + r) / 4.0, (1 + r) / 4.0, (1 - r) / 4.0],
        ]
    )
    jac = f @ gc.T

    if abs(r - 1.0) < 1.0e-6 or abs(r + 1.0) < 1.0e-6:
        dl = np.sqrt(jac[0, 0] ** 2 + jac[1, 1] ** 2)
    elif abs(s - 1.0) < 1.0e-6 or abs(s + 1.0) < 1.0e-6:
        dl = np.sqrt(jac[1, 0] ** 2 + jac[1, 1] ** 2)
    else:  # pragma: no cover - only edges are integrated
        dl = 0.0

    n_thn = h * np.outer(n, n)
    n_tht = h * t_ambient * n

    integrand_el1 = n_thn * dl
    integrand_fl1 = n_tht * dl
    return integrand_el1, integrand_fl1


def element_temp(
    pad_index,
    current_element,
    energy_mesh,
    kx_e,
    ky_e,
    mx_e,
    my_e,
    p_e,
    q_e,
    pad_length,
    pad_thickness,
    convec_edges,
    convec_back,
    t_ambient,
):
    """Element matrix/column of the full energy equation (2x2 Gauss + edges).

    Sums the 4 interior Gauss-point integrands and adds the leading-edge,
    trailing-edge and back convection line integrals where the element touches
    those boundaries.

    Returns
    -------
    e_matrix : numpy.ndarray
        Shape ``(4, 4)``.
    e_column : numpy.ndarray
        Length 4.
    """
    pad = pad_index
    ce = current_element
    a = 3.0
    g = 1.0 / np.sqrt(a)

    # Interior 2x2 Gauss quadrature: delegate to the JIT kernel that inlines
    # ``integrand_e_temp`` over all four Gauss points.
    e_matrix, e_column = element_temp_interior_jit(
        pad,
        ce,
        np.ascontiguousarray(energy_mesh.node_1, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_2, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_3, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_4, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.x, dtype=np.float64),
        np.ascontiguousarray(energy_mesh.y, dtype=np.float64),
        float(kx_e),
        float(ky_e),
        float(mx_e),
        float(my_e),
        float(p_e),
        float(q_e),
    )

    n1 = energy_mesh.node_1[ce]
    n3 = energy_mesh.node_3[ce]

    def _add_line(h, r_fixed=None, s_fixed=None):
        for g2 in (-g, g):
            r = r_fixed if r_fixed is not None else g2
            s = s_fixed if s_fixed is not None else g2
            iel, ifl = integrand_line1(
                pad_index,
                current_element,
                energy_mesh,
                t_ambient,
                h,
                r,
                s,
            )
            e_matrix[:] = e_matrix + iel
            e_column[:] = e_column + ifl

    # Leading edge of the pad (x = 0) in the solid.
    if abs(energy_mesh.x[pad, n1]) < 1.0e-6 and (
        energy_mesh.y[pad, n3] < pad_thickness
        or abs(energy_mesh.y[pad, n3] - pad_thickness) < 1.0e-6
    ):
        _add_line(-convec_edges, r_fixed=-1.0)

    # Trailing edge of the pad (x = pad_length) in the solid.
    if abs(energy_mesh.x[pad, n3] - pad_length[pad]) < 1.0e-6 and (
        energy_mesh.y[pad, n3] < pad_thickness
        or abs(energy_mesh.y[pad, n3] - pad_thickness) < 1.0e-6
    ):
        _add_line(-convec_edges, r_fixed=1.0)

    # Back of the pad (y = 0).
    if abs(energy_mesh.y[pad, n1]) < 1.0e-6:
        _add_line(-convec_back[pad], s_fixed=-1.0)

    return e_matrix, e_column


def temp_xy(
    temp_j_type,
    operating,
    tz_type,
    pad_index,
    pads,
    pad_conduct,
    lube,
    vis_effect_3d,
    velocity_x_n,
    velocity_y_n,
    velocity_z_n,
    conduct_effect,
    dudy_n,
    dwdy_n,
    total_e_y_film,
    energy_mesh,
    total_e_z_film,
    z_3d,
    temp_inlet,
    temp_j,
    t_ambient,
    convec_edges,
    convec_back,
    temp_full,
):
    """Solve the full 2-D energy equation on one pad's (x, y) energy mesh.

    Selects the nodal-coefficient routine by ``operating_type``, assembles and
    solves the (non-symmetric) condensed system, and writes the result into the
    ``pad_index`` row of ``temp_full``.

    Parameters
    ----------
    temp_j_type : str
        Journal-surface temperature treatment, one of
        :data:`~ross.bearings.fluid_film.constants.TEMP_J_TYPES`; passed through to
        :func:`temp_bc`, which skips the journal-surface BC for
        ``"insulated_shaft_surface"``.
    operating : OperatingPoint
        Speed and pressure conditions of the case; ``operating_type``
        selects the nodal-coefficient routine.
    tz_type : numpy.ndarray
        Per-node axial-profile flag (only used when
        ``operating_type == "high_ambient_pressure"``), shape
        ``(total_pads, dim_xy)``.
    pad_index : int
        0-based pad index being solved.
    pads : PadGeometry
        Per-pad geometry.
    pad_conduct : float
        Pad thermal conductivity, W/(m*K).
    lube : Lubricant
        Lubricant properties.
    vis_effect_3d : numpy.ndarray
        Effective (laminar + eddy) viscosity at the 3-D nodes, shape
        ``(total_pads, dim_3d)``, Pa*s.
    velocity_x_n, velocity_y_n, velocity_z_n : numpy.ndarray
        Circumferential, radial and axial film velocity at the 3-D nodes,
        same shape, m/s.
    conduct_effect : numpy.ndarray
        Effective heat conductivity at the 3-D nodes, same shape, W/(m*K).
    dudy_n, dwdy_n : numpy.ndarray
        Radial derivatives of the circumferential and axial velocity, same
        shape, 1/s.
    total_e_y_film : int
        Radial film element count, used to locate the film layers.
    energy_mesh : EnergyMesh
        Film+pad cross-section (x-y) mesh.
    total_e_z_film : int
        Axial film element count.
    z_3d : numpy.ndarray
        Axial 3-D coordinate field, shape ``(total_pads, dim_3d)``, m.
    temp_inlet : numpy.ndarray
        Pad inlet temperature, shape ``(total_pads,)``, K.
    temp_j : float
        Journal surface temperature, K.
    t_ambient : float
        Ambient temperature for the convection line integrals, K.
    convec_edges : float
        Convection coefficient on the pad leading / trailing edges,
        W/(m^2*K).
    convec_back : numpy.ndarray
        Convection coefficient on the pad back, per pad, W/(m^2*K).
    temp_full : numpy.ndarray
        2-D film+pad temperature field, shape ``(total_pads, dim_xy)``, K;
        updated for ``pad_index`` and returned.

    Returns
    -------
    numpy.ndarray
        ``temp_full`` with the ``pad_index`` row filled in.
    """
    pad = pad_index
    dim_xy = energy_mesh.x.shape[1]
    dim_z = energy_mesh.match_nodes_xy.shape[1]

    temp_bc_index, prescribed_temp, total_bc_energy = temp_bc(
        temp_j_type,
        pad_index,
        energy_mesh,
        total_e_y_film,
        total_e_z_film,
        temp_j,
        temp_inlet,
        velocity_x_n,
    )

    if operating.operating_type in (
        "regular_flooded",
        "starved_condition_even",
        "starved_condition_uneven",
        "oil_ring_lubricated",
    ):
        kx_n, ky_n, mx_n, my_n, p_n, q_n = energy_coeffs_flooded(
            pad_index,
            energy_mesh,
            total_e_z_film,
            pads.pad_thickness,
            lube,
            pad_conduct,
            pads.axial_length,
            vis_effect_3d,
            conduct_effect,
            dudy_n,
            dwdy_n,
            velocity_x_n,
            velocity_y_n,
            z_3d,
            dim_z,
        )
    elif operating.operating_type == "axial_flow":
        kx_n, ky_n, mx_n, my_n, p_n, q_n = energy_coeffs_axial_flow(
            pad_index,
            energy_mesh,
            total_e_z_film,
            pads.pad_thickness,
            lube,
            pad_conduct,
            pads.axial_length,
            vis_effect_3d,
            conduct_effect,
            dudy_n,
            dwdy_n,
            velocity_x_n,
            velocity_y_n,
            velocity_z_n,
            operating.temp_supply,
            z_3d,
            dim_z,
        )
    elif operating.operating_type == "high_ambient_pressure":
        kx_n, ky_n, mx_n, my_n, p_n, q_n = energy_coeffs_high_ambient(
            pad_index,
            energy_mesh,
            total_e_z_film,
            tz_type,
            pads.pad_thickness,
            lube,
            pad_conduct,
            pads.axial_length,
            vis_effect_3d,
            conduct_effect,
            dudy_n,
            dwdy_n,
            velocity_x_n,
            velocity_y_n,
            velocity_z_n,
            operating.temp_supply,
            z_3d,
            dim_z,
        )

    # Band storage: the energy system only touches ``2*bw - 1`` columns.
    global_matrix = np.zeros((dim_xy, 2 * energy_mesh.bandwidth - 1))
    global_column = np.zeros(dim_xy)

    # Interior 2x2 Gauss for every element in a single JIT call (fuses the
    # element_temp interior + the _assemble for every element).
    temp_xy_assemble_all_jit(
        pad,
        int(energy_mesh.total_elements),
        np.ascontiguousarray(energy_mesh.e_index, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_1, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_2, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_3, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_4, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.x, dtype=np.float64),
        np.ascontiguousarray(energy_mesh.y, dtype=np.float64),
        np.ascontiguousarray(kx_n, dtype=np.float64),
        np.ascontiguousarray(ky_n, dtype=np.float64),
        np.ascontiguousarray(mx_n, dtype=np.float64),
        np.ascontiguousarray(my_n, dtype=np.float64),
        np.ascontiguousarray(p_n, dtype=np.float64),
        np.ascontiguousarray(q_n, dtype=np.float64),
        float(pads.pad_thickness),
        int(energy_mesh.bandwidth),
        global_matrix,
        global_column,
    )

    # Boundary-line contributions: only the elements touching an LE / TE / back
    # edge contribute. The edge predicates, two-point Gauss line integrals
    # (``integrand_line1``) and assembly all run in a single JIT call; the
    # interior contribution is already in the global system.
    temp_xy_boundary_all_jit(
        pad,
        int(energy_mesh.total_elements),
        np.ascontiguousarray(energy_mesh.e_index, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_1, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_2, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_3, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.node_4, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.x, dtype=np.float64),
        np.ascontiguousarray(energy_mesh.y, dtype=np.float64),
        float(pads.pad_length[pad]),
        float(pads.pad_thickness),
        float(t_ambient),
        float(convec_edges),
        float(convec_back[pad]),
        int(energy_mesh.bandwidth),
        global_matrix,
        global_column,
    )

    _include_prescribed(
        global_matrix,
        global_column,
        energy_mesh.bandwidth,
        total_bc_energy,
        temp_bc_index,
        prescribed_temp,
        energy_mesh.total_nodes,
    )

    global_matrix, a_lower, index1, _d = banded.lu_factor(
        global_matrix, energy_mesh.total_nodes, energy_mesh.bandwidth
    )
    global_column = banded.lu_solve(
        global_matrix,
        energy_mesh.total_nodes,
        energy_mesh.bandwidth,
        a_lower,
        index1,
        global_column,
    )

    temp_full[pad, : energy_mesh.total_nodes] = global_column[: energy_mesh.total_nodes]

    return temp_full


# ===========================================================================
# Full driver
# ===========================================================================
def axial_profile(
    pad_index,
    mesh,
    energy_mesh,
    tz_type,
    axial_length,
    ambient_press1,
    nodal_pressure,
):
    """Choose constant vs parabolic axial profile per energy node.

    Used for high-ambient-pressure bearings. Below-ambient
    pressure along the centre line selects the parabolic profile
    (``tz_type = 1``); the choice is broadcast circumferentially onto the
    energy mesh.

    Returns
    -------
    numpy.ndarray
        Updated ``tz_type``, shape ``(total_pads, dim_xy)``.
    """
    pad = pad_index
    x = np.zeros(mesh.dim_x)
    tz = np.zeros(mesh.dim_x, dtype=int)

    k = 0
    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        if abs(mesh.z[pad, node] - 0.5 * axial_length[pad]) < 1.0e-6:
            x[k] = mesh.x[pad, node]
            if (
                nodal_pressure[pad, node] > ambient_press1
                or abs(nodal_pressure[pad, node] - ambient_press1) < 1.0e-6
            ):
                tz[k] = 0
            else:
                tz[k] = 1
            k += 1

    for i in range(1, mesh.total_e_x_film + 1 + 1):
        for j in range(energy_mesh.total_nodes):
            node = energy_mesh.n_index[j]
            if abs(energy_mesh.x[pad, node] - x[i - 1]) < 1.0e-6:
                tz_type[pad, node] = tz[i - 1]

    return tz_type


def effective_conduct(
    pad_index,
    mesh,
    lube,
    vis_n_3d,
    conduct_effect,
    vis_eddy_3d,
):
    """Effective heat conductivity including turbulent eddy transport.

    For laminar flow this equals the lubricant conductivity; for superlaminar
    flow the eddy-viscosity term is added.

    Returns
    -------
    numpy.ndarray
        Updated ``conduct_effect``, shape ``(total_pads, dim_3d)``.
    """
    pad = pad_index
    conduct_effect = np.ascontiguousarray(conduct_effect, dtype=np.float64)
    effective_conduct_jit(
        pad,
        mesh.total_nodes,
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        mesh.total_e_y_film,
        np.ascontiguousarray(mesh.match_nodes_xz, dtype=np.int64),
        float(lube.conduct),
        float(lube.cp),
        np.ascontiguousarray(vis_n_3d, dtype=np.float64),
        conduct_effect,
        np.ascontiguousarray(vis_eddy_3d, dtype=np.float64),
        PR_TURB,
    )
    return conduct_effect


def temp_average_full(
    pad_index, total_n_energy, n_index_energy, temp_full, temp_full_old
):
    """Store the old full temperature and return its pad average.

    Returns the updated ``temp_full_old`` and the average over the pad's active
    energy nodes.
    """
    pad = pad_index
    nodes = np.asarray(n_index_energy[:total_n_energy], dtype=np.intp)
    temp_full_old[pad, nodes] = temp_full[pad, nodes]
    return temp_full_old, temp_full_old[pad, nodes].sum() / total_n_energy


def expand_film_temp_flooded(
    pad_index,
    energy_mesh,
    total_e_z_film,
    pad_thickness,
    temp_full,
    temp_3d,
):
    """Expand the 2-D full temperature to 3-D film (constant axial profile).

    Returns the updated ``temp_3d``.
    """
    pad = pad_index
    temp_3d = np.ascontiguousarray(temp_3d, dtype=np.float64)
    expand_film_temp_flooded_jit(
        pad,
        energy_mesh.total_nodes,
        np.ascontiguousarray(energy_mesh.n_index, dtype=np.int64),
        np.ascontiguousarray(energy_mesh.match_nodes_xy, dtype=np.int64),
        total_e_z_film,
        np.ascontiguousarray(energy_mesh.y, dtype=np.float64),
        float(pad_thickness),
        np.ascontiguousarray(temp_full, dtype=np.float64),
        temp_3d,
    )
    return temp_3d


def expand_film_temp_axial_flow(
    pad_index,
    energy_mesh,
    total_e_z_film,
    n_index_reynolds,
    pad_thickness,
    nodal_pressure,
    axial_length,
    temp_supply,
    temp_full,
    temp_3d,
):
    """Expand the 2-D full temperature to 3-D film (linear axial profile).

    Used for bearings with axial flow. The known supply
    temperature may be on either axial end; the side is chosen from the
    nodal-pressure gradient.

    Returns
    -------
    numpy.ndarray
        Updated ``temp_3d``.
    """
    pad = pad_index
    step_length = axial_length[pad] / total_e_z_film
    al = axial_length[pad]

    p_first = nodal_pressure[pad, n_index_reynolds[0]]
    p_last = nodal_pressure[pad, n_index_reynolds[total_e_z_film + 1 - 1]]

    for i in range(energy_mesh.total_nodes):
        node = energy_mesh.n_index[i]
        y = energy_mesh.y[pad, node]
        if y > pad_thickness or abs(y - pad_thickness) < 1.0e-6:
            for j in range(1, total_e_z_film + 1 + 1):
                m = energy_mesh.match_nodes_xy[node, j - 1]
                if p_first > p_last:
                    temp_3d[pad, m] = (
                        2.0
                        * (temp_full[pad, node] - temp_supply)
                        * (j - 1)
                        * step_length
                        / al
                        + temp_supply
                    )
                else:
                    temp_3d[pad, m] = (
                        2.0
                        * (temp_full[pad, node] - temp_supply)
                        * (1 - ((j - 1) * step_length / al))
                        + temp_supply
                    )
    return temp_3d


def expand_film_temp_high_ambient(
    pad_index,
    energy_mesh,
    total_e_z_film,
    tz_type,
    pad_thickness,
    axial_length,
    temp_supply,
    temp_full,
    temp_3d,
):
    """Expand the 2-D full temperature to 3-D film (mixed const/parabolic).

    Used for high-ambient-pressure bearings. Constant in the
    convergent half (``tz_type == 0``), parabolic in the divergent half.

    Returns
    -------
    numpy.ndarray
        Updated ``temp_3d``.
    """
    pad = pad_index
    step_length = axial_length[pad] / total_e_z_film
    al = axial_length[pad]

    for i in range(energy_mesh.total_nodes):
        node = energy_mesh.n_index[i]
        y = energy_mesh.y[pad, node]
        if y > pad_thickness or abs(y - pad_thickness) < 1.0e-6:
            if tz_type[pad, node] == 0:
                for j in range(1, total_e_z_film + 1 + 1):
                    m = energy_mesh.match_nodes_xy[node, j - 1]
                    temp_3d[pad, m] = temp_full[pad, node]
            else:
                for j in range(1, total_e_z_film + 1 + 1):
                    m = energy_mesh.match_nodes_xy[node, j - 1]
                    frac = (j - 1) * step_length / al
                    temp_3d[pad, m] = (
                        4.0 * (temp_supply - temp_full[pad, node]) * (frac**2 - frac)
                        + temp_supply
                    )
    return temp_3d


def update_vis(
    pad_index,
    flow_regime_dam,
    mesh,
    lube,
    speed_surface,
    temp_3d,
    pad_thickness,
    vis_n_3d,
    vis_n_average,
    h_n,
    scale_turb_dam,
    turbulence,
):
    """Update viscosity, averaged viscosity and flow regime (full model).

    Used by the smooth-pad full thermal model. Updates the nodal
    3-D viscosity from the new temperature, the radially averaged viscosity
    and Reynolds number, and the dam flow regime / turbulence scaling factor.

    Returns
    -------
    vis_n_3d, vis_n_average : numpy.ndarray
        Updated viscosity fields.
    flow_regime_dam : numpy.ndarray
        Updated flow regime.
    scale_turb_dam : numpy.ndarray
        Updated turbulence scaling factor.
    """
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    y_3d = mesh.y_3d
    pad = pad_index
    dim_yf = match_nodes_xz.shape[1]
    dim_xz = mesh.x.shape[1]

    # Capture the np.ascontiguousarray results so the (possibly-copied) array
    # that the JIT mutates is the one we return -- otherwise a dtype mismatch
    # in the caller would silently drop the updates.
    vis_n_3d = np.ascontiguousarray(vis_n_3d, dtype=np.float64)
    vis_n_average = np.ascontiguousarray(vis_n_average, dtype=np.float64)
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)
    match_nodes_xz = np.ascontiguousarray(match_nodes_xz, dtype=np.int64)
    temp_3d = np.ascontiguousarray(temp_3d, dtype=np.float64)
    y_3d = np.ascontiguousarray(y_3d, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)

    re_max_dam = update_vis_jit(
        pad,
        mesh.total_nodes,
        n_index_reynolds,
        mesh.total_e_y_film,
        match_nodes_xz,
        float(lube.density),
        float(speed_surface),
        temp_3d,
        float(pad_thickness),
        y_3d,
        vis_n_3d,
        vis_n_average,
        h_n,
        float(lube.viscosity1),
        float(lube.viscosity2),
        float(lube.temp1),
        float(lube.temp2),
        dim_yf,
        dim_xz,
    )

    _update_flow_regime(
        flow_regime_dam,
        scale_turb_dam,
        pad,
        re_max_dam,
        turbulence,
    )

    return vis_n_3d, vis_n_average, flow_regime_dam, scale_turb_dam


def temp_full_residual(
    total_pads, total_n_energy, n_index_energy, temp_full1, temp_full_old
):
    """Root-mean-square change of the full 2-D temperature.

    Returns the RMS over all pads / active energy nodes of ``temp_full1 -
    temp_full_old``.
    """
    nodes = np.asarray(n_index_energy[:total_n_energy], dtype=np.intp)
    diff = temp_full1[:total_pads, nodes] - temp_full_old[:total_pads, nodes]
    return np.sqrt(np.sum(diff * diff) / (total_pads * total_n_energy))


def thermal_full(
    total_pads,
    operating,
    pads,
    pad_conduct,
    lube,
    vis_n_3d,
    vis_eddy_3d,
    vis_effect_3d,
    vis_n_average,
    mesh,
    energy_mesh,
    z_3d,
    flow_regime_dam,
    h_n,
    scale_turb_dam,
    temp_inlet,
    temp_j,
    t_ambient,
    convec_edges,
    convec_back,
    velocity_x_n,
    velocity_y_n,
    velocity_z_n,
    dudy_n,
    dwdy_n,
    nodal_pressure,
    temp_full,
    temp_3d,
    turbulence,
    relax_t_max,
    temp_j_type,
    conduct_effect=None,
    tz_type=None,
):
    """Full (conducting-pad) thermal solution for all pads (one iteration).

    Runs on the integrated film+pad mesh (smooth pads). For each
    pad it (1) selects the axial profile for high-ambient-pressure bearings,
    (2) computes the effective conductivity, (3) stores the old temperature,
    (4) solves the generalized 2-D energy equation, (5) under-relaxes to limit
    the average temperature change to 10 K (and ``relax_t_max``), (6)
    expands to the 3-D film with the appropriate axial profile and (7) updates
    the viscosity / flow regime. Returns the RMS temperature change.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    operating : OperatingPoint
        Speed and pressure conditions of the case; ``operating_type``
        selects the axial profile and the film-expansion routine.
    pads : PadGeometry
        Per-pad geometry.
    pad_conduct : float
        Pad thermal conductivity, W/(m*K).
    lube : Lubricant
        Lubricant properties.
    vis_n_3d : numpy.ndarray
        Laminar nodal viscosity at the 3-D nodes, shape
        ``(total_pads, dim_3d)``, Pa*s; updated.
    vis_eddy_3d : numpy.ndarray
        Eddy viscosity at the 3-D nodes, same shape, Pa*s.
    vis_effect_3d : numpy.ndarray
        Effective (laminar + eddy) viscosity at the 3-D nodes, same shape,
        Pa*s.
    vis_n_average : numpy.ndarray
        Radially averaged nodal viscosity, shape ``(total_pads, dim_xz)``,
        Pa*s; updated.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    energy_mesh : EnergyMesh
        Film+pad cross-section (x-y) mesh.
    z_3d : numpy.ndarray
        Axial 3-D coordinate field, shape ``(total_pads, dim_3d)``, m.
    flow_regime_dam : numpy.ndarray
        Per-pad dam flow-regime flag (0 laminar, 1 transition, 2 turbulent);
        updated.
    h_n : numpy.ndarray
        Nodal film thickness, shape ``(total_pads, dim_xz)``, m.
    scale_turb_dam : numpy.ndarray
        Per-pad dam turbulence scaling factor; updated with the regime.
    temp_inlet : numpy.ndarray
        Pad inlet temperature, shape ``(total_pads,)``, K.
    temp_j : float
        Journal surface temperature, K.
    t_ambient : float
        Ambient temperature for the convection line integrals, K.
    convec_edges : float
        Convection coefficient on the pad leading / trailing edges,
        W/(m^2*K).
    convec_back : numpy.ndarray
        Convection coefficient on the pad back, per pad, W/(m^2*K).
    velocity_x_n, velocity_y_n, velocity_z_n : numpy.ndarray
        Circumferential, radial and axial film velocity at the 3-D nodes,
        shape ``(total_pads, dim_3d)``, m/s.
    dudy_n, dwdy_n : numpy.ndarray
        Radial derivatives of the circumferential and axial velocity, same
        shape, 1/s.
    nodal_pressure : numpy.ndarray
        Nodal film pressure, shape ``(total_pads, dim_xz)``, Pa; only read
        for the axial-flow and high-ambient-pressure models.
    temp_full : numpy.ndarray
        2-D film+pad temperature field, shape ``(total_pads, dim_xy)``, K;
        the starting value and the under-relaxed result.
    temp_3d : numpy.ndarray
        3-D film temperature, shape ``(total_pads, dim_3d)``, K; updated.
    turbulence : Turbulence
        Turbulence-model constants.
    relax_t_max : float
        Upper bound on the temperature relaxation factor.
    temp_j_type : str
        Journal-surface temperature treatment, one of
        :data:`~ross.bearings.fluid_film.constants.TEMP_J_TYPES`; passed straight
        through to :func:`temp_xy` / :func:`temp_bc`, which omit the
        journal-surface BC for ``"insulated_shaft_surface"``.
    conduct_effect : numpy.ndarray, optional
        Scratch effective-conductivity field, shape ``(total_pads, dim_3d)``;
        allocated internally (zeros) if not supplied.
    tz_type : numpy.ndarray, optional
        Scratch per-node axial-profile flag, shape ``(total_pads, dim_xy)``;
        allocated internally (zeros) if not supplied.

    Returns
    -------
    temp_full : numpy.ndarray
        Updated (under-relaxed) 2-D film+pad temperature.
    temp_3d : numpy.ndarray
        Updated 3-D film temperature.
    vis_n_3d, vis_n_average : numpy.ndarray
        Updated viscosity fields.
    flow_regime_dam : numpy.ndarray
        Updated dam flow regime.
    scale_turb_dam : numpy.ndarray
        Updated dam turbulence scaling factor.
    rms_temp : float
        RMS temperature change between iterations.

    Notes
    -----
    Unlike :func:`thermal_adiabatic`, the full model updates only the *dam*
    flow regime; the track regime is left untouched. The same last-pad-scratch
    behaviour in the residual applies as noted for the adiabatic driver.
    """
    dim_3d = mesh.y_3d.shape[1]
    dim_xy = energy_mesh.x.shape[1]
    if conduct_effect is None:
        conduct_effect = np.zeros((total_pads, dim_3d))
    if tz_type is None:
        tz_type = np.zeros((total_pads, dim_xy), dtype=int)

    temp_full_old = temp_full.copy()
    temp_full1 = temp_full.copy()

    for pad_index in range(total_pads):
        pad = pad_index

        if operating.operating_type == "high_ambient_pressure":
            tz_type = axial_profile(
                pad_index,
                mesh,
                energy_mesh,
                tz_type,
                pads.axial_length,
                operating.ambient_press1,
                nodal_pressure,
            )

        conduct_effect = effective_conduct(
            pad_index,
            mesh,
            lube,
            vis_n_3d,
            conduct_effect,
            vis_eddy_3d,
        )

        temp_full_old, temp_average_old = temp_average_full(
            pad_index,
            energy_mesh.total_nodes,
            energy_mesh.n_index,
            temp_full,
            temp_full_old,
        )

        temp_full1 = temp_xy(
            temp_j_type,
            operating,
            tz_type,
            pad_index,
            pads,
            pad_conduct,
            lube,
            vis_effect_3d,
            velocity_x_n,
            velocity_y_n,
            velocity_z_n,
            conduct_effect,
            dudy_n,
            dwdy_n,
            mesh.total_e_y_film,
            energy_mesh,
            mesh.total_e_z_film,
            z_3d,
            temp_inlet,
            temp_j,
            t_ambient,
            convec_edges,
            convec_back,
            temp_full1,
        )

        total = 0.0
        for i in range(energy_mesh.total_nodes):
            total += temp_full1[pad, energy_mesh.n_index[i]]
        temp_average = total / energy_mesh.total_nodes

        if abs(temp_average - temp_average_old) > 10.0:
            relax_t = 10.0 / abs(temp_average - temp_average_old)
        else:
            relax_t = 1.0
        relax_t = min(relax_t, relax_t_max)
        for i in range(energy_mesh.total_nodes):
            node = energy_mesh.n_index[i]
            temp_full[pad, node] = (
                relax_t * temp_full1[pad, node]
                + (1.0 - relax_t) * temp_full_old[pad, node]
            )

        if operating.operating_type in (
            "regular_flooded",
            "starved_condition_even",
            "starved_condition_uneven",
            "oil_ring_lubricated",
        ):
            temp_3d = expand_film_temp_flooded(
                pad_index,
                energy_mesh,
                mesh.total_e_z_film,
                pads.pad_thickness,
                temp_full,
                temp_3d,
            )
        elif operating.operating_type == "axial_flow":
            temp_3d = expand_film_temp_axial_flow(
                pad_index,
                energy_mesh,
                mesh.total_e_z_film,
                mesh.n_index,
                pads.pad_thickness,
                nodal_pressure,
                pads.axial_length,
                operating.temp_supply,
                temp_full,
                temp_3d,
            )
        elif operating.operating_type == "high_ambient_pressure":
            temp_3d = expand_film_temp_high_ambient(
                pad_index,
                energy_mesh,
                mesh.total_e_z_film,
                tz_type,
                pads.pad_thickness,
                pads.axial_length,
                operating.temp_supply,
                temp_full,
                temp_3d,
            )

        (vis_n_3d, vis_n_average, flow_regime_dam, scale_turb_dam) = update_vis(
            pad_index,
            flow_regime_dam,
            mesh,
            lube,
            operating.speed_surface,
            temp_3d,
            pads.pad_thickness,
            vis_n_3d,
            vis_n_average,
            h_n,
            scale_turb_dam,
            turbulence,
        )

    rms_temp = temp_full_residual(
        total_pads,
        energy_mesh.total_nodes,
        energy_mesh.n_index,
        temp_full1,
        temp_full_old,
    )

    return (
        temp_full,
        temp_3d,
        vis_n_3d,
        vis_n_average,
        flow_regime_dam,
        scale_turb_dam,
        rms_temp,
    )
