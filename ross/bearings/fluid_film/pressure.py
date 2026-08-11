"""Finite-element solve of the Reynolds equation for the film pressure.

Solves the pressure distribution on a single pad using the generalized
Reynolds equation (Safar & Szeri formulation), accounting for a variable
cross-film viscosity and turbulence -- both folded into the supplied
``vis_effect_3d`` field. :func:`press` is the driver; the rest of the module
is the helpers it calls.

Indexing / data-structure convention
-------------------------------------
Every mesh-coupled array is 0-based, matching what
:func:`ross.bearings.fluid_film.mesh.mesh_reynolds` produces. The stored node and
element numbers (``mesh.node_i[e]``, ``mesh.n_index[i]``, ...) index the
coordinate and field arrays directly. Concretely:

* 1-D index/coordinate arrays (``mesh.n_index``, ``mesh.node_i`` and the
  other connectivity arrays, ``mesh.e_index``, ``film_onset``, ...) have
  length ``dim_xz`` (or ``total_pads``);
* 2-D per-pad arrays (``mesh.x``, ``mesh.z``, ``h_n``, ``nodal_pressure``,
  ``dpdx_n``, ``dpdz_n``, ``mesh.e_length``, ``mesh.e_width``, ...) have
  shape ``(total_pads, dim_xz)`` and are indexed ``[pad_index, node]``;
* ``mesh.y_3d`` and ``vis_effect_3d`` have shape ``(total_pads, dim_3d)``;
* ``mesh.match_nodes_xz`` has shape ``(dim_xz, dim_yf)`` and stores 3-D node
  numbers, with ``-1`` marking an unused slot;
* ``mesh.dx`` has shape ``(total_pads, dim_xz, 4)``, the trailing axis
  holding the four derivative components.

The banded global system (``global_matrix_p``, ``a_lower``) is stored with
shape ``(dim_xz, 2 * bandwidth - 1)``; the band diagonal sits at column
``mesh.bandwidth - 1``. The Reynolds cavitation clamp is applied during
back substitution, in :func:`ross.bearings.fluid_film.banded.lu_solve_cavitating`.

What the caller must provide
----------------------------
:func:`press` owns no state. The caller passes, with the conventions above:

* the mesh, as a :class:`~ross.bearings.fluid_film.state.ReynoldsMesh` built by
  ``mesh_reynolds`` / ``mesh_3d``, and the pad shapes as a
  :class:`~ross.bearings.fluid_film.state.PadGeometry`;
* the current film thickness ``h_n`` and the cross-film viscosity-effect
  field ``vis_effect_3d`` from the hydrodynamics layer;
* the case conditions as an
  :class:`~ross.bearings.fluid_film.state.OperatingPoint` (surface speed, ambient
  and cavitation pressures, bearing and operating type), together with the
  per-pad cavitation state ``film_onset``, ``h_min``, ``x_hmin`` and
  ``full_cavitate``.

:func:`press` returns the updated ``nodal_pressure``, ``dpdx_n`` and
``dpdz_n`` rather than mutating them in place.

References
----------
.. [1] Safar, Z., & Szeri, A. Z. (1974). Thermohydrodynamic lubrication in
       laminar and turbulent regimes. ASME Journal of Lubrication
       Technology, 96(1), 48-56.
.. [2] Allaire, P. E., Nicholas, J. C., & Gunter, E. J. (1977). Systems of
       finite elements for finite bearings. ASME Journal of Lubrication
       Technology, 99(2), 187-197.
.. [3] Swift, H. W. (1932). The stability of lubricating films in journal
       bearings. Minutes of Proceedings of the Institution of Civil
       Engineers, 233, 267-288. (Together with Stieber's 1933 Das
       Schwimmlager, the origin of the zero-gradient cavitation boundary
       condition applied here.)
"""

import numpy as np

from ross.bearings.fluid_film import banded
from ross.bearings.fluid_film._numba_kernels import (
    assemble_press_jit,
    element_press_jit,
    gamma_g_loop_jit,
    include_press_jit,
    press_assemble_all_jit,
    press_gradient_node_jit,
)
from ross.bearings.fluid_film.constants import PI


def press_bc(
    mesh,
    operating,
    pad_index,
    pads,
    film_onset,
):
    """Build the prescribed pressure boundary conditions for a pad.

    Sets the pressure on the four pad edges and (when starvation begins
    downstream of the film onset) on the leading edge. For a flooded bearing
    all edge pressures are zero; for the ``"high_ambient_pressure"`` and
    ``"axial_flow"`` operating types the leading/trailing edges vary linearly
    between the two ambient pressures. The Reynolds cavitation condition itself
    is applied later, during back substitution in
    :func:`ross.bearings.fluid_film.banded.lu_solve_cavitating`.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
    operating : OperatingPoint
        Speed and pressure conditions of the case.
    pad_index : int
        0-based pad number.
    pads : PadGeometry
        Per-pad geometry.
    film_onset : array_like of int
        Per-pad film-onset element row.

    Returns
    -------
    total_bc_reynolds : int
        Number of prescribed boundary conditions found.
    press_bc_index : numpy.ndarray of int
        Prescribed node ids (0-based), length ``dim_xz``.
    prescribed_press : numpy.ndarray of float
        Prescribed nodal pressures, length ``dim_xz``, Pa.
    """
    x_reynolds = mesh.x
    z_reynolds = mesh.z
    x_reynolds = np.asarray(x_reynolds, dtype=float)
    z_reynolds = np.asarray(z_reynolds, dtype=float)

    press_bc_index = np.zeros(mesh.dim_xz, dtype=np.int64)
    prescribed_press = np.zeros(mesh.dim_xz, dtype=float)

    p = pad_index
    is_360 = abs(pads.arc_length_rad[0] - 2.0 * PI) < 1.0e-6

    # Vectorized over the node scan: each node hits at most the first matching
    # branch (if/elif precedence), and matches are emitted in scan order, so
    # boolean masking on the ordered node list reproduces the sequential fill.
    idx = np.asarray(mesh.n_index[: mesh.total_nodes], dtype=np.int64)
    z = z_reynolds[p, idx]
    x = x_reynolds[p, idx]

    # The four edge predicates, with if/elif precedence.
    on_edge1 = np.abs(z) < 1.0e-6
    on_edge2 = ~on_edge1 & (np.abs(z - pads.axial_length[p]) < 1.0e-6)
    before_onset = (
        ~on_edge1 & ~on_edge2 & (idx < (film_onset[p] + 1) * (mesh.total_e_z_film + 1))
    )
    on_te = (
        ~on_edge1
        & ~on_edge2
        & ~before_onset
        & (np.abs(x - pads.pad_length[p]) < 1.0e-6)
    )

    # Leading/trailing edges are skipped for a 360-degree pad
    # ``goto 100``); the axial edges always prescribe.
    le_te = (before_onset | on_te) & (not is_360)
    selected = on_edge1 | on_edge2 | le_te

    if operating.operating_type in ("axial_flow", "high_ambient_pressure"):
        edge1_press = operating.ambient_press1
        edge2_press = operating.ambient_press2
        le_te_press = (
            z * (operating.ambient_press2 - operating.ambient_press1)
        ) / pads.axial_length[p] + operating.ambient_press1
    else:
        # regular_flooded / starved / oil-ring: zero everywhere except the
        # axial edges of the non-flooded cases.
        edge1_press = (
            0.0
            if operating.operating_type == "regular_flooded"
            else operating.ambient_press1
        )
        edge2_press = (
            0.0
            if operating.operating_type == "regular_flooded"
            else operating.ambient_press2
        )
        le_te_press = np.zeros_like(z)

    values = np.select(
        [on_edge1, on_edge2, le_te],
        [np.full_like(z, edge1_press), np.full_like(z, edge2_press), le_te_press],
    )

    total_bc_reynolds = int(np.count_nonzero(selected))
    press_bc_index[:total_bc_reynolds] = idx[selected]
    prescribed_press[:total_bc_reynolds] = values[selected]
    return total_bc_reynolds, press_bc_index, prescribed_press


def gamma_g(
    mesh,
    pad_index,
    pads,
    h_n,
    vis_effect_3d,
):
    """Compute the ``Gamma`` and ``G`` cross-film functions at every node.

    ``Gamma`` and ``G`` collect the cross-film viscosity integrals (the
    ``Xi1``/``Xi2`` quantities in the Safar-Szeri generalized Reynolds
    equation) using the trapezoidal rule over the through-film node column
    matched to each Reynolds node. Pocket, dam and pocket-edge nodes use
    different integration limits.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
        Supplies the ``match_nodes_xz`` Reynolds-node -> 3-D-node map (shape
        ``(dim_xz, dim_yf)``) and the ``y_3d`` radial coordinates.
    pad_index : int
        0-based pad number.
    pads : PadGeometry
        Per-pad geometry, including the pressure-dam pocket dimensions that
        set the integration limits.
    h_n : array_like of float
        Film thickness at each Reynolds node, shape
        ``(total_pads, dim_xz)``, m.
    vis_effect_3d : array_like of float
        Cross-film viscosity-effect field, shape ``(total_pads, dim_3d)``.

    Returns
    -------
    gamma : numpy.ndarray of float
        ``Gamma`` at each node, length ``dim_xz``.
    g : numpy.ndarray of float
        ``G`` at each node, length ``dim_xz``.
    """
    match_nodes_xz = mesh.match_nodes_xz
    n_index_reynolds = mesh.n_index
    x_reynolds = mesh.x
    y_3d = mesh.y_3d
    z_reynolds = mesh.z
    x_reynolds = np.ascontiguousarray(x_reynolds, dtype=np.float64)
    z_reynolds = np.ascontiguousarray(z_reynolds, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)
    y_3d = np.ascontiguousarray(y_3d, dtype=np.float64)
    vis_effect_3d = np.ascontiguousarray(vis_effect_3d, dtype=np.float64)
    match_nodes_xz = np.ascontiguousarray(match_nodes_xz, dtype=np.int64)
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)

    gamma = np.zeros(mesh.dim_xz, dtype=np.float64)
    g = np.zeros(mesh.dim_xz, dtype=np.float64)

    p = pad_index

    # Drive the per-node cross-film integration via the 0-based JIT loop.
    gamma_g_loop_jit(
        p,
        mesh.total_nodes,
        n_index_reynolds,
        mesh.total_e_y_film,
        match_nodes_xz,
        int(mesh.total_e_y_trackbl[p]),
        int(mesh.total_e_y_trackcore[p]),
        float(pads.pad_thickness),
        float(pads.pad_length[p]),
        float(pads.axial_length[p]),
        float(pads.depth_track[p]),
        float(pads.length_track[p]),
        float(pads.axial_length_track[p]),
        float(pads.axial_length_dam[p]),
        x_reynolds,
        z_reynolds,
        y_3d,
        h_n,
        vis_effect_3d,
        gamma,
        g,
    )
    return gamma, g


def zero_pressure_system(dim_xz, total_column_reynolds):
    """Allocate/zero the banded global matrix and the global column.

    Returns freshly zeroed 0-based arrays rather than mutating in place; the
    caller assigns them. The full band (all stored non-zero entries) is zeroed.

    Parameters
    ----------
    dim_xz : int
        Declared dimension.
    total_column_reynolds : int
        Number of band columns used (``2 * bandwidth_reynolds - 1``).

    Returns
    -------
    global_matrix_p : numpy.ndarray
        Zeroed ``(dim_xz, total_column_reynolds)`` banded matrix (0-based).
    global_column_p : numpy.ndarray
        Zeroed ``(dim_xz,)`` column vector (0-based).
    """
    global_matrix_p = np.zeros((dim_xz, total_column_reynolds), dtype=float)
    global_column_p = np.zeros(dim_xz, dtype=float)
    return global_matrix_p, global_column_p


def element_press(k_x, k_z, q, l_e, w_e):
    """Build the 4-node element matrix and column of the Reynolds equation.

    The closed-form Q4 element stiffness for the 2nd-order Reynolds operator,
    after Allaire.

    Parameters
    ----------
    k_x, k_z : float
        Circumferential / axial diffusion coefficients
        (``h_e**3 * gamma_e``); ``k_x`` is forced to zero for a 360-degree pad.
    q : float
        Source term (``speed_surface * g_e * dhdx_e``).
    l_e, w_e : float
        Element circumferential length and axial width.

    Returns
    -------
    e_matrix_reynolds : numpy.ndarray, shape (4, 4)
        Element matrix.
    e_column_reynolds : numpy.ndarray, shape (4,)
        Element column.
    """
    return element_press_jit(float(k_x), float(k_z), float(q), float(l_e), float(w_e))


def assemble_press(
    e_matrix_reynolds,
    e_column_reynolds,
    local_coordinates,
    bandwidth_reynolds,
    global_matrix_p,
    global_column_p,
):
    """Assemble a Q4 element into the banded global system.

    ``global_matrix_p`` is stored in banded storage (0-based): row
    ``irow`` (0-based node id), band column ``jcol`` (0-based, diagonal at
    ``bandwidth_reynolds - 1``).

    Parameters
    ----------
    e_matrix_reynolds : array_like, shape (4, 4)
        Element matrix (local slots 0..3 for the four element corners).
    e_column_reynolds : array_like, shape (4,)
        Element column.
    local_coordinates : array_like of int, length 4
        0-based global node ids of the element's four local corners; indexed
        by the local corner slot ``0..3``.
    bandwidth_reynolds : int
        Half-bandwidth of the band storage.
    global_matrix_p : array_like, shape (dim_xz, total_column)
        Banded global matrix (a mutated copy is returned).
    global_column_p : array_like, shape (dim_xz,)
        Global column (a mutated copy is returned).

    Returns
    -------
    global_matrix_p : numpy.ndarray
        Updated banded global matrix.
    global_column_p : numpy.ndarray
        Updated global column.
    """
    e_matrix_reynolds = np.ascontiguousarray(e_matrix_reynolds, dtype=np.float64)
    e_column_reynolds = np.ascontiguousarray(e_column_reynolds, dtype=np.float64)
    global_matrix_p = np.ascontiguousarray(global_matrix_p, dtype=np.float64)
    global_column_p = np.ascontiguousarray(global_column_p, dtype=np.float64)
    local_coordinates = np.ascontiguousarray(local_coordinates, dtype=np.int64)

    # ``assemble_press_jit`` expects a length-4 connectivity of node numbers
    # and a band matrix.
    if local_coordinates.shape[0] == 5:
        local_coordinates = local_coordinates[1:]
    return assemble_press_jit(
        e_matrix_reynolds,
        e_column_reynolds,
        local_coordinates,
        bandwidth_reynolds,
        global_matrix_p,
        global_column_p,
    )


def include_press(
    global_matrix_p,
    global_column_p,
    bandwidth_reynolds,
    total_bc_reynolds,
    press_bc_index,
    prescribed_press,
    total_n_reynolds,
):
    """Impose prescribed nodal pressures on the banded global system.

    For each constrained node the coupled band
    entries are moved to the right-hand side and zeroed, leaving the diagonal so
    the node equals its prescribed pressure.

    Parameters
    ----------
    global_matrix_p : array_like, shape (dim_xz, total_column)
        Banded global matrix (a mutated copy is returned).
    global_column_p : array_like, shape (dim_xz,)
        Global column (a mutated copy is returned).
    bandwidth_reynolds : int
        Half-bandwidth of the band storage.
    total_bc_reynolds : int
        Number of prescribed boundary conditions.
    press_bc_index : array_like of int
        Constrained node ids (0-based), length ``dim_xz``.
    prescribed_press : array_like of float
        Prescribed nodal pressures, length ``dim_xz``.
    total_n_reynolds : int
        Number of active rows.

    Returns
    -------
    global_matrix_p : numpy.ndarray
        Updated banded global matrix.
    global_column_p : numpy.ndarray
        Updated global column.
    """
    global_matrix_p = np.ascontiguousarray(global_matrix_p, dtype=np.float64)
    global_column_p = np.ascontiguousarray(global_column_p, dtype=np.float64)
    press_bc_index = np.ascontiguousarray(press_bc_index, dtype=np.int64)
    prescribed_press = np.ascontiguousarray(prescribed_press, dtype=np.float64)

    # ``include_press_jit`` operates on a 0-based band matrix and indexes the
    # constrained rows from 0-based node ids stored in ``press_bc_index``.
    return include_press_jit(
        global_matrix_p,
        global_column_p,
        bandwidth_reynolds,
        total_bc_reynolds,
        press_bc_index[:total_bc_reynolds],
        prescribed_press[:total_bc_reynolds],
        total_n_reynolds,
    )


def press_gradient_node(
    pad_index,
    mesh,
    nodal_pressure,
    axial_length,
    dpdx_n,
    dpdz_n,
):
    """Compute nodal pressure gradients (general, second-order one-sided/central).

    The circumferential derivative uses a three-point backward difference on
    the last two circumferential layers and a three-point forward difference
    elsewhere; the axial derivative uses a three-point forward/backward
    difference depending on the axial half, and a three-point central
    difference on the centerline.

    Parameters
    ----------
    pad_index : int
        0-based pad number.
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
    nodal_pressure : array_like of float
        Nodal pressure, shape ``(total_pads, dim_xz)``, Pa.
    axial_length : array_like of float
        Per-pad axial length (shape ``(total_pads,)``), m.
    dpdx_n, dpdz_n : array_like of float
        Output gradient arrays, shape ``(total_pads, dim_xz)`` (mutated
        copies returned), Pa/m.

    Returns
    -------
    dpdx_n : numpy.ndarray
        Circumferential pressure gradient, shape ``(total_pads, dim_xz)``.
    dpdz_n : numpy.ndarray
        Axial pressure gradient, shape ``(total_pads, dim_xz)``.
    """
    n_index_reynolds = mesh.n_index
    x_reynolds = mesh.x
    z_reynolds = mesh.z
    nodal_pressure = np.asarray(nodal_pressure, dtype=float)
    x_reynolds = np.asarray(x_reynolds, dtype=float)
    z_reynolds = np.asarray(z_reynolds, dtype=float)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)

    p = pad_index
    n_index_reynolds = np.ascontiguousarray(n_index_reynolds, dtype=np.int64)
    axial_length = np.ascontiguousarray(axial_length, dtype=np.float64)
    press_gradient_node_jit(
        p,
        mesh.total_nodes,
        n_index_reynolds,
        nodal_pressure,
        axial_length,
        x_reynolds,
        z_reynolds,
        mesh.total_e_x_film,
        mesh.total_e_z_film,
        dpdx_n,
        dpdz_n,
    )
    return dpdx_n, dpdz_n


def press_gradient_node_dam(
    pad_index,
    mesh,
    nodal_pressure,
    axial_length,
    dpdx_n,
    dpdz_n,
):
    """Compute nodal pressure gradients with simple two-point differences.

    Used for ``bearing_type == "pressure_dam"`` and the oil
    seal. The circumferential derivative is a two-point forward difference on
    the leading edge and a two-point backward difference elsewhere; the axial
    derivative is a two-point forward/backward difference by axial half and a
    two-point central difference on the centerline.

    Parameters
    ----------
    pad_index : int
        0-based pad number.
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
    nodal_pressure : array_like of float
        Nodal pressure, shape ``(total_pads, dim_xz)``, Pa.
    axial_length : array_like of float
        Per-pad axial length (shape ``(total_pads,)``), m.
    dpdx_n, dpdz_n : array_like of float
        Output gradient arrays, shape ``(total_pads, dim_xz)`` (mutated
        copies returned), Pa/m.

    Returns
    -------
    dpdx_n : numpy.ndarray
        Circumferential pressure gradient.
    dpdz_n : numpy.ndarray
        Axial pressure gradient.
    """
    x_reynolds = mesh.x
    z_reynolds = mesh.z
    nodal_pressure = np.asarray(nodal_pressure, dtype=float)
    x_reynolds = np.asarray(x_reynolds, dtype=float)
    z_reynolds = np.asarray(z_reynolds, dtype=float)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)

    p = pad_index
    pr = nodal_pressure
    x = x_reynolds
    z = z_reynolds
    step = mesh.total_e_z_film + 1

    for i in range(mesh.total_nodes):
        node = int(mesh.n_index[i])

        # Circumferential derivative.
        if abs(x[p, node]) < 1.0e-6:
            # Leading edge: two-point forward difference.
            nf = node + step
            dpdx_n[p, node] = (-pr[p, node] + pr[p, nf]) / (x[p, nf] - x[p, node])
        else:
            # Two-point backward difference.
            nb = node - step
            dpdx_n[p, node] = (pr[p, node] - pr[p, nb]) / (x[p, node] - x[p, nb])

        # Axial derivative.
        if z[p, node] < 0.5 * axial_length[p]:
            nf = node + 1
            dpdz_n[p, node] = (-pr[p, node] + pr[p, nf]) / (z[p, nf] - z[p, node])
        elif z[p, node] > 0.5 * axial_length[p]:
            nb = node - 1
            dpdz_n[p, node] = (pr[p, node] - pr[p, nb]) / (z[p, node] - z[p, nb])
        else:
            nf = node + 1
            nb = node - 1
            dpdz_n[p, node] = (pr[p, nf] - pr[p, nb]) / (z[p, nf] - z[p, nb])

    return dpdx_n, dpdz_n


def press(
    mesh,
    operating,
    pad_index,
    pads,
    vis_effect_3d,
    film_onset,
    h_n,
    h_min,
    x_hmin,
    nodal_pressure,
    dpdx_n,
    dpdz_n,
    full_cavitate,
):
    """Solve the generalized-Reynolds pressure distribution on one pad.

    The sequence is: apply the
    pressure boundary conditions (:func:`press_bc`), evaluate the cross-film
    ``Gamma``/``G`` functions (:func:`gamma_g`), assemble the banded FE system
    element-by-element (:func:`element_press`, :func:`assemble_press`), impose
    the prescribed pressures (:func:`include_press`), solve with the banded LU
    that also enforces the Reynolds cavitation floor
    (:mod:`ross.bearings.fluid_film.banded`), apply the various cavitation /
    full-cavitation corrections, and finally compute the nodal pressure
    gradients (:func:`press_gradient_node` or :func:`press_gradient_node_dam`).

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps
        (see the module docstring for the array shapes).
    operating : OperatingPoint
        Speed and pressure conditions of the case. ``bearing_type ==
        "pressure_dam"`` (the oil seal) selects the simple gradient and an
        alternate cavitation rule.
    pad_index : int
        0-based pad number.
    pads : PadGeometry
        Per-pad geometry.
    vis_effect_3d : array_like of float
        Cross-film viscosity-effect field, shape ``(total_pads, dim_3d)``.
    film_onset : array_like of int
        Per-pad film-onset element row.
    h_n : array_like of float
        Film thickness at each Reynolds node, shape
        ``(total_pads, dim_xz)``, m.
    h_min, x_hmin : array_like of float
        Per-pad minimum film thickness and its circumferential location
        (shape ``(total_pads,)``), m.
    nodal_pressure, dpdx_n, dpdz_n : array_like of float
        Output pressure and gradient fields, shape
        ``(total_pads, dim_xz)`` (mutated copies returned), in Pa and Pa/m.
    full_cavitate : array_like of bool
        Per-pad full-cavitation flag (shape ``(total_pads,)``).

    Returns
    -------
    nodal_pressure : numpy.ndarray
        Updated nodal pressure, shape ``(total_pads, dim_xz)``.
    dpdx_n : numpy.ndarray
        Circumferential pressure gradient, same shape.
    dpdz_n : numpy.ndarray
        Axial pressure gradient, same shape.
    """
    arc_length_rad = pads.arc_length_rad
    dx_reynolds = mesh.dx
    e_length_reynolds = mesh.e_length
    e_width_reynolds = mesh.e_width
    x_reynolds = mesh.x
    z_reynolds = mesh.z
    arc_length_rad = np.asarray(arc_length_rad, dtype=float)
    x_reynolds = np.asarray(x_reynolds, dtype=float)
    z_reynolds = np.asarray(z_reynolds, dtype=float)
    e_length_reynolds = np.asarray(e_length_reynolds, dtype=float)
    e_width_reynolds = np.asarray(e_width_reynolds, dtype=float)
    dx_reynolds = np.asarray(dx_reynolds, dtype=float)
    h_n = np.asarray(h_n, dtype=float)
    nodal_pressure = np.ascontiguousarray(nodal_pressure, dtype=np.float64)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)

    p = pad_index
    total_column_reynolds = 2 * mesh.bandwidth - 1

    # Apply the pressure boundary conditions.
    total_bc_reynolds, press_bc_index, prescribed_press = press_bc(
        mesh,
        operating,
        pad_index,
        pads,
        film_onset,
    )

    # Cross-film Gamma and G functions at each node.
    gamma, g = gamma_g(
        mesh,
        pad_index,
        pads,
        h_n,
        vis_effect_3d,
    )

    # Zero the global system.
    global_matrix_p, global_column_p = zero_pressure_system(
        mesh.dim_xz, total_column_reynolds
    )

    is_360 = abs(arc_length_rad[0] - 2.0 * PI) < 1.0e-6

    # Assemble every element in a single JIT call (eliminates ~14k
    # per-element Python<->JIT crossings on the full mesh).
    press_assemble_all_jit(
        p,
        mesh.total_elements,
        np.ascontiguousarray(mesh.e_index, dtype=np.int64),
        np.ascontiguousarray(mesh.node_i, dtype=np.int64),
        np.ascontiguousarray(mesh.node_j, dtype=np.int64),
        np.ascontiguousarray(mesh.node_k, dtype=np.int64),
        np.ascontiguousarray(mesh.node_l, dtype=np.int64),
        np.ascontiguousarray(gamma, dtype=np.float64),
        np.ascontiguousarray(g, dtype=np.float64),
        h_n,
        np.ascontiguousarray(dx_reynolds, dtype=np.float64),
        e_length_reynolds,
        e_width_reynolds,
        float(operating.speed_surface),
        bool(is_360),
        int(mesh.bandwidth),
        global_matrix_p,
        global_column_p,
    )

    # Include the prescribed nodal pressures.
    global_matrix_p, global_column_p = include_press(
        global_matrix_p,
        global_column_p,
        mesh.bandwidth,
        total_bc_reynolds,
        press_bc_index,
        prescribed_press,
        mesh.total_nodes,
    )

    # Banded LU solve with the Reynolds cavitation floor.
    global_matrix_p, a_lower, index1, _d = banded.lu_factor(
        global_matrix_p, mesh.total_nodes, mesh.bandwidth
    )
    global_column_p = banded.lu_solve_cavitating(
        global_matrix_p,
        mesh.total_nodes,
        mesh.bandwidth,
        a_lower,
        index1,
        global_column_p,
        operating.press_cavitate,
    )

    # Assign the nodal pressure.
    for i in range(mesh.total_nodes):
        node = int(mesh.n_index[i])
        nodal_pressure[p, node] = global_column_p[node]

    # A fully cavitated pad has zero pressure everywhere.
    if full_cavitate[p]:
        for i in range(mesh.total_nodes):
            node = int(mesh.n_index[i])
            nodal_pressure[p, node] = 0.0

    # Cavitation due to insufficient lubricant in a divergent clearance
    # (flooded condition, no pressure dam, non-oil-seal only).
    if operating.operating_type == "regular_flooded":
        for i in range(mesh.total_nodes - mesh.total_e_z_film - 1):
            node = int(mesh.n_index[i])
            if (
                h_n[p, node] > h_n[p, node + mesh.total_e_z_film + 1]
                and h_n[p, node] > h_min[p]
                and x_reynolds[p, node] > x_hmin[p]
                and operating.bearing_type != "pressure_dam"
            ):
                nodal_pressure[p, node] = 0.0

    # Nodal pressure gradients.
    if operating.bearing_type == "pressure_dam":
        dpdx_n, dpdz_n = press_gradient_node_dam(
            pad_index,
            mesh,
            nodal_pressure,
            pads.axial_length,
            dpdx_n,
            dpdz_n,
        )
    else:
        dpdx_n, dpdz_n = press_gradient_node(
            pad_index,
            mesh,
            nodal_pressure,
            pads.axial_length,
            dpdx_n,
            dpdz_n,
        )

    return nodal_pressure, dpdx_n, dpdz_n
