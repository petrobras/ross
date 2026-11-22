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
``dpdz_n`` rather than mutating them in place, and the mask of the nodes held
at the cavitation pressure (``cavitated``), which the perturbation solves of
:mod:`ross.bearings.fluid_film.coefficients` hold fixed.

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
    press_bc_jit,
    press_gradient_node_jit,
    press_postprocess_jit,
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
    p = pad_index
    press_bc_index = np.zeros(mesh.dim_xz, dtype=np.int64)
    prescribed_press = np.zeros(mesh.dim_xz, dtype=float)
    is_360 = abs(pads.arc_length_rad[0] - 2.0 * PI) < 1.0e-6
    linear_edges = operating.operating_type in ("axial_flow", "high_ambient_pressure")
    if linear_edges or operating.operating_type != "regular_flooded":
        edge1_press = operating.ambient_press1
        edge2_press = operating.ambient_press2
    else:
        # Flooded: zero on every edge.
        edge1_press = edge2_press = 0.0

    # One pass over the nodes in scan order; each node takes the first edge
    # rule it meets (axial edge z = 0, z = L, then the leading/trailing
    # edges, which a 360 degree land does not have).
    total_bc_reynolds = press_bc_jit(
        p,
        int(mesh.total_nodes),
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        np.ascontiguousarray(mesh.x, dtype=np.float64),
        np.ascontiguousarray(mesh.z, dtype=np.float64),
        float(pads.axial_length[p]),
        float(pads.pad_length[p]),
        int(film_onset[p]),
        int(mesh.total_e_z_film),
        bool(is_360),
        bool(linear_edges),
        float(edge1_press),
        float(edge2_press),
        float(operating.ambient_press1),
        float(operating.ambient_press2),
        press_bc_index,
        prescribed_press,
    )
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
    cavitated=None,
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
    cavitated : array_like of bool, optional
        Cavitated-node mask of the other pads, shape ``(total_pads, dim_xz)``
        (copied; row ``pad_index`` is replaced). Default is None (all False).

    Returns
    -------
    nodal_pressure : numpy.ndarray
        Updated nodal pressure, shape ``(total_pads, dim_xz)``.
    dpdx_n : numpy.ndarray
        Circumferential pressure gradient, same shape.
    dpdz_n : numpy.ndarray
        Axial pressure gradient, same shape.
    cavitated : numpy.ndarray of bool
        Nodes held at the cavitation pressure, same shape: raised to it by the
        clamp of the back substitution, on a fully cavitated pad, or in the
        divergent clearance downstream of the minimum film thickness
        (flooded operation without a pressure dam). The perturbation
        solves of the dynamic coefficients hold these nodes fixed.
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
    global_column_p, clamped = banded.lu_solve_cavitating(
        global_matrix_p,
        mesh.total_nodes,
        mesh.bandwidth,
        a_lower,
        index1,
        global_column_p,
        operating.press_cavitate,
        return_clamped=True,
    )

    nodal_pressure, cavitated = _postprocess(
        mesh,
        operating,
        p,
        global_column_p,
        clamped,
        bool(full_cavitate[p]),
        h_n,
        h_min,
        x_hmin,
        x_reynolds,
        nodal_pressure,
        cavitated,
    )

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

    return nodal_pressure, dpdx_n, dpdz_n, cavitated


def _postprocess(
    mesh,
    operating,
    p,
    global_column_p,
    clamped,
    full_cavitate_p,
    h_n,
    h_min,
    x_hmin,
    x_reynolds,
    nodal_pressure,
    cavitated,
):
    """Write one pad's solution, its clamp mask and the cavitation rules.

    The nodal pressure takes the solution and ``cavitated`` the nodes the
    clamp held at the cavitation pressure; a fully cavitated pad is at the
    cavitation pressure everywhere; in flooded operation without a pressure
    dam, the divergent clearance downstream of the minimum film is
    cavitated (``press_postprocess_jit``).

    Returns
    -------
    nodal_pressure, cavitated : numpy.ndarray
        Updated copies (``cavitated`` is a new array).
    """
    if cavitated is None:
        cavitated = np.zeros(nodal_pressure.shape, dtype=bool)
    cavitated = np.array(cavitated, dtype=bool, copy=True)
    press_postprocess_jit(
        int(p),
        int(mesh.total_nodes),
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        int(mesh.total_e_z_film),
        np.ascontiguousarray(global_column_p, dtype=np.float64),
        np.ascontiguousarray(clamped, dtype=np.bool_),
        bool(full_cavitate_p),
        float(operating.press_cavitate),
        operating.operating_type == "regular_flooded"
        and operating.bearing_type != "pressure_dam",
        np.ascontiguousarray(h_n, dtype=np.float64),
        float(h_min[p]),
        float(x_hmin[p]),
        np.ascontiguousarray(x_reynolds, dtype=np.float64),
        nodal_pressure,
        cavitated,
    )
    return nodal_pressure, cavitated


def press_full_cavitation(
    mesh,
    operating,
    pad_index,
    pads,
    h_n,
    h_min,
    x_hmin,
    nodal_pressure,
    dpdx_n,
    dpdz_n,
    cavitated=None,
):
    """Pressure of a fully cavitated pad, without a Reynolds solve.

    A pad whose film diverges from its leading edge on (``full_cavitate``,
    see :func:`ross.bearings.fluid_film.hydrodynamics.film_thickness`) is at
    the cavitation pressure everywhere: :func:`press` overwrites its
    Reynolds solution with it. This returns the same fields as
    :func:`press` for such a pad, bit for bit, without the solve.

    Parameters
    ----------
    mesh, operating, pad_index, pads, h_n, h_min, x_hmin, nodal_pressure, \
dpdx_n, dpdz_n, cavitated
        As for :func:`press`.

    Returns
    -------
    nodal_pressure, dpdx_n, dpdz_n, cavitated : numpy.ndarray
        As returned by :func:`press` with ``full_cavitate[pad_index]`` true.
    """
    p = pad_index
    nodal_pressure = np.ascontiguousarray(nodal_pressure, dtype=np.float64)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)
    nodal_pressure, cavitated = _postprocess(
        mesh,
        operating,
        p,
        np.zeros(mesh.dim_xz, dtype=np.float64),
        np.zeros(mesh.dim_xz, dtype=np.bool_),
        True,
        np.asarray(h_n, dtype=float),
        h_min,
        x_hmin,
        np.asarray(mesh.x, dtype=float),
        nodal_pressure,
        cavitated,
    )
    gradient = (
        press_gradient_node_dam
        if operating.bearing_type == "pressure_dam"
        else press_gradient_node
    )
    dpdx_n, dpdz_n = gradient(
        pad_index, mesh, nodal_pressure, pads.axial_length, dpdx_n, dpdz_n
    )
    return nodal_pressure, dpdx_n, dpdz_n, cavitated


CAVITATION_MODELS = ("reynolds", "mass_conserving")
# Tolerance of the mass-conserving film solves (scaled residual, see
# ``solve_elrod_adams``).
FILM_TOLERANCE = 1e-13
MASS_CONSERVING_THERMAL_TYPES = (None, "adiabatic")
MASS_CONSERVING_OPERATING_TYPES = ("regular_flooded", "starved_condition_even")


def check_cavitation_model(
    cavitation_model,
    thermal_type,
    deform_type,
    operating_type,
    bearing_type,
    pad_arc,
    cavitation_pressure,
    ambient_pressure_1=0.0,
    ambient_pressure_2=0.0,
):
    """Check that the engine supports the cavitation model for a bearing.

    Parameters
    ----------
    cavitation_model : str
        ``"reynolds"`` or ``"mass_conserving"``.
    thermal_type, deform_type, operating_type, bearing_type : str or None
        Model flags of the bearing.
    pad_arc : array_like
        Per-pad arc, rad.
    cavitation_pressure : float
        Cavitation pressure, Pa (gauge).
    ambient_pressure_1, ambient_pressure_2 : float, optional
        Ambient pressures of the axial edges, Pa (gauge). Default is 0.

    Raises
    ------
    ValueError
        Unknown ``cavitation_model``, or a mass-conserving film whose
        boundary pressures are below the cavitation pressure.
    NotImplementedError
        The mass-conserving model with a thermal model other than None or
        ``"adiabatic"``, pad deformation, an operating type other than
        ``"regular_flooded"`` or ``"starved_condition_even"``, a pressure dam
        or a 360 deg pad.

    Examples
    --------
    >>> check_cavitation_model("mass_conserving", None, None,
    ...                        "regular_flooded", "fixed_geometry", [3.0, 3.0], 0.0)
    >>> check_cavitation_model("mass_conserving", "full", None,
    ...                        "regular_flooded", "fixed_geometry", [3.0, 3.0], 0.0)
    Traceback (most recent call last):
    ...
    NotImplementedError: cavitation_model='mass_conserving' needs thermal_type None or 'adiabatic', got 'full'
    """
    if cavitation_model not in CAVITATION_MODELS:
        raise ValueError(
            f"cavitation_model must be one of {list(CAVITATION_MODELS)}, "
            f"not {cavitation_model!r}"
        )
    if cavitation_model == "reynolds":
        return
    model = "cavitation_model='mass_conserving'"
    if thermal_type not in MASS_CONSERVING_THERMAL_TYPES:
        raise NotImplementedError(
            f"{model} needs thermal_type None or 'adiabatic', got {thermal_type!r}"
        )
    if deform_type is not None:
        raise NotImplementedError(
            f"{model} needs rigid pads (deform_type None), got {deform_type!r}"
        )
    if operating_type not in MASS_CONSERVING_OPERATING_TYPES:
        raise NotImplementedError(
            f"{model} needs operating_type 'regular_flooded' or "
            f"'starved_condition_even', got {operating_type!r}"
        )
    if bearing_type == "pressure_dam":
        raise NotImplementedError(f"{model} does not model the pressure dam")
    if abs(float(np.atleast_1d(pad_arc)[0]) - 2.0 * PI) < 1.0e-6:
        raise NotImplementedError(
            f"{model} does not model the 360 deg pad (short-bearing oil seal)"
        )
    boundary = 0.0
    if operating_type != "regular_flooded":
        boundary = min(0.0, float(ambient_pressure_1), float(ambient_pressure_2))
    if float(cavitation_pressure) > boundary:
        raise ValueError(
            f"{model} needs the film boundary pressures (lowest {boundary} Pa) at "
            f"or above cavitation_pressure, got {float(cavitation_pressure)} Pa"
        )


class MassConservingFilm:
    """Film-fraction state of the mass-conserving (Elrod-Adams) pressure solves.

    One instance per case. It holds, per pad, the inlet film fraction, the
    last solution (the start of the next solve of the pad) and its solver
    record, and it counts the solves.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    dim_xz : int
        Nodes per pad.
    charge : callable, optional
        Called once before each solve (the work budget of the case). Default
        is None (not counted).

    Attributes
    ----------
    film_fraction_in : numpy.ndarray
        Film fraction of the oil entering each pad at its leading edge: 1 for
        a flooded inlet, less for a starved one.
    film_fraction : numpy.ndarray
        Film fraction of the last solve of each pad, shape
        ``(total_pads, dim_xz)``.
    inlet_capacity : numpy.ndarray
        Flow each pad would draw at its leading edge with a full inlet film
        (the Couette flow ``U G h`` through the leading control-volume
        faces), m**3/s.
    inlet_flow : numpy.ndarray
        Net flow each pad drew through its leading edge in its last solve
        (the Couette inflow less the pressure-driven backflow into the
        supply), m**3/s.
    records : list
        Per pad, the :class:`~ross.bearings.film.cavitation.CavitationSolution`
        of the last solve (None before the first one).
    full_film : numpy.ndarray of bool
        Per pad, whether the last solve has a full-film (uncavitated) free
        node; False for a pad whose film is cavitated everywhere.
    solves : int
        Solves done.
    """

    def __init__(self, total_pads, dim_xz, charge=None):
        self.film_fraction_in = np.ones(total_pads)
        self.film_fraction = np.ones((total_pads, dim_xz))
        self.inlet_capacity = np.zeros(total_pads)
        self.inlet_flow = np.zeros(total_pads)
        self.records = [None] * total_pads
        self.full_film = np.ones(total_pads, dtype=bool)
        self.solves = 0
        self.charge = charge

    def _count(self):
        if self.charge is not None:
            self.charge()
        self.solves += 1

    def summary(self):
        """Return the worst pad's record of the last solves.

        Returns
        -------
        dict
            ``iterations`` (most Newton iterations of a pad), ``residual``
            (largest final residual), ``converged`` (every pad),
            ``reason`` (of the first pad that did not converge, else
            ``"converged"``), ``mass_balance`` (per pad) and ``solves``.
        """
        solved = [sol for sol in self.records if sol is not None]
        if not solved:
            return {
                "iterations": 0,
                "residual": 0.0,
                "converged": True,
                "reason": "converged",
                "mass_balance": [0.0] * len(self.records),
                "solves": self.solves,
            }
        failed = [sol for sol in solved if not sol.converged]
        return {
            "iterations": max(sol.record.iterations for sol in solved),
            "residual": max(sol.record.final_residual for sol in solved),
            "converged": not failed,
            "reason": failed[0].record.reason if failed else "converged",
            "mass_balance": [
                0.0 if sol is None else float(sol.mass_balance) for sol in self.records
            ],
            "solves": self.solves,
        }


def cavitation_grid(
    mesh,
    operating,
    pad_index,
    pads,
    vis_effect_3d,
    film_onset,
    h_n,
    film_fraction_in=1.0,
):
    """Build the finite-volume grid of one pad on the Reynolds nodes.

    The control volumes are centred on the nodes of the engine's Reynolds
    mesh, so the pressure, film fraction and cavitated mask map one to one
    to the nodal arrays. The flux per unit width is
    ``q = h**3 Gamma grad(p) + U G h theta e_x`` with the engine's
    cross-film functions ``Gamma`` (negative) and ``G``
    (:func:`gamma_g`) and the surface speed ``U``; the nodes with a
    prescribed pressure (:func:`press_bc`: axial edges, leading edge and the
    nodes upstream of the film onset, trailing edge) are the Dirichlet
    nodes, and the oil entering from them has the film fraction
    ``film_fraction_in``.

    Parameters
    ----------
    mesh, operating, pad_index, pads, vis_effect_3d, film_onset, h_n
        As for :func:`press`.
    film_fraction_in : float, optional
        Film fraction of the oil at the leading-edge nodes (and the nodes
        upstream of the film onset). Default is 1 (flooded).

    Returns
    -------
    grid : ross.bearings.film.cavitation.CavitationGrid
        The pad's grid, shape ``(total_e_x_film + 1, total_e_z_film + 1)``.
    inlet_capacity : float
        Couette flow through the faces from the inlet nodes into the film
        with a full inlet film, m**3/s.
    """
    from ross.bearings.film.cavitation import CavitationGrid

    p = pad_index
    nx = mesh.total_e_x_film + 1
    nz = mesh.total_e_z_film + 1
    n = nx * nz
    if mesh.total_nodes != n or np.any(np.asarray(mesh.n_index[:n]) != np.arange(n)):
        raise NotImplementedError(
            "the mass-conserving film needs a structured Reynolds mesh "
            "(no pocket nodes)"
        )
    x = np.asarray(mesh.x, dtype=float)[p, :n].reshape(nx, nz)
    z = np.asarray(mesh.z, dtype=float)[p, :n].reshape(nx, nz)
    h = np.asarray(h_n, dtype=float)[p, :n].reshape(nx, nz)
    gamma, g = gamma_g(mesh, pad_index, pads, h_n, vis_effect_3d)
    total_bc, bc_index, bc_value = press_bc(
        mesh, operating, pad_index, pads, film_onset
    )
    dirichlet = np.zeros(n, dtype=bool)
    dirichlet[bc_index[:total_bc]] = True
    pressure_bc = np.zeros(n)
    pressure_bc[bc_index[:total_bc]] = bc_value[:total_bc]
    inlet_columns = int(film_onset[p]) + 1
    fraction = np.ones((nx, nz))
    fraction[:inlet_columns] = float(film_fraction_in)
    grid = CavitationGrid(
        x[:, 0],
        z[0],
        h,
        -(h**3) * gamma[:n].reshape(nx, nz),
        float(operating.speed_surface) * g[:n].reshape(nx, nz),
        dirichlet.reshape(nx, nz),
        pressure_bc.reshape(nx, nz),
        film_fraction_in=fraction,
    )
    inlet = (grid.face_left < inlet_columns * nz) & ~dirichlet[grid.face_right]
    inlet_capacity = float(np.sum(np.clip(grid.face_couette[inlet], 0.0, None)))
    return grid, inlet_capacity


def press_mass_conserving(
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
    cavitated=None,
    film=None,
    keep=True,
):
    """Solve the mass-conserving (Elrod-Adams) film pressure on one pad.

    The drop-in replacement of :func:`press` for
    ``cavitation_model="mass_conserving"``: the pad's finite-volume grid
    (:func:`cavitation_grid`) is solved by
    :func:`ross.bearings.film.cavitation.solve_elrod_adams`, starting from
    the pad's previous solution when ``film`` holds one (a start that does
    not converge is solved again from the full film). Rupture and
    reformation follow from the mass balance; no divergent-zone rule or
    full-cavitation shortcut is applied.

    Parameters
    ----------
    mesh, operating, pad_index, pads, vis_effect_3d, film_onset, h_n, \
h_min, x_hmin, nodal_pressure, dpdx_n, dpdz_n, full_cavitate, cavitated
        As for :func:`press` (``h_min``, ``x_hmin`` and ``full_cavitate`` are
        not used).
    film : MassConservingFilm, optional
        Inlet film fractions, previous solutions and solve counter of the
        case. Default is None (flooded inlet, cold start, not counted).
    keep : bool, optional
        Store the solution in ``film`` (the start of the pad's next solve
        and its reported record). False for the perturbed solves of a finite
        difference. Default is True.

    Returns
    -------
    nodal_pressure, dpdx_n, dpdz_n, cavitated : numpy.ndarray
        As returned by :func:`press`; ``cavitated`` is the solver's mask of
        the free nodes in the cavitated zone (film fraction below 1, or at
        the cavitation pressure on a rupture boundary).
    """
    import warnings

    from ross.bearings.film.cavitation import solve_elrod_adams
    from ross.bearings.film.loops import ConvergenceWarning

    p = pad_index
    fraction_in = 1.0 if film is None else float(film.film_fraction_in[p])
    grid, capacity = cavitation_grid(
        mesh, operating, pad_index, pads, vis_effect_3d, film_onset, h_n, fraction_in
    )
    previous = None if film is None else film.records[p]
    p_cav = float(operating.press_cavitate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if film is not None:
            film._count()
        if previous is not None:
            solution = solve_elrod_adams(
                grid,
                p_cav,
                tol=FILM_TOLERANCE,
                p_init=previous.p,
                theta_init=previous.theta,
            )
            if not solution.converged:
                if film is not None:
                    film._count()
                solution = solve_elrod_adams(grid, p_cav, tol=FILM_TOLERANCE)
        else:
            solution = solve_elrod_adams(grid, p_cav, tol=FILM_TOLERANCE)
    n = solution.p.size
    if film is not None and keep:
        film.records[p] = solution
        film.full_film[p] = bool(np.any(~grid.dirichlet & ~solution.cavitated))
        film.inlet_capacity[p] = capacity
        inlet_columns = int(film_onset[p]) + 1
        film.inlet_flow[p] = float(np.sum(solution.boundary_flow[:inlet_columns]))
        film.film_fraction[p, :n] = solution.theta.ravel()

    nodal_pressure = np.array(nodal_pressure, dtype=np.float64, copy=True)
    nodal_pressure[p, :n] = solution.p.ravel()
    if cavitated is None:
        cavitated = np.zeros(nodal_pressure.shape, dtype=bool)
    cavitated = np.array(cavitated, dtype=bool, copy=True)
    cavitated[p] = False
    cavitated[p, :n] = solution.cavitated.ravel()
    dpdx_n, dpdz_n = press_gradient_node(
        pad_index,
        mesh,
        nodal_pressure,
        pads.axial_length,
        np.ascontiguousarray(dpdx_n, dtype=np.float64),
        np.ascontiguousarray(dpdz_n, dtype=np.float64),
    )
    return nodal_pressure, dpdx_n, dpdz_n, cavitated
