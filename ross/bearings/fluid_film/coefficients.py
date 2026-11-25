"""Dynamic coefficients and the equilibrium Jacobian.

For a converged hydrodynamic (zeroth-order) pressure solution, this module
computes:

* :func:`jacobian` -- the journal-equilibrium Jacobian, whose elements are the
  (negative) stiffness coefficients ``kxx, kyx, kxy, kyy`` and, for tilting
  pads, the tilt-DOF blocks ``kdeltax, kdeltay, kxdelta, kydelta,
  kdeltadelta`` plus the per-pad direct blocks.
* :func:`damping` -- the corresponding velocity-perturbation damping
  coefficients ``cxx, cyx, cxy, cyy`` and their tilt-DOF / per-pad blocks.

The pad tilt (and pivot) degrees of freedom are condensed at the whirl
frequency by :class:`~ross.bearings.fluid_film.padmatrices.PadMatrices`.

Each coefficient comes from a perturbation solve of the Reynolds equation:
:func:`stiffness_press` perturbs the journal position and pad tilt,
:func:`damping_press_dxdot` / :func:`damping_press_dydot` /
:func:`damping_press_dtiltdot` perturb the corresponding velocities.

Index conventions
-----------------
Arrays are 0-based throughout: ``pad_index`` is the pad row, the node and
element index arrays (``node_i_reynolds``, ``e_index_reynolds``,
``n_index_reynolds``, ...) store numbers used directly as indices, and per-pad
2-D arrays such as ``h_n[pad, node]`` are indexed ``[pad_index, node]``.
``match_nodes_xz`` stores 3-D node numbers with ``-1`` for empty cross-film
columns. In the banded kernels node ``k`` sits at row ``k`` with the band
diagonal at column ``bandwidth - 1``.

Injected dependencies
---------------------
The element-assembly and linear-algebra helpers live in the ``pressure``
module, and are passed in rather than imported to avoid an import cycle. Each
public solver here takes a ``helpers`` argument -- an object (e.g. a
``SimpleNamespace``) exposing those callables; :func:`_require_helpers`
documents and validates the contract.

Nodal inputs are shaped ``(total_pads, dim_xz)``, 3-D film quantities
``(total_pads, dim_3d)``, ``match_nodes_xz`` is ``(dim_xz, dim_yf)``, and
``dx_reynolds`` / ``dz_reynolds`` are ``(total_pads, dim_xz, 4)``.
"""

import numpy as np

from ross.bearings.fluid_film import banded
from ross.bearings.fluid_film._numba_kernels import (
    gamma_g_loop_jit,
    integrate_forces_jit,
    pert_press_assemble_all_jit,
    pert_press_rhs_all_jit,
    press_bc_pert_jit,
    stiffness_source_all_jit,
    stiffness_source_terms_jit,
)
from ross.bearings.fluid_film.constants import PI
from ross.bearings.fluid_film.state import CoefficientBlock


# ---------------------------------------------------------------------------
# helper-injection contract
# ---------------------------------------------------------------------------
def _require_helpers(helpers):
    """Validate the injected FE/linear-algebra helper bundle.

    Parameters
    ----------
    helpers : object
        Namespace exposing the following callables:

        ``element_press(k_x, k_z, q, l_e, w_e) -> (e_matrix, e_column)``
            The 4x4 element matrix and length-4 element column.
        ``include_press(global_matrix, global_column, bandwidth, total_bc,
        press_bc_index, prescribed_press, total_n) -> (global_matrix,
        global_column)``
            Apply the prescribed nodal pressures.
        ``zero_pressure_system(dim_xz, total_n, total_column) ->
        (global_matrix, global_column)``
            Allocate and zero the banded global system.
        ``integrate_xz(pad_index, total_e, e_index, node_i, node_j, node_k,
        node_l, e_length, e_width, f) -> inte_f``
            Surface integral of the nodal field ``f``.

    Returns
    -------
    object
        The validated ``helpers`` bundle.
    """
    required = (
        "element_press",
        "include_press",
        "zero_pressure_system",
        "integrate_xz",
    )
    missing = [name for name in required if not callable(getattr(helpers, name, None))]
    if missing:
        raise ValueError(f"helpers bundle is missing callables: {missing}")
    return helpers


# ---------------------------------------------------------------------------
# Perturbation boundary conditions, cross-film functions and banded solve
# ---------------------------------------------------------------------------
def press_bc_pert(
    mesh,
    pad_index,
    film_onset,
    pads,
    cavitated,
):
    """Build the prescribed-pressure boundary condition for a perturbation.

    Returns the list of constrained nodes and their prescribed (always zero)
    pressures: the film edges, the nodes before the film onset and the nodes
    the zeroth-order solve held at the cavitation pressure.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    pad_index : int
        0-based current pad.
    film_onset : ndarray, shape (total_pads,)
        Per-pad film-onset element offset.
    pads : PadGeometry
        Per-pad geometry (``pad_length``, ``axial_length``,
        ``arc_length_rad``).
    cavitated : ndarray of bool, shape (total_pads, dim_xz)
        Nodes held at the cavitation pressure by the zeroth-order solve
        (returned by :func:`ross.bearings.fluid_film.pressure.press`).

    Returns
    -------
    total_bc_reynolds : int
        Number of constrained nodes.
    press_bc_index : ndarray, shape (dim_xz,)
        0-based ids of constrained nodes (only first ``total_bc_reynolds``
        entries are meaningful).
    prescribed_press : ndarray, shape (dim_xz,)
        Prescribed values (all zero).
    """
    p = pad_index
    total_bc_reynolds, press_bc_index, prescribed_press = press_bc_pert_jit(
        p,
        int(mesh.dim_xz),
        int(mesh.total_e_z_film),
        int(film_onset[p]),
        float(pads.pad_length[p]),
        float(pads.axial_length[p]),
        float(pads.arc_length_rad[0]),
        np.ascontiguousarray(cavitated, dtype=np.bool_),
        int(mesh.total_nodes),
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        np.ascontiguousarray(mesh.x, dtype=np.float64),
        np.ascontiguousarray(mesh.z, dtype=np.float64),
    )
    return total_bc_reynolds, press_bc_index, prescribed_press


def gamma_g_pert(
    mesh,
    pad_index,
    pads,
    h_n,
    vis_effect_3d,
):
    """Compute the cross-film ``Gamma`` and ``G`` functions at every node.

    Identical in form to :func:`ross.bearings.fluid_film.pressure.gamma_g`, but for
    the perturbation solve.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity, the ``match_nodes_xz``
        map to the 3-D cross-film node columns, and the per-pad
        babbitt/core layer element counts.
    pad_index : int
        0-based current pad.
    pads : PadGeometry
        Per-pad geometry (pad thickness/length, pocket and dam extents).
    h_n : ndarray, shape (total_pads, dim_xz)
        Nodal film thickness, m.
    vis_effect_3d : ndarray, shape (total_pads, dim_3d)
        Viscosity-effect field of the 3-D mesh.

    Returns
    -------
    gamma : ndarray, shape (dim_xz,)
        Cross-film ``Gamma`` at every node.
    g : ndarray, shape (dim_xz,)
        Cross-film ``G`` at every node.
    """
    p = pad_index
    gamma = np.zeros(mesh.dim_xz, dtype=np.float64)
    g = np.zeros(mesh.dim_xz, dtype=np.float64)

    # Drive the per-node cross-film integration via the fully 0-based JIT loop
    # (0-based pad row, 0-based node values, 0-based match_nodes values).
    gamma_g_loop_jit(
        p,
        mesh.total_nodes,
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        mesh.total_e_y_film,
        np.ascontiguousarray(mesh.match_nodes_xz, dtype=np.int64),
        int(mesh.total_e_y_trackbl[p]),
        int(mesh.total_e_y_trackcore[p]),
        float(pads.pad_thickness),
        float(pads.pad_length[p]),
        float(pads.axial_length[p]),
        float(pads.depth_track[p]),
        float(pads.length_track[p]),
        float(pads.axial_length_track[p]),
        float(pads.axial_length_dam[p]),
        np.ascontiguousarray(mesh.x, dtype=np.float64),
        np.ascontiguousarray(mesh.z, dtype=np.float64),
        np.ascontiguousarray(mesh.y_3d, dtype=np.float64),
        np.ascontiguousarray(h_n, dtype=np.float64),
        np.ascontiguousarray(vis_effect_3d, dtype=np.float64),
        gamma,
        g,
    )
    return gamma, g


def _solve_pert_pressure(
    mesh,
    pad_index,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    element_source_all,
    helpers,
):
    """Solve one perturbed Reynolds problem and return nodal perturbed pressure.

    A single right-hand side of :func:`_solve_pert_pressures`.

    Returns
    -------
    n_press_pert : ndarray, shape (dim_xz,)
        Perturbed nodal pressure (only Reynolds nodes are set).
    """
    return _solve_pert_pressures(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        cavitated,
        [element_source_all],
        helpers,
    )[0]


def _solve_pert_pressures(
    mesh,
    pad_index,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    element_sources,
    helpers,
):
    """Solve the perturbed Reynolds problems of one pad with one factorisation.

    Shared backbone of the damping (:func:`damping_press_dxdot` /
    :func:`damping_press_dydot` / :func:`damping_press_dtiltdot`) and stiffness
    (:func:`stiffness_press`) perturbation solves. The perturbations of a pad
    share the matrix (film thickness, cross-film ``Gamma``, the prescribed
    nodes, and for the short-bearing oil-seal case the zeroed circumferential
    conduction) and differ only in the per-element source term ``Q``. The
    matrix is assembled and LU-factored once; each source gives one
    right-hand side, solved with the same factors (the earlier versions
    refactored the matrix per right-hand side, with the same result).

    Each ``element_sources[k]`` computes ``Q`` for **all** elements at once
    as ``element_source_all(elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma,
    g_e_arr, theta_e_arr, oil_seal)`` with the pre-gathered element ids,
    local node ids, per-pad ``gamma`` nodal array, per-element averages
    ``g_E`` / ``theta_E`` and the oil-seal flag, and returns the
    length-``total_e_reynolds`` source array (C-family: a vectorized numpy
    expression; K-family: the fused ``stiffness_source_all_jit`` kernel).

    Returns
    -------
    list of ndarray, shape (dim_xz,)
        Perturbed nodal pressure of each source (only Reynolds nodes set).
    """
    p = pad_index
    total_column_reynolds = 2 * mesh.bandwidth - 1
    oil_seal = abs(pads.arc_length_rad[0] - 2.0 * PI) < 1.0e-6

    total_bc, press_bc_index, prescribed_press = press_bc_pert(
        mesh,
        pad_index,
        film_onset,
        pads,
        cavitated,
    )

    gamma, g = gamma_g_pert(
        mesh,
        pad_index,
        pads,
        h_n,
        vis_effect_3d,
    )

    # Gather the per-element topology once (vectorized) and compute every
    # element's source value in a single call per source.
    e_index_reynolds = np.ascontiguousarray(mesh.e_index, dtype=np.int64)
    node_i_reynolds = np.ascontiguousarray(mesh.node_i, dtype=np.int64)
    node_j_reynolds = np.ascontiguousarray(mesh.node_j, dtype=np.int64)
    node_k_reynolds = np.ascontiguousarray(mesh.node_k, dtype=np.int64)
    node_l_reynolds = np.ascontiguousarray(mesh.node_l, dtype=np.int64)
    e_length = np.ascontiguousarray(mesh.e_length, dtype=np.float64)
    e_width = np.ascontiguousarray(mesh.e_width, dtype=np.float64)
    elem = e_index_reynolds[: int(mesh.total_elements)]
    ni_arr = node_i_reynolds[elem]
    nj_arr = node_j_reynolds[elem]
    nk_arr = node_k_reynolds[elem]
    nl_arr = node_l_reynolds[elem]
    g_e_arr = (g[ni_arr] + g[nj_arr] + g[nk_arr] + g[nl_arr]) / 4.0
    theta_e_arr = (
        mesh.x_rad[p, ni_arr]
        + mesh.x_rad[p, nj_arr]
        + mesh.x_rad[p, nk_arr]
        + mesh.x_rad[p, nl_arr]
    ) / 4.0
    sources = [
        np.ascontiguousarray(
            source(
                elem,
                ni_arr,
                nj_arr,
                nk_arr,
                nl_arr,
                gamma,
                g_e_arr,
                theta_e_arr,
                oil_seal,
            ),
            dtype=np.float64,
        )
        for source in element_sources
    ]

    # Matrix and the first right-hand side in one pass; the other
    # right-hand sides by the column part of the same assembly.
    global_matrix_p, first_column = helpers.zero_pressure_system(
        mesh.dim_xz, mesh.total_nodes, total_column_reynolds
    )
    pert_press_assemble_all_jit(
        p,
        int(mesh.total_elements),
        e_index_reynolds,
        node_i_reynolds,
        node_j_reynolds,
        node_k_reynolds,
        node_l_reynolds,
        np.ascontiguousarray(gamma, dtype=np.float64),
        np.ascontiguousarray(h_n, dtype=np.float64),
        e_length,
        e_width,
        sources[0],
        bool(oil_seal),
        int(mesh.bandwidth),
        global_matrix_p,
        first_column,
    )
    columns = [first_column]
    for q_per_element in sources[1:]:
        column = np.zeros(first_column.shape, dtype=np.float64)
        pert_press_rhs_all_jit(
            p,
            int(mesh.total_elements),
            e_index_reynolds,
            node_i_reynolds,
            node_j_reynolds,
            node_k_reynolds,
            node_l_reynolds,
            e_length,
            e_width,
            q_per_element,
            column,
        )
        columns.append(column)

    # The prescribed nodes change the matrix the same way for every
    # right-hand side; each column gets its own pass on a copy of the
    # assembled matrix, so the column operations are those of a single solve.
    assembled = global_matrix_p.copy() if len(columns) > 1 else global_matrix_p
    idx = np.asarray(mesh.n_index, dtype=np.int64)[: int(mesh.total_nodes)]
    results = []
    factors = None
    for k, column in enumerate(columns):
        matrix = global_matrix_p if k == 0 else assembled.copy()
        matrix, column = helpers.include_press(
            matrix,
            column,
            mesh.bandwidth,
            total_bc,
            press_bc_index,
            prescribed_press,
            mesh.total_nodes,
        )
        if factors is None:
            factors = banded.lu_factor(matrix, mesh.total_nodes, mesh.bandwidth)
        a, a_lower, index1, _d = factors
        column = banded.lu_solve(
            a, mesh.total_nodes, mesh.bandwidth, a_lower, index1, column
        )
        n_press_pert = np.zeros(mesh.dim_xz, dtype=np.float64)
        n_press_pert[idx] = column[idx]
        results.append(n_press_pert)
    return results


def _integrate_forces(
    mesh,
    pad_index,
    n_press_pert,
    pads,
):
    """Integrate a perturbed nodal pressure into x/y forces and a pivot moment.

    Shared tail of every perturbation solve: forms ``p*cos(...)``,
    ``p*sin(...)`` and the moment integrand ``p*sin(theta - theta_pivot)``,
    integrates each via ``helpers.integrate_xz`` and scales the moment by
    the journal radius + pad thickness.

    Returns
    -------
    fx : float
        Integral of the x-force component.
    fy : float
        Integral of the y-force component.
    moment : float
        ``(journal radius + pad thickness) * \\int p \\sin(theta - theta_pivot)``.
    """
    p = pad_index
    return integrate_forces_jit(
        p,
        int(mesh.dim_xz),
        int(mesh.total_elements),
        np.ascontiguousarray(mesh.e_index, dtype=np.int64),
        np.ascontiguousarray(mesh.node_i, dtype=np.int64),
        np.ascontiguousarray(mesh.node_j, dtype=np.int64),
        np.ascontiguousarray(mesh.node_k, dtype=np.int64),
        np.ascontiguousarray(mesh.node_l, dtype=np.int64),
        np.ascontiguousarray(mesh.e_length, dtype=np.float64),
        np.ascontiguousarray(mesh.e_width, dtype=np.float64),
        np.ascontiguousarray(mesh.n_index, dtype=np.int64),
        int(mesh.total_nodes),
        np.ascontiguousarray(n_press_pert, dtype=np.float64),
        float(pads.leading_angle_rad[p]),
        np.ascontiguousarray(mesh.x_rad, dtype=np.float64),
        float(pads.x_pivot_rad[p]),
        float(pads.journal_radius),
        float(pads.pad_thickness),
    )


# ---------------------------------------------------------------------------
# Damping: velocity-perturbation pressure solves
# ---------------------------------------------------------------------------
def _damping_sources(pads, pad_index, tilting=True):
    """Element sources of the velocity perturbations of one pad.

    Returns the ``element_source_all`` callables of the x-velocity
    (``Q = -cos(leading_angle + theta_E)``), y-velocity
    (``Q = -sin(leading_angle + theta_E)``) and, when ``tilting``, the
    tilt-velocity (``Q = -(R + t) sin(theta_E - theta_pivot)``)
    perturbations, in that order.
    """
    p = pad_index

    def source_dx(
        elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
    ):
        return -np.cos(pads.leading_angle_rad[p] + theta_e_arr)

    def source_dy(
        elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
    ):
        return -np.sin(pads.leading_angle_rad[p] + theta_e_arr)

    def source_dtilt(
        elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
    ):
        return -(pads.journal_radius + pads.pad_thickness) * np.sin(
            theta_e_arr - pads.x_pivot_rad[p]
        )

    if tilting:
        return [source_dx, source_dy, source_dtilt]
    return [source_dx, source_dy]


def damping_press_dxdot(
    mesh,
    pad_index,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    helpers,
):
    """Solve the x-velocity perturbed Reynolds equation (damping ``Cxx/Cyx``).

    The element source is ``Q = -cos(leading_angle + theta_E)``.

    Returns
    -------
    cxx_i : float
        Direct x damping contribution of this pad.
    cyx_i : float
        Cross x->y damping contribution.
    cdeltax_i : float
        Pivot-moment x damping contribution.
    """
    _require_helpers(helpers)
    source_all = _damping_sources(pads, pad_index)[0]

    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        cavitated,
        source_all,
        helpers,
    )
    cxx_i, cyx_i, cdeltax_i = _integrate_forces(
        mesh,
        pad_index,
        n_press_pert,
        pads,
    )
    return cxx_i, cyx_i, cdeltax_i


def damping_press_dydot(
    mesh,
    pad_index,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    helpers,
):
    """Solve the y-velocity perturbed Reynolds equation (damping ``Cxy/Cyy``).

    Element source ``Q = -sin(leading_angle + theta_E)``.

    Returns
    -------
    cxy_i, cyy_i, cdeltay_i : float
        Cross y->x, direct y, and pivot-moment y damping contributions.
    """
    _require_helpers(helpers)
    source_all = _damping_sources(pads, pad_index)[1]

    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        cavitated,
        source_all,
        helpers,
    )
    cxy_i, cyy_i, cdeltay_i = _integrate_forces(
        mesh,
        pad_index,
        n_press_pert,
        pads,
    )
    return cxy_i, cyy_i, cdeltay_i


def damping_press_dtiltdot(
    mesh,
    pad_index,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    helpers,
):
    """Solve the pad-tilt velocity perturbed Reynolds equation (damping tilt DOF).

    Element source ``Q = -(journal radius + pad thickness) * sin(theta_e -
    theta_pivot)``.

    Returns
    -------
    cxdelta_i, cydelta_i, cdeltadelta_i : float
        Tilt-induced x, y forces and the tilt-tilt damping for this pad.
    """
    _require_helpers(helpers)
    source_all = _damping_sources(pads, pad_index)[2]

    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        cavitated,
        source_all,
        helpers,
    )
    cxdelta_i, cydelta_i, cdeltadelta_i = _integrate_forces(
        mesh,
        pad_index,
        n_press_pert,
        pads,
    )
    return cxdelta_i, cydelta_i, cdeltadelta_i


def damping(
    total_pads,
    mesh,
    operating,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    unloaded,
    helpers,
):
    """Assemble the bearing damping coefficients over all pads.

    Loops over pads, solving the three velocity-perturbation Reynolds problems
    and summing the journal DOF contributions; tilt-DOF blocks are kept per
    pad.  Unloaded pads (under high ambient pressure for tilting-pad operating
    types 1/2) drop all blocks.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    operating : OperatingPoint
        Bearing/operating type flags and the cavitation pressure; the
        tilting-pad families get tilt-DOF blocks, and the axial-flow /
        high-ambient operating modes drop unloaded pads.
    film_onset : ndarray, shape (total_pads,)
        Per-pad film-onset element offset.
    pads : PadGeometry
        Per-pad geometry.
    vis_effect_3d : ndarray, shape (total_pads, dim_3d)
        Viscosity-effect field of the 3-D mesh.
    h_n : ndarray, shape (total_pads, dim_xz)
        Nodal film thickness, m.
    cavitated : ndarray of bool, shape (total_pads, dim_xz)
        Nodes held at the cavitation pressure by the zeroth-order solve; the
        perturbation solves hold them fixed.
    unloaded : ndarray of bool, shape (total_pads,)
        Per-pad unloaded flag.
    helpers : object
        FE/linear-algebra helper bundle (see :func:`_require_helpers`).

    Returns
    -------
    cxx, cyx, cxy, cyy : float
        Summed journal damping coefficients.
    cdeltax, cdeltay, cxdelta, cydelta, cdeltadelta : ndarray, shape (total_pads,)
        Per-pad tilt-DOF damping blocks.
    cxxi, cyxi, cxyi, cyyi : ndarray, shape (total_pads,)
        Per-pad direct journal damping contributions (for pivot reduction).
    """
    _require_helpers(helpers)
    cxx = cyx = cxy = cyy = 0.0
    cdeltax = np.zeros(total_pads, dtype=np.float64)
    cdeltay = np.zeros(total_pads, dtype=np.float64)
    cxdelta = np.zeros(total_pads, dtype=np.float64)
    cydelta = np.zeros(total_pads, dtype=np.float64)
    cdeltadelta = np.zeros(total_pads, dtype=np.float64)
    cxxi = np.zeros(total_pads, dtype=np.float64)
    cyxi = np.zeros(total_pads, dtype=np.float64)
    cxyi = np.zeros(total_pads, dtype=np.float64)
    cyyi = np.zeros(total_pads, dtype=np.float64)

    tilting = operating.bearing_type in (
        "conventional_tilting_pad",
        "inlet_groove_tilting_pad",
        "spray_bar_tilting_pad",
    )
    for pad_index in range(total_pads):
        # One factorisation per pad for its two or three velocity
        # perturbations.
        n_press = _solve_pert_pressures(
            mesh,
            pad_index,
            film_onset,
            pads,
            vis_effect_3d,
            h_n,
            cavitated,
            _damping_sources(pads, pad_index, tilting),
            helpers,
        )
        cxx_i, cyx_i, cdeltax_i = _integrate_forces(mesh, pad_index, n_press[0], pads)
        cxy_i, cyy_i, cdeltay_i = _integrate_forces(mesh, pad_index, n_press[1], pads)

        if tilting:
            cxdelta_i, cydelta_i, cdeltadelta_i = _integrate_forces(
                mesh, pad_index, n_press[2], pads
            )
        else:
            cdeltax_i = cdeltay_i = cxdelta_i = cydelta_i = cdeltadelta_i = 0.0

        if (
            operating.operating_type in ("axial_flow", "high_ambient_pressure")
            and tilting
            and bool(unloaded[pad_index])
        ):
            cdeltax_i = cdeltay_i = cxdelta_i = cydelta_i = cdeltadelta_i = 0.0
            cxx_i = cxy_i = cyx_i = cyy_i = 0.0

        cxx += cxx_i
        cyx += cyx_i
        cxy += cxy_i
        cyy += cyy_i
        cdeltax[pad_index] = cdeltax_i
        cdeltay[pad_index] = cdeltay_i
        cxdelta[pad_index] = cxdelta_i
        cydelta[pad_index] = cydelta_i
        cdeltadelta[pad_index] = cdeltadelta_i
        cxxi[pad_index] = cxx_i
        cyxi[pad_index] = cyx_i
        cxyi[pad_index] = cxy_i
        cyyi[pad_index] = cyy_i

    return CoefficientBlock(
        cxx,
        cyx,
        cxy,
        cyy,
        cdeltax,
        cdeltay,
        cxdelta,
        cydelta,
        cdeltadelta,
        cxxi,
        cyxi,
        cxyi,
        cyyi,
    )


# ---------------------------------------------------------------------------
# Stiffness: position-perturbation pressure solves, and the Jacobian
# ---------------------------------------------------------------------------
def _stiffness_source_terms(
    p,
    current_element,
    nodes,
    gamma,
    g_e,
    speed_surface,
    dx_reynolds,
    dz_reynolds,
    x_reynolds_rad,
    h_n,
    dpdx_n,
    dpdz_n,
    angle_fn,
    scale,
    angle_offset,
):
    """Compute the three Jacobian source terms (Term I/II/III) for one element.

    ``angle_fn`` is :data:`numpy.cos` (x-perturbation) or :data:`numpy.sin`
    (y/tilt); ``angle_offset`` is ``leading_angle`` for the x/y perturbations
    and ``-x_pivot`` for tilt; ``scale`` is 1 for x/y and
    the journal radius + pad thickness for tilt.
    """
    ni, nj, nk, nl = nodes
    # angle_fn is either np.cos or np.sin; route on an int the JIT can read.
    angle_kind = 0 if angle_fn is np.cos else 1
    # Array inputs are expected contiguous float64: ``gamma`` is built that way
    # by :func:`gamma_g_pert`, and the loop-invariant mesh/state arrays are
    # pre-converted once in :func:`stiffness_press`. Reconverting them here would
    # add ~7 redundant ``ascontiguousarray`` calls per element (~355k calls).
    return stiffness_source_terms_jit(
        p,
        int(current_element),
        int(ni),
        int(nj),
        int(nk),
        int(nl),
        gamma,
        float(g_e),
        float(speed_surface),
        dx_reynolds,
        dz_reynolds,
        x_reynolds_rad,
        h_n,
        dpdx_n,
        dpdz_n,
        angle_kind,
        float(scale),
        float(angle_offset[0]),
        float(angle_offset[1]),
        float(angle_offset[2]),
        float(angle_offset[3]),
    )


def _stiffness_source(direction, pad_index, pads, operating, h_n, dpdx_n, dpdz_n, mesh):
    """Element source of one displacement perturbation of a pad.

    ``direction`` selects the angular function and scaling of the Jacobian
    source terms: ``"dx"`` (cosine, offset by the pad leading angle),
    ``"dy"`` (sine, same offset) or ``"dtilt"`` (sine, offset by
    ``-x_pivot``, scaled by the pivot radius ``journal_radius +
    pad_thickness``). Returns the ``element_source_all`` callable of
    :func:`_solve_pert_pressures`.
    """
    p = pad_index
    # ``stiffness_source_all_jit`` requires contiguous float64 inputs.
    dx_reynolds = np.ascontiguousarray(mesh.dx, dtype=np.float64)
    dz_reynolds = np.ascontiguousarray(mesh.dz, dtype=np.float64)
    x_reynolds_rad = np.ascontiguousarray(mesh.x_rad, dtype=np.float64)
    h_n = np.ascontiguousarray(h_n, dtype=np.float64)
    dpdx_n = np.ascontiguousarray(dpdx_n, dtype=np.float64)
    dpdz_n = np.ascontiguousarray(dpdz_n, dtype=np.float64)

    # ``angle_kind`` selects the angular function inside
    # ``stiffness_source_all_jit``: 0 -> cosine, 1 -> sine.
    if direction == "dx":
        angle_kind = 0
        scale = 1.0
        angle_offset_base = pads.leading_angle_rad[p]
    elif direction == "dy":
        angle_kind = 1
        scale = 1.0
        angle_offset_base = pads.leading_angle_rad[p]
    elif direction == "dtilt":
        angle_kind = 1
        scale = pads.journal_radius + pads.pad_thickness
        angle_offset_base = -pads.x_pivot_rad[p]
    else:
        raise ValueError(f"direction must be 'dx', 'dy' or 'dtilt', not {direction!r}")

    def source_all(
        elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
    ):
        return stiffness_source_all_jit(
            p,
            elem,
            ni_arr,
            nj_arr,
            nk_arr,
            nl_arr,
            np.ascontiguousarray(gamma, dtype=np.float64),
            g_e_arr,
            float(operating.speed_surface),
            dx_reynolds,
            dz_reynolds,
            x_reynolds_rad,
            h_n,
            dpdx_n,
            dpdz_n,
            angle_kind,
            float(scale),
            float(angle_offset_base),
            bool(oil_seal),
        )

    return source_all


def stiffness_press(
    direction,
    mesh,
    pad_index,
    pads,
    operating,
    vis_effect_3d,
    film_onset,
    h_n,
    cavitated,
    dpdx_n,
    dpdz_n,
    helpers,
):
    """One stiffness perturbation solve of a pad.

    ``direction`` selects the perturbation (see :func:`_stiffness_source`):

    ``"dx"``
        cosine, offset by the pad leading angle.
    ``"dy"``
        sine, offset by the pad leading angle.
    ``"dtilt"``
        sine, offset by ``-x_pivot``, scaled by the pivot radius
        ``journal_radius + pad_thickness``.

    For the oil-seal short-bearing case ``k_x`` is zeroed and ``q`` drops its
    first term. :func:`jacobian` solves the directions of a pad together
    with one factorisation.

    Returns
    -------
    fx, fy, moment : float
        Direct, cross and pivot-moment stiffness contributions.
    """
    _require_helpers(helpers)
    source_all = _stiffness_source(
        direction, pad_index, pads, operating, h_n, dpdx_n, dpdz_n, mesh
    )
    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        cavitated,
        source_all,
        helpers,
    )
    fx, fy, moment = _integrate_forces(
        mesh,
        pad_index,
        n_press_pert,
        pads,
    )
    return fx, fy, moment


def jacobian(
    total_pads,
    mesh,
    operating,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    dpdx_n,
    dpdz_n,
    unloaded,
    helpers,
):
    """Assemble the equilibrium Jacobian (negative stiffness coefficients).

    Loops over pads, solving the three displacement-perturbation Reynolds
    problems and summing the journal DOF contributions; tilt-DOF blocks are
    kept per pad.  Unloaded pads (under high ambient pressure for tilting-pad
    operating types 1/2) drop all blocks.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    mesh : ReynoldsMesh
        Film (x-z) mesh: coordinates, connectivity and index maps.
    operating : OperatingPoint
        Bearing/operating type flags, surface speed and cavitation pressure.
    film_onset : ndarray, shape (total_pads,)
        Per-pad film-onset element offset.
    pads : PadGeometry
        Per-pad geometry.
    vis_effect_3d : ndarray, shape (total_pads, dim_3d)
        Viscosity-effect field of the 3-D mesh.
    h_n : ndarray, shape (total_pads, dim_xz)
        Nodal film thickness, m.
    cavitated : ndarray of bool, shape (total_pads, dim_xz)
        Nodes held at the cavitation pressure by the zeroth-order solve; the
        perturbation solves hold them fixed.
    dpdx_n, dpdz_n : ndarray, shape (total_pads, dim_xz)
        Zeroth-order nodal pressure gradients.
    unloaded : ndarray of bool, shape (total_pads,)
        Per-pad unloaded flag.
    helpers : object
        FE/linear-algebra helper bundle (see :func:`_require_helpers`).

    Returns
    -------
    kxx, kyx, kxy, kyy : float
        Summed journal stiffness coefficients.
    kdeltax, kdeltay, kxdelta, kydelta, kdeltadelta : ndarray, shape (total_pads,)
        Per-pad tilt-DOF stiffness blocks.
    kxxi, kyxi, kxyi, kyyi : ndarray, shape (total_pads,)
        Per-pad direct journal stiffness contributions (for pivot reduction).
    """
    _require_helpers(helpers)
    kxx = kyx = kxy = kyy = 0.0
    kdeltax = np.zeros(total_pads, dtype=np.float64)
    kdeltay = np.zeros(total_pads, dtype=np.float64)
    kxdelta = np.zeros(total_pads, dtype=np.float64)
    kydelta = np.zeros(total_pads, dtype=np.float64)
    kdeltadelta = np.zeros(total_pads, dtype=np.float64)
    kxxi = np.zeros(total_pads, dtype=np.float64)
    kyxi = np.zeros(total_pads, dtype=np.float64)
    kxyi = np.zeros(total_pads, dtype=np.float64)
    kyyi = np.zeros(total_pads, dtype=np.float64)

    tilting = operating.bearing_type in (
        "conventional_tilting_pad",
        "inlet_groove_tilting_pad",
        "spray_bar_tilting_pad",
    )
    directions = ("dx", "dy", "dtilt") if tilting else ("dx", "dy")
    for pad_index in range(total_pads):
        # One factorisation per pad for its two or three displacement
        # perturbations.
        n_press = _solve_pert_pressures(
            mesh,
            pad_index,
            film_onset,
            pads,
            vis_effect_3d,
            h_n,
            cavitated,
            [
                _stiffness_source(
                    direction, pad_index, pads, operating, h_n, dpdx_n, dpdz_n, mesh
                )
                for direction in directions
            ],
            helpers,
        )
        kxx_i, kyx_i, kdeltax_i = _integrate_forces(mesh, pad_index, n_press[0], pads)
        kxy_i, kyy_i, kdeltay_i = _integrate_forces(mesh, pad_index, n_press[1], pads)

        if tilting:
            kxdelta_i, kydelta_i, kdeltadelta_i = _integrate_forces(
                mesh, pad_index, n_press[2], pads
            )
        else:
            kdeltax_i = kdeltay_i = kxdelta_i = kydelta_i = kdeltadelta_i = 0.0

        if (
            operating.operating_type in ("axial_flow", "high_ambient_pressure")
            and tilting
            and bool(unloaded[pad_index])
        ):
            kdeltax_i = kdeltay_i = kxdelta_i = kydelta_i = kdeltadelta_i = 0.0
            kxx_i = kxy_i = kyx_i = kyy_i = 0.0

        kxx += kxx_i
        kyx += kyx_i
        kxy += kxy_i
        kyy += kyy_i
        kdeltax[pad_index] = kdeltax_i
        kdeltay[pad_index] = kdeltay_i
        kxdelta[pad_index] = kxdelta_i
        kydelta[pad_index] = kydelta_i
        kdeltadelta[pad_index] = kdeltadelta_i
        kxxi[pad_index] = kxx_i
        kyxi[pad_index] = kyx_i
        kxyi[pad_index] = kxy_i
        kyyi[pad_index] = kyy_i

    return CoefficientBlock(
        kxx,
        kyx,
        kxy,
        kyy,
        kdeltax,
        kdeltay,
        kxdelta,
        kydelta,
        kdeltadelta,
        kxxi,
        kyxi,
        kxyi,
        kyyi,
    )
