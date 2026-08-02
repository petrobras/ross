"""Dynamic coefficients, Jacobian and dynamic reduction.

For a converged hydrodynamic (zeroth-order) pressure solution, this module
computes:

* :func:`jacobian` -- the journal-equilibrium Jacobian, whose elements are the
  (negative) stiffness coefficients ``kxx, kyx, kxy, kyy`` and, for tilting
  pads, the tilt-DOF blocks ``kdeltax, kdeltay, kxdelta, kydelta,
  kdeltadelta`` plus the per-pad direct blocks.
* :func:`damping` -- the corresponding velocity-perturbation damping
  coefficients ``cxx, cyx, cxy, cyy`` and their tilt-DOF / per-pad blocks.
* :func:`dynamic_reduction` / :func:`dynamic_reduction_pivot` -- condensation
  of the pad tilt (and, with a flexible pivot, the pivot translation) degrees
  of freedom at a given excitation frequency, yielding the reduced 2x2 ``K``
  and ``C`` matrices.

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
    nodal_pressure,
    press_cavitate,
):
    """Build the prescribed-pressure boundary condition for a perturbation.

    Returns the list of constrained nodes and their prescribed (always zero)
    pressures.

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
    nodal_pressure : ndarray, shape (total_pads, dim_xz)
        Zeroth-order nodal pressure (used to detect the cavitation region).
    press_cavitate : float
        Cavitation pressure, Pa.

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
        np.ascontiguousarray(nodal_pressure, dtype=np.float64),
        float(press_cavitate),
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
    press_cavitate,
    nodal_pressure,
    element_source_all,
    helpers,
):
    """Solve one perturbed Reynolds problem and return nodal perturbed pressure.

    Shared backbone of the damping (:func:`damping_press_dxdot` /
    :func:`damping_press_dydot` / :func:`damping_press_dtiltdot`) and stiffness
    (:func:`stiffness_press`) perturbation solves.  The
    perturbation only changes the per-element source term ``Q`` (and, for the
    short-bearing oil-seal case, whether the circumferential conduction term
    is zeroed and whether ``Q``
    drops its circumferential term).  ``element_source_all`` computes ``Q`` for
    **all** elements at once — it is called exactly once per solve as
    ``element_source_all(elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr,
    theta_e_arr, oil_seal)`` with the pre-gathered element ids, local node ids,
    per-pad ``gamma`` nodal array, per-element averages ``g_E`` / ``theta_E``
    and the oil-seal flag, and must return the length-``total_e_reynolds``
    source array (C-family: a vectorized numpy expression; K-family: the fused
    ``stiffness_source_all_jit`` kernel).

    Returns
    -------
    n_press_pert : ndarray, shape (dim_xz,)
        Perturbed nodal pressure (only Reynolds nodes are set).
    """
    e_index_reynolds = mesh.e_index
    node_i_reynolds = mesh.node_i
    node_j_reynolds = mesh.node_j
    node_k_reynolds = mesh.node_k
    node_l_reynolds = mesh.node_l
    p = pad_index
    total_column_reynolds = 2 * mesh.bandwidth - 1
    oil_seal = abs(pads.arc_length_rad[0] - 2.0 * PI) < 1.0e-6

    total_bc, press_bc_index, prescribed_press = press_bc_pert(
        mesh,
        pad_index,
        film_onset,
        pads,
        nodal_pressure,
        press_cavitate,
    )

    gamma, g = gamma_g_pert(
        mesh,
        pad_index,
        pads,
        h_n,
        vis_effect_3d,
    )

    global_matrix_p, global_column_p = helpers.zero_pressure_system(
        mesh.dim_xz, mesh.total_nodes, total_column_reynolds
    )

    # Gather the per-element topology once (vectorized) and compute every
    # element's source value in a single ``element_source_all`` call — one
    # fused JIT call (K-family) or one vectorized numpy expression (C-family)
    # instead of one Python callback / JIT crossing per element.
    e_index_reynolds = np.ascontiguousarray(e_index_reynolds, dtype=np.int64)
    node_i_reynolds = np.ascontiguousarray(node_i_reynolds, dtype=np.int64)
    node_j_reynolds = np.ascontiguousarray(node_j_reynolds, dtype=np.int64)
    node_k_reynolds = np.ascontiguousarray(node_k_reynolds, dtype=np.int64)
    node_l_reynolds = np.ascontiguousarray(node_l_reynolds, dtype=np.int64)
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
    q_per_element = np.ascontiguousarray(
        element_source_all(
            elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
        ),
        dtype=np.float64,
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
        np.ascontiguousarray(mesh.e_length, dtype=np.float64),
        np.ascontiguousarray(mesh.e_width, dtype=np.float64),
        q_per_element,
        bool(oil_seal),
        int(mesh.bandwidth),
        global_matrix_p,
        global_column_p,
    )

    global_matrix_p, global_column_p = helpers.include_press(
        global_matrix_p,
        global_column_p,
        mesh.bandwidth,
        total_bc,
        press_bc_index,
        prescribed_press,
        mesh.total_nodes,
    )

    a, a_lower, index1, _d = banded.lu_factor(
        global_matrix_p, mesh.total_nodes, mesh.bandwidth
    )
    global_column_p = banded.lu_solve(
        a, mesh.total_nodes, mesh.bandwidth, a_lower, index1, global_column_p
    )

    n_press_pert = np.zeros(mesh.dim_xz, dtype=np.float64)
    for i in range(mesh.total_nodes):
        nid = int(mesh.n_index[i])
        n_press_pert[nid] = global_column_p[nid]
    return n_press_pert


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
def damping_press_dxdot(
    mesh,
    pad_index,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    press_cavitate,
    nodal_pressure,
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
    p = pad_index

    def source_all(
        elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
    ):
        return -np.cos(pads.leading_angle_rad[p] + theta_e_arr)

    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        press_cavitate,
        nodal_pressure,
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
    press_cavitate,
    nodal_pressure,
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
    p = pad_index

    def source_all(
        elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
    ):
        return -np.sin(pads.leading_angle_rad[p] + theta_e_arr)

    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        press_cavitate,
        nodal_pressure,
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
    press_cavitate,
    nodal_pressure,
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
    p = pad_index

    def source_all(
        elem, ni_arr, nj_arr, nk_arr, nl_arr, gamma, g_e_arr, theta_e_arr, oil_seal
    ):
        return -(pads.journal_radius + pads.pad_thickness) * np.sin(
            theta_e_arr - pads.x_pivot_rad[p]
        )

    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        press_cavitate,
        nodal_pressure,
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
    nodal_pressure,
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
    nodal_pressure : ndarray, shape (total_pads, dim_xz)
        Zeroth-order nodal pressure, Pa.
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
        cxx_i, cyx_i, cdeltax_i = damping_press_dxdot(
            mesh,
            pad_index,
            film_onset,
            pads,
            vis_effect_3d,
            h_n,
            operating.press_cavitate,
            nodal_pressure,
            helpers,
        )
        cxy_i, cyy_i, cdeltay_i = damping_press_dydot(
            mesh,
            pad_index,
            film_onset,
            pads,
            vis_effect_3d,
            h_n,
            operating.press_cavitate,
            nodal_pressure,
            helpers,
        )

        if tilting:
            cxdelta_i, cydelta_i, cdeltadelta_i = damping_press_dtiltdot(
                mesh,
                pad_index,
                film_onset,
                pads,
                vis_effect_3d,
                h_n,
                operating.press_cavitate,
                nodal_pressure,
                helpers,
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


def stiffness_press(
    direction,
    mesh,
    pad_index,
    pads,
    operating,
    vis_effect_3d,
    film_onset,
    h_n,
    nodal_pressure,
    dpdx_n,
    dpdz_n,
    helpers,
):
    """Shared body of the three stiffness perturbation solves.

    ``direction`` selects the angular function and scaling of the Jacobian
    source terms:

    ``"dx"``
        cosine, offset by the pad leading angle.
    ``"dy"``
        sine, offset by the pad leading angle.
    ``"dtilt"``
        sine, offset by ``-x_pivot``, scaled by the pivot radius
        ``journal_radius + pad_thickness``.

    For the oil-seal short-bearing case ``k_x`` is zeroed and ``q`` drops its
    first term.

    Returns
    -------
    fx, fy, moment : float
        Direct, cross and pivot-moment stiffness contributions.
    """
    dx_reynolds = mesh.dx
    dz_reynolds = mesh.dz
    x_reynolds_rad = mesh.x_rad
    _require_helpers(helpers)
    p = pad_index

    # ``stiffness_source_all_jit`` requires contiguous float64 inputs; convert
    # once here (``gamma`` is built contiguous inside ``_solve_pert_pressure``).
    dx_reynolds = np.ascontiguousarray(dx_reynolds, dtype=np.float64)
    dz_reynolds = np.ascontiguousarray(dz_reynolds, dtype=np.float64)
    x_reynolds_rad = np.ascontiguousarray(x_reynolds_rad, dtype=np.float64)
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

    n_press_pert = _solve_pert_pressure(
        mesh,
        pad_index,
        film_onset,
        pads,
        vis_effect_3d,
        h_n,
        operating.press_cavitate,
        nodal_pressure,
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
    nodal_pressure,
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
    nodal_pressure : ndarray, shape (total_pads, dim_xz)
        Zeroth-order nodal pressure, Pa.
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
    for pad_index in range(total_pads):
        kxx_i, kyx_i, kdeltax_i = stiffness_press(
            "dx",
            mesh,
            pad_index,
            pads,
            operating,
            vis_effect_3d,
            film_onset,
            h_n,
            nodal_pressure,
            dpdx_n,
            dpdz_n,
            helpers,
        )
        kxy_i, kyy_i, kdeltay_i = stiffness_press(
            "dy",
            mesh,
            pad_index,
            pads,
            operating,
            vis_effect_3d,
            film_onset,
            h_n,
            nodal_pressure,
            dpdx_n,
            dpdz_n,
            helpers,
        )

        if tilting:
            kxdelta_i, kydelta_i, kdeltadelta_i = stiffness_press(
                "dtilt",
                mesh,
                pad_index,
                pads,
                operating,
                vis_effect_3d,
                film_onset,
                h_n,
                nodal_pressure,
                dpdx_n,
                dpdz_n,
                helpers,
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


# ---------------------------------------------------------------------------
# Dynamic reduction over the pad tilt degrees of freedom
# ---------------------------------------------------------------------------
def dynamic_reduction(
    total_pads,
    stiffness,
    damping_block,
    pads,
    pad_density,
    excit_rad,
    ip,
    k_rotate,
):
    """Condense the pad tilt DOFs (pivot stiffness *not* included).

    Forms the complex dynamic stiffness ``D(s) = Kuu + s Cuu - A1 A3^{-1} A4``
    at ``s = i * excit_rad`` where the tilt block ``A3`` is diagonal, and
    returns the real/imag parts as reduced 2x2 ``K``/``C``.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    stiffness, damping_block : CoefficientBlock
        Journal 2x2 block plus the tilt-DOF coupling/diagonal blocks, as
        returned by :func:`jacobian` and :func:`damping`.
    pads : PadGeometry
        Pad geometry (``pad_length``, ``pad_thickness``, ``axial_length``).
    pad_density : float
        Pad material density, kg/m^3.
    excit_rad : float
        Excitation frequency, rad/s.
    ip : ndarray, shape (total_pads,)
        Pad polar moment of inertia buffer (recomputed, returned).
    k_rotate : ndarray, shape (total_pads,)
        Per-pad rotational stiffness added to the tilt diagonal.

    Returns
    -------
    kxx_reduced, kxy_reduced, kyx_reduced, kyy_reduced : float
    cxx_reduced, cxy_reduced, cyx_reduced, cyy_reduced : float
    ip : ndarray, shape (total_pads,)
        Updated pad polar moments of inertia.

    Examples
    --------
    With no tilt coupling (all ``delta`` blocks zero) the reduced
    coefficients equal the journal ones:

    >>> import numpy as np
    >>> from types import SimpleNamespace
    >>> from ross.bearings.fluid_film.state import CoefficientBlock
    >>> z = np.zeros(1)
    >>> k = CoefficientBlock(xx=10.0, yx=1.0, xy=2.0, yy=20.0, deltax=z,
    ...                      deltay=z, xdelta=z, ydelta=z, deltadelta=z)
    >>> c = CoefficientBlock(xx=3.0, yx=0.5, xy=0.5, yy=6.0, deltax=z,
    ...                      deltay=z, xdelta=z, ydelta=z, deltadelta=z)
    >>> pads = SimpleNamespace(
    ...     pad_length=np.array([1.0]), pad_thickness=0.1,
    ...     axial_length=np.array([1.0]),
    ... )
    >>> out = dynamic_reduction(1, k, c, pads, 0.3, 377.0, np.array([1.0]), z)
    >>> round(float(out[0]), 6), round(float(out[3]), 6)
    (10.0, 20.0)
    >>> round(float(out[4]), 6), round(float(out[7]), 6)
    (3.0, 6.0)
    """
    ip = np.array(ip, dtype=np.float64, copy=True)
    s = complex(0.0, excit_rad)
    t = s * s

    mass_pad = np.zeros(total_pads, dtype=np.float64)
    for k in range(total_pads):
        mass_pad[k] = (
            pads.pad_length[k] * pads.pad_thickness * pads.axial_length[k] * pad_density
        )
        ip[k] = mass_pad[k] * (pads.pad_thickness**2 + pads.pad_length[k] ** 2) / 12.0

    kuu = np.array(
        [[stiffness.xx, stiffness.xy], [stiffness.yx, stiffness.yy]], dtype=complex
    )
    cuu = np.array(
        [[damping_block.xx, damping_block.xy], [damping_block.yx, damping_block.yy]],
        dtype=complex,
    )

    kudelta = np.zeros((2, total_pads), dtype=complex)
    cudelta = np.zeros((2, total_pads), dtype=complex)
    kdeltau = np.zeros((total_pads, 2), dtype=complex)
    cdeltau = np.zeros((total_pads, 2), dtype=complex)
    kdeldel = np.zeros((total_pads, total_pads), dtype=complex)
    cdeldel = np.zeros((total_pads, total_pads), dtype=complex)
    mass_ip = np.zeros((total_pads, total_pads), dtype=complex)

    for k in range(total_pads):
        kudelta[0, k] = stiffness.xdelta[k]
        kudelta[1, k] = stiffness.ydelta[k]
        cudelta[0, k] = damping_block.xdelta[k]
        cudelta[1, k] = damping_block.ydelta[k]
        kdeltau[k, 0] = stiffness.deltax[k]
        kdeltau[k, 1] = stiffness.deltay[k]
        cdeltau[k, 0] = damping_block.deltax[k]
        cdeltau[k, 1] = damping_block.deltay[k]
        kdeldel[k, k] = stiffness.deltadelta[k] + k_rotate[k]
        cdeldel[k, k] = damping_block.deltadelta[k]
        mass_ip[k, k] = ip[k]

    a1 = s * cudelta + kudelta
    a3 = t * mass_ip + s * cdeldel + kdeldel
    # A3 is diagonal: invert element-wise.
    a3_inverse = np.zeros((total_pads, total_pads), dtype=complex)
    for k in range(total_pads):
        a3_inverse[k, k] = 1.0 / a3[k, k]
    a4 = s * cdeltau + kdeltau
    a5 = a1 @ a3_inverse
    a6 = a5 @ a4
    a7 = s * cuu + kuu
    ds = a7 - a6

    kxx_reduced = ds[0, 0].real
    kxy_reduced = ds[0, 1].real
    kyx_reduced = ds[1, 0].real
    kyy_reduced = ds[1, 1].real
    cxx_reduced = ds[0, 0].imag / excit_rad
    cxy_reduced = ds[0, 1].imag / excit_rad
    cyx_reduced = ds[1, 0].imag / excit_rad
    cyy_reduced = ds[1, 1].imag / excit_rad

    return (
        kxx_reduced,
        kxy_reduced,
        kyx_reduced,
        kyy_reduced,
        cxx_reduced,
        cxy_reduced,
        cyx_reduced,
        cyy_reduced,
        ip,
    )


def dynamic_reduction_pivot(
    total_pads,
    pads,
    pad_density,
    ip,
    stiffness,
    damping_block,
    k_pivot,
    excit_rad,
    k_rotate,
):
    """Condense pad tilt *and* pivot-translation DOFs (pivot flexibility on).

    Builds the ``2`` x ``2*total_pads`` coupling matrices and the
    ``2*total_pads`` x ``2*total_pads`` tilt/pivot blocks (tilt DOFs first,
    pivot-translation DOFs second), then performs the same complex condensation
    using a dense (Gauss) inverse of ``A3``.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    pads : PadGeometry
        Per-pad geometry (``leading_angle_rad``, ``x_pivot_rad``,
        ``pad_length``, ``axial_length``, ``pad_thickness``).
    pad_density : float
        Pad material density, kg/m^3.
    ip : ndarray, shape (total_pads,)
        Pad polar moment of inertia (recomputed in place, returned).
    stiffness, damping_block : CoefficientBlock
        Journal 2x2 block, tilt-DOF coupling blocks and the per-pad direct
        blocks (``xxi``..``yyi``) from :func:`jacobian` / :func:`damping`.
    k_pivot : ndarray, shape (total_pads,)
        Per-pad pivot translational stiffness, N/m.
    excit_rad : float
        Excitation frequency, rad/s.
    k_rotate : ndarray, shape (total_pads,)
        Per-pad rotational stiffness added to the tilt diagonal.

    Returns
    -------
    kxx_reduced, kxy_reduced, kyx_reduced, kyy_reduced : float
    cxx_reduced, cxy_reduced, cyx_reduced, cyy_reduced : float
    ip : ndarray, shape (total_pads,)
        Updated pad polar moments of inertia.
    """
    ip = np.array(ip, dtype=np.float64, copy=True)
    n = 2 * total_pads

    kuu = np.array(
        [[stiffness.xx, stiffness.xy], [stiffness.yx, stiffness.yy]], dtype=complex
    )
    cuu = np.array(
        [[damping_block.xx, damping_block.xy], [damping_block.yx, damping_block.yy]],
        dtype=complex,
    )

    kutp = np.zeros((2, n), dtype=complex)
    cutp = np.zeros((2, n), dtype=complex)
    ktpu = np.zeros((n, 2), dtype=complex)
    ctpu = np.zeros((n, 2), dtype=complex)
    ktptp = np.zeros((n, n), dtype=complex)
    ctptp = np.zeros((n, n), dtype=complex)
    mass_matrix = np.zeros((n, n), dtype=complex)
    mass_pad = np.zeros(total_pads, dtype=np.float64)

    for k in range(total_pads):
        mass_pad[k] = (
            pads.pad_length[k] * pads.pad_thickness * pads.axial_length[k] * pad_density
        )
        ip[k] = mass_pad[k] * (pads.pad_thickness**2 + pads.pad_length[k] ** 2) / 12.0
        theta_pi = pads.leading_angle_rad[k] + pads.x_pivot_rad[k]
        c = np.cos(theta_pi)
        sn = np.sin(theta_pi)

        # Kutp / Cutp (2 x n): tilt columns then pivot columns.
        kutp[0, k] = stiffness.xdelta[k]
        kutp[1, k] = stiffness.ydelta[k]
        kutp[0, total_pads + k] = -stiffness.xxi[k] * c - stiffness.xyi[k] * sn
        kutp[1, total_pads + k] = -stiffness.yxi[k] * c - stiffness.yyi[k] * sn
        cutp[0, k] = damping_block.xdelta[k]
        cutp[1, k] = damping_block.ydelta[k]
        cutp[0, total_pads + k] = -damping_block.xxi[k] * c - damping_block.xyi[k] * sn
        cutp[1, total_pads + k] = -damping_block.yxi[k] * c - damping_block.yyi[k] * sn

        # Ktpu / Ctpu (n x 2).
        ktpu[k, 0] = stiffness.deltax[k]
        ktpu[k, 1] = stiffness.deltay[k]
        ktpu[total_pads + k, 0] = -stiffness.xxi[k] * c - stiffness.yxi[k] * sn
        ktpu[total_pads + k, 1] = -stiffness.xyi[k] * c - stiffness.yyi[k] * sn
        ctpu[k, 0] = damping_block.deltax[k]
        ctpu[k, 1] = damping_block.deltay[k]
        ctpu[total_pads + k, 0] = -damping_block.xxi[k] * c - damping_block.yxi[k] * sn
        ctpu[total_pads + k, 1] = -damping_block.xyi[k] * c - damping_block.yyi[k] * sn

        # Diagonal tilt/pivot blocks (same-pad entries only).
        ktptp[k, k] = stiffness.deltadelta[k] + k_rotate[k]
        ktptp[k, total_pads + k] = -stiffness.deltax[k] * c - stiffness.deltay[k] * sn
        ktptp[total_pads + k, k] = -stiffness.xdelta[k] * c - stiffness.ydelta[k] * sn
        ktptp[total_pads + k, total_pads + k] = (
            c * (stiffness.xxi[k] * c + stiffness.xyi[k] * sn)
            + sn * (stiffness.yxi[k] * c + stiffness.yyi[k] * sn)
            + k_pivot[k]
        )
        ctptp[k, k] = damping_block.deltadelta[k]
        ctptp[k, total_pads + k] = (
            -damping_block.deltax[k] * c - damping_block.deltay[k] * sn
        )
        ctptp[total_pads + k, k] = (
            -damping_block.xdelta[k] * c - damping_block.ydelta[k] * sn
        )
        ctptp[total_pads + k, total_pads + k] = c * (
            damping_block.xxi[k] * c + damping_block.xyi[k] * sn
        ) + sn * (damping_block.yxi[k] * c + damping_block.yyi[k] * sn)
        mass_matrix[k, k] = ip[k]
        mass_matrix[total_pads + k, total_pads + k] = mass_pad[k]

    s = complex(0.0, excit_rad)
    t = s * s

    a1 = s * cutp + kutp
    a3 = t * mass_matrix + s * ctptp + ktptp
    a3_inverse = _inverse_complex(a3)
    a4 = s * ctpu + ktpu
    a5 = a1 @ a3_inverse
    a6 = a5 @ a4
    a7 = s * cuu + kuu
    ds = a7 - a6

    kxx_reduced = ds[0, 0].real
    kxy_reduced = ds[0, 1].real
    kyx_reduced = ds[1, 0].real
    kyy_reduced = ds[1, 1].real
    cxx_reduced = ds[0, 0].imag / excit_rad
    cxy_reduced = ds[0, 1].imag / excit_rad
    cyx_reduced = ds[1, 0].imag / excit_rad
    cyy_reduced = ds[1, 1].imag / excit_rad

    return (
        kxx_reduced,
        kxy_reduced,
        kyx_reduced,
        kyy_reduced,
        cxx_reduced,
        cxy_reduced,
        cyx_reduced,
        cyy_reduced,
        ip,
    )


def _inverse_complex(a):
    """Invert a complex matrix by naive Gauss elimination (no pivoting).

    Deliberately not ``numpy.linalg.inv``: the no-pivot algorithm is
    kept so that near-singular behaviour matches the pinned regression
    fixtures.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Complex matrix to invert (not modified).

    Returns
    -------
    ndarray, shape (n, n)
        The inverse.

    Examples
    --------
    >>> import numpy as np
    >>> m = np.array([[2.0 + 0j, 1.0], [1.0, 3.0]])
    >>> inv = _inverse_complex(m)
    >>> bool(np.allclose(m @ inv, np.eye(2)))
    True
    """
    n = a.shape[0]
    lu = np.array(a, dtype=complex, copy=True)

    # Gauss elimination (store multipliers in lower triangle).
    for i in range(n - 1):
        for j in range(i + 1, n):
            xmult = lu[j, i] / lu[i, i]
            for k in range(i + 1, n):
                lu[j, k] = lu[j, k] - xmult * lu[i, k]
            lu[j, i] = xmult

    y = np.zeros((n, n), dtype=complex)
    for col in range(n):
        b = np.zeros(n, dtype=complex)
        b[col] = 1.0
        # Forward update of the RHS.
        for i in range(n - 1):
            for j in range(i + 1, n):
                b[j] = b[j] - lu[j, i] * b[i]
        # Back substitution.
        x = np.zeros(n, dtype=complex)
        x[n - 1] = b[n - 1] / lu[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            ssum = b[i]
            for j in range(i + 1, n):
                ssum = ssum - lu[i, j] * x[j]
            x[i] = ssum / lu[i, i]
        y[:, col] = x
    return y
