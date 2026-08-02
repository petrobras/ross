"""Pad/shaft/shell elastic and thermal deformation.

This module solves the 2-D plane-strain elastic + thermal deformation of a
bearing pad with a 4-node bilinear (Q4) finite-element mesh, plus the driver
that loops over pads (:func:`elasto`) and the routine that turns the
hydrodynamic pressure/temperature fields into nodal loads
(:func:`pad_temp_and_load`).

Indexing conventions
--------------------
Node and element index arrays (``n_index_pad``, ``e_index_pad``,
``node_1_pad``, ...) store node and element numbers used directly as indices;
per-pad 2-D fields (``x_pad``, ``pad_temp``, ``nodal_force``, ...) are shaped
``(total_pads, dim...)`` and indexed ``[pad_index, node]``. Degrees of freedom
are numbered so that pad-local node ``n`` has x-DOF ``2*n`` (circumferential)
and y-DOF ``2*n + 1`` (radial).

The banded global stiffness matrix ``global_matrix_d`` is stored with shape
``(dim_xy2, 2 * bandwidth - 1)`` and the diagonal at band column
``bandwidth_deform - 1``.

Injected dependencies
---------------------
:func:`pad_temp_and_load` takes two integrators from the orchestrator as
callables:

``integrate_xz(pad_index, mesh, f) -> inte_f``
    Surface integral of the nodal field ``f`` over the Reynolds (film) mesh.

``trapezoid(t, f, start, stop) -> inte``
    Trapezoidal integral of ``f`` against ``t`` over the samples
    ``t[start:stop]``.
"""

import numpy as np

from ross.bearings.fluid_film import banded
from ross.bearings.fluid_film._numba_kernels import deform_assemble_all_jit


def deform_bc_fixed_geometry(
    total_n_pad,
    n_index_pad,
    total_bc_deform,
    deform_bc_index,
    y_pad,
    prescribed_deform,
):
    """Prescribe pad boundary conditions for a fixed-geometry bearing.

    The radial displacement of every node on the back of the pad (``y_pad ==
    0``) is set to zero.

    Parameters
    ----------
    total_n_pad : int
        Number of pad nodes.
    n_index_pad : array_like of int
        0-based global node numbers of the pad nodes, length ``dim_xy2``.
    total_bc_deform : int
        Ignored on input; recomputed and returned.
    deform_bc_index : array_like of int
        Work array for the constrained DOF numbers, length ``dim_xy2``.
    y_pad : array_like of float
        Radial (thickness-direction) nodal coordinate, length ``dim_xy2``,
        indexed by 0-based node number.
    prescribed_deform : array_like of float
        Work array for the prescribed DOF values, length ``dim_xy2``.

    Returns
    -------
    total_bc_deform : int
        Number of prescribed boundary conditions found.
    deform_bc_index : numpy.ndarray of int
        Updated constrained-DOF list (0-based DOF numbers).
    prescribed_deform : numpy.ndarray of float
        Updated prescribed values (all zero here).
    """
    deform_bc_index = np.ascontiguousarray(deform_bc_index, dtype=np.int64)
    prescribed_deform = np.ascontiguousarray(prescribed_deform, dtype=np.float64)

    j = 0
    for i in range(total_n_pad):
        node = int(n_index_pad[i])
        if abs(y_pad[node]) < 1.0e-6:
            # 0-based node ``node`` -> y-DOF ``2*node + 1``.
            deform_bc_index[j] = node * 2 + 1
            prescribed_deform[j] = 0.0
            j += 1
    total_bc_deform = j
    return total_bc_deform, deform_bc_index, prescribed_deform


def deform_bc_tilting_pad(
    pad_index,
    total_n_pad,
    n_index_pad,
    total_bc_deform,
    deform_bc_index,
    pad_length,
    x_pivot,
    x_pad,
    y_pad,
    prescribed_deform,
):
    """Prescribe pad boundary conditions for a tilting pad (line contact).

    Constrains the circumferential displacement of the node column nearest the
    pivot, and additionally the radial displacement of the pivot-column node on
    the back of the pad.

    Parameters
    ----------
    pad_index : int
        0-based pad number.
    total_n_pad : int
        Number of pad nodes.
    n_index_pad : array_like of int
        0-based global node numbers of the pad nodes, length ``dim_xy2``.
    total_bc_deform : int
        Ignored on input; recomputed and returned.
    deform_bc_index : array_like of int
        Work array for the constrained DOF numbers, length ``dim_xy2``.
    pad_length : array_like of float
        Pad arc length per pad, length ``total_pads``.
    x_pivot : array_like of float
        Circumferential pivot coordinate per pad, length ``total_pads``.
    x_pad : array_like of float
        Circumferential nodal coordinate, shape ``(total_pads, dim_xy2)``.
    y_pad : array_like of float
        Radial nodal coordinate, length ``dim_xy2``.
    prescribed_deform : array_like of float
        Work array for the prescribed DOF values, length ``dim_xy2``.

    Returns
    -------
    total_bc_deform : int
        Number of prescribed boundary conditions found.
    deform_bc_index : numpy.ndarray of int
        Updated constrained-DOF list (0-based DOF numbers).
    prescribed_deform : numpy.ndarray of float
        Updated prescribed values (all zero here).
    """
    deform_bc_index = np.ascontiguousarray(deform_bc_index, dtype=np.int64)
    prescribed_deform = np.ascontiguousarray(prescribed_deform, dtype=np.float64)
    x_pad = np.asarray(x_pad, dtype=float)

    p = pad_index

    # Find the closest nodal distance to the pivot on the pad surface.
    dx_min = pad_length[p]
    for i in range(total_n_pad):
        node = int(n_index_pad[i])
        dist = abs(x_pad[p, node] - x_pivot[p])
        dx_min = min(dx_min, dist)

    j = 0
    for i in range(total_n_pad):
        node = int(n_index_pad[i])
        # Node at the pivot circumferential location.
        if abs(abs(x_pad[p, node] - x_pivot[p]) - dx_min) < 1.0e-6:
            # 0-based node ``node`` -> x-DOF ``2*node``, y-DOF ``2*node + 1``.
            deform_bc_index[j] = node * 2
            prescribed_deform[j] = 0.0
            j += 1
            # Node also on the back of the pad.
            if abs(y_pad[node]) < 1.0e-6:
                deform_bc_index[j] = node * 2 + 1
                prescribed_deform[j] = 0.0
                j += 1
    total_bc_deform = j
    return total_bc_deform, deform_bc_index, prescribed_deform


def zero_deform_system(dim_xy2, total_column_pad):
    """Allocate/zero the banded global matrix and the global column.

    Returns freshly zeroed arrays rather than mutating in place; the caller
    assigns them.

    Parameters
    ----------
    dim_xy2 : int
        Declared dimension.
    total_column_pad : int
        Number of band columns used (``2 * bandwidth_deform - 1``).

    Returns
    -------
    global_matrix_d : numpy.ndarray
        Zeroed ``(dim_xy2, total_column_pad)`` banded matrix.
    global_column_d : numpy.ndarray
        Zeroed ``(dim_xy2,)`` column vector.
    """
    global_matrix_d = np.zeros((dim_xy2, total_column_pad), dtype=float)
    global_column_d = np.zeros(dim_xy2, dtype=float)
    return global_matrix_d, global_column_d


def integrand_e_deform(
    pad_index,
    current_element,
    node_1_pad,
    node_2_pad,
    node_3_pad,
    node_4_pad,
    x_pad,
    y_pad,
    young,
    poisson,
    pad_expand,
    delta_t_e,
    r,
    s,
):
    """Integrand of the Q4 element stiffness matrix and thermal column.

    Evaluates the plane-strain element contribution at a single Gauss point
    ``(r, s)``.

    Parameters
    ----------
    pad_index : int
        0-based pad number.
    current_element : int
        0-based element number.
    node_1_pad, node_2_pad, node_3_pad, node_4_pad : array_like of int
        Element connectivity arrays (0-based node numbers), length ``dim_xy2``,
        indexed by 0-based element number.
    x_pad : array_like of float
        Circumferential nodal coordinate, shape ``(total_pads, dim_xy2)``.
    y_pad : array_like of float
        Radial nodal coordinate, length ``dim_xy2``.
    young : float
        Young's modulus.
    poisson : float
        Poisson's ratio.
    pad_expand : float
        Thermal expansion coefficient.
    delta_t_e : float
        Element temperature rise above the reference.
    r, s : float
        Gauss-point natural coordinates.

    Returns
    -------
    integrand_e : numpy.ndarray, shape (8, 8)
        Element stiffness integrand at this Gauss point.
    integrand_f : numpy.ndarray, shape (8,)
        Thermal load integrand at this Gauss point.
    """
    x_pad = np.asarray(x_pad, dtype=float)
    y_pad = np.asarray(y_pad, dtype=float)
    p = pad_index
    el = current_element  # 0-based element id; index connectivity directly

    n1 = int(node_1_pad[el])
    n2 = int(node_2_pad[el])
    n3 = int(node_3_pad[el])
    n4 = int(node_4_pad[el])

    # global_coord[axis, local_node] (0-based local node 0..3)
    global_coord = np.zeros((2, 4), dtype=float)
    global_coord[0, 0] = x_pad[p, n1]
    global_coord[0, 1] = x_pad[p, n2]
    global_coord[0, 2] = x_pad[p, n3]
    global_coord[0, 3] = x_pad[p, n4]
    global_coord[1, 0] = y_pad[n1]
    global_coord[1, 1] = y_pad[n2]
    global_coord[1, 2] = y_pad[n3]
    global_coord[1, 3] = y_pad[n4]

    # Plane-strain material matrix.
    alpha = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    # fmt: off
    e_mat = np.array([
        [alpha * (1.0 - poisson), alpha * poisson,         0.0],
        [alpha * poisson,         alpha * (1.0 - poisson), 0.0],
        [0.0,                     0.0, alpha * (1.0 - 2.0 * poisson) * 0.5],
    ], dtype=float)
    # fmt: on

    epsilon0 = np.array(
        [pad_expand * delta_t_e, pad_expand * delta_t_e, 0.0], dtype=float
    )

    # Shape-function natural derivatives F (2 x 4).
    # fmt: off
    f_deriv = np.array([
        [-(1 - s) / 4.0, (1 - s) / 4.0, (1 + s) / 4.0, -(1 + s) / 4.0],
        [-(1 - r) / 4.0, -(1 + r) / 4.0, (1 + r) / 4.0, (1 - r) / 4.0],
    ], dtype=float)
    # fmt: on

    # Jacobian: jac[i, j] = sum_k f_deriv[i, k] * global_coord[j, k].
    jac = f_deriv @ global_coord.T
    det_j = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]

    j_inverse = np.array(
        [
            [jac[1, 1] / det_j, -jac[0, 1] / det_j],
            [-jac[1, 0] / det_j, jac[0, 0] / det_j],
        ],
        dtype=float,
    )

    # B (2 x 4) = J_inverse * F.
    b = j_inverse @ f_deriv

    # Strain-displacement matrix DN (3 x 8).
    dn = np.zeros((3, 8), dtype=float)
    for k in range(4):  # 0-based local node
        dn[0, 2 * k] = b[0, k]
        dn[1, 2 * k + 1] = b[1, k]
        dn[2, 2 * k] = b[1, k]
        dn[2, 2 * k + 1] = b[0, k]

    integrand_e = (dn.T @ e_mat @ dn) * det_j
    integrand_f = (dn.T @ (e_mat @ epsilon0)) * det_j
    return integrand_e, integrand_f


def element_deform(
    pad_index,
    current_element,
    node_1_pad,
    node_2_pad,
    node_3_pad,
    node_4_pad,
    x_pad,
    y_pad,
    young,
    poisson,
    pad_expand,
    delta_t_e,
):
    """Q4 element stiffness matrix and thermal load by 2x2 Gauss quadrature.

    Sums :func:`integrand_e_deform` over the four
    Gauss points ``(+/-1/sqrt(3), +/-1/sqrt(3))``.

    Parameters
    ----------
    pad_index : int
        0-based pad number.
    current_element : int
        0-based element number.
    node_1_pad, node_2_pad, node_3_pad, node_4_pad : array_like of int
        Element connectivity (0-based node numbers).
    x_pad : array_like of float
        Circumferential nodal coordinate, shape ``(total_pads, dim_xy2)``.
    y_pad : array_like of float
        Radial nodal coordinate, length ``dim_xy2``.
    young, poisson, pad_expand : float
        Material properties.
    delta_t_e : float
        Element temperature rise above reference.

    Returns
    -------
    e_matrix_pad : numpy.ndarray, shape (8, 8)
        Element stiffness matrix.
    e_column_pad : numpy.ndarray, shape (8,)
        Element thermal load vector.
    """
    a = 3.0
    g = 1.0 / np.sqrt(a)
    gauss_points = [(-g, -g), (g, -g), (-g, g), (g, g)]

    e_matrix_pad = np.zeros((8, 8), dtype=float)
    e_column_pad = np.zeros(8, dtype=float)
    for r, s in gauss_points:
        integrand_e, integrand_f = integrand_e_deform(
            pad_index,
            current_element,
            node_1_pad,
            node_2_pad,
            node_3_pad,
            node_4_pad,
            x_pad,
            y_pad,
            young,
            poisson,
            pad_expand,
            delta_t_e,
            r,
            s,
        )
        e_matrix_pad += integrand_e
        e_column_pad += integrand_f
    return e_matrix_pad, e_column_pad


def assemble_pad(
    e_matrix_pad,
    e_column_pad,
    local_coordinates,
    bandwidth_deform,
    global_matrix_d,
    global_column_d,
):
    """Assemble a Q4 element into the banded global system.

    ``global_matrix_d`` is stored in banded storage: row ``irow``
    (0-based DOF), band column (0-based, diagonal at ``bandwidth_deform - 1``).

    Parameters
    ----------
    e_matrix_pad : array_like, shape (8, 8)
        Element stiffness matrix.
    e_column_pad : array_like, shape (8,)
        Element load vector.
    local_coordinates : array_like of int, length 4
        0-based global node numbers of the element's four local nodes.
    bandwidth_deform : int
        Half-bandwidth (in nodes-times-two terms) of the band storage.
    global_matrix_d : array_like, shape (dim_xy2, total_column_pad)
        Banded global stiffness (mutated copy returned).
    global_column_d : array_like, shape (dim_xy2,)
        Global load vector (mutated copy returned).

    Returns
    -------
    global_matrix_d : numpy.ndarray
        Updated banded global matrix.
    global_column_d : numpy.ndarray
        Updated global load vector.
    """
    e_matrix_pad = np.asarray(e_matrix_pad, dtype=float)
    e_column_pad = np.asarray(e_column_pad, dtype=float)
    global_matrix_d = np.ascontiguousarray(global_matrix_d, dtype=np.float64)
    global_column_d = np.ascontiguousarray(global_column_d, dtype=np.float64)

    # 0-based DOF: node ``n`` -> x-DOF ``2*n`` (local e-matrix slot ``2*i``),
    # y-DOF ``2*n + 1`` (slot ``2*i + 1``). Band diagonal sits at column
    # ``bandwidth_deform - 1``; the band-column offset is a DOF difference and
    # so is invariant under the uniform 0-based shift.
    for i in range(4):  # local node 0..3
        ix = 2 * i
        iy = ix + 1
        for j in range(4):
            jx = 2 * j
            jy = jx + 1
            irow_x = int(local_coordinates[i]) * 2
            irow_y = irow_x + 1
            icol_x = int(local_coordinates[j]) * 2
            jcol_xx = icol_x - irow_x + (bandwidth_deform - 1)
            jcol_xy = jcol_xx + 1
            jcol_yx = jcol_xx - 1
            global_matrix_d[irow_x, jcol_xx] += e_matrix_pad[ix, jx]
            global_matrix_d[irow_y, jcol_xx] += e_matrix_pad[iy, jy]
            global_matrix_d[irow_x, jcol_xy] += e_matrix_pad[ix, jy]
            global_matrix_d[irow_y, jcol_yx] += e_matrix_pad[iy, jx]
        irow_x = int(local_coordinates[i]) * 2
        irow_y = irow_x + 1
        global_column_d[irow_x] += e_column_pad[ix]
        global_column_d[irow_y] += e_column_pad[iy]
    return global_matrix_d, global_column_d


def include_forces(
    pad_index,
    total_n_pad,
    n_index_pad,
    pad_thickness,
    y_pad,
    nodal_force,
    global_column_d,
):
    """Add the concentrated nodal film forces to the global column.

    For each pad node on the film interface (``y_pad == pad_thickness``) the
    pre-lumped radial force ``nodal_force`` is added to the radial DOF.

    Parameters
    ----------
    pad_index : int
        0-based pad number.
    total_n_pad : int
        Number of pad nodes.
    n_index_pad : array_like of int
        0-based global node numbers, length ``dim_xy2``.
    pad_thickness : float
        Pad thickness (radial coordinate of the film interface).
    y_pad : array_like of float
        Radial nodal coordinate, length ``dim_xy2``.
    nodal_force : array_like of float
        Lumped radial nodal force, shape ``(total_pads, dim_x)``; the surface
        nodes are stored densely in order (0-based counter ``k``).
    global_column_d : array_like, shape (dim_xy2,)
        Global load vector (mutated copy returned).

    Returns
    -------
    global_column_d : numpy.ndarray
        Updated global load vector.
    """
    nodal_force = np.asarray(nodal_force, dtype=float)
    y_pad = np.asarray(y_pad, dtype=float)
    global_column_d = np.ascontiguousarray(global_column_d, dtype=np.float64)
    p = pad_index

    k = 0
    for i in range(total_n_pad):
        node = int(n_index_pad[i])
        if abs(y_pad[node] - pad_thickness) < 1.0e-6:
            nn = 2 * node + 1  # 0-based radial (y) DOF
            global_column_d[nn] += nodal_force[p, k]
            k += 1
    return global_column_d


def include_deform(
    global_matrix_d,
    global_column_d,
    bandwidth_deform,
    total_bc_deform,
    deform_bc_index,
    prescribed_deform,
    total_row_pad,
):
    """Impose prescribed nodal displacements on the banded global system.

    For each constrained DOF the column is moved to
    the right-hand side and the row/column zeroed, leaving the diagonal so the
    DOF equals its prescribed value.

    Parameters
    ----------
    global_matrix_d : array_like, shape (dim_xy2, total_column_pad)
        Banded global stiffness (mutated copy returned).
    global_column_d : array_like, shape (dim_xy2,)
        Global load vector (mutated copy returned).
    bandwidth_deform : int
        Half-bandwidth of the band storage.
    total_bc_deform : int
        Number of prescribed boundary conditions.
    deform_bc_index : array_like of int
        Constrained DOF numbers (0-based), length ``dim_xy2``.
    prescribed_deform : array_like of float
        Prescribed DOF values, length ``dim_xy2``.
    total_row_pad : int
        Number of active rows (``2 * total_n_pad``).

    Returns
    -------
    global_matrix_d : numpy.ndarray
        Updated banded global matrix.
    global_column_d : numpy.ndarray
        Updated global load vector.
    """
    global_matrix_d = np.ascontiguousarray(global_matrix_d, dtype=np.float64)
    global_column_d = np.ascontiguousarray(global_column_d, dtype=np.float64)

    twb = 2 * bandwidth_deform
    for i in range(total_bc_deform):
        irow = int(deform_bc_index[i])  # 0-based constrained DOF

        # Move the coupled terms to the right-hand side and zero them.
        for j in range(2, twb + 1):
            jrow = irow - bandwidth_deform + j - 1
            jcol = twb - j  # 0-based band column
            if 0 <= jrow < total_row_pad:
                global_column_d[jrow] -= (
                    global_matrix_d[jrow, jcol] * prescribed_deform[i]
                )
                if jrow != irow:
                    global_matrix_d[jrow, jcol] = 0.0

        # Zero the constrained row except the diagonal, set the RHS.
        for jcol in range(twb - 1):
            if jcol != bandwidth_deform - 1:
                global_matrix_d[irow, jcol] = 0.0
        global_column_d[irow] = (
            global_matrix_d[irow, bandwidth_deform - 1] * prescribed_deform[i]
        )
    return global_matrix_d, global_column_d


def deformation(
    total_pads,
    dim_xy2,
    pad_index,
    bearing_type,
    pads,
    young,
    poisson,
    pad_expand,
    temp_ref,
    pad_temp,
    nodal_force,
    total_e_x_film,
    total_e_y_pad,
    total_n_pad,
    n_index_pad,
    total_e_pad,
    e_index_pad,
    bandwidth_deform,
    node_1_pad,
    node_2_pad,
    node_3_pad,
    node_4_pad,
    x_pad,
    y_pad,
    deform_r_surface,
):
    """Solve the 2-D plane-strain pad deformation for one pad.

    Builds
    and solves the banded FE system for the pad's elastic + thermal
    deformation, then extracts the radial surface deformation that perturbs the
    film thickness.

    Parameters
    ----------
    total_pads, dim_xy2 : int
        Declared dimensions.
    pad_index : int
        0-based pad number.
    bearing_type : str
        ``"fixed_geometry"`` uses :func:`deform_bc_fixed_geometry`; the
        tilting-pad types use :func:`deform_bc_tilting_pad`.
    pads : PadGeometry
        Per-pad geometry.
    young, poisson, pad_expand, temp_ref : float
        Material properties (Pa, dimensionless, 1/K) and reference
        temperature (K).
    pad_temp : array_like of float
        Nodal pad temperature, shape ``(total_pads, dim_xy2)``, indexed by
        0-based node number (as filled by :func:`pad_temp_and_load`).
    nodal_force : array_like of float
        Lumped radial nodal force, shape ``(total_pads, dim_x)``.
    total_e_x_film : int
        Number of film elements in the circumferential direction.
    total_e_y_pad : int
        Number of pad elements through the thickness.
    total_n_pad : int
        Number of pad nodes.
    n_index_pad : array_like of int
        0-based global node numbers, length ``dim_xy2``.
    total_e_pad : int
        Number of pad elements.
    e_index_pad : array_like of int
        0-based global element numbers, length ``dim_xy2``.
    bandwidth_deform : int
        Half-bandwidth of the band storage.
    node_1_pad, node_2_pad, node_3_pad, node_4_pad : array_like of int
        Element connectivity (0-based node numbers), length ``dim_xy2``.
    x_pad : array_like of float
        Circumferential nodal coordinate, shape ``(total_pads, dim_xy2)``.
    y_pad : array_like of float
        Radial nodal coordinate, length ``dim_xy2``.
    deform_r_surface : array_like of float
        Radial film-surface deformation, shape ``(total_pads, dim_x)``; the row
        for ``pad_index`` is overwritten and the full array returned.

    Returns
    -------
    deform_r_surface : numpy.ndarray
        Updated ``(total_pads, dim_x)`` radial surface deformation.
    nodal_deform : numpy.ndarray
        Full nodal solution for this pad, shape ``(total_pads, dim_xy2)``
        (DOF-ordered: ``2*i-1`` = x, ``2*i`` = y for pad-local node ``i``).
    deform_x : numpy.ndarray
        Circumferential nodal deformation, shape ``(total_pads, dim_xy2)``,
        indexed by 0-based global node number.
    deform_y : numpy.ndarray
        Radial nodal deformation, shape ``(total_pads, dim_xy2)``.
    """
    x_pad = np.asarray(x_pad, dtype=float)
    y_pad = np.asarray(y_pad, dtype=float)
    pad_temp = np.asarray(pad_temp, dtype=float)
    deform_r_surface = np.ascontiguousarray(deform_r_surface, dtype=np.float64)
    p = pad_index

    total_row_pad = 2 * total_n_pad
    total_column_pad = 2 * bandwidth_deform - 1

    deform_bc_index = np.zeros(dim_xy2, dtype=np.int64)
    prescribed_deform = np.zeros(dim_xy2, dtype=float)

    # Prescribed boundary conditions.
    if bearing_type == "fixed_geometry":
        total_bc_deform, deform_bc_index, prescribed_deform = deform_bc_fixed_geometry(
            total_n_pad,
            n_index_pad,
            0,
            deform_bc_index,
            y_pad,
            prescribed_deform,
        )
    elif bearing_type in (
        "conventional_tilting_pad",
        "inlet_groove_tilting_pad",
        "spray_bar_tilting_pad",
    ):
        total_bc_deform, deform_bc_index, prescribed_deform = deform_bc_tilting_pad(
            pad_index,
            total_n_pad,
            n_index_pad,
            0,
            deform_bc_index,
            pads.pad_length,
            pads.x_pivot,
            x_pad,
            y_pad,
            prescribed_deform,
        )
    else:
        total_bc_deform = 0

    global_matrix_d, global_column_d = zero_deform_system(dim_xy2, total_column_pad)

    # Assemble every element in a single JIT call (fuses the 2x2 Gauss
    # integrand, the element matrix and the band assembly; eliminates ~16k
    # per-element/per-Gauss-point Python<->JIT crossings).
    deform_assemble_all_jit(
        p,
        int(total_e_pad),
        np.ascontiguousarray(e_index_pad, dtype=np.int64),
        np.ascontiguousarray(node_1_pad, dtype=np.int64),
        np.ascontiguousarray(node_2_pad, dtype=np.int64),
        np.ascontiguousarray(node_3_pad, dtype=np.int64),
        np.ascontiguousarray(node_4_pad, dtype=np.int64),
        np.ascontiguousarray(x_pad, dtype=np.float64),
        np.ascontiguousarray(y_pad, dtype=np.float64),
        np.ascontiguousarray(pad_temp, dtype=np.float64),
        float(temp_ref),
        float(young),
        float(poisson),
        float(pad_expand),
        int(bandwidth_deform),
        global_matrix_d,
        global_column_d,
    )

    # Concentrated film forces.
    global_column_d = include_forces(
        pad_index,
        total_n_pad,
        n_index_pad,
        pads.pad_thickness,
        y_pad,
        nodal_force,
        global_column_d,
    )

    # Prescribed nodal displacements.
    global_matrix_d, global_column_d = include_deform(
        global_matrix_d,
        global_column_d,
        bandwidth_deform,
        total_bc_deform,
        deform_bc_index,
        prescribed_deform,
        total_row_pad,
    )

    # Banded LU solve.
    global_matrix_d, a_lower, index1, _d = banded.lu_factor(
        global_matrix_d, total_row_pad, bandwidth_deform
    )
    global_column_d = banded.lu_solve(
        global_matrix_d,
        total_row_pad,
        bandwidth_deform,
        a_lower,
        index1,
        global_column_d,
    )

    # Store the nodal deformation solution.
    nodal_deform = np.zeros((total_pads, dim_xy2), dtype=float)
    for i in range(total_row_pad):
        nodal_deform[p, i] = global_column_d[i]

    deform_x = np.zeros((total_pads, dim_xy2), dtype=float)
    deform_y = np.zeros((total_pads, dim_xy2), dtype=float)
    # 0-based local pad-node ``i`` -> x-DOF slot ``2*i``, y-DOF slot ``2*i + 1``.
    for i in range(total_n_pad):
        node = int(n_index_pad[i])
        deform_x[p, node] = nodal_deform[p, 2 * i]
        deform_y[p, node] = nodal_deform[p, 2 * i + 1]

    # Radial deformation on the pad/film interface (drives the film thickness).
    for i in range(total_e_x_film + 1):
        node = int(n_index_pad[(i + 1) * (total_e_y_pad + 1) - 1])
        deform_r_surface[p, i] = deform_y[p, node]

    return deform_r_surface, nodal_deform, deform_x, deform_y


def pad_temp_and_load(
    mesh,
    pad_index,
    deform_type,
    bearing_type,
    energy_mesh,
    pads,
    temp_ref,
    nodal_pressure,
    pressback_n,
    temp_full,
    nodal_force,
    force_pivot,
    pad_temp,
    integrate_xz,
    trapezoid,
):
    """Build pad nodal pressure load, temperature, and pivot force for one pad.

    Computes the net film pressure, integrates the
    pivot force, axially averages the pressure into a circumferential line load,
    lumps it onto pad nodes, and assigns nodal temperatures depending on the
    deformation/bearing type.

    Parameters
    ----------
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
    pad_index : int
        0-based pad number.
    deform_type : str
        Deformation model: the purely mechanical modes give free thermal
        expansion (temperature = ``temp_ref``); the thermal modes use the
        full temperature field.
    bearing_type : str
        Bearing type; ``"fixed_geometry"`` also forces free expansion.
    energy_mesh : EnergyMesh
        Film+pad cross-section (``x``-``y``) mesh.
    pads : PadGeometry
        Per-pad geometry.
    temp_ref : float
        Reference temperature, K.
    nodal_pressure, pressback_n : array_like of float
        Film and back pressures at Reynolds nodes, shape
        ``(total_pads, dim_xz)``, Pa.
    temp_full : array_like of float
        Full temperature field on the energy mesh, shape ``(total_pads, dim_xy)``.
    nodal_force : array_like of float
        Output lumped nodal force, shape ``(total_pads, dim_x)`` (returned).
    force_pivot : array_like of float
        Output pivot force per pad, length ``total_pads`` (returned).
    pad_temp : array_like of float
        Output pad nodal temperature, shape ``(total_pads, dim_xy2)`` (returned).
    integrate_xz : callable
        Surface integrator over the Reynolds mesh (see module docstring).
    trapezoid : callable
        Unequal-spacing trapezoidal integrator (see module docstring).

    Returns
    -------
    nodal_force : numpy.ndarray
        Updated lumped nodal force, shape ``(total_pads, dim_x)``.
    force_pivot : numpy.ndarray
        Updated pivot force per pad, length ``total_pads``.
    pad_temp : numpy.ndarray
        Updated pad nodal temperature, shape ``(total_pads, dim_xy2)``.
    """
    y_energy = energy_mesh.y
    x_reynolds_rad = mesh.x_rad
    z_reynolds = mesh.z
    nodal_pressure = np.asarray(nodal_pressure, dtype=float)
    pressback_n = np.asarray(pressback_n, dtype=float)
    z_reynolds = np.asarray(z_reynolds, dtype=float)
    y_energy = np.asarray(y_energy, dtype=float)
    x_reynolds_rad = np.asarray(x_reynolds_rad, dtype=float)
    temp_full = np.asarray(temp_full, dtype=float)
    nodal_force = np.ascontiguousarray(nodal_force, dtype=np.float64)
    force_pivot = np.ascontiguousarray(force_pivot, dtype=np.float64)
    pad_temp = np.ascontiguousarray(pad_temp, dtype=np.float64)
    p = pad_index

    # Net pressure on the pad and its radial component through the pivot.
    press_net = np.zeros(mesh.dim_xz, dtype=float)
    integrand = np.zeros(mesh.dim_xz, dtype=float)
    for i in range(mesh.total_nodes):
        node = int(mesh.n_index[i])
        press_net[node] = nodal_pressure[p, node] - pressback_n[p, node]
        integrand[node] = press_net[node] * np.cos(
            x_reynolds_rad[p, node] - pads.x_pivot_rad[p]
        )

    # Force on the pivot.
    inte_f_pivot = integrate_xz(
        pad_index,
        mesh,
        integrand,
    )
    force_pivot[p] = inte_f_pivot
    force_pivot[p] = max(force_pivot[p], 0.0)

    limit = mesh.total_e_z_film + 1

    # Axially average the nodal pressure into a circumferential line load.
    n_press_average = np.zeros(mesh.dim_x, dtype=float)
    for i in range(mesh.total_e_x_film + 1):
        t = np.zeros(mesh.dim_z, dtype=float)
        f = np.zeros(mesh.dim_z, dtype=float)
        for j in range(limit):
            idx = i * (mesh.total_e_z_film + 1) + j  # 0-based Reynolds node
            t[j] = z_reynolds[p, idx]
            f[j] = press_net[idx]
        # ``t``/``f`` are local scratch buffers filled at slots 0..limit-1;
        inte_trap = trapezoid(t, f, 0, limit)
        n_press_average[i] = inte_trap / pads.axial_length[p]

    step_length2 = pads.pad_length[p] / mesh.total_e_x_film

    # Lump the pressure onto the adjacent nodes (0-based circumferential node).
    for i in range(mesh.total_e_x_film + 1):
        if i == 0:
            nodal_force[p, i] = (
                -(n_press_average[i] + n_press_average[i + 1]) * step_length2 / 4.0
            )
        elif i == mesh.total_e_x_film:
            nodal_force[p, i] = (
                -(n_press_average[i - 1] + n_press_average[i]) * step_length2 / 4.0
            )
        else:
            nodal_force[p, i] = (
                -(n_press_average[i - 1] + n_press_average[i]) * step_length2 / 4.0
            ) + (-(n_press_average[i] + n_press_average[i + 1]) * step_length2 / 4.0)

    # Assign the pad nodal temperatures.
    m = 0
    for i in range(energy_mesh.total_nodes):
        node = int(energy_mesh.n_index[i])
        in_pad = (
            y_energy[p, node] < pads.pad_thickness
            or abs(y_energy[p, node] - pads.pad_thickness) < 1.0e-6
        )
        if in_pad:
            if (
                deform_type in ("pad_mechanical", "pad_pivot_mechanical")
                or bearing_type == "fixed_geometry"
            ):
                pad_temp[p, m] = temp_ref
            elif deform_type in (
                "pad_mechanical_thermal",
                "pad_mechanical_thermal_shaft_shell_thermal",
                "pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical",
            ):
                pad_temp[p, m] = temp_full[p, node]
            m += 1
    return nodal_force, force_pivot, pad_temp


def elasto(
    total_pads,
    mesh,
    energy_mesh,
    bearing_type,
    deform_type,
    pads,
    deform_r_surface,
    young,
    poisson,
    pad_expand,
    temp_ref,
    total_n_pad,
    n_index_pad,
    total_e_pad,
    e_index_pad,
    bandwidth_deform,
    node_1_pad,
    node_2_pad,
    node_3_pad,
    node_4_pad,
    x_pad,
    y_pad,
    nodal_pressure,
    pressback_n,
    temp_full,
    force_pivot,
    pad_temp,
    integrate_xz,
    trapezoid,
):
    """Driver: pad thermal + mechanical deformation for every pad.

    For each pad
    it calls :func:`pad_temp_and_load` then :func:`deformation`, accumulating the
    radial surface deformation that perturbs the film thickness.

    Parameters
    ----------
    total_pads : int
        Pad count.
    mesh : ReynoldsMesh
        Film (``x``-``z``) mesh: coordinates, connectivity and index maps.
    energy_mesh : EnergyMesh
        Film+pad cross-section (``x``-``y``) mesh.
    bearing_type, deform_type : str
        Bearing geometry and deformation model.
    pads : PadGeometry
        Per-pad geometry.
    deform_r_surface : array_like of float
        Radial film-surface deformation, shape ``(total_pads, dim_x)``
        (returned), m.
    young, poisson, pad_expand, temp_ref : float
        Pad material properties (Pa, dimensionless, 1/K) and reference
        temperature (K).
    total_n_pad, n_index_pad, total_e_pad, e_index_pad : mixed
        Pad-mesh sizes and 0-based index arrays.
    bandwidth_deform : int
        Deformation band half-bandwidth.
    node_1_pad, node_2_pad, node_3_pad, node_4_pad : array_like of int
        Pad element connectivity (0-based).
    x_pad : array_like of float
        Circumferential nodal coordinate, shape ``(total_pads, dim_xy2)``, m.
    y_pad : array_like of float
        Radial nodal coordinate, length ``dim_xy2``, m.
    nodal_pressure, pressback_n : array_like of float
        Film/back pressures, shape ``(total_pads, dim_xz)``, Pa.
    temp_full : array_like of float
        Full temperature field, shape ``(total_pads, dim_xy)``.
    force_pivot : array_like of float
        Pivot force per pad, length ``total_pads`` (returned).
    pad_temp : array_like of float
        Pad nodal temperature work array, shape ``(total_pads, dim_xy2)``
        (returned).
    integrate_xz, trapezoid : callable
        Upstream integrators (see module docstring).

    Returns
    -------
    deform_r_surface : numpy.ndarray
        Updated radial surface deformation, shape ``(total_pads, dim_x)``.
    force_pivot : numpy.ndarray
        Updated pivot force per pad, length ``total_pads``.
    pad_temp : numpy.ndarray
        Updated pad nodal temperature, shape ``(total_pads, dim_xy2)``.
    nodal_force : numpy.ndarray
        Lumped nodal force used internally, shape ``(total_pads, dim_x)``.
    """
    deform_r_surface = np.ascontiguousarray(deform_r_surface, dtype=np.float64)
    force_pivot = np.ascontiguousarray(force_pivot, dtype=np.float64)
    pad_temp = np.ascontiguousarray(pad_temp, dtype=np.float64)
    nodal_force = np.zeros((total_pads, mesh.dim_x), dtype=float)

    for pad_index in range(total_pads):
        nodal_force, force_pivot, pad_temp = pad_temp_and_load(
            mesh,
            pad_index,
            deform_type,
            bearing_type,
            energy_mesh,
            pads,
            temp_ref,
            nodal_pressure,
            pressback_n,
            temp_full,
            nodal_force,
            force_pivot,
            pad_temp,
            integrate_xz,
            trapezoid,
        )

        deform_r_surface, _nodal_deform, _deform_x, _deform_y = deformation(
            total_pads,
            energy_mesh.dim_xy2,
            pad_index,
            bearing_type,
            pads,
            young,
            poisson,
            pad_expand,
            temp_ref,
            pad_temp,
            nodal_force,
            mesh.total_e_x_film,
            energy_mesh.total_e_y_pad,
            total_n_pad,
            n_index_pad,
            total_e_pad,
            e_index_pad,
            bandwidth_deform,
            node_1_pad,
            node_2_pad,
            node_3_pad,
            node_4_pad,
            x_pad,
            y_pad,
            deform_r_surface,
        )

    return deform_r_surface, force_pivot, pad_temp, nodal_force


def deform_pivots(
    total_pads,
    deform_type,
    pivot_type,
    poisson,
    young,
    pivot_diameter,
    house_diameter,
    axial_length,
    pivot_stiff,
    force_pivot,
    deform_pivot,
    k_pivot,
):
    """Compute pivot deformation and stiffness per pad.

    For
    ``deform_type in ("pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical", "pad_pivot_mechanical")`` it sets ``deform_pivot`` and ``k_pivot`` from
    the per-pad ``force_pivot`` and the geometry/material constants, using
    one of four contact models selected by ``pivot_type``:

    1. Sphere in sphere.
    2. Sphere in cylinder.
    3. Cylinder in cylinder (closed form in ``axial_length`` only).
    4. User-supplied ``pivot_stiff``.

    For any other ``deform_type`` (``0/1/2/3``), ``deform_pivot`` is zeroed
    and ``k_pivot`` is left untouched.

    Parameters
    ----------
    total_pads : int
        Number of pads.
    deform_type, pivot_type : int
        Program option flags.
    poisson, young : float
        Pivot/housing material constants.
    pivot_diameter, house_diameter : float
        Pivot and housing diameters (used by types 1 and 2).
    axial_length : array_like of float
        Pad axial length per pad, length ``total_pads``.
    pivot_stiff : float
        User-supplied pivot stiffness (used by type 4).
    force_pivot : array_like of float
        Net pivot force per pad (set by :func:`pad_temp_and_load`), length
        ``total_pads``.
    deform_pivot, k_pivot : array_like of float
        Per-pad output buffers, length ``total_pads`` -- overwritten and
        returned.

    Returns
    -------
    deform_pivot : numpy.ndarray
        Pivot displacement under the current load, length ``total_pads``.
    k_pivot : numpy.ndarray
        Pivot stiffness, length ``total_pads``.

    References
    ----------
    Hertz contact formulas (sphere-in-sphere, sphere-in-cylinder,
    cylinder-in-cylinder).
    """
    axial_length = np.asarray(axial_length, dtype=float)
    force_pivot = np.asarray(force_pivot, dtype=float)
    deform_pivot = np.ascontiguousarray(deform_pivot, dtype=np.float64)
    k_pivot = np.ascontiguousarray(k_pivot, dtype=np.float64)

    for pad_index in range(total_pads):
        p = pad_index
        if deform_type in (
            "pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical",
            "pad_pivot_mechanical",
        ):
            if pivot_type == "ball_in_socket":
                ce = 2.0 * (1.0 - poisson**2) / young
                cd2 = (pivot_diameter * house_diameter) / (
                    house_diameter - pivot_diameter
                )
                deform_pivot[p] = 1.04 * (force_pivot[p] ** 2 * ce**2 / cd2) ** (
                    1.0 / 3.0
                )
                k_pivot[p] = 1.442 * (cd2 * force_pivot[p] / ce**2) ** (1.0 / 3.0)
            elif pivot_type == "button":
                ce = 2.0 * (1.0 - poisson**2) / young
                cd2 = (pivot_diameter * house_diameter) / (
                    house_diameter - pivot_diameter
                )
                deform_pivot[p] = (
                    0.52
                    * (force_pivot[p] ** 2 * ce**2) ** (1.0 / 3.0)
                    * (1.0 / pivot_diameter + 1.0 / cd2) ** (1.0 / 3.0)
                )
                k_pivot[p] = 2.885 * (
                    ((pivot_diameter * cd2) / (pivot_diameter + cd2))
                    * (force_pivot[p] / ce**2)
                ) ** (1.0 / 3.0)
            elif pivot_type == "rocker_back":
                # SI-derived from the imperial fit
                # ``d[in] = 4.36e-7 * F[lbf]^0.9 / L[in]^0.8`` and
                # ``K[lbf/in] = 2.55e6 * F[lbf]^0.1 * L[in]^0.8``.
                # Picking unit inputs and converting in/lbf <-> m/N gives the
                # coefficients below; the round-trip matches to 16 digits.
                deform_pivot[p] = (
                    1.530451e-10 * force_pivot[p] ** 0.9 / axial_length[p] ** 0.8
                )
                k_pivot[p] = 7.264526e9 * force_pivot[p] ** 0.1 * axial_length[p] ** 0.8
            elif pivot_type == "user_specified_stiffness":
                k_pivot[p] = pivot_stiff
                deform_pivot[p] = force_pivot[p] / k_pivot[p]
        else:
            deform_pivot[p] = 0.0
    return deform_pivot, k_pivot
